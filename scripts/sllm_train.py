"""Train the speech LLM (Whisper-small encoder frozen, projector + LoRA on
Qwen2.5-0.5B-Instruct) on real LibriSpeech + SLURP audio with DDP.

  torchrun --nproc_per_node 2 scripts/sllm_train.py --data_root /tmp/data --out_dir runs/sllm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import speechdata as D  # noqa: E402
from speechlab import speechllm as SL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--out_dir", default="runs/sllm")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--libri_max", type=int, default=20000, help="cap on train-clean-100 utterances (0 = none)")
ap.add_argument("--slurp_max", type=int, default=0, help="cap on SLURP train recordings (0 = all)")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--max_steps", type=int, default=None)
ap.add_argument("--batch", type=int, default=8, help="per GPU")
ap.add_argument("--grad_accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--projector_lr", type=float, default=1e-3)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--grad_ckpt", type=int, default=1)
ap.add_argument("--log_every", type=int, default=25)
args = ap.parse_args()

ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
local = int(os.environ.get("LOCAL_RANK", 0))
cuda = torch.cuda.is_available()
device = f"cuda:{local}" if cuda else "cpu"
if cuda:
    torch.cuda.set_device(device)
torch.manual_seed(args.seed)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------- data
root = Path(args.data_root)


def build_items():
    items = []
    if (root / "LibriSpeech" / "train-clean-100").exists():
        lib = D.librispeech_items(root / "LibriSpeech", "train-clean-100")
        if args.libri_max:
            lib = [lib[i] for i in np.random.default_rng(args.seed).permutation(len(lib))[: args.libri_max]]
        items += lib
    sl = D.slurp_items(root / "slurp_repo" / "dataset" / "slurp", root / "slurp_audio" / "slurp_real", "train",
                       assign_one_task_seed=args.seed)
    if args.slurp_max:
        sl = [sl[i] for i in np.random.default_rng(args.seed + 1).permutation(len(sl))[: args.slurp_max]]
    return items + sl


raw_items = build_items()  # deterministic, identical on every rank
if rank == 0:
    items, n_long = D.filter_by_duration(raw_items, root / "durations.json", write_cache=True)
if ddp:
    dist.barrier()
if rank != 0:
    items, n_long = D.filter_by_duration(raw_items, root / "durations.json", write_cache=False)
counts = {}
for it in items:
    k = f"{it['source']}:{it['task']}"
    counts[k] = counts.get(k, 0) + 1
log(f"train items {len(items)} (dropped >30s: {n_long}) {counts}")

# ---------------------------------------------------------------- model
if rank == 0:
    model, tok = SL.build(args.whisper, args.llm, args.lora_r, device)
if ddp:
    dist.barrier()
if rank != 0:
    model, tok = SL.build(args.whisper, args.llm, args.lora_r, device)
fe = SL.feature_extractor(args.whisper)
if args.grad_ckpt:
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.config.use_cache = False
model.train()
lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
n_train = sum(p.numel() for p in lora_params) + sum(p.numel() for p in model.projector.parameters())
log(f"trainable params {n_train/1e6:.2f}M")
net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None) if ddp else model
opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                         {"params": model.projector.parameters(), "lr": args.projector_lr}], weight_decay=0.0)

ds = D.ItemDataset(items)


def make_loader(epoch: int):
    """Fresh length-grouped shuffle each epoch, identical across ranks."""
    order = D.length_grouped_order(items, args.batch * world, args.seed + epoch)
    # equal-length shards: otherwise one rank runs an extra batch and DDP hangs
    order = order[: len(order) - len(order) % (args.batch * world)]
    order = order[rank::world]  # disjoint shard per rank
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, order), batch_size=args.batch, shuffle=False,
                                       num_workers=args.num_workers, collate_fn=D.Collator(fe, tok, items, True),
                                       drop_last=True)


dl = make_loader(0)
steps_per_epoch = len(dl) // args.grad_accum
total = args.max_steps or max(1, int(args.epochs * steps_per_epoch))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.projector_lr], total_steps=total,
                                            pct_start=0.05, anneal_strategy="cos")
scaler = torch.amp.GradScaler("cuda", enabled=cuda)
log(f"world={world} optimizer steps={total} (per-GPU batch {args.batch} x accum {args.grad_accum})")

step, micro, losses, t0, timed_samples = 0, 0, [], None, 0
history = []
epoch = 0
while step < total:
    if epoch > 0:
        dl = make_loader(epoch)
    for idx, feats, n_samples, pieces in dl:
        feats = feats.to(device, dtype=next(model.encoder.parameters()).dtype, non_blocking=True)
        sync = (micro + 1) % args.grad_accum == 0
        ctx = net.no_sync() if (ddp and not sync) else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            loss = net(feats, n_samples, pieces) / args.grad_accum
        scaler.scale(loss).backward()
        micro += 1
        losses.append(loss.item() * args.grad_accum)
        if t0 is not None:
            timed_samples += len(idx) * world
        if not sync:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lora_params + list(model.projector.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step == 10:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
        if step % args.log_every == 0 or step == total:
            m = float(np.mean(losses[-args.log_every * args.grad_accum:]))
            history.append({"step": step, "loss": round(m, 4)})
            log(f"step {step}/{total} loss {m:.4f}")
        if step >= total:
            break
    epoch += 1

if cuda:
    torch.cuda.synchronize()
elapsed = time.time() - t0 if t0 else float("nan")
if rank == 0:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(out / "lora")
    torch.save(model.projector.state_dict(), out / "projector.pt")
    stats = {"world_size": world, "steps": step, "per_gpu_batch": args.batch, "grad_accum": args.grad_accum,
             "train_items": len(items), "item_counts": counts, "dropped_over_30s": n_long,
             "trainable_params_m": round(n_train / 1e6, 2),
             "samples_per_sec": round(timed_samples / elapsed, 2) if timed_samples else None,
             "train_minutes": round(elapsed / 60, 1) if t0 else None,
             "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None,
             "gpu": torch.cuda.get_device_name(device) if cuda else "cpu",
             "loss_history": history, "whisper": args.whisper, "llm": args.llm, "lora_r": args.lora_r}
    json.dump(stats, open(out / "train_stats.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "loss_history"}, indent=2))
if ddp:
    dist.destroy_process_group()
