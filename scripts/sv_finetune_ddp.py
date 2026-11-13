"""Multi-GPU (DDP) fine-tuning of the pretrained ECAPA-TDNN on SLURP
train-split speakers (headset + far-field), with AAM-softmax and on-the-fly
babble augmentation. Test speakers are never seen.

Also measures training throughput so the 1-GPU vs 2-GPU scaling efficiency
can be reported from real runs:
  torchrun --nproc_per_node 1 scripts/sv_finetune_ddp.py ... --max_steps 300 --tag ws1
  torchrun --nproc_per_node 2 scripts/sv_finetune_ddp.py ... --max_steps 300 --tag ws2
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import audiocache  # noqa: E402
from speechlab import slurp as S  # noqa: E402
from speechlab.audio import load_audio, make_babble, mix_at_snr  # noqa: E402
from speechlab.speaker import AAMSoftmaxHead, load_ecapa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slurp_dir", required=True)
ap.add_argument("--audio_dir", required=True)
ap.add_argument("--split", default="results/sv/speaker_split.json")
ap.add_argument("--out_dir", default="runs/ecapa_ft")
ap.add_argument("--tag", default="main")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--max_steps", type=int, default=None)
ap.add_argument("--batch", type=int, default=64, help="per-GPU batch")
ap.add_argument("--crop_sec", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--head_lr", type=float, default=1e-2)
ap.add_argument("--freeze_blocks", type=int, default=0, help="freeze the first k ECAPA blocks")
ap.add_argument("--p_babble", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--save", type=int, default=1)
ap.add_argument("--cache_dir", default=None, help="decode audio once into an int16 memmap here")
args = ap.parse_args()

ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
else:
    rank, world, local = 0, 1, 0
device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda"):
    torch.cuda.set_device(device)
torch.manual_seed(args.seed + rank)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


# ------------------------------------------------------------------ data
split = json.load(open(args.split))
train_spk = set(split["train_speakers"])
assert not train_spk & set(split["test_speakers"]), "speaker leakage"
recs = [r for r in S.load_recordings(args.slurp_dir) if r.speaker in train_spk]
spk_ids = {s: i for i, s in enumerate(sorted({r.speaker for r in recs}))}
audio_dir = Path(args.audio_dir)
rng = random.Random(args.seed)
_hs = [r.file for r in recs if not r.farfield]
babble_pool_files = rng.sample(_hs, min(200, len(_hs)))


if args.cache_dir:
    if rank == 0:
        files = sorted({r.file for r in recs})
        audiocache.build([audio_dir / f for f in files], files, args.cache_dir, args.num_workers)
    if ddp:
        dist.barrier()
    reader = audiocache.Reader(args.cache_dir)
else:
    reader = None


def read(f):
    return reader.get(f) if reader is not None else load_audio(audio_dir / f)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = [(r.file, spk_ids[r.speaker]) for r in recs]
        self.crop = int(args.crop_sec * 16000)
        self.pool = None

    def __len__(self):
        return len(self.items)

    def _pool(self):
        if self.pool is None:  # loaded lazily inside each worker
            self.pool = [read(f) for f in babble_pool_files]
        return self.pool

    def __getitem__(self, i):
        f, y = self.items[i]
        w = read(f)
        g = np.random.default_rng()
        if len(w) >= self.crop:
            s = int(g.integers(0, len(w) - self.crop + 1))
            w = w[s : s + self.crop]
        else:
            w = np.pad(w, (0, self.crop - len(w)), mode="wrap")
        if g.random() < args.p_babble:
            pool = self._pool()
            noise = make_babble([pool[k] for k in g.choice(len(pool), 3, replace=False)], len(w), g)
            w = mix_at_snr(w, noise, float(g.uniform(0, 20)), g)
        return torch.from_numpy(w), y


ds = CropDataset()
sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed) if ddp else None
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=True,
                                 persistent_workers=args.num_workers > 0)

# ----------------------------------------------------------------- model
if rank == 0:
    emb_model = load_ecapa(device=device)
if ddp:
    dist.barrier()
if rank != 0:
    emb_model = load_ecapa(device=device)
for blk in list(emb_model.embedding_model.blocks)[: args.freeze_blocks]:
    for p in blk.parameters():
        p.requires_grad_(False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = emb_model
        self.head = AAMSoftmaxHead(192, len(spk_ids))

    def forward(self, x, y):
        rel = torch.ones(x.size(0), device=x.device)
        return self.head(self.emb(x, rel), y)


net = Net().to(device).train()
if ddp:
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local] if device.startswith("cuda") else None)
core = net.module if ddp else net
opt = torch.optim.AdamW([
    {"params": [p for p in core.emb.parameters() if p.requires_grad], "lr": args.lr},
    {"params": core.head.parameters(), "lr": args.head_lr},
], weight_decay=2e-5)
steps_per_epoch = len(dl)
total = args.max_steps or int(args.epochs * steps_per_epoch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=total, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
log(f"world={world} train_speakers={len(spk_ids)} recordings={len(ds)} steps/epoch={steps_per_epoch} total_steps={total}")

# ------------------------------------------------------------------ train
step, epoch, t0, timed, losses = 0, 0, None, 0, []
wait_s, t_last = 0.0, None
WARMUP = 20
while step < total:
    if sampler:
        sampler.set_epoch(epoch)
    t_last = time.time()
    for x, y in dl:
        if t0 is not None:
            wait_s += time.time() - t_last  # time blocked on the DataLoader
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            loss = net(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        losses.append(loss.item())
        if step == WARMUP:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
        elif step > WARMUP:
            timed += 1
        if step % 50 == 0:
            log(f"step {step}/{total} loss {np.mean(losses[-50:]):.3f}")
        if step >= total:
            break
        t_last = time.time()
    epoch += 1

if device.startswith("cuda"):
    torch.cuda.synchronize()
elapsed = time.time() - t0 if t0 else float("nan")
stats = {
    "tag": args.tag,
    "world_size": world,
    "per_gpu_batch": args.batch,
    "steps": step,
    "timed_steps": timed,
    "global_samples_per_sec": round(timed * args.batch * world / elapsed, 2) if timed else None,
    "data_wait_fraction": round(wait_s / elapsed, 3) if timed else None,
    "audio_cache": bool(args.cache_dir),
    "final_loss_mean_last100": round(float(np.mean(losses[-100:])), 4),
    "gpu": torch.cuda.get_device_name(local) if device.startswith("cuda") else "cpu",
    "train_speakers": len(spk_ids),
    "train_recordings": len(ds),
}
if rank == 0:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(out / f"train_stats_{args.tag}.json", "w"), indent=2)
    if args.save:
        torch.save(core.emb.embedding_model.state_dict(), out / f"embedding_model_{args.tag}.pt")
    print(json.dumps(stats, indent=2))
if ddp:
    dist.destroy_process_group()
