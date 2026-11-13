"""GRPO on the end-to-end speech LLM: audio -> intent JSON, same verifiable
reward as the text stage (format + exact intent + entity F1).

Start point: the SFT speech LLM from sllm_train.py. Its LoRA is merged into
the base LLM, which then serves as the frozen KL reference; a fresh LoRA is
trained on top. Encoder and projector stay frozen, so policy and reference
see identical audio tokens.

  python scripts/sllm_grpo.py --data_root D --sft_dir runs/sllm --out_dir runs/sllm_grpo
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import grpo as G  # noqa: E402
from speechlab import slurp as S  # noqa: E402
from speechlab import speechdata as D  # noqa: E402
from speechlab import speechllm as SL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--sft_dir", default="runs/sllm")
ap.add_argument("--out_dir", default="runs/sllm_grpo")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--prompts", type=int, default=4, help="recordings per step")
ap.add_argument("--group", type=int, default=4, help="samples per recording")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=50)
ap.add_argument("--eval_n", type=int, default=300)
ap.add_argument("--eval_batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
root = Path(args.data_root)
slurp_dir = root / "slurp_repo" / "dataset" / "slurp"
audio_dir = root / "slurp_audio" / "slurp_real"

# ------------------------------------------------------------- model
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from peft.utils import load_peft_weights  # noqa: E402

model, tok = SL.build(args.whisper, args.llm, args.lora_r, device)
set_peft_model_state_dict(model.llm, load_peft_weights(str(Path(args.sft_dir) / "lora"), device=device))
model.projector.load_state_dict(torch.load(Path(args.sft_dir) / "projector.pt", map_location=device))
for p in model.projector.parameters():
    p.requires_grad_(False)
model.llm = model.llm.merge_and_unload()  # SFT weights become the reference
tm = ["q_proj", "k_proj", "v_proj", "o_proj"]
if os.environ.get("SPEECHLAB_TEST_TINY") != "1":
    tm += ["gate_proj", "up_proj", "down_proj"]
model.llm = get_peft_model(model.llm, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                                 target_modules=tm))
if cuda:
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
fe = SL.feature_extractor(args.whisper)
enc_dtype = next(model.encoder.parameters()).dtype
opt = torch.optim.AdamW([p for p in model.llm.parameters() if p.requires_grad], lr=args.lr)
scaler = torch.amp.GradScaler("cuda", enabled=cuda)

# -------------------------------------------------------------- data
golds = {ex["slurp_id"]: S.gold_target(ex) for sp in ("train", "devel") for ex in S.load_jsonl(slurp_dir / f"{sp}.jsonl")}
train = D.slurp_items(slurp_dir, audio_dir, "train", tasks=("slu",))
dev = D.slurp_items(slurp_dir, audio_dir, "devel", tasks=("slu",))
dev = random.Random(args.seed).sample(dev, min(args.eval_n, len(dev)))
collate_t = D.Collator(fe, tok, train, with_target=False)
collate_d = D.Collator(fe, tok, dev, with_target=False)
ds_t, ds_d = D.ItemDataset(train), D.ItemDataset(dev)


def dev_reward():
    model.eval()
    rewards, parsed = [], 0
    for s in range(0, len(dev), args.eval_batch):
        idx, feats, n, pieces = collate_d([ds_d[i] for i in range(s, min(s + args.eval_batch, len(dev)))])
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            texts = model.generate(feats.to(device, dtype=enc_dtype), n, pieces, tok)
        for i, t in zip(idx, texts):
            rewards.append(S.slu_reward(t, golds[dev[i]["slurp_id"]]))
            parsed += S.parse_prediction(t) is not None
    return float(np.mean(rewards)), parsed / len(dev)


log = []
r0, p0 = dev_reward()
log.append({"step": 0, "dev_reward": round(r0, 4), "dev_parse_rate": round(p0, 4)})
print(log[-1], flush=True)
rng = random.Random(args.seed + 11)
t0 = time.time()
for step in range(1, args.steps + 1):
    pick = rng.sample(range(len(train)), args.prompts)
    idx, feats, n, pieces = collate_t([ds_t[i] for i in pick])
    st = G.sllm_grpo_step(model, tok, feats.to(device, dtype=enc_dtype), n, pieces,
                          [golds[train[i]["slurp_id"]] for i in idx], S.slu_reward, opt, device,
                          group=args.group, beta=args.beta, temperature=args.temperature,
                          scaler=scaler if cuda else None, autocast=cuda)
    rec = {"step": step, **{k: round(v, 4) for k, v in st.__dict__.items()}}
    if step % args.eval_every == 0 or step == args.steps:
        r, pr = dev_reward()
        rec.update({"dev_reward": round(r, 4), "dev_parse_rate": round(pr, 4)})
        print(rec, flush=True)
    elif step % 10 == 0:
        print(rec, flush=True)
    log.append(rec)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
model.llm.save_pretrained(out / "lora")
json.dump({"sft_dir": args.sft_dir, "steps": args.steps, "prompts_per_step": args.prompts, "group": args.group,
           "beta": args.beta, "lr": args.lr, "train_recordings": len(train), "dev_recordings": len(dev),
           "minutes": round((time.time() - t0) / 60, 1), "log": log}, open(out / "grpo_stats.json", "w"), indent=2)
