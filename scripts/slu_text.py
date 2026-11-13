"""Text SLU on real Whisper ASR transcripts of SLURP audio: SFT, then GRPO
with a verifiable reward (JSON format + exact intent + entity F1), evaluated
with the official SLURP metric.

  python scripts/slu_text.py sft  --data_root D --asr_dir results/asr --run_dir runs/slu
  python scripts/slu_text.py grpo --data_root D --asr_dir results/asr --run_dir runs/slu
  python scripts/slu_text.py eval --data_root D --asr_dir results/asr --run_dir runs/slu --out results/slu
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from speechlab import grpo as G  # noqa: E402
from speechlab import slurp as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["sft", "grpo", "eval"])
ap.add_argument("--data_root", required=True)
ap.add_argument("--asr_dir", default="results/asr")
ap.add_argument("--run_dir", default="runs/slu")
ap.add_argument("--out", default="results/slu")
ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--grpo_pool_frac", type=float, default=0.25, help="share of train utterances held out of SFT for GRPO")
ap.add_argument("--sft_epochs", type=float, default=1.0)
ap.add_argument("--sft_batch", type=int, default=16)
ap.add_argument("--sft_lr", type=float, default=2e-4)
ap.add_argument("--grpo_steps", type=int, default=300)
ap.add_argument("--grpo_prompts", type=int, default=8)
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--grpo_lr", type=float, default=2e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--eval_n", type=int, default=400)
ap.add_argument("--eval_batch", type=int, default=32)
ap.add_argument("--max_test", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
root = Path(args.data_root)
slurp_dir = root / "slurp_repo" / "dataset" / "slurp"
run = Path(args.run_dir)
run.mkdir(parents=True, exist_ok=True)
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


def gold_by_id(split):
    return {ex["slurp_id"]: ex for ex in S.load_jsonl(slurp_dir / f"{split}.jsonl")}


def load_hyps(set_name):
    rows = []
    merged = Path(args.asr_dir) / f"hyp_{set_name}.jsonl"
    paths = [str(merged)] if merged.exists() else sorted(glob.glob(str(Path(args.asr_dir) / f"hyp_{set_name}.shard*.jsonl")))
    for p in paths:
        rows += S.load_jsonl(p)
    if not rows:
        raise SystemExit(f"no ASR hypotheses for {set_name} in {args.asr_dir}; run scripts/asr_whisper.py first")
    return rows


def train_pools():
    golds = gold_by_id("train")
    rows = load_hyps("slurp_train")
    ids = sorted({r["slurp_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    grpo_ids = set(ids[: int(len(ids) * args.grpo_pool_frac)])  # split by utterance, not recording
    sft, grpo = [], []
    for r in rows:
        ex = (G.slu_prompt_text(r["hyp"]), S.gold_target(golds[r["slurp_id"]]))
        (grpo if r["slurp_id"] in grpo_ids else sft).append(ex)
    return sft, grpo


def load_base():
    if os.environ.get("SPEECHLAB_TEST_TINY") == "1":
        from speechlab.speechllm import tiny_llm

        return tiny_llm()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.float16 if cuda else torch.float32,
                                               attn_implementation="sdpa").to(device)
    return llm, tok


def new_lora(llm, dropout):
    from peft import LoraConfig, get_peft_model

    tm = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if os.environ.get("SPEECHLAB_TEST_TINY") != "1":
        tm += ["gate_proj", "up_proj", "down_proj"]
    return get_peft_model(llm, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=dropout,
                                          target_modules=tm))


def sft_merged():
    from peft import PeftModel

    llm, tok = load_base()
    llm = PeftModel.from_pretrained(llm, str(run / "sft")).merge_and_unload()
    return llm, tok


def evaluate(model, tok, pairs, batch):
    """pairs: list of (prompt, gold) -> (mean reward, parse rate, predictions)."""
    model.eval()
    preds = []
    for s in range(0, len(pairs), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            preds += G.greedy(model, tok, [p for p, _ in pairs[s : s + batch]], device)
    rewards = [S.slu_reward(t, g) for t, (_, g) in zip(preds, pairs)]
    parsed = [S.parse_prediction(t) for t in preds]
    return float(np.mean(rewards)), float(np.mean([x is not None for x in parsed])), parsed


def dev_pairs():
    try:
        rows = load_hyps("slurp_devel")
    except SystemExit:
        return None
    golds = gold_by_id("devel")
    rng = random.Random(args.seed)
    rows = rng.sample(rows, min(args.eval_n, len(rows)))
    return [(G.slu_prompt_text(r["hyp"]), S.gold_target(golds[r["slurp_id"]])) for r in rows]


# ======================================================================
if args.step == "sft":
    sft, grpo_pool = train_pools()
    llm, tok = load_base()
    model = new_lora(llm, 0.05)
    if cuda:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.sft_lr)
    order = list(range(len(sft)))
    random.shuffle(order)
    total = max(1, int(args.sft_epochs * len(order) / args.sft_batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.sft_lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    model.train()
    hist, t0 = [], time.time()
    for step in range(total):
        sel = [sft[order[(step * args.sft_batch + k) % len(order)]] for k in range(args.sft_batch)]
        ids, mask, lab = G.sft_batch(tok, [p for p, _ in sel], [S.target_to_text(g) for _, g in sel], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % 50 == 0 or step + 1 == total:
            hist.append({"step": step + 1, "loss": round(loss.item(), 4)})
            print(hist[-1], flush=True)
    model.save_pretrained(run / "sft")
    json.dump({"sft_examples": len(sft), "grpo_pool_examples": len(grpo_pool), "steps": total,
               "batch": args.sft_batch, "minutes": round((time.time() - t0) / 60, 1), "loss_history": hist},
              open(run / "sft_stats.json", "w"), indent=2)

elif args.step == "grpo":
    _, pool = train_pools()
    llm, tok = sft_merged()
    model = new_lora(llm, 0.0)  # no dropout: sampling and scoring must use the same policy
    if cuda:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.grpo_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    dev = dev_pairs()
    log, rng = [], random.Random(args.seed + 7)
    if dev:
        r0, pr0, _ = evaluate(model, tok, dev, args.eval_batch)
        log.append({"step": 0, "dev_reward": round(r0, 4), "dev_parse_rate": round(pr0, 4)})
        print(log[-1], flush=True)
    t0 = time.time()
    for step in range(1, args.grpo_steps + 1):
        batch = rng.sample(pool, args.grpo_prompts)
        st = G.grpo_step(model, tok, [p for p, _ in batch], [g for _, g in batch], S.slu_reward, opt, device,
                         group=args.group, beta=args.beta, temperature=args.temperature,
                         scaler=scaler if cuda else None, autocast=cuda)
        rec = {"step": step, **{k: round(v, 4) for k, v in st.__dict__.items()}}
        if step % 10 == 0:
            print(rec, flush=True)
        if dev and (step % args.eval_every == 0 or step == args.grpo_steps):
            r, pr, _ = evaluate(model, tok, dev, args.eval_batch)
            rec.update({"dev_reward": round(r, 4), "dev_parse_rate": round(pr, 4)})
            print(rec, flush=True)
        log.append(rec)
    model.save_pretrained(run / "grpo")
    json.dump({"pool_examples": len(pool), "steps": args.grpo_steps, "prompts_per_step": args.grpo_prompts,
               "group": args.group, "beta": args.beta, "lr": args.grpo_lr,
               "minutes": round((time.time() - t0) / 60, 1), "log": log},
              open(run / "grpo_stats.json", "w"), indent=2)

else:  # eval
    from peft import PeftModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    golds = gold_by_id("test")
    rows = load_hyps("slurp_test")
    if args.max_test:
        rows = random.Random(args.seed).sample(rows, min(args.max_test, len(rows)))
    inputs = {
        "asr_headset": [r for r in rows if not r["farfield"]],
        "asr_farfield": [r for r in rows if r["farfield"]],
    }
    seen, gold_rows = set(), []
    for r in rows:  # gold transcript upper bound: one row per utterance
        if r["slurp_id"] not in seen:
            seen.add(r["slurp_id"])
            gold_rows.append({**r, "hyp": S.transcript(golds[r["slurp_id"]])})
    inputs["gold_transcript"] = gold_rows

    systems = {}
    llm, tok = load_base()
    systems["sft"] = PeftModel.from_pretrained(llm, str(run / "sft"))
    if (run / "grpo").exists():
        m2, _ = sft_merged()
        systems["sft+grpo"] = PeftModel.from_pretrained(m2, str(run / "grpo"))
    results = {}
    for sname, model in systems.items():
        for iname, rs in inputs.items():
            pairs = [(G.slu_prompt_text(r["hyp"]), S.gold_target(golds[r["slurp_id"]])) for r in rs]
            mean_r, parse, parsed = evaluate(model, tok, pairs, args.eval_batch)
            pf = out / f"pred_{sname.replace('+', '_')}_{iname}.jsonl"
            with open(pf, "w") as f:
                for r, pr in zip(rs, parsed):
                    f.write(json.dumps(S.prediction_record(r["id"], pr)) + "\n")
            off = out / f"official_{sname.replace('+', '_')}_{iname}.json"
            subprocess.run([sys.executable, str(HERE / "slurp_official_eval.py"), str(root / "slurp_repo"),
                            str(slurp_dir / "test.jsonl"), str(pf), str(off)], check=True, capture_output=True)
            results[f"{sname}|{iname}"] = {"n": len(rs), "mean_reward": round(mean_r, 4),
                                          "json_parse_rate": round(parse, 4), "official": json.load(open(off))}
            o = results[f"{sname}|{iname}"]["official"]
            print(sname, iname, "intent F1", o.get("intent (scen_act)", {}).get("f1"),
                  "SLU-F1", o.get("slu f1", {}).get("f1"), flush=True)
    json.dump(results, open(out / "slu_eval.json", "w"), indent=2)
