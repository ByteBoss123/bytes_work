"""Multilingual spoken intent recognition with the speech LLM on
Speech-MASSIVE (FBK-MT, real crowd-sourced recordings, CC BY-NC-SA 4.0).

Speech-MASSIVE releases speech training data only for de-DE and fr-FR, so
the model trains on German + French audio and is evaluated on the
validation split of all 12 languages: 2 seen and 10 zero-shot. A
majority-intent baseline is reported per language.

  python scripts/multi_slu.py prepare --out /tmp/data/smassive
  python scripts/multi_slu.py train   --data /tmp/data/smassive --run_dir runs/multi
  python scripts/multi_slu.py eval    --data /tmp/data/smassive --run_dir runs/multi --out results/multi
"""
import argparse
import io
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import speechdata as D  # noqa: E402
from speechlab import speechllm as SL  # noqa: E402
from speechlab.audio import SR  # noqa: E402

LANGS = ["ar-SA", "de-DE", "es-ES", "fr-FR", "hu-HU", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "tr-TR", "vi-VN"]
TRAIN_LANGS = ["de-DE", "fr-FR"]

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prepare", "train", "eval"])
ap.add_argument("--repo", default="FBK-MT/Speech-MASSIVE")
ap.add_argument("--data", default="/tmp/data/smassive")
ap.add_argument("--out", default="results/multi")
ap.add_argument("--run_dir", default="runs/multi")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--train_per_lang", type=int, default=6000)
ap.add_argument("--eval_per_lang", type=int, default=500)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--grad_accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--projector_lr", type=float, default=1e-3)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


def pick(ex, keys):
    for k in keys:
        if k in ex and ex[k] not in (None, ""):
            return ex[k]
    raise KeyError(f"none of {keys} in {sorted(ex)}")


def stream(lang, split):
    """Yield rows of <lang>/<split>-*.parquet. Downloads whole parquet files
    with huggingface_hub (resumable, timeouts) and reads them with pyarrow,
    instead of datasets streaming, which stalled on this repo."""
    import pyarrow.parquet as pq

    if Path(args.repo).exists():  # local parquet copy (tests)
        files = sorted(str(p) for p in (Path(args.repo) / lang).glob(f"{split}*.parquet"))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = [f for f in HfApi().list_repo_files(args.repo, repo_type="dataset")
                 if f.startswith(f"{lang}/{split}-") and f.endswith(".parquet")]
        print(f"{lang}/{split}: {len(names)} parquet files", flush=True)
        files = [hf_hub_download(args.repo, f, repo_type="dataset", etag_timeout=60) for f in sorted(names)]
    if not files:
        raise FileNotFoundError(f"no parquet for {lang}/{split}")

    def rows():  # files are resolved eagerly above so missing languages fail at call time
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                yield from batch.to_pylist()

    return rows()


def decode(a):
    b = a.get("bytes")
    wav, sr = sf.read(io.BytesIO(b) if b else a["path"], dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav.astype(np.float32)


def load_manifest(split):
    return [json.loads(line) for line in open(Path(args.data) / f"{split}.jsonl")]


if args.step == "prepare":
    out = Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    plan = [(lang, "train", args.train_per_lang) for lang in TRAIN_LANGS] + \
           [(lang, "validation", args.eval_per_lang) for lang in LANGS]
    first_keys, counts = None, {}
    for split_name in ("train", "validation"):
        with open(out / f"{split_name}.jsonl", "w") as f:
            for lang, split, cap in [p for p in plan if p[1] == split_name]:
                n = 0
                try:
                    it = stream(lang, split)
                except Exception as e:  # noqa: BLE001  (logged and reported, never silent)
                    counts[f"{lang}/{split}"] = f"unavailable: {str(e)[:120]}"
                    print("SKIP", lang, split, str(e)[:200], flush=True)
                    continue
                for k, ex in enumerate(it):
                    if first_keys is None:
                        first_keys = sorted(ex)
                        print("Speech-MASSIVE fields:", first_keys, flush=True)
                    intent = pick(ex, ("intent_str", "intent"))
                    if isinstance(intent, int):
                        raise SystemExit("intent label is an int index; need the string field (intent_str)")
                    wav = decode(ex["audio"])
                    if not 0.3 <= len(wav) / SR <= 30:
                        continue
                    p = out / lang / f"{split}_{k:05d}.wav"
                    p.parent.mkdir(exist_ok=True)
                    sf.write(p, wav, SR)
                    f.write(json.dumps({"id": f"{lang}/{split}/{k}", "path": str(p), "lang": lang, "task": "intent",
                                        "text": str(intent), "utt": str(ex.get("utt", ""))}) + "\n")
                    n += 1
                    if n >= cap:
                        break
                counts[f"{lang}/{split}"] = n
                print(lang, split, n, flush=True)
    json.dump({"fields": first_keys, "counts": counts}, open(out / "prepare_stats.json", "w"), indent=2)
    if not all(isinstance(counts.get(f"{lng}/train"), int) and counts[f"{lng}/train"] > 0 for lng in TRAIN_LANGS):
        raise SystemExit(f"training languages missing: {counts}")
    sys.exit(0)

random.seed(args.seed)
torch.manual_seed(args.seed)
model, tok = SL.build(args.whisper, args.llm, device=device)
fe = SL.feature_extractor(args.whisper)
enc_dtype = next(model.encoder.parameters()).dtype
run = Path(args.run_dir)

if args.step == "train":
    items = load_manifest("train")
    for it in items:
        it["seconds"] = 0.0
    if cuda:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lora = [p for p in model.llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr},
                             {"params": model.projector.parameters(), "lr": args.projector_lr}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.projector_lr], total_steps=args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    col = D.Collator(fe, tok, items, with_target=True)
    ds = D.ItemDataset(items)
    step, micro, hist, t0 = 0, 0, [], time.time()
    while step < args.steps:
        order = list(range(len(items)))
        random.shuffle(order)
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, order), batch_size=args.batch,
                                         num_workers=args.num_workers, collate_fn=col, drop_last=True)
        for idx, feats, n, pieces in dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = model(feats.to(device, dtype=enc_dtype), n, pieces) / args.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora + list(model.projector.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0 or step == args.steps:
                hist.append({"step": step, "loss": round(loss.item() * args.grad_accum, 4)})
                print(hist[-1], flush=True)
            if step >= args.steps:
                break
    run.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(run / "lora")
    torch.save(model.projector.state_dict(), run / "projector.pt")
    langs = Counter(i["lang"] for i in items)
    json.dump({"train_items": len(items), "train_langs": dict(langs), "steps": step, "batch": args.batch,
               "grad_accum": args.grad_accum, "minutes": round((time.time() - t0) / 60, 1), "loss_history": hist,
               "gpu": torch.cuda.get_device_name(0) if cuda else "cpu"}, open(run / "train_stats.json", "w"), indent=2)
    sys.exit(0)

# ------------------------------------------------------------------ eval
from peft import set_peft_model_state_dict  # noqa: E402
from peft.utils import load_peft_weights  # noqa: E402

set_peft_model_state_dict(model.llm, load_peft_weights(str(run / "lora"), device=device))
model.projector.load_state_dict(torch.load(run / "projector.pt", map_location=device))
model.eval()
train_labels = Counter(i["text"] for i in load_manifest("train"))
majority = train_labels.most_common(1)[0][0]
items = load_manifest("validation")
col = D.Collator(fe, tok, items, with_target=False)
dl = torch.utils.data.DataLoader(D.ItemDataset(items), batch_size=16, num_workers=args.num_workers, collate_fn=col)
preds = [None] * len(items)
for idx, feats, n, pieces in dl:
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        out = model.generate(feats.to(device, dtype=enc_dtype), n, pieces, tok, max_new_tokens=16)
    for i, t in zip(idx, out):
        preds[i] = t.strip().split()[0] if t.strip() else ""
res = {"train_langs": TRAIN_LANGS, "majority_intent": majority, "per_lang": {}}
for lang in LANGS:
    sel = [k for k, it in enumerate(items) if it["lang"] == lang]
    if not sel:
        continue
    acc = float(np.mean([preds[k] == items[k]["text"] for k in sel]))
    base = float(np.mean([items[k]["text"] == majority for k in sel]))
    res["per_lang"][lang] = {"n": len(sel), "intent_accuracy": round(acc, 4), "majority_baseline": round(base, 4),
                             "seen_in_training": lang in TRAIN_LANGS}
for grp, langs in (("seen", TRAIN_LANGS), ("zero_shot", [lng for lng in LANGS if lng not in TRAIN_LANGS])):
    v = [res["per_lang"][lng] for lng in langs if lng in res["per_lang"]]
    if v:
        res[f"{grp}_mean_accuracy"] = round(float(np.mean([x["intent_accuracy"] for x in v])), 4)
        res[f"{grp}_mean_majority_baseline"] = round(float(np.mean([x["majority_baseline"] for x in v])), 4)
Path(args.out).mkdir(parents=True, exist_ok=True)
json.dump(res, open(Path(args.out) / "multi_eval.json", "w"), indent=2)
with open(Path(args.out) / "pred_multi.jsonl", "w") as f:
    for it, p in zip(items, preds):
        f.write(json.dumps({"id": it["id"], "lang": it["lang"], "gold": it["text"], "pred": p}) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "per_lang"}, indent=2))
