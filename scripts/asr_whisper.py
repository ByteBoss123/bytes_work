"""Whisper-small ASR baseline. Writes hypotheses (input to the text SLU /
GRPO stage) and WER with the same normaliser used for the speech LLM.

  python scripts/asr_whisper.py --data_root D --set slurp_test --out results/asr
  python scripts/asr_whisper.py --data_root D --set slurp_train --max_items 20000 --out results/asr
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import speechdata as D  # noqa: E402
from speechlab.metrics import wer, whisper_normalizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--set", required=True, choices=["librispeech_test_clean", "slurp_test", "slurp_devel", "slurp_train"])
ap.add_argument("--out", default="results/asr")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--max_items", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", default="0/1", help="i/n: process every n-th item starting at i (for 2 GPUs)")
ap.add_argument("--merge", action="store_true", help="merge shard outputs of --set and score them")
args = ap.parse_args()
out = Path(args.out)


def score(items, hyps, n_long):
    norm = whisper_normalizer(args.whisper)
    res = {"set": args.set, "model": args.whisper, "n_items": len(items), "dropped_over_30s": n_long,
           "all": wer([i["text"] if "text" in i else i["ref"] for i in items], hyps, norm)}
    if args.set.startswith("slurp"):
        for name, far in (("headset", False), ("farfield", True)):
            sel = [k for k, it in enumerate(items) if it["farfield"] == far]
            if sel:
                ref = [items[k]["text"] if "text" in items[k] else items[k]["ref"] for k in sel]
                res[name] = wer(ref, [hyps[k] for k in sel], norm)
    return res


if args.merge:
    import glob

    rows = []
    for p in sorted(glob.glob(str(out / f"hyp_{args.set}.shard*of*.jsonl"))):
        rows += [json.loads(line) for line in open(p)]
    with open(out / f"hyp_{args.set}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = score(rows, [r["hyp"] for r in rows], None)
    json.dump(res, open(out / f"wer_{args.set}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    sys.exit(0)

root = Path(args.data_root)
if args.set == "librispeech_test_clean":
    items = D.librispeech_items(root / "LibriSpeech", "test-clean")
else:
    split = args.set.split("_")[1]
    items = D.slurp_items(root / "slurp_repo" / "dataset" / "slurp", root / "slurp_audio" / "slurp_real", split,
                          tasks=("asr",))
if args.max_items and len(items) > args.max_items:
    items = [items[i] for i in sorted(np.random.default_rng(args.seed).permutation(len(items))[: args.max_items])]
items, n_long = D.filter_by_duration(items)
si, sn = (int(x) for x in args.shard.split("/"))
items = items[si::sn]

from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
proc = WhisperProcessor.from_pretrained(args.whisper)
model = WhisperForConditionalGeneration.from_pretrained(args.whisper, dtype=dtype).to(device).eval()


def collate(batch):
    idx = [b[0] for b in batch]
    feats = proc.feature_extractor([b[1] for b in batch], sampling_rate=16000, return_tensors="pt").input_features
    return idx, feats


dl = torch.utils.data.DataLoader(D.ItemDataset(items), batch_size=args.batch, num_workers=args.num_workers,
                                 collate_fn=collate)
hyps = [None] * len(items)
with torch.no_grad():
    for idx, feats in dl:
        ids = model.generate(feats.to(device, dtype=dtype), language="en", task="transcribe", max_new_tokens=128)
        for i, t in zip(idx, proc.batch_decode(ids, skip_special_tokens=True)):
            hyps[i] = t.strip()

out.mkdir(parents=True, exist_ok=True)
tag = args.set + (f".shard{si}of{sn}" if sn > 1 else "")
with open(out / f"hyp_{tag}.jsonl", "w") as f:
    for it, h in zip(items, hyps):
        f.write(json.dumps({"id": it["id"], "slurp_id": it.get("slurp_id"), "farfield": it.get("farfield"),
                            "ref": it["text"], "hyp": h}) + "\n")
res = score(items, hyps, n_long)
json.dump(res, open(out / f"wer_{tag}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
