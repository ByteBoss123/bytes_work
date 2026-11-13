"""Leave-corpus-out evaluation of directed-speech detection with shortcut
baselines (see src/speechlab/ddsd.py for the rationale).

  python scripts/ddsd_eval.py --feat_dir results/ddsd
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import ddsd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feat_dir", default="results/ddsd")
ap.add_argument("--per_corpus", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corpus_norm", type=int, default=0, help="z-score dense features within each corpus")
args = ap.parse_args()

d = Path(args.feat_dir)
z = np.load(d / "features.npz")
feats = {i: {"acoustic": a, "channel": c, "length": length}
         for i, a, c, length in zip(z["ids"], z["acoustic"], z["channel"], z["length"])}
rows = [json.loads(line) for line in open(d / "rows.jsonl")]
kept = [r for r in rows if ddsd.keep_utterance(r["duration"], r["hyp"])]
counts = {}
for r in kept:
    counts[r["corpus"]] = counts.get(r["corpus"], 0) + 1
print("after the shared filter (ASR >= 4 words, 1.5-15 s):", counts)

present = set(counts)
folds = ddsd.make_folds(present)
res = {"filtered_counts": counts, "per_corpus_cap": args.per_corpus, "corpus_norm": bool(args.corpus_norm),
       "folds": ddsd.run_folds(kept, feats, folds, args.per_corpus, args.seed, bool(args.corpus_norm))}
json.dump(res, open(d / ("ddsd_eval_cnorm.json" if args.corpus_norm else "ddsd_eval.json"), "w"), indent=2)
for name, f in res["folds"].items():
    print(f"\n{name}  (n_train={f['n_train']})")
    for m, v in f["models"].items():
        print(f"  {m:13s} in-domain AUC {v['in_domain']['auc']:.3f}  cross-corpus AUC {v['cross_corpus']['auc']:.3f}"
              f"  EER {v['cross_corpus']['eer_percent']}%")
