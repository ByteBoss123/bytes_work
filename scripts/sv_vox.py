"""Reproduce the pretrained ECAPA-TDNN baseline on VoxCeleb1-O (cleaned).

This is the sanity check for the whole verification pipeline: the model card
reports 0.80% EER on this list, so a large deviation means a pipeline bug.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab.metrics import eer, min_dcf  # noqa: E402
from speechlab.speaker import embed_files, load_ecapa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vox_root", required=True, help="dir containing wav/ and veri_test2.txt")
ap.add_argument("--out", default="results/sv_vox1o.json")
ap.add_argument("--device", default="cuda")
ap.add_argument("--ckpt", default=None, help="optional fine-tuned embedding_model state_dict")
args = ap.parse_args()

root = Path(args.vox_root)
trials = [line.split() for line in open(root / "veri_test2.txt") if line.strip()]
labels = np.array([int(t[0]) for t in trials])
files = sorted({t[1] for t in trials} | {t[2] for t in trials})
missing = [f for f in files if not (root / "wav" / f).exists()]
if missing:
    raise SystemExit(f"{len(missing)} trial files missing, e.g. {missing[:3]}")

model = load_ecapa(device=args.device)
if args.ckpt:
    import torch

    model.embedding_model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
emb = embed_files(model, [root / "wav" / f for f in files], args.device)
pos = {f: i for i, f in enumerate(files)}
a = emb[[pos[t[1]] for t in trials]]
b = emb[[pos[t[2]] for t in trials]]
scores = (a / np.linalg.norm(a, axis=1, keepdims=True) * (b / np.linalg.norm(b, axis=1, keepdims=True))).sum(1)
e, thr = eer(scores, labels)
res = {
    "trial_list": "VoxCeleb1-O cleaned (veri_test2.txt)",
    "n_trials": len(trials),
    "n_target": int(labels.sum()),
    "n_files": len(files),
    "model": "speechbrain/spkrec-ecapa-voxceleb" + (f" + {args.ckpt}" if args.ckpt else ""),
    "scoring": "cosine, no score normalisation",
    "eer_percent": round(100 * e, 3),
    "min_dcf_p0.01": round(min_dcf(scores, labels), 4),
    "model_card_eer_percent": 0.80,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
np.savez_compressed(Path(args.out).with_suffix(".emb.npz"), files=np.array(files), emb=emb)
print(json.dumps(res, indent=2))
