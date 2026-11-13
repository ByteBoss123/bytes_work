"""Out-of-domain speaker-verification check on LibriSpeech test-clean and
test-other (publicly downloadable; replaces the VoxCeleb1-O check, whose
data now requires registration). Every utterance pair of a set is a trial.

  python scripts/sv_libri.py --libri_root D/LibriSpeech [--ckpt runs/ecapa_ft/embedding_model_X.pt] --out results/sv_libri.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab.metrics import eer, min_dcf  # noqa: E402
from speechlab.speaker import embed_files, l2n, load_ecapa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--libri_root", required=True)
ap.add_argument("--parts", default="test-clean,test-other")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--out", default="results/sv_libri.json")
ap.add_argument("--device", default="cuda")
ap.add_argument("--num_workers", type=int, default=4)
args = ap.parse_args()

import torch  # noqa: E402

device = args.device if torch.cuda.is_available() else "cpu"
model = load_ecapa(device=device)
if args.ckpt:
    model.embedding_model.load_state_dict(torch.load(args.ckpt, map_location=device))
res = {"model": "speechbrain/spkrec-ecapa-voxceleb" + (f" + {args.ckpt}" if args.ckpt else ""), "parts": {}}
for part in args.parts.split(","):
    files = sorted((Path(args.libri_root) / part).rglob("*.flac"))
    if not files:
        continue
    spk = np.array([f.name.split("-")[0] for f in files])
    e = l2n(embed_files(model, files, device, num_workers=args.num_workers))
    iu = np.triu_indices(len(files), k=1)
    scores = (e @ e.T)[iu]
    labels = (spk[:, None] == spk[None, :])[iu].astype(np.int8)
    ee, _ = eer(scores, labels)
    res["parts"][part] = {"utterances": len(files), "speakers": int(len(set(spk))), "trials": int(len(labels)),
                          "target_trials": int(labels.sum()), "eer_percent": round(100 * ee, 3),
                          "min_dcf_p0.01": round(min_dcf(scores, labels), 4)}
    print(part, res["parts"][part], flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
