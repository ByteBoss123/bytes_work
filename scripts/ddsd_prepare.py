"""Export the four real corpora for directed-speech detection to 16 kHz wav
plus one manifest.

  directed (label 1):     SLURP test recordings (voice-assistant commands, headset + far-field)
                          MInDS-14 en-US/en-GB/en-AU (PolyAI; spoken to an e-banking voice
                          assistant over the phone, CC BY 4.0)
  non-directed (label 0): AMI ihm test (people talking to people in meetings, CC BY 4.0)
                          VoxCeleb1 test (interviews)

  python scripts/ddsd_prepare.py --data_root /tmp/data --out /tmp/data/ddsd
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import slurp as S  # noqa: E402
from speechlab.audio import SR, load_audio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_per_corpus", type=int, default=5000, help="raw cap before the ASR-based filter")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corpora", default="slurp,minds14,ami,vox1")
ap.add_argument("--ami_repo", default="edinburghcstr/ami")
ap.add_argument("--minds_repo", default="PolyAI/minds14")
args = ap.parse_args()

root, out = Path(args.data_root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
want = set(args.corpora.split(","))
rows = []


def resample(wav, sr):
    if sr == SR:
        return wav.astype(np.float32)
    import torch
    import torchaudio.functional as AF

    return AF.resample(torch.from_numpy(wav.astype(np.float32)), sr, SR).numpy()


def decode(audio_field):
    """datasets Audio(decode=False) gives {'bytes', 'path'}; decode ourselves so
    the pipeline does not depend on the datasets audio backend."""
    b = audio_field.get("bytes")
    if b:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    else:
        wav, sr = sf.read(audio_field["path"], dtype="float32", always_2d=True)
    return resample(wav.mean(1), sr)


def write(corpus, uid, wav, label, group, ref=""):
    d = out / corpus
    d.mkdir(exist_ok=True)
    p = d / f"{uid}.wav"
    if not p.exists():
        sf.write(p, wav, SR)
    rows.append({"id": f"{corpus}/{uid}", "path": str(p), "corpus": corpus, "label": label, "group": str(group),
                 "duration": round(len(wav) / SR, 3), "ref": ref})


def hf_stream(repo, config, split):
    from datasets import Audio, load_dataset

    if Path(repo).exists():  # local parquet export (tests, or a pre-downloaded copy)
        ds = load_dataset("parquet", data_files=str(Path(repo) / config / split / "*.parquet"), split="train",
                          streaming=True)
        return ds.cast_column("audio", Audio(decode=False))
    errs = []
    for kw in ({}, {"revision": "refs/convert/parquet"}):  # script datasets: use the parquet export
        try:
            ds = load_dataset(repo, config, split=split, streaming=True, **kw)
            return ds.cast_column("audio", Audio(decode=False))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kw or 'default'}: {e}")
    raise SystemExit(f"could not load {repo}/{config}/{split}: {errs}")


def first_text(ex, keys):
    for k in keys:
        if isinstance(ex.get(k), str) and ex[k].strip():
            return ex[k]
    return ""


if "slurp" in want:
    recs = S.load_recordings(root / "slurp_repo" / "dataset" / "slurp", splits=("test",))
    sid_text = {ex["slurp_id"]: S.transcript(ex) for ex in S.load_jsonl(root / "slurp_repo" / "dataset" / "slurp" / "test.jsonl")}
    pick = rng.permutation(len(recs))[: args.max_per_corpus]
    for i in sorted(pick):
        r = recs[i]
        write("slurp", Path(r.file).stem, load_audio(root / "slurp_audio" / "slurp_real" / r.file), 1, r.speaker,
              sid_text[r.slurp_id])

if "vox1" in want:
    files = sorted((root / "vox1" / "wav").rglob("*.wav"))
    pick = rng.permutation(len(files))[: args.max_per_corpus]
    for i in sorted(pick):
        f = files[i]
        rel = f.relative_to(root / "vox1" / "wav")
        write("vox1", "_".join(rel.with_suffix("").parts), load_audio(f), 0, rel.parts[0])

if "ami" in want:
    n = 0
    for ex in hf_stream(args.ami_repo, "ihm", "test"):
        text = first_text(ex, ("text",))
        if len(text.split()) < 4:  # cheap pre-filter; the real filter runs on ASR output
            continue
        write("ami", ex["audio_id"], decode(ex["audio"]), 0, ex.get("speaker_id", ex.get("meeting_id")), text.lower())
        n += 1
        if n >= args.max_per_corpus:
            break

if "minds14" in want:
    n = 0
    for cfg in ("en-US", "en-GB", "en-AU"):
        for k, ex in enumerate(hf_stream(args.minds_repo, cfg, "train")):
            text = first_text(ex, ("transcription", "english_transcription"))
            # no speaker ids are published; each recording is its own group
            write("minds14", f"{cfg}_{k:05d}", decode(ex["audio"]), 1, f"{cfg}_{k}", text.lower())
            n += 1
            if n >= args.max_per_corpus:
                break

with open(out / "manifest.jsonl", "a" if (out / "manifest.jsonl").exists() else "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
counts = {}
for r in rows:
    counts[r["corpus"]] = counts.get(r["corpus"], 0) + 1
print(json.dumps(counts))
