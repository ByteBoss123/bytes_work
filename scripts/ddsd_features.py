"""Whisper-small transcripts + mean-pooled encoder embeddings, channel-only
and length-only features for every utterance in the DDSD manifest.

  python scripts/ddsd_features.py --ddsd_dir /tmp/data/ddsd --out results/ddsd
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import ddsd  # noqa: E402
from speechlab.audio import load_audio  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ddsd_dir", required=True)
ap.add_argument("--out", default="results/ddsd")
ap.add_argument("--whisper", default="openai/whisper-small")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--num_workers", type=int, default=4)
args = ap.parse_args()

rows = {}
for line in open(Path(args.ddsd_dir) / "manifest.jsonl"):
    r = json.loads(line)
    rows[r["id"]] = r  # dedupe re-runs: last write wins
rows = list(rows.values())

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
if os.environ.get("SPEECHLAB_TEST_TINY") == "1":
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

    fe = WhisperFeatureExtractor(feature_size=80)
    model = WhisperForConditionalGeneration(WhisperConfig(d_model=32, encoder_layers=1, decoder_layers=1,
                                                          encoder_attention_heads=2, decoder_attention_heads=2,
                                                          encoder_ffn_dim=64, decoder_ffn_dim=64, vocab_size=60,
                                                          pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                                          decoder_start_token_id=1, max_target_positions=64))
    decode = lambda ids: [" ".join(f"w{t}" for t in seq.tolist() if t > 2) for seq in ids]  # noqa: E731
    gen_kw = {"max_new_tokens": 8}
else:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(args.whisper)
    fe = proc.feature_extractor
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper, dtype=dtype)
    decode = lambda ids: [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]  # noqa: E731
    gen_kw = {"language": "en", "task": "transcribe", "max_new_tokens": 96}
model = model.to(device, dtype=dtype).eval()


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(rows)

    def __getitem__(self, i):
        w = load_audio(rows[i]["path"])[: 30 * 16000]
        return i, w, ddsd.channel_features(w)


def collate(b):
    feats = fe([x[1] for x in b], sampling_rate=16000, return_tensors="pt").input_features
    return [x[0] for x in b], feats, [len(x[1]) for x in b], np.stack([x[2] for x in b])


hyps, acoustic, channel = [None] * len(rows), np.zeros((len(rows), model.config.d_model), np.float32), \
    np.zeros((len(rows), 162), np.float32)
dl = torch.utils.data.DataLoader(DS(), batch_size=args.batch, num_workers=args.num_workers, collate_fn=collate)
with torch.no_grad():
    for idx, feats, n, ch in dl:
        feats = feats.to(device, dtype=dtype)
        h = model.model.encoder(feats).last_hidden_state.float()
        for j, (i, ns) in enumerate(zip(idx, n)):
            t = min(h.size(1), math.ceil(ns / 320))
            acoustic[i] = h[j, :t].mean(0).cpu().numpy()  # pool over real audio only, not padding
            channel[i] = ch[j]
        for i, t in zip(idx, decode(model.generate(input_features=feats, **gen_kw))):
            hyps[i] = t

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
length = np.stack([ddsd.length_features(r["duration"], h) for r, h in zip(rows, hyps)])
np.savez_compressed(out / "features.npz", ids=np.array([r["id"] for r in rows]), acoustic=acoustic,
                    channel=channel, length=length)
with open(out / "rows.jsonl", "w") as f:
    for r, h in zip(rows, hyps):
        f.write(json.dumps({**r, "hyp": h}) + "\n")
print(f"{len(rows)} utterances featurised")
