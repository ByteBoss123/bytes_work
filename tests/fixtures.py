"""Tiny SLURP-shaped fixture for pipeline tests. The audio is synthetic
tones, used ONLY to exercise code paths; no reported metric comes from it."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

INTENTS = [("calendar", "set"), ("weather", "query"), ("play", "music")]


def make_slurp(root: Path, n_speakers=8, utts_per_speaker=12, seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    ann, audio = root / "slurp", root / "audio"
    ann.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    meta, splits = {}, {"train": [], "devel": [], "test": []}
    recid = 0
    sid = 0
    for s in range(n_speakers):
        spk = f"{'FM'[s % 2]}E-{s:03d}"
        f0 = 120 + 25 * s
        for u in range(utts_per_speaker):
            recid += 1
            sid += 1
            scen, act = INTENTS[u % len(INTENTS)]
            words = ["set", "alarm", "for", "seven"] if scen == "calendar" else ["what", "is", "the", "weather"]
            files = []
            for far in (False, True):
                name = f"audio-{1000000 + recid}{'' if far else '-headset'}.flac"
                t = np.arange(int(16000 * seconds)) / 16000
                w = 0.3 * np.sin(2 * np.pi * f0 * t) + (0.05 if far else 0.01) * rng.standard_normal(len(t))
                sf.write(audio / name, w.astype(np.float32), 16000)
                files.append(name)
                meta.setdefault(str(sid), {"nlub_id": sid, "recordings": {}})["recordings"][name] = {
                    "ansid": 1, "recid": recid, "status": "correct", "usrid": spk}
            ex = {
                "slurp_id": sid, "sentence": " ".join(words), "intent": f"{scen}_{act}",
                "scenario": scen, "action": act,
                "tokens": [{"surface": w, "id": i} for i, w in enumerate(words)],
                "entities": [{"span": [3], "type": "time"}] if scen == "calendar" else [],
                "recordings": [{"file": f, "wer": 0.0, "ent_wer": 0.0, "status": "correct"} for f in files],
            }
            splits[["train", "train", "devel", "test"][u % 4]].append(ex)
    json.dump(meta, open(ann / "metadata.json", "w"))
    for k, v in splits.items():
        with open(ann / f"{k}.jsonl", "w") as f:
            for ex in v:
                f.write(json.dumps(ex) + "\n")
    return ann, audio
