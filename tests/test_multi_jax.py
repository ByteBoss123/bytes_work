"""Multilingual intent pipeline and the JAX directed-speech head, run on
tiny random models / synthetic data (mechanics only)."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def _parquet(root: Path, lang: str, split: str, n: int, intents):
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = []
    for k in range(n):
        buf = io.BytesIO()
        sf.write(buf, (0.1 * np.sin(np.arange(16000) / (3 + k))).astype(np.float32), 16000, format="WAV")
        recs.append({"audio": {"bytes": buf.getvalue(), "path": f"{k}.wav"}, "utt": "wake me up at seven",
                     "intent_str": intents[k % len(intents)], "intent_idx": k % len(intents), "locale": lang})
    (root / lang).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(recs), root / lang / f"{split}-0.parquet")


def test_multilingual_prepare_train_eval(tmp_path):
    repo = tmp_path / "sm"
    for lang in ("de-DE", "fr-FR"):
        _parquet(repo, lang, "train", 6, ["alarm_set", "weather_query"])
    for lang in ("de-DE", "fr-FR", "ko-KR"):
        _parquet(repo, lang, "validation", 3, ["alarm_set", "weather_query"])
    env = dict(os.environ, SPEECHLAB_TEST_TINY="1")
    common = ["--repo", str(repo), "--data", str(tmp_path / "data"), "--num_workers", "0"]
    for step, extra in (("prepare", []), ("train", ["--run_dir", str(tmp_path / "run"), "--steps", "2", "--batch", "2",
                                                    "--grad_accum", "1"]),
                        ("eval", ["--run_dir", str(tmp_path / "run"), "--out", str(tmp_path / "res")])):
        out = subprocess.run([sys.executable, "scripts/multi_slu.py", step, *common, *extra], cwd=ROOT,
                             capture_output=True, text=True, env=env, timeout=900)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    stats = json.load(open(tmp_path / "data" / "prepare_stats.json"))
    assert stats["counts"]["de-DE/train"] == 6 and stats["counts"]["ko-KR/validation"] == 3
    res = json.load(open(tmp_path / "res" / "multi_eval.json"))
    assert res["per_lang"]["ko-KR"]["seen_in_training"] is False and "zero_shot_mean_accuracy" in res


def test_jax_head_learns_and_matches_folds(tmp_path):
    rng = np.random.default_rng(0)
    rows, ids, ac = [], [], []
    for c, label in (("slurp", 1), ("ami", 0), ("minds14", 1), ("vox1", 0)):
        for k in range(200):
            uid = f"{c}/{k}"
            x = rng.normal(size=16)
            x[0] += 2.0 if label else -2.0
            rows.append({"id": uid, "corpus": c, "label": label, "group": f"{c}{k % 20}", "duration": 3.0,
                         "hyp": "one two three four"})
            ids.append(uid)
            ac.append(x)
    d = tmp_path / "ddsd"
    d.mkdir()
    np.savez_compressed(d / "features.npz", ids=np.array(ids), acoustic=np.array(ac, np.float32),
                        channel=np.zeros((len(ids), 162), np.float32), length=np.zeros((len(ids), 3), np.float32))
    with open(d / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = subprocess.run([sys.executable, "scripts/ddsd_jax.py", "--feat_dir", str(d), "--per_corpus", "200"],
                         cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.load(open(d / "ddsd_jax.json"))
    assert len(res["folds"]) == 4 and res["backend"].startswith("jax")
    for f in res["folds"].values():
        assert f["jax_mlp"]["cross_corpus"]["auc"] > 0.9  # real signal is learnable in JAX
