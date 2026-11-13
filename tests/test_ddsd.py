"""Directed-speech pipeline: preparation (incl. HF-style parquet audio),
featurisation, and a methodology check that leave-corpus-out evaluation
exposes a planted channel shortcut. Synthetic signals only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from fixtures import make_slurp  # noqa: E402
from speechlab import ddsd  # noqa: E402


def test_channel_features_shape_and_noise_floor():
    rng = np.random.default_rng(0)
    clean = np.r_[np.zeros(8000), 0.3 * np.sin(np.arange(16000) / 5), np.zeros(8000)].astype(np.float32)
    noisy = clean + 0.05 * rng.standard_normal(len(clean)).astype(np.float32)
    a, b = ddsd.channel_features(clean), ddsd.channel_features(noisy)
    assert a.shape == (162,) and b[160] > a[160]  # higher noise floor is visible


def test_keep_rule():
    assert ddsd.keep_utterance(2.0, "set an alarm please")
    assert not ddsd.keep_utterance(2.0, "yeah okay")
    assert not ddsd.keep_utterance(0.8, "set an alarm please")


def _synthetic(corpora, n=300, seed=0):
    """label signal in 'acoustic' dim 0 for every corpus; channel dim 0 is a
    corpus fingerprint that correlates with the label ONLY in training corpora."""
    rng = np.random.default_rng(seed)
    rows, feats = [], {}
    chan_sign = {"A": 1, "B": -1, "C": -1, "D": 1}  # A(+),B(-) train; C(+),D(-) test -> flipped
    for c, label in corpora:
        for k in range(n):
            uid = f"{c}/{k}"
            rows.append({"id": uid, "corpus": c, "label": label, "group": f"{c}{k % 30}",
                         "hyp": ("set alarm" if label else "we think") + f" w{k % 7} x y"})
            ac = rng.normal(size=8)
            ac[0] += 1.5 if label else -1.5
            ch = rng.normal(size=162)
            ch[0] += 3 * chan_sign[c]
            feats[uid] = {"acoustic": ac, "channel": ch, "length": rng.normal(size=3)}
    return rows, feats


def test_leave_corpus_out_exposes_channel_shortcut():
    rows, feats = _synthetic([("A", 1), ("B", 0), ("C", 1), ("D", 0)])
    res = ddsd.run_folds(rows, feats, [{"name": "f", "train": ["A", "B"], "test": ["C", "D"]}], per_corpus=300)
    m = res["f"]["models"]
    assert m["channel_only"]["in_domain"]["auc"] > 0.95       # shortcut looks perfect in-domain
    assert m["channel_only"]["cross_corpus"]["auc"] < 0.2      # and fails once the corpus changes
    assert m["acoustic"]["cross_corpus"]["auc"] > 0.8          # real signal transfers
    assert m["length_only"]["cross_corpus"]["auc"] == pytest.approx(0.5, abs=0.1)
    with pytest.raises(AssertionError):
        ddsd.run_folds(rows, feats, [{"name": "bad", "train": ["A", "B"], "test": ["B", "C"]}], per_corpus=50)


def _hf_parquet(path: Path, config: str, n: int, text_key: str, sr: int, extra: dict):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = []
    for k in range(n):
        buf = io.BytesIO()
        sf.write(buf, (0.1 * np.sin(np.arange(sr * 2) / (3 + k))).astype(np.float32), sr, format="WAV")
        recs.append({"audio": {"bytes": buf.getvalue(), "path": f"{k}.wav"}, text_key: "one two three four five",
                     **{key: f(k) for key, f in extra.items()}})
    d = path / config / ("test" if config == "ihm" else "train")
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(recs), d / "0.parquet")


def test_prepare_and_features_end_to_end(tmp_path):
    data = tmp_path / "data"
    ann, audio = make_slurp(tmp_path / "raw", n_speakers=3, utts_per_speaker=8)
    (data / "slurp_repo" / "dataset").mkdir(parents=True)
    import shutil

    shutil.copytree(ann, data / "slurp_repo" / "dataset" / "slurp")
    (data / "slurp_audio").mkdir()
    shutil.copytree(audio, data / "slurp_audio" / "slurp_real")
    for spk in ("id10270", "id10271"):
        p = data / "vox1" / "wav" / spk / "v"
        p.mkdir(parents=True)
        sf.write(p / "00001.wav", np.zeros(32000, np.float32), 16000)
    ami, minds = tmp_path / "ami", tmp_path / "minds"
    _hf_parquet(ami, "ihm", 4, "text", 16000, {"audio_id": lambda k: f"AMI_{k}", "speaker_id": lambda k: f"S{k % 2}"})
    for cfg in ("en-US", "en-GB", "en-AU"):
        _hf_parquet(minds, cfg, 2, "transcription", 8000, {})
    env = dict(os.environ, SPEECHLAB_TEST_TINY="1")
    out = subprocess.run([sys.executable, "scripts/ddsd_prepare.py", "--data_root", str(data), "--out",
                          str(tmp_path / "ddsd"), "--ami_repo", str(ami), "--minds_repo", str(minds)],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    counts = json.loads(out.stdout.strip().splitlines()[-1])
    assert counts == {"slurp": 12, "vox1": 2, "ami": 4, "minds14": 6}
    w, sr = sf.read(next((tmp_path / "ddsd" / "minds14").glob("*.wav")))
    assert sr == 16000 and len(w) == 32000  # 8 kHz phone audio resampled
    out = subprocess.run([sys.executable, "scripts/ddsd_features.py", "--ddsd_dir", str(tmp_path / "ddsd"),
                          "--out", str(tmp_path / "res"), "--batch", "4", "--num_workers", "0"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    z = np.load(tmp_path / "res" / "features.npz")
    assert z["acoustic"].shape == (24, 32) and z["channel"].shape == (24, 162)


def test_make_folds_uses_present_corpora_only():
    f = ddsd.make_folds({"slurp", "minds14", "ami", "libri"})
    assert len(f) == 4
    for x in f:
        assert not set(x["train"]) & set(x["test"])
        assert "vox1" not in x["train"] + x["test"]
    assert ddsd.make_folds({"slurp", "ami"}) == []


def test_corpus_standardize_removes_corpus_offsets():
    rows = [{"corpus": "A"}] * 50 + [{"corpus": "B"}] * 50
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(5, 1, (50, 3)), rng.normal(-5, 2, (50, 3))]
    Z = ddsd.corpus_standardize(rows, X)
    assert np.allclose(Z[:50].mean(0), 0, atol=1e-6) and np.allclose(Z[50:].std(0), 1, atol=1e-3)


def test_corpus_norm_folds_run():
    rows, feats = _synthetic([("A", 1), ("B", 0), ("C", 1), ("D", 0)])
    res = ddsd.run_folds(rows, feats, [{"name": "f", "train": ["A", "B"], "test": ["C", "D"]}], per_corpus=300,
                         corpus_norm=True)
    # after per-corpus z-scoring the corpus fingerprint carries no label information
    assert abs(res["f"]["models"]["channel_only"]["in_domain"]["auc"] - 0.5) < 0.2
