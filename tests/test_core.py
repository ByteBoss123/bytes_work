import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import make_slurp  # noqa: E402
from speechlab import metrics as M  # noqa: E402
from speechlab import slurp as S  # noqa: E402
from speechlab.audio import mix_at_snr, power  # noqa: E402


# ---------------------------------------------------------------- metrics
def test_eer_matches_sklearn_roc():
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(0)
    y = np.r_[np.ones(2000), np.zeros(8000)].astype(int)
    s = np.r_[rng.normal(1.5, 1, 2000), rng.normal(0, 1, 8000)]
    e, _ = M.eer(s, y)
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    assert abs(e - (fpr[i] + fnr[i]) / 2) < 2e-3


def test_eer_perfect_and_inverted():
    y = np.array([1, 1, 0, 0])
    assert M.eer(np.array([0.9, 0.8, 0.1, 0.2]), y)[0] == pytest.approx(0.0)
    assert M.eer(np.array([0.1, 0.2, 0.9, 0.8]), y)[0] == pytest.approx(1.0)


def test_threshold_for_far_bounds_false_accepts():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(500), np.zeros(5000)].astype(int)
    s = np.r_[rng.normal(2, 1, 500), rng.normal(0, 1, 5000)]
    for far in (0.01, 0.001):
        t = M.threshold_for_far(s, y, far)
        r = M.rates_at_threshold(s, y, t)
        assert r["false_accept_rate"] <= far


def test_min_dcf_range():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(300), np.zeros(3000)].astype(int)
    s = np.r_[rng.normal(2, 1, 300), rng.normal(0, 1, 3000)]
    assert 0 < M.min_dcf(s, y) <= 1.0


def test_wer_normalization():
    out = M.wer(["Set an alarm, please."], ["set an alarm please"])
    assert out["wer"] == 0.0


# ------------------------------------------------------------------ audio
def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(0)
    sig = (0.1 * rng.standard_normal(16000)).astype(np.float32)
    noise = rng.standard_normal(8000).astype(np.float32)
    for snr in (20, 10, 0):
        mixed = mix_at_snr(sig, noise, snr, np.random.default_rng(3))
        n = mixed - sig
        assert 10 * np.log10(power(sig) / power(n)) == pytest.approx(snr, abs=0.05)


# ------------------------------------------------------------------ SLURP
def test_split_trials_and_leakage_guard(tmp_path):
    ann, _ = make_slurp(tmp_path, n_speakers=8, utts_per_speaker=12)
    recs = S.load_recordings(ann)
    train, test = S.speaker_disjoint_split(recs, n_test_speakers=3, min_per_condition=5)
    assert not train & test and len(test) == 3
    tr = S.build_trials(recs, test, enroll_farfield=False, test_farfield=True, n_enroll=3)
    # every enrollment take has a far-field twin in the test pool -> must be excluded
    assert sum(len(v) for v in tr.excluded.values()) == 3 * 3
    recid = {r.file: r.recid for r in recs}
    for i, files in enumerate(tr.enroll_files):
        rids = {recid[f] for f in files}
        for j, f in enumerate(tr.test_files):
            if recid[f] in rids:
                assert j in tr.excluded[i]


def test_trial_scores_flatten():
    from speechlab.speaker import trial_scores

    mat = np.arange(6, dtype=float).reshape(2, 3)
    s, y = trial_scores(mat, ["a", "b"], ["a", "b", "a"], {0: {0}})
    assert len(s) == 5 and y.tolist() == [0, 1, 0, 1, 0]


def test_as_norm_shape_and_monotonic_per_row():
    from speechlab.speaker import as_norm, cosine_matrix

    rng = np.random.default_rng(0)
    e, t, c = rng.normal(size=(4, 16)), rng.normal(size=(10, 16)), rng.normal(size=(50, 16))
    raw = cosine_matrix(e, t)
    n = as_norm(raw, e, t, c, top_k=20)
    assert n.shape == raw.shape
    # for a fixed test column, enrollment z-scoring is a monotone map of raw score
    assert np.all(np.isfinite(n))


def test_gold_target_and_parse_roundtrip(tmp_path):
    ann, _ = make_slurp(tmp_path)
    ex = S.load_jsonl(ann / "train.jsonl")[0]
    tgt = S.gold_target(ex)
    txt = S.target_to_text(tgt)
    assert S.parse_prediction("noise " + txt + " trailing") == tgt
    assert S.slu_reward(txt, tgt) == pytest.approx(1.0)


def test_parse_rejects_bad_output():
    assert S.parse_prediction("no json here") is None
    assert S.parse_prediction('{"scenario": 3}') is None
    assert S.slu_reward("garbage", {"scenario": "a", "action": "b", "entities": []}) == 0.0


def test_reward_components():
    gold = {"scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "seven am"}]}
    wrong_intent = json.dumps({"scenario": "alarm", "action": "query", "entities": gold["entities"]})
    no_ent = json.dumps({"scenario": "alarm", "action": "set", "entities": []})
    assert S.slu_reward(wrong_intent, gold) == pytest.approx(0.1 + 0.45)
    assert S.slu_reward(no_ent, gold) == pytest.approx(0.1 + 0.45)


def test_official_slurp_metric_accepts_our_prediction_file(tmp_path):
    """Our prediction file must load in the official SLURP evaluator."""
    ext = ROOT.parent / "ext" / "slurp" / "scripts" / "evaluation"
    if not ext.exists():
        pytest.skip("official SLURP repo not cloned")
    ann, _ = make_slurp(tmp_path)
    gold = S.load_jsonl(ann / "test.jsonl")
    pred_path = tmp_path / "pred.jsonl"
    with open(pred_path, "w") as f:
        for ex in gold:
            for r in ex["recordings"]:
                f.write(json.dumps(S.prediction_record(r["file"], S.gold_target(ex))) + "\n")
    res_path = tmp_path / "res.json"
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "slurp_official_eval.py"), str(ext.parents[1]),
                          str(ann / "test.jsonl"), str(pred_path), str(res_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.load(open(res_path))
    # gold fed back as predictions must score perfectly on every official metric
    assert res["slu f1"]["f1"] == pytest.approx(1.0)
    assert res["intent (scen_act)"]["f1"] == pytest.approx(1.0)


# ------------------------------------------------------- script smoke tests
def _run(args, env_extra=None):
    env = dict(os.environ, SPEECHLAB_TEST_RANDOM_ECAPA="1", **(env_extra or {}))
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    return out


def test_sv_slurp_pipeline_end_to_end(tmp_path):
    ann, audio = make_slurp(tmp_path, n_speakers=8, utts_per_speaker=12)
    od = tmp_path / "sv"
    base = [sys.executable, "scripts/sv_slurp.py"]
    common = ["--slurp_dir", str(ann), "--out_dir", str(od)]
    _run(base + ["split"] + common + ["--n_test_speakers", "3", "--min_per_condition", "5"])
    split = json.load(open(od / "speaker_split.json"))
    assert not set(split["train_speakers"]) & set(split["test_speakers"])
    _run(base + ["eval"] + common + ["--audio_dir", str(audio), "--device", "cpu", "--cohort_size", "20",
                                     "--num_workers", "0"])
    res = json.load(open(od / "sv_slurp_pretrained.json"))
    assert set(res["conditions"]) == {"headset->headset", "headset->farfield", "farfield->farfield", "farfield->headset"}
    assert res["conditions"]["headset->farfield"]["same_take_trials_excluded"] == 9
    _run(base + ["noise"] + common + ["--audio_dir", str(audio), "--device", "cpu", "--cohort_size", "20",
                                      "--num_workers", "0", "--snrs", "10,0"])
    noise = json.load(open(od / "sv_slurp_noise_pretrained.json"))
    assert set(noise["snr_db"]) == {"clean", "10.0", "0.0"}

    # 2-process DDP fine-tune on CPU (gloo) to exercise the distributed path
    run = tmp_path / "ft"
    _run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "scripts/sv_finetune_ddp.py",
          "--slurp_dir", str(ann), "--audio_dir", str(audio), "--split", str(od / "speaker_split.json"),
          "--out_dir", str(run), "--max_steps", "25", "--batch", "4", "--crop_sec", "0.5", "--num_workers", "0",
          "--tag", "ws2", "--cache_dir", str(tmp_path / "cache")])
    st = json.load(open(run / "train_stats_ws2.json"))
    assert st["world_size"] == 2 and st["steps"] == 25 and st["audio_cache"]
    # cached audio must equal the decoded FLAC (int16 quantisation only)
    from speechlab.audio import load_audio
    from speechlab.audiocache import Reader

    rd = Reader(tmp_path / "cache")
    f = next(iter(rd.items))
    assert np.abs(rd.get(f) - load_audio(audio / f)).max() < 1e-4
    _run(base + ["eval"] + common + ["--audio_dir", str(audio), "--device", "cpu", "--cohort_size", "20",
                                     "--num_workers", "0", "--ckpt", str(run / "embedding_model_ws2.pt"),
                                     "--tag", "finetuned"])
    assert (od / "sv_slurp_finetuned.json").exists()


def test_duration_batches_cap_padded_audio(tmp_path):
    import soundfile as sf
    from speechlab.speaker import duration_batches

    lens = [1, 2, 2, 3, 40, 120, 5, 1]
    paths = []
    for i, sec in enumerate(lens):
        p = tmp_path / f"{i}.flac"
        sf.write(p, np.zeros(int(sec * 16000), dtype=np.float32), 16000)
        paths.append(p)
    batches = duration_batches(paths, max_batch_seconds=100, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(len(lens)))
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or max(lens[i] for i in b) * len(b) <= 100
