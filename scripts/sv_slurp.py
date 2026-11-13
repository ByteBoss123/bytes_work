"""Speaker verification on real far-field voice-assistant audio (SLURP).

Steps
  split : speaker-disjoint train/test speaker split (written once, reused)
  eval  : embed test speakers + cohort, score 4 enrollment/test mic
          conditions with cosine and AS-norm, report EER / minDCF / FRR at
          fixed false-accept rates
  noise : same, with real babble (3 overlapped SLURP train-speaker
          utterances) mixed into the far-field test audio at fixed SNRs

Example
  python scripts/sv_slurp.py split --slurp_dir D/slurp_repo/dataset/slurp --out_dir results/sv
  python scripts/sv_slurp.py eval  --slurp_dir ... --audio_dir D/slurp_audio/slurp_real --out_dir results/sv
  python scripts/sv_slurp.py eval  ... --ckpt runs/ecapa_ft/embedding_model.pt --tag finetuned
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import slurp as S  # noqa: E402
from speechlab.audio import load_audio, make_babble, mix_at_snr  # noqa: E402
from speechlab.metrics import eer, min_dcf, rates_at_threshold, threshold_for_far  # noqa: E402

CONDITIONS = [  # (name, enroll_farfield, test_farfield)
    ("headset->headset", False, False),
    ("headset->farfield", False, True),
    ("farfield->farfield", True, True),
    ("farfield->headset", True, False),
    ("mixed->farfield", "mixed", True),   # enroll on 2 headset + 1 far-field take
]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "eval", "noise"])
    ap.add_argument("--slurp_dir", required=True)
    ap.add_argument("--audio_dir")
    ap.add_argument("--out_dir", default="results/sv")
    ap.add_argument("--n_test_speakers", type=int, default=40)
    ap.add_argument("--min_per_condition", type=int, default=30)
    ap.add_argument("--n_enroll", type=int, default=3)
    ap.add_argument("--cohort_size", type=int, default=3000)
    ap.add_argument("--max_test_per_speaker", type=int, default=None)
    ap.add_argument("--snrs", default="20,10,5,0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="pretrained")
    ap.add_argument("--num_workers", type=int, default=4)
    return ap.parse_args()


def load_split(out_dir):
    sp = json.load(open(Path(out_dir) / "speaker_split.json"))
    return set(sp["train_speakers"]), set(sp["test_speakers"])


def summarize(scores, labels):
    e, thr = eer(scores, labels)
    out = {
        "n_trials": int(len(labels)),
        "n_target": int(labels.sum()),
        "eer_percent": round(100 * e, 3),
        "min_dcf_p0.01": round(min_dcf(scores, labels), 4),
    }
    for far in (0.01, 0.001):
        t = threshold_for_far(scores, labels, far)
        r = rates_at_threshold(scores, labels, t)
        out[f"frr_percent_at_far{far}"] = round(100 * r["false_reject_rate"], 3)
    return out


def score_condition(trials, emb_of, cohort):
    from speechlab.speaker import as_norm, cosine_matrix, trial_scores

    enroll = np.stack([np.mean([emb_of(f) for f in fs], axis=0) for fs in trials.enroll_files])
    test = np.stack([emb_of(f) for f in trials.test_files])
    raw = cosine_matrix(enroll, test)
    res = {}
    for name, mat in (("cosine", raw), ("as_norm", as_norm(raw, enroll, test, cohort))):
        s, y = trial_scores(mat, trials.enroll_speakers, trials.test_speakers, trials.excluded)
        res[name] = summarize(s, y)
    return res


def main():
    a = get_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = S.load_recordings(a.slurp_dir)

    if a.step == "split":
        train, test = S.speaker_disjoint_split(recs, a.n_test_speakers, a.min_per_condition, seed=a.seed)
        n_tr = sum(r.speaker in train for r in recs)
        n_te = sum(r.speaker in test for r in recs)
        json.dump({"seed": a.seed, "train_speakers": sorted(train), "test_speakers": sorted(test),
                   "n_train_recordings": n_tr, "n_test_recordings": n_te},
                  open(out / "speaker_split.json", "w"), indent=2)
        print(f"train speakers {len(train)} ({n_tr} recs) | test speakers {len(test)} ({n_te} recs) | overlap {len(train & test)}")
        return

    import torch
    from speechlab.speaker import embed_files, load_ecapa

    train_spk, test_spk = load_split(out)
    assert not train_spk & test_spk
    audio = Path(a.audio_dir)
    model = load_ecapa(device=a.device)
    if a.ckpt:
        model.embedding_model.load_state_dict(torch.load(a.ckpt, map_location=a.device))

    rng = random.Random(a.seed)
    train_recs = [r for r in recs if r.speaker in train_spk]
    cohort_recs = rng.sample(train_recs, min(a.cohort_size, len(train_recs)))
    cache = out / f"emb_{a.tag}_clean.npz"
    test_files = sorted({r.file for r in recs if r.speaker in test_spk})
    if cache.exists():
        z = np.load(cache)
        files, emb = list(z["files"]), z["emb"]
    else:
        files = test_files + [r.file for r in cohort_recs]
        emb = embed_files(model, [audio / f for f in files], a.device, num_workers=a.num_workers)
        np.savez_compressed(cache, files=np.array(files), emb=emb)
    pos = {f: i for i, f in enumerate(files)}
    cohort = emb[[pos[r.file] for r in cohort_recs]]

    if a.step == "eval":
        result = {"tag": a.tag, "ckpt": a.ckpt, "n_test_speakers": len(test_spk), "n_enroll_utts": a.n_enroll,
                  "cohort": len(cohort_recs), "conditions": {}}
        for name, ef, tf in CONDITIONS:
            trials = S.build_trials(recs, test_spk, ef, tf, a.n_enroll, a.max_test_per_speaker, seed=a.seed)
            result["conditions"][name] = score_condition(trials, lambda f: emb[pos[f]], cohort)
            n_ex = sum(len(v) for v in trials.excluded.values())
            result["conditions"][name]["same_take_trials_excluded"] = n_ex
            print(name, json.dumps(result["conditions"][name]))
        json.dump(result, open(out / f"sv_slurp_{a.tag}.json", "w"), indent=2)
        return

    # ---- noise robustness: headset enrollment, babble-corrupted far-field test
    hs = [r for r in train_recs if not r.farfield]
    pool_recs = rng.sample(hs, min(300, len(hs)))
    pool = [load_audio(audio / r.file) for r in pool_recs]
    trials = S.build_trials(recs, test_spk, False, True, a.n_enroll,
                            a.max_test_per_speaker or 100, seed=a.seed)
    result = {"tag": a.tag, "condition": "headset->farfield + babble", "babble_source":
              "3 overlapped SLURP utterances from train-split speakers", "snr_db": {}}
    clean = score_condition(trials, lambda f: emb[pos[f]], cohort)
    result["snr_db"]["clean"] = clean
    for snr in [float(x) for x in a.snrs.split(",")]:
        def transform(i, w, snr=snr):
            g = np.random.default_rng(a.seed * 1_000_003 + i)
            noise = make_babble([pool[k] for k in g.choice(len(pool), 3, replace=False)], len(w), g)
            return mix_at_snr(w, noise, snr, g)

        noisy = embed_files(model, [audio / f for f in trials.test_files], a.device,
                            num_workers=a.num_workers, transform=transform)
        npos = {f: i for i, f in enumerate(trials.test_files)}

        def emb_of(f, noisy=noisy, npos=npos):
            return noisy[npos[f]] if f in npos else emb[pos[f]]

        result["snr_db"][str(snr)] = score_condition(trials, emb_of, cohort)
        print(snr, json.dumps(result["snr_db"][str(snr)]))
    json.dump(result, open(out / f"sv_slurp_noise_{a.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
