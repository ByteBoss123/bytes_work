"""Device-directed speech detection (DDSD) with shortcut controls.

No public corpus records assistant-directed and person-directed speech under
matched conditions, so every directed/non-directed contrast built from
public data is confounded with the corpus (mic, room, speakers, topic).
Instead of hiding that, this module measures it:

* leave-corpus-out folds: train on one directed + one non-directed corpus,
  test on a *different* directed + non-directed pair, so recording-channel
  shortcuts learned in training do not carry over;
* shortcut baselines trained on the same folds: a channel-only probe (spectra
  of the quietest frames, i.e. background/microphone), and a length-only
  probe (duration, word count);
* the in-domain vs cross-corpus gap of every model.
A model is only credited with learning directedness to the extent it beats
the shortcut probes on held-out corpora.
"""
from __future__ import annotations

import numpy as np

from .metrics import eer

# --------------------------------------------------------------- features
_MEL = {}


def channel_features(wav: np.ndarray, sr: int = 16000, quiet_frac: float = 0.2) -> np.ndarray:
    """Mean/std log-mel of the quietest frames (background + mic colouration),
    plus noise floor and speech-to-floor level. 162 dims."""
    import torch
    import torchaudio

    key = sr
    if key not in _MEL:
        _MEL[key] = torchaudio.transforms.MelSpectrogram(sr, n_fft=400, hop_length=160, n_mels=80)
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    if x.numel() < 800:
        x = torch.nn.functional.pad(x, (0, 800 - x.numel()))
    m = torch.log(_MEL[key](x) + 1e-8).T.numpy()  # (frames, 80)
    energy = m.mean(1)
    k = max(1, int(len(energy) * quiet_frac))
    quiet = m[np.argsort(energy)[:k]]
    loud = m[np.argsort(energy)[-k:]]
    return np.concatenate([quiet.mean(0), quiet.std(0),
                           [quiet.mean(), loud.mean() - quiet.mean()]]).astype(np.float32)


def length_features(duration_s: float, transcript: str) -> np.ndarray:
    n = len(transcript.split())
    return np.array([duration_s, n, n / max(duration_s, 1e-3)], dtype=np.float32)


# ------------------------------------------------------------- data rules
def keep_utterance(duration_s: float, transcript: str, min_words: int = 4, min_s: float = 1.5,
                   max_s: float = 15.0) -> bool:
    """Same filter for every corpus, applied to ASR output (not references),
    so backchannels ('yeah', 'hmm') cannot make negatives trivially short."""
    return min_s <= duration_s <= max_s and len(transcript.split()) >= min_words


def balanced_sample(rows: list[dict], per_corpus: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted({r["corpus"] for r in rows}):
        rs = [r for r in rows if r["corpus"] == c]
        idx = rng.permutation(len(rs))[:per_corpus]
        out += [rs[i] for i in sorted(idx)]
    return out


def group_split(rows: list[dict], test_frac: float, seed: int):
    """Split by speaker group within each corpus, so one speaker never sits on
    both sides and every corpus (hence both classes) appears in both parts."""
    rng = np.random.default_rng(seed)
    test_g = set()
    for c in sorted({r["corpus"] for r in rows}):
        groups = sorted({r["group"] for r in rows if r["corpus"] == c})
        rng.shuffle(groups)
        test_g |= {(c, g) for g in groups[: max(1, int(len(groups) * test_frac))]}
    tr = [r for r in rows if (r["corpus"], r["group"]) not in test_g]
    te = [r for r in rows if (r["corpus"], r["group"]) in test_g]
    return tr, te


# ----------------------------------------------------------------- models
def _dense_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _text_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=3000, C=4.0))


FEATURE_SETS = ("length_only", "channel_only", "text", "acoustic", "fusion")


def fit_predict(train: list[dict], test: list[dict], feats: dict) -> dict[str, np.ndarray]:
    """feats[id] -> {'channel','length','acoustic'} arrays; rows carry 'hyp'.
    Returns P(directed) on the test rows for every feature set."""
    y = np.array([r["label"] for r in train])
    out = {}
    for name, key in (("length_only", "length"), ("channel_only", "channel"), ("acoustic", "acoustic")):
        Xtr = np.stack([feats[r["id"]][key] for r in train])
        Xte = np.stack([feats[r["id"]][key] for r in test])
        out[name] = _dense_model().fit(Xtr, y).predict_proba(Xte)[:, 1]
    tm = _text_model().fit([r["hyp"] for r in train], y)
    out["text"] = tm.predict_proba([r["hyp"] for r in test])[:, 1]
    out["fusion"] = 0.5 * (out["text"] + out["acoustic"])  # late fusion, no extra fitting
    return out


def score(labels, probs) -> dict:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    e, _ = eer(np.asarray(probs), labels)
    return {"auc": round(float(roc_auc_score(labels, probs)), 4), "eer_percent": round(100 * e, 2),
            "n": int(len(labels)), "n_directed": int(labels.sum())}


def run_folds(rows: list[dict], feats: dict, folds: list[dict], per_corpus: int, seed: int = 0) -> dict:
    """folds: [{'name', 'train': [corpora], 'test': [corpora]}] (disjoint corpora).
    Reports in-domain (speaker-held-out within the train corpora) and
    cross-corpus scores for every feature set."""
    res = {}
    for f in folds:
        assert not set(f["train"]) & set(f["test"]), "train/test corpora must be disjoint"
        tr_all = balanced_sample([r for r in rows if r["corpus"] in f["train"]], per_corpus, seed)
        te = balanced_sample([r for r in rows if r["corpus"] in f["test"]], per_corpus, seed)
        tr, te_in = group_split(tr_all, 0.2, seed)
        p_in = fit_predict(tr, te_in, feats)
        p_x = fit_predict(tr, te, feats)
        fold = {"train_corpora": f["train"], "test_corpora": f["test"], "n_train": len(tr), "models": {}}
        for name in FEATURE_SETS:
            fold["models"][name] = {
                "in_domain": score([r["label"] for r in te_in], p_in[name]),
                "cross_corpus": score([r["label"] for r in te], p_x[name]),
            }
        res[f["name"]] = fold
    return res


def make_folds(present):
    """Leave-corpus-out folds: every (directed, non-directed) train pair is
    tested on the other directed corpus plus a different non-directed one."""
    D = [c for c in ("slurp", "minds14") if c in present]
    N = [c for c in ("ami", "vox1", "libri") if c in present]
    folds = []
    for d_tr in D:
        for n_tr in N:
            for d_te in D:
                for n_te in N:
                    if d_te != d_tr and n_te != n_tr:
                        folds.append({"name": f"train {d_tr}+{n_tr} -> test {d_te}+{n_te}",
                                      "train": [d_tr, n_tr], "test": [d_te, n_te]})
    return folds
