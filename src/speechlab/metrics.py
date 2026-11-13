"""Verification and recognition metrics.

EER / minDCF follow the standard NIST-style definitions used in VoxCeleb
evaluations (p_target=0.01, C_miss=C_fa=1, normalized DCF).
"""
from __future__ import annotations

import re

import numpy as np


def _det_curve(scores: np.ndarray, labels: np.ndarray):
    """Return (thresholds, fnr, fpr) swept over every unique score."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() == labels.max():
        raise ValueError("need both target (1) and non-target (0) trials")
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n_tar, n_non = y.sum(), len(y) - y.sum()
    # threshold t_i = s[i]: accept score >= t. misses = targets with score < t
    cum_tar = np.concatenate([[0], np.cumsum(y)])
    cum_non = np.concatenate([[0], np.cumsum(1 - y)])
    fnr = cum_tar[:-1] / n_tar
    fpr = (n_non - cum_non[:-1]) / n_non
    # append the "reject everything" point
    fnr = np.concatenate([fnr, [1.0]])
    fpr = np.concatenate([fpr, [0.0]])
    thr = np.concatenate([s, [np.inf]])
    return thr, fnr, fpr


def eer(scores, labels) -> tuple[float, float]:
    """Equal error rate (fraction, not %) and the threshold where it occurs."""
    thr, fnr, fpr = _det_curve(scores, labels)
    i = int(np.argmin(np.abs(fnr - fpr)))
    # linear interpolation between neighbouring points for a stable estimate
    j = i + 1 if i + 1 < len(fnr) else i - 1
    d1, d2 = fnr[i] - fpr[i], fnr[j] - fpr[j]
    if d1 != d2 and np.sign(d1) != np.sign(d2):
        a = d1 / (d1 - d2)
        e = fpr[i] + a * (fpr[j] - fpr[i])
    else:
        e = (fnr[i] + fpr[i]) / 2
    return float(e), float(thr[i])


def min_dcf(scores, labels, p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    _, fnr, fpr = _det_curve(scores, labels)
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(dcf.min() / c_def)


def rates_at_threshold(scores, labels, threshold: float) -> dict:
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    acc = scores >= threshold
    return {
        "false_reject_rate": float((~acc[labels]).mean()),
        "false_accept_rate": float(acc[~labels].mean()),
    }


def threshold_for_far(scores, labels, target_far: float) -> float:
    """Smallest threshold whose false-accept rate is <= target_far."""
    non_desc = np.sort(np.asarray(scores, dtype=np.float64)[np.asarray(labels) == 0])[::-1]
    k = int(np.floor(target_far * len(non_desc)))  # false accepts we may allow
    if k >= len(non_desc):
        return float(-np.inf)
    # accept iff score >= thr; placing thr just above the (k+1)-th highest
    # non-target score admits at most k non-targets (ties are rejected)
    return float(np.nextafter(non_desc[k], np.inf))


_PUNCT = re.compile(r"[^a-z0-9' ]+")


def basic_normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def wer(refs: list[str], hyps: list[str], normalizer=basic_normalize) -> dict:
    import jiwer

    r = [normalizer(x) for x in refs]
    h = [normalizer(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x.strip()]
    r = [r[i] for i in keep]
    h = [h[i] if h[i].strip() else "<empty>" for i in keep]
    out = jiwer.process_words(r, h)
    return {
        "wer": float(out.wer),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "n_utts": len(r),
    }


def whisper_normalizer(whisper_name: str = "openai/whisper-small"):
    """Whisper's EnglishTextNormalizer (numbers, spellings, punctuation),
    applied identically to references and hypotheses of every system."""
    import os

    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    if os.environ.get("SPEECHLAB_TEST_TINY") == "1":
        return EnglishTextNormalizer({})
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained(whisper_name).normalize
