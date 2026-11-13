"""Run the unmodified official SLURP evaluator (scripts/evaluation/evaluate.py
from github.com/pswietojanski/slurp) with current jiwer.

The evaluator was written for jiwer 2.0 (wer(truth=..., hypothesis=...)).
Newer jiwer renamed the arguments to reference/hypothesis; this shim only
adapts the keyword names, the metric code itself is untouched.

Usage: python scripts/slurp_official_eval.py SLURP_REPO GOLD.jsonl PRED.jsonl [OUT.json]
"""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import jiwer

_orig = jiwer.wer


def _compat_wer(*args, truth=None, hypothesis=None, **kw):
    if truth is not None:
        return _orig(reference=truth, hypothesis=hypothesis, **kw)
    return _orig(*args, hypothesis=hypothesis, **kw) if hypothesis is not None else _orig(*args, **kw)


jiwer.wer = _compat_wer


def main():
    repo, gold, pred = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    out_json = sys.argv[4] if len(sys.argv) > 4 else None
    ev = repo / "scripts" / "evaluation"
    sys.path.insert(0, str(ev))
    sys.argv = ["evaluate.py", "-g", gold, "-p", pred, "--table-layout", "csv"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ev / "evaluate.py"), run_name="__main__")
    text = buf.getvalue()
    print(text)
    # parse "OVERALL,P,R,F" rows that follow each table header
    res, label = {}, None
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) >= 4 and parts[1] == "Precision":
            label = parts[0].lower()
        elif label and parts[0] == "OVERALL":
            res[label] = {"precision": float(parts[1]), "recall": float(parts[2]), "f1": float(parts[3])}
            label = None
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
