"""SLURP helpers: annotations, speaker metadata, speaker-disjoint splits,
verification trial construction, and the JSON target format shared by the
speech LLM, the text SLU model, and the GRPO reward.

SLURP release facts this module relies on (verified against the GitHub
release in dataset/slurp/):
  * train/devel/test.jsonl list every recording of an utterance under
    "recordings"; headset files end in "-headset.flac", the far-field twin of
    the same take has the same name without the suffix.
  * metadata.json maps every recording to a speaker id ("usrid") and a
    recording-session id ("recid") shared by the headset/far-field twins.
  * The official splits share speakers, so speaker verification needs its own
    speaker-disjoint split (built here).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------- loading
def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_entities(example: dict) -> list[dict]:
    """Entities in the official evaluation format (lower-cased surface tokens)."""
    toks = example["tokens"]
    return [
        {"type": e["type"], "filler": " ".join(toks[i]["surface"].lower() for i in e["span"])}
        for e in example["entities"]
    ]


def gold_target(example: dict) -> dict:
    return {
        "scenario": example["scenario"],
        "action": example["action"],
        "entities": gold_entities(example),
    }


def target_to_text(target: dict) -> str:
    """Canonical compact JSON string the models are trained to emit."""
    return json.dumps(
        {"scenario": target["scenario"], "action": target["action"], "entities": target["entities"]},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def transcript(example: dict) -> str:
    return " ".join(t["surface"] for t in example["tokens"]).lower()


@dataclass(frozen=True)
class Recording:
    file: str
    speaker: str
    recid: int
    farfield: bool
    slurp_id: int
    split: str


def load_recordings(slurp_dir, splits=("train", "devel", "test")) -> list[Recording]:
    slurp_dir = Path(slurp_dir)
    meta = json.load(open(slurp_dir / "metadata.json"))
    rec_meta = {}
    for v in meta.values():
        for f, r in v["recordings"].items():
            rec_meta[f] = r
    out = []
    for sp in splits:
        for ex in load_jsonl(slurp_dir / f"{sp}.jsonl"):
            for r in ex["recordings"]:
                m = rec_meta[r["file"]]
                out.append(
                    Recording(
                        file=r["file"],
                        speaker=m["usrid"],
                        recid=int(m["recid"]),
                        farfield=not r["file"].endswith("-headset.flac"),
                        slurp_id=int(ex["slurp_id"]),
                        split=sp,
                    )
                )
    return out


# ------------------------------------------------- speaker-disjoint split
def speaker_disjoint_split(
    recs: list[Recording],
    n_test_speakers: int = 40,
    min_per_condition: int = 30,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Pick test speakers among those with enough headset AND far-field audio.

    Returns (train_speakers, test_speakers); the two sets never overlap.
    Speakers with too little audio for a fair test go to train.
    """
    per = defaultdict(lambda: [0, 0])
    for r in recs:
        per[r.speaker][int(r.farfield)] += 1
    eligible = sorted(s for s, (h, f) in per.items() if h >= min_per_condition and f >= min_per_condition)
    if len(eligible) < n_test_speakers:
        raise ValueError(f"only {len(eligible)} eligible speakers for {n_test_speakers} requested")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    test = set(eligible[:n_test_speakers])
    train = set(per) - test
    return train, test


# ------------------------------------------------------- trial building
@dataclass
class TrialSet:
    """Enrollment models x test recordings; every pair is a trial."""

    enroll_speakers: list[str]            # model i belongs to this speaker
    enroll_files: list[list[str]]         # files averaged into model i
    test_files: list[str]
    test_speakers: list[str]
    excluded: dict                        # (model_idx) -> set of test idx not scored


def build_trials(
    recs: list[Recording],
    test_speakers: set[str],
    enroll_farfield: bool,
    test_farfield: bool,
    n_enroll: int = 3,
    max_test_per_speaker: int | None = None,
    seed: int = 0,
) -> TrialSet:
    """Full cross-product trials with leakage guard.

    Leakage guard: a test recording is never scored against a model whose
    enrollment contains the same take (same recid), because the headset and
    far-field twins are simultaneous recordings of one utterance.
    """
    rng = random.Random(seed)
    by_spk = defaultdict(lambda: {False: [], True: []})
    for r in recs:
        if r.speaker in test_speakers:
            by_spk[r.speaker][r.farfield].append(r)
    for s in by_spk:
        for c in (False, True):
            by_spk[s][c].sort(key=lambda r: r.file)

    enroll_speakers, enroll_files, enroll_recids = [], [], []
    for s in sorted(by_spk):
        pool = by_spk[s][enroll_farfield]
        chosen = rng.sample(pool, n_enroll)
        enroll_speakers.append(s)
        enroll_files.append([r.file for r in chosen])
        enroll_recids.append({r.recid for r in chosen})

    test_recs = []
    for s in sorted(by_spk):
        pool = list(by_spk[s][test_farfield])
        if max_test_per_speaker and len(pool) > max_test_per_speaker:
            pool = rng.sample(pool, max_test_per_speaker)
            pool.sort(key=lambda r: r.file)
        test_recs.extend(pool)

    excluded = {}
    for i, rid in enumerate(enroll_recids):
        bad = {j for j, r in enumerate(test_recs) if r.speaker == enroll_speakers[i] and r.recid in rid}
        if bad:
            excluded[i] = bad
    return TrialSet(
        enroll_speakers=enroll_speakers,
        enroll_files=enroll_files,
        test_files=[r.file for r in test_recs],
        test_speakers=[r.speaker for r in test_recs],
        excluded=excluded,
    )


# ------------------------------------------------------ parsing outputs
def parse_prediction(text: str) -> dict | None:
    """Parse a model completion into {scenario, action, entities} or None."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    scen, act, ents = obj.get("scenario"), obj.get("action"), obj.get("entities", [])
    if not isinstance(scen, str) or not isinstance(act, str) or not isinstance(ents, list):
        return None
    clean = []
    for e in ents:
        if isinstance(e, dict) and isinstance(e.get("type"), str) and isinstance(e.get("filler"), str):
            clean.append({"type": e["type"], "filler": " ".join(e["filler"].lower().split())})
    return {"scenario": scen, "action": act, "entities": clean}


def prediction_record(file: str, pred: dict | None) -> dict:
    """One line of the official SLURP prediction file."""
    if pred is None:
        pred = {"scenario": "none", "action": "none", "entities": []}
    return {"file": file, "scenario": pred["scenario"], "action": pred["action"], "entities": pred["entities"]}


def entity_f1(gold: list[dict], pred: list[dict]) -> float:
    g = [(e["type"], e["filler"]) for e in gold]
    p = [(e["type"], e["filler"]) for e in pred]
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    gc = defaultdict(int)
    for x in g:
        gc[x] += 1
    tp = 0
    for x in p:
        if gc[x] > 0:
            tp += 1
            gc[x] -= 1
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def slu_reward(completion: str, gold: dict, w_format: float = 0.1, w_intent: float = 0.45, w_ent: float = 0.45) -> float:
    """Verifiable reward for GRPO: format + exact intent + entity F1.

    Unparseable output gets 0. The intent term needs scenario AND action to
    match, mirroring the official intent (scenario_action) metric.
    """
    pred = parse_prediction(completion)
    if pred is None:
        return 0.0
    intent_ok = pred["scenario"] == gold["scenario"] and pred["action"] == gold["action"]
    return w_format + w_intent * float(intent_ok) + w_ent * entity_f1(gold["entities"], pred["entities"])
