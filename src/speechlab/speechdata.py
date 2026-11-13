"""Manifests and batching for the speech LLM and the ASR baseline."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from . import slurp as S
from .audio import load_audio

MAX_SECONDS = 30.0  # whisper encoder window


def librispeech_items(root, part: str) -> list[dict]:
    base = Path(root) / part
    items = []
    for tr in sorted(base.rglob("*.trans.txt")):
        for line in open(tr):
            uid, text = line.strip().split(" ", 1)
            items.append({"id": uid, "path": str(tr.parent / f"{uid}.flac"), "task": "asr",
                          "text": text.lower(), "source": f"librispeech/{part}"})
    return items


def slurp_items(slurp_dir, audio_dir, split: str, tasks=("asr", "slu"), farfield=(False, True),
                assign_one_task_seed: int | None = None, p_slu: float = 0.6) -> list[dict]:
    """One item per (recording, task). With assign_one_task_seed set, each
    recording gets a single task (SLU with prob p_slu) - used for training."""
    rng = random.Random(assign_one_task_seed)
    items = []
    for ex in S.load_jsonl(Path(slurp_dir) / f"{split}.jsonl"):
        tgt = S.target_to_text(S.gold_target(ex))
        for r in ex["recordings"]:
            far = not r["file"].endswith("-headset.flac")
            if far not in farfield:
                continue
            base = {"id": r["file"], "path": str(Path(audio_dir) / r["file"]), "farfield": far,
                    "slurp_id": ex["slurp_id"], "source": f"slurp/{split}"}
            ts = tasks
            if assign_one_task_seed is not None:
                ts = ("slu",) if rng.random() < p_slu else ("asr",)
            for t in ts:
                items.append({**base, "task": t, "text": tgt if t == "slu" else S.transcript(ex)})
    return items


def filter_by_duration(items: list[dict], cache_path=None, max_seconds=MAX_SECONDS,
                       write_cache: bool = True) -> tuple[list[dict], int]:
    import soundfile as sf

    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = json.load(open(cache_path))
    for it in items:
        if it["path"] not in cache:
            info = sf.info(it["path"])
            cache[it["path"]] = info.frames / info.samplerate
    if cache_path and write_cache:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    keep = [it for it in items if cache[it["path"]] <= max_seconds]
    for it in keep:
        it["seconds"] = cache[it["path"]]
    return keep, len(items) - len(keep)


class ItemDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        w = load_audio(it["path"])[: int(MAX_SECONDS * 16000)]
        return i, w


class Collator:
    """Whisper log-mel features (padded to 30 s as the encoder requires) +
    prompt token pieces."""

    def __init__(self, feature_extractor, tokenizer, items, with_target: bool):
        from .speechllm import prompt_pieces

        self.fe, self.tok, self.items, self.with_target = feature_extractor, tokenizer, items, with_target
        self.prompt_pieces = prompt_pieces

    def __call__(self, batch):
        idx = [b[0] for b in batch]
        wavs = [b[1] for b in batch]
        feats = self.fe(wavs, sampling_rate=16000, return_tensors="pt").input_features
        pieces = [self.prompt_pieces(self.tok, self.items[i]["task"],
                                     self.items[i]["text"] if self.with_target else None) for i in idx]
        return idx, feats, [len(w) for w in wavs], pieces


def length_grouped_order(items, batch_size: int, seed: int) -> list[int]:
    """Shuffle, then sort within mega-batches by duration to cut padding."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    mega = batch_size * 50
    out = []
    for s in range(0, len(order), mega):
        chunk = sorted(order[s : s + mega], key=lambda i: items[i].get("seconds", 0.0))
        batches = [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
        rng.shuffle(batches)
        out.extend(i for b in batches for i in b)
    return [int(i) for i in out]
