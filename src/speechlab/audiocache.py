"""Decode a file list once into a single int16 memmap so training workers
slice arrays instead of decoding FLAC (Kaggle gives 4 vCPUs for 2 GPUs)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .audio import SR, load_audio


class _Dec(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        w = load_audio(self.paths[i], SR)
        return i, np.clip(np.round(w * 32767), -32768, 32767).astype(np.int16)


def build(paths: list, keys: list[str], cache_dir, num_workers: int = 4) -> None:
    cache_dir = Path(cache_dir)
    if (cache_dir / "index.json").exists():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    dl = torch.utils.data.DataLoader(_Dec(list(paths)), batch_size=None, num_workers=num_workers)
    index, off = {}, 0
    tmp = open(cache_dir / "audio.int16", "wb")
    for i, a in dl:
        a = a.numpy() if isinstance(a, torch.Tensor) else a
        tmp.write(a.tobytes())
        index[keys[i]] = [off, len(a)]
        off += len(a)
    tmp.close()
    json.dump({"sr": SR, "n_samples": off, "items": index}, open(cache_dir / "index.json.tmp", "w"))
    (cache_dir / "index.json.tmp").rename(cache_dir / "index.json")  # atomic "done" marker


class Reader:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        meta = json.load(open(self.cache_dir / "index.json"))
        self.items = meta["items"]
        self._mm = None

    def __contains__(self, key):
        return key in self.items

    def get(self, key: str) -> np.ndarray:
        if self._mm is None:  # opened lazily so each DataLoader worker has its own handle
            self._mm = np.memmap(self.cache_dir / "audio.int16", dtype=np.int16, mode="r")
        off, n = self.items[key]
        return self._mm[off : off + n].astype(np.float32) / 32767.0
