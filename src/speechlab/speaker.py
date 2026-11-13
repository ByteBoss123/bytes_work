"""ECAPA-TDNN speaker embeddings, cosine scoring and adaptive score norm."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class EcapaEmbedder(nn.Module):
    """fbank -> sentence mean norm -> ECAPA-TDNN, i.e. the same path as
    speechbrain's EncoderClassifier.encode_batch(normalize=False), exposed as
    a plain nn.Module so it can be fine-tuned and wrapped in DDP."""

    def __init__(self, compute_features, mean_var_norm, embedding_model):
        super().__init__()
        self.compute_features = compute_features
        self.mean_var_norm = mean_var_norm
        self.embedding_model = embedding_model

    def forward(self, wavs: torch.Tensor, rel_lens: torch.Tensor) -> torch.Tensor:
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, rel_lens)
        emb = self.embedding_model(feats, rel_lens)
        return emb.squeeze(1)


def load_ecapa(savedir: str = "pretrained/ecapa", device: str = "cuda") -> EcapaEmbedder:
    import os

    if os.environ.get("SPEECHLAB_TEST_RANDOM_ECAPA") == "1":
        # unit tests only: same architecture, random weights, no download
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        torch.manual_seed(0)
        m = EcapaEmbedder(Fbank(n_mels=80), InputNormalization(norm_type="sentence", std_norm=False),
                          ECAPA_TDNN(80, channels=[64, 64, 64, 64, 192], lin_neurons=192))
        return m.to(device).eval()
    from speechbrain.inference.speaker import EncoderClassifier

    clf = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=savedir, run_opts={"device": device})
    m = EcapaEmbedder(clf.mods.compute_features, clf.mods.mean_var_norm, clf.mods.embedding_model)
    return m.to(device).eval()


def pad_batch(wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(w) for w in wavs)
    out = torch.zeros(len(wavs), n)
    for i, w in enumerate(wavs):
        out[i, : len(w)] = torch.from_numpy(w)
    rel = torch.tensor([len(w) / n for w in wavs], dtype=torch.float32)
    return out, rel


class _FileDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from .audio import load_audio

        w = load_audio(self.paths[i])
        if self.transform is not None:
            w = self.transform(i, w)
        return i, w


def _collate(batch):
    idx = [b[0] for b in batch]
    x, rel = pad_batch([b[1] for b in batch])
    return torch.tensor(idx), x, rel


def duration_batches(paths, max_batch_seconds: float, max_batch: int = 64) -> list[list[int]]:
    """Length-sorted batches capped by total padded audio (largest item x
    count), so a few multi-minute utterances cannot exhaust GPU memory."""
    import soundfile as sf

    dur = []
    for p in paths:
        info = sf.info(str(p))
        dur.append(info.frames / info.samplerate)
    order = np.argsort(dur)
    batches, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * dur[i] > max_batch_seconds or len(cur) >= max_batch):
            batches.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def embed_files(model: nn.Module, paths, device: str, max_batch_seconds: float = 400.0, num_workers: int = 4,
                transform=None, progress: bool = True, fp16: bool = False) -> np.ndarray:
    """Embed full-length audio files with parallel loading. transform(i, wav)
    -> wav is applied in the workers (used for noise injection)."""
    paths = list(paths)
    ds = _FileDataset(paths, transform)
    batches = duration_batches(paths, max_batch_seconds)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=num_workers, collate_fn=_collate)
    out = np.zeros((len(ds), 192), dtype=np.float32)
    it = dl
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(dl, desc="embed", mininterval=30)
    for idx, x, rel in it:
        with torch.autocast("cuda", enabled=fp16 and device.startswith("cuda"), dtype=torch.float16):
            e = model(x.to(device, non_blocking=True), rel.to(device))
        out[idx.numpy()] = e.float().cpu().numpy()
    return out


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2n(a) @ l2n(b).T


def as_norm(raw: np.ndarray, enroll: np.ndarray, test: np.ndarray, cohort: np.ndarray, top_k: int = 300) -> np.ndarray:
    """Adaptive symmetric score normalisation (AS-norm, Matejka et al. 2017).

    raw: (n_enroll, n_test) cosine scores. Cohort statistics use the top_k
    most similar cohort embeddings of each side.
    """
    ce = cosine_matrix(enroll, cohort)
    ct = cosine_matrix(test, cohort)
    k = min(top_k, cohort.shape[0])
    te = -np.sort(-ce, axis=1)[:, :k]
    tt = -np.sort(-ct, axis=1)[:, :k]
    mu_e, sd_e = te.mean(1, keepdims=True), te.std(1, keepdims=True) + 1e-6
    mu_t, sd_t = tt.mean(1)[None, :], tt.std(1)[None, :] + 1e-6
    return 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t) / sd_t)


def trial_scores(score_mat: np.ndarray, enroll_speakers, test_speakers, excluded: dict):
    """Flatten an (n_model, n_test) score matrix into (scores, labels)."""
    es = np.asarray(enroll_speakers)[:, None]
    ts = np.asarray(test_speakers)[None, :]
    labels = (es == ts).astype(np.int8)
    mask = np.ones_like(labels, dtype=bool)
    for i, bad in excluded.items():
        mask[i, list(bad)] = False
    return score_mat[mask], labels[mask]


class AAMSoftmaxHead(nn.Module):
    """Additive angular margin softmax (ArcFace-style) for fine-tuning."""

    def __init__(self, dim: int, n_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.size(1)).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y)
