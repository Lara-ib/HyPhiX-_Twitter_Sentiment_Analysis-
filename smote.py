"""Sequence-aware SMOTE in the BERT embedding space.

Classical SMOTE (Chawla et al., 2002) creates a synthetic minority sample as
    x_new = x_i + lambda * (x_nn - x_i),   lambda ~ U(0, 1),
where x_nn is one of the k nearest neighbours of x_i within the same class.

Applied to pooled [CLS] vectors only, SMOTE produces vectors that the CNN and BiLSTM
branches cannot consume, because these branches need the token-level sequence
H in R^{L x d}. This module therefore
  1. finds the k nearest neighbours of each minority sample in the [CLS] space, and
  2. applies the *same* interpolation (same base sample, neighbour and lambda) to the
     complete token-embedding sequence, the attention mask (union of the two masks) and,
     when present, the psycholinguistic vector.
Every synthetic sample therefore has aligned token-level, pooled and psycholinguistic
representations. Oversampling is applied to the training partition only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plan_smote(cls_vecs: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 2018,
               target: str | dict = "auto"):
    """Choose (base, neighbour, lambda, label) for every synthetic sample.

    ``target='auto'`` balances every class to the size of the majority class, like
    imbalanced-learn's SMOTE(sampling_strategy='auto') in the multi-class case.
    Returns arrays base_idx, nn_idx (indices into the original arrays), lam and labels.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    want = ({c: int(n_max) for c in classes} if target == "auto" else {int(c): int(v) for c, v in target.items()})
    bases, nbs, lams, labs = [], [], [], []
    for c, n_c in zip(classes, counts):
        need = want.get(int(c), n_c) - n_c
        if need <= 0:
            continue
        idx_c = np.flatnonzero(y == c)
        kk = min(k, len(idx_c) - 1)
        if kk < 1:
            raise ValueError(f"class {c} has too few samples for SMOTE")
        X = cls_vecs[idx_c].astype(np.float32)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        neigh = nn.kneighbors(X, return_distance=False)[:, 1:]          # drop the sample itself
        b = rng.integers(0, len(idx_c), size=need)
        j = rng.integers(0, kk, size=need)
        bases.append(idx_c[b])
        nbs.append(idx_c[neigh[b, j]])
        lams.append(rng.random(need).astype(np.float32))
        labs.append(np.full(need, c, dtype=np.int64))
    if not bases:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32), e
    return np.concatenate(bases), np.concatenate(nbs), np.concatenate(lams), np.concatenate(labs)


def apply_smote(seq: np.ndarray, mask: np.ndarray, cls_vecs: np.ndarray, y: np.ndarray,
                psycho: np.ndarray | None = None, k: int = 5, seed: int = 2018,
                out_dir: str | Path | None = None, chunk: int = 2048):
    """Create the synthetic samples. ``seq`` may be a NumPy memmap of shape (N, L, d).

    Returns a dict with the synthetic arrays ``seq``, ``mask``, ``cls``, ``psycho`` (or None),
    ``y`` and the provenance arrays ``base``, ``neighbour`` and ``lam``. When ``out_dir`` is
    given, the synthetic token sequences are written to a memory-mapped file there.
    """
    base, nb, lam, ys = plan_smote(cls_vecs, y, k=k, seed=seed)
    n, L, d = len(base), seq.shape[1], seq.shape[2]
    if out_dir is not None and n > 0:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        syn = np.lib.format.open_memmap(out_dir / "smote_seq.npy", mode="w+", dtype=seq.dtype, shape=(n, L, d))
    else:
        syn = np.zeros((n, L, d), dtype=seq.dtype)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        bi, ni, lm = base[s:e], nb[s:e], lam[s:e, None, None]
        a = np.asarray(seq[bi], dtype=np.float32)
        b = np.asarray(seq[ni], dtype=np.float32)
        syn[s:e] = (a + lm * (b - a)).astype(seq.dtype)
    if isinstance(syn, np.memmap):
        syn.flush()
    syn_mask = np.maximum(mask[base], mask[nb]).astype(mask.dtype)
    l1 = lam[:, None]
    syn_cls = cls_vecs[base] + l1 * (cls_vecs[nb] - cls_vecs[base])
    syn_psy = None if psycho is None else psycho[base] + l1 * (psycho[nb] - psycho[base])
    return {"seq": syn, "mask": syn_mask, "cls": syn_cls.astype(np.float32),
            "psycho": None if syn_psy is None else syn_psy.astype(np.float32),
            "y": ys, "base": base, "neighbour": nb, "lam": lam}


def smote_summary(y_before: np.ndarray, y_synthetic: np.ndarray, names) -> list[dict]:
    rows = []
    for c, name in enumerate(names):
        b = int((y_before == c).sum())
        s = int((y_synthetic == c).sum())
        rows.append({"class_index": c, "class": name, "original": b, "synthetic": s, "after_smote": b + s})
    return rows


def check_against_imblearn(cls_vecs: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 2018) -> dict:
    """Sanity check: the number of synthetic samples per class equals imbalanced-learn's
    SMOTE(sampling_strategy='auto', k_neighbors=k) applied to the [CLS] vectors."""
    from imblearn.over_sampling import SMOTE

    _, y_res = SMOTE(k_neighbors=k, random_state=seed).fit_resample(cls_vecs, y)
    _, _, _, ys = plan_smote(cls_vecs, y, k=k, seed=seed)
    ours = np.bincount(np.concatenate([y, ys]), minlength=int(y.max()) + 1)
    theirs = np.bincount(y_res, minlength=int(y.max()) + 1)
    return {"ours": ours.tolist(), "imblearn": theirs.tolist(), "identical_counts": bool((ours == theirs).all())}
