"""Training, prediction and feature extraction.

Stage A (all models): end-to-end training as in the notebooks - batch size 32, AdamW
(lr 2e-5, eps 1e-6, no weight decay), gradient clipping at 1.0, 5 epochs; the epoch with
the best validation accuracy is kept. (In the notebooks ``best_model_state = model.state_dict()``
stores references to the live tensors, so the saved weights were in fact those of the last
epoch; here a real copy of the best epoch is kept. Use ``select_best_by: last`` to reproduce
the notebook behaviour exactly.)

Stage B (``--balance smote``): the fine-tuned encoder is frozen, the token embeddings of
the training partition are extracted, rebalanced with sequence-aware SMOTE (smote.py),
and the head is trained further on the balanced embeddings. Validation and evaluation
data are never oversampled.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


# --------------------------------------------------------------------------- datasets
class ArrayDataset(Dataset):
    """Rows of aligned arrays (ids/mask/psycho/label); arrays may be memmaps."""

    def __init__(self, *arrays):
        self.arrays = [a for a in arrays]
        self.n = len(arrays[0])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return tuple(torch.as_tensor(np.asarray(a[i])) for a in self.arrays)


class FeatureDataset(Dataset):
    """Original + synthetic token embeddings for head training (Stage B)."""

    def __init__(self, seq_orig, mask_orig, y_orig, psy_orig=None, seq_syn=None, mask_syn=None,
                 y_syn=None, psy_syn=None):
        self.parts = [(seq_orig, mask_orig, y_orig, psy_orig)]
        if seq_syn is not None and len(y_syn):
            self.parts.append((seq_syn, mask_syn, y_syn, psy_syn))
        self.offsets = np.cumsum([0] + [len(p[2]) for p in self.parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        j = i - self.offsets[k]
        seq, mask, y, psy = self.parts[k]
        H = torch.as_tensor(np.asarray(seq[j], dtype=np.float32))
        out = [H, torch.as_tensor(mask[j]), torch.as_tensor(y[j])]
        out.append(torch.as_tensor(psy[j]) if psy is not None else torch.zeros(1))
        return tuple(out)


def _unpack(batch, device, has_psycho):
    if has_psycho:
        ids, mask, psy, y = [b.to(device) for b in batch]
    else:
        ids, mask, y = [b.to(device) for b in batch]
        psy = None
    return ids, mask, psy, y


def make_loader(ids, mask, y, psycho=None, batch_size=32, shuffle=False, seed=2018, num_workers=0):
    arrays = [ids, mask] + ([psycho] if psycho is not None else []) + [y]
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ArrayDataset(*arrays), batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers)


def make_optimizer(model, lr, eps=1e-6, weight_decay=0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, eps=eps, weight_decay=weight_decay)


# --------------------------------------------------------------------------- loops
def run_epoch(model, loader, device, optimizer=None, grad_clip=1.0, has_psycho=False,
              extra_loss=None, log_every=200, desc=""):
    """One pass over ``loader``; trains when an optimizer is given. Returns (loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    crit = nn.NLLLoss()
    tot_loss, correct, n = 0.0, 0, 0
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        ids, mask, psy, y = _unpack(batch, device, has_psycho)
        with torch.set_grad_enabled(train):
            logp = model(ids, mask, psy)
            loss = crit(logp, y)
            if train and extra_loss is not None:
                loss = loss + extra_loss()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        tot_loss += float(loss.item()) * len(y)
        correct += int((logp.argmax(1) == y).sum())
        n += len(y)
        if log_every and step % log_every == 0:
            print(f"  {desc} step {step}/{len(loader)} loss={tot_loss / n:.4f} acc={correct / n:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return tot_loss / max(n, 1), correct / max(n, 1)


def fit(model, train_loader, val_loader, device, epochs, lr, eps=1e-6, weight_decay=0.0, grad_clip=1.0,
        has_psycho=False, select_best_by="val_accuracy", extra_loss=None, log=print):
    """Train for ``epochs`` epochs and restore the best (or last) epoch. Returns the history."""
    opt = make_optimizer(model, lr, eps, weight_decay)
    history, best, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, grad_clip, has_psycho, extra_loss,
                                    desc=f"epoch {ep}")
        va_loss, va_acc = run_epoch(model, val_loader, device, None, has_psycho=has_psycho, log_every=0)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_accuracy": tr_acc, "val_loss": va_loss,
                        "val_accuracy": va_acc, "seconds": round(time.time() - t0, 1)})
        log(f"epoch {ep}/{epochs}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s")
        if select_best_by == "last" or va_acc > best:
            best = va_acc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model, loader, device, has_psycho=False):
    """Return class probabilities (N, C) and, for HyPhiX, the gate weights (N, S) or None."""
    model.eval()
    probs, alphas = [], []
    for batch in loader:
        ids, mask, psy, _ = _unpack(batch, device, has_psycho)
        probs.append(model(ids, mask, psy).exp().cpu().numpy())
        a = getattr(model, "last_alpha", None)
        if a is not None:
            alphas.append(a.cpu().numpy())
    return np.concatenate(probs), (np.concatenate(alphas) if alphas else None)


# --------------------------------------------------------------------------- Stage B: SMOTE
@torch.no_grad()
def extract_features(model, ids, mask, device, out_path: Path, batch_size=64, dtype="float16"):
    """Encode token sequences with the (frozen) encoder and store H (N, L, d) as a .npy memmap.
    Also returns the [CLS] vectors (N, d) used for the SMOTE neighbour search."""
    model.eval()
    N, L = ids.shape
    d = model.hidden
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(N, L, d))
    cls = np.zeros((N, d), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        H = model.encode(torch.as_tensor(ids[s:e]).to(device), torch.as_tensor(mask[s:e]).to(device))
        seq[s:e] = H.cpu().numpy().astype(dtype)
        cls[s:e] = H[:, 0, :].float().cpu().numpy()
    seq.flush()
    return seq, cls


class _HeadOnly(nn.Module):
    """Wraps a BERT model so that forward(H, mask, psycho) only runs the head."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_alpha = None

    def forward(self, H, mask, psycho=None):
        out = self.model.head(H, mask, psycho)
        self.last_alpha = getattr(self.model, "last_alpha", None)
        return out


def fit_head_on_features(model, feat_ds, val_loader, device, epochs, lr, batch_size, seed,
                         has_psycho=False, grad_clip=1.0, log=print):
    """Train only the head on (original + SMOTE) token embeddings; select by validation accuracy
    computed end-to-end on the original validation tweets."""
    model.freeze_encoder()
    head = _HeadOnly(model).to(device)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True, generator=g)
    opt = make_optimizer(head, lr, eps=1e-8, weight_decay=0.0)
    crit = nn.NLLLoss()
    history, best, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        head.train()
        t0, tot, cor, n = time.time(), 0.0, 0, 0
        for H, m, y, p in loader:
            H, m, y = H.to(device), m.to(device), y.to(device)
            p = p.to(device) if has_psycho else None
            logp = head(H, m, p)
            loss = crit(logp, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item()) * len(y)
            cor += int((logp.argmax(1) == y).sum())
            n += len(y)
        va_loss, va_acc = run_epoch(model, val_loader, device, None, has_psycho=has_psycho, log_every=0)
        history.append({"epoch": ep, "train_loss": tot / n, "train_accuracy": cor / n, "val_loss": va_loss,
                        "val_accuracy": va_acc, "seconds": round(time.time() - t0, 1)})
        log(f"[SMOTE head] epoch {ep}/{epochs}: train loss {tot / n:.4f} acc {cor / n:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return history
