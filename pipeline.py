"""End-to-end experiment runner used by the command-line scripts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import LABEL_NAMES
from .data import build_data, class_distribution
from .evaluation import classification_table, compute_metrics, save_predictions
from .features import KerasLikeTokenizer, PsycholinguisticExtractor, bert_encode, get_tokenizer
from .models import build_model
from .plots import plot_confusion, plot_history
from .training import (FeatureDataset, extract_features, fit, fit_head_on_features, make_loader, predict)
from .utils import environment_info, get_device, resolve, save_json, set_seed

BERT_MODELS = {"cb", "bb", "hyphix", "bert"}
EVAL_SETS = ["test", "validation", "validation_dedup"]


class Encoded:
    """Model inputs for every partition (ids, mask, psycho, labels)."""

    def __init__(self):
        self.parts = {}

    def add(self, name, ids, mask, y, psycho=None):
        self.parts[name] = {"ids": ids, "mask": mask, "y": y, "psycho": psycho}

    def __getitem__(self, name):
        return self.parts[name]


def encode_all(model_name, cfg, data, pretrained=None, log=print):
    """Tokenise every partition for the chosen model family."""
    enc = Encoded()
    extra = {}
    sets = {"train": data.train, "val": data.val, "test": data.test, "validation": data.eval_file,
            "validation_dedup": data.eval_dedup}
    psy = None
    if model_name == "hyphix" and cfg["hyphix"]["use_psycho"]:
        psy = PsycholinguisticExtractor(cfg["hyphix"]["psycho_categories"])
        extra["psycho_extractor"] = psy
    if model_name in BERT_MODELS:
        tok = get_tokenizer(pretrained or cfg["bert"]["pretrained"])
        L = cfg["baselines"]["bert"]["max_len"] if model_name == "bert" else cfg["bert"]["max_len"]
        for k, df in sets.items():
            ids, mask = bert_encode(df["text"], tok, L)
            p = psy(df["raw_text"].tolist()) if psy is not None else None
            enc.add(k, ids, mask, df["label"].to_numpy(np.int64), p)
        extra.update({"tokenizer": tok, "max_len": L})
    elif model_name in {"cnn", "bilstm"}:
        b = cfg["baselines"][model_name]
        if model_name == "cnn":
            tok = KerasLikeTokenizer(num_words=b["max_words"]).fit(data.train["text"])
            L = max(1, max(len(s) for s in tok.texts_to_sequences(data.train["text"])))
        else:
            n_unique = data.train["text"].nunique()
            tok = KerasLikeTokenizer(num_words=n_unique, oov_token="<OOV>").fit(data.train["text"])
            L = b["seq_len"]
        for k, df in sets.items():
            ids = tok.pad(tok.texts_to_sequences(df["text"]), L)
            enc.add(k, ids, (ids > 0).astype(np.int64), df["label"].to_numpy(np.int64))
        extra.update({"tokenizer": tok, "max_len": L, "vocab_size": tok.vocab_size})
    else:
        raise ValueError(model_name)
    log(f"[encode] {model_name}: sequence length {extra['max_len']}")
    return enc, extra


def train_run(model_name: str, cfg: dict, run_dir: Path, balance: str = "none", pretrained: str | None = None,
              device=None, eval_sets=EVAL_SETS, data=None, save_model=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    seed = cfg["train"]["seed"]
    set_seed(seed)
    device = device or get_device()
    log(f"=== {model_name} | balance={balance} | seed={seed} | device={device} ===")
    data = data or build_data(cfg)
    split_ids = pd.concat([d[["tweet_id", "entity", "sentiment"]].assign(partition=p)
                           for p, d in [("train", data.train), ("val", data.val), ("test", data.test),
                                        ("evaluation_file", data.eval_file)]])
    split_ids.to_csv(run_dir / "split_identifiers.csv", index=False)
    class_distribution(pd.concat([data.train, data.val, data.test])).to_csv(run_dir / "class_distribution.csv", index=False)

    enc, extra = encode_all(model_name, cfg, data, pretrained, log)
    has_psy = enc["train"]["psycho"] is not None
    psy_dim = enc["train"]["psycho"].shape[1] if has_psy else 2
    model = build_model(model_name, cfg, psycho_dim=psy_dim, vocab_size=extra.get("vocab_size"),
                        pretrained=pretrained).to(device)

    # ---------------- hyper-parameters per family
    tr = cfg["train"]
    if model_name in {"cb", "bb", "hyphix"}:
        hp = dict(epochs=tr["epochs"], lr=tr["lr"], eps=tr["adam_eps"], wd=tr["weight_decay"], bs=tr["batch_size"])
    elif model_name == "bert":
        b = cfg["baselines"]["bert"]
        hp = dict(epochs=b["epochs"], lr=b["lr"], eps=b["adam_eps"], wd=0.0, bs=b["batch_size"])
    else:
        b = cfg["baselines"][model_name]
        hp = dict(epochs=b["epochs"], lr=b["lr"], eps=1e-7 if model_name == "cnn" else 1e-8, wd=0.0, bs=b["batch_size"])
    E = lambda k, shuffle=False: make_loader(enc[k]["ids"], enc[k]["mask"], enc[k]["y"], enc[k]["psycho"],
                                             batch_size=hp["bs"], shuffle=shuffle, seed=seed,
                                             num_workers=0)
    extra_loss = model.l2_penalty if hasattr(model, "l2_penalty") else None
    grad_clip = None if model_name == "cnn" else tr["grad_clip"]

    # ---------------- Stage A: end-to-end training
    history = fit(model, E("train", True), E("val"), device, hp["epochs"], hp["lr"], hp["eps"], hp["wd"],
                  grad_clip, has_psy, tr.get("select_best_by", "val_accuracy"), extra_loss, log)
    save_json(history, run_dir / "history_stage_a.json")
    plot_history(history, run_dir / "history_stage_a.png")

    # ---------------- Stage B: SMOTE in the embedding space (training partition only)
    smote_info = None
    if balance == "smote":
        if model_name not in {"cb", "bb", "hyphix"}:
            raise ValueError("SMOTE stage is implemented for the BERT hybrids (cb, bb, hyphix)")
        from .smote import apply_smote, check_against_imblearn, smote_summary

        s = cfg["smote"]
        log("[SMOTE] extracting token embeddings of the training partition with the fine-tuned encoder")
        feat_dir = run_dir / "features"
        seq, cls = extract_features(model, enc["train"]["ids"], enc["train"]["mask"], device,
                                    feat_dir / "train_seq.npy", batch_size=s["batch_size"], dtype=s["feature_dtype"])
        y = enc["train"]["y"]
        syn = apply_smote(seq, enc["train"]["mask"], cls, y, enc["train"]["psycho"], k=s["k_neighbors"],
                          seed=seed, out_dir=feat_dir)
        smote_info = {"per_class": smote_summary(y, syn["y"], LABEL_NAMES), "k_neighbors": s["k_neighbors"],
                      "neighbour_space": "[CLS] embeddings of the fine-tuned encoder",
                      "interpolated": "token-embedding sequence, attention mask (union), [CLS] vector"
                                      + (", psycholinguistic vector" if has_psy else ""),
                      "imblearn_check": check_against_imblearn(cls, y, s["k_neighbors"], seed)}
        np.savez_compressed(feat_dir / "smote_provenance.npz", base=syn["base"], neighbour=syn["neighbour"],
                            lam=syn["lam"], y=syn["y"])
        log(f"[SMOTE] {json.dumps(smote_info['per_class'])}")
        ds = FeatureDataset(seq, enc["train"]["mask"], y, enc["train"]["psycho"], syn["seq"], syn["mask"], syn["y"],
                            syn["psycho"])
        hist_b = fit_head_on_features(model, ds, E("val"), device, s["head_epochs"], s["head_lr"], s["batch_size"],
                                      seed, has_psy, tr["grad_clip"], log)
        save_json(hist_b, run_dir / "history_stage_b_smote.json")
        save_json(smote_info, run_dir / "smote_summary.json")

    # ---------------- evaluation on every set
    results = {}
    for name in eval_sets:
        probs, alphas = predict(model, E(name), device, has_psy)
        df = data.get(name)
        preds = save_predictions(df, probs, run_dir / f"predictions_{name}.csv", alphas)
        m = compute_metrics(preds["true_label"], preds["pred_label"], cfg["evaluation"]["bootstrap_resamples"],
                            cfg["evaluation"]["ci_level"], seed)
        if alphas is not None:
            m["mean_gate_weights"] = alphas.mean(0).tolist()
        save_json(m, run_dir / f"metrics_{name}.json")
        classification_table(m).to_csv(run_dir / f"classification_{name}.csv", index=False)
        plot_confusion(m["confusion_matrix"], run_dir / f"confusion_{name}.png",
                       title=f"{model_name.upper()} - {name} (n = {m['n']})")
        plot_confusion(m["confusion_matrix"], run_dir / f"confusion_{name}_normalised.png", normalise=True)
        results[name] = m
        log(f"[{name}] n={m['n']} accuracy={m['accuracy']:.4f} (95% CI {m['accuracy_ci_wilson'][0]:.4f}-"
            f"{m['accuracy_ci_wilson'][1]:.4f}) macro-F1={m['macro_f1']:.4f}")

    meta = {"model": model_name, "balance": balance, "seed": seed, "pretrained": pretrained or cfg["bert"]["pretrained"],
            "max_len": extra["max_len"], "hyperparameters": hp, "minutes": round((time.time() - t_start) / 60, 1),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "partition_sizes": {k: int(len(v["y"])) for k, v in enc.parts.items()}}
    save_json(meta, run_dir / "run_info.json")
    save_json(cfg, run_dir / "config_used.json")
    save_json(environment_info(), run_dir / "environment.json")
    if save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
        if model_name in {"cnn", "bilstm"}:
            save_json({"word_index": extra["tokenizer"].word_index, "num_words": extra["tokenizer"].num_words,
                       "oov_token": extra["tokenizer"].oov_token, "max_len": extra["max_len"]}, run_dir / "vocab.json")
    if balance == "smote":   # the memory-mapped features are large; keep only the provenance file
        for f in ["train_seq.npy", "smote_seq.npy"]:
            p = run_dir / "features" / f
            if p.exists():
                p.unlink()
    logf.close()
    return results, model, extra, data


def load_run(run_dir: Path, pretrained: str | None = None, device=None):
    """Rebuild a trained model, its data and its encoder from a run folder."""
    from .utils import load_json

    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config_used.json")
    info = load_json(run_dir / "run_info.json")
    device = device or get_device()
    data = build_data(cfg, verbose=False)
    pre = pretrained or info["pretrained"]
    enc, extra = encode_all(info["model"], cfg, data, pre, log=lambda *_: None)
    psy_dim = enc["train"]["psycho"].shape[1] if enc["train"]["psycho"] is not None else 2
    model = build_model(info["model"], cfg, psycho_dim=psy_dim, vocab_size=extra.get("vocab_size"), pretrained=pre)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), cfg, info, data, enc, extra
