"""SHAP explanations for the BERT-based models.

Settings (all recorded in ``shap_settings.json`` of every run):
  explainer       shap.Explainer(f, masker) -> PartitionExplainer (hierarchical Owen values)
  masker          shap.maskers.Text(tokenizer) - WordPiece tokens of the *preprocessed* tweet are
                  hidden by replacing them with [MASK]; the [CLS]/[SEP] tokens are never masked
  model output    class probabilities (softmax scale) for all four classes
  explained class the predicted class of each tweet (global ranking); signed values are also
                  stored for every class
  sample          ``n_samples`` tweets of the chosen evaluation set (stratified random sample)
  aggregation     WordPiece pieces are merged into words; a word's global importance is the mean
                  absolute SHAP value over all its occurrences (words seen fewer than ``min_count``
                  times are excluded)
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import LABEL_NAMES


def make_predict_fn(model, tokenizer, max_len, device, psycho_fn=None, batch_size=64):
    """Return f(list[str]) -> probabilities (N, C) for SHAP."""

    @torch.no_grad()
    def f(texts):
        texts = [str(t) for t in texts]
        out = []
        model.eval()
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            enc = tokenizer(chunk, max_length=max_len, padding="max_length", truncation=True,
                            return_tensors="pt", return_token_type_ids=False)
            psy = None
            if psycho_fn is not None:
                psy = torch.as_tensor(psycho_fn([t.replace(tokenizer.mask_token, " ") for t in chunk])).to(device)
            logp = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), psy)
            out.append(logp.exp().cpu().numpy())
        return np.concatenate(out)

    return f


def _merge_wordpieces(tokens, values):
    """Merge WordPiece tokens ('play', '##ing') into words, summing their SHAP values."""
    words, vals = [], []
    for t, v in zip(tokens, values):
        s = str(t).strip()
        if not s:
            continue
        if s.startswith("##") and words:
            words[-1] += s[2:]
            vals[-1] = vals[-1] + v
        else:
            words.append(s.lower())
            vals.append(np.array(v, dtype=float))
    return words, vals


def explain(model, tokenizer, df: pd.DataFrame, out_dir: Path, max_len: int, device, n_samples=500,
            max_evals=300, batch_size=64, top_k=20, min_count=5, n_instance_plots=12, seed=2018,
            psycho_fn=None, model_name="model"):
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_samples < len(df):
        parts = [g.sample(n=max(1, round(n_samples * len(g) / len(df))), random_state=seed) for _, g in df.groupby("label")]
        sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        sample = df.reset_index(drop=True)
    texts = sample["text"].astype(str).tolist()

    f = make_predict_fn(model, tokenizer, max_len, device, psycho_fn, batch_size)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f, masker, output_names=LABEL_NAMES)
    sv = explainer(texts, max_evals=max_evals, batch_size=batch_size)

    probs = f(texts)
    pred = probs.argmax(1)
    with open(out_dir / "shap_values.pkl", "wb") as fh:
        pickle.dump({"values": sv.values, "base_values": sv.base_values, "data": sv.data, "texts": texts,
                     "tweet_id": sample["tweet_id"].tolist(), "true": sample["label"].tolist(),
                     "pred": pred.tolist()}, fh)

    # ---- word-level records
    rows, absvals, signed = [], defaultdict(list), {c: defaultdict(list) for c in range(len(LABEL_NAMES))}
    for i in range(len(texts)):
        words, vals = _merge_wordpieces(sv.data[i], sv.values[i])
        for w, v in zip(words, vals):
            if w in {"[cls]", "[sep]", "[pad]", ""}:
                continue
            absvals[w].append(abs(v[pred[i]]))
            for c in range(len(LABEL_NAMES)):
                signed[c][w].append(v[c])
            rows.append({"sample": i, "tweet_id": sample["tweet_id"][i], "word": w, "pred_class": LABEL_NAMES[pred[i]],
                         **{f"shap_{LABEL_NAMES[c]}": float(v[c]) for c in range(len(LABEL_NAMES))}})
    pd.DataFrame(rows).to_csv(out_dir / "shap_word_level.csv", index=False)

    glob = pd.DataFrame([{"word": w, "count": len(v), "mean_abs_shap": float(np.mean(v))}
                         for w, v in absvals.items() if len(v) >= min_count],
                        columns=["word", "count", "mean_abs_shap"])
    glob = glob.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    glob.to_csv(out_dir / "shap_global_ranking.csv", index=False)

    sig = []
    for c in range(len(LABEL_NAMES)):
        for w, v in signed[c].items():
            if len(v) >= min_count:
                sig.append({"class": LABEL_NAMES[c], "word": w, "count": len(v), "mean_shap": float(np.mean(v))})
    sig = pd.DataFrame(sig, columns=["class", "word", "count", "mean_shap"])
    sig.to_csv(out_dir / "shap_signed_by_class.csv", index=False)

    # ---- global figure (unsigned, as Figures 8-9 of the manuscript)
    top = glob.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.2))
    ax.barh(top["word"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| for the predicted class (probability scale)")
    ax.set_title(f"{model_name}: top {len(top)} words (n = {len(texts)} tweets)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_global_top_words.png", dpi=400)
    plt.close(fig)

    # ---- signed figure: most positive and most negative words for each class
    if len(sig):
        fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(4.2 * len(LABEL_NAMES), 5))
        for ax, name in zip(axes, LABEL_NAMES):
            d = sig[sig["class"] == name].sort_values("mean_shap")
            d = pd.concat([d.head(8), d.tail(8)]).drop_duplicates("word")
            ax.barh(d["word"], d["mean_shap"], color=["#C44E52" if v < 0 else "#55A868" for v in d["mean_shap"]])
            ax.axvline(0, color="black", lw=0.6)
            ax.set_title(name)
            ax.set_xlabel("mean signed SHAP")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_signed_by_class.png", dpi=300)
        plt.close(fig)

    # ---- instance-level explanations (signed, for the predicted class)
    inst = out_dir / "instances"
    inst.mkdir(exist_ok=True)
    first = list(range(min(n_instance_plots // 2, len(texts))))
    wrong = [i for i in range(len(texts)) if pred[i] != sample["label"][i] and i not in first]
    chosen = first + wrong[: n_instance_plots - len(first)]   # correct and misclassified examples
    html_parts = []
    for i in chosen:
        c = int(pred[i])
        html_parts.append(f"<h3>tweet_id {sample['tweet_id'][i]} | true: {LABEL_NAMES[sample['label'][i]]} | "
                          f"predicted: {LABEL_NAMES[c]} (p = {probs[i, c]:.3f})</h3>")
        html_parts.append(shap.plots.text(sv[i, :, c], display=False))
        plt.figure()
        shap.plots.bar(sv[i, :, c], max_display=12, show=False)
        plt.title(f"tweet_id {sample['tweet_id'][i]}: {LABEL_NAMES[c]}")
        plt.tight_layout()
        plt.savefig(inst / f"instance_{i:03d}_{LABEL_NAMES[c]}.png", dpi=200)
        plt.close("all")
    (out_dir / "instance_explanations.html").write_text(
        "<html><head><meta charset='utf-8'></head><body>" + "\n".join(html_parts) + "</body></html>", encoding="utf-8")

    settings = {"explainer": type(explainer).__name__, "masker": "shap.maskers.Text(tokenizer), mask token "
                f"{tokenizer.mask_token}", "output": "class probabilities (softmax)",
                "explained_class": "predicted class (global ranking); all classes (signed values)",
                "n_explained_tweets": len(texts), "max_evals_per_tweet": max_evals,
                "aggregation": "WordPiece pieces summed into words; mean |SHAP| over occurrences",
                "min_count": min_count, "text_input": "preprocessed tweet (the model input)",
                "instances_saved": len(chosen), "shap_version": shap.__version__}
    pd.Series(settings).to_json(out_dir / "shap_settings.json", indent=2)
    return glob, settings
