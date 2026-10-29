#!/usr/bin/env python
"""Compare trained runs on the same evaluation set: metric table, pairwise McNemar tests and
figures in the style of the manuscript (macro metrics, accuracy and confusion matrices).

  python scripts/compare_runs.py --runs runs/cnn_seed2018 runs/bilstm_seed2018 runs/bert_seed2018 \
         runs/cb_seed2018 runs/bb_seed2018 --labels CNN BiLSTM BERT CB BB --eval-set validation
"""
import argparse
import itertools
from pathlib import Path

from _common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--eval-set", default="validation")
    p.add_argument("--out", default=str(ROOT / "reports" / "comparison"))
    args = p.parse_args()

    import pandas as pd

    from hyphix.evaluation import mcnemar_test
    from hyphix.plots import plot_accuracy_bars, plot_grouped_metrics
    from hyphix.utils import load_json, save_json

    labels = args.labels or [Path(r).name for r in args.runs]
    out = Path(args.out) / args.eval_set
    out.mkdir(parents=True, exist_ok=True)
    rows, preds, grouped = [], {}, {}
    for lab, r in zip(labels, args.runs):
        m = load_json(Path(r) / f"metrics_{args.eval_set}.json")
        preds[lab] = pd.read_csv(Path(r) / f"predictions_{args.eval_set}.csv")
        row = {"model": lab, "n": m["n"], "accuracy": m["accuracy"], "acc_ci_low": m["accuracy_ci_wilson"][0],
               "acc_ci_high": m["accuracy_ci_wilson"][1], "macro_precision": m["macro_precision"],
               "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"]}
        if "macro_f1_ci_bootstrap" in m:
            row.update({"f1_ci_low": m["macro_f1_ci_bootstrap"][0], "f1_ci_high": m["macro_f1_ci_bootstrap"][1]})
        for c in m["per_class"]:
            for k in ["precision", "recall", "f1"]:
                row[f"{k}_{c['class_index']}"] = c[k]
        rows.append(row)
        grouped[lab] = {"Precision": m["macro_precision"], "Recall": m["macro_recall"], "F1-score": m["macro_f1"]}
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics_table.csv", index=False)
    print(table[["model", "n", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1"]].to_string(index=False))

    tests = []
    for a, b in itertools.combinations(labels, 2):
        pa, pb = preds[a], preds[b]
        if not (pa["tweet_id"].values == pb["tweet_id"].values).all():
            raise SystemExit(f"{a} and {b} were not evaluated on the same tweets")
        t = mcnemar_test(pa["true_label"], pa["pred_label"], pb["pred_label"])
        tests.append({"model_a": a, "model_b": b, **t})
    pd.DataFrame(tests).to_csv(out / "mcnemar_tests.csv", index=False)
    save_json({"eval_set": args.eval_set, "runs": dict(zip(labels, args.runs))}, out / "runs.json")
    lo = min(r["accuracy"] for r in rows)
    plot_accuracy_bars({r["model"]: r["accuracy"] for r in rows}, out / "accuracy.png", ylim=(max(0, lo - 0.05), 1.0))
    plot_grouped_metrics(grouped, out / "macro_metrics.png")
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
