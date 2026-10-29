#!/usr/bin/env python
"""SHAP explanations of a trained BERT-based model (global, signed and instance-level).

  python scripts/explain_shap.py --run runs/cb_smote_seed2018 --eval-set validation --n-samples 500
Outputs: runs/<run>/shap_<eval-set>/ (rankings, figures, HTML instance explanations, raw values).
"""
import argparse
from pathlib import Path

from _common import ROOT  # noqa: F401  (adds src to the path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--eval-set", default="validation", choices=["test", "validation", "validation_dedup"])
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--max-evals", type=int, default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from hyphix.explain import explain
    from hyphix.pipeline import load_run

    model, cfg, info, data, enc, extra = load_run(Path(args.run), args.pretrained, args.device)
    if info["model"] not in {"cb", "bb", "hyphix", "bert"}:
        raise SystemExit("SHAP explanations are provided for the BERT-based models")
    s = cfg["shap"]
    psy = extra.get("psycho_extractor")
    device = next(model.parameters()).device
    glob, settings = explain(model, extra["tokenizer"], data.get(args.eval_set), Path(args.run) / f"shap_{args.eval_set}",
                             extra["max_len"], device, n_samples=args.n_samples or s["n_samples"],
                             max_evals=args.max_evals or s["max_evals"], batch_size=s["batch_size"], top_k=s["top_k"],
                             min_count=s["min_count"], n_instance_plots=s["n_instance_plots"], seed=info["seed"],
                             psycho_fn=psy, model_name=info["model"].upper())
    print(glob.head(20).to_string(index=False))
    print(settings)


if __name__ == "__main__":
    main()
