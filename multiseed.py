#!/usr/bin/env python
"""Repeat a configuration over several random seeds and summarise mean, SD and 95% CI.

  python scripts/multiseed.py --model cb --balance smote --seeds 2018 42 7 123 2024
  python scripts/multiseed.py --model cb --seeds 2018 42 7 --vary-split   # also re-draw the split
Writes runs/multiseed/<name>_seed<s>/ for every seed and reports/multiseed/<name>.json.
With --compare-to <other summary JSON> a paired t-test and Wilcoxon test over seeds are added.
"""
from pathlib import Path

from _common import ROOT, base_parser, config_from_args, maybe_limit


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, choices=["cnn", "bilstm", "bert", "cb", "bb", "hyphix"])
    p.add_argument("--balance", choices=["none", "smote"], default="none")
    p.add_argument("--fusion", choices=["gate", "concat", "equal"], default=None)
    p.add_argument("--no-psycho", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[2018, 42, 7, 123, 2024])
    p.add_argument("--vary-split", action="store_true", help="use each seed for the data split as well")
    p.add_argument("--eval-sets", nargs="+", default=["test", "validation", "validation_dedup"])
    p.add_argument("--name", default=None)
    p.add_argument("--compare-to", default=None)
    args = p.parse_args()

    from hyphix.data import build_data
    from hyphix.evaluation import aggregate_seeds, paired_seed_test
    from hyphix.pipeline import train_run
    from hyphix.utils import load_json, save_json

    name = args.name or "_".join(filter(None, [args.model, None if args.balance == "none" else args.balance,
                                               args.fusion, "nopsy" if args.no_psycho else None,
                                               "varysplit" if args.vary_split else None]))
    per_seed = {k: [] for k in args.eval_sets}
    for s in args.seeds:
        args.seed = s
        if args.vary_split:
            args.split_seed = s
        cfg = config_from_args(args)
        if args.fusion:
            cfg["hyphix"]["fusion"] = args.fusion
        if args.no_psycho:
            cfg["hyphix"]["use_psycho"] = False
        data = maybe_limit(build_data(cfg), args.limit, s)
        res, *_ = train_run(args.model, cfg, ROOT / "runs" / "multiseed" / f"{name}_seed{s}", args.balance, args.pretrained,
                            args.device, args.eval_sets, data, save_model=False)
        for k in args.eval_sets:
            per_seed[k].append(res[k])
    summary = {"name": name, "seeds": args.seeds, "vary_split": args.vary_split,
               "by_eval_set": {k: aggregate_seeds(v) for k, v in per_seed.items()}}
    if args.compare_to:
        other = load_json(args.compare_to)
        summary["paired_tests_vs_" + other["name"]] = {
            k: {m: paired_seed_test(summary["by_eval_set"][k][m]["values"], other["by_eval_set"][k][m]["values"])
                for m in ["accuracy", "macro_f1"]} for k in args.eval_sets if k in other["by_eval_set"]}
    out = ROOT / "reports" / "multiseed"
    save_json(summary, out / f"{name}.json")
    for k, v in summary["by_eval_set"].items():
        print(f"{k}: accuracy {v['accuracy']['mean']:.4f} ± {v['accuracy']['sd']:.4f} | "
              f"macro-F1 {v['macro_f1']['mean']:.4f} ± {v['macro_f1']['sd']:.4f} (n = {v['n_runs']})")


if __name__ == "__main__":
    main()
