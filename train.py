#!/usr/bin/env python
"""Train and evaluate one model.

Examples
  python scripts/train.py --model cb                         # BERT-CNN, no oversampling
  python scripts/train.py --model cb --balance smote         # BERT-CNN + sequence-aware SMOTE
  python scripts/train.py --model bb --balance smote
  python scripts/train.py --model bert                       # BERT baseline
  python scripts/train.py --model cnn | bilstm               # word-level baselines
  python scripts/train.py --model hyphix --fusion gate       # complete three-stream model
  python scripts/train.py --model hyphix --fusion concat     # ablation: concatenation
  python scripts/train.py --model hyphix --fusion equal      # ablation: fixed equal weights
  python scripts/train.py --model hyphix --no-psycho         # ablation: without psycholinguistic stream
Outputs are written to runs/<run-name>/.
"""
from _common import ROOT, base_parser, config_from_args, maybe_limit


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, choices=["cnn", "bilstm", "bert", "cb", "bb", "hyphix"])
    p.add_argument("--balance", choices=["none", "smote"], default="none")
    p.add_argument("--fusion", choices=["gate", "concat", "equal"], default=None)
    p.add_argument("--no-psycho", action="store_true", help="HyPhiX without the psycholinguistic stream")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--eval-sets", nargs="+", default=["test", "validation", "validation_dedup"])
    p.add_argument("--no-save-model", action="store_true")
    args = p.parse_args()

    cfg = config_from_args(args)
    if args.fusion:
        cfg["hyphix"]["fusion"] = args.fusion
    if args.no_psycho:
        cfg["hyphix"]["use_psycho"] = False
    from hyphix.data import build_data
    from hyphix.pipeline import train_run

    data = maybe_limit(build_data(cfg), args.limit, cfg["train"]["seed"])
    name = args.run_name or "_".join(filter(None, [
        args.model, args.balance if args.balance != "none" else None,
        (cfg["hyphix"]["fusion"] + ("" if cfg["hyphix"]["use_psycho"] else "_nopsy")) if args.model == "hyphix" else None,
        cfg["split"]["strategy"] if cfg["split"]["strategy"] != "stratified" else None,
        f"seed{cfg['train']['seed']}"]))
    from pathlib import Path

    train_run(args.model, cfg, Path(args.runs_dir) / name, balance=args.balance, pretrained=args.pretrained,
              device=args.device, eval_sets=args.eval_sets, data=data, save_model=not args.no_save_model)


if __name__ == "__main__":
    main()
