#!/usr/bin/env python
"""Dataset statistics, partition sizes and the train/evaluation overlap (leakage) analysis.

  python scripts/data_report.py                 # writes reports/data_report.json and CSV tables
"""
from _common import ROOT, base_parser, config_from_args


def main():
    p = base_parser(__doc__)
    p.add_argument("--out", default=str(ROOT / "reports"))
    args = p.parse_args()
    cfg = config_from_args(args)

    from pathlib import Path

    import pandas as pd

    from hyphix.data import build_data, class_distribution, leakage_report, split_indices
    from hyphix.utils import save_json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = build_data(cfg)
    full = pd.concat([data.train, data.val, data.test])
    rep = {"retained_tweets": len(full), "partitions": {k: len(data.get(k)) for k in
                                                        ["train", "val", "test", "validation", "validation_dedup"]},
           "split": cfg["split"], "leakage": leakage_report(cfg)}
    # leakage between the internal partitions themselves (same tweet_id in train and test)
    tr_ids = set(data.train.tweet_id)
    rep["internal_test_rows_whose_tweet_id_is_in_train"] = int(data.test.tweet_id.isin(tr_ids).sum())
    class_distribution(full).to_csv(out / "class_distribution_retained.csv", index=False)
    for k in ["test", "validation", "validation_dedup"]:
        class_distribution(data.get(k)).to_csv(out / f"class_distribution_{k}.csv", index=False)
    save_json(rep, out / "data_report.json")
    import json

    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
