"""Shared command-line helpers (adds ``src`` to the import path)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML configuration file")
    p.add_argument("--pretrained", default=None, help="BERT checkpoint name or local folder (default from config)")
    p.add_argument("--seed", type=int, default=None, help="model seed (train.seed)")
    p.add_argument("--split", choices=["stratified", "group"], default=None, help="partitioning strategy")
    p.add_argument("--split-seed", type=int, default=None, help="seed of the 80/10/10 split")
    p.add_argument("--keep-negations", action="store_true", help="keep negation words during stopword removal")
    p.add_argument("--full-files", action="store_true",
                   help="read the first CSV row as data (the notebooks lost it as a header row)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None, help="BERT sequence length for CB/BB/HyPhiX")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="use only N training tweets (quick tests)")
    return p


def config_from_args(args):
    from hyphix.utils import load_config

    ov = {"train.seed": args.seed, "split.strategy": args.split, "split.seed": args.split_seed,
          "train.epochs": args.epochs, "bert.max_len": args.max_len}
    if args.keep_negations:
        ov["data.keep_negations"] = True
    if args.full_files:
        ov["data.legacy_header"] = False
    return load_config(args.config, ov)


def maybe_limit(data, n, seed=0):
    if not n:
        return data
    data.train = data.train.sample(n=min(n, len(data.train)), random_state=seed).reset_index(drop=True)
    k = max(64, n // 8)
    data.val = data.val.sample(n=min(k, len(data.val)), random_state=seed).reset_index(drop=True)
    data.test = data.test.sample(n=min(k, len(data.test)), random_state=seed).reset_index(drop=True)
    return data
