"""Data loading, preprocessing, label encoding, partitioning and leakage analysis.

The preprocessing reproduces the original notebooks step by step:
  1. read the CSV (optionally with pandas' default header handling, which
     drops the first tweet of each file, exactly as in the notebooks);
  2. drop the entity column and remove exact duplicate rows;
  3. convert the text to str (missing tweets become the string "nan");
  4. lower-case and remove NLTK English stopwords;
  5. tokenise with RegexpTokenizer(r"\\w+|\\d+") and re-join;
  6. lemmatise with WordNet using POS tags.
The text before step 4 is kept in the column ``raw_text`` so that the
psycholinguistic features can be computed before stopword removal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABEL_NAMES, LABEL_TO_ID

COLUMNS = ["tweet_id", "entity", "sentiment", "text"]
NEGATIONS = {"not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere",
             "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't",
             "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "haven't", "hasn't", "hadn't",
             "mustn't", "needn't", "mightn't", "shan't", "ain't",
             "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "wouldn", "couldn",
             "shouldn", "haven", "hasn", "hadn", "mustn", "needn", "mightn", "shan", "ain"}


# --------------------------------------------------------------------------- NLTK
def ensure_nltk() -> None:
    """Download the NLTK resources used by the pipeline (no-op if present)."""
    import nltk

    for res, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"),
                      ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                      ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                      ("opinion_lexicon", "corpora/opinion_lexicon")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


# --------------------------------------------------------------------------- loading
def read_raw(path: str | Path, legacy_header: bool = True) -> pd.DataFrame:
    """Read a Kaggle CSV file. ``legacy_header=True`` mimics ``pd.read_csv(path)``
    in the notebooks, where the first tweet is consumed as the header row."""
    if legacy_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None)
    df.columns = COLUMNS
    return df


def preprocess(df: pd.DataFrame, keep_negations: bool = False, drop_duplicates: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """Apply the notebook preprocessing and return a new frame with the columns
    tweet_id, entity, sentiment, label, raw_text and text."""
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer

    n0 = len(df)
    # The notebooks drop the entity ("place") column before removing duplicates,
    # so duplicates are defined on (tweet_id, sentiment, text).
    key = df[["tweet_id", "sentiment", "text"]]
    if drop_duplicates:
        keep = ~key.duplicated()
        df = df.loc[keep].copy()
    else:
        df = df.copy()
    n_dup = n0 - len(df)
    n_missing = int(df["text"].isna().sum())
    # str() reproduces pandas<3 astype(str): a missing tweet becomes the string "nan"
    df["raw_text"] = df["text"].map(lambda v: str(v))

    stop = set(stopwords.words("english"))
    if keep_negations:
        stop = stop - NEGATIONS
    tok = RegexpTokenizer(r"\w+|\d+")
    lemm = _Lemmatizer()

    def clean(x: str) -> str:
        x = " ".join(w.lower() for w in x.split() if w.lower() not in stop)
        x = " ".join(tok.tokenize(x))
        return lemm(x)

    df["text"] = df["raw_text"].map(clean)
    df["label"] = df["sentiment"].map(LABEL_TO_ID).astype(int)
    df = df.reset_index(drop=True)
    if verbose:
        print(f"[preprocess] {n0} records -> removed {n_dup} exact duplicates -> {len(df)} tweets "
              f"({n_missing} missing texts converted to the string 'nan')")
    return df[["tweet_id", "entity", "sentiment", "label", "raw_text", "text"]]


class _Lemmatizer:
    """POS-aware WordNet lemmatiser identical to ``lemmatize_sentence`` in the notebooks."""

    def __init__(self):
        import nltk
        from nltk.stem import WordNetLemmatizer

        self.nltk = nltk
        self.lem = WordNetLemmatizer()
        from nltk.corpus import wordnet

        self.wn = wordnet
        self.cache: dict = {}

    def _pos(self, tag: str):
        wn = self.wn
        if tag.startswith("J"):
            return wn.ADJ
        if tag.startswith("V"):
            return wn.VERB
        if tag.startswith("N"):
            return wn.NOUN
        if tag.startswith("R"):
            return wn.ADV
        return wn.NOUN

    def __call__(self, sentence: str) -> str:
        words = sentence.split()
        if not words:
            return ""
        tags = self.nltk.pos_tag(words)
        out = []
        for w, t in tags:
            k = (w, t[:1])
            if k not in self.cache:
                self.cache[k] = self.lem.lemmatize(w, self._pos(t))
            out.append(self.cache[k])
        return " ".join(out)


# --------------------------------------------------------------------------- dataset bundle
@dataclass
class DataBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame            # internal test partition of the training file
    eval_file: pd.DataFrame       # separate evaluation file (twitter_validation.csv)
    eval_dedup: pd.DataFrame      # evaluation tweets whose text does not occur in the training file

    def get(self, name: str) -> pd.DataFrame:
        return {"train": self.train, "val": self.val, "test": self.test,
                "validation": self.eval_file, "eval": self.eval_file,
                "validation_dedup": self.eval_dedup, "eval_dedup": self.eval_dedup}[name]


def split_indices(df: pd.DataFrame, strategy: str = "stratified", seed: int = 2018,
                  test_size: float = 0.2, val_fraction: float = 0.5):
    """Return (train_idx, val_idx, test_idx).

    ``stratified`` reproduces the two calls of ``train_test_split(..., random_state=2018,
    stratify=labels)`` in the notebooks (80 / 10 / 10). ``group`` keeps all rows that share
    a tweet_id (the paraphrased variants of one tweet) in the same partition.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    if strategy == "stratified":
        tr, tmp, _, ytmp = train_test_split(idx, y, random_state=seed, test_size=test_size, stratify=y)
        va, te, _, _ = train_test_split(tmp, ytmp, random_state=seed, test_size=val_fraction, stratify=ytmp)
        return tr, va, te
    if strategy == "group":
        groups = df["tweet_id"].to_numpy()
        g1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, tmp = next(g1.split(idx, y, groups))
        g2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        a, b = next(g2.split(tmp, y[tmp], groups[tmp]))
        return tr, tmp[a], tmp[b]
    raise ValueError(f"unknown split strategy: {strategy}")


def normalise_for_overlap(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def build_data(cfg: dict, verbose: bool = True) -> DataBundle:
    from .utils import resolve

    d = cfg["data"]
    train_raw = read_raw(resolve(d["train_file"]), d.get("legacy_header", True))
    eval_raw = read_raw(resolve(d["eval_file"]), d.get("legacy_header", True))
    full = preprocess(train_raw, keep_negations=d.get("keep_negations", False), verbose=verbose)
    ev = preprocess(eval_raw, keep_negations=d.get("keep_negations", False), verbose=verbose)
    s = cfg["split"]
    tr, va, te = split_indices(full, s["strategy"], s["seed"], s["test_size"], s["val_fraction"])
    train_texts = set(full["raw_text"].map(normalise_for_overlap))
    in_train = ev["raw_text"].map(normalise_for_overlap).isin(train_texts)
    ev_dedup = ev.loc[~in_train].reset_index(drop=True)
    if verbose:
        print(f"[split:{s['strategy']}] train={len(tr)} val={len(va)} test={len(te)} | "
              f"evaluation file={len(ev)} (text-deduplicated subset={len(ev_dedup)})")
    return DataBundle(full.iloc[tr].reset_index(drop=True), full.iloc[va].reset_index(drop=True),
                      full.iloc[te].reset_index(drop=True), ev, ev_dedup)


# --------------------------------------------------------------------------- reports
def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(range(len(LABEL_NAMES)), fill_value=0)
    return pd.DataFrame({"class_index": range(len(LABEL_NAMES)), "class": LABEL_NAMES,
                         "tweets": counts.values,
                         "share_percent": (100 * counts.values / counts.values.sum()).round(1)})


def leakage_report(cfg: dict) -> dict:
    """Quantify the overlap between the evaluation file and the training file."""
    from .utils import resolve

    d = cfg["data"]
    tr = read_raw(resolve(d["train_file"]), d.get("legacy_header", True))
    ev = read_raw(resolve(d["eval_file"]), d.get("legacy_header", True))
    tr_text = set(tr["text"].map(lambda v: normalise_for_overlap(str(v))))
    ev_norm = ev["text"].map(lambda v: normalise_for_overlap(str(v)))
    exact = ev_norm.isin(tr_text)
    id_in = ev["tweet_id"].isin(set(tr["tweet_id"]))
    same_label = ev.merge(tr[["tweet_id", "sentiment"]].drop_duplicates(), on="tweet_id",
                          suffixes=("", "_train"))
    agree = (same_label["sentiment"] == same_label["sentiment_train"]).groupby(same_label["tweet_id"]).any()
    return {
        "training_records": int(len(tr)),
        "evaluation_records": int(len(ev)),
        "evaluation_texts_found_verbatim_in_training": int(exact.sum()),
        "share_found_verbatim_percent": round(100 * float(exact.mean()), 1),
        "evaluation_tweet_ids_present_in_training": int(id_in.sum()),
        "evaluation_ids_with_same_label_in_training": int(agree.sum()),
    }
