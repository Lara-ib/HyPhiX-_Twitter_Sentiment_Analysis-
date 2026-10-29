"""Create a tiny randomly initialised BERT (and WordPiece vocabulary) for offline smoke tests.
It is NOT a substitute for bert-base-uncased; it only checks that the pipeline runs."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "tiny_bert"


def main(hidden=32, layers=2, heads=2, vocab_words=3000):
    from transformers import BertConfig, BertModel, BertTokenizer

    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "twitter_training.csv", header=None)
    words = Counter(w for t in df[3].map(lambda v: str(v).lower()) for w in t.split() if w.isalpha())
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [w for w, _ in words.most_common(vocab_words)]
    OUT.mkdir(exist_ok=True)
    (OUT / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizer(str(OUT / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(OUT)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                     intermediate_size=2 * hidden, max_position_embeddings=160)
    BertModel(cfg).save_pretrained(OUT)
    print("tiny BERT written to", OUT)


if __name__ == "__main__":
    main()
