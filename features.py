"""Text encoders: BERT tokenisation, word vocabularies for the baselines and
psycholinguistic features.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


# --------------------------------------------------------------------------- BERT
def get_tokenizer(pretrained: str):
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(pretrained)


def bert_encode(texts, tokenizer, max_len: int):
    """Identical to ``data_process`` in the notebooks: pad/truncate to ``max_len`` tokens
    with [CLS] ... [SEP] and return int64 input ids and attention masks."""
    enc = tokenizer(list(map(str, texts)), max_length=max_len, padding="max_length",
                    truncation=True, return_token_type_ids=False)
    return (np.asarray(enc["input_ids"], dtype=np.int64),
            np.asarray(enc["attention_mask"], dtype=np.int64))


# --------------------------------------------------------------------------- word vocabularies
class KerasLikeTokenizer:
    """Minimal port of ``tf.keras.preprocessing.text.Tokenizer`` + ``pad_sequences``:
    words are ranked by frequency (index 1 = most frequent), only the ``num_words - 1``
    most frequent words are kept, padding and truncation are applied at the start ("pre")."""

    def __init__(self, num_words: int | None = None, oov_token: str | None = None):
        self.num_words, self.oov_token = num_words, oov_token
        self.word_index: dict = {}

    @staticmethod
    def _split(text: str):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        t = str(text).lower().translate(str.maketrans({c: " " for c in filters}))
        return [w for w in t.split(" ") if w]

    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(self._split(t))
        ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        if self.oov_token is not None:
            ranked = [self.oov_token] + ranked
        self.word_index = {w: i + 1 for i, w in enumerate(ranked)}
        return self

    def texts_to_sequences(self, texts):
        out = []
        oov = self.word_index.get(self.oov_token) if self.oov_token else None
        for t in texts:
            seq = []
            for w in self._split(t):
                i = self.word_index.get(w)
                if i is not None and (self.num_words is None or i < self.num_words):
                    seq.append(i)
                elif oov is not None:
                    seq.append(oov)
            out.append(seq)
        return out

    @staticmethod
    def pad(seqs, maxlen: int):
        arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
        for i, s in enumerate(seqs):
            s = s[-maxlen:]
            if s:
                arr[i, -len(s):] = s
        return arr

    @property
    def vocab_size(self) -> int:
        return len(self.word_index) + 1


# --------------------------------------------------------------------------- psycholinguistic features
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
                "i'm", "i've", "i'd", "i'll", "im", "ive", "id", "ill"}
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
                  "cannot", "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
                  "weren't", "won't", "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't",
                  "hadn't", "ain't", "dont", "doesnt", "didnt", "isnt", "cant", "wont"}


class PsycholinguisticExtractor:
    """Normalised lexicon proportions f_m = (1/T) * sum_t 1[w_t in lexicon_m] (Eq. 19).

    Categories
      self_reference   first-person singular and plural pronouns (list above)
      negative_affect  negative words of the Hu & Liu (2004) opinion lexicon (NLTK)
      positive_affect  positive words of the Hu & Liu (2004) opinion lexicon (NLTK)
      negation         negation words (list above)
    Features are computed on the raw text **before** stopword removal, so pronouns and
    negations are retained. A word that belongs to several lexicons counts in each of them.
    """

    def __init__(self, categories=("self_reference", "negative_affect")):
        from nltk.corpus import opinion_lexicon

        from .data import ensure_nltk

        ensure_nltk()
        self.categories = list(categories)
        self.lexicons = {
            "self_reference": FIRST_PERSON,
            "negative_affect": set(opinion_lexicon.negative()),
            "positive_affect": set(opinion_lexicon.positive()),
            "negation": NEGATION_WORDS,
        }
        unknown = set(self.categories) - set(self.lexicons)
        if unknown:
            raise ValueError(f"unknown psycholinguistic categories: {unknown}")

    @property
    def dim(self) -> int:
        return len(self.categories)

    def __call__(self, texts) -> np.ndarray:
        import re

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z']+", str(t).lower())
            if not words:
                continue
            T = len(words)
            for j, c in enumerate(self.categories):
                lex = self.lexicons[c]
                out[i, j] = sum(w in lex or w.replace("'", "") in lex for w in words) / T
        return out
