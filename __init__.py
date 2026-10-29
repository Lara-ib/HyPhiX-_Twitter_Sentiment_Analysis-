"""HyPhiX: explainable hybrid BERT models for four-class Twitter sentiment analysis."""

__version__ = "1.0.0"

LABEL_NAMES = ["Irrelevant", "Negative", "Neutral", "Positive"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
