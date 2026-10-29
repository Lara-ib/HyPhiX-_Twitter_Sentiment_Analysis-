"""Model definitions.

* ``BertCNN``     - BERT-CNN (CB), identical to BERT_CNN.ipynb.
* ``BertBiLSTM``  - BERT-BiLSTM (BB), identical to BERT_BILSTM.ipynb.
* ``HyPhiX``      - complete three-stream model (CNN branch, BiLSTM branch and a
                    psycholinguistic stream) with gated, concatenation or equal-weight fusion.
* ``BertClassifier`` - BERT baseline (BertForSequenceClassification, as in BERT_final.ipynb).
* ``CNNBaseline`` / ``BiLSTMBaseline`` - PyTorch ports of the Keras baselines.

Every BERT-based model exposes
    encode(input_ids, mask)            -> token embeddings H (B, L, d)
    head(H, mask, psycho=None)         -> log-probabilities (B, C)
    forward(input_ids, mask, psycho)   -> head(encode(...))
so that the head can be trained on (SMOTE-balanced) embeddings with a frozen encoder.
All heads return log-softmax outputs; training uses NLLLoss, which is equivalent to the
notebooks' CrossEntropyLoss applied to LogSoftmax outputs.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def load_bert(pretrained: str):
    from transformers import BertModel

    return BertModel.from_pretrained(pretrained)


class _BertBase(nn.Module):
    uses_psycho = False

    def __init__(self, pretrained: str, freeze_encoder: bool = False):
        super().__init__()
        self.bert = load_bert(pretrained)
        self.hidden = self.bert.config.hidden_size
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = False

    def encode(self, input_ids, mask):
        return self.bert(input_ids=input_ids, attention_mask=mask).last_hidden_state

    def forward(self, input_ids, mask, psycho=None):
        return self.head(self.encode(input_ids, mask), mask, psycho)


class BertCNN(_BertBase):
    """CB: Conv2d(1, 13, (3, d), padding=(1, 0)) -> BatchNorm2d -> ReLU ->
    MaxPool2d((3, 1), stride 1) -> Flatten -> Linear(13 * (L - 2), C) -> LogSoftmax."""

    def __init__(self, pretrained: str, num_classes: int = 4, max_len: int = 36, conv_channels: int = 13,
                 kernel_height: int = 3, pool_height: int = 3, head_dropout: float = 0.0,
                 freeze_encoder: bool = False):
        super().__init__(pretrained, freeze_encoder)
        self.conv = nn.Conv2d(1, conv_channels, kernel_size=(kernel_height, self.hidden),
                              padding=(kernel_height // 2, 0))
        self.bn = nn.BatchNorm2d(conv_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(pool_height, 1), stride=1)
        self.dropout = nn.Dropout(head_dropout)
        self.flat = nn.Flatten()
        self.out_dim = conv_channels * (max_len - pool_height + 1)   # 13 * 34 = 442 for L = 36
        self.fc = nn.Linear(self.out_dim, num_classes)

    def branch(self, H):
        x = self.conv(H.unsqueeze(1))
        x = self.relu(self.bn(x))
        x = self.pool(x)
        return self.flat(x)

    def head(self, H, mask=None, psycho=None):
        return F.log_softmax(self.fc(self.dropout(self.branch(H))), dim=1)


class BertBiLSTM(_BertBase):
    """BB: Dropout(0.1) on BERT embeddings -> 2-layer BiLSTM(128) -> Linear(256, C) -> LogSoftmax.
    ``pooling='last'`` uses lstm_out[:, -1, :] exactly as the notebook (with right padding this is
    the state at the last padded position); ``pooling='mean'`` uses a masked mean instead."""

    def __init__(self, pretrained: str, num_classes: int = 4, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, pooling: str = "last", freeze_encoder: bool = False):
        super().__init__(pretrained, freeze_encoder)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.pooling = pooling
        self.out_dim = 2 * hidden_size
        self.fc = nn.Linear(self.out_dim, num_classes)

    def branch(self, H, mask):
        out, _ = self.lstm(self.dropout(H))
        if self.pooling == "last":
            return out[:, -1, :]
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def head(self, H, mask=None, psycho=None):
        return F.log_softmax(self.fc(self.branch(H, mask)), dim=1)


class HyPhiX(_BertBase):
    """Complete HyPhiX design (Section 3.4 of the manuscript).

    z_cnn = W1 v_cnn,  z_bilstm = W2 v_bilstm,  z_psy = W3 p           (projection, Eq. 22)
    alpha = softmax(Wg [z_cnn; z_bilstm; z_psy] + bg)                  (gate, Eq. 23)
    z     = alpha_1 z_cnn + alpha_2 z_bilstm + alpha_3 z_psy            (fusion, Eq. 24)
    fusion = 'concat' replaces the gate by concatenation; 'equal' uses fixed weights 1/S.
    use_psycho=False removes the psycholinguistic stream (two-stream ablation).
    """

    uses_psycho = True

    def __init__(self, pretrained: str, num_classes: int = 4, max_len: int = 36, psycho_dim: int = 2,
                 fusion: str = "gate", use_psycho: bool = True, fusion_dim: int = 256,
                 conv_channels: int = 13, lstm_hidden: int = 128, lstm_layers: int = 2,
                 lstm_dropout: float = 0.1, pooling: str = "mean", freeze_encoder: bool = False):
        super().__init__(pretrained, freeze_encoder)
        self.fusion, self.use_psycho = fusion, use_psycho
        # CNN branch (same layers as CB)
        self.conv = nn.Conv2d(1, conv_channels, kernel_size=(3, self.hidden), padding=(1, 0))
        self.bn = nn.BatchNorm2d(conv_channels)
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1)
        cnn_dim = conv_channels * (max_len - 2)
        # BiLSTM branch (same layers as BB)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.lstm = nn.LSTM(self.hidden, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True)
        self.pooling = pooling
        # projections to the common fusion space
        self.proj_cnn = nn.Linear(cnn_dim, fusion_dim)
        self.proj_lstm = nn.Linear(2 * lstm_hidden, fusion_dim)
        self.n_streams = 3 if use_psycho else 2
        if use_psycho:
            self.proj_psy = nn.Sequential(nn.Linear(psycho_dim, fusion_dim), nn.ReLU())
        if fusion == "gate":
            self.gate = nn.Linear(self.n_streams * fusion_dim, self.n_streams)
        out_in = self.n_streams * fusion_dim if fusion == "concat" else fusion_dim
        self.fc = nn.Linear(out_in, num_classes)
        self.last_alpha = None

    def streams(self, H, mask, psycho):
        v_cnn = self.pool(F.relu(self.bn(self.conv(H.unsqueeze(1))))).flatten(1)
        out, _ = self.lstm(self.lstm_dropout(H))
        if self.pooling == "last":
            v_lstm = out[:, -1, :]
        else:
            m = mask.unsqueeze(-1).to(out.dtype)
            v_lstm = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        z = [self.proj_cnn(v_cnn), self.proj_lstm(v_lstm)]
        if self.use_psycho:
            if psycho is None:
                raise ValueError("HyPhiX with use_psycho=True needs the psycholinguistic features")
            z.append(self.proj_psy(psycho.to(H.dtype)))
        return z

    def head(self, H, mask=None, psycho=None):
        z = self.streams(H, mask, psycho)
        if self.fusion == "concat":
            fused = torch.cat(z, dim=1)
            self.last_alpha = None
        else:
            stack = torch.stack(z, dim=1)                       # (B, S, d)
            if self.fusion == "gate":
                alpha = torch.softmax(self.gate(torch.cat(z, dim=1)), dim=1)
            else:                                               # fixed equal weights
                alpha = torch.full((H.size(0), self.n_streams), 1.0 / self.n_streams, device=H.device)
            self.last_alpha = alpha.detach()
            fused = (alpha.unsqueeze(-1) * stack).sum(1)
        return F.log_softmax(self.fc(fused), dim=1)


class BertClassifier(nn.Module):
    """BERT baseline: BertForSequenceClassification (pooled [CLS] + linear layer)."""

    uses_psycho = False

    def __init__(self, pretrained: str, num_classes: int = 4):
        super().__init__()
        from transformers import BertForSequenceClassification

        self.model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=num_classes)

    def forward(self, input_ids, mask, psycho=None):
        return F.log_softmax(self.model(input_ids=input_ids, attention_mask=mask).logits, dim=1)


# --------------------------------------------------------------------------- word-level baselines
class CNNBaseline(nn.Module):
    """Port of the Keras CNN: Embedding(V+1, 20) -> 3 x [Conv1D(k=3)+ReLU] with MaxPool1D(3)
    after the first two -> GlobalMaxPool -> Dense(256, ReLU) -> Dropout(0.075) -> Dense(C)."""

    uses_psycho = False

    def __init__(self, vocab_size: int, num_classes: int = 4, embedding_dim: int = 20, dropout: float = 0.075):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.c1 = nn.Conv1d(embedding_dim, 32, 3)
        self.c2 = nn.Conv1d(32, 64, 3)
        self.c3 = nn.Conv1d(64, 128, 3)
        self.fc1 = nn.Linear(128, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, ids, mask=None, psycho=None):
        x = self.emb(ids).transpose(1, 2)
        x = F.max_pool1d(F.relu(self.c1(x)), 3)
        x = F.max_pool1d(F.relu(self.c2(x)), 3)
        x = F.relu(self.c3(x)).amax(dim=2)
        x = self.drop(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


class BiLSTMBaseline(nn.Module):
    """Port of the Keras BiLSTM: Embedding(V, 128) -> BiLSTM(128, input dropout 0.5) -> BatchNorm ->
    BiLSTM(128, input dropout 0.5) -> BatchNorm -> Dense(C). The Keras kernel L2 penalty (0.01)
    is added to the loss through ``l2_penalty()``."""

    uses_psycho = False

    def __init__(self, vocab_size: int, num_classes: int = 4, embedding_dim: int = 128,
                 hidden_size: int = 128, dropout: float = 0.5, l2: float = 0.01):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.l1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(2 * hidden_size)
        self.l2_layer = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.l2 = l2

    def forward(self, ids, mask=None, psycho=None):
        x, _ = self.l1(self.drop(self.emb(ids)))
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.l2_layer(self.drop(x))
        x = torch.cat([h[-2], h[-1]], dim=1)          # final forward and backward states
        return F.log_softmax(self.fc(self.bn2(x)), dim=1)

    def l2_penalty(self):
        w = [p for n, p in self.named_parameters() if "weight_ih" in n]
        return self.l2 * sum((p ** 2).sum() for p in w)


def build_model(name: str, cfg: dict, psycho_dim: int = 2, vocab_size: int | None = None,
                pretrained: str | None = None):
    """Create a model from the configuration. ``name`` in {cb, bb, hyphix, bert, cnn, bilstm}."""
    b = cfg["bert"]
    pre = pretrained or b["pretrained"]
    if name == "cb":
        c = cfg["cb"]
        return BertCNN(pre, max_len=b["max_len"], conv_channels=c["conv_channels"],
                       kernel_height=c["kernel_height"], pool_height=c["pool_height"],
                       head_dropout=c["head_dropout"], freeze_encoder=b["freeze_encoder"])
    if name == "bb":
        c = cfg["bb"]
        return BertBiLSTM(pre, hidden_size=c["hidden_size"], num_layers=c["num_layers"],
                          dropout=c["dropout"], pooling=c["pooling"], freeze_encoder=b["freeze_encoder"])
    if name == "hyphix":
        h = cfg["hyphix"]
        return HyPhiX(pre, max_len=b["max_len"], psycho_dim=psycho_dim, fusion=h["fusion"],
                      use_psycho=h["use_psycho"], fusion_dim=h["fusion_dim"],
                      freeze_encoder=b["freeze_encoder"])
    if name == "bert":
        return BertClassifier(pre)
    if name == "cnn":
        c = cfg["baselines"]["cnn"]
        return CNNBaseline(vocab_size, embedding_dim=c["embedding_dim"], dropout=c["dropout"])
    if name == "bilstm":
        c = cfg["baselines"]["bilstm"]
        return BiLSTMBaseline(vocab_size, embedding_dim=c["embedding_dim"], hidden_size=c["hidden_size"],
                              dropout=c["dropout"], l2=c["l2"])
    raise ValueError(f"unknown model {name}")
