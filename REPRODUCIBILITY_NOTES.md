# Reproducibility notes

This document records how the code relates to the original notebooks and to the manuscript, and which
reported values the original code could or could not produce. Please read it before updating the paper.

## 1. Results printed in the original notebooks

| Model | Notebook | Evaluated on | Accuracy |
|---|---|---|---|
| BERT–CNN (CB) | `BERT_CNN.ipynb` | internal test split (7,199) | 0.9018 |
| BERT–CNN (CB) | `BERT_MODELS_testing_code.ipynb` | evaluation file (999) | 0.9550 |
| BERT–BiLSTM (BB) | `BERT_BILSTM.ipynb` | internal test split (7,199) | 0.8997 |
| BERT–BiLSTM (BB) | `BERT_MODELS_testing_code.ipynb` | evaluation file (999) | 0.9419 |
| BERT | `BERT_final.ipynb` | 20 % random split of the training file | 0.9028 |
| CNN | `CNN_LSTM_BiLSTM testing code.ipynb` | evaluation file (999) | 0.8819 |
| BiLSTM | `CNN_LSTM_BiLSTM testing code.ipynb` | evaluation file (999) | 0.9159 |

The class-wise reports for CB and BB on the 999 tweets equal the “pre-balancing” columns of Tables 7 and 8
of the manuscript.

## 2. Components that the original code did not contain

The original notebooks contain **no SMOTE, no SHAP and no confusion-matrix code**. Consequently the
following values of the manuscript cannot be reproduced from the original code and must be regenerated with
this repository before publication:

* the “post-balancing” columns of Tables 7 and 8 (CB 0.968 / BB 0.954) and the accuracies derived from them;
* the confusion matrix of Figure 7 (967 of 999 correct) and Table 9, including the confidence intervals;
* the SHAP rankings of Figures 8 and 9.

This repository implements these steps (`smote.py`, `evaluation.py`, `plots.py`, `explain.py`). Report the
numbers that these scripts produce.

## 3. Settings: code versus manuscript (Table 3)

| Setting | Original code (reproduced here) | Current manuscript text |
|---|---|---|
| Max. sequence length (CB, BB) | **36** tokens | 128 tokens |
| Max. sequence length (BERT baseline) | 128 tokens | — |
| Dropout | BB: 0.1 on BERT outputs; CB: 0.1 defined but never applied | 0.3 before the classifier |
| Optimiser | AdamW (transformers), ε = 10⁻⁶, no weight decay | Adam |
| L2 regularisation | none in CB/BB (0.01 kernel L2 only in the Keras LSTM baselines) | λ = 10⁻⁴ |
| Split / seed | stratified 80/10/10, `random_state = 2018` | seed 42 |
| Encoder | fine-tuned (the notebooks freeze a *separate* `bert` object, so the model’s own BERT is trained) | fine-tuned |
| BERT baseline | TFBertForSequenceClassification, lr 3×10⁻⁵, 10 epochs, own 80/20 split | — |
| Model selection | the notebook stores `model.state_dict()` by reference, so the *last* epoch is saved | best validation accuracy |

Defaults in `configs/default.yaml` follow the original code. Table 3 of the manuscript should be updated to
the values actually used in the runs that produce the reported numbers.

## 4. Deliberate corrections in this repository

* The best epoch is stored as a real copy (`select_best_by: last` reproduces the notebook behaviour).
* The testing notebook defines BERT_CNN without BatchNorm2d although the trained model contains it; here one
  definition is used for training and testing.
* `pd.read_csv(...).astype(str)` behaviour of pandas < 3 (missing text → "nan") is reproduced explicitly.
* All models share one protocol (same split and evaluation sets), so comparisons are paired; the original BERT
  and CNN/BiLSTM notebooks used their own splits.
* Optional, off by default: keeping negations (`--keep-negations`), masked-mean pooling in BB (`bb.pooling: mean`),
  the leakage-aware `--split group`.

## 5. Data overlap

* 715 of the 999 evaluation tweets (71.6 %) appear verbatim (after lower-casing and removal of punctuation) in
  the training file; all 999 tweet ids appear there with the same label.
* With the stratified split, 7,189 of the 7,199 internal test rows share their tweet id with training rows.
* `--split group` removes the id overlap between partitions; `validation_dedup` (284 tweets) removes the
  verbatim overlap with the evaluation file.

These facts should be reported in the limitations of the manuscript, ideally together with the scores on the
leakage-aware sets.
