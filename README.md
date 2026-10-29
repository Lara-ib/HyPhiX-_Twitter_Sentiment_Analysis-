# HyPhiX – Hybrid BERT Models for Four-Class Twitter Sentiment Analysis

Code accompanying the manuscript **“HyPhiX: An Explainable Hybrid Fusion Framework for Multi-Class
Sentiment Analysis of Social Media Text”** (submitted to *Algorithms*, MDPI).

The repository trains and evaluates the BERT–CNN (CB) and BERT–BiLSTM (BB) models, the CNN, BiLSTM and BERT
baselines and the complete three-stream HyPhiX model on the Kaggle *Twitter Entity Sentiment Analysis*
dataset (positive, negative, neutral and irrelevant). Besides training and testing, it provides:

| Component | Where | What it does |
|---|---|---|
| Sequence-aware **SMOTE** | `src/hyphix/smote.py` | Oversamples the training partition in the BERT embedding space; neighbours are searched on the `[CLS]` vectors and the same interpolation is applied to the complete token-embedding sequence, the attention mask and the psycholinguistic vector, so CNN/BiLSTM heads receive valid synthetic inputs |
| **Confusion matrices** and metrics | `src/hyphix/evaluation.py`, `plots.py` | Case-level predictions, class-wise precision/recall/F1, raw and normalised confusion matrices |
| **Statistics** | `src/hyphix/evaluation.py` | Wilson intervals (accuracy, class recall), bootstrap interval of macro-F1, exact McNemar test, multi-seed mean ± SD with paired t-test and Wilcoxon test |
| **SHAP** | `src/hyphix/explain.py` | Partition explainer with a text masker: global word ranking, signed per-class attributions and instance-level explanations (HTML and PNG) |
| **Full HyPhiX model** | `src/hyphix/models.py` | CNN + BiLSTM + psycholinguistic streams with gated, concatenation or fixed-equal fusion, and the ablations requested by the reviewers |
| **Leakage analysis** | `scripts/data_report.py` | Overlap between the evaluation file and the training file; leakage-aware `group` split |

> **Attribution.** The preprocessing and the CB/BB/CNN/BiLSTM/BERT architectures reproduce the notebooks of
> the public repository *Twitter-Sentiment-Analysis* by A. R. Avula, P. K. Rohan and M. Shashidhara
> (Apache License 2.0). This repository re-implements them as a single PyTorch pipeline and adds the
> components listed above. See `NOTICE` and `docs/REPRODUCIBILITY_NOTES.md`.

---

## 1. Installation

**Google Colab (recommended, GPU runtime):** upload `HyPhiX-Twitter-Sentiment.zip` to *My Drive*, open
`notebooks/HyPhiX_Colab.ipynb` in Colab (File → Upload notebook), select a T4 GPU and run the cells. The notebook
stores all results directly in `My Drive/HyPhiX_runs` and `My Drive/HyPhiX_reports` and skips finished runs, so
after a disconnection you simply run the cells again.

**Local machine:**
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ['stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','opinion_lexicon']]"
```
A CUDA GPU with at least 12 GB of memory is recommended (the experiments of the manuscript used Google Colab
Pro with an NVIDIA Tesla T4 or V100). The first run downloads `bert-base-uncased` from Hugging Face.

## 2. Data

`data/twitter_training.csv` (74,682 rows) and `data/twitter_validation.csv` (1,000 rows) are the two files of
the Kaggle dataset <https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis>
(columns: tweet id, entity, sentiment, text; see the Kaggle page for the licence).

Exactly as in the original notebooks, both files are read with pandas’ default header handling, so the first
row of each file is used as the header: **74,681** training records and **999** evaluation tweets. After the
removal of 2,700 exact duplicates, **71,981** tweets remain (21,787 negative, 19,810 positive, 17,800 neutral,
12,584 irrelevant). Use `--full-files` to keep the first rows.

| Class index | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Label | Irrelevant | Negative | Neutral | Positive |

Partitions (`split.strategy: stratified`, seed 2018, as in the notebooks): train 57,584 · validation 7,198 ·
internal test 7,199 (supports 1,259 / 2,179 / 1,780 / 1,981). Every model is additionally evaluated on
the separate evaluation file (999 tweets; supports 171 / 266 / 285 / 277) and on its text-deduplicated subset.

### Important: overlap between the files

`python scripts/data_report.py` shows that **715 of the 999 evaluation tweets (71.6 %) occur verbatim in the
training file**, and every evaluation tweet id also occurs there with the same label. The dataset contains
several paraphrased variants of each tweet id, and with the stratified split 7,189 of the 7,199 internal-test
rows share their tweet id with training rows. Scores on these sets are therefore optimistic. Two leakage-aware
alternatives are built in:

* `validation_dedup` – the 284 evaluation tweets whose text does not occur in the training file (reported automatically);
* `--split group` – a split that keeps all variants of one tweet id in the same partition (0 shared ids).

## 3. Quick start

```bash
python scripts/data_report.py                           # dataset statistics and leakage report
python scripts/train.py --model cb                      # BERT-CNN, no oversampling
python scripts/train.py --model cb --balance smote      # BERT-CNN + SMOTE
python scripts/explain_shap.py --run runs/cb_smote_seed2018 --eval-set validation
```

Every run writes to `runs/<name>/`:

| File | Content |
|---|---|
| `metrics_<set>.json` | accuracy + Wilson 95 % CI, macro/weighted metrics, macro-F1 bootstrap CI, per-class metrics with recall CIs, confusion matrix (HyPhiX: mean gate weights) |
| `classification_<set>.csv` | class-wise table (Tables 6–9 of the manuscript) |
| `predictions_<set>.csv` | case-level outputs: tweet id, text, true label, prediction, class probabilities (and gate weights) |
| `confusion_<set>.png`, `confusion_<set>_normalised.png` | confusion matrices (Figure 7) |
| `history_stage_a.json/.png`, `history_stage_b_smote.json` | training curves |
| `smote_summary.json`, `features/smote_provenance.npz` | class counts before/after SMOTE, imbalanced-learn check, base/neighbour/λ of every synthetic sample |
| `split_identifiers.csv` | partition of every tweet (train / val / test / evaluation file) |
| `config_used.json`, `run_info.json`, `environment.json` | full configuration, parameter counts, hardware and library versions |
| `model.pt` | trained weights |

`<set>` is `test` (internal test partition), `validation` (the 999-tweet evaluation file) or `validation_dedup`.

## 4. Reproducing all experiments

```bash
bash scripts/run_all.sh            # baselines, CB/BB with and without SMOTE, HyPhiX ablations, SHAP, comparisons
```
or step by step:

```bash
# baselines and hybrids (Table 6, Figures 4-5)
for m in cnn bilstm bert cb bb; do python scripts/train.py --model $m; done
# oversampling (Tables 7-8)
python scripts/train.py --model cb --balance smote
python scripts/train.py --model bb --balance smote
# complete HyPhiX model and ablations (hypothesis H2)
python scripts/train.py --model hyphix --fusion gate              # three streams, learned gate
python scripts/train.py --model hyphix --fusion concat            # gate -> concatenation
python scripts/train.py --model hyphix --fusion equal             # gate -> fixed equal weights
python scripts/train.py --model hyphix --fusion gate --no-psycho  # without the psycholinguistic stream
# comparison table, McNemar tests and figures
python scripts/compare_runs.py --runs runs/cnn_seed2018 runs/bilstm_seed2018 runs/bert_seed2018 \
       runs/cb_seed2018 runs/cb_smote_seed2018 runs/bb_seed2018 runs/bb_smote_seed2018 \
       --labels CNN BiLSTM BERT CB CB+SMOTE BB BB+SMOTE --eval-set validation
# multi-seed experiments (mean ± SD, 95 % CI, paired tests)
python scripts/multiseed.py --model cb --balance smote --seeds 2018 42 7 123 2024
python scripts/multiseed.py --model bb --balance smote --seeds 2018 42 7 123 2024 \
       --compare-to reports/multiseed/cb_smote.json
# leakage-aware protocol
python scripts/train.py --model cb --balance smote --split group
```

Approximate GPU times (T4): CB or BB ≈ 20–30 min per run (5 epochs, 57,584 tweets, 36 tokens); SMOTE stage
+10–15 min; BERT baseline (128 tokens, 10 epochs) ≈ 1.5–2 h; CNN and BiLSTM a few minutes; SHAP (500 tweets)
≈ 10–20 min.

## 5. Methods implemented

**Preprocessing** (`src/hyphix/data.py`): exact-duplicate removal on (tweet id, sentiment, text), lower-casing,
NLTK English stopword removal (optionally keeping negations, `--keep-negations`), `RegexpTokenizer(r"\w+|\d+")`,
POS-aware WordNet lemmatisation. The raw text before stopword removal is kept for the psycholinguistic features.

**CB / BB** (`src/hyphix/models.py`): `bert-base-uncased` (fine-tuned), 36 tokens.
CB: Conv2d(1→13, kernel 3×768, padding 1×0) → BatchNorm2d → ReLU → MaxPool2d(3×1, stride 1) → Linear(442→4).
BB: Dropout 0.1 → 2-layer BiLSTM (128 per direction) → Linear(256→4) on the last time step.
Training: batch 32, AdamW (lr 2×10⁻⁵, ε 10⁻⁶, no weight decay), gradient clipping 1.0, 5 epochs, best
validation accuracy. All settings are in `configs/default.yaml`.

**SMOTE** (`src/hyphix/smote.py`): Stage A trains the model end to end; Stage B freezes the fine-tuned encoder,
extracts the token embeddings of the training partition, adds synthetic minority samples until every class
matches the majority class (k = 5 neighbours in the `[CLS]` space, λ ~ U(0, 1), same λ for the whole sequence)
and trains the head further on the balanced embeddings. Validation and evaluation tweets are never oversampled.
The per-class counts are checked against `imblearn.over_sampling.SMOTE`.

**HyPhiX** (`src/hyphix/models.py`): CNN branch (as CB) and BiLSTM branch (masked mean pooling) on the shared
BERT embeddings plus psycholinguistic features (self-reference and negative affect, Hu & Liu opinion lexicon,
computed before stopword removal); each stream is projected to 256 dimensions and fused by a softmax gate
α = softmax(W_g[z₁; z₂; z₃]) (`gate`), by concatenation (`concat`) or by fixed weights 1/3 (`equal`).

**SHAP** (`src/hyphix/explain.py`): `shap.Explainer(f, shap.maskers.Text(tokenizer))` (PartitionExplainer);
f returns the class probabilities of the preprocessed tweet; WordPiece pieces are merged into words; global
importance = mean |SHAP| for the predicted class over all occurrences (words seen ≥ 5 times); signed per-class
means and instance-level explanations are also saved. The settings are written to `shap_settings.json`.

**Statistics** (`src/hyphix/evaluation.py`): Wilson score intervals, 10,000-resample percentile bootstrap of
macro-F1, exact McNemar test between models on the same tweets, t-based intervals over seeds, paired t-test and
Wilcoxon signed-rank test over seeds.

## 6. Mapping to the manuscript

| Manuscript | Produced by |
|---|---|
| Table 2 (class distribution) | `scripts/data_report.py` → `reports/class_distribution_retained.csv` |
| Table 3 (settings), Tables 4–5 (environment) | `configs/default.yaml`, `runs/*/run_info.json`, `runs/*/environment.json` |
| Table 6, Figures 4–5 | `scripts/compare_runs.py` (CNN, BiLSTM, BERT, CB, BB) |
| Tables 7–8 (with/without SMOTE) | `runs/cb_seed2018`, `runs/cb_smote_seed2018`, `runs/bb_*` → `classification_validation.csv` |
| Figure 7, Table 9 (confusion matrix, CIs) | `runs/cb_smote_seed2018/confusion_validation.png`, `metrics_validation.json` |
| Figures 8–9 (SHAP) | `scripts/explain_shap.py` → `shap_validation/shap_global_top_words.png` |
| Multi-seed results, significance tests | `scripts/multiseed.py`, `scripts/compare_runs.py` |
| Ablations of H2 (future work in the current manuscript) | `scripts/train.py --model hyphix ...` |

Numbers obtained with this code replace any value in the manuscript that was not produced by it; see
`docs/REPRODUCIBILITY_NOTES.md` for the differences between the original notebooks and the manuscript text.

## 7. Tests

```bash
pip install pytest
python tests/make_tiny_bert.py     # tiny random BERT, no download needed
pytest -q tests                    # runs every model and option on 400 tweets in a few minutes (CPU)
```

## 8. Citation

If you use this code, please cite the manuscript (details will be added upon publication) and the original
repository by Avula, Rohan and Shashidhara.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`).
