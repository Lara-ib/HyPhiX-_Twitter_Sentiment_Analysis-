#!/usr/bin/env bash
# Full experimental plan of the manuscript. Run from the repository root on a GPU machine.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/data_report.py

# 1. baselines and hybrids without oversampling (Table 6, Figures 4-5)
for m in cnn bilstm bert cb bb; do python scripts/train.py --model "$m"; done

# 2. hybrids with SMOTE (Tables 7-8, Figure 7, Table 9)
python scripts/train.py --model cb --balance smote
python scripts/train.py --model bb --balance smote

# 3. complete HyPhiX model and ablations (hypothesis H2)
python scripts/train.py --model hyphix --fusion gate
python scripts/train.py --model hyphix --fusion concat
python scripts/train.py --model hyphix --fusion equal
python scripts/train.py --model hyphix --fusion gate --no-psycho

# 4. SHAP explanations (Figures 8-9)
python scripts/explain_shap.py --run runs/cb_smote_seed2018 --eval-set validation
python scripts/explain_shap.py --run runs/bb_smote_seed2018 --eval-set validation

# 5. comparison tables, McNemar tests and figures on every evaluation set
for s in validation test validation_dedup; do
  python scripts/compare_runs.py --eval-set "$s" \
    --runs runs/cnn_seed2018 runs/bilstm_seed2018 runs/bert_seed2018 runs/cb_seed2018 runs/cb_smote_seed2018 \
           runs/bb_seed2018 runs/bb_smote_seed2018 runs/hyphix_gate_seed2018 runs/hyphix_concat_seed2018 \
           runs/hyphix_equal_seed2018 runs/hyphix_gate_nopsy_seed2018 \
    --labels CNN BiLSTM BERT CB CB+SMOTE BB BB+SMOTE HyPhiX-gate HyPhiX-concat HyPhiX-equal HyPhiX-noPsy
done

# 6. multi-seed experiments with paired tests
python scripts/multiseed.py --model cb --balance smote --seeds 2018 42 7 123 2024
python scripts/multiseed.py --model bb --balance smote --seeds 2018 42 7 123 2024 --compare-to reports/multiseed/cb_smote.json

# 7. leakage-aware protocol (grouped split)
python scripts/train.py --model cb --balance smote --split group
python scripts/train.py --model bb --balance smote --split group
