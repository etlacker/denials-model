# denials-model

## Overview

Synthetic claims/denial dataset augmented to ~30% denials, cleaned/enriched for ML, and trained denial-prediction models (XGBoost primary, LightGBM baseline).

## Data pipeline

1. Augmentation:
   - `augment_denials.py` -> writes `artifacts/claims_and_billing_augmented.csv`, `artifacts/denials_augmented.csv`, summary JSON.
2. Cleaning/enrichment:
   - `uv run python data_cleaning.py`
   - Uses augmented inputs if present; outputs:
     - `artifacts/claims_and_billing_cleaned.csv`
     - `artifacts/claims_enriched.csv`
     - `artifacts/claims_enriched_train.csv`
     - `artifacts/eval/claims_enriched_eval.csv`
3. Modeling:
   - XGBoost sweep + optional calibration: `uv run python stage1a_model.py`
     - Saves model `artifacts/claim_denial_model.joblib`
     - Metrics `artifacts/claim_denial_metrics.json`
   - LightGBM baseline: `uv run python stage1b_model.py`
     - Saves model `artifacts/claim_denial_model_lgbm.joblib`
     - Metrics `artifacts/claim_denial_metrics_lgbm.json`
   - CatBoost (categorical handling): `uv run python stage1c_model.py`
     - Saves model `artifacts/claim_denial_model_catboost.cbm`
     - Metrics `artifacts/claim_denial_metrics_catboost.json`

## Current model stats (eval set, 30% positives)

 XGBoost (stage1a, best config from sweep; uncalibrated chosen)

- AP: 0.665
- F1 @ 0.5: 0.571
- Best F1: 0.581 @ threshold ≈ 0.400
- Confusion @ best F1: tp 3,054 / fp 3,237 / fn 1,162 / tn 4,442
- Threshold tips: use ~0.40 for best F1; cost-based thresholds in metrics JSON.

LightGBM (stage1b baseline)

- CV AP: ~0.63 (3-fold)
- Eval metrics are in `claim_denial_metrics_lgbm.json` (baseline trails XGBoost).
- Confusion/more details: see metrics JSON.

 CatBoost (stage1c, categorical model)
 
 - Eval AP: 0.701
 - F1 @ 0.5: 0.502
 - Best F1: 0.600 @ threshold ≈ 0.321
 - Confusion @ best F1: tp 2,577 / fp 1,793 / fn 1,639 / tn 5,886
 - Best-cost threshold (FN cost=5, FP=1): ~0.20 (higher recall, more FP)

Improvement ideas (with rough expected lift)

- Better domain features: explicit days-to-file, duplicate flags, payer-age interactions, denial-type priors. Likely +0.01–0.03 AP if signals are informative.
- Broader hyperparameter sweep (depth/leaves/lr) or CatBoost: incremental, +0.005–0.02 AP.
- Probability calibration (kept optional): may improve decision thresholds even if AP unchanged.
- Operational tuning: choose thresholds based on FP/FN cost; current best-F1 ~0.40. Lower threshold -> more recall, higher threshold -> fewer false alarms.

## Usage

1. Ensure env: `uv sync` (Python 3.13). Activate: `source .venv/bin/activate`.
2. Run cleaning/enrichment: `uv run python data_cleaning.py`.
3. Train XGBoost sweep: `uv run python stage1a_model.py`.
4. Train LightGBM baseline: `uv run python stage1b_model.py`.
