# denials-model

## Overview

Synthetic claims/denial dataset augmented to ~30% denials, cleaned/enriched for ML, with multiple denial-prediction models evaluated. **CatBoost (stage1c) achieves best performance** with 0.701 AP and 0.600 F1. LLM reasoning augmentation (stage1d) did not improve results.

## Artifacts Directory Structure

```
artifacts/
├── augment_denials/     # Output from augment_denials.py
│   ├── claims_and_billing_augmented.csv
│   ├── denials_augmented.csv
│   └── denial_augmentation_summary.json
├── data_cleaning/       # Output from data_cleaning.py
│   ├── claims_and_billing_cleaned.csv
│   ├── claims_enriched.csv
│   ├── claims_enriched_train.csv
│   └── claims_enriched_eval.csv
├── stage1a/             # XGBoost model
├── stage1b/             # LightGBM model
├── stage1c/             # CatBoost model
└── stage1d/             # Reasoning-augmented model
```

## Data Pipeline

1. **Augmentation:** `uv run python augment_denials.py`

   - Outputs to `artifacts/augment_denials/`

2. **Cleaning/enrichment:** `uv run python data_cleaning.py`

   - Reads from `artifacts/augment_denials/` if present, otherwise `raw_data/`
   - Outputs to `artifacts/data_cleaning/`

3. **Modeling:** (each stage reads from `artifacts/data_cleaning/`)
   - XGBoost: `uv run python stage1a_model.py` → `artifacts/stage1a/`
   - LightGBM: `uv run python stage1b_model.py` → `artifacts/stage1b/`
   - CatBoost: `uv run python stage1c_model.py` → `artifacts/stage1c/`
   - Reasoning-augmented: `uv run python stage1d_reasoning_model.py` → `artifacts/stage1d/`
     - Requires `GEMINI_API_KEY` in `.env` file

## Current Model Stats (eval set, 30% positives)

| Model                   | AP        | Best F1   | Threshold | Notes               |
| ----------------------- | --------- | --------- | --------- | ------------------- |
| CatBoost (stage1c)      | **0.701** | **0.600** | 0.321     | **Best model**      |
| Reasoning-Aug (stage1d) | 0.694     | 0.595     | 0.259     | LLM embeddings hurt |
| XGBoost (stage1a)       | 0.665     | 0.581     | 0.400     |                     |
| LightGBM (stage1b)      | ~0.63     | -         | -         | CV only             |

### XGBoost (stage1a)

- AP: 0.665
- Best F1: 0.581 @ threshold ≈ 0.400

### LightGBM (stage1b)

- CV AP: ~0.63 (3-fold)
- Baseline, trails XGBoost

### CatBoost (stage1c) - Best Model

- Eval AP: 0.701
- Best F1: 0.600 @ threshold ≈ 0.321

### Reasoning-Augmented CatBoost (stage1d)

- Uses Gemini 3.0 Flash to generate denial risk reasoning per claim
- Embeddings via sentence-transformers (all-MiniLM-L6-v2, 384 dims)
- Parallelized API calls (~28 claims/sec with 50 workers)
- **Eval AP: 0.694** | **Best F1: 0.595** @ threshold ≈ 0.259
- **Result: Slightly underperforms baseline CatBoost (-0.7% AP, -0.5% F1)**

#### Why didn't LLM reasoning help?

1. **Information redundancy** - LLM reasons about features the model already sees
2. **Noise injection** - 384 embedding dims may add more noise than signal
3. **Synthetic data** - LLM's real-world denial knowledge doesn't match synthetic patterns
4. **Embedding mismatch** - General-purpose encoder may not capture medical reasoning well

**Conclusion:** Simpler is better here. CatBoost (stage1c) remains the best model.

## Usage

```bash
# 1. Setup environment
uv sync
source .venv/bin/activate

# 2. Create .env with GEMINI_API_KEY (for stage1d)
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run full pipeline
uv run python augment_denials.py
uv run python data_cleaning.py
uv run python stage1a_model.py
uv run python stage1b_model.py
uv run python stage1c_model.py
uv run python stage1d_reasoning_model.py
```
