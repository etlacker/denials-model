# denials-model

## Overview

Synthetic claims/denial dataset augmented to ~35% denials, cleaned/enriched for ML, with multiple denial-prediction models evaluated. **CatBoost (stage1c) achieves best performance** with 0.701 AP and 0.600 F1. LLM reasoning augmentation (stage1d) did not improve results.

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

## Current Model Stats (eval set, 35% denials)

| Model                   | AP        | Best F1   | Threshold | Notes               |
| ----------------------- | --------- | --------- | --------- | ------------------- |
| CatBoost (stage1c)      | **0.701** | **0.600** | 0.321     | **Best model**      |
| Reasoning-Aug (stage1d) | 0.694     | 0.595     | 0.259     | LLM embeddings hurt |
| XGBoost (stage1a)       | 0.665     | 0.581     | 0.400     |                     |
| LightGBM (stage1b)      | 0.649     | 0.567     | 0.281     |                     |

## Plain English Performance Summary

The evaluation set contains **11,895 claims** total:

- **4,203 actual denials** (35.3%)
- **7,692 valid claims** (64.7%)

### CatBoost (stage1c) - Best Model

At the **optimal threshold (0.275)**:

- **Caught 2,901 of 4,203 denials** (69% recall) — the model correctly identified about 7 in 10 claims that would be denied
- **Missed 1,302 denials** — these claims were predicted valid but actually got denied
- **Flagged 2,689 valid claims as denials** (35% false positive rate) — about 1 in 3 valid claims was incorrectly flagged
- **When the model predicts "denial," it's right 52% of the time** (precision)

At a **high-confidence threshold (0.5)**:

- **Caught 1,438 of 4,203 denials** (34% recall) — only catches the most obvious denials
- **Missed 2,765 denials** — most denials slip through
- **Flagged only 303 valid claims as denials** (4% false positive rate) — very few false alarms
- **When the model predicts "denial," it's right 83% of the time** (precision)

**Trade-off**: Lower thresholds catch more denials but create more false alarms. Higher thresholds are more precise but miss more denials.

### XGBoost (stage1a)

At the **optimal threshold (0.419)**:

- **Caught 2,913 of 4,203 denials** (69% recall)
- **Flagged 3,008 valid claims as denials** (39% false positive rate)
- **When predicting "denial," right 49% of the time** (precision)

### LightGBM (stage1b)

At the **optimal threshold (0.281)**:

- **Caught 2,967 of 4,203 denials** (71% recall)
- **Flagged 3,290 valid claims as denials** (43% false positive rate)
- **When predicting "denial," right 47% of the time** (precision)

### Reasoning-Augmented CatBoost (stage1d)

- Uses Gemini 3.0 Flash to generate denial risk reasoning per claim
- Embeddings via sentence-transformers (all-MiniLM-L6-v2, 384 dims)
- Parallelized API calls (~28 claims/sec with 50 workers)

At the **optimal threshold (0.259)**:

- **Caught 3,098 of 4,203 denials** (74% recall)
- **Flagged 3,116 valid claims as denials** (41% false positive rate)
- **When predicting "denial," right 50% of the time** (precision)

**Result: Slightly underperforms baseline CatBoost (-0.7% AP, -0.5% F1)** despite higher recall, the extra false positives hurt overall balance.

#### Why didn't LLM reasoning help?

1. **Information redundancy** - LLM reasons about features the model already sees
2. **Noise injection** - 384 embedding dims may add more noise than signal
3. **Synthetic data** - LLM's real-world denial knowledge doesn't match synthetic patterns
4. **Embedding mismatch** - General-purpose encoder may not capture medical reasoning well

**Conclusion:** Simpler is better here. CatBoost (stage1c) remains the best model.

## Glossary

- **AP (Average Precision)**: Overall ranking quality, 0-1 scale. Higher = model ranks denials above valid claims more consistently.
- **F1 Score**: Balance between precision and recall, 0-1 scale. Higher = better trade-off between catching denials and avoiding false alarms.
- **Recall**: % of actual denials the model catches
- **Precision**: When model predicts "denial," how often it's correct
- **False Positive Rate**: % of valid claims incorrectly flagged as denials
- **Threshold**: Confidence cutoff — predictions above this are labeled "denial"

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
