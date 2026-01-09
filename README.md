# denials-model

## Overview

Synthetic claims/denial dataset with multiple denial-prediction models evaluated on three dataset variants:

1. **Raw** (8.5% denials) — Original imbalanced data
2. **Augmented** (35% denials) — Synthetic denials added to increase denial rate
3. **Balanced** (35% denials) — Raw data with valid claims downsampled

**Key finding:** On augmented data, **CatBoost achieves 0.689 AP and 0.591 F1**. Balanced downsampling achieves ~0.35 AP — better than raw but worse than augmented due to smaller training set.

## Data Setup

Before running any scripts, download the raw data from Kaggle:

1. Download the dataset from: https://www.kaggle.com/datasets/rajkumarpadmanabhan/ca-hospital-dataset-q1-2025
2. Extract the CSV files into the `raw_data/` directory

The `raw_data/` folder should contain:

```
raw_data/
├── claims_and_billing.csv
├── denials.csv
├── diagnoses.csv
├── encounters.csv
├── lab_tests.csv
├── medications.csv
├── patients.csv
├── procedures.csv
└── providers.csv
```

## Artifacts Directory Structure

```
artifacts/
├── augment_denials/           # Output from augment_denials.py
│   ├── claims_and_billing_augmented.csv
│   ├── denials_augmented.csv
│   └── denial_augmentation_summary.json
├── data_cleaning_raw/         # Cleaned data from raw inputs (8.5% denials, 59,639 claims)
├── data_cleaning_augmented/   # Cleaned data from augmented inputs (35% denials, 59,639 claims)
├── data_cleaning_balanced/    # Downsampled raw data (35% denials, 17,137 claims)
├── stage1a_raw/               # XGBoost on raw data
├── stage1a_augmented/         # XGBoost on augmented data
├── stage1a_balanced/          # XGBoost on balanced data
├── stage1b_*/                 # LightGBM variants
├── stage1c_*/                 # CatBoost variants
└── stage1d/                   # Reasoning-augmented model
```

## Data Pipeline

### 1. Augmentation (optional)

```bash
uv run python augment_denials.py
```

- Creates synthetic denials to reach ~35% denial rate
- Outputs to `artifacts/augment_denials/`

### 2. Data Cleaning/Enrichment

```bash
# Generate all three datasets (raw, augmented, balanced)
uv run python data_cleaning.py --all

# Or generate specific dataset
uv run python data_cleaning.py --no-augment    # Raw only
uv run python data_cleaning.py                  # Augmented only (default)
uv run python data_cleaning.py --balanced       # Balanced (downsampled) only
uv run python data_cleaning.py --both           # Raw + Augmented
```

### 3. Model Training

```bash
# Train on all three datasets
uv run python stage1a_model.py --both
uv run python stage1b_model.py --both
uv run python stage1c_model.py --both

# Or train on specific dataset
uv run python stage1a_model.py --data-dir artifacts/data_cleaning_raw
uv run python stage1a_model.py --data-dir artifacts/data_cleaning_augmented
uv run python stage1a_model.py --data-dir artifacts/data_cleaning_balanced
```

## Model Performance Comparison

### Augmented Data (35% denials, 59,639 train claims, 11,895 eval)

| Model              | AP        | Best F1   | Threshold |
| ------------------ | --------- | --------- | --------- |
| CatBoost (stage1c) | **0.689** | **0.591** | 0.292     |
| XGBoost (stage1a)  | 0.646     | 0.572     | 0.406     |
| LightGBM (stage1b) | 0.645     | 0.565     | 0.281     |

### Balanced Data (35% denials, 17,137 train claims, 3,419 eval)

| Model              | AP    | Best F1 | Threshold |
| ------------------ | ----- | ------- | --------- |
| XGBoost (stage1a)  | 0.357 | 0.524   | 0.092     |
| LightGBM (stage1b) | 0.352 | 0.525   | 0.026     |
| CatBoost (stage1c) | 0.350 | 0.525   | 0.133     |

### Raw Data (8.5% denials, 59,639 train claims, 11,895 eval)

| Model              | AP    | Best F1 | Threshold |
| ------------------ | ----- | ------- | --------- |
| XGBoost (stage1a)  | 0.103 | 0.182   | 0.178     |
| CatBoost (stage1c) | 0.101 | 0.182   | 0.066     |
| LightGBM (stage1b) | 0.098 | 0.182   | 0.017     |

## Analysis

### Why Augmented > Balanced > Raw?

| Dataset   | Denial Rate | Train Size | AP Range    | Why?                                      |
| --------- | ----------- | ---------- | ----------- | ----------------------------------------- |
| Augmented | 35%         | 47,744     | 0.65 - 0.69 | More data + balanced classes              |
| Balanced  | 35%         | 13,718     | 0.35 - 0.36 | Balanced but fewer examples to learn from |
| Raw       | 8.5%        | 47,744     | 0.10 - 0.10 | Extreme imbalance kills learning          |

**Key insights:**

1. **Class balance matters more than data size** — Balanced beats Raw despite 3.5x fewer training examples
2. **More data still helps** — Augmented beats Balanced by ~2x AP with same denial rate but 3.5x more data
3. **Augmentation adds learnable patterns** — Synthetic denials are rule-based, making them more predictable

### Balanced Dataset Approach

The balanced dataset is created by:

1. Keeping all 5,998 denied claims from raw data
2. Randomly sampling 11,139 valid claims (to achieve 35% denial rate)
3. Total: 17,137 claims (vs 59,639 in raw/augmented)

This tests whether the augmented model's improvement comes from class balance alone or also from having more training examples.

**Result:** Balanced achieves F1 ~0.52 vs Augmented F1 ~0.59, confirming that both class balance AND more data contribute to performance.

## Plain English Performance Summary

### Augmented Data - CatBoost (Best Model)

The evaluation set contains **11,895 claims** total:

- **4,203 actual denials** (35.3%)
- **7,692 valid claims** (64.7%)

At the **optimal threshold (0.292)**:

- **Caught ~70% of denials** — correctly identifies about 7 in 10 claims that would be denied
- **Precision ~52%** — when predicting "denial," right about half the time
- **F1 Score: 0.591** — good balance between catching denials and avoiding false alarms

### Balanced Data - All Models Similar

The evaluation set contains **3,419 claims** (smaller due to downsampling):

- **1,215 actual denials** (35.5%)
- **2,204 valid claims** (64.5%)

At optimal thresholds:

- **Caught ~100% of denials** (at very low thresholds)
- **Precision ~35%** — many false positives
- **F1 Score: ~0.52** — decent but not as good as augmented

### Raw Data Limitations

On raw data (8.5% denials), all models achieve only ~0.10 AP because:

1. **Extreme class imbalance** — only 1 in 12 claims is denied
2. **Models learn to predict "not denied"** — achieves 91.5% accuracy but catches zero denials
3. **All F1 scores cluster around 0.18** — the best possible given the imbalance

## Data Features

The enriched dataset includes:

- **Patient info:** age, gender, ethnicity, insurance_type, marital_status
- **Encounter info:** visit_type, department, reason_for_visit, admission_type
- **Provider info:** specialty, years_experience, npi, location
- **Clinical codes:** diagnosis_code, proc_codes, lab_codes, med_codes (pipe-separated)
- **Aggregate counts:** diag_count, proc_count, lab_count, med_count
- **Billing:** billed_amount, claim_billing_date
- **IDs preserved:** claim_id, encounter_id, patient_id (for traceability)

## Glossary

- **AP (Average Precision)**: Overall ranking quality, 0-1 scale. Higher = model ranks denials above valid claims more consistently.
- **F1 Score**: Balance between precision and recall, 0-1 scale. Higher = better trade-off between catching denials and avoiding false alarms.
- **Recall**: % of actual denials the model catches
- **Precision**: When model predicts "denial," how often it's correct
- **Threshold**: Confidence cutoff — predictions above this are labeled "denial"

## Quick Start

```bash
# 1. Setup environment
uv sync

# 2. Run full pipeline (all three datasets)
uv run python augment_denials.py
uv run python data_cleaning.py --all
uv run python stage1a_model.py --both
uv run python stage1b_model.py --both
uv run python stage1c_model.py --both

# 3. (Optional) Run reasoning-augmented model (requires GEMINI_API_KEY in .env)
echo "GEMINI_API_KEY=your_key_here" > .env
uv run python stage1d_reasoning_model.py
```
