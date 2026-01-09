"""
Script version of data_cleaning.ipynb to build cleaned and enriched claim data.
Steps:
1) Clean claims/denials, reconcile denial reasons, write cleaned CSV.
2) Join encounters/patients/providers and per-encounter aggregates, write enriched CSV.
3) Grouped split by patient into train/eval CSVs.

Augmented data support:
- By default, uses augmented data if available.
- Use --no-augment to force using raw data.
- Use --both to generate both augmented and non-augmented datasets.
- Use --balanced to downsample valid claims to match augmented denial rate (~35%).
- Use --all to generate all three datasets (raw, augmented, balanced).

Outputs are written to:
- artifacts/data_cleaning_augmented/ (when using augmented data)
- artifacts/data_cleaning_raw/ (when using raw data)
- artifacts/data_cleaning_balanced/ (when using balanced raw data)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGET_DENIAL_RATE = 0.35  # Target denial rate for balanced dataset

RAW_DATA_DIR = Path("raw_data")
ARTIFACTS_DIR = Path("artifacts")
AUGMENT_DIR = ARTIFACTS_DIR / "augment_denials"

RAW_CLAIMS_PATH = RAW_DATA_DIR / "claims_and_billing.csv"
RAW_DENIALS_PATH = RAW_DATA_DIR / "denials.csv"
AUG_CLAIMS_PATH = AUGMENT_DIR / "claims_and_billing_augmented.csv"
AUG_DENIALS_PATH = AUGMENT_DIR / "denials_augmented.csv"


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[.]+$", "", regex=True)
    )


def deduplicate_claims(claims: pd.DataFrame) -> pd.DataFrame:
    dup_mask = claims["claim_id"].duplicated(keep=False)
    if dup_mask.any():
        print(f"[warn] Found {dup_mask.sum()} rows with duplicate claim_id; keeping first per claim_id.")
    return claims.drop_duplicates(subset=["claim_id"], keep="first")


def reconcile_reasons(claims: pd.DataFrame, denials: pd.DataFrame):
    claims = claims.copy()
    denials = denials.copy()

    claims["claim_status"] = claims["claim_status"].str.strip()
    claims = claims[claims["claim_status"].isin(["Paid", "Denied"])]

    claims["denial_reason_norm"] = normalize_text(claims.get("denial_reason", pd.Series(dtype=str)))
    denials["denial_reason_norm"] = normalize_text(denials.get("denial_reason_description", pd.Series(dtype=str)))

    claims_counts = claims.loc[claims["claim_status"] == "Denied", "denial_reason_norm"].value_counts()
    denials_counts = denials["denial_reason_norm"].value_counts()
    mapping = {}
    if len(claims_counts) == len(denials_counts):
        mapping = {long: short for short, long in zip(claims_counts.index, denials_counts.index)}

    denials_small = denials[["claim_id", "denial_reason_norm"]].drop_duplicates("claim_id")
    claims = claims.merge(denials_small, on="claim_id", how="left", suffixes=("", "_denials"))

    def choose_reason(row):
        r_claims = row["denial_reason_norm"]
        r_denials = row.get("denial_reason_norm_denials", "")
        if r_claims:
            return r_claims
        if r_denials:
            return mapping.get(r_denials, r_denials)
        return ""

    claims["denial_reason_clean"] = claims.apply(choose_reason, axis=1)
    claims.loc[claims["claim_status"] != "Denied", "denial_reason_clean"] = ""
    return claims, mapping


def summarize(claims: pd.DataFrame, mapping: dict):
    total = len(claims)
    denied = (claims["claim_status"] == "Denied").sum()
    print(f"[info] total claims: {total}, denied: {denied}, paid: {total - denied}")
    missing_reason = ((claims["claim_status"] == "Denied") & (claims["denial_reason_clean"] == "")).sum()
    print(f"[info] denied with missing clean reason: {missing_reason}")
    if mapping:
        print(f"[info] mapped {len(mapping)} denial descriptions to short forms.")
    top_reasons = claims.loc[claims["claim_status"] == "Denied", "denial_reason_clean"].value_counts().head(10)
    print("\nTop denial reasons (clean):")
    print(top_reasons)


def get_output_dir(mode: str) -> Path:
    """Get output directory based on data source mode.
    
    Args:
        mode: One of 'augmented', 'raw', or 'balanced'
    """
    stage_dir = ARTIFACTS_DIR / f"data_cleaning_{mode}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def resolve_source_paths(no_augment: bool = False):
    use_augmented = not no_augment and AUG_CLAIMS_PATH.exists() and AUG_DENIALS_PATH.exists()
    claims_path = AUG_CLAIMS_PATH if use_augmented else RAW_CLAIMS_PATH
    denials_path = AUG_DENIALS_PATH if use_augmented else RAW_DENIALS_PATH
    source = "augmented" if use_augmented else "raw"
    print(f"[info] using {source} inputs -> claims: {claims_path}, denials: {denials_path}")
    return claims_path, denials_path, use_augmented


def downsample_valid_claims(claims: pd.DataFrame, target_denial_rate: float = TARGET_DENIAL_RATE) -> pd.DataFrame:
    """Downsample valid claims to achieve target denial rate.
    
    Keeps all denied claims and randomly samples valid claims to achieve
    the target denial rate (matching augmented dataset).
    """
    denied = claims[claims["claim_status"] == "Denied"]
    valid = claims[claims["claim_status"] != "Denied"]
    
    n_denied = len(denied)
    # Calculate how many valid claims to keep: n_denied / target_rate - n_denied
    n_total_target = int(n_denied / target_denial_rate)
    n_valid_target = n_total_target - n_denied
    
    if n_valid_target >= len(valid):
        print(f"[info] No downsampling needed, keeping all {len(valid)} valid claims")
        return claims
    
    print(f"[info] Downsampling valid claims: {len(valid)} -> {n_valid_target} (keeping all {n_denied} denied)")
    
    # Sample valid claims randomly with fixed seed for reproducibility
    valid_sampled = valid.sample(n=n_valid_target, random_state=42)
    
    # Combine and return
    balanced = pd.concat([denied, valid_sampled], ignore_index=True)
    return balanced


def clean_claims_and_denials(
    claims_path: Path | None = None,
    denials_path: Path | None = None,
    no_augment: bool = False,
    balanced: bool = False,
) -> tuple[pd.DataFrame, Path]:
    if claims_path is None or denials_path is None:
        claims_path, denials_path, use_augmented = resolve_source_paths(no_augment=no_augment)
    else:
        use_augmented = not no_augment
    
    # Determine output mode
    if balanced:
        mode = "balanced"
    elif use_augmented:
        mode = "augmented"
    else:
        mode = "raw"
    
    output_dir = get_output_dir(mode)
    clean_claims_path = output_dir / "claims_and_billing_cleaned.csv"
    
    claims = pd.read_csv(claims_path)
    denials = pd.read_csv(denials_path)
    claims = deduplicate_claims(claims)
    claims, mapping = reconcile_reasons(claims, denials)
    
    # Apply downsampling for balanced mode
    if balanced:
        claims = downsample_valid_claims(claims)
    
    summarize(claims, mapping)
    claims.to_csv(clean_claims_path, index=False)
    print(f"\n[done] wrote cleaned claims to {clean_claims_path}")
    return claims, output_dir


def add_counts_and_codes(df: pd.DataFrame, table: pd.DataFrame, key: str, col: str, prefix: str):
    """Add counts, distinct counts, and aggregated codes from a related table."""
    counts = table.groupby(key).size().rename(f"{prefix}_count")
    distinct = table.groupby(key)[col].nunique().rename(f"{prefix}_distinct")
    # Aggregate codes as pipe-separated string for each encounter
    codes_agg = table.groupby(key)[col].apply(lambda x: "|".join(sorted(set(x.dropna().astype(str))))).rename(f"{prefix}_codes")
    
    df = df.merge(counts, left_on=key, right_index=True, how="left")
    df = df.merge(distinct, left_on=key, right_index=True, how="left")
    df = df.merge(codes_agg, left_on=key, right_index=True, how="left")
    return df


def build_enriched_dataset(claims: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Load raw tables
    encounters = pd.read_csv(RAW_DATA_DIR / "encounters.csv")
    diagnoses = pd.read_csv(RAW_DATA_DIR / "diagnoses.csv")
    procedures = pd.read_csv(RAW_DATA_DIR / "procedures.csv")
    labs = pd.read_csv(RAW_DATA_DIR / "lab_tests.csv")
    meds = pd.read_csv(RAW_DATA_DIR / "medications.csv")
    patients = pd.read_csv(RAW_DATA_DIR / "patients.csv")
    providers = pd.read_csv(RAW_DATA_DIR / "providers.csv")

    # Deduplicate keys for safer joins
    encounters = encounters.drop_duplicates(subset=["encounter_id"], keep="first")
    patients = patients.drop_duplicates(subset=["patient_id"], keep="first")
    providers = providers.drop_duplicates(subset=["provider_id"], keep="first")

    # Keep only claims that have encounter_id and patient_id
    claims = claims[claims["encounter_id"].notna() & claims["patient_id"].notna()].copy()

    # Merge core dimensions
    joined = claims.merge(encounters, on="encounter_id", how="left", suffixes=("", "_enc"))
    joined = joined.merge(patients, on="patient_id", how="left", suffixes=("", "_pat"))
    joined = joined.merge(providers, on="provider_id", how="left", suffixes=("", "_prov"))

    # Per-encounter aggregates with actual codes preserved
    joined = add_counts_and_codes(joined, diagnoses, "encounter_id", "diagnosis_code", "diag")
    joined = add_counts_and_codes(joined, procedures, "encounter_id", "procedure_code", "proc")
    joined = add_counts_and_codes(joined, labs, "encounter_id", "test_code", "lab")
    joined = add_counts_and_codes(joined, meds, "encounter_id", "drug_name", "med")

    # Fill count NA with 0 and codes NA with empty string
    for col in [c for c in joined.columns if c.endswith("_count") or c.endswith("_distinct")]:
        joined[col] = joined[col].fillna(0).astype(int)
    for col in [c for c in joined.columns if c.endswith("_codes")]:
        joined[col] = joined[col].fillna("")

    # Labels
    joined["label_denied"] = (joined["claim_status"] == "Denied").astype(int)
    joined["label_reason"] = joined["denial_reason_clean"].fillna("")

    # Drop leakage/PII columns not suitable for modeling
    # Keep claim_id and encounter_id for traceability
    leak_cols = {
        "label_reason",
        "denial_reason",
        "denial_reason_norm",
        "denial_reason_norm_denials",
        "denial_reason_clean",
        "claim_status",
        "paid_amount",
    }
    pii_cols = {
        "billing_id",
        "provider_id",
        "patient_id_enc",
        "first_name",
        "last_name",
        "address",
        "city",
        "state",
        "zip",
        "phone",
        "email",
        "registration_date",
        "name",
        "department_prov",
        "contact_info",
        "email_prov",
    }
    joined = joined.drop(columns=list(leak_cols | pii_cols), errors="ignore")

    # Basic sanity checks
    print("joined rows", len(joined))
    print("label_denied distribution:", joined["label_denied"].value_counts().to_dict())

    # Save enriched dataset
    output_path = output_dir / "claims_enriched.csv"
    joined.to_csv(output_path, index=False)
    print(f"[done] wrote enriched claims to {output_path}")
    return joined


def split_train_eval(joined: pd.DataFrame, output_dir: Path):
    groups = joined["patient_id"]
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, eval_idx = next(splitter.split(joined, groups=groups))

    train_df = joined.iloc[train_idx].reset_index(drop=True)
    eval_df = joined.iloc[eval_idx].reset_index(drop=True)

    print("Train rows:", len(train_df), "Eval rows:", len(eval_df))
    print("Train label distribution:", train_df["label_denied"].value_counts().to_dict())
    print("Eval label distribution:", eval_df["label_denied"].value_counts().to_dict())

    train_path = output_dir / "claims_enriched_train.csv"
    eval_path = output_dir / "claims_enriched_eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    print(f"[done] wrote train -> {train_path}\n[done] wrote eval -> {eval_path}")


def run_pipeline(no_augment: bool = False, balanced: bool = False):
    """Run the full data cleaning pipeline for a single data source."""
    cleaned_claims, output_dir = clean_claims_and_denials(no_augment=no_augment, balanced=balanced)
    enriched = build_enriched_dataset(cleaned_claims, output_dir)
    split_train_eval(enriched, output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Clean and enrich claims data for denial prediction.")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Force using raw data instead of augmented data even if augmented files exist.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both augmented and non-augmented datasets.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Downsample valid claims from raw data to match augmented denial rate (~35%%).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all three datasets: raw, augmented, and balanced.",
    )
    args = parser.parse_args()

    if args.all:
        print("=" * 60)
        print("Processing RAW (non-augmented) data")
        print("=" * 60)
        run_pipeline(no_augment=True, balanced=False)
        
        print("\n" + "=" * 60)
        print("Processing AUGMENTED data")
        print("=" * 60)
        if AUG_CLAIMS_PATH.exists() and AUG_DENIALS_PATH.exists():
            run_pipeline(no_augment=False, balanced=False)
        else:
            print("[warn] Augmented data files not found, skipping augmented pipeline.")
            print(f"  Expected: {AUG_CLAIMS_PATH}")
            print(f"  Expected: {AUG_DENIALS_PATH}")
        
        print("\n" + "=" * 60)
        print("Processing BALANCED (downsampled) data")
        print("=" * 60)
        run_pipeline(no_augment=True, balanced=True)
        
    elif args.both:
        print("=" * 60)
        print("Processing RAW (non-augmented) data")
        print("=" * 60)
        run_pipeline(no_augment=True, balanced=False)
        
        print("\n" + "=" * 60)
        print("Processing AUGMENTED data")
        print("=" * 60)
        if AUG_CLAIMS_PATH.exists() and AUG_DENIALS_PATH.exists():
            run_pipeline(no_augment=False, balanced=False)
        else:
            print("[warn] Augmented data files not found, skipping augmented pipeline.")
            print(f"  Expected: {AUG_CLAIMS_PATH}")
            print(f"  Expected: {AUG_DENIALS_PATH}")
    elif args.balanced:
        run_pipeline(no_augment=True, balanced=True)
    else:
        run_pipeline(no_augment=args.no_augment, balanced=False)


if __name__ == "__main__":
    main()

