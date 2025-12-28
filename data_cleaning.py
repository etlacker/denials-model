"""
Script version of data_cleaning.ipynb to build cleaned and enriched claim data.
Steps:
1) Clean claims/denials, reconcile denial reasons, write cleaned CSV.
2) Join encounters/patients/providers and per-encounter aggregates, write enriched CSV.
3) Grouped split by patient into train/eval CSVs.

Augmented data support:
- If augmented files exist at artifacts/claims_and_billing_augmented.csv and artifacts/denials_augmented.csv, they are used as inputs.
- Otherwise raw_data/*.csv are used.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_DATA_DIR = Path("raw_data")
ARTIFACTS_DIR = Path("artifacts")
AUGMENT_DIR = ARTIFACTS_DIR / "augment_denials"
STAGE_DIR = ARTIFACTS_DIR / "data_cleaning"
STAGE_DIR.mkdir(parents=True, exist_ok=True)

RAW_CLAIMS_PATH = RAW_DATA_DIR / "claims_and_billing.csv"
RAW_DENIALS_PATH = RAW_DATA_DIR / "denials.csv"
AUG_CLAIMS_PATH = AUGMENT_DIR / "claims_and_billing_augmented.csv"
AUG_DENIALS_PATH = AUGMENT_DIR / "denials_augmented.csv"

CLEAN_CLAIMS_PATH = STAGE_DIR / "claims_and_billing_cleaned.csv"
OUTPUT_JOINED_PATH = STAGE_DIR / "claims_enriched.csv"
TRAIN_PATH = STAGE_DIR / "claims_enriched_train.csv"
EVAL_PATH = STAGE_DIR / "claims_enriched_eval.csv"


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


def resolve_source_paths():
    use_augmented = AUG_CLAIMS_PATH.exists() and AUG_DENIALS_PATH.exists()
    claims_path = AUG_CLAIMS_PATH if use_augmented else RAW_CLAIMS_PATH
    denials_path = AUG_DENIALS_PATH if use_augmented else RAW_DENIALS_PATH
    source = "augmented" if use_augmented else "raw"
    print(f"[info] using {source} inputs -> claims: {claims_path}, denials: {denials_path}")
    return claims_path, denials_path


def clean_claims_and_denials(claims_path: Path | None = None, denials_path: Path | None = None) -> pd.DataFrame:
    if claims_path is None or denials_path is None:
        claims_path, denials_path = resolve_source_paths()
    claims = pd.read_csv(claims_path)
    denials = pd.read_csv(denials_path)
    claims = deduplicate_claims(claims)
    claims, mapping = reconcile_reasons(claims, denials)
    summarize(claims, mapping)
    claims.to_csv(CLEAN_CLAIMS_PATH, index=False)
    print(f"\n[done] wrote cleaned claims to {CLEAN_CLAIMS_PATH}")
    return claims


def add_counts(df: pd.DataFrame, table: pd.DataFrame, key: str, col: str, prefix: str):
    counts = table.groupby(key).size().rename(f"{prefix}_count")
    distinct = table.groupby(key)[col].nunique().rename(f"{prefix}_distinct")
    return df.merge(counts, left_on=key, right_index=True, how="left").merge(
        distinct, left_on=key, right_index=True, how="left"
    )


def build_enriched_dataset(claims: pd.DataFrame) -> pd.DataFrame:
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

    # Per-encounter aggregates
    joined = add_counts(joined, diagnoses, "encounter_id", "diagnosis_code", "diag")
    joined = add_counts(joined, procedures, "encounter_id", "procedure_code", "proc")
    joined = add_counts(joined, labs, "encounter_id", "test_code", "lab")
    joined = add_counts(joined, meds, "encounter_id", "drug_name", "med")

    # Fill count NA with 0
    for col in [c for c in joined.columns if c.endswith("_count") or c.endswith("_distinct")]:
        joined[col] = joined[col].fillna(0).astype(int)

    # Labels
    joined["label_denied"] = (joined["claim_status"] == "Denied").astype(int)
    joined["label_reason"] = joined["denial_reason_clean"].fillna("")

    # Drop leakage/PII columns not suitable for modeling
    leak_cols = {
        "label_reason",
        "denial_reason",
        "denial_reason_norm",
        "denial_reason_norm_denials",
        "denial_reason_clean",
        "claim_status",
        "paid_amount",
    }
    id_cols = {
        "billing_id",
        "claim_id",
        "encounter_id",
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
    joined = joined.drop(columns=list(leak_cols | id_cols), errors="ignore")

    # Basic sanity checks
    print("joined rows", len(joined))
    print("label_denied distribution:", joined["label_denied"].value_counts().to_dict())

    # Save enriched dataset
    joined.to_csv(OUTPUT_JOINED_PATH, index=False)
    print(f"[done] wrote enriched claims to {OUTPUT_JOINED_PATH}")
    return joined


def split_train_eval(joined: pd.DataFrame):
    groups = joined["patient_id"]
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, eval_idx = next(splitter.split(joined, groups=groups))

    train_df = joined.iloc[train_idx].reset_index(drop=True)
    eval_df = joined.iloc[eval_idx].reset_index(drop=True)

    print("Train rows:", len(train_df), "Eval rows:", len(eval_df))
    print("Train label distribution:", train_df["label_denied"].value_counts().to_dict())
    print("Eval label distribution:", eval_df["label_denied"].value_counts().to_dict())

    train_df.to_csv(TRAIN_PATH, index=False)
    eval_df.to_csv(EVAL_PATH, index=False)
    print(f"[done] wrote train -> {TRAIN_PATH}\n[done] wrote eval -> {EVAL_PATH}")


def main():
    cleaned_claims = clean_claims_and_denials()
    enriched = build_enriched_dataset(cleaned_claims)
    split_train_eval(enriched)


if __name__ == "__main__":
    main()

