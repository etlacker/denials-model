"""
Build a flattened knowledge base of denied claims only (excluding self-pay),
preserving original text fields (no normalization of denial reasons).

Outputs:
- CSV: artifacts/knowledge_base/denied_claims_knowledge_base.csv (by default)
  Contains all denied claims EXCEPT a random holdout set for testing.
- MD: artifacts/knowledge_base/denied_claims_holdout.md (by default)
  Contains a random sample of 50 denied claims (or fewer if not available),
  each as a JSON block with a header describing expected outcome.

Usage:
  python knowledge_base_prep.py
  python knowledge_base_prep.py --output artifacts/knowledge_base/custom.csv --holdout-md artifacts/knowledge_base/custom_holdout.md --holdout-size 50

Notes:
- Excludes self-pay using filtering only (values are not modified in the output):
  • claims.payment_method in {"self-pay", "self pay"}
  • claims.insurance_provider in {"self-pay", "self pay"}
  • patients.insurance_type in {"self-pay", "self pay"}
- Preserves raw denial_reason fields (no normalization/reconciliation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("raw_data")
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "knowledge_base"
DEFAULT_CSV_PATH = DEFAULT_OUT_DIR / "denied_claims_knowledge_base.csv"
DEFAULT_HOLDOUT_MD_PATH = DEFAULT_OUT_DIR / "denied_claims_holdout.md"


# Raw file paths
RAW_CLAIMS_PATH = RAW_DATA_DIR / "claims_and_billing.csv"
RAW_DENIALS_PATH = RAW_DATA_DIR / "denials.csv"
RAW_ENCOUNTERS_PATH = RAW_DATA_DIR / "encounters.csv"
RAW_DIAGNOSES_PATH = RAW_DATA_DIR / "diagnoses.csv"
RAW_PROCEDURES_PATH = RAW_DATA_DIR / "procedures.csv"
RAW_LABS_PATH = RAW_DATA_DIR / "lab_tests.csv"
RAW_MEDS_PATH = RAW_DATA_DIR / "medications.csv"
RAW_PATIENTS_PATH = RAW_DATA_DIR / "patients.csv"
RAW_PROVIDERS_PATH = RAW_DATA_DIR / "providers.csv"


def ensure_output_dir(path: Path) -> None:
    out_dir = path.parent if path.suffix else path
    out_dir.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame, key: str) -> pd.DataFrame:
    dup_mask = df[key].duplicated(keep=False)
    if dup_mask.any():
        print(f"[warn] Found {dup_mask.sum()} rows with duplicate {key}; keeping first per {key}.")
    return df.drop_duplicates(subset=[key], keep="first")


def norm_for_filter(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def is_self_pay_series(series: pd.Series) -> pd.Series:
    s = norm_for_filter(series)
    return s.isin({"self-pay", "self pay"})


def aggregate_pipe(series: pd.Series) -> str:
    vals = [str(v) for v in series.dropna().astype(str) if str(v).strip() != ""]
    if not vals:
        return ""
    uniq = sorted(set(vals))
    return "|".join(uniq)


def aggregate_any_yes(series: pd.Series) -> str:
    s = norm_for_filter(series)
    return "Yes" if (s == "yes").any() else "No"


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_counts_codes_desc(
    base: pd.DataFrame,
    table: pd.DataFrame,
    key: str,
    code_col: str,
    prefix: str,
    desc_col: str | None = None,
) -> pd.DataFrame:
    if table.empty:
        base[f"{prefix}_count"] = 0
        base[f"{prefix}_distinct"] = 0
        base[f"{prefix}_codes"] = ""
        if desc_col:
            base[f"{prefix}_descriptions"] = ""
        return base

    grp = table.groupby(key, dropna=False)
    counts = grp.size().rename(f"{prefix}_count")
    distinct = grp[code_col].nunique(dropna=True).rename(f"{prefix}_distinct")
    codes_agg = grp[code_col].apply(aggregate_pipe).rename(f"{prefix}_codes")

    base = base.merge(counts, left_on=key, right_index=True, how="left")
    base = base.merge(distinct, left_on=key, right_index=True, how="left")
    base = base.merge(codes_agg, left_on=key, right_index=True, how="left")

    if desc_col:
        descs_agg = grp[desc_col].apply(aggregate_pipe).rename(f"{prefix}_descriptions")
        base = base.merge(descs_agg, left_on=key, right_index=True, how="left")

    return base


def aggregate_denials(denials: pd.DataFrame) -> pd.DataFrame:
    if denials.empty:
        return pd.DataFrame(columns=[
            "claim_id",
            "denial_ids",
            "denial_reason_codes",
            "denial_reason_descriptions",
            "denial_dates",
            "denied_amount_sum",
            "denied_amount_max",
            "appeal_filed_any",
            "appeal_statuses",
            "appeal_resolution_dates",
            "final_outcomes",
        ])

    d = denials.copy()
    d["denied_amount_num"] = to_numeric_safe(d.get("denied_amount", pd.Series(dtype=float))).fillna(0.0)

    grp = d.groupby("claim_id", dropna=False)
    agg = pd.DataFrame({
        "denial_ids": grp["denial_id"].apply(aggregate_pipe),
        "denial_reason_codes": grp["denial_reason_code"].apply(aggregate_pipe),
        "denial_reason_descriptions": grp["denial_reason_description"].apply(aggregate_pipe),
        "denial_dates": grp["denial_date"].apply(aggregate_pipe),
        "denied_amount_sum": grp["denied_amount_num"].sum(),
        "denied_amount_max": grp["denied_amount_num"].max(),
        "appeal_filed_any": grp["appeal_filed"].apply(aggregate_any_yes),
        "appeal_statuses": grp["appeal_status"].apply(aggregate_pipe),
        "appeal_resolution_dates": grp["appeal_resolution_date"].apply(aggregate_pipe),
        "final_outcomes": grp["final_outcome"].apply(aggregate_pipe),
    }).reset_index()

    return agg


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Numeric -> 0
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].fillna(0)

    # For count/distinct columns, ensure int
    for col in out.columns:
        if col.endswith("_count") or col.endswith("_distinct"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # Everything else -> ""
    obj_cols = out.select_dtypes(include=["object"]).columns
    out[obj_cols] = out[obj_cols].fillna("")

    return out


def build_knowledge_base() -> pd.DataFrame:
    # Load raw tables
    claims = read_csv(RAW_CLAIMS_PATH)
    denials = read_csv(RAW_DENIALS_PATH)
    encounters = read_csv(RAW_ENCOUNTERS_PATH)
    diagnoses = read_csv(RAW_DIAGNOSES_PATH)
    procedures = read_csv(RAW_PROCEDURES_PATH)
    labs = read_csv(RAW_LABS_PATH)
    meds = read_csv(RAW_MEDS_PATH)
    patients = read_csv(RAW_PATIENTS_PATH)
    providers = read_csv(RAW_PROVIDERS_PATH)

    # Deduplicate keys for safer joins
    claims = deduplicate(claims, "claim_id")
    encounters = deduplicate(encounters, "encounter_id")
    patients = deduplicate(patients, "patient_id")
    providers = deduplicate(providers, "provider_id")

    # Filter to denied only
    claim_status_norm = norm_for_filter(claims.get("claim_status", pd.Series(dtype=str)))
    claims_denied = claims[claim_status_norm == "denied"].copy()

    print(f"[info] claims total: {len(claims)}, denied: {len(claims_denied)}")

    # Bring in patient insurance_type just for filtering self-pay
    claims_denied = claims_denied.merge(
        patients[["patient_id", "insurance_type"]],
        on="patient_id",
        how="left",
        suffixes=("", "_patfilter"),
    )

    # Self-pay exclusion mask (values preserved later; this is filter-only)
    mask_self_pay = (
        is_self_pay_series(claims_denied.get("payment_method")) |
        is_self_pay_series(claims_denied.get("insurance_provider")) |
        is_self_pay_series(claims_denied.get("insurance_type"))
    )
    n_self_pay = int(mask_self_pay.sum())
    if n_self_pay:
        print(f"[info] excluding {n_self_pay} denied claims due to self-pay filter")
    claims_denied = claims_denied.loc[~mask_self_pay].copy()

    # Drop temporary filter column if it collided (keep the original names otherwise)
    if "insurance_type" in claims.columns:
        # The column exists in claims, so merge didn't introduce a new column name.
        pass
    else:
        # Remove the temporary column used only for filtering if it came from patients and not desired as dup
        # However, having insurance_type in the output can be useful, so we keep it.
        pass

    # Join core dimensions (retain columns from all tables; use suffixes to avoid collisions)
    joined = claims_denied.merge(encounters, on="encounter_id", how="left", suffixes=("", "_enc"))
    joined = joined.merge(patients, on="patient_id", how="left", suffixes=("", "_pat"))
    joined = joined.merge(providers, on="provider_id", how="left", suffixes=("", "_prov"))

    # Clinical per-encounter aggregates
    joined = add_counts_codes_desc(joined, diagnoses, "encounter_id", "diagnosis_code", "diag", desc_col="diagnosis_description")
    joined = add_counts_codes_desc(joined, procedures, "encounter_id", "procedure_code", "proc", desc_col="procedure_description")
    joined = add_counts_codes_desc(joined, labs, "encounter_id", "test_code", "lab", desc_col="test_name")
    # Medications: names only (no codes in provided schema)
    if not meds.empty:
        grp = meds.groupby("encounter_id", dropna=False)
        med_count = grp.size().rename("med_count")
        med_distinct = grp["drug_name"].nunique(dropna=True).rename("med_distinct")
        med_names = grp["drug_name"].apply(aggregate_pipe).rename("med_names")
        joined = joined.merge(med_count, left_on="encounter_id", right_index=True, how="left")
        joined = joined.merge(med_distinct, left_on="encounter_id", right_index=True, how="left")
        joined = joined.merge(med_names, left_on="encounter_id", right_index=True, how="left")
    else:
        joined["med_count"] = 0
        joined["med_distinct"] = 0
        joined["med_names"] = ""

    # Denials per-claim aggregates (preserve original text)
    den_agg = aggregate_denials(denials)
    joined = joined.merge(den_agg, on="claim_id", how="left")

    # Final fill
    joined = fill_missing_values(joined)

    # Basic summary
    print(f"[info] final denied (post-filter) rows: {len(joined)}")
    if "denial_reason" in joined.columns:
        top = pd.Series(joined["denial_reason"]).value_counts(dropna=False).head(10)
        print("\nTop claim denial_reason (as-is):")
        print(top)

    return joined


def write_holdout_markdown(df: pd.DataFrame, path: Path, holdout_size: int, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        ensure_output_dir(path)
        path.write_text("# Denied Claims Holdout\n\n_No denied claims available._\n", encoding="utf-8")
        return df

    n = min(holdout_size, len(df))
    holdout = df.sample(n=n, random_state=seed).copy()

    lines = []
    lines.append("# Denied Claims Holdout\n")
    lines.append(f"_Random sample of {n} denied claims held out for testing._\n")
    for _, row in holdout.iterrows():
        claim_id = row.get("claim_id", "")
        reason = row.get("denial_reason", "")
        payer = row.get("insurance_provider", "")
        first_name = row.get("first_name", "")
        last_name = row.get("last_name", "")
        header = f"### Claim {claim_id} — Expected: Denied — Reason: {reason} — Payer: {payer} — Patient: {first_name} {last_name}"
        lines.append(header)
        # Full flattened row as JSON
        as_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        lines.append("```json")
        lines.append(json.dumps(as_dict, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")  # blank line

    content = "\n".join(lines) + "\n"

    ensure_output_dir(path)
    path.write_text(content, encoding="utf-8")

    return holdout


def main():
    parser = argparse.ArgumentParser(description="Prepare a flattened knowledge base of denied claims only (excluding self-pay).")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV_PATH, help="Output CSV path for the knowledge base.")
    parser.add_argument("--holdout-md", type=Path, default=DEFAULT_HOLDOUT_MD_PATH, help="Output Markdown path for the holdout set.")
    parser.add_argument("--holdout-size", type=int, default=50, help="Number of denied claims to hold out for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout sampling.")
    args = parser.parse_args()

    # Build full denied claims KB
    kb = build_knowledge_base()

    # Create holdout MD and exclude from KB CSV
    holdout = write_holdout_markdown(kb, args.holdout_md, args.holdout_size, seed=args.seed)
    holdout_ids = set(holdout["claim_id"].tolist()) if not holdout.empty else set()
    kb_remainder = kb[~kb["claim_id"].isin(holdout_ids)].copy()

    # Save CSV
    ensure_output_dir(args.output)
    kb_remainder.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[done] wrote knowledge base CSV (excluding {len(holdout_ids)} holdout claims) -> {args.output}")
    print(f"[done] wrote holdout markdown -> {args.holdout_md}")


if __name__ == "__main__":
    main()