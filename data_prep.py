from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DataBundle:
    claims: pd.DataFrame
    denials: pd.DataFrame
    encounters: pd.DataFrame
    diagnoses: pd.DataFrame
    procedures: pd.DataFrame
    labs: pd.DataFrame
    meds: pd.DataFrame
    patients: pd.DataFrame
    providers: pd.DataFrame


def load_data(data_dir: str) -> DataBundle:
    claims = pd.read_csv(f"{data_dir}/claims_and_billing.csv")
    denials = pd.read_csv(f"{data_dir}/denials.csv")
    encounters = pd.read_csv(f"{data_dir}/encounters.csv")
    diagnoses = pd.read_csv(f"{data_dir}/diagnoses.csv")
    procedures = pd.read_csv(f"{data_dir}/procedures.csv")
    labs = pd.read_csv(f"{data_dir}/lab_tests.csv")
    meds = pd.read_csv(f"{data_dir}/medications.csv")
    patients = pd.read_csv(f"{data_dir}/patients.csv")
    providers = pd.read_csv(f"{data_dir}/providers.csv")

    claims["claim_billing_date"] = pd.to_datetime(
        claims["claim_billing_date"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    claims["paid_amount"] = pd.to_numeric(claims["paid_amount"], errors="coerce")
    claims["billed_amount"] = pd.to_numeric(claims["billed_amount"], errors="coerce")

    encounters["visit_date"] = pd.to_datetime(
        encounters["visit_date"], format="%d-%m-%Y", errors="coerce"
    )

    return DataBundle(
        claims=claims,
        denials=denials,
        encounters=encounters,
        diagnoses=diagnoses,
        procedures=procedures,
        labs=labs,
        meds=meds,
        patients=patients,
        providers=providers,
    )


def _safe_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def compute_top_codes(
    df: pd.DataFrame,
    group_key: str,
    code_col: str,
    top_n: int,
    allowed_groups: pd.Index | pd.Series | set | None = None,
) -> list:
    subset = df
    if allowed_groups is not None:
        subset = df[df[group_key].isin(allowed_groups)]
    return subset[code_col].value_counts().head(top_n).index.tolist()


def aggregate_codes(
    df: pd.DataFrame,
    group_key: str,
    code_col: str,
    top_n: int,
    prefix: str,
    top_codes: list | None = None,
) -> pd.DataFrame:
    counts = df.groupby(group_key)[code_col].agg(
        **{f"{prefix}_total_codes": "count", f"{prefix}_distinct_codes": lambda x: x.nunique()}
    )
    top_codes = top_codes or df[code_col].value_counts().head(top_n).index.tolist()
    flags = (
        df[df[code_col].isin(top_codes)]
        .assign(flag=1)
        .pivot_table(
            index=group_key,
            columns=code_col,
            values="flag",
            aggfunc="max",
            fill_value=0,
        )
    )
    flags.columns = [f"{prefix}_{code_col}_flag_{c}" for c in flags.columns]
    return counts.join(flags, how="left").reset_index()


def build_claim_features(
    data: DataBundle,
    top_n_codes: int,
    top_code_maps: dict | None = None,
    claims_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return claim-grain feature frame, stage1 label, stage2 label, claim dates, patient ids.

    top_code_maps optionally supplies pre-computed top code lists keyed by
    {"diag", "proc", "lab", "med"} to avoid data leakage (e.g., derive from
    training split only). If absent, top codes are computed on the full data.
    """
    claims = claims_df.copy() if claims_df is not None else data.claims.copy()
    claims = claims[claims["claim_id"].notna()].copy()
    claims = claims.reset_index(drop=True).reset_index().rename(columns={"index": "row_id"})

    y_denied = claims.set_index("row_id")["claim_status"].eq("Denied").astype(int)

    denial_reason_map = (
        data.denials.drop_duplicates("claim_id")
        .set_index("claim_id")["denial_reason_description"]
    )
    claims["denial_reason_filled"] = claims["denial_reason"].fillna(
        claims["claim_id"].map(denial_reason_map)
    )
    y_reason = claims.set_index("row_id")["denial_reason_filled"]

    base_cols = [
        "row_id",
        "billing_id",
        "claim_id",
        "patient_id",
        "encounter_id",
        "insurance_provider",
        "payment_method",
        "billed_amount",
    ]
    feats = claims[base_cols].copy()
    feats["claim_month"] = claims["claim_billing_date"].dt.to_period("M").astype(str)

    patient_cols = ["patient_id", "age", "gender", "ethnicity", "insurance_type", "marital_status"]
    feats = feats.merge(data.patients[patient_cols], on="patient_id", how="left")

    enc_cols = [
        "encounter_id",
        "provider_id",
        "visit_type",
        "department",
        "reason_for_visit",
        "diagnosis_code",
        "admission_type",
    ]
    feats = feats.merge(data.encounters[enc_cols], on="encounter_id", how="left")

    provider_cols = ["provider_id", "specialty", "inhouse", "location", "years_experience"]
    feats = feats.merge(data.providers[provider_cols], on="provider_id", how="left")

    diag = data.diagnoses.copy()
    diag["chronic_flag"] = _safe_bool(diag["chronic_flag"])
    diag_agg = diag.groupby("encounter_id").agg(
        diag_count=("diagnosis_id", "count"),
        chronic_count=("chronic_flag", "sum"),
    )
    diag_top_codes = None
    if top_code_maps:
        diag_top_codes = top_code_maps.get("diag")
    diag_top = aggregate_codes(
        diag, "encounter_id", "diagnosis_code", top_n_codes, prefix="diag", top_codes=diag_top_codes
    )
    diag_agg = diag_agg.reset_index().merge(diag_top, on="encounter_id", how="left")
    feats = feats.merge(diag_agg, on="encounter_id", how="left")

    proc_top_codes = top_code_maps.get("proc") if top_code_maps else None
    proc_agg = aggregate_codes(
        data.procedures, "encounter_id", "procedure_code", top_n_codes, prefix="proc", top_codes=proc_top_codes
    )
    feats = feats.merge(proc_agg, on="encounter_id", how="left")

    lab_top_codes = top_code_maps.get("lab") if top_code_maps else None
    lab_agg = aggregate_codes(
        data.labs, "encounter_id", "test_code", top_n_codes, prefix="lab", top_codes=lab_top_codes
    )
    feats = feats.merge(lab_agg, on="encounter_id", how="left")

    med_top_codes = top_code_maps.get("med") if top_code_maps else None
    med_agg = aggregate_codes(
        data.meds, "encounter_id", "drug_name", top_n_codes, prefix="med", top_codes=med_top_codes
    )
    feats = feats.merge(med_agg, on="encounter_id", how="left")

    # Preserve IDs for grouping before dropping them from features
    patient_ids = feats["patient_id"].copy()

    # Fill numeric aggregates only; leave categoricals for imputers in the pipeline
    numeric_fill_cols = [
        c
        for c in feats.columns
        if c.startswith(("diag_", "proc_", "lab_", "med_"))
        or c.endswith("_count")
        or c.endswith("_total_codes")
        or c.endswith("_distinct_codes")
    ]
    feats[numeric_fill_cols] = feats[numeric_fill_cols].fillna(0)

    feats = feats.set_index("row_id")
    feats = feats.drop(columns=["billing_id", "claim_id", "encounter_id", "patient_id"], errors="ignore")
    claim_dates = claims.set_index("row_id")["claim_billing_date"]
    return feats, y_denied, y_reason, claim_dates, patient_ids


def bucket_reasons(y_reason: pd.Series, top_n: int) -> pd.Series:
    counts = y_reason.value_counts()
    top_reasons = counts.head(top_n).index
    return y_reason.where(y_reason.isin(top_reasons), other="Other")


def time_or_stratified_split(
    y: pd.Series, date_series: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Index, pd.Index]:
    if pd.api.types.is_datetime64_any_dtype(date_series):
        order = date_series.sort_values().index
        split_idx = int(len(order) * (1 - test_size))
        train_idx = order[:split_idx]
        test_idx = order[split_idx:]
        return train_idx, test_idx
    return train_test_split(y.index, test_size=test_size, stratify=y, random_state=42)


def group_shuffle_split(
    index: pd.Index, groups: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.Index, pd.Index]:
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(index, groups=groups))
    return index[train_idx], index[test_idx]

