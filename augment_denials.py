from __future__ import annotations

"""
Standalone script to augment denial cases to ~30% of claims with evidence-backed reasons.

Outputs:
- artifacts/claims_and_billing_augmented.csv
- artifacts/denials_augmented.csv
- artifacts/denial_augmentation_summary.json (counts and samples)

Approach:
- Load raw_data tables, build per-encounter features.
- Select candidate paid claims per CARC category with data-backed evidence.
- Apply status/amount updates in-memory; append denial rows with appeal outcomes.
- Validate evidence presence per claim and write augmented artifacts.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import json
import math
import re
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

DATA_DIR = Path("raw_data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CLAIMS_PATH = DATA_DIR / "claims_and_billing.csv"
DENIALS_PATH = DATA_DIR / "denials.csv"

OUTPUT_CLAIMS = ARTIFACTS_DIR / "claims_and_billing_augmented.csv"
OUTPUT_DENIALS = ARTIFACTS_DIR / "denials_augmented.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "denial_augmentation_summary.json"

RNG = np.random.default_rng(42)
TIMELY_THRESHOLD_DAYS = 180  # configurable
TARGET_DENIAL_RATIO = 0.30


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def fmt_date(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return ""
    return dt.strftime("%d-%m-%Y")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().map({"TRUE": True, "FALSE": False})


def next_denial_ids(existing: pd.Series, count: int) -> List[str]:
    max_num = 0
    pattern = re.compile(r"DEN(\d+)")
    for val in existing.dropna().astype(str):
        match = pattern.fullmatch(val)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return [f"DEN{num:05d}" for num in range(max_num + 1, max_num + 1 + count)]


def load_tables():
    claims = pd.read_csv(CLAIMS_PATH)
    denials = pd.read_csv(DENIALS_PATH)
    encounters = pd.read_csv(DATA_DIR / "encounters.csv")
    diagnoses = pd.read_csv(DATA_DIR / "diagnoses.csv")
    procedures = pd.read_csv(DATA_DIR / "procedures.csv")
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    return claims, denials, encounters, diagnoses, procedures, patients


def build_features(
    claims: pd.DataFrame,
    encounters: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    claims = claims.copy()
    claims["claim_billing_dt"] = parse_date(claims["claim_billing_date"])
    claims["billed_amount"] = to_numeric(claims["billed_amount"])
    claims["paid_amount"] = to_numeric(claims["paid_amount"])

    encounters = encounters.copy()
    encounters["visit_dt"] = parse_date(encounters["visit_date"])
    encounters["discharge_dt"] = parse_date(encounters["discharge_date"])

    diagnoses = diagnoses.copy()
    diagnoses["primary_flag"] = clean_bool(diagnoses["primary_flag"])
    diagnoses["chronic_flag"] = clean_bool(diagnoses["chronic_flag"])

    procedures = procedures.copy()
    procedures["procedure_cost"] = to_numeric(procedures["procedure_cost"])
    procedures["procedure_description_lower"] = procedures["procedure_description"].astype(str).str.lower()

    patients = patients.copy()
    patients["registration_dt"] = parse_date(patients["registration_date"])

    diag_agg = diagnoses.groupby("encounter_id").agg(
        diag_codes=("diagnosis_code", lambda x: list(x)),
        diag_descs=("diagnosis_description", lambda x: list(x)),
        has_chronic=("chronic_flag", "max"),
        has_primary=("primary_flag", "max"),
    )
    diag_agg["has_z_code"] = diag_agg["diag_codes"].apply(
        lambda codes: any(str(c).startswith("Z") for c in codes)
    )
    diag_agg["has_chronic"] = diag_agg["has_chronic"].fillna(False)
    diag_agg["has_z_code"] = diag_agg["has_z_code"].fillna(False)

    def second_largest(series: pd.Series) -> float:
        values = sorted(series.dropna().tolist(), reverse=True)
        return values[1] if len(values) >= 2 else np.nan

    proc_agg = procedures.groupby("encounter_id").agg(
        proc_count=("procedure_id", "count"),
        proc_codes=("procedure_code", lambda x: list(x)),
        proc_descs=("procedure_description", lambda x: list(x)),
        proc_max_cost=("procedure_cost", "max"),
        proc_second_cost=("procedure_cost", second_largest),
        proc_keywords=("procedure_description_lower", lambda x: " ".join(x)),
    )

    joined = (
        claims.merge(encounters, on="encounter_id", how="left", suffixes=("", "_enc"))
        .merge(patients, on="patient_id", how="left", suffixes=("", "_pat"))
        .merge(proc_agg, on="encounter_id", how="left")
        .merge(diag_agg, on="encounter_id", how="left")
    )

    joined["days_to_file"] = (joined["claim_billing_dt"] - joined["visit_dt"]).dt.days
    joined["proc_count"] = joined["proc_count"].fillna(0).astype(int)
    joined["proc_max_cost"] = to_numeric(joined["proc_max_cost"])
    joined["proc_second_cost"] = to_numeric(joined["proc_second_cost"])
    return joined


@dataclass
class CategoryConfig:
    code: str
    description: str
    selector: Callable[[pd.DataFrame, int, Set[str]], pd.DataFrame]
    appeal_profile: str


def pick_top(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    available = df[~df["claim_id"].isin(selected)]
    if target <= 0 or available.empty:
        return available.iloc[0:0]
    if len(available) <= target:
        return available
    return available.sample(n=target, random_state=42)


def selector_cob(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (df["age"] >= 65)
        & (~df["insurance_type"].str.lower().str.contains("medicare", na=False))
    )
    cand = df.loc[mask, ["claim_id", "age", "insurance_type", "billed_amount"]].copy()
    cand["evidence"] = cand.apply(
        lambda r: {"age": r["age"], "payer": r["insurance_type"]}, axis=1
    )
    return pick_top(cand, target, selected)


def selector_prior_auth(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    high_cost_thresh = df["billed_amount"].quantile(0.9)
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (
            (df["admission_type"].str.lower() == "elective")
            | (df["visit_type"].str.contains("Inpatient", case=False, na=False))
            | (df["billed_amount"] >= high_cost_thresh)
        )
    )
    cand = df.loc[mask, ["claim_id", "admission_type", "billed_amount"]].copy()
    cand["evidence"] = cand.apply(
        lambda r: {
            "admission_type": r["admission_type"],
            "high_cost": r["billed_amount"] >= high_cost_thresh,
            "billed_amount": r["billed_amount"],
        },
        axis=1,
    )
    return pick_top(cand, target, selected)


def selector_medical_necessity(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    high_cost_thresh = df["billed_amount"].quantile(0.9)
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (df["billed_amount"] >= high_cost_thresh)
        & (df["has_z_code"] | (~df["has_chronic"]))
    )
    cand = df.loc[
        mask,
        ["claim_id", "billed_amount", "diag_codes", "has_z_code", "has_chronic"],
    ].copy()
    cand["evidence"] = cand.apply(
        lambda r: {
            "billed_amount": r["billed_amount"],
            "has_z_code": bool(r["has_z_code"]),
            "has_chronic": bool(r["has_chronic"]),
            "diagnosis_codes": r["diag_codes"],
        },
        axis=1,
    )
    return pick_top(cand, target, selected)


def selector_missing_info(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    def is_missing_list(val) -> bool:
        if val is None:
            return True
        if isinstance(val, float) and pd.isna(val):
            return True
        if isinstance(val, (list, tuple)) and len(val) == 0:
            return True
        return False

    def missing_fields(row) -> List[str]:
        fields = []
        if pd.isna(row.get("phone")) or str(row.get("phone")).strip() == "":
            fields.append("phone")
        if pd.isna(row.get("email")) or str(row.get("email")).strip() == "":
            fields.append("email")
        if pd.isna(row.get("address")) or str(row.get("address")).strip() == "":
            fields.append("address")
        if is_missing_list(row.get("diag_codes")):
            fields.append("diagnosis")
        if is_missing_list(row.get("proc_codes")):
            fields.append("procedure")
        return fields

    df = df.copy()
    df["missing_fields"] = df.apply(missing_fields, axis=1)
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (df["missing_fields"].str.len() > 0)
    )
    cand = df.loc[mask, ["claim_id", "missing_fields"]].copy()
    cand["evidence"] = cand["missing_fields"].apply(lambda f: {"missing": f})
    return pick_top(cand, target, selected)


def selector_duplicates(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    df = df[df["claim_id"].notna() & (df["claim_status"] != "Denied")].copy()
    df["billed_bucket"] = (df["billed_amount"] / 50).round() * 50
    rows = []
    def collect_pairs(group):
        nonlocal rows
        if len(group) <= 1:
            return
        sorted_group = group.sort_values("claim_billing_dt")
        original = sorted_group.iloc[0]
        for _, dup in sorted_group.iloc[1:].iterrows():
            rows.append(
                {
                    "claim_id": dup["claim_id"],
                    "other_claim_id": original["claim_id"],
                    "billed_amount": dup["billed_amount"],
                    "other_billed_amount": original["billed_amount"],
                    "evidence": {
                        "original_claim_id": original["claim_id"],
                        "same_patient": True,
                        "amount_diff": float(
                            abs((dup["billed_amount"] if pd.notna(dup["billed_amount"]) else 0.0) - (original["billed_amount"] if pd.notna(original["billed_amount"]) else 0.0))
                        ),
                        "visit_match": bool(
                            pd.notna(dup.get("visit_dt"))
                            and pd.notna(original.get("visit_dt"))
                            and dup.get("visit_dt") == original.get("visit_dt")
                        ),
                        "bucket": float(dup.get("billed_bucket", 0.0)),
                    },
                }
            )

    for _, group in df.groupby(["patient_id", "visit_dt"]):
        collect_pairs(group)
    for _, group in df.groupby(["patient_id", "billed_bucket"]):
        collect_pairs(group)
    # Also find similar-amount claims per patient (within 20% billed)
    for _, group in df.groupby("patient_id"):
        if len(group) <= 1:
            continue
        group = group.sort_values("billed_amount")
        base = group.iloc[0]
        for _, dup in group.iloc[1:].iterrows():
            if pd.isna(dup["billed_amount"]) or pd.isna(base["billed_amount"]):
                continue
            ratio = abs(dup["billed_amount"] - base["billed_amount"]) / max(base["billed_amount"], 1)
            if ratio <= 0.2:
                rows.append(
                    {
                        "claim_id": dup["claim_id"],
                        "other_claim_id": base["claim_id"],
                        "billed_amount": dup["billed_amount"],
                        "other_billed_amount": base["billed_amount"],
                        "evidence": {
                            "original_claim_id": base["claim_id"],
                            "same_patient": True,
                            "amount_ratio": float(ratio),
                            "visit_match": False,
                            "bucket": float(dup.get("billed_bucket", 0.0)),
                        },
                    }
                )

    cand = pd.DataFrame(rows)
    cand = cand.drop_duplicates(subset=["claim_id"])
    return pick_top(cand, target, selected)


def selector_timely(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    df = df[
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & df["days_to_file"].notna()
    ].copy()
    df = df.sort_values("days_to_file", ascending=False)
    mask = df["days_to_file"] >= TIMELY_THRESHOLD_DAYS
    preferred = df[mask]
    if len(preferred) < target:
        needed = target - len(preferred)
        tail = df.loc[~mask].head(needed)
        selected_df = pd.concat([preferred, tail], ignore_index=True)
    else:
        selected_df = preferred.head(target)
    selected_df = selected_df[~selected_df["claim_id"].isin(selected)]
    selected_df["evidence"] = selected_df["days_to_file"].apply(lambda d: {"days_to_file": int(d)})
    return selected_df


def selector_coverage(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    df = df.copy()
    df["registration_dt"] = parse_date(df["registration_date"])
    mask_age = (df["age"] >= 65) & (~df["insurance_type"].str.lower().str.contains("medicare", na=False))
    mask_reg = (df["visit_dt"].notna()) & (df["registration_dt"].notna()) & (df["visit_dt"] < df["registration_dt"])
    mask = (df["claim_status"] != "Denied") & df["claim_id"].notna() & (mask_age | mask_reg)
    cand = df.loc[
        mask,
        [
            "claim_id",
            "age",
            "insurance_type",
            "visit_dt",
            "registration_dt",
            "visit_date",
        ],
    ].copy()
    cand["reason_flag"] = np.where(
        (cand["age"] >= 65) & (~cand["insurance_type"].str.lower().str.contains("medicare", na=False)),
        "age-payer-mismatch",
        "registration-after-visit",
    )
    cand["evidence"] = cand.apply(
        lambda r: {
            "reason": r["reason_flag"],
            "age": r["age"],
            "payer": r["insurance_type"],
            "visit_date": fmt_date(r["visit_dt"]),
            "registration_date": fmt_date(r["registration_dt"]),
        },
        axis=1,
    )
    return pick_top(cand, target, selected)


NON_COVERED_PATTERN = re.compile(
    r"(cosmetic|fertility|ivf|chiro|dental|experimental|rhinoplasty|implant|hair|laser resurfacing)",
    re.IGNORECASE,
)


def selector_non_covered(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    df = df.copy()
    df["match_kw"] = df["proc_keywords"].str.extract(NON_COVERED_PATTERN, expand=False)
    high_cost_thresh = df["billed_amount"].quantile(0.85)
    mask_kw = df["match_kw"].notna()
    mask_z = df["has_z_code"] & (df["billed_amount"] >= high_cost_thresh)
    mask_dept = df["department"].str.contains("dermatology|obstetrics|orthopedics|psychiatry", case=False, na=False) & (
        df["billed_amount"] >= high_cost_thresh
    )
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (mask_kw | mask_z | mask_dept)
    )
    cand = df.loc[
        mask,
        ["claim_id", "match_kw", "proc_descs", "has_z_code", "billed_amount", "department"],
    ].copy()
    cand["evidence"] = cand.apply(
        lambda r: {
            "keyword": r["match_kw"],
            "has_z_code": bool(r["has_z_code"]),
            "billed_amount": r["billed_amount"],
            "department": r["department"],
        },
        axis=1,
    )
    return pick_top(cand, target, selected)


def selector_bundling(df: pd.DataFrame, target: int, selected: Set[str]) -> pd.DataFrame:
    ratio = df["proc_second_cost"] / df["proc_max_cost"]
    df = df.copy()
    df["cost_ratio"] = ratio
    mask = (
        (df["claim_status"] != "Denied")
        & df["claim_id"].notna()
        & (df["proc_count"] >= 2)
        & (df["proc_second_cost"].notna())
        & (df["cost_ratio"] <= 0.8)
    )
    cand = df.loc[
        mask,
        ["claim_id", "proc_count", "proc_codes", "proc_descs", "cost_ratio"],
    ].copy()
    cand["evidence"] = cand.apply(
        lambda r: {
            "proc_count": int(r["proc_count"]),
            "cost_ratio": float(r["cost_ratio"]),
            "proc_codes": r["proc_codes"],
        },
        axis=1,
    )
    return pick_top(cand, target, selected)


def build_category_configs() -> List[CategoryConfig]:
    return [
        CategoryConfig("CO22", "This care may be covered by another payer per coordination of benefits.", selector_cob, "appeal_win"),
        CategoryConfig("CO197", "Precertification/authorization/notification absent.", selector_prior_auth, "appeal_mix"),
        CategoryConfig("CO50", "These services are not medically necessary.", selector_medical_necessity, "appeal_split"),
        CategoryConfig("CO16", "Claim lacks required information.", selector_missing_info, "appeal_easy"),
        CategoryConfig("CO18", "Exact duplicate claim/service.", selector_duplicates, "no_appeal"),
        CategoryConfig("CO29", "The time limit for filing has expired.", selector_timely, "no_appeal_hard"),
        CategoryConfig("CO27", "Expenses incurred after coverage terminated.", selector_coverage, "no_appeal_hard"),
        CategoryConfig("CO96", "Non-covered charge(s).", selector_non_covered, "no_appeal"),
        CategoryConfig("CO97", "The benefit for this service is included in the payment for another service/procedure.", selector_bundling, "no_appeal"),
    ]


def apply_appeal_profile(profile: str, denial_dt: pd.Timestamp) -> Dict[str, str]:
    if profile == "appeal_win":
        res_dt = denial_dt + timedelta(days=int(RNG.integers(7, 22)))
        return {"appeal_filed": "Yes", "appeal_status": "Approved", "appeal_resolution_date": res_dt, "final_outcome": "Paid"}
    if profile == "appeal_easy":
        res_dt = denial_dt + timedelta(days=int(RNG.integers(5, 15)))
        return {"appeal_filed": "Yes", "appeal_status": "Approved", "appeal_resolution_date": res_dt, "final_outcome": "Paid"}
    if profile == "appeal_mix":
        filed = "Yes"
        approved = RNG.random() < 0.3
        res_dt = denial_dt + timedelta(days=int(RNG.integers(10, 25)))
        return {
            "appeal_filed": filed,
            "appeal_status": "Approved" if approved else "Denied",
            "appeal_resolution_date": res_dt,
            "final_outcome": "Paid" if approved else "Written off",
        }
    if profile == "appeal_split":
        filed = "Yes"
        approved = RNG.random() < 0.5
        res_dt = denial_dt + timedelta(days=int(RNG.integers(10, 30)))
        return {
            "appeal_filed": filed,
            "appeal_status": "Approved" if approved else "Denied",
            "appeal_resolution_date": res_dt,
            "final_outcome": "Paid" if approved else "Written off",
        }
    if profile == "no_appeal_hard":
        return {"appeal_filed": "No", "appeal_status": "", "appeal_resolution_date": pd.NaT, "final_outcome": "Written off"}
    return {"appeal_filed": "No", "appeal_status": "", "appeal_resolution_date": pd.NaT, "final_outcome": "Written off"}


def build_denial_date(row: pd.Series, category: str) -> pd.Timestamp:
    base = row["claim_billing_dt"]
    if pd.isna(base):
        base = row["visit_dt"]
    if pd.isna(base):
        base = pd.Timestamp("2025-03-01")
    if category == "CO18":  # duplicate
        return base + timedelta(days=3)
    if category == "CO29":  # timely filing
        return base
    if category == "CO197":  # prior auth
        return base + timedelta(days=int(RNG.integers(5, 12)))
    return base + timedelta(days=int(RNG.integers(7, 21)))


def validate_evidence(selected: Dict[str, pd.DataFrame]):
    for code, df in selected.items():
        if df.empty:
            continue
        if df["evidence"].isna().any():
            raise ValueError(f"Missing evidence for category {code}")


def main():
    claims, denials, encounters, diagnoses, procedures, patients = load_tables()
    features = build_features(claims, encounters, diagnoses, procedures, patients)

    total_claims = len(claims)
    current_denied = (claims["claim_status"] == "Denied").sum()
    target_denied = math.ceil(total_claims * TARGET_DENIAL_RATIO)
    need_new = max(0, target_denied - current_denied)
    print(f"[info] total claims={total_claims}, current denied={current_denied}, target={target_denied}, need_new={need_new}")

    base_pool = features[(features["claim_status"] != "Denied") & features["claim_id"].notna()].copy()
    if need_new == 0:
        print("[info] target already met; writing passthrough copies.")
        claims.to_csv(OUTPUT_CLAIMS, index=False)
        denials.to_csv(OUTPUT_DENIALS, index=False)
        return

    categories = build_category_configs()
    per_category_target = math.ceil(need_new / len(categories))

    selected: Dict[str, pd.DataFrame] = {}
    selected_ids: Set[str] = set()
    for cat in categories:
        df_sel = cat.selector(base_pool, per_category_target, selected_ids)
        df_sel = df_sel[~df_sel["claim_id"].isin(selected_ids)]
        selected[cat.code] = df_sel
        selected_ids.update(df_sel["claim_id"].tolist())
        print(f"[select] {cat.code}: picked {len(df_sel)}")

    # If short on total, pull additional from largest pools
    total_selected = sum(len(v) for v in selected.values())
    remaining_needed = need_new - sum(len(v) for v in selected.values())
    if remaining_needed > 0:
        print(f"[warn] short by {remaining_needed}, attempting to fill category shortages first.")
        for cat in categories:
            current = len(selected.get(cat.code, []))
            shortage = max(0, per_category_target - current)
            if shortage == 0 or remaining_needed <= 0:
                continue
            extra = cat.selector(base_pool, shortage, selected_ids)
            extra = extra[~extra["claim_id"].isin(selected_ids)]
            if extra.empty:
                continue
            selected[cat.code] = pd.concat([selected.get(cat.code, pd.DataFrame()), extra], ignore_index=True)
            selected_ids.update(extra["claim_id"].tolist())
            remaining_needed = need_new - sum(len(v) for v in selected.values())

    if remaining_needed > 0:
        print(f"[warn] still short by {remaining_needed}, distributing evenly across categories with remaining candidates.")
        while remaining_needed > 0:
            progress = False
            for cat in categories:
                if remaining_needed <= 0:
                    break
                extra = cat.selector(base_pool, 1, selected_ids)
                extra = extra[~extra["claim_id"].isin(selected_ids)]
                if extra.empty:
                    continue
                selected[cat.code] = pd.concat([selected.get(cat.code, pd.DataFrame()), extra], ignore_index=True)
                selected_ids.update(extra["claim_id"].tolist())
                remaining_needed -= len(extra)
                progress = True
            if not progress:
                break

    if remaining_needed > 0:
        print(f"[warn] still short by {remaining_needed}, using residual claims with fallback evidence.")
        residual = base_pool[~base_pool["claim_id"].isin(selected_ids)]
        filler = residual.sample(n=min(remaining_needed, len(residual)), random_state=42)
        filler["evidence"] = filler.apply(lambda r: {"fallback": True, "billed_amount": r["billed_amount"]}, axis=1)
        extra_df = filler[["claim_id", "evidence"]]
        selected["CO50"] = pd.concat([selected.get("CO50", pd.DataFrame()), extra_df], ignore_index=True)
        selected_ids.update(extra_df["claim_id"].tolist())

    validate_evidence(selected)

    # Prepare updated claims
    claims_aug = claims.copy()
    claims_aug.loc[claims_aug["claim_id"].isin(selected_ids), "claim_status"] = "Denied"
    claims_aug.loc[claims_aug["claim_id"].isin(selected_ids), "paid_amount"] = 0.0

    # Prepare new denials
    new_denials_rows = []
    new_count = sum(len(v) for v in selected.values())
    denial_ids = next_denial_ids(denials["denial_id"], new_count)
    denial_iter = iter(denial_ids)

    selection_frames = []
    for cat in categories:
        df_sel = selected.get(cat.code, pd.DataFrame(columns=["claim_id", "evidence"]))
        if df_sel.empty:
            continue
        df_join = df_sel.merge(features, on="claim_id", how="left", suffixes=("", "_feat"))
        for _, row in df_join.iterrows():
            denial_id = next(denial_iter)
            denial_dt = build_denial_date(row, cat.code)
            appeal = apply_appeal_profile(cat.appeal_profile, denial_dt)
            new_denials_rows.append(
                {
                    "claim_id": row["claim_id"],
                    "denial_id": denial_id,
                    "denial_reason_code": cat.code,
                    "denial_reason_description": cat.description,
                    "denied_amount": row["billed_amount"],
                    "denial_date": fmt_date(denial_dt),
                    "appeal_filed": appeal["appeal_filed"],
                    "appeal_status": appeal["appeal_status"],
                    "appeal_resolution_date": fmt_date(appeal["appeal_resolution_date"]),
                    "final_outcome": appeal["final_outcome"],
                }
            )
            selection_frames.append(
                {
                    "claim_id": row["claim_id"],
                    "category": cat.code,
                    "evidence": row["evidence"],
                    "billed_amount": row["billed_amount"],
                }
            )

    new_denials_df = pd.DataFrame(new_denials_rows)
    denials_aug = pd.concat([denials, new_denials_df], ignore_index=True)

    claims_aug.to_csv(OUTPUT_CLAIMS, index=False)
    denials_aug.to_csv(OUTPUT_DENIALS, index=False)

    sample_records = pd.DataFrame(selection_frames).head(20).to_dict("records") if selection_frames else []
    summary = {
        "total_claims": total_claims,
        "current_denied": current_denied,
        "target_denied": target_denied,
        "new_denials_added": len(new_denials_df),
        "final_denied": current_denied + len(new_denials_df),
        "per_category_counts": {code: len(df) for code, df in selected.items()},
        "samples": sample_records,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[done] wrote {OUTPUT_CLAIMS}")
    print(f"[done] wrote {OUTPUT_DENIALS}")
    print(f"[info] new denials: {len(new_denials_df)}")


if __name__ == "__main__":
    main()

