from __future__ import annotations

from pathlib import Path
import joblib

from data_prep import build_claim_features, load_data


DEFAULT_ARTIFACTS = Path("artifacts")
PREPARED_PATH = DEFAULT_ARTIFACTS / "prepared_data.joblib"


def prepare_training_data(data_dir: str, top_codes: int, artifacts_dir: Path = DEFAULT_ARTIFACTS):
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    data = load_data(data_dir)
    feats, y_denied, y_reason, claim_dates, patient_ids = build_claim_features(data, top_codes)
    payload = {
        "features": feats,
        "y_denied": y_denied,
        "y_reason": y_reason,
        "claim_dates": claim_dates,
        "patient_ids": patient_ids,
        "top_codes": top_codes,
    }
    joblib.dump(payload, artifacts_dir / "prepared_data.joblib")
    return payload


def load_prepared_data(artifacts_dir: Path = DEFAULT_ARTIFACTS):
    path = artifacts_dir / "prepared_data.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Prepared data not found at {path}. Run prepare_training_data first.")
    return joblib.load(path)


def load_or_prepare(data_dir: str, top_codes: int, artifacts_dir: Path = DEFAULT_ARTIFACTS):
    path = artifacts_dir / "prepared_data.joblib"
    if path.exists():
        return joblib.load(path)
    return prepare_training_data(data_dir, top_codes, artifacts_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data artifacts.")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSVs")
    parser.add_argument("--top-codes", type=int, default=15, help="Top-N codes per table")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts output directory")
    args = parser.parse_args()

    payload = prepare_training_data(args.data_dir, args.top_codes, Path(args.artifacts_dir))
    print(
        f"Prepared data saved to {Path(args.artifacts_dir) / 'prepared_data.joblib'} "
        f"with features shape {payload['features'].shape}"
    )

