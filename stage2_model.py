from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data_prep import bucket_reasons, group_shuffle_split
from prepare_training_data import load_prepared_data


def _to_str(X):
    return X.astype(str)


def build_transformer(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    cat_pipeline = Pipeline(
        [
            ("to_str", FunctionTransformer(_to_str, validate=False)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ]
    )
    return preprocessor


def train_stage2(
    X: pd.DataFrame,
    y_reason: pd.Series,
    y_denied: pd.Series,
    patient_ids: pd.Series,
    top_n: int,
):
    mask = (y_denied == 1) & y_reason.notna() & (y_reason != "")
    X2 = X[mask].copy()
    y2 = bucket_reasons(y_reason[mask], top_n)
    groups = patient_ids.loc[X2.index]

    pre = build_transformer(X2)
    model = LogisticRegression(max_iter=400, class_weight="balanced")
    clf = Pipeline([("pre", pre), ("clf", model)])

    train_idx, test_idx = group_shuffle_split(
        X2.index, groups=groups, test_size=0.2, random_state=42
    )

    clf.fit(X2.loc[train_idx], y2.loc[train_idx])
    preds = clf.predict(X2.loc[test_idx])
    metrics = {
        "macro_f1": f1_score(y2.loc[test_idx], preds, average="macro"),
        "accuracy": accuracy_score(y2.loc[test_idx], preds),
        "report": classification_report(y2.loc[test_idx], preds, output_dict=False),
        "label_counts": y2.value_counts().to_dict(),
    }
    return clf, metrics


DEFAULT_DATA_DIR = "data"


def run_stage2(top_codes: int, top_reasons: int, data_dir: str = DEFAULT_DATA_DIR):
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    prepared = load_or_prepare(data_dir, top_codes, artifacts_dir)
    feats = prepared["features"]
    y_denied = prepared["y_denied"]
    y_reason = prepared["y_reason"]
    patient_ids = prepared["patient_ids"]

    stage2_model, stage2_metrics = train_stage2(
        feats, y_reason, y_denied, patient_ids, top_reasons
    )

    print("\nStage 2a (Denial Reason):")
    for k, v in stage2_metrics.items():
        if k == "report":
            print("\nClassification report:\n", v)
        else:
            print(f"  {k}: {v}")

    import joblib

    joblib.dump(stage2_model, artifacts_dir / "stage2_reason.joblib")
    print(f"\nSaved model to {artifacts_dir / 'stage2_reason.joblib'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 denial reason model")
    parser.add_argument(
        "--top-reasons",
        type=int,
        default=8,
        help="Top-N denial reasons to keep; others -> Other (reduced for class balance)",
    )
    return parser.parse_args()


DEFAULT_ARTIFACTS_DIR = Path("artifacts")


def run_stage2(top_reasons: int, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    prepared = load_prepared_data(artifacts_dir)
    feats = prepared["features"]
    y_denied = prepared["y_denied"]
    y_reason = prepared["y_reason"]
    patient_ids = prepared["patient_ids"]

    stage2_model, stage2_metrics = train_stage2(
        feats, y_reason, y_denied, patient_ids, top_reasons
    )

    print("\nStage 2a (Denial Reason):")
    for k, v in stage2_metrics.items():
        if k == "report":
            print("\nClassification report:\n", v)
        else:
            print(f"  {k}: {v}")

    import joblib

    joblib.dump(stage2_model, artifacts_dir / "stage2_reason.joblib")
    print(f"\nSaved model to {artifacts_dir / 'stage2_reason.joblib'}")


if __name__ == "__main__":
    args = parse_args()
    run_stage2(args.top_reasons)

