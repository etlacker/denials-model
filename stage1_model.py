from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

from data_prep import (
    bucket_reasons,
    group_shuffle_split,
)
from prepare_training_data import load_prepared_data


def _to_str(X):
    return X.astype(str)


def build_transformer(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(_to_str, validate=False)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ]
    )
    return preprocessor


def _make_model(df):
    pre = build_transformer(df)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", model)])


def _choose_threshold(base_clf, X_train, y_train, groups):
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=123)
    train_idx, val_idx = next(splitter.split(X_train, y_train, groups=groups))
    clf = clone(base_clf)
    clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    proba_val = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train.iloc[val_idx], proba_val)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = f1_scores.argmax()
    return float(thresholds[best_idx])


def train_stage1(X_train, y_train, X_eval, y_eval, groups_train):
    base_clf = _make_model(X_train)
    best_threshold = _choose_threshold(base_clf, X_train, y_train, groups_train)

    clf = clone(base_clf)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_eval)[:, 1]
    preds_default = (proba >= 0.5).astype(int)
    preds_tuned = (proba >= best_threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_eval, proba)

    def precision_at_recall(target):
        if len(recall) == 0:
            return 0.0
        idx = (abs(recall - target)).argmin()
        return precision[idx]

    def confusion(y_true, preds):
        tp = int(((preds == 1) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        return tp, tn, fp, fn

    tp_def, tn_def, fp_def, fn_def = confusion(y_eval, preds_default)
    tp_tuned, tn_tuned, fp_tuned, fn_tuned = confusion(y_eval, preds_tuned)

    metrics = {
        "ap": average_precision_score(y_eval, proba),
        "f1_default": f1_score(y_eval, preds_default),
        "accuracy_default": accuracy_score(y_eval, preds_default),
        "precision_at_50": precision[np.argmin(np.abs(recall - 0.5))],
        "threshold_tuned": best_threshold,
        "f1_tuned": f1_score(y_eval, preds_tuned),
        "accuracy_tuned": accuracy_score(y_eval, preds_tuned),
        "precision_at_recall_20": float(precision_at_recall(0.2)),
        "precision_at_recall_40": float(precision_at_recall(0.4)),
        "precision_at_recall_60": float(precision_at_recall(0.6)),
        "count_eval": int(len(y_eval)),
        "tp_default": tp_def,
        "tn_default": tn_def,
        "fp_default": fp_def,
        "fn_default": fn_def,
        "tp_tuned": tp_tuned,
        "tn_tuned": tn_tuned,
        "fp_tuned": fp_tuned,
        "fn_tuned": fn_tuned,
    }
    return clf, metrics


DEFAULT_ARTIFACTS_DIR = Path("artifacts")


def run_stage1(top_reasons: int, balance_eval: bool, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    prepared = load_prepared_data(artifacts_dir)
    feats = prepared["features"]
    y_denied = prepared["y_denied"]
    y_reason = prepared["y_reason"]
    patient_ids = prepared["patient_ids"]

    train_idx, eval_idx = group_shuffle_split(
        feats.index, patient_ids.loc[feats.index], test_size=0.2, random_state=42
    )
    train_patients = set(patient_ids.loc[train_idx])
    eval_patients = set(patient_ids.loc[eval_idx])
    overlap = train_patients & eval_patients
    if overlap:
        raise ValueError(f"Group split failed; overlap found for {len(overlap)} patient_ids")

    X_train, X_eval = feats.loc[train_idx], feats.loc[eval_idx]
    y_train, y_eval = y_denied.loc[train_idx], y_denied.loc[eval_idx]

    print(
        "Stage 1 label counts - train:",
        y_train.value_counts().to_dict(),
        "eval:",
        y_eval.value_counts().to_dict(),
    )

    if balance_eval:
        pos_idx = y_eval[y_eval == 1].index
        neg_idx = y_eval[y_eval == 0].index
        n = min(len(pos_idx), len(neg_idx))
        if n > 0:
            pos_sample = pos_idx.to_series().sample(n, random_state=42)
            neg_sample = neg_idx.to_series().sample(n, random_state=42)
            balanced_idx = pd.Index(pos_sample.tolist() + neg_sample.tolist())
            X_eval = X_eval.loc[balanced_idx]
            y_eval = y_eval.loc[balanced_idx]
            eval_idx = balanced_idx

    eval_dir = artifacts_dir / "eval"
    eval_dir.mkdir(exist_ok=True, parents=True)
    eval_df = X_eval.copy()
    eval_df["label_denied"] = y_eval
    eval_df["label_reason_raw"] = y_reason.loc[eval_idx]
    eval_df["label_reason_bucketed"] = bucket_reasons(
        y_reason.loc[eval_idx], top_reasons
    )
    eval_path = eval_dir / "eval_set_stage1.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved holdout evaluation set to {eval_path} (not used for training)")

    stage1_model, stage1_metrics = train_stage1(X_train, y_train, X_eval, y_eval, patient_ids.loc[train_idx])
    print("\nStage 1 (Denied vs Not Denied) - evaluated on holdout:")
    for k, v in stage1_metrics.items():
        print(f"  {k}: {v}")

    import joblib

    joblib.dump(stage1_model, artifacts_dir / "stage1_denial.joblib")
    threshold_path = artifacts_dir / "stage1_threshold.txt"
    threshold_path.write_text(str(stage1_metrics.get("threshold_tuned", 0.5)))
    print(f"\nSaved model to {artifacts_dir / 'stage1_denial.joblib'} and threshold to {threshold_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 denial model")
    parser.add_argument(
        "--top-reasons",
        type=int,
        default=8,
        help="Top-N denial reasons to keep; others -> Other (for eval bucketing)",
    )
    parser.add_argument(
        "--balance-eval",
        action="store_true",
        help="Balance eval set to 50/50 denied vs not-denied for inspection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage1(args.top_reasons, args.balance_eval)

