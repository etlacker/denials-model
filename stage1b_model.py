"""
LightGBM baseline for denial prediction on the cleaned/enriched dataset.
Uses the same feature engineering as stage1a (log billed_amount, count buckets).

Usage:
  python stage1b_model.py                           # Uses augmented data if available
  python stage1b_model.py --data-dir artifacts/data_cleaning_raw  # Uses raw data
  python stage1b_model.py --both                    # Runs on both datasets
"""
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress sklearn feature name warnings (harmless, caused by pipeline transformations)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("lightgbm is required for this script; please install it.") from exc


ARTIFACTS_DIR = Path("artifacts")

label_col = "label_denied"
group_col = "patient_id"
COST_FP = 1.0
COST_FN = 5.0
SWEEP_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def build_feature_frames(df: pd.DataFrame):
    drop_cols = {label_col}
    if group_col in df.columns:
        drop_cols.add(group_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return df[feature_cols].copy(), cat_cols, num_cols


def add_engineered_features(df: pd.DataFrame, train_df: pd.DataFrame | None = None):
    df = df.copy()
    if "billed_amount" in df.columns:
        df["billed_amount_log"] = np.log1p(df["billed_amount"].astype(float).fillna(0))
    for col in ["proc_count", "diag_count", "lab_count", "med_count"]:
        if col in df.columns:
            qs = (train_df if train_df is not None else df)[col].quantile([0.25, 0.5, 0.75]).values
            edges = [-np.inf] + sorted(set(qs)) + [np.inf]
            df[f"{col}_bucket"] = pd.cut(
                df[col],
                bins=edges,
                labels=[f"bin_{i}" for i in range(len(edges) - 1)],
                duplicates="drop",
            ).astype(str)
    return df


def compute_metrics(y_true, proba, threshold: float):
    preds = (proba >= threshold).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_grid = (2 * precision[:-1] * recall[:-1]) / np.clip(
        precision[:-1] + recall[:-1], a_min=1e-12, a_max=None
    )
    best_idx = int(np.argmax(f1_grid)) if len(f1_grid) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else threshold
    return {
        "ap": average_precision_score(y_true, proba),
        "f1_at_threshold": f1_score(y_true, preds),
        "accuracy_at_threshold": accuracy_score(y_true, preds),
        "best_f1_threshold": best_threshold,
        "best_f1": float(f1_grid[best_idx]) if len(f1_grid) else float("nan"),
    }


def confusion_stats(y_true, proba, threshold: float):
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def sweep_costs(y_true, proba, thresholds, cost_fp: float, cost_fn: float):
    rows = []
    for t in thresholds:
        stats = confusion_stats(y_true, proba, threshold=t)
        fp, fn, tp, tn = stats["fp"], stats["fn"], stats["tp"], stats["tn"]
        cost = cost_fp * fp + cost_fn * fn
        total = fp + fn + tp + tn
        rows.append(
            {
                "threshold": round(t, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "f1": float(tp * 2 / (2 * tp + fp + fn)) if tp + fp + fn else 0.0,
                "cost": float(cost),
                "cost_per_example": float(cost / total) if total else 0.0,
            }
        )
    return rows


def run_training(data_dir: Path, output_dir: Path):
    """Run the full training pipeline for a given data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "claims_enriched_train.csv"
    eval_path = data_dir / "claims_enriched_eval.csv"
    model_path = output_dir / "claim_denial_model.joblib"
    metrics_path = output_dir / "claim_denial_metrics.json"
    
    print(f"[info] Loading data from {data_dir}")
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_df = add_engineered_features(train_df, train_df)
    eval_df = add_engineered_features(eval_df, train_df)

    X_train, cat_cols, num_cols = build_feature_frames(train_df)
    X_eval, _, _ = build_feature_frames(eval_df)
    y_train = train_df[label_col]
    y_eval = eval_df[label_col]

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", num_pipeline, num_cols),
        ]
    )

    groups = train_df[group_col] if group_col in train_df.columns else None
    cv = (
        GroupKFold(n_splits=3)
        if groups is not None
        else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("pre", pre), ("clf", lgbm)])

    cv_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    cv_metrics = compute_metrics(y_train, cv_proba, threshold=0.5)
    cv_conf_default = confusion_stats(y_train, cv_proba, threshold=0.5)
    cv_conf_best = confusion_stats(y_train, cv_proba, threshold=cv_metrics["best_f1_threshold"])
    print("CV metrics (LightGBM):", cv_metrics)
    print("CV confusion@0.5:", cv_conf_default)
    print("CV confusion@best_f1:", cv_conf_best)

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_eval)[:, 1]
    eval_metrics_default = compute_metrics(y_eval, proba, threshold=0.5)
    best_threshold = eval_metrics_default["best_f1_threshold"]
    eval_metrics_best = compute_metrics(y_eval, proba, threshold=best_threshold)
    eval_costs = sweep_costs(y_eval, proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    best_cost_row = min(eval_costs, key=lambda r: r["cost"])
    eval_conf_default = confusion_stats(y_eval, proba, threshold=0.5)
    eval_conf_best = confusion_stats(y_eval, proba, threshold=best_threshold)

    joblib.dump(pipeline, model_path)
    metrics_payload = {
        "cv_metrics": cv_metrics,
        "cv_confusion_0_5": cv_conf_default,
        "cv_confusion_best_f1": cv_conf_best,
        "eval_metrics_default": eval_metrics_default,
        "eval_confusion_0_5": eval_conf_default,
        "eval_metrics_best_f1": eval_metrics_best,
        "eval_confusion_best_f1": eval_conf_best,
        "eval_cost_sweep": eval_costs,
        "eval_best_cost_threshold": best_cost_row,
        "recommended_threshold": {
            "best_f1": best_threshold,
            "best_cost": best_cost_row["threshold"],
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"[done] saved LGBM model to {model_path}")
    print(f"[done] wrote metrics to {metrics_path}")
    
    return metrics_payload


def get_default_data_dir() -> Path:
    """Get the default data directory (augmented if available, else raw)."""
    aug_dir = ARTIFACTS_DIR / "data_cleaning_augmented"
    raw_dir = ARTIFACTS_DIR / "data_cleaning_raw"
    if (aug_dir / "claims_enriched_train.csv").exists():
        return aug_dir
    return raw_dir


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM denial prediction model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing train/eval CSVs (default: auto-detect augmented or raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save model and metrics (default: based on data-dir)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run training on both augmented and raw datasets.",
    )
    args = parser.parse_args()

    if args.both:
        results = {}
        for suffix in ["raw", "augmented", "balanced"]:
            data_dir = ARTIFACTS_DIR / f"data_cleaning_{suffix}"
            if not (data_dir / "claims_enriched_train.csv").exists():
                print(f"[warn] Skipping {suffix}: {data_dir} not found")
                continue
            output_dir = ARTIFACTS_DIR / f"stage1b_{suffix}"
            print("\n" + "=" * 60)
            print(f"Training on {suffix.upper()} data")
            print("=" * 60)
            metrics = run_training(data_dir, output_dir)
            results[suffix] = metrics
        return results
    else:
        data_dir = args.data_dir or get_default_data_dir()
        # Determine output dir based on data source
        if args.output_dir:
            output_dir = args.output_dir
        elif "balanced" in str(data_dir):
            output_dir = ARTIFACTS_DIR / "stage1b_balanced"
        elif "raw" in str(data_dir):
            output_dir = ARTIFACTS_DIR / "stage1b_raw"
        elif "augmented" in str(data_dir):
            output_dir = ARTIFACTS_DIR / "stage1b_augmented"
        else:
            output_dir = ARTIFACTS_DIR / "stage1b"
        
        return run_training(data_dir, output_dir)


if __name__ == "__main__":
    main()


