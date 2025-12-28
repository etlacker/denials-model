"""
Denial prediction model trained on the cleaned/enriched dataset (augmented inputs).

- Uses GroupKFold over patient_id when available.
- Reports cross-validated PR-AUC/cost sweep on train and eval metrics with best-F1 threshold.
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress sklearn feature name warnings (harmless, caused by pipeline transformations)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required for this script; please install it.") from exc


ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = ARTIFACTS_DIR / "data_cleaning"
STAGE_DIR = ARTIFACTS_DIR / "stage1a"
STAGE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = DATA_DIR / "claims_enriched_train.csv"
EVAL_PATH = DATA_DIR / "claims_enriched_eval.csv"
MODEL_PATH = STAGE_DIR / "claim_denial_model.joblib"
METRICS_PATH = STAGE_DIR / "claim_denial_metrics.json"

label_col = "label_denied"
group_col = "patient_id"
COST_FP = 1.0  # cost of flagging a paid claim as denied (false positive)
COST_FN = 5.0  # cost of missing a denial (false negative)
SWEEP_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
HPARAM_SWEEP = [
    # around previous best (depth 6)
    {"n_estimators": 650, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": 1.1, "min_child_weight": 1.0},
    {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.045, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.2, "min_child_weight": 1.0},
    # slightly deeper
    {"n_estimators": 850, "max_depth": 7, "learning_rate": 0.045, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.3, "min_child_weight": 1.2},
    # lower lr, more trees
    {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.035, "subsample": 0.92, "colsample_bytree": 0.92, "reg_lambda": 1.2, "min_child_weight": 1.0},
    # heavier depth option
    {"n_estimators": 900, "max_depth": 8, "learning_rate": 0.04, "subsample": 0.88, "colsample_bytree": 0.9, "reg_lambda": 1.4, "min_child_weight": 1.5},
]


def build_feature_frames(df: pd.DataFrame):
    drop_cols = {label_col}
    if group_col in df.columns:
        drop_cols.add(group_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return df[feature_cols].copy(), cat_cols, num_cols


def add_engineered_features(df: pd.DataFrame, train_df: pd.DataFrame | None = None):
    """Add simple, leak-safe engineered features."""
    df = df.copy()
    # Log of billed amount to reduce skew
    if "billed_amount" in df.columns:
        df["billed_amount_log"] = np.log1p(df["billed_amount"].astype(float).fillna(0))
    # Bucket counts to capture non-linear effects without deeper trees
    for col in ["proc_count", "diag_count", "lab_count", "med_count"]:
        if col in df.columns:
            if train_df is not None:
                qs = train_df[col].quantile([0.25, 0.5, 0.75]).values
            else:
                qs = df[col].quantile([0.25, 0.5, 0.75]).values
            # Ensure unique, monotonically increasing edges
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
    # Precision/recall arrays have length len(thresholds)+1; align when deriving F1 grid
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


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    # Add engineered features using train quantiles for buckets
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

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    sweep_results = []
    best = None
    best_params = None

    for i, params in enumerate(HPARAM_SWEEP, 1):
        clf = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        pipeline = Pipeline([("pre", pre), ("clf", clf)])

        # Cross-validated PR-AUC on train (grouped by patient if available)
        groups = train_df[group_col] if group_col in train_df.columns else None
        cv = (
            GroupKFold(n_splits=3)
            if groups is not None
            else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )
        cv_proba = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            method="predict_proba",
            n_jobs=1,  # xgboost already uses parallelism
        )[:, 1]
        cv_metrics = compute_metrics(y_train, cv_proba, threshold=0.5)

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate on held-out eval set
        proba = pipeline.predict_proba(X_eval)[:, 1]
        eval_metrics_default = compute_metrics(y_eval, proba, threshold=0.5)
        best_threshold = eval_metrics_default["best_f1_threshold"]
        eval_metrics_best = compute_metrics(y_eval, proba, threshold=best_threshold)
        cost_sweep = sweep_costs(y_eval, proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
        best_cost_row = min(cost_sweep, key=lambda r: r["cost"])

        result = {
            "config_id": i,
            "params": params,
            "cv_metrics": cv_metrics,
            "eval_metrics_default": eval_metrics_default,
            "eval_metrics_best_f1": eval_metrics_best,
            "best_cost": best_cost_row,
            "best_threshold": {
                "best_f1": best_threshold,
                "best_cost": best_cost_row["threshold"],
            },
            "ap_eval": eval_metrics_default["ap"],
        }
        sweep_results.append(result)

        if best is None or result["ap_eval"] > best["ap_eval"]:
            best = result
            best_params = params

    # Rebuild best pipeline and recompute metrics for reporting/saving
    if best_params is None:
        raise SystemExit("No model trained in sweep.")

    best_clf = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **best_params,
    )
    pipeline = Pipeline([("pre", pre), ("clf", best_clf)])

    groups = train_df[group_col] if group_col in train_df.columns else None
    cv = (
        GroupKFold(n_splits=5)
        if groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
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
    print("Cross-validated (train) metrics (best config):")
    for k, v in cv_metrics.items():
        print(f"  {k}: {v}")
    cv_conf_default = confusion_stats(y_train, cv_proba, threshold=0.5)
    cv_conf_best = confusion_stats(y_train, cv_proba, threshold=cv_metrics["best_f1_threshold"])
    print(f"  confusion@0.5: {cv_conf_default}")
    print(f"  confusion@best_f1: {cv_conf_best}")
    cv_costs = sweep_costs(y_train, cv_proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    print("\nCross-validated cost sweep (threshold -> cost, precision, recall):")
    for row in cv_costs:
        print(
            f"  t={row['threshold']:.3f} cost={row['cost']:.0f} "
            f"c/ex={row['cost_per_example']:.4f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} f1={row['f1']:.3f}"
        )

    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_eval)[:, 1]
    eval_metrics_default = compute_metrics(y_eval, proba, threshold=0.5)
    best_threshold = eval_metrics_default["best_f1_threshold"]
    eval_metrics_best = compute_metrics(y_eval, proba, threshold=best_threshold)
    cost_sweep = sweep_costs(y_eval, proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    best_cost_row = min(cost_sweep, key=lambda r: r["cost"])

    # Optional calibration on top of the best config
    calib_clf = CalibratedClassifierCV(
        estimator=XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            **best_params,
        ),
        method="isotonic",
        cv=3,
        n_jobs=1,
    )
    pipeline_calibrated = Pipeline([("pre", pre), ("clf", calib_clf)])
    pipeline_calibrated.fit(X_train, y_train)
    proba_cal = pipeline_calibrated.predict_proba(X_eval)[:, 1]
    eval_metrics_default_cal = compute_metrics(y_eval, proba_cal, threshold=0.5)
    best_threshold_cal = eval_metrics_default_cal["best_f1_threshold"]
    eval_metrics_best_cal = compute_metrics(y_eval, proba_cal, threshold=best_threshold_cal)
    cost_sweep_cal = sweep_costs(y_eval, proba_cal, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    best_cost_row_cal = min(cost_sweep_cal, key=lambda r: r["cost"])

    # Choose calibrated if it improves AP; otherwise keep uncalibrated
    use_calibrated = eval_metrics_default_cal["ap"] > eval_metrics_default["ap"]
    final_pipeline = pipeline_calibrated if use_calibrated else pipeline
    final_eval_metrics_default = eval_metrics_default_cal if use_calibrated else eval_metrics_default
    final_eval_metrics_best = eval_metrics_best_cal if use_calibrated else eval_metrics_best
    final_best_threshold = best_threshold_cal if use_calibrated else best_threshold
    final_best_cost_row = best_cost_row_cal if use_calibrated else best_cost_row

    print("\nEval metrics (best pipeline):")
    print("  default threshold=0.5")
    for k, v in final_eval_metrics_default.items():
        print(f"    {k}: {v}")
    eval_conf_default = confusion_stats(y_eval, proba if not use_calibrated else proba_cal, threshold=0.5)
    print(f"    confusion@0.5: {eval_conf_default}")

    print(f"  best F1 threshold={final_best_threshold:.4f}")
    for k, v in final_eval_metrics_best.items():
        print(f"    {k}: {v}")
    eval_conf_best = confusion_stats(y_eval, proba if not use_calibrated else proba_cal, threshold=final_best_threshold)
    print(f"    confusion@best_f1: {eval_conf_best}")
    eval_costs = cost_sweep_cal if use_calibrated else cost_sweep
    print("\nEval cost sweep (threshold -> cost, precision, recall):")
    for row in eval_costs:
        print(
            f"  t={row['threshold']:.3f} cost={row['cost']:.0f} "
            f"c/ex={row['cost_per_example']:.4f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} f1={row['f1']:.3f}"
        )

    joblib.dump(final_pipeline, MODEL_PATH)
    metrics_payload = {
        "sweep_results": sweep_results,
        "best_params": best_params,
        "use_calibrated": use_calibrated,
        "cv_metrics": cv_metrics,
        "cv_confusion_0_5": cv_conf_default,
        "cv_confusion_best_f1": cv_conf_best,
        "eval_metrics_default": final_eval_metrics_default,
        "eval_confusion_0_5": eval_conf_default,
        "eval_metrics_best_f1": final_eval_metrics_best,
        "eval_confusion_best_f1": eval_conf_best,
        "eval_cost_sweep": eval_costs,
        "eval_best_cost_threshold": final_best_cost_row,
        "recommended_threshold": {
            "best_f1": final_best_threshold,
            "best_cost": final_best_cost_row["threshold"],
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))
    print(f"\n[done] saved model to {MODEL_PATH}")
    print(f"[done] wrote metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

