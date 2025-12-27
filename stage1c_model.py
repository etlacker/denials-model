"""
CatBoost model for denial prediction with categorical handling and engineered features.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier, Pool

ARTIFACTS_DIR = Path("artifacts")
TRAIN_PATH = ARTIFACTS_DIR / "claims_enriched_train.csv"
EVAL_PATH = ARTIFACTS_DIR / "eval" / "claims_enriched_eval.csv"
MODEL_PATH = ARTIFACTS_DIR / "claim_denial_model_catboost.cbm"
METRICS_PATH = ARTIFACTS_DIR / "claim_denial_metrics_catboost.json"

label_col = "label_denied"
group_col = "patient_id"
COST_FP = 1.0
COST_FN = 5.0
SWEEP_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


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


def split_features(df: pd.DataFrame):
    drop_cols = {label_col}
    if group_col in df.columns:
        drop_cols.add(group_col)
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, cat_cols, num_cols


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


def prepare_data(train_df, eval_df):
    train_df = add_engineered_features(train_df, train_df)
    eval_df = add_engineered_features(eval_df, train_df)

    X_train, cat_cols, num_cols = split_features(train_df)
    X_eval, _, _ = split_features(eval_df)
    y_train = train_df[label_col]
    y_eval = eval_df[label_col]

    # Simple imputations
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_eval[num_cols] = num_imputer.transform(X_eval[num_cols])
    X_train[cat_cols] = X_train[cat_cols].fillna("missing")
    X_eval[cat_cols] = X_eval[cat_cols].fillna("missing")
    return X_train, X_eval, y_train, y_eval, cat_cols


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    X_train, X_eval, y_train, y_eval, cat_cols = prepare_data(train_df, eval_df)

    # CV strategy
    groups = train_df[group_col] if group_col in train_df.columns else None
    cv = (
        GroupKFold(n_splits=3)
        if groups is not None
        else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )

    lgb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC:hints=skip_train~false",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.2,
        "random_seed": 42,
        "iterations": 1200,
        "verbose": False,
        "use_best_model": False,
        "thread_count": -1,
    }

    # CV predictions
    # Manual CV loop because CatBoost is not a sklearn estimator with tags
    cv_probs = np.zeros(len(X_train))
    for train_idx, val_idx in cv.split(X_train, y_train, groups=groups):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        model_cv = CatBoostClassifier(
            **lgb_params,
            cat_features=[X_train.columns.get_loc(c) for c in cat_cols],
        )
        model_cv.fit(X_tr, y_tr, verbose=False)
        cv_probs[val_idx] = model_cv.predict_proba(X_va)[:, 1]

    cv_metrics = compute_metrics(y_train, cv_probs, threshold=0.5)
    cv_conf_default = confusion_stats(y_train, cv_probs, threshold=0.5)
    cv_conf_best = confusion_stats(y_train, cv_probs, threshold=cv_metrics["best_f1_threshold"])
    print("CV metrics (CatBoost):", cv_metrics)

    # Fit final model
    model = CatBoostClassifier(**lgb_params, cat_features=[X_train.columns.get_loc(c) for c in cat_cols])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_eval)[:, 1]
    eval_metrics_default = compute_metrics(y_eval, proba, threshold=0.5)
    best_threshold = eval_metrics_default["best_f1_threshold"]
    eval_metrics_best = compute_metrics(y_eval, proba, threshold=best_threshold)
    eval_costs = sweep_costs(y_eval, proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    best_cost_row = min(eval_costs, key=lambda r: r["cost"])
    eval_conf_default = confusion_stats(y_eval, proba, threshold=0.5)
    eval_conf_best = confusion_stats(y_eval, proba, threshold=best_threshold)

    model.save_model(MODEL_PATH)
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
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))
    print(f"[done] saved CatBoost model to {MODEL_PATH}")
    print(f"[done] wrote metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()


