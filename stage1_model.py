# Train a denial prediction model (structured features)
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, accuracy_score
import joblib

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required for this script; please install it.") from exc

ARTIFACTS_DIR = Path("artifacts")
TRAIN_PATH = ARTIFACTS_DIR / "claims_enriched_train.csv"
EVAL_PATH = ARTIFACTS_DIR / "eval" / "claims_enriched_eval.csv"
MODEL_PATH = ARTIFACTS_DIR / "claim_denial_model.joblib"

train_df = pd.read_csv(TRAIN_PATH)
eval_df = pd.read_csv(EVAL_PATH)

# Define labels
label_col = "label_denied"

# Separate features by type (after notebook cleaned schema)
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in train_df.columns if c not in cat_cols + [label_col]]

feature_df_train = train_df[cat_cols + num_cols]
feature_df_eval = eval_df[cat_cols + num_cols]

cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),  # sparse by default
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

# Model choice: XGBoost (handles sparse OHE efficiently, non-linear)
neg, pos = (train_df[label_col] == 0).sum(), (train_df[label_col] == 1).sum()
scale_pos_weight = neg / max(pos, 1)

clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
)

pipeline = Pipeline([("pre", pre), ("clf", clf)])

X_train = feature_df_train
y_train = train_df[label_col]
X_eval = feature_df_eval
y_eval = eval_df[label_col]

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_eval)[:, 1]
preds = (proba >= 0.5).astype(int)
precision, recall, thresholds = precision_recall_curve(y_eval, proba)

metrics = {
    "ap": average_precision_score(y_eval, proba),
    "f1": f1_score(y_eval, preds),
    "accuracy": accuracy_score(y_eval, preds),
    "precision_at_50": precision[np.argmin(np.abs(recall - 0.5))],
}

print("Eval metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

joblib.dump(pipeline, MODEL_PATH)
print(f"[done] saved model to {MODEL_PATH}")
