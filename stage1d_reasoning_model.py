"""
Reasoning-augmented denial prediction model.

Uses Gemini LLM to generate reasoning about denial risk for each claim,
embeds the reasonings using sentence-transformers, and trains CatBoost
on the combined original features + reasoning embeddings.

Pipeline:
1. Load claims data
2. Generate reasoning strings via Gemini API (cached to disk, parallelized)
3. Embed reasonings using sentence-transformers
4. Concatenate embeddings with original features
5. Train CatBoost on augmented features
6. Evaluate and save metrics
"""

from pathlib import Path
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment variables
load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = ARTIFACTS_DIR / "data_cleaning"
STAGE_DIR = ARTIFACTS_DIR / "stage1d"
STAGE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = DATA_DIR / "claims_enriched_train.csv"
EVAL_PATH = DATA_DIR / "claims_enriched_eval.csv"
MODEL_PATH = STAGE_DIR / "claim_denial_model.cbm"
METRICS_PATH = STAGE_DIR / "claim_denial_metrics.json"
REASONINGS_TRAIN_PATH = STAGE_DIR / "reasonings_train.json"
REASONINGS_EVAL_PATH = STAGE_DIR / "reasonings_eval.json"
EMBEDDINGS_TRAIN_PATH = STAGE_DIR / "embeddings_train.npy"
EMBEDDINGS_EVAL_PATH = STAGE_DIR / "embeddings_eval.npy"

label_col = "label_denied"
group_col = "patient_id"
COST_FP = 1.0
COST_FN = 5.0
SWEEP_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Gemini configuration
GEMINI_MODEL = "gemini-3-flash-preview"
MAX_WORKERS = 50  # Number of parallel API calls
MAX_RETRIES = 3  # Number of retries per claim
SAVE_INTERVAL = 100  # Save cache every N completed claims


def get_gemini_client() -> genai.Client:
    """Initialize Gemini client with API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file.")
    return genai.Client(api_key=api_key)


def build_claim_description(row: pd.Series) -> str:
    """Build a text description of a claim for the LLM."""
    return f"""Patient: {row.get('age', 'Unknown')} years old, {row.get('gender', 'Unknown')}, {row.get('ethnicity', 'Unknown')}
Insurance: {row.get('insurance_type', 'Unknown')} via {row.get('insurance_provider', 'Unknown')}
Payment Method: {row.get('payment_method', 'Unknown')}
Visit Type: {row.get('visit_type', 'Unknown')} in {row.get('department', 'Unknown')}
Reason for Visit: {row.get('reason_for_visit', 'Unknown')}
Diagnosis Code: {row.get('diagnosis_code', 'Unknown')}
Admission Type: {row.get('admission_type', 'N/A')}
Billed Amount: ${row.get('billed_amount', 0):.2f}
Length of Stay: {row.get('length_of_stay', 'N/A')} days
Provider Specialty: {row.get('specialty', 'Unknown')} ({row.get('years_experience', 'Unknown')} years experience)
Procedures: {row.get('proc_count', 0)}, Diagnoses: {row.get('diag_count', 0)}, Labs: {row.get('lab_count', 0)}, Medications: {row.get('med_count', 0)}"""


def build_single_claim_prompt(claim_description: str) -> str:
    """Build a prompt for a single claim."""
    return f"""You are a healthcare claims denial prediction expert. Analyze the denial risk for this claim.

Provide:
1. 2-3 specific reasons this claim might be DENIED (consider: insurance coverage issues, medical necessity, prior authorization requirements, billing errors, timely filing)
2. 2-3 specific reasons this claim might be APPROVED (consider: appropriate coverage, medical necessity documentation, proper coding)
3. An overall risk assessment: LOW, MEDIUM, or HIGH

Format your response as a JSON object:
{{"denial_reasons": ["reason1", "reason2"], "approval_reasons": ["reason1", "reason2"], "risk_level": "MEDIUM", "reasoning_summary": "Brief 1-2 sentence summary of key risk factors"}}

Claim to analyze:

{claim_description}

Respond ONLY with the JSON object, no other text."""


def parse_single_response(response_text: str) -> dict:
    """Parse Gemini response for a single claim."""
    try:
        text = response_text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
        
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "denial_reasons": ["Unable to analyze"],
            "approval_reasons": ["Unable to analyze"],
            "risk_level": "MEDIUM",
            "reasoning_summary": "Analysis failed - using default assessment"
        }


def format_reasoning(result: dict) -> str:
    """Format a parsed result into a reasoning string."""
    denial_reasons = "; ".join(result.get("denial_reasons", ["Unknown"]))
    approval_reasons = "; ".join(result.get("approval_reasons", ["Unknown"]))
    risk_level = result.get("risk_level", "MEDIUM")
    summary = result.get("reasoning_summary", "")
    
    return (
        f"Denial risk: {risk_level}. "
        f"Denial factors: {denial_reasons}. "
        f"Approval factors: {approval_reasons}. "
        f"Summary: {summary}"
    )


def process_single_claim(
    client: genai.Client,
    idx: int,
    claim_description: str,
) -> tuple[int, str]:
    """Process a single claim with retries. Returns (index, reasoning_string)."""
    prompt = build_single_claim_prompt(claim_description)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            result = parse_single_response(response.text)
            return (idx, format_reasoning(result))
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                return (idx, "Denial risk: MEDIUM. Analysis unavailable due to API error.")
    
    return (idx, "Denial risk: MEDIUM. Analysis unavailable.")


def generate_reasonings_parallel(
    client: genai.Client,
    df: pd.DataFrame,
    cache_path: Path,
) -> list[str]:
    """Generate reasoning strings for all claims using parallel Gemini API calls."""
    
    # Load existing cache (partial or complete)
    cached_reasonings: dict[int, str] = {}
    if cache_path.exists():
        print(f"[info] Loading cached reasonings from {cache_path}")
        with open(cache_path, "r") as f:
            cached_list = json.load(f)
        # Check if complete
        if isinstance(cached_list, list) and len(cached_list) == len(df):
            # Check if all entries are non-empty (complete)
            if all(r and r != "" for r in cached_list):
                print(f"[info] Cache complete with {len(cached_list)} reasonings")
                return cached_list
        # If it's a dict (partial cache), load it
        if isinstance(cached_list, dict):
            cached_reasonings = {int(k): v for k, v in cached_list.items()}
            print(f"[info] Loaded {len(cached_reasonings)} cached reasonings, resuming...")
        elif isinstance(cached_list, list):
            # Convert list to dict, keeping non-empty entries
            cached_reasonings = {i: r for i, r in enumerate(cached_list) if r and r != ""}
            print(f"[info] Loaded {len(cached_reasonings)} cached reasonings from list, resuming...")
    
    # Identify claims that need processing
    indices_to_process = [i for i in range(len(df)) if i not in cached_reasonings]
    
    if not indices_to_process:
        print(f"[info] All {len(df)} claims already cached")
        return [cached_reasonings[i] for i in range(len(df))]
    
    print(f"[info] Generating reasonings for {len(indices_to_process)} claims using {MAX_WORKERS} parallel workers...", flush=True)
    
    # Build claim descriptions for uncached claims
    print(f"[info] Building claim descriptions...", flush=True)
    claim_descriptions = {}
    for idx in indices_to_process:
        row = df.iloc[idx]
        claim_descriptions[idx] = build_claim_description(row)
    print(f"[info] Built {len(claim_descriptions)} descriptions.", flush=True)
    
    # Thread-safe counter and lock for progress
    completed = [len(cached_reasonings)]  # Use list for mutability in closure
    lock = threading.Lock()
    total = len(df)
    
    def save_cache():
        """Save current state to cache."""
        with lock:
            with open(cache_path, "w") as f:
                json.dump(cached_reasonings, f)
    
    def process_with_progress(idx: int) -> tuple[int, str]:
        """Process a claim and update progress."""
        result = process_single_claim(client, idx, claim_descriptions[idx])
        
        with lock:
            cached_reasonings[result[0]] = result[1]
            completed[0] += 1
            current = completed[0]
        
        # Progress update every 100 claims or at milestones
        if current % 100 == 0 or current == total or current <= 10:
            print(f"[progress] Processed {current}/{total} claims ({100*current/total:.1f}%)", flush=True)
        
        # Save cache periodically
        if current % SAVE_INTERVAL == 0:
            save_cache()
        
        return result
    
    # Process claims in parallel
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_with_progress, idx): idx for idx in indices_to_process}
        
        for future in as_completed(futures):
            try:
                future.result()  # Raises exception if the task failed
            except Exception as e:
                idx = futures[future]
                print(f"[error] Claim {idx} failed: {e}", flush=True)
                with lock:
                    cached_reasonings[idx] = "Denial risk: MEDIUM. Analysis unavailable."
                    completed[0] += 1
    
    elapsed = time.time() - start_time
    print(f"[info] Completed {len(indices_to_process)} claims in {elapsed:.1f}s ({len(indices_to_process)/elapsed:.1f} claims/sec)")
    
    # Final cache save
    save_cache()
    
    # Convert to list format
    all_reasonings = [cached_reasonings.get(i, "Denial risk: MEDIUM. Analysis unavailable.") for i in range(len(df))]
    
    # Save as list for final format
    with open(cache_path, "w") as f:
        json.dump(all_reasonings, f)
    print(f"[info] Saved final reasonings to {cache_path}")
    
    return all_reasonings


def embed_reasonings(
    reasonings: list[str],
    cache_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed reasoning strings using sentence-transformers."""
    
    # Check cache
    if cache_path.exists():
        print(f"[info] Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    
    print(f"[info] Embedding {len(reasonings)} reasonings using {model_name}...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(reasonings, show_progress_bar=True, batch_size=64)
    
    # Cache
    np.save(cache_path, embeddings)
    print(f"[info] Cached embeddings to {cache_path}")
    
    return embeddings


def add_engineered_features(df: pd.DataFrame, train_df: pd.DataFrame | None = None):
    """Add engineered features (same as stage1c)."""
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
    """Split dataframe into features, identifying categorical and numeric columns."""
    drop_cols = {label_col}
    if group_col in df.columns:
        drop_cols.add(group_col)
    X = df[[c for c in df.columns if c not in drop_cols]].copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, cat_cols, num_cols


def compute_metrics(y_true, proba, threshold: float):
    """Compute classification metrics."""
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
    """Compute confusion matrix stats."""
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def sweep_costs(y_true, proba, thresholds, cost_fp: float, cost_fn: float):
    """Sweep thresholds and compute cost-based metrics."""
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
    print("=" * 60)
    print("Stage 1d: Reasoning-Augmented Denial Prediction Model")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    print(f"  Train: {len(train_df)} samples, Eval: {len(eval_df)} samples")
    
    # Initialize Gemini
    print("\n[2/6] Initializing Gemini API...")
    gemini_client = get_gemini_client()
    print(f"  Using model: {GEMINI_MODEL}")
    
    # Generate reasonings
    print("\n[3/6] Generating LLM reasonings...")
    train_reasonings = generate_reasonings_parallel(gemini_client, train_df, REASONINGS_TRAIN_PATH)
    eval_reasonings = generate_reasonings_parallel(gemini_client, eval_df, REASONINGS_EVAL_PATH)
    
    # Sample reasoning output
    print("\n  Sample reasoning (train[0]):")
    print(f"  {train_reasonings[0][:200]}...")
    
    # Embed reasonings
    print("\n[4/6] Embedding reasonings...")
    train_embeddings = embed_reasonings(train_reasonings, EMBEDDINGS_TRAIN_PATH)
    eval_embeddings = embed_reasonings(eval_reasonings, EMBEDDINGS_EVAL_PATH)
    print(f"  Embedding dimension: {train_embeddings.shape[1]}")
    
    # Prepare features
    print("\n[5/6] Preparing features...")
    train_df = add_engineered_features(train_df, train_df)
    eval_df = add_engineered_features(eval_df, train_df)
    
    X_train, cat_cols, num_cols = split_features(train_df)
    X_eval, _, _ = split_features(eval_df)
    y_train = train_df[label_col]
    y_eval = eval_df[label_col]
    
    # Impute missing values
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_eval[num_cols] = num_imputer.transform(X_eval[num_cols])
    X_train[cat_cols] = X_train[cat_cols].fillna("missing")
    X_eval[cat_cols] = X_eval[cat_cols].fillna("missing")
    
    # Add embedding features
    embedding_cols = [f"reasoning_emb_{i}" for i in range(train_embeddings.shape[1])]
    train_embeddings_df = pd.DataFrame(train_embeddings, columns=embedding_cols, index=X_train.index)
    eval_embeddings_df = pd.DataFrame(eval_embeddings, columns=embedding_cols, index=X_eval.index)
    
    X_train_aug = pd.concat([X_train.reset_index(drop=True), train_embeddings_df.reset_index(drop=True)], axis=1)
    X_eval_aug = pd.concat([X_eval.reset_index(drop=True), eval_embeddings_df.reset_index(drop=True)], axis=1)
    
    # Update column lists
    num_cols_aug = num_cols + embedding_cols
    
    print(f"  Original features: {len(X_train.columns)}")
    print(f"  Embedding features: {len(embedding_cols)}")
    print(f"  Total features: {len(X_train_aug.columns)}")
    
    # Train model
    print("\n[6/6] Training CatBoost model...")
    
    groups = train_df[group_col] if group_col in train_df.columns else None
    cv = (
        GroupKFold(n_splits=3)
        if groups is not None
        else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )
    
    cat_feature_indices = [X_train_aug.columns.get_loc(c) for c in cat_cols]
    
    catboost_params = {
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
    
    # Cross-validation
    cv_probs = np.zeros(len(X_train_aug))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_aug, y_train, groups=groups)):
        X_tr, X_va = X_train_aug.iloc[train_idx], X_train_aug.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        
        model_cv = CatBoostClassifier(**catboost_params, cat_features=cat_feature_indices)
        model_cv.fit(X_tr, y_tr, verbose=False)
        cv_probs[val_idx] = model_cv.predict_proba(X_va)[:, 1]
        print(f"  Fold {fold_idx + 1}/3 complete")
    
    cv_metrics = compute_metrics(y_train, cv_probs, threshold=0.5)
    cv_conf_default = confusion_stats(y_train, cv_probs, threshold=0.5)
    cv_conf_best = confusion_stats(y_train, cv_probs, threshold=cv_metrics["best_f1_threshold"])
    
    print("\n  CV Metrics:")
    print(f"    AP: {cv_metrics['ap']:.4f}")
    print(f"    Best F1: {cv_metrics['best_f1']:.4f} @ threshold {cv_metrics['best_f1_threshold']:.3f}")
    
    # Train final model
    print("\n  Training final model on full training set...")
    model = CatBoostClassifier(**catboost_params, cat_features=cat_feature_indices)
    model.fit(X_train_aug, y_train, verbose=False)
    
    # Evaluate
    proba = model.predict_proba(X_eval_aug)[:, 1]
    eval_metrics_default = compute_metrics(y_eval, proba, threshold=0.5)
    best_threshold = eval_metrics_default["best_f1_threshold"]
    eval_metrics_best = compute_metrics(y_eval, proba, threshold=best_threshold)
    eval_costs = sweep_costs(y_eval, proba, SWEEP_THRESHOLDS, cost_fp=COST_FP, cost_fn=COST_FN)
    best_cost_row = min(eval_costs, key=lambda r: r["cost"])
    eval_conf_default = confusion_stats(y_eval, proba, threshold=0.5)
    eval_conf_best = confusion_stats(y_eval, proba, threshold=best_threshold)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nEval Metrics (Reasoning-Augmented CatBoost):")
    print(f"  AP: {eval_metrics_default['ap']:.4f}")
    print(f"  F1 @ 0.5: {eval_metrics_default['f1_at_threshold']:.4f}")
    print(f"  Best F1: {eval_metrics_best['best_f1']:.4f} @ threshold {best_threshold:.3f}")
    print(f"  Confusion @ best F1: tp={eval_conf_best['tp']}, fp={eval_conf_best['fp']}, fn={eval_conf_best['fn']}, tn={eval_conf_best['tn']}")
    
    print(f"\nComparison to baseline CatBoost (from stage1c):")
    print(f"  Baseline AP: 0.701, This model AP: {eval_metrics_default['ap']:.3f}")
    print(f"  Baseline Best F1: 0.600, This model Best F1: {eval_metrics_best['best_f1']:.3f}")
    
    # Save model and metrics
    model.save_model(MODEL_PATH)
    
    metrics_payload = {
        "model": "CatBoost + Reasoning Embeddings",
        "reasoning_model": GEMINI_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": train_embeddings.shape[1],
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
    
    print(f"\n[done] Saved model to {MODEL_PATH}")
    print(f"[done] Saved metrics to {METRICS_PATH}")
    print(f"[done] Reasonings cached at {REASONINGS_TRAIN_PATH}, {REASONINGS_EVAL_PATH}")
    print(f"[done] Embeddings cached at {EMBEDDINGS_TRAIN_PATH}, {EMBEDDINGS_EVAL_PATH}")


if __name__ == "__main__":
    main()

