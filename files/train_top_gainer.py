from __future__ import annotations

"""
Train Top Gainer Classifier from collected dataset.

Reads top_gainer_dataset.jsonl, trains multi-output CatBoost model,
and saves to top_gainer_model.json.

Multi-output targets:
  - label_top5:   P(coin in top 5 gainers by EOD)
  - label_top10:  P(coin in top 10)
  - label_top20:  P(coin in top 20)
  - label_top50:  P(coin in top 50)

Walk-forward validation:
  - Train on first 80% of days, validate on last 20%
  - Report precision, recall, F1 per tier

Usage:
    python train_top_gainer.py
    python train_top_gainer.py --min-samples 500
    python train_top_gainer.py --output top_gainer_model_v2.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from top_gainer_model import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_FILE = Path(__file__).resolve().parent / "top_gainer_dataset.jsonl"
MODEL_FILE = Path(__file__).resolve().parent / "top_gainer_model.json"


def load_dataset(path: Path, min_samples: int = 100) -> Tuple[np.ndarray, dict]:
    """Load dataset from JSONL. Returns (feature_matrix, labels_dict)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(records) < min_samples:
        log.error("Insufficient data: %d records (need %d)", len(records), min_samples)
        return np.array([]), {}

    log.info("Loaded %d records from %s", len(records), path)

    # Extract features
    feature_names = list(FEATURE_NAMES)
    X = np.zeros((len(records), len(feature_names)))
    labels = {
        "top5": np.zeros(len(records)),
        "top10": np.zeros(len(records)),
        "top20": np.zeros(len(records)),
        "top50": np.zeros(len(records)),
        "return": np.zeros(len(records)),
        "ts": np.zeros(len(records)),
        "symbol": [],
    }

    for idx, rec in enumerate(records):
        feat = rec.get("features", {})
        for fi, fn in enumerate(feature_names):
            X[idx, fi] = float(feat.get(fn, 0.0))
        labels["top5"][idx] = float(rec.get("label_top5", 0))
        labels["top10"][idx] = float(rec.get("label_top10", 0))
        labels["top20"][idx] = float(rec.get("label_top20", 0))
        labels["top50"][idx] = float(rec.get("label_top50", 0))
        labels["return"][idx] = float(rec.get("eod_return_pct", 0))
        labels["ts"][idx] = float(rec.get("ts", 0))
        labels["symbol"].append(rec.get("symbol", ""))

    return X, labels


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_name: str,
) -> Tuple[Any, dict]:
    """
    Train a single gradient boosting classifier.

    Tries CatBoost first, falls back to custom decision stump ensemble.
    Returns (model_payload, metrics).
    """
    # Try CatBoost
    try:
        from catboost import CatBoostClassifier, Pool

        pos_weight = max(1.0, (1 - y_train.mean()) / max(0.01, y_train.mean()))
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
            eval_metric="AUC",
            early_stopping_rounds=50,
        )

        train_pool = Pool(X_train, y_train, weight=sample_weight)
        val_pool = Pool(X_val, y_val)
        model.fit(train_pool, eval_set=val_pool)

        # Predict
        val_proba = model.predict_proba(X_val)[:, 1]
        metrics = _compute_metrics(y_val, val_proba, target_name)

        # Feature importance
        importance = model.get_feature_importance()
        top_features = sorted(
            zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True
        )[:10]

        log.info("  %s (CatBoost): AUC=%.3f, P@0.3=%.3f, R@0.3=%.3f",
                 target_name, metrics["auc"], metrics["precision_at_03"],
                 metrics["recall_at_03"])
        log.info("  Top features: %s", ", ".join(f"{n}={v:.1f}" for n, v in top_features[:5]))

        # Serialize to temp file then read back
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_path = tf.name
        try:
            model.save_model(tmp_path, format="json")
            model_json_str = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            os.unlink(tmp_path)

        payload = {
            "model_type": "catboost",
            "target": target_name,
            "model_json": model_json_str,
            "feature_names": list(FEATURE_NAMES),
            "metrics": metrics,
            "top_features": {n: float(v) for n, v in top_features},
        }
        return payload, metrics

    except ImportError:
        log.warning("CatBoost not available, using numpy decision stump ensemble")

    # Fallback: simple gradient boosting with decision stumps
    payload, metrics = _train_stump_ensemble(X_train, y_train, X_val, y_val, target_name)
    return payload, metrics


def _train_stump_ensemble(
    X_train, y_train, X_val, y_val, target_name,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
) -> Tuple[dict, dict]:
    """Simple gradient boosting with decision stumps (no external dependencies)."""
    n_train = len(y_train)
    n_features = X_train.shape[1]

    # Initialize with log-odds
    pos = y_train.sum()
    neg = n_train - pos
    init_pred = float(np.log(max(pos, 1) / max(neg, 1)))

    train_pred = np.full(n_train, init_pred)
    val_pred = np.full(len(y_val), init_pred)

    stumps = []

    for t in range(n_estimators):
        # Compute gradients (logistic loss)
        p = 1.0 / (1.0 + np.exp(-train_pred))
        grad = p - y_train  # negative gradient = y - p

        # Find best split
        best_gain = -1
        best_feature = 0
        best_threshold = 0.0
        best_left_val = 0.0
        best_right_val = 0.0

        for f in range(n_features):
            # Simple: try percentile thresholds
            for pctl in [25, 50, 75]:
                thr = float(np.percentile(X_train[:, f], pctl))
                left_mask = X_train[:, f] <= thr
                right_mask = ~left_mask

                if left_mask.sum() < 5 or right_mask.sum() < 5:
                    continue

                left_val = -float(np.mean(grad[left_mask]))
                right_val = -float(np.mean(grad[right_mask]))
                gain = (left_mask.sum() * left_val ** 2 +
                        right_mask.sum() * right_val ** 2)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thr
                    best_left_val = left_val
                    best_right_val = right_val

        if best_gain <= 0:
            break

        # Update predictions
        left_mask_train = X_train[:, best_feature] <= best_threshold
        train_pred[left_mask_train] += learning_rate * best_left_val
        train_pred[~left_mask_train] += learning_rate * best_right_val

        left_mask_val = X_val[:, best_feature] <= best_threshold
        val_pred[left_mask_val] += learning_rate * best_left_val
        val_pred[~left_mask_val] += learning_rate * best_right_val

        stumps.append({
            "feature": best_feature,
            "threshold": best_threshold,
            "left_value": best_left_val * learning_rate,
            "right_value": best_right_val * learning_rate,
        })

    # Final probabilities
    val_proba = 1.0 / (1.0 + np.exp(-val_pred))
    metrics = _compute_metrics(y_val, val_proba, target_name)

    log.info("  %s (stumps): AUC=%.3f, P@0.3=%.3f, R@0.3=%.3f",
             target_name, metrics["auc"], metrics["precision_at_03"],
             metrics["recall_at_03"])

    payload = {
        "model_type": "stump_ensemble",
        "target": target_name,
        "init_pred": init_pred,
        "learning_rate": learning_rate,
        "stumps": stumps,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
    }
    return payload, metrics


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, name: str) -> dict:
    """Compute classification metrics."""
    from collections import Counter

    # AUC (simple trapezoidal)
    sorted_idx = np.argsort(-y_proba)
    y_sorted = y_true[sorted_idx]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        tp = 0
        fp = 0
        auc = 0.0
        prev_fpr = 0.0
        for i, y in enumerate(y_sorted):
            if y == 1:
                tp += 1
            else:
                fp += 1
                fpr = fp / n_neg
                tpr = tp / n_pos
                auc += (fpr - prev_fpr) * tpr
                prev_fpr = fpr
        auc += (1.0 - prev_fpr) * (tp / n_pos)

    # Precision/Recall at threshold 0.3
    pred_03 = (y_proba >= 0.3).astype(int)
    tp_03 = int(((pred_03 == 1) & (y_true == 1)).sum())
    fp_03 = int(((pred_03 == 1) & (y_true == 0)).sum())
    fn_03 = int(((pred_03 == 0) & (y_true == 1)).sum())
    precision_03 = tp_03 / (tp_03 + fp_03) if (tp_03 + fp_03) > 0 else 0.0
    recall_03 = tp_03 / (tp_03 + fn_03) if (tp_03 + fn_03) > 0 else 0.0

    # Precision/Recall at threshold 0.5
    pred_05 = (y_proba >= 0.5).astype(int)
    tp_05 = int(((pred_05 == 1) & (y_true == 1)).sum())
    fp_05 = int(((pred_05 == 1) & (y_true == 0)).sum())
    fn_05 = int(((pred_05 == 0) & (y_true == 1)).sum())
    precision_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0.0
    recall_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0.0

    return {
        "name": name,
        "auc": round(auc, 4),
        "n_samples": len(y_true),
        "n_positive": n_pos,
        "positive_rate": round(n_pos / len(y_true), 4) if len(y_true) > 0 else 0,
        "precision_at_03": round(precision_03, 4),
        "recall_at_03": round(recall_03, 4),
        "precision_at_05": round(precision_05, 4),
        "recall_at_05": round(recall_05, 4),
    }


def train_and_save(
    min_samples: int = 100,
    val_ratio: float = 0.2,
    output: str = "",
) -> dict:
    """
    Programmatic entry point for training. Returns result dict with metrics.
    Used by daily_learning.py for automated retraining.
    """
    output_path = Path(output) if output else MODEL_FILE

    if not DATASET_FILE.exists():
        return {"status": "error", "error": f"dataset not found: {DATASET_FILE}"}

    X, labels = load_dataset(DATASET_FILE, min_samples=min_samples)
    if len(X) == 0:
        return {"status": "error", "error": "empty dataset"}

    timestamps = labels["ts"]
    sort_idx = np.argsort(timestamps)
    X = X[sort_idx]
    for key in ["top5", "top10", "top20", "top50", "return", "ts"]:
        labels[key] = labels[key][sort_idx]
    labels["symbol"] = [labels["symbol"][i] for i in sort_idx]

    split_idx = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]

    models = {}
    all_metrics = {}
    for tier in ["top5", "top10", "top20", "top50"]:
        y_train = labels[tier][:split_idx]
        y_val = labels[tier][split_idx:]
        model_payload, metrics = train_gradient_boosting(
            X_train, y_train, X_val, y_val, tier,
        )
        models[tier] = model_payload
        all_metrics[tier] = metrics

    combined = {
        "model_type": models["top20"].get("model_type", "stump_ensemble"),
        "feature_names": list(FEATURE_NAMES),
        "tier_models": models,
        "metrics": all_metrics,
        "thresholds": {"top5": 0.15, "top10": 0.20, "top20": 0.30, "top50": 0.40},
        "train_samples": split_idx,
        "val_samples": len(X_val),
    }

    output_path.write_text(json.dumps(combined, indent=2, default=str))
    log.info("Model saved to %s", output_path)

    m20 = all_metrics.get("top20", {})
    return {
        "status": "ok",
        "n_records": len(X),
        "train_samples": split_idx,
        "val_samples": len(X_val),
        "auc_top20": m20.get("auc"),
        "recall_at_03_top20": m20.get("recall_at_03"),
        "precision_at_03_top20": m20.get("precision_at_03"),
        "metrics": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Top Gainer Model")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum dataset size to train")
    parser.add_argument("--output", type=str, default=str(MODEL_FILE),
                        help="Output model path")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation ratio (walk-forward)")
    args = parser.parse_args()

    if not DATASET_FILE.exists():
        log.error("Dataset not found: %s", DATASET_FILE)
        log.error("Run: python backfill_top_gainer_dataset.py --daily")
        return

    X, labels = load_dataset(DATASET_FILE, min_samples=args.min_samples)
    if len(X) == 0:
        return

    # Walk-forward split (time-based, NOT random)
    timestamps = labels["ts"]
    sort_idx = np.argsort(timestamps)
    X = X[sort_idx]
    for key in ["top5", "top10", "top20", "top50", "return", "ts"]:
        labels[key] = labels[key][sort_idx]
    labels["symbol"] = [labels["symbol"][i] for i in sort_idx]

    split_idx = int(len(X) * (1 - args.val_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]

    log.info("Dataset: %d total, %d train, %d val", len(X), len(X_train), len(X_val))

    # Train per-tier models
    models = {}
    all_metrics = {}

    for tier in ["top5", "top10", "top20", "top50"]:
        y_train = labels[tier][:split_idx]
        y_val = labels[tier][split_idx:]

        log.info("\nTraining %s classifier (pos_rate=%.1f%%)...",
                 tier, y_train.mean() * 100)

        model_payload, metrics = train_gradient_boosting(
            X_train, y_train, X_val, y_val, tier,
        )
        models[tier] = model_payload
        all_metrics[tier] = metrics

    # Combine into single model file
    combined = {
        "model_type": models["top20"].get("model_type", "stump_ensemble"),
        "feature_names": list(FEATURE_NAMES),
        "tier_models": models,
        "metrics": all_metrics,
        "thresholds": {
            "top5": 0.15,
            "top10": 0.20,
            "top20": 0.30,
            "top50": 0.40,
        },
        "train_samples": split_idx,
        "val_samples": len(X_val),
    }

    Path(args.output).write_text(json.dumps(combined, indent=2, default=str))
    log.info("\nModel saved to %s", args.output)

    # Summary
    log.info("\n=== TRAINING SUMMARY ===")
    for tier, m in all_metrics.items():
        log.info("  %s: AUC=%.3f | P@0.3=%.3f R@0.3=%.3f | P@0.5=%.3f R@0.5=%.3f | pos_rate=%.1f%%",
                 tier, m["auc"], m["precision_at_03"], m["recall_at_03"],
                 m["precision_at_05"], m["recall_at_05"], m["positive_rate"] * 100)


if __name__ == "__main__":
    main()
