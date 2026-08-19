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

    # A delisted pair keeps answering the ticker endpoint that feeds the
    # snapshot, so 3.98% of this file describes instruments that had already
    # stopped trading on the day the row claims. EOSUSDT carried
    # tg_return_since_open=6.79 with no candle since May 2025. Judged as of each
    # row's own day, so rows written while the symbol was still live survive.
    try:
        import phantom_filter as _pf
        records, _dropped = _pf.drop_phantom_rows(records)
        if _dropped:
            log.info("phantom filter: dropped %d of %d rows",
                     _dropped, _dropped + len(records))
    except Exception:
        log.exception("phantom filter unavailable; training on the raw file")

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

    # The entry-score bonus ladder used fixed cuts (0.3/0.3/0.35/0.4) tuned
    # against a model whose AUC was 0.99 because it could read the label. An
    # honestly-labelled model spreads its probabilities out, so those same cuts
    # fired three times as often — 12.2% -> 37.1% of live candidates got a
    # bonus, a large loosening of the entry gate smuggled in with a metric fix.
    # Predicting positives as often as they actually occur is self-calibrating:
    # it depends on this model's distribution, not on the previous model's.
    base = n_pos / len(y_true) if len(y_true) else 0.0
    bonus_threshold = (float(np.quantile(y_proba, 1.0 - base))
                       if 0.0 < base < 1.0 else 0.5)

    return {
        "name": name,
        "auc": round(auc, 4),
        "bonus_threshold": round(bonus_threshold, 6),
        "n_samples": len(y_true),
        "n_positive": n_pos,
        "positive_rate": round(n_pos / len(y_true), 4) if len(y_true) > 0 else 0,
        "precision_at_03": round(precision_03, 4),
        "recall_at_03": round(recall_03, 4),
        "precision_at_05": round(precision_05, 4),
        "recall_at_05": round(recall_05, 4),
    }


def _utc_days(ts_array) -> list:
    """`ts` is written in seconds by some snapshot paths and milliseconds by
    others; both appear in the dataset, so normalise rather than assume."""
    from datetime import datetime, timezone
    out = []
    for t in ts_array:
        t = float(t)
        if t > 1e11:
            t /= 1000.0
        out.append(datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d"))
    return out


def _apply_immutable_labels(X, labels):
    """Replace the snapshot tier labels with immutable later-EOD ones.

    Returns `(X, labels, stats)` unchanged when the flag is off, so the default
    path is byte-identical to before. `stats` is `None` in that case, and the
    caller reports the leaky provenance — an artifact that names the wrong
    label source is worse than one that admits the label is leaky.
    """
    try:
        import config as _cfg
        enabled = bool(getattr(_cfg, "TRAIN_IMMUTABLE_LABELS_ENABLED", False))
        floor = float(getattr(_cfg, "TRAIN_IMMUTABLE_LABEL_MIN_PCT", 5.0))
    except Exception:
        return X, labels, None
    if not enabled:
        return X, labels, None

    import immutable_labels as IL
    tiers = (5, 10, 20, 50)
    keep, new_labels, stats = IL.tier_labels(
        _utc_days(labels["ts"]), labels["symbol"], tiers=tiers, floor=floor)
    if not keep:
        return X, {}, stats

    idx = np.asarray(keep, dtype=int)
    out = {f"top{n}": np.asarray(new_labels[f"top{n}"], dtype=float)
           for n in tiers}
    out["return"] = labels["return"][idx]
    out["ts"] = labels["ts"][idx]
    out["symbol"] = [labels["symbol"][i] for i in keep]
    log.info("immutable labels: %d/%d rows kept, base rates %s",
             stats["n_labelled"], stats["n_rows_in"],
             {k: round(v, 4) for k, v in stats["base_rate"].items()})
    return X[idx], out, stats


def _evaluation_scope(day_grouped: bool, label_stats) -> str:
    """Name both defects independently. Fixing the split does not fix the label,
    and a scope string that implies it did would let one hide behind the other."""
    split = "day_grouped" if day_grouped else "time_sorted_row"
    label = "immutable_later_eod_label" if label_stats else "same_snapshot_label"
    return f"{split}_holdout_{label}"


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

    # TH-03: the dataset's own tier labels come from the same rolling-24h
    # snapshot that produced the features, so `tg_return_since_open` is an input
    # AND very nearly the answer. Relabelling from the immutable store drops
    # rows it cannot label rather than calling them negatives.
    X, labels, label_stats = _apply_immutable_labels(X, labels)
    if not labels:
        return {"status": "error", "error": "immutable relabelling left no rows"}

    # TH-04: cutting by row index lands the boundary INSIDE a UTC day, so part
    # of a day trains the model and the rest validates it. The tier labels are
    # per-day ranks, so knowing part of a day tells you about the rest.
    # Flagged off by default — this model feeds the ranker's hard veto, so
    # retraining on different rows changes live gating indirectly.
    day_grouped = False
    try:
        import config as _cfg
        day_grouped = bool(getattr(_cfg, "TRAIN_DAY_GROUPED_SPLIT_ENABLED", False))
        embargo = int(getattr(_cfg, "TRAIN_SPLIT_EMBARGO_DAYS", 0))
    except Exception:
        embargo = 0

    train_idx = val_idx = None
    if day_grouped:
        try:
            from day_split import split_indices_by_day
            train_idx, val_idx = split_indices_by_day(
                labels["ts"], train_frac=1 - val_ratio, embargo_days=embargo)
        except ValueError as exc:
            # Not enough distinct days to split honestly. Falling back to the
            # row cut would produce the very number this flag exists to stop
            # reporting, so the run fails instead.
            return {"status": "error",
                    "error": f"day-grouped split impossible: {exc}"}

    if train_idx is None:
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        n_train, n_val = split_idx, len(X) - split_idx
    else:
        X_train, X_val = X[train_idx], X[val_idx]
        n_train, n_val = len(train_idx), len(val_idx)

    scope = _evaluation_scope(day_grouped, label_stats)
    _timing = ("immutable_later_eod_close" if label_stats
               else "same_snapshot_current_24h_leaderboard")
    _encoding = [] if label_stats else ["tg_return_since_open"]

    models = {}
    all_metrics = {}
    for tier in ["top5", "top10", "top20", "top50"]:
        if train_idx is None:
            y_train = labels[tier][:split_idx]
            y_val = labels[tier][split_idx:]
        else:
            y_train = labels[tier][train_idx]
            y_val = labels[tier][val_idx]
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
        # Calibrated to each tier's own base rate: the ladder fires as often as
        # the tier actually occurs. The fixed 0.3/0.3/0.35/0.4 were tuned for
        # the leaky model's peaked probabilities and do not transfer.
        "bonus_thresholds": {t: m["bonus_threshold"] for t, m in
                             all_metrics.items() if "bonus_threshold" in m},
        "train_samples": n_train,
        "val_samples": len(X_val),
        "evaluation_scope": scope,
        "label_timing": _timing,
        "label_encoding_features": _encoding,
        "label_base_rate": (label_stats or {}).get("base_rate"),
        "n_records_labelled": (label_stats or {}).get("n_labelled"),
    }

    output_path.write_text(json.dumps(combined, indent=2, default=str))
    log.info("Model saved to %s", output_path)

    m20 = all_metrics.get("top20", {})
    return {
        "status": "ok",
        "n_records": len(X),
        "train_samples": n_train,
        "val_samples": len(X_val),
        "auc_top20": m20.get("auc"),
        "recall_at_03_top20": m20.get("recall_at_03"),
        "precision_at_03_top20": m20.get("precision_at_03"),
        "evaluation_scope": scope,
        "label_timing": _timing,
        "label_encoding_features": _encoding,
        # TH-01: the base rate travels with every ratio above it, because the
        # two label sets do not share one and AUC alone would not show that.
        "label_base_rate": (label_stats or {}).get("base_rate"),
        "n_records_labelled": (label_stats or {}).get("n_labelled"),
        "label_dropped_unlabelled": (label_stats or {}).get("dropped_unlabelled"),
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

    # Delegate to train_and_save rather than keeping a second copy of the
    # pipeline. The duplicate had drifted: it referenced `n_train` and `scope`,
    # which were never defined here, so this CLI path raised NameError before
    # writing anything — and it was where the leaky `label_timing` string
    # survived the relabelling. One path cannot disagree with itself.
    res = train_and_save(min_samples=args.min_samples,
                         val_ratio=args.val_ratio, output=args.output)
    if res.get("status") != "ok":
        log.error("training failed: %s", res.get("error"))
        return

    log.info("=== TRAINING SUMMARY ===")
    log.info("  scope=%s  labels=%s", res.get("evaluation_scope"),
             res.get("label_timing"))
    if res.get("n_records_labelled") is not None:
        log.info("  rows labelled=%s (dropped unlabelled=%s)",
                 res["n_records_labelled"], res.get("label_dropped_unlabelled"))
    base = res.get("label_base_rate") or {}
    for tier, m in (res.get("metrics") or {}).items():
        # TH-01: the base rate sits beside the ratio, never on its own line.
        br = base.get(tier, m.get("positive_rate", 0.0))
        log.info("  %s: AUC=%.3f | P@0.3=%.3f R@0.3=%.3f | base=%.2f%%",
                 tier, m["auc"], m["precision_at_03"], m["recall_at_03"],
                 br * 100)


if __name__ == "__main__":
    main()
