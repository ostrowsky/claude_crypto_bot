"""Train fast-reversal classifier (RM-3 Step 2).

Reads `critic_dataset.jsonl` records with `label_fast_reversal` set,
trains a binary classifier on entry-bar features, reports AUC + feature
importance, saves model to `fast_reversal_model.json`.

Walk-forward validation:
    Train on first 80% of records (by ts_signal), validate on last 20%.

Usage:
    pyembed\\python.exe files\\train_fast_reversal.py
    pyembed\\python.exe files\\train_fast_reversal.py --min-samples 200

Output:
    - fast_reversal_model.json: trained logistic regression weights
    - fast_reversal_report.json: AUC, feature importance, calibration

Implements stdlib-only logistic regression (no scipy/sklearn/catboost
dependency) so the script can run inside pyembed without extra packages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATASET_FILE = HERE / "critic_dataset.jsonl"
MODEL_FILE = HERE / "fast_reversal_model.json"
REPORT_FILE = HERE / "fast_reversal_report.json"

# Features from the scalar `f` dict (primary source — well-named, matches
# ml_signal_model.SAFE_SCALAR_FEATURES) + the `decision` block.
F_FEATURES = [
    "slope",
    "rsi",
    "adx",
    "vol_x",
    "macd_hist_norm",
    "atr_pct",
    "daily_range",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema200",
    "ema20_vs_ema50",
    "btc_vs_ema50",
    "btc_momentum_4h",
]
DECISION_FEATURES = [
    "candidate_score",
    "ml_proba",
    "forecast_return_pct",
    "today_change_pct",
]
FEATURE_NAMES = F_FEATURES + DECISION_FEATURES


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def load_labeled_records(
    path: Path,
    *,
    ret3_fallback: bool = True,
    ret3_threshold: float = -0.3,
    take_only: bool = True,
) -> List[dict]:
    """Load records with a fast-reversal label.

    The RM-3 spec label (`labels.label_fast_reversal`, trail-stop-hit in
    next 3 bars) requires `entry_context.atr_pct`+`trail_k` which were
    never logged to production critic_dataset (0/17437 records). To make
    the model trainable on real history, fall back to an outcome-based
    proxy: `label_fast_reversal = 1 if labels.ret_3 <= ret3_threshold`.
    This mirrors `_backtest_fast_reversal_by_mode.py`'s validated v1
    definition (price reversed within 3 bars) and is computable on all
    historical records that have ret_3 filled.

    Records get a synthetic `labels.label_fast_reversal` in-memory so the
    rest of the pipeline is unchanged.
    """
    if not path.exists():
        print(f"[!] {path} not found")
        return []

    records = []
    skipped = 0
    n_explicit = 0
    n_proxy = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            labels = rec.get("labels", {}) or {}
            if take_only:
                act = (rec.get("decision") or {}).get("action")
                if act != "take":
                    continue
            lab = labels.get("label_fast_reversal")
            if lab is not None:
                n_explicit += 1
                records.append(rec)
                continue
            if ret3_fallback:
                r3 = labels.get("ret_3")
                if r3 is None:
                    continue
                rec.setdefault("labels", {})["label_fast_reversal"] = (
                    1 if float(r3) <= ret3_threshold else 0
                )
                n_proxy += 1
                records.append(rec)

    if skipped > 0:
        print(f"[*] Skipped {skipped} malformed lines")
    print(f"[*] Labels: {n_explicit} explicit (spec trail-stop), "
          f"{n_proxy} proxy (ret_3 <= {ret3_threshold}%)")
    return records


def extract_features(rec: dict) -> List[float]:
    """Feature vector from the scalar `f` dict + `decision` block.

    Production critic_dataset stores rich scalar features under `f`
    (close_vs_ema20, slope, rsi, adx, vol_x, ...). The earlier version
    read a non-existent `seq_features` dict, which silently zeroed every
    technical feature and capped model AUC at ~0.53. Falls back to the
    entry bar of the `seq` matrix (column order = seq_feature_names) when
    a value is missing from `f`.
    """
    f = rec.get("f") or {}
    decision = rec.get("decision") or {}

    # seq[-1] fallback indexed by seq_feature_names
    seq = rec.get("seq") or []
    names = rec.get("seq_feature_names") or []
    seq_last = {}
    if seq and isinstance(seq[-1], (list, tuple)) and names:
        seq_last = {n: v for n, v in zip(names, seq[-1])}

    def pick(key: str) -> float:
        v = f.get(key)
        if v is None:
            v = seq_last.get(key)
        return _safe_float(v)

    vec = [pick(name) for name in F_FEATURES]
    vec += [_safe_float(decision.get(name)) for name in DECISION_FEATURES]
    return vec


def report_label_distribution(records: List[dict]) -> Dict[str, Any]:
    by_mode: Dict[str, Dict[str, int]] = {}
    overall = {"pos": 0, "neg": 0}

    for rec in records:
        label = int(rec.get("labels", {}).get("label_fast_reversal", 0))
        sig_type = rec.get("signal_type", "unknown")
        by_mode.setdefault(sig_type, {"pos": 0, "neg": 0})

        if label == 1:
            by_mode[sig_type]["pos"] += 1
            overall["pos"] += 1
        else:
            by_mode[sig_type]["neg"] += 1
            overall["neg"] += 1

    total = overall["pos"] + overall["neg"]
    print(f"\n{'='*70}")
    print(f"LABEL DISTRIBUTION ({total} records)")
    print(f"{'='*70}")
    print(f"  Overall reversal rate: {overall['pos']:5d} / {total:5d}  ({100*overall['pos']/max(total,1):.1f}%)")
    print(f"\n  Per-mode:")
    for mode in sorted(by_mode):
        m = by_mode[mode]
        mode_total = m["pos"] + m["neg"]
        if mode_total > 0:
            pct = 100 * m["pos"] / mode_total
            print(f"    {mode:20s}: {m['pos']:4d} / {mode_total:4d}  ({pct:5.1f}% reversal)")

    return {
        "total_records": total,
        "overall_positive_rate": overall["pos"] / max(total, 1),
        "by_mode": by_mode,
    }


def compute_auc(y_true: List[int], y_score: List[float]) -> float:
    """Compute AUC via Mann-Whitney U formula (stdlib-only)."""
    pos_scores = [s for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos_scores or not neg_scores:
        return 0.5

    wins = 0
    ties = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1
            elif ps == ns:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))


def train_logistic(
    X_train: List[List[float]],
    y_train: List[int],
    n_iter: int = 300,
    lr: float = 0.1,
    l2: float = 0.01,
) -> Tuple[List[float], float, List[float], List[float]]:
    """Train logistic regression with stdlib gradient descent."""
    if not X_train:
        return [], 0.0, [], []
    n_features = len(X_train[0])

    # Z-score normalization
    means = [sum(row[j] for row in X_train) / len(X_train) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        var = sum((row[j] - means[j]) ** 2 for row in X_train) / len(X_train)
        stds.append(math.sqrt(var) if var > 1e-10 else 1.0)

    X_norm = [
        [(row[j] - means[j]) / stds[j] for j in range(n_features)]
        for row in X_train
    ]

    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(n_iter):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for x, y in zip(X_norm, y_train):
            z = bias + sum(weights[j] * x[j] for j in range(n_features))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            for j in range(n_features):
                grad_w[j] += err * x[j]
            grad_b += err

        n = len(X_norm)
        for j in range(n_features):
            grad_w[j] = grad_w[j] / n + l2 * weights[j]
            weights[j] -= lr * grad_w[j]
        bias -= lr * grad_b / n

    return weights, bias, means, stds


def predict_proba(
    x: List[float], weights: List[float], bias: float,
    means: List[float], stds: List[float],
) -> float:
    n = len(weights)
    z = bias
    for j in range(n):
        z += weights[j] * (x[j] - means[j]) / stds[j]
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train fast-reversal classifier (RM-3)")
    ap.add_argument("--min-samples", type=int, default=200,
                    help="Minimum labeled records required (default: 200)")
    ap.add_argument("--output", type=Path, default=MODEL_FILE,
                    help=f"Output model file (default: {MODEL_FILE.name})")
    ap.add_argument("--report", type=Path, default=REPORT_FILE,
                    help=f"Output report file (default: {REPORT_FILE.name})")
    ap.add_argument("--dataset", type=Path, default=DATASET_FILE,
                    help="Input critic_dataset.jsonl")
    ap.add_argument("--ret3-threshold", type=float, default=-0.3,
                    help="ret_3 %% at/below which a take is a fast-reversal "
                         "(proxy label; default -0.3)")
    ap.add_argument("--no-ret3-fallback", action="store_true",
                    help="disable ret_3 proxy; require explicit "
                         "labels.label_fast_reversal")
    args = ap.parse_args()

    print(f"[*] Loading labeled records from {args.dataset}")
    records = load_labeled_records(
        args.dataset,
        ret3_fallback=not args.no_ret3_fallback,
        ret3_threshold=args.ret3_threshold,
    )
    print(f"[*] Found {len(records)} records with label_fast_reversal set")

    if len(records) < args.min_samples:
        print(f"[!] Need at least {args.min_samples} labeled records, have {len(records)}")
        print("[!] Data accumulation in progress. Re-run after EOD jobs add labels.")
        return 1

    dist = report_label_distribution(records)

    records.sort(key=lambda r: r.get("ts_signal", ""))
    split_idx = int(len(records) * 0.8)
    train_recs = records[:split_idx]
    val_recs = records[split_idx:]
    print(f"\n[*] Split: train={len(train_recs)}, val={len(val_recs)}")

    X_train = [extract_features(r) for r in train_recs]
    y_train = [int(r.get("labels", {}).get("label_fast_reversal", 0)) for r in train_recs]
    X_val = [extract_features(r) for r in val_recs]
    y_val = [int(r.get("labels", {}).get("label_fast_reversal", 0)) for r in val_recs]

    print("[*] Training logistic regression (stdlib)...")
    weights, bias, means, stds = train_logistic(X_train, y_train)

    y_train_pred = [predict_proba(x, weights, bias, means, stds) for x in X_train]
    y_val_pred = [predict_proba(x, weights, bias, means, stds) for x in X_val]
    auc_train = compute_auc(y_train, y_train_pred)
    auc_val = compute_auc(y_val, y_val_pred)

    print(f"\n[*] AUC train: {auc_train:.4f}")
    print(f"[*] AUC val:   {auc_val:.4f}")

    importance = sorted(zip(FEATURE_NAMES, weights), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE (by |weight|)")
    print(f"{'='*70}")
    for fname, w in importance:
        print(f"  {fname:25s}: {w:+.4f}")

    model_data = {
        "model_type": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "bias": bias,
        "means": means,
        "stds": stds,
        "trained_on": len(train_recs),
        "auc_train": auc_train,
        "auc_val": auc_val,
        "label_positive_rate": dist["overall_positive_rate"],
    }
    args.output.write_text(json.dumps(model_data, indent=2))
    print(f"\n[+] Model saved to {args.output}")

    report = {
        "training_records": len(train_recs),
        "validation_records": len(val_recs),
        "auc_train": auc_train,
        "auc_val": auc_val,
        "label_distribution": dist,
        "feature_importance": [{"feature": f, "weight": w} for f, w in importance],
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[+] Report saved to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
