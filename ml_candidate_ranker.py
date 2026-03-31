from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ml_signal_model import (
    DATASET_FILE as _UNUSED_SIGNAL_DATASET_FILE,
    DEFAULT_MODEL_FILE as _UNUSED_SIGNAL_MODEL_FILE,
    DEFAULT_REPORT_FILE as _UNUSED_SIGNAL_REPORT_FILE,
    DatasetBundle,
    LogisticModel,
    MLPModel,
    StandardScaler,
    _iter_jsonl,
    _parse_ts,
    _safe_float,
    build_runtime_record as _build_signal_runtime_record,
    evaluate_predictions,
    find_best_threshold,
    permutation_importance,
    predict_proba_from_payload,
    render_text as _unused_render_text,
    roc_auc_score_np as _unused_roc_auc_score_np,
    rule_baseline_metrics,
    save_json,
    safe_feature_names as signal_safe_feature_names,
    vectorize_record as _unused_vectorize_record,
)


ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "critic_dataset.jsonl"
DEFAULT_MODEL_FILE = ROOT / "ml_candidate_ranker.json"
DEFAULT_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"

DECISION_FEATURES = [
    "candidate_score",
    "base_score",
    "score_floor",
    "forecast_return_pct",
    "today_change_pct",
    "ml_proba",
    "mtf_soft_penalty",
    "fresh_priority",
    "catchup",
    "continuation_profile",
    "flag_entry_ok",
    "flag_breakout_ok",
    "flag_retest_ok",
    "flag_surge_ok",
    "flag_impulse_ok",
    "flag_alignment_ok",
]


def build_runtime_candidate_record(
    *,
    sym: str,
    tf: str,
    signal_type: str,
    is_bull_day: bool,
    bar_ts: int,
    feat: dict,
    data: np.ndarray,
    i: int,
    candidate_score: float,
    base_score: float,
    score_floor: float,
    forecast_return_pct: float,
    today_change_pct: float,
    ml_proba: Optional[float],
    mtf_soft_penalty: float,
    fresh_priority: bool,
    catchup: bool,
    continuation_profile: bool,
    signal_flags: Optional[Dict[str, bool]] = None,
    btc_vs_ema50: float = 0.0,
    btc_momentum_4h: float = 0.0,
    market_vol_24h: float = 0.0,
) -> dict:
    rec = _build_signal_runtime_record(
        sym=sym,
        tf=tf,
        signal_type=signal_type,
        is_bull_day=is_bull_day,
        bar_ts=bar_ts,
        feat=feat,
        data=data,
        i=i,
        btc_vs_ema50=btc_vs_ema50,
        btc_momentum_4h=btc_momentum_4h,
        market_vol_24h=market_vol_24h,
    )
    rec["decision"] = {
        "candidate_score": float(candidate_score),
        "base_score": float(base_score),
        "score_floor": float(score_floor),
        "forecast_return_pct": float(forecast_return_pct),
        "today_change_pct": float(today_change_pct),
        "ml_proba": None if ml_proba is None else float(ml_proba),
        "mtf_soft_penalty": float(mtf_soft_penalty),
        "fresh_priority": bool(fresh_priority),
        "catchup": bool(catchup),
        "continuation_profile": bool(continuation_profile),
        "signal_flags": signal_flags or {},
    }
    return rec


def safe_feature_names() -> List[str]:
    return list(signal_safe_feature_names()) + DECISION_FEATURES


def build_feature_dict(rec: dict) -> Dict[str, float]:
    from ml_signal_model import build_feature_dict as build_signal_feature_dict

    out = build_signal_feature_dict(rec)
    decision = rec.get("decision") or {}
    signal_flags = decision.get("signal_flags") or {}
    out["candidate_score"] = _safe_float(decision.get("candidate_score"))
    out["base_score"] = _safe_float(decision.get("base_score"))
    out["score_floor"] = _safe_float(decision.get("score_floor"))
    out["forecast_return_pct"] = _safe_float(decision.get("forecast_return_pct"))
    out["today_change_pct"] = _safe_float(decision.get("today_change_pct"))
    out["ml_proba"] = _safe_float(decision.get("ml_proba"))
    out["mtf_soft_penalty"] = _safe_float(decision.get("mtf_soft_penalty"))
    out["fresh_priority"] = 1.0 if decision.get("fresh_priority") else 0.0
    out["catchup"] = 1.0 if decision.get("catchup") else 0.0
    out["continuation_profile"] = 1.0 if decision.get("continuation_profile") else 0.0
    out["flag_entry_ok"] = 1.0 if signal_flags.get("entry_ok") else 0.0
    out["flag_breakout_ok"] = 1.0 if signal_flags.get("breakout_ok") else 0.0
    out["flag_retest_ok"] = 1.0 if signal_flags.get("retest_ok") else 0.0
    out["flag_surge_ok"] = 1.0 if signal_flags.get("surge_ok") else 0.0
    out["flag_impulse_ok"] = 1.0 if signal_flags.get("impulse_ok") else 0.0
    out["flag_alignment_ok"] = 1.0 if signal_flags.get("alignment_ok") else 0.0
    return out


def vectorize_record(rec: dict, feature_names: Sequence[str]) -> np.ndarray:
    fmap = build_feature_dict(rec)
    return np.array([_safe_float(fmap.get(name)) for name in feature_names], dtype=float)


def predict_proba_from_candidate_payload(payload: dict, rec: dict) -> float:
    feature_names = payload["feature_names"]
    x = vectorize_record(rec, feature_names)
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    model = payload["model"]
    if model["type"] == "logistic":
        w = np.asarray(model["weights"], dtype=float)
        b = float(model["bias"])
        z = float(x @ w + b)
        z = max(-35.0, min(35.0, z))
        return float(1.0 / (1.0 + math.exp(-z)))
    if model["type"] == "mlp":
        W1 = np.asarray(model["W1"], dtype=float)
        b1 = np.asarray(model["b1"], dtype=float)
        W2 = np.asarray(model["W2"], dtype=float)
        b2 = np.asarray(model["b2"], dtype=float)
        z1 = x @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = float((a1 @ W2 + b2).reshape(-1)[0])
        z2 = max(-35.0, min(35.0, z2))
        return float(1.0 / (1.0 + math.exp(-z2)))
    raise ValueError(f"Unsupported candidate-ranker model type: {model['type']}")


def load_training_rows(path: Path, min_ts: Optional[datetime] = None) -> List[dict]:
    rows: List[dict] = []
    for rec in _iter_jsonl(path):
        if str(rec.get("signal_type", "none")) == "none":
            continue
        labels = rec.get("labels") or {}
        if labels.get("ret_5") is None:
            continue
        ts = _parse_ts(rec["ts_signal"])
        if min_ts and ts < min_ts:
            continue
        rec["_dt"] = ts
        rows.append(rec)
    rows.sort(key=lambda r: r["_dt"])
    return rows


def dataset_coverage_days(rows: list) -> float:
    """Return number of unique calendar days in dataset."""
    if not rows:
        return 0.0
    dates = set()
    for r in rows:
        ts = r.get("ts_signal", "")
        if ts and len(ts) >= 10:
            dates.add(ts[:10])
    return float(len(dates))


def _temporal_feature_names(days_covered: float) -> List[str]:
    """
    PATCH: Remove dow_sin / dow_cos when dataset covers < 60 days.
    With < 60 days these features overfit to specific weekdays rather than
    learning real market patterns.
    They remain available once enough diverse weekday data is collected.
    """
    exclude = set()
    if days_covered < 60:
        exclude.update({"dow_sin", "dow_cos"})
    base = list(signal_safe_feature_names()) + DECISION_FEATURES
    return [f for f in base if f not in exclude]


def _target_return(rec: dict) -> float:
    """
    PATCH: Improved target combining exit PnL and forward returns.
    Uses worst-case (min) approach when exit_pnl available to be risk-aware.
    ret_3 added as fast-exit proxy for early divergence.
    """
    labels = rec.get("labels") or {}
    ret_3 = _safe_float(labels.get("ret_3")) if labels.get("ret_3") is not None else None
    ret_5 = _safe_float(labels.get("ret_5"))
    trade_taken = bool(labels.get("trade_taken"))
    trade_exit = labels.get("trade_exit_pnl")
    if trade_taken and trade_exit is not None:
        # PATCH: worst-case target - min of forward return and actual exit
        # This penalises entries where price moved against us even if exit was managed
        combined = min(_safe_float(trade_exit), ret_5)
        if ret_3 is not None:
            combined = min(combined, ret_3)  # early momentum check
        return combined
    # No exit data: use min(ret_3, ret_5) if both available
    if ret_3 is not None:
        return min(ret_3, ret_5)
    return ret_5


def build_dataset(rows: List[dict], positive_ret_threshold: float = 0.0,
                  days_covered: float = 0.0) -> DatasetBundle:
    # PATCH: use temporal-aware feature selection (drops dow_sin/cos if < 60 days)
    feature_names = _temporal_feature_names(days_covered) if days_covered > 0 else safe_feature_names()
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    r = np.zeros(len(rows), dtype=float)

    for idx, rec in enumerate(rows):
        X[idx] = vectorize_record(rec, feature_names)
        target_ret = _target_return(rec)
        r[idx] = target_ret
        y[idx] = 1.0 if target_ret > positive_ret_threshold else 0.0

    n = len(rows)
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1) if n > 2 else n

    return DatasetBundle(
        feature_names=feature_names,
        X_train=X[:train_end],
        y_train=y[:train_end],
        r_train=r[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        r_val=r[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
        r_test=r[val_end:],
        meta_test=rows[val_end:],
    )


def train_and_evaluate(
    dataset_path: Path,
    positive_ret_threshold: float = 0.0,
    min_ts: Optional[datetime] = None,
    min_rows: int = 500,
) -> dict:
    rows = load_training_rows(dataset_path, min_ts=min_ts)
    if len(rows) < min_rows:
        raise RuntimeError("Not enough labeled candidate rows for ranker training")

    days = dataset_coverage_days(rows)
    excluded = [] if days >= 60 else ["dow_sin", "dow_cos"]
    if excluded:
        import logging as _log
        _log.getLogger(__name__).warning(
            "PATCH: dataset covers %.0f days (<60) - excluding temporal features %s to prevent weekday overfitting",
            days, excluded,
        )
    bundle = build_dataset(rows, positive_ret_threshold=positive_ret_threshold, days_covered=days)
    scaler = StandardScaler().fit(bundle.X_train)
    X_train = scaler.transform(bundle.X_train)
    X_val = scaler.transform(bundle.X_val)
    X_test = scaler.transform(bundle.X_test)

    models = {
        "logistic": LogisticModel(X_train.shape[1]).fit(X_train, bundle.y_train),
        "mlp": MLPModel(X_train.shape[1]).fit(X_train, bundle.y_train),
    }

    validation: Dict[str, dict] = {}
    best_name = ""
    best_threshold = 0.5
    best_score = -1e9
    for name, model in models.items():
        val_score = model.predict_proba(X_val)
        threshold, metrics = find_best_threshold(bundle.y_val, val_score, bundle.r_val)
        validation[name] = metrics
        score = metrics["selected_ret5_avg"] * (0.35 + metrics["coverage"]) + 0.15 * metrics["precision"]
        if score > best_score:
            best_score = score
            best_name = name
            best_threshold = threshold

    best_model = models[best_name]
    test_score = best_model.predict_proba(X_test)
    baseline = rule_baseline_metrics(bundle.r_test, bundle.y_test)
    filtered = evaluate_predictions(bundle.y_test, test_score, bundle.r_test, best_threshold)
    importances = permutation_importance(
        best_model,
        X_test,
        bundle.y_test,
        bundle.r_test,
        bundle.feature_names,
        best_threshold,
    )
    model_payload = {
        "model_name": best_name,
        "feature_names": bundle.feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": best_threshold,
        "positive_ret_threshold": positive_ret_threshold,
        "model": best_model.to_dict(),
    }
    suggestions = [
        "Использовать ranker сначала как quality-score bonus для top-N кандидатов, а не как hard-block движок.",
        "Собирать disagreement samples: bot_take vs ranker_skip и bot_skip vs ranker_take.",
        "Отдельно обучать future exit/hold model, не смешивая её с entry ranker.",
    ]
    return {
        "dataset_file": str(dataset_path),
        "rows_total": len(rows),
        "train_rows": int(bundle.X_train.shape[0]),
        "val_rows": int(bundle.X_val.shape[0]),
        "test_rows": int(bundle.X_test.shape[0]),
        "feature_count": len(bundle.feature_names),
        "positive_ret_threshold": positive_ret_threshold,
        "chosen_model": best_name,
        "validation": validation,
        "test_baseline_candidates": baseline,
        "test_ranker": filtered,
        "improvement_delta": {
            "ret5_avg_delta": round(filtered["selected_ret5_avg"] - baseline["selected_ret5_avg"], 4),
            "win_rate_delta": round(filtered["selected_win_rate"] - baseline["selected_win_rate"], 4),
            "coverage_delta": round(filtered["coverage"] - baseline["coverage"], 4),
        },
        "top_feature_importance": importances,
        "suggestions": suggestions,
        "dataset_days_covered": round(days, 1),
        "temporal_features_excluded": excluded,
        "model_payload": model_payload,
    }


def render_text(report: dict) -> str:
    lines = [
        "ML Candidate Ranker",
        f"Rows: {report['rows_total']} (train={report['train_rows']}, val={report['val_rows']}, test={report['test_rows']})",
        f"Chosen model: {report['chosen_model']}",
        "",
        "Validation:",
    ]
    for name, metrics in report["validation"].items():
        lines.append(
            f"  {name}: thr={metrics['threshold']:.2f} f1={metrics['f1']:.3f} "
            f"prec={metrics['precision']:.3f} cov={metrics['coverage']:.3f} "
            f"ret={metrics['selected_ret5_avg']:+.4f}%"
        )
    lines.extend(
        [
            "",
            "Test comparison:",
            f"  all candidates: cov={report['test_baseline_candidates']['coverage']:.3f} "
            f"wr={report['test_baseline_candidates']['selected_win_rate']:.3f} "
            f"ret={report['test_baseline_candidates']['selected_ret5_avg']:+.4f}%",
            f"  ranker: cov={report['test_ranker']['coverage']:.3f} "
            f"wr={report['test_ranker']['selected_win_rate']:.3f} "
            f"ret={report['test_ranker']['selected_ret5_avg']:+.4f}%",
            f"  delta: ret={report['improvement_delta']['ret5_avg_delta']:+.4f}% "
            f"wr={report['improvement_delta']['win_rate_delta']:+.4f} "
            f"cov={report['improvement_delta']['coverage_delta']:+.4f}",
            "",
            "Top features:",
        ]
    )
    for item in report["top_feature_importance"]:
        lines.append(f"  {item['feature']}: {item['importance']:+.6f}")
    lines.append("")
    lines.append("Suggestions:")
    for item in report["suggestions"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def build_live_model_payload(report: dict) -> dict:
    return dict(report.get("model_payload", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate a candidate ranker for crypto bot decisions")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--positive-ret-threshold", type=float, default=0.0)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    min_ts = _parse_ts(args.min_date) if args.min_date else None
    report = train_and_evaluate(
        args.dataset,
        positive_ret_threshold=args.positive_ret_threshold,
        min_ts=min_ts,
    )
    save_json(args.model_out, build_live_model_payload(report))
    save_json(args.report_out, {k: v for k, v in report.items() if k != "model_payload"})

    if args.as_json:
        print(json.dumps({k: v for k, v in report.items() if k != "model_payload"}, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"Model saved to: {args.model_out}")
        print(f"Report saved to: {args.report_out}")


if __name__ == "__main__":
    main()
