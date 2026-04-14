from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

    CATBOOST_AVAILABLE = True
    CATBOOST_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local environment
    CatBoostClassifier = None  # type: ignore[assignment]
    CatBoostRanker = None  # type: ignore[assignment]
    CatBoostRegressor = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = str(exc)

from ml_signal_model import (
    DATASET_FILE as _UNUSED_SIGNAL_DATASET_FILE,
    DEFAULT_MODEL_FILE as _UNUSED_SIGNAL_MODEL_FILE,
    DEFAULT_REPORT_FILE as _UNUSED_SIGNAL_REPORT_FILE,
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
DEFAULT_EV_LAMBDA = 0.75
DEFAULT_QUALITY_SCORE_WEIGHT = 0.35
DEFAULT_RANK_SCORE_WEIGHT = 0.20
DEFAULT_TOP_GAINER_SCORE_WEIGHT = 0.25
DEFAULT_CAPTURE_SCORE_WEIGHT = 0.15
CATBOOST_RUNTIME_CACHE_DIR = ROOT.parent / ".runtime" / "catboost_model_cache"
CATBOOST_RANDOM_SEED = 42
CATBOOST_ITERATIONS = 250
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_L2_LEAF_REG = 6.0

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
    "near_miss",
    "flag_entry_ok",
    "flag_breakout_ok",
    "flag_retest_ok",
    "flag_surge_ok",
    "flag_impulse_ok",
    "flag_alignment_ok",
]


@dataclass
class DatasetBundle:
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    r_train: np.ndarray
    er_train: np.ndarray
    dd_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    r_val: np.ndarray
    er_val: np.ndarray
    dd_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    r_test: np.ndarray
    er_test: np.ndarray
    dd_test: np.ndarray
    tg_train: np.ndarray
    cap_train: np.ndarray
    tmask_train: np.ndarray
    tg_val: np.ndarray
    cap_val: np.ndarray
    tmask_val: np.ndarray
    tg_test: np.ndarray
    cap_test: np.ndarray
    tmask_test: np.ndarray
    meta_train: List[dict]
    meta_val: List[dict]
    meta_test: List[dict]


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
    near_miss: bool = False,
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
        "near_miss": bool(near_miss),
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
    out["near_miss"] = 1.0 if decision.get("near_miss") else 0.0
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


def _transform_payload_vector(payload: dict, rec: dict) -> np.ndarray:
    feature_names = payload["feature_names"]
    x = vectorize_record(rec, feature_names)
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)
    scale[scale < 1e-8] = 1.0
    return (x - mean) / scale


def _sigmoid_scalar(z: float) -> float:
    z = max(-35.0, min(35.0, float(z)))
    return float(1.0 / (1.0 + math.exp(-z)))


_CATBOOST_LOADED_MODELS: dict[str, Any] = {}


def _catboost_blob_to_dict(model: Any, *, model_type: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        model.save_model(str(tmp_path), format="cbm")
        blob = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
    digest = hashlib.sha1(blob).hexdigest()
    return {
        "type": model_type,
        "format": "cbm",
        "sha1": digest,
        "blob_b64": base64.b64encode(blob).decode("ascii"),
    }


def _load_catboost_runtime_model(model: dict) -> Any:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError(f"CatBoost is not installed: {CATBOOST_IMPORT_ERROR}")
    blob_b64 = str(model.get("blob_b64") or "")
    if not blob_b64:
        raise ValueError("CatBoost payload missing blob_b64")
    digest = str(model.get("sha1") or "")
    if not digest:
        digest = hashlib.sha1(base64.b64decode(blob_b64.encode("ascii"))).hexdigest()
    if digest in _CATBOOST_LOADED_MODELS:
        return _CATBOOST_LOADED_MODELS[digest]

    CATBOOST_RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CATBOOST_RUNTIME_CACHE_DIR / f"{digest}.cbm"
    if not path.exists():
        path.write_bytes(base64.b64decode(blob_b64.encode("ascii")))

    model_type = str(model.get("type", ""))
    if model_type == "catboost_classifier":
        runtime_model = CatBoostClassifier()
    elif model_type == "catboost_regressor":
        runtime_model = CatBoostRegressor()
    elif model_type == "catboost_ranker":
        runtime_model = CatBoostRanker()
    else:
        raise ValueError(f"Unsupported CatBoost model type: {model_type}")
    runtime_model.load_model(str(path), format=str(model.get("format", "cbm") or "cbm"))
    _CATBOOST_LOADED_MODELS[digest] = runtime_model
    return runtime_model


def _predict_model_score(model: dict, x: np.ndarray) -> float:
    model_type = str(model.get("type", ""))
    if model_type == "logistic":
        w = np.asarray(model["weights"], dtype=float)
        b = float(model["bias"])
        return _sigmoid_scalar(float(x @ w + b))
    if model_type == "mlp":
        W1 = np.asarray(model["W1"], dtype=float)
        b1 = np.asarray(model["b1"], dtype=float)
        W2 = np.asarray(model["W2"], dtype=float)
        b2 = np.asarray(model["b2"], dtype=float)
        z1 = x @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = float((a1 @ W2 + b2).reshape(-1)[0])
        return _sigmoid_scalar(z2)
    if model_type in {"ridge", "pairwise_linear"}:
        w = np.asarray(model["weights"], dtype=float)
        b = float(model["bias"])
        return float(x @ w + b)
    if model_type == "constant":
        return float(model.get("value", 0.0))
    if model_type == "catboost_classifier":
        runtime_model = _load_catboost_runtime_model(model)
        proba = runtime_model.predict_proba(np.asarray([x], dtype=float))
        return float(np.asarray(proba, dtype=float).reshape(1, -1)[0, -1])
    if model_type in {"catboost_regressor", "catboost_ranker"}:
        runtime_model = _load_catboost_runtime_model(model)
        pred = runtime_model.predict(np.asarray([x], dtype=float))
        return float(np.asarray(pred, dtype=float).reshape(-1)[0])
    raise ValueError(f"Unsupported candidate-ranker model type: {model_type}")


def _apply_calibration(calibrator: Optional[dict], raw_score: float) -> float:
    if not calibrator:
        return float(max(0.0, min(1.0, raw_score)))
    if str(calibrator.get("type", "")) != "platt":
        return float(max(0.0, min(1.0, raw_score)))
    p = min(1.0 - 1e-5, max(1e-5, float(raw_score)))
    logit = math.log(p / (1.0 - p))
    z = float(calibrator.get("a", 1.0)) * logit + float(calibrator.get("b", 0.0))
    return _sigmoid_scalar(z)


def predict_components_from_candidate_payload(payload: dict, rec: dict) -> dict:
    x = _transform_payload_vector(payload, rec)
    payload_version = int(payload.get("payload_version", 1) or 1)
    threshold = float(payload.get("threshold", 0.5))
    if payload_version < 2 or "quality_model" not in payload:
        quality = _predict_model_score(payload["model"], x)
        return {
            "quality_proba": float(quality),
            "expected_return": 0.0,
            "expected_drawdown": 0.0,
            "ev_raw": 0.0,
            "rank_score": 0.0,
            "rank_score_norm": 0.0,
            "final_score": float(quality - threshold),
        }

    quality_raw = _predict_model_score(payload["quality_model"], x)
    quality_proba = _apply_calibration(payload.get("quality_calibrator"), quality_raw)
    expected_return = _predict_model_score(payload["return_model"], x)
    expected_drawdown = max(0.0, _predict_model_score(payload["drawdown_model"], x))
    ev_lambda = float(payload.get("ev_lambda", DEFAULT_EV_LAMBDA))
    ev_raw = float(expected_return - ev_lambda * expected_drawdown)
    rank_score = _predict_model_score(payload["rank_model"], x)
    rank_mean = float(payload.get("rank_score_mean", 0.0))
    rank_std = max(1e-8, float(payload.get("rank_score_std", 1.0)))
    rank_score_norm = float((rank_score - rank_mean) / rank_std)
    quality_weight = float(payload.get("quality_score_weight", DEFAULT_QUALITY_SCORE_WEIGHT))
    rank_weight = float(payload.get("rank_score_weight", DEFAULT_RANK_SCORE_WEIGHT))
    top_gainer_prob = 0.0
    capture_ratio_pred = 0.0
    if payload_version >= 3 and "top_gainer_model" in payload and "capture_model" in payload:
        top_raw = _predict_model_score(payload["top_gainer_model"], x)
        top_gainer_prob = float(_apply_calibration(payload.get("top_gainer_calibrator"), top_raw))
        capture_ratio_pred = float(max(0.0, _predict_model_score(payload["capture_model"], x)))
    top_rate_mean = float(payload.get("top_gainer_rate_mean", 0.0))
    capture_mean = float(payload.get("capture_ratio_mean", 0.0))
    top_weight = float(payload.get("top_gainer_score_weight", DEFAULT_TOP_GAINER_SCORE_WEIGHT))
    capture_weight = float(payload.get("capture_score_weight", DEFAULT_CAPTURE_SCORE_WEIGHT))
    final_score = float(
        ev_raw
        + quality_weight * (quality_proba - threshold)
        + rank_weight * rank_score_norm
        + top_weight * (top_gainer_prob - top_rate_mean)
        + capture_weight * (capture_ratio_pred - capture_mean)
    )
    return {
        "quality_proba": float(quality_proba),
        "expected_return": float(expected_return),
        "expected_drawdown": float(expected_drawdown),
        "ev_raw": float(ev_raw),
        "rank_score": float(rank_score),
        "rank_score_norm": float(rank_score_norm),
        "top_gainer_prob": float(top_gainer_prob),
        "capture_ratio_pred": float(capture_ratio_pred),
        "final_score": float(final_score),
    }


def predict_proba_from_candidate_payload(payload: dict, rec: dict) -> float:
    return float(predict_components_from_candidate_payload(payload, rec)["quality_proba"])


def predict_final_score_from_candidate_payload(payload: dict, rec: dict) -> float:
    return float(predict_components_from_candidate_payload(payload, rec)["final_score"])


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


def _target_return(rec: dict) -> float:
    labels = rec.get("labels") or {}
    ret_5 = _safe_float(labels.get("ret_5"))
    trade_taken = bool(labels.get("trade_taken"))
    trade_exit = labels.get("trade_exit_pnl")
    if trade_taken and trade_exit is not None:
        return 0.6 * _safe_float(trade_exit) + 0.4 * ret_5
    return ret_5


def _available_returns(rec: dict) -> List[float]:
    labels = rec.get("labels") or {}
    values: List[float] = []
    for key in ("ret_3", "ret_5", "ret_10"):
        value = labels.get(key)
        if value is not None:
            values.append(_safe_float(value))
    exit_pnl = labels.get("trade_exit_pnl")
    if exit_pnl is not None:
        values.append(_safe_float(exit_pnl))
    return values


def _target_expected_return(rec: dict) -> float:
    labels = rec.get("labels") or {}
    weighted: List[tuple[float, float]] = []
    if labels.get("ret_3") is not None:
        weighted.append((0.20, _safe_float(labels.get("ret_3"))))
    if labels.get("ret_5") is not None:
        weighted.append((0.35, _safe_float(labels.get("ret_5"))))
    if labels.get("ret_10") is not None:
        weighted.append((0.15, _safe_float(labels.get("ret_10"))))
    if labels.get("trade_exit_pnl") is not None:
        weighted.append((0.30, _safe_float(labels.get("trade_exit_pnl"))))
    if not weighted:
        return _target_return(rec)
    total_w = sum(w for w, _ in weighted)
    return sum(w * v for w, v in weighted) / max(1e-9, total_w)


def _target_expected_drawdown(rec: dict) -> float:
    values = _available_returns(rec)
    if not values:
        return 0.0
    return max(0.0, -min(values))


def _target_ev(rec: dict, ev_lambda: float = DEFAULT_EV_LAMBDA) -> float:
    return _target_expected_return(rec) - ev_lambda * _target_expected_drawdown(rec)


def _target_trade_quality(rec: dict, positive_ret_threshold: float = 0.0, ev_lambda: float = DEFAULT_EV_LAMBDA) -> float:
    return 1.0 if _target_ev(rec, ev_lambda=ev_lambda) > positive_ret_threshold else 0.0


def _teacher_payload(rec: dict) -> dict:
    teacher = rec.get("teacher") or {}
    if isinstance(teacher.get("final"), dict):
        return teacher["final"]
    if isinstance(teacher.get("midday"), dict):
        return teacher["midday"]
    return {}


def _teacher_present(rec: dict) -> bool:
    return bool(_teacher_payload(rec))


def _target_teacher_top_gainer(rec: dict) -> float:
    teacher = _teacher_payload(rec)
    return 1.0 if teacher.get("watchlist_top_gainer") else 0.0


def _target_teacher_capture_ratio(rec: dict) -> float:
    teacher = _teacher_payload(rec)
    value = teacher.get("capture_ratio")
    if value is None:
        return 0.0
    return max(0.0, min(1.5, _safe_float(value)))


def _decision_group_key(rec: dict) -> str:
    return str(rec.get("ts_signal") or rec.get("bar_ts") or "")


def build_dataset(
    rows: List[dict],
    positive_ret_threshold: float = 0.0,
    ev_lambda: float = DEFAULT_EV_LAMBDA,
) -> DatasetBundle:
    feature_names = safe_feature_names()
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    r = np.zeros(len(rows), dtype=float)
    er = np.zeros(len(rows), dtype=float)
    dd = np.zeros(len(rows), dtype=float)
    tg = np.zeros(len(rows), dtype=float)
    cap = np.zeros(len(rows), dtype=float)
    tmask = np.zeros(len(rows), dtype=float)

    for idx, rec in enumerate(rows):
        X[idx] = vectorize_record(rec, feature_names)
        er[idx] = _target_expected_return(rec)
        dd[idx] = _target_expected_drawdown(rec)
        target_ev = er[idx] - ev_lambda * dd[idx]
        r[idx] = target_ev
        y[idx] = _target_trade_quality(rec, positive_ret_threshold=positive_ret_threshold, ev_lambda=ev_lambda)
        if _teacher_present(rec):
            tmask[idx] = 1.0
            tg[idx] = _target_teacher_top_gainer(rec)
            cap[idx] = _target_teacher_capture_ratio(rec)

    n = len(rows)
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1) if n > 2 else n

    return DatasetBundle(
        feature_names=feature_names,
        X_train=X[:train_end],
        y_train=y[:train_end],
        r_train=r[:train_end],
        er_train=er[:train_end],
        dd_train=dd[:train_end],
        tg_train=tg[:train_end],
        cap_train=cap[:train_end],
        tmask_train=tmask[:train_end],
        X_val=X[train_end:val_end],
        y_val=y[train_end:val_end],
        r_val=r[train_end:val_end],
        er_val=er[train_end:val_end],
        dd_val=dd[train_end:val_end],
        tg_val=tg[train_end:val_end],
        cap_val=cap[train_end:val_end],
        tmask_val=tmask[train_end:val_end],
        X_test=X[val_end:],
        y_test=y[val_end:],
        r_test=r[val_end:],
        er_test=er[val_end:],
        dd_test=dd[val_end:],
        tg_test=tg[val_end:],
        cap_test=cap[val_end:],
        tmask_test=tmask[val_end:],
        meta_train=rows[:train_end],
        meta_val=rows[train_end:val_end],
        meta_test=rows[val_end:],
    )


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        if X.shape[0] == 0:
            self.w = np.zeros(X.shape[1], dtype=float)
            self.b = 0.0
            return self
        ones = np.ones((X.shape[0], 1), dtype=float)
        Xb = np.hstack([X, ones])
        reg = np.eye(Xb.shape[1], dtype=float) * self.alpha
        reg[-1, -1] = 0.0
        beta = np.linalg.pinv(Xb.T @ Xb + reg) @ (Xb.T @ y)
        self.w = beta[:-1]
        self.b = float(beta[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return X @ self.w + self.b

    def to_dict(self) -> dict:
        return {
            "type": "ridge",
            "alpha": self.alpha,
            "weights": (self.w.tolist() if self.w is not None else []),
            "bias": self.b,
        }


class ConstantScoreModel:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def fit(self, *_args, **_kwargs) -> "ConstantScoreModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=float)

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=float)

    def to_dict(self) -> dict:
        return {"type": "constant", "value": float(self.value)}


class CatBoostBinaryClassifier:
    def __init__(self) -> None:
        if not CATBOOST_AVAILABLE:
            raise RuntimeError(f"CatBoost is not available: {CATBOOST_IMPORT_ERROR}")
        self.model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            l2_leaf_reg=CATBOOST_L2_LEAF_REG,
            random_seed=CATBOOST_RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "CatBoostBinaryClassifier":
        eval_set = (X_val, y_val) if X_val.shape[0] > 0 else None
        self.model.fit(X, y, eval_set=eval_set, use_best_model=bool(eval_set))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=float)[:, -1]

    def to_dict(self) -> dict:
        return _catboost_blob_to_dict(self.model, model_type="catboost_classifier")


class CatBoostValueRegressor:
    def __init__(self) -> None:
        if not CATBOOST_AVAILABLE:
            raise RuntimeError(f"CatBoost is not available: {CATBOOST_IMPORT_ERROR}")
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            l2_leaf_reg=CATBOOST_L2_LEAF_REG,
            random_seed=CATBOOST_RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "CatBoostValueRegressor":
        eval_set = (X_val, y_val) if X_val.shape[0] > 0 else None
        self.model.fit(X, y, eval_set=eval_set, use_best_model=bool(eval_set))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def to_dict(self) -> dict:
        return _catboost_blob_to_dict(self.model, model_type="catboost_regressor")


class CatBoostGroupRanker:
    def __init__(self) -> None:
        if not CATBOOST_AVAILABLE:
            raise RuntimeError(f"CatBoost is not available: {CATBOOST_IMPORT_ERROR}")
        self.model = CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=3",
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            l2_leaf_reg=CATBOOST_L2_LEAF_REG,
            random_seed=CATBOOST_RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_id: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        group_id_val: np.ndarray,
    ) -> "CatBoostGroupRanker":
        train_pool = Pool(X, label=y, group_id=group_id)
        eval_pool = Pool(X_val, label=y_val, group_id=group_id_val) if X_val.shape[0] > 0 else None
        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=bool(eval_pool))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def to_dict(self) -> dict:
        return _catboost_blob_to_dict(self.model, model_type="catboost_ranker")


class PlattCalibrator:
    def __init__(self, lr: float = 0.05, epochs: int = 300) -> None:
        self.a = 1.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if raw_scores.size == 0 or len(np.unique(y)) < 2:
            self.a = 1.0
            self.b = 0.0
            return self
        p = np.clip(raw_scores.astype(float), 1e-5, 1.0 - 1e-5)
        x = np.log(p / (1.0 - p))
        n = max(1, x.shape[0])
        for _ in range(self.epochs):
            z = self.a * x + self.b
            pred = self._sigmoid(z)
            err = pred - y
            grad_a = float((err * x).sum() / n)
            grad_b = float(err.mean())
            self.a -= self.lr * grad_a
            self.b -= self.lr * grad_b
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        p = np.clip(raw_scores.astype(float), 1e-5, 1.0 - 1e-5)
        x = np.log(p / (1.0 - p))
        return self._sigmoid(self.a * x + self.b)

    def to_dict(self) -> dict:
        return {"type": "platt", "a": float(self.a), "b": float(self.b)}


class PairwiseLinearRanker:
    def __init__(self, n_features: int, lr: float = 0.03, epochs: int = 120, l2: float = 1e-4) -> None:
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, utility: np.ndarray, groups: Sequence[Sequence[int]]) -> "PairwiseLinearRanker":
        if X.shape[0] == 0:
            return self
        for _ in range(self.epochs):
            grad_w = np.zeros_like(self.w)
            grad_b = 0.0
            total_weight = 0.0
            for group in groups:
                idx = list(group)
                if len(idx) < 2:
                    continue
                for a in range(len(idx)):
                    ia = idx[a]
                    for b in range(a + 1, len(idx)):
                        ib = idx[b]
                        diff_u = float(utility[ia] - utility[ib])
                        if abs(diff_u) < 1e-8:
                            continue
                        if diff_u > 0:
                            xdiff = X[ia] - X[ib]
                        else:
                            xdiff = X[ib] - X[ia]
                            diff_u = -diff_u
                        z = float(xdiff @ self.w + self.b)
                        p = float(self._sigmoid(np.array([z]))[0])
                        weight = max(0.05, min(3.0, abs(diff_u)))
                        err = (p - 1.0) * weight
                        grad_w += err * xdiff
                        grad_b += err
                        total_weight += weight
            if total_weight <= 0:
                break
            grad_w = grad_w / total_weight + self.l2 * self.w
            grad_b = grad_b / total_weight
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def to_dict(self) -> dict:
        return {"type": "pairwise_linear", "weights": self.w.tolist(), "bias": float(self.b)}


def _train_teacher_top_model(bundle: DatasetBundle, X_train: np.ndarray, X_val: np.ndarray) -> tuple[Any, Optional[PlattCalibrator], float]:
    mask_train = bundle.tmask_train > 0.5
    mask_val = bundle.tmask_val > 0.5
    train_rate = float(np.mean(bundle.tg_train[mask_train])) if np.any(mask_train) else 0.0
    if (not CATBOOST_AVAILABLE) or np.sum(mask_train) < 20 or len(np.unique(bundle.tg_train[mask_train])) < 2:
        return ConstantScoreModel(train_rate), None, train_rate

    model = CatBoostBinaryClassifier().fit(
        X_train[mask_train],
        bundle.tg_train[mask_train],
        X_val[mask_val] if np.any(mask_val) else X_val[:0],
        bundle.tg_val[mask_val] if np.any(mask_val) else bundle.tg_val[:0],
    )
    calibrator: Optional[PlattCalibrator] = None
    if np.sum(mask_val) >= 10 and len(np.unique(bundle.tg_val[mask_val])) >= 2:
        raw = model.predict_proba(X_val[mask_val])
        calibrator = PlattCalibrator().fit(raw, bundle.tg_val[mask_val])
    return model, calibrator, train_rate


def _train_teacher_capture_model(bundle: DatasetBundle, X_train: np.ndarray, X_val: np.ndarray) -> tuple[Any, float]:
    mask_train = bundle.tmask_train > 0.5
    mask_val = bundle.tmask_val > 0.5
    capture_mean = float(np.mean(bundle.cap_train[mask_train])) if np.any(mask_train) else 0.0
    if (not CATBOOST_AVAILABLE) or np.sum(mask_train) < 20:
        return ConstantScoreModel(capture_mean), capture_mean

    model = CatBoostValueRegressor().fit(
        X_train[mask_train],
        bundle.cap_train[mask_train],
        X_val[mask_val] if np.any(mask_val) else X_val[:0],
        bundle.cap_val[mask_val] if np.any(mask_val) else bundle.cap_val[:0],
    )
    return model, capture_mean


def _group_indices(meta_rows: Sequence[dict]) -> List[List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(meta_rows):
        groups[_decision_group_key(rec)].append(idx)
    out = [group for _, group in sorted(groups.items(), key=lambda item: item[0]) if len(group) >= 2]
    return out


def _group_id_array(meta_rows: Sequence[dict]) -> np.ndarray:
    mapping: Dict[str, int] = {}
    out: List[int] = []
    next_id = 0
    for rec in meta_rows:
        key = _decision_group_key(rec)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out.append(mapping[key])
    return np.asarray(out, dtype=np.int32)


def _selection_metrics(rows: Sequence[dict]) -> dict:
    target_returns = [_target_return(r) for r in rows]
    ret5 = [_safe_float((r.get("labels") or {}).get("ret_5")) for r in rows]
    ev_values = [_target_ev(r) for r in rows]
    drawdowns = [_target_expected_drawdown(r) for r in rows]
    teacher_rows = [r for r in rows if _teacher_present(r)]
    teacher_top = [_target_teacher_top_gainer(r) for r in teacher_rows]
    teacher_capture = [_target_teacher_capture_ratio(r) for r in teacher_rows]
    exit_pnls = [
        _safe_float((r.get("labels") or {}).get("trade_exit_pnl"))
        for r in rows
        if (r.get("labels") or {}).get("trade_exit_pnl") is not None
    ]
    return {
        "count": len(rows),
        "avg_target_return": round(float(np.mean(target_returns)) if target_returns else 0.0, 4),
        "avg_ret5": round(float(np.mean(ret5)) if ret5 else 0.0, 4),
        "avg_ev": round(float(np.mean(ev_values)) if ev_values else 0.0, 4),
        "avg_drawdown": round(float(np.mean(drawdowns)) if drawdowns else 0.0, 4),
        "avg_exit_pnl": round(float(np.mean(exit_pnls)) if exit_pnls else 0.0, 4),
        "win_rate": round(float(np.mean([1.0 if v > 0.0 else 0.0 for v in target_returns])) if target_returns else 0.0, 4),
        "teacher_rows": len(teacher_rows),
        "teacher_top_gainer_rate": round(float(np.mean(teacher_top)) if teacher_top else 0.0, 4),
        "teacher_capture_ratio": round(float(np.mean(teacher_capture)) if teacher_capture else 0.0, 4),
    }


def _evaluate_grouped_ranking(
    meta_rows: Sequence[dict],
    ranker_scores: np.ndarray,
    *,
    top_ns: Sequence[int] = (1, 3, 5),
) -> dict:
    groups = _group_indices(meta_rows)
    top_reports: List[dict] = []
    for top_n in top_ns:
        eligible = [group for group in groups if len(group) >= top_n]
        baseline_selected: List[dict] = []
        ranker_selected: List[dict] = []
        overlap_total = 0
        for group in eligible:
            baseline = sorted(
                group,
                key=lambda idx: (
                    _safe_float((meta_rows[idx].get("decision") or {}).get("candidate_score")),
                    _safe_float((meta_rows[idx].get("decision") or {}).get("forecast_return_pct")),
                    str(meta_rows[idx].get("sym", "")),
                ),
                reverse=True,
            )[:top_n]
            ranker = sorted(
                group,
                key=lambda idx: (
                    float(ranker_scores[idx]),
                    _safe_float((meta_rows[idx].get("decision") or {}).get("candidate_score")),
                    str(meta_rows[idx].get("sym", "")),
                ),
                reverse=True,
            )[:top_n]
            baseline_selected.extend(meta_rows[idx] for idx in baseline)
            ranker_selected.extend(meta_rows[idx] for idx in ranker)
            overlap_total += len(set(baseline) & set(ranker))

        baseline_metrics = _selection_metrics(baseline_selected)
        ranker_metrics = _selection_metrics(ranker_selected)
        top_reports.append(
            {
                "top_n": int(top_n),
                "eligible_groups": len(eligible),
                "baseline": baseline_metrics,
                "ranker": ranker_metrics,
                "delta": {
                    "avg_target_return": round(ranker_metrics["avg_target_return"] - baseline_metrics["avg_target_return"], 4),
                    "avg_ret5": round(ranker_metrics["avg_ret5"] - baseline_metrics["avg_ret5"], 4),
                    "avg_ev": round(ranker_metrics["avg_ev"] - baseline_metrics["avg_ev"], 4),
                    "avg_drawdown": round(ranker_metrics["avg_drawdown"] - baseline_metrics["avg_drawdown"], 4),
                    "win_rate": round(ranker_metrics["win_rate"] - baseline_metrics["win_rate"], 4),
                    "teacher_top_gainer_rate": round(ranker_metrics["teacher_top_gainer_rate"] - baseline_metrics["teacher_top_gainer_rate"], 4),
                    "teacher_capture_ratio": round(ranker_metrics["teacher_capture_ratio"] - baseline_metrics["teacher_capture_ratio"], 4),
                },
                "overlap_ratio": round(overlap_total / max(1, len(eligible) * top_n), 4),
            }
        )

    head_to_head = {"wins": 0, "losses": 0, "ties": 0, "eligible_groups": len(groups)}
    for group in groups:
        baseline_idx = sorted(
            group,
            key=lambda idx: (
                _safe_float((meta_rows[idx].get("decision") or {}).get("candidate_score")),
                str(meta_rows[idx].get("sym", "")),
            ),
            reverse=True,
        )[0]
        ranker_idx = sorted(
            group,
            key=lambda idx: (
                float(ranker_scores[idx]),
                _safe_float((meta_rows[idx].get("decision") or {}).get("candidate_score")),
            ),
            reverse=True,
        )[0]
        delta = _target_return(meta_rows[ranker_idx]) - _target_return(meta_rows[baseline_idx])
        if delta > 0:
            head_to_head["wins"] += 1
        elif delta < 0:
            head_to_head["losses"] += 1
        else:
            head_to_head["ties"] += 1

    return {
        "grouped_competitions": len(groups),
        "top_n": top_reports,
        "top1_head_to_head": head_to_head,
    }


def _calibration_report(probas: np.ndarray, target_quality: np.ndarray, utility: np.ndarray, buckets: int = 5) -> List[dict]:
    if probas.size == 0:
        return []
    order = np.argsort(-probas)
    sorted_p = probas[order]
    sorted_y = target_quality[order]
    sorted_u = utility[order]
    bucket_count = min(buckets, max(1, len(sorted_p)))
    bucket_size = max(1, len(sorted_p) // bucket_count)
    out: List[dict] = []
    for idx in range(bucket_count):
        lo = idx * bucket_size
        hi = len(sorted_p) if idx == bucket_count - 1 else min(len(sorted_p), (idx + 1) * bucket_size)
        p = sorted_p[lo:hi]
        y = sorted_y[lo:hi]
        u = sorted_u[lo:hi]
        if p.size == 0:
            continue
        out.append(
            {
                "bucket": idx + 1,
                "rows": int(p.size),
                "proba_min": round(float(np.min(p)), 4),
                "proba_max": round(float(np.max(p)), 4),
                "avg_quality": round(float(np.mean(y)), 4),
                "avg_target_return": round(float(np.mean(u)), 4),
            }
        )
    return out


def train_and_evaluate(
    dataset_path: Path,
    positive_ret_threshold: float = 0.0,
    min_ts: Optional[datetime] = None,
    min_rows: int = 500,
    ev_lambda: float = DEFAULT_EV_LAMBDA,
    require_catboost: bool = True,
) -> dict:
    if require_catboost and not CATBOOST_AVAILABLE:
        detail = CATBOOST_IMPORT_ERROR or "unknown import error"
        raise RuntimeError(f"CatBoost is required for candidate ranker training but unavailable: {detail}")
    rows = load_training_rows(dataset_path, min_ts=min_ts)
    if len(rows) < min_rows:
        raise RuntimeError("Not enough labeled candidate rows for ranker training")

    bundle = build_dataset(rows, positive_ret_threshold=positive_ret_threshold, ev_lambda=ev_lambda)
    scaler = StandardScaler().fit(bundle.X_train)
    X_train = scaler.transform(bundle.X_train)
    X_val = scaler.transform(bundle.X_val)
    X_test = scaler.transform(bundle.X_test)
    train_groups = _group_indices(bundle.meta_train)
    train_group_ids = _group_id_array(bundle.meta_train)
    val_group_ids = _group_id_array(bundle.meta_val)

    if CATBOOST_AVAILABLE:
        return_model = CatBoostValueRegressor().fit(X_train, bundle.er_train, X_val, bundle.er_val)
        drawdown_model = CatBoostValueRegressor().fit(X_train, bundle.dd_train, X_val, bundle.dd_val)
        rank_model = CatBoostGroupRanker().fit(
            X_train,
            bundle.r_train,
            train_group_ids,
            X_val,
            bundle.r_val,
            val_group_ids,
        )
        regression_backend = "catboost"
        ranking_backend = "catboost"
    else:
        return_model = RidgeRegressor(alpha=2.0).fit(X_train, bundle.er_train)
        drawdown_model = RidgeRegressor(alpha=2.0).fit(X_train, bundle.dd_train)
        rank_model = PairwiseLinearRanker(X_train.shape[1]).fit(X_train, bundle.r_train, train_groups)
        regression_backend = "ridge"
        ranking_backend = "pairwise_linear"
    top_gainer_model, top_gainer_calibrator, top_gainer_rate_mean = _train_teacher_top_model(bundle, X_train, X_val)
    capture_model, capture_ratio_mean = _train_teacher_capture_model(bundle, X_train, X_val)
    rank_train_scores = rank_model.score(X_train)
    rank_score_mean = float(np.mean(rank_train_scores)) if rank_train_scores.size else 0.0
    rank_score_std = max(1e-8, float(np.std(rank_train_scores)) if rank_train_scores.size else 1.0)

    models: Dict[str, Any] = {}
    if require_catboost:
        models["catboost"] = CatBoostBinaryClassifier().fit(X_train, bundle.y_train, X_val, bundle.y_val)
    else:
        models = {
            "logistic": LogisticModel(X_train.shape[1]).fit(X_train, bundle.y_train),
            "mlp": MLPModel(X_train.shape[1]).fit(X_train, bundle.y_train),
        }
        if CATBOOST_AVAILABLE:
            models["catboost"] = CatBoostBinaryClassifier().fit(X_train, bundle.y_train, X_val, bundle.y_val)

    validation: Dict[str, dict] = {}
    best_name = ""
    best_threshold = 0.5
    best_calibrator: Optional[PlattCalibrator] = None
    best_score = -1e9
    for name, model in models.items():
        val_raw = model.predict_proba(X_val)
        calibrator = PlattCalibrator().fit(val_raw, bundle.y_val)
        val_score = calibrator.predict(val_raw)
        threshold, quality_metrics = find_best_threshold(bundle.y_val, val_score, bundle.r_val)
        val_return = return_model.predict(X_val)
        val_drawdown = np.maximum(0.0, drawdown_model.predict(X_val))
        val_ev = val_return - ev_lambda * val_drawdown
        val_rank = (rank_model.score(X_val) - rank_score_mean) / rank_score_std
        val_top_raw = top_gainer_model.predict_proba(X_val) if hasattr(top_gainer_model, "predict_proba") else top_gainer_model.predict(X_val)
        if top_gainer_calibrator is not None:
            val_top = top_gainer_calibrator.predict(np.asarray(val_top_raw, dtype=float))
        else:
            val_top = np.asarray(val_top_raw, dtype=float)
        val_capture = np.maximum(0.0, np.asarray(capture_model.predict(X_val), dtype=float))
        val_final = (
            val_ev
            + DEFAULT_QUALITY_SCORE_WEIGHT * (val_score - threshold)
            + DEFAULT_RANK_SCORE_WEIGHT * val_rank
            + DEFAULT_TOP_GAINER_SCORE_WEIGHT * (val_top - top_gainer_rate_mean)
            + DEFAULT_CAPTURE_SCORE_WEIGHT * (val_capture - capture_ratio_mean)
        )
        grouped = _evaluate_grouped_ranking(bundle.meta_val, val_final, top_ns=(1, 3))
        validation[name] = {
            "quality": quality_metrics,
            "group_ranking": grouped,
        }
        top1 = grouped["top_n"][0] if grouped["top_n"] else {"ranker": {"avg_target_return": 0.0, "win_rate": 0.0}}
        top3 = grouped["top_n"][1] if len(grouped["top_n"]) > 1 else {"delta": {"avg_target_return": 0.0}}
        score = (
            top1["ranker"]["avg_target_return"]
            + 0.35 * top1["ranker"]["win_rate"]
            + 0.25 * top3["delta"]["avg_target_return"]
            + 0.25 * top1.get("delta", {}).get("teacher_top_gainer_rate", 0.0)
            + 0.20 * top1.get("delta", {}).get("teacher_capture_ratio", 0.0)
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_threshold = threshold
            best_calibrator = calibrator

    best_model = models[best_name]
    assert best_calibrator is not None
    test_raw_score = best_model.predict_proba(X_test)
    test_score = best_calibrator.predict(test_raw_score)
    baseline = rule_baseline_metrics(bundle.r_test, bundle.y_test)
    filtered = evaluate_predictions(bundle.y_test, test_score, bundle.r_test, best_threshold)
    test_return = return_model.predict(X_test)
    test_drawdown = np.maximum(0.0, drawdown_model.predict(X_test))
    test_ev = test_return - ev_lambda * test_drawdown
    test_rank = (rank_model.score(X_test) - rank_score_mean) / rank_score_std
    test_top_raw = top_gainer_model.predict_proba(X_test) if hasattr(top_gainer_model, "predict_proba") else top_gainer_model.predict(X_test)
    if top_gainer_calibrator is not None:
        test_top = top_gainer_calibrator.predict(np.asarray(test_top_raw, dtype=float))
    else:
        test_top = np.asarray(test_top_raw, dtype=float)
    test_capture = np.maximum(0.0, np.asarray(capture_model.predict(X_test), dtype=float))
    test_final = (
        test_ev
        + DEFAULT_QUALITY_SCORE_WEIGHT * (test_score - best_threshold)
        + DEFAULT_RANK_SCORE_WEIGHT * test_rank
        + DEFAULT_TOP_GAINER_SCORE_WEIGHT * (test_top - top_gainer_rate_mean)
        + DEFAULT_CAPTURE_SCORE_WEIGHT * (test_capture - capture_ratio_mean)
    )
    grouped_test = _evaluate_grouped_ranking(bundle.meta_test, test_final, top_ns=(1, 3, 5))
    importances = permutation_importance(
        best_model,
        X_test,
        bundle.y_test,
        bundle.r_test,
        bundle.feature_names,
        best_threshold,
    )
    top1_test = grouped_test["top_n"][0] if grouped_test["top_n"] else {
        "baseline": {"avg_target_return": 0.0, "avg_ret5": 0.0, "avg_ev": 0.0, "win_rate": 0.0},
        "ranker": {"avg_target_return": 0.0, "avg_ret5": 0.0, "avg_ev": 0.0, "win_rate": 0.0},
        "delta": {"avg_target_return": 0.0, "avg_ret5": 0.0, "avg_ev": 0.0, "win_rate": 0.0},
    }
    model_payload = {
        "payload_version": 3,
        "model_name": best_name,
        "available_model_families": {
            "logistic": not require_catboost,
            "mlp": not require_catboost,
            "catboost": CATBOOST_AVAILABLE,
        },
        "regression_backend": regression_backend,
        "ranking_backend": ranking_backend,
        "feature_names": bundle.feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": best_threshold,
        "positive_ret_threshold": positive_ret_threshold,
        "ev_lambda": ev_lambda,
        "quality_score_weight": DEFAULT_QUALITY_SCORE_WEIGHT,
        "rank_score_weight": DEFAULT_RANK_SCORE_WEIGHT,
        "top_gainer_score_weight": DEFAULT_TOP_GAINER_SCORE_WEIGHT,
        "capture_score_weight": DEFAULT_CAPTURE_SCORE_WEIGHT,
        "rank_score_mean": rank_score_mean,
        "rank_score_std": rank_score_std,
        "top_gainer_rate_mean": top_gainer_rate_mean,
        "capture_ratio_mean": capture_ratio_mean,
        "model": best_model.to_dict(),
        "quality_model": best_model.to_dict(),
        "quality_calibrator": best_calibrator.to_dict(),
        "return_model": return_model.to_dict(),
        "drawdown_model": drawdown_model.to_dict(),
        "rank_model": rank_model.to_dict(),
        "top_gainer_model": top_gainer_model.to_dict(),
        "top_gainer_calibrator": None if top_gainer_calibrator is None else top_gainer_calibrator.to_dict(),
        "capture_model": capture_model.to_dict(),
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
        "catboost_available": CATBOOST_AVAILABLE,
        "catboost_import_error": CATBOOST_IMPORT_ERROR,
        "require_catboost": require_catboost,
        "positive_ret_threshold": positive_ret_threshold,
        "ev_lambda": ev_lambda,
        "top_gainer_score_weight": DEFAULT_TOP_GAINER_SCORE_WEIGHT,
        "capture_score_weight": DEFAULT_CAPTURE_SCORE_WEIGHT,
        "chosen_model": best_name,
        "regression_backend": regression_backend,
        "ranking_backend": ranking_backend,
        "validation": validation,
        "test_baseline_candidates": baseline,
        "test_ranker": filtered,
        "test_group_ranking": grouped_test,
        "test_calibration": _calibration_report(test_score, bundle.y_test, bundle.r_test),
        "improvement_delta": {
            "ret5_avg_delta": round(top1_test["delta"]["avg_ret5"], 4),
            "win_rate_delta": round(top1_test["delta"]["win_rate"], 4),
            "coverage_delta": round(filtered["coverage"] - baseline["coverage"], 4),
            "ev_avg_delta": round(top1_test["delta"]["avg_ev"], 4),
            "target_return_delta": round(top1_test["delta"]["avg_target_return"], 4),
            "teacher_top_gainer_rate_delta": round(top1_test["delta"].get("teacher_top_gainer_rate", 0.0), 4),
            "teacher_capture_ratio_delta": round(top1_test["delta"].get("teacher_capture_ratio", 0.0), 4),
        },
        "teacher_label_coverage": {
            "train_rows": int(np.sum(bundle.tmask_train)),
            "val_rows": int(np.sum(bundle.tmask_val)),
            "test_rows": int(np.sum(bundle.tmask_test)),
            "train_top_gainer_rate": round(float(np.mean(bundle.tg_train[bundle.tmask_train > 0.5])) if np.any(bundle.tmask_train > 0.5) else 0.0, 4),
            "train_capture_ratio": round(float(np.mean(bundle.cap_train[bundle.tmask_train > 0.5])) if np.any(bundle.tmask_train > 0.5) else 0.0, 4),
        },
        "top_feature_importance": importances,
        "suggestions": suggestions,
        "model_payload": model_payload,
    }


def render_text(report: dict) -> str:
    top1 = ((report.get("test_group_ranking") or {}).get("top_n") or [{}])[0]
    lines = [
        "ML Candidate Ranker",
        f"Rows: {report['rows_total']} (train={report['train_rows']}, val={report['val_rows']}, test={report['test_rows']})",
        f"Chosen model: {report['chosen_model']}",
        f"Backends: regression={report.get('regression_backend', 'ridge')} ranking={report.get('ranking_backend', 'pairwise_linear')} catboost={report.get('catboost_available', False)}",
        f"Objective: EV = expected_return - {report.get('ev_lambda', DEFAULT_EV_LAMBDA):.2f} * expected_drawdown",
        f"Teacher weights: top_gainer={report.get('top_gainer_score_weight', DEFAULT_TOP_GAINER_SCORE_WEIGHT):.2f} capture={report.get('capture_score_weight', DEFAULT_CAPTURE_SCORE_WEIGHT):.2f}",
        "",
        "Validation:",
    ]
    for name, metrics in report["validation"].items():
        quality = metrics["quality"]
        group_top1 = metrics["group_ranking"]["top_n"][0] if metrics["group_ranking"]["top_n"] else {
            "ranker": {"avg_target_return": 0.0},
            "delta": {"avg_target_return": 0.0},
        }
        lines.append(
            f"  {name}: thr={quality['threshold']:.2f} f1={quality['f1']:.3f} "
            f"prec={quality['precision']:.3f} cov={quality['coverage']:.3f} "
            f"ret={quality['selected_ret5_avg']:+.4f}% "
            f"top1={group_top1['ranker']['avg_target_return']:+.4f}% "
            f"delta={group_top1['delta']['avg_target_return']:+.4f}% "
            f"tg={group_top1['delta'].get('teacher_top_gainer_rate', 0.0):+.4f} "
            f"cap={group_top1['delta'].get('teacher_capture_ratio', 0.0):+.4f}"
        )
    lines.extend(
        [
            "",
            "Teacher labels:",
            f"  train={report.get('teacher_label_coverage', {}).get('train_rows', 0)} "
            f"val={report.get('teacher_label_coverage', {}).get('val_rows', 0)} "
            f"test={report.get('teacher_label_coverage', {}).get('test_rows', 0)} "
            f"train_top_rate={report.get('teacher_label_coverage', {}).get('train_top_gainer_rate', 0.0):.4f} "
            f"train_capture={report.get('teacher_label_coverage', {}).get('train_capture_ratio', 0.0):.4f}",
            "",
            "Test quality filter:",
            f"  all candidates: cov={report['test_baseline_candidates']['coverage']:.3f} "
            f"wr={report['test_baseline_candidates']['selected_win_rate']:.3f} "
            f"ret={report['test_baseline_candidates']['selected_ret5_avg']:+.4f}%",
            f"  ranker: cov={report['test_ranker']['coverage']:.3f} "
            f"wr={report['test_ranker']['selected_win_rate']:.3f} "
            f"ret={report['test_ranker']['selected_ret5_avg']:+.4f}%",
            f"  delta: ret={report['test_ranker']['selected_ret5_avg'] - report['test_baseline_candidates']['selected_ret5_avg']:+.4f}% "
            f"wr={report['test_ranker']['selected_win_rate'] - report['test_baseline_candidates']['selected_win_rate']:+.4f} "
            f"cov={report['improvement_delta']['coverage_delta']:+.4f}",
            "",
            "Test grouped top-1:",
            f"  baseline: ev={top1.get('baseline', {}).get('avg_ev', 0.0):+.4f} "
            f"ret={top1.get('baseline', {}).get('avg_target_return', 0.0):+.4f}% "
            f"ret5={top1.get('baseline', {}).get('avg_ret5', 0.0):+.4f}% "
            f"wr={top1.get('baseline', {}).get('win_rate', 0.0):.3f}",
            f"  ranker: ev={top1.get('ranker', {}).get('avg_ev', 0.0):+.4f} "
            f"ret={top1.get('ranker', {}).get('avg_target_return', 0.0):+.4f}% "
            f"ret5={top1.get('ranker', {}).get('avg_ret5', 0.0):+.4f}% "
            f"wr={top1.get('ranker', {}).get('win_rate', 0.0):.3f}",
            f"  delta: ev={report['improvement_delta'].get('ev_avg_delta', 0.0):+.4f} "
            f"ret={report['improvement_delta'].get('target_return_delta', 0.0):+.4f}% "
            f"ret5={report['improvement_delta']['ret5_avg_delta']:+.4f}% "
            f"wr={report['improvement_delta']['win_rate_delta']:+.4f} "
            f"tg={report['improvement_delta'].get('teacher_top_gainer_rate_delta', 0.0):+.4f} "
            f"cap={report['improvement_delta'].get('teacher_capture_ratio_delta', 0.0):+.4f}",
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
    parser.add_argument("--ev-lambda", type=float, default=DEFAULT_EV_LAMBDA)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_FILE)
    parser.set_defaults(require_catboost=True)
    parser.add_argument("--require-catboost", dest="require_catboost", action="store_true")
    parser.add_argument("--allow-fallback", dest="require_catboost", action="store_false")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    min_ts = _parse_ts(args.min_date) if args.min_date else None
    report = train_and_evaluate(
        args.dataset,
        positive_ret_threshold=args.positive_ret_threshold,
        min_ts=min_ts,
        ev_lambda=args.ev_lambda,
        require_catboost=args.require_catboost,
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
