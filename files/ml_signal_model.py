from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "ml_dataset.jsonl"
DEFAULT_MODEL_FILE = ROOT / "ml_signal_model.json"
DEFAULT_REPORT_FILE = ROOT / "ml_signal_report.json"

SAFE_SCALAR_FEATURES = [
    "adx",
    "atr_pct",
    "body_pct",
    "btc_momentum_4h",
    "btc_vs_ema50",
    "close_vs_ema20",
    "close_vs_ema200",
    "close_vs_ema50",
    "daily_range",
    "ema20_vs_ema50",
    "ema50_vs_ema200",
    "lower_wick_pct",
    "macd_hist_norm",
    "market_vol_24h",
    "rsi",
    "slope",
    "upper_wick_pct",
    "vol_x",
]
LEAKY_FEATURES = {"r1", "r3", "r5", "r10"}
SIGNAL_TYPES = [
    "trend",
    "strong_trend",
    "retest",
    "breakout",
    "impulse_speed",
    "impulse",
    "alignment",
]
TIMEFRAME_VALUES = ["15m", "1h"]
SEQ_INDEX = {
    "close_norm": 0,
    "vol_x": 4,
    "slope": 5,
    "adx": 6,
    "rsi": 7,
    "macd_hist_norm": 8,
    "atr_pct": 9,
}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _safe_float(value: object) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _cyclical(value: int, period: int) -> Tuple[float, float]:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def _seq_feature_vector(seq: Sequence[Sequence[float]]) -> Dict[str, float]:
    arr = np.asarray(seq, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {}

    out: Dict[str, float] = {}
    for name, idx in SEQ_INDEX.items():
        col = arr[:, idx]
        out[f"seq_last_{name}"] = _safe_float(col[-1])
        out[f"seq_mean_{name}"] = _safe_float(col.mean())
        out[f"seq_trend_{name}"] = _safe_float(col[-1] - col[0])
        tail = col[-3:] if col.shape[0] >= 3 else col
        out[f"seq_tail_{name}"] = _safe_float(tail.mean())
    return out


def build_runtime_record(
    *,
    sym: str,
    tf: str,
    signal_type: str,
    is_bull_day: bool,
    bar_ts: int,
    feat: dict,
    data: np.ndarray,
    i: int,
    btc_vs_ema50: float = 0.0,
    btc_momentum_4h: float = 0.0,
    market_vol_24h: float = 0.0,
) -> dict:
    c_arr = data["c"].astype(float)
    o_arr = data["o"].astype(float)
    h_arr = data["h"].astype(float)
    l_arr = data["l"].astype(float)

    close_sig = _safe_float(c_arr[i])
    ema20 = _safe_float(feat["ema_fast"][i])
    ema50 = _safe_float(feat["ema_slow"][i])
    ema200 = _safe_float(feat["ema200"][i]) if "ema200" in feat else 0.0
    atr = _safe_float(feat["atr"][i])
    o_sig = _safe_float(o_arr[i])
    h_sig = _safe_float(h_arr[i])
    l_sig = _safe_float(l_arr[i])
    body = abs(close_sig - o_sig) / close_sig * 100 if close_sig > 0 else 0.0
    upper_wick = (h_sig - max(o_sig, close_sig)) / close_sig * 100 if close_sig > 0 else 0.0
    lower_wick = (min(o_sig, close_sig) - l_sig) / close_sig * 100 if close_sig > 0 else 0.0
    macd_norm = _safe_float(feat["macd_hist"][i]) / close_sig * 100 if close_sig > 0 else 0.0
    atr_pct = atr / close_sig * 100 if close_sig > 0 else 0.0

    scalar_f = {
        "close_vs_ema20": round((close_sig / ema20 - 1) * 100, 4) if ema20 > 0 else 0.0,
        "close_vs_ema50": round((close_sig / ema50 - 1) * 100, 4) if ema50 > 0 else 0.0,
        "close_vs_ema200": round((close_sig / ema200 - 1) * 100, 4) if ema200 > 0 else 0.0,
        "ema20_vs_ema50": round((ema20 / ema50 - 1) * 100, 4) if ema50 > 0 else 0.0,
        "ema50_vs_ema200": round((ema50 / ema200 - 1) * 100, 4) if ema200 > 0 else 0.0,
        "slope": round(_safe_float(feat["slope"][i]), 4),
        "rsi": round(_safe_float(feat["rsi"][i]), 2),
        "adx": round(_safe_float(feat["adx"][i]), 2),
        "vol_x": round(_safe_float(feat["vol_x"][i]), 4),
        "macd_hist_norm": round(macd_norm, 6),
        "atr_pct": round(atr_pct, 4),
        "daily_range": round(_safe_float(feat["daily_range_pct"][i]), 4),
        "body_pct": round(body, 4),
        "upper_wick_pct": round(upper_wick, 4),
        "lower_wick_pct": round(lower_wick, 4),
        "btc_vs_ema50": round(_safe_float(btc_vs_ema50), 4),
        "btc_momentum_4h": round(_safe_float(btc_momentum_4h), 4),
        "market_vol_24h": round(_safe_float(market_vol_24h), 4),
    }

    seq: List[List[float]] = []
    start = max(0, i - 20 + 1)
    for k in range(start, i + 1):
        c_n = _safe_float(c_arr[k]) / close_sig if close_sig > 0 else 1.0
        h_n = _safe_float(h_arr[k]) / close_sig if close_sig > 0 else 1.0
        l_n = _safe_float(l_arr[k]) / close_sig if close_sig > 0 else 1.0
        o_n = _safe_float(o_arr[k]) / close_sig if close_sig > 0 else 1.0
        seq.append([
            round(c_n, 6),
            round(h_n, 6),
            round(l_n, 6),
            round(o_n, 6),
            round(_safe_float(feat["vol_x"][k]), 4),
            round(_safe_float(feat["slope"][k]), 4),
            round(_safe_float(feat["adx"][k]), 2),
            round(_safe_float(feat["rsi"][k]), 2),
            round(_safe_float(feat["macd_hist"][k]) / close_sig * 100 if close_sig > 0 else 0.0, 6),
            round(_safe_float(feat["atr"][k]) / close_sig * 100 if close_sig > 0 else 0.0, 4),
        ])
    if len(seq) < 20:
        seq = [[0.0] * 10] * (20 - len(seq)) + seq

    dt = datetime.fromtimestamp(bar_ts / 1000)
    return {
        "sym": sym,
        "tf": tf,
        "signal_type": signal_type,
        "is_bull_day": bool(is_bull_day),
        "hour_utc": dt.hour,
        "dow": dt.weekday(),
        "f": scalar_f,
        "seq": seq,
    }


def safe_feature_names() -> List[str]:
    names: List[str] = []
    names.extend(SAFE_SCALAR_FEATURES)
    names.extend(["is_bull_day", "hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    names.extend([f"signal_{name}" for name in SIGNAL_TYPES])
    names.extend([f"tf_{name}" for name in TIMEFRAME_VALUES])
    seq_names = list(_seq_feature_vector(np.zeros((20, 10), dtype=float)).keys())
    names.extend(seq_names)
    return names


def build_feature_dict(rec: dict) -> Dict[str, float]:
    feat = rec.get("f") or {}
    out: Dict[str, float] = {}
    for name in SAFE_SCALAR_FEATURES:
        out[name] = _safe_float(feat.get(name))
    for bad in LEAKY_FEATURES:
        out.pop(bad, None)

    hour_sin, hour_cos = _cyclical(int(rec.get("hour_utc", 0)), 24)
    dow_sin, dow_cos = _cyclical(int(rec.get("dow", 0)), 7)
    out["is_bull_day"] = 1.0 if rec.get("is_bull_day") else 0.0
    out["hour_sin"] = hour_sin
    out["hour_cos"] = hour_cos
    out["dow_sin"] = dow_sin
    out["dow_cos"] = dow_cos

    signal_type = str(rec.get("signal_type", "none"))
    for name in SIGNAL_TYPES:
        out[f"signal_{name}"] = 1.0 if signal_type == name else 0.0

    tf = str(rec.get("tf", "15m"))
    for name in TIMEFRAME_VALUES:
        out[f"tf_{name}"] = 1.0 if tf == name else 0.0

    out.update(_seq_feature_vector(rec.get("seq") or []))
    return out


def vectorize_record(rec: dict, feature_names: Sequence[str]) -> np.ndarray:
    fmap = build_feature_dict(rec)
    return np.array([_safe_float(fmap.get(name)) for name in feature_names], dtype=float)


@dataclass
class DatasetBundle:
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    r_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    r_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    r_test: np.ndarray
    meta_test: List[dict]


def load_training_rows(path: Path, min_ts: Optional[datetime] = None) -> List[dict]:
    rows: List[dict] = []
    for rec in _iter_jsonl(path):
        signal_type = str(rec.get("signal_type", "none"))
        if signal_type == "none":
            continue
        labels = rec.get("labels") or {}
        ret_5 = labels.get("ret_5")
        if ret_5 is None:
            continue
        ts = _parse_ts(rec["ts_signal"])
        if min_ts and ts < min_ts:
            continue
        rec["_dt"] = ts
        rows.append(rec)
    rows.sort(key=lambda r: r["_dt"])
    return rows


def build_dataset(rows: List[dict], positive_ret_threshold: float = 0.0) -> DatasetBundle:
    feature_names = safe_feature_names()
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    r = np.zeros(len(rows), dtype=float)

    for idx, rec in enumerate(rows):
        fmap = build_feature_dict(rec)
        X[idx] = np.array([_safe_float(fmap.get(name)) for name in feature_names], dtype=float)
        ret_5 = _safe_float((rec.get("labels") or {}).get("ret_5"))
        r[idx] = ret_5
        y[idx] = 1.0 if ret_5 > positive_ret_threshold else 0.0

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


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_


class LogisticModel:
    def __init__(self, n_features: int, lr: float = 0.03, epochs: int = 400, l2: float = 1e-3) -> None:
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        n = max(1, X.shape[0])
        for _ in range(self.epochs):
            logits = X @ self.w + self.b
            p = self._sigmoid(logits)
            err = p - y
            grad_w = (X.T @ err) / n + self.l2 * self.w
            grad_b = float(err.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    def to_dict(self) -> dict:
        return {"type": "logistic", "weights": self.w.tolist(), "bias": self.b}


class MLPModel:
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 24,
        lr: float = 0.01,
        epochs: int = 250,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.15, size=(n_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2 = rng.normal(0.0, 0.15, size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        y2 = y.reshape(-1, 1)
        n = max(1, X.shape[0])
        for _ in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ self.W2 + self.b2
            p = self._sigmoid(z2)

            dz2 = p - y2
            dW2 = (a1.T @ dz2) / n + self.l2 * self.W2
            db2 = dz2.mean(axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0.0)
            dW1 = (X.T @ dz1) / n + self.l2 * self.W1
            db1 = dz1.mean(axis=0)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2).reshape(-1)

    def to_dict(self) -> dict:
        return {
            "type": "mlp",
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }


def predict_proba_from_payload(payload: dict, rec: dict) -> float:
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
        z2 = float(a1 @ W2 + b2)
        z2 = max(-35.0, min(35.0, z2))
        return float(1.0 / (1.0 + math.exp(-z2)))
    raise ValueError(f"Unsupported model type: {model['type']}")


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos = int(y_true.sum())
    neg = int((1.0 - y_true).sum())
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    sum_pos = ranks[y_true > 0.5].sum()
    auc = (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ret_5: np.ndarray,
    threshold: float,
) -> dict:
    pred = y_score >= threshold
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    total = max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    coverage = float(pred.mean()) if len(pred) else 0.0
    selected = ret_5[pred]
    selected_avg = float(selected.mean()) if selected.size else 0.0
    selected_win = float((selected > 0.0).mean()) if selected.size else 0.0
    brier = float(np.mean((y_score - y_true) ** 2)) if len(y_true) else 0.0
    auc = roc_auc_score_np(y_true, y_score)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": round(coverage, 4),
        "selected_count": int(pred.sum()),
        "selected_ret5_avg": round(selected_avg, 4),
        "selected_win_rate": round(selected_win, 4),
        "brier": round(brier, 4),
        "auc": None if auc is None else round(auc, 4),
    }


def rule_baseline_metrics(ret_5: np.ndarray, y_true: np.ndarray) -> dict:
    all_positive = np.ones_like(y_true, dtype=float)
    metrics = evaluate_predictions(y_true, all_positive, ret_5, threshold=0.5)
    metrics["coverage"] = 1.0
    metrics["selected_count"] = int(len(ret_5))
    return metrics


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, ret_5: np.ndarray) -> Tuple[float, dict]:
    best_thr = 0.5
    best_metrics = evaluate_predictions(y_true, y_score, ret_5, best_thr)
    best_score = -1e9
    for thr in np.linspace(0.35, 0.8, 19):
        metrics = evaluate_predictions(y_true, y_score, ret_5, float(thr))
        if metrics["selected_count"] < 25:
            continue
        score = metrics["selected_ret5_avg"] * (0.35 + metrics["coverage"])
        score += 0.15 * metrics["precision"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics
    return best_thr, best_metrics


def permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    ret_5: np.ndarray,
    feature_names: Sequence[str],
    threshold: float,
    top_k: int = 12,
) -> List[dict]:
    if X.shape[0] == 0:
        return []
    base_score = evaluate_predictions(y, model.predict_proba(X), ret_5, threshold)["selected_ret5_avg"]
    out: List[dict] = []
    rng = np.random.default_rng(42)
    for idx, name in enumerate(feature_names):
        perturbed = X.copy()
        rng.shuffle(perturbed[:, idx])
        score = evaluate_predictions(y, model.predict_proba(perturbed), ret_5, threshold)["selected_ret5_avg"]
        out.append({"feature": name, "importance": round(base_score - score, 6)})
    out.sort(key=lambda item: item["importance"], reverse=True)
    return out[:top_k]


def build_improvement_hints(rows: List[dict], importances: List[dict]) -> List[str]:
    hints: List[str] = []
    if not rows:
        return hints

    non_bull = [r for r in rows if not r.get("is_bull_day")]
    for signal in ("strong_trend", "trend", "retest", "breakout"):
        sig_rows = [r for r in non_bull if r.get("signal_type") == signal]
        if not sig_rows:
            continue
        ret5 = [_safe_float((r.get("labels") or {}).get("ret_5")) for r in sig_rows]
        win = sum(1 for x in ret5 if x > 0.0)
        loss = sum(1 for x in ret5 if x < -0.5)
        if len(ret5) >= 20 and win / len(ret5) < 0.45:
            hints.append(
                f"`{signal}` в non-bull режиме шумный: win-rate по ret_5 только {win / len(ret5):.1%}, "
                f"сильных провалов {loss / len(ret5):.1%}. Его стоит пропускать через отдельный quality-score."
            )

    important = {item["feature"] for item in importances[:6]}
    if {"btc_vs_ema50", "is_bull_day"} & important:
        hints.append(
            "Режим рынка вошёл в топ факторов. Для слабых дней стоит обучать и калибровать ML-слой отдельно от bullish-сессий."
        )
    if {"seq_trend_macd_hist_norm", "seq_tail_macd_hist_norm", "macd_hist_norm"} & important:
        hints.append(
            "Последовательная форма MACD важнее простого порога. Это хороший кандидат для ML-подтверждения раннего продолжения тренда."
        )
    if {"daily_range", "close_vs_ema20", "close_vs_ema50"} & important:
        hints.append(
            "Растянутость движения вошла в важные признаки. ML может лучше rule-based логики отсеивать поздние входы после уже зрелого выноса."
        )
    return hints


def _train_single_scope(
    rows: List[dict],
    positive_ret_threshold: float,
    min_rows: int,
) -> Optional[dict]:
    if len(rows) < min_rows:
        return None

    bundle = build_dataset(rows, positive_ret_threshold=positive_ret_threshold)
    if bundle.X_train.shape[0] < max(30, min_rows // 2) or bundle.X_test.shape[0] < 20:
        return None

    scaler = StandardScaler().fit(bundle.X_train)
    X_train = scaler.transform(bundle.X_train)
    X_val = scaler.transform(bundle.X_val)
    X_test = scaler.transform(bundle.X_test)

    model = LogisticModel(X_train.shape[1]).fit(X_train, bundle.y_train)
    val_score = model.predict_proba(X_val)
    threshold, validation = find_best_threshold(bundle.y_val, val_score, bundle.r_val)
    test_score = model.predict_proba(X_test)
    baseline = rule_baseline_metrics(bundle.r_test, bundle.y_test)
    filtered = evaluate_predictions(bundle.y_test, test_score, bundle.r_test, threshold)

    return {
        "rows_total": len(rows),
        "train_rows": int(bundle.X_train.shape[0]),
        "val_rows": int(bundle.X_val.shape[0]),
        "test_rows": int(bundle.X_test.shape[0]),
        "threshold": round(threshold, 4),
        "validation": validation,
        "baseline": baseline,
        "filtered": filtered,
        "delta": {
            "ret5_avg_delta": round(filtered["selected_ret5_avg"] - baseline["selected_ret5_avg"], 4),
            "win_rate_delta": round(filtered["selected_win_rate"] - baseline["selected_win_rate"], 4),
            "coverage_delta": round(filtered["coverage"] - baseline["coverage"], 4),
        },
        "model_payload": {
            "model_name": "logistic",
            "feature_names": bundle.feature_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "threshold": threshold,
            "positive_ret_threshold": positive_ret_threshold,
            "model": model.to_dict(),
        },
    }


def train_segment_models(
    rows: List[dict],
    positive_ret_threshold: float = 0.0,
    min_rows: int = 160,
) -> Dict[str, dict]:
    segments: Dict[str, List[dict]] = {}
    for rec in rows:
        key = f"{rec.get('signal_type')}|{'bull' if rec.get('is_bull_day') else 'nonbull'}"
        segments.setdefault(key, []).append(rec)

    reports: Dict[str, dict] = {}
    for key, segment_rows in sorted(segments.items()):
        rep = _train_single_scope(segment_rows, positive_ret_threshold, min_rows=min_rows)
        if rep is not None:
            reports[key] = rep
    return reports


def train_and_evaluate(
    dataset_path: Path,
    positive_ret_threshold: float = 0.0,
    min_ts: Optional[datetime] = None,
    min_rows: int = 500,
    segment_min_rows: int = 160,
) -> dict:
    rows = load_training_rows(dataset_path, min_ts=min_ts)
    if len(rows) < min_rows:
        raise RuntimeError("Not enough labeled signal rows for ML training")

    bundle = build_dataset(rows, positive_ret_threshold=positive_ret_threshold)
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

    suggestions = build_improvement_hints(rows, importances)
    suggestions.append(
        "Интегрировать модель сначала как quality-filter поверх текущих правил, а не как автономный торговый движок."
    )
    suggestions.append(
        "Использовать ML-score для ранжирования кандидатов при portfolio-full, особенно в non-bull режиме."
    )
    segment_reports = train_segment_models(
        rows,
        positive_ret_threshold=positive_ret_threshold,
        min_rows=segment_min_rows,
    )
    if segment_reports:
        suggestions.append(
            "Для следующей итерации перспективнее раздельные модели по signal_type x regime, чем один общий фильтр на все сигналы."
        )

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
        "test_baseline_rules": baseline,
        "test_rules_plus_ml": filtered,
        "improvement_delta": {
            "ret5_avg_delta": round(filtered["selected_ret5_avg"] - baseline["selected_ret5_avg"], 4),
            "win_rate_delta": round(filtered["selected_win_rate"] - baseline["selected_win_rate"], 4),
            "coverage_delta": round(filtered["coverage"] - baseline["coverage"], 4),
        },
        "segment_reports": segment_reports,
        "segment_model_payloads": {
            key: seg["model_payload"] for key, seg in segment_reports.items()
        },
        "top_feature_importance": importances,
        "suggestions": suggestions,
        "model_payload": model_payload,
    }


def render_text(report: dict) -> str:
    lines = [
        "ML Signal Model",
        f"Rows: {report['rows_total']} (train={report['train_rows']}, val={report['val_rows']}, test={report['test_rows']})",
        f"Chosen model: {report['chosen_model']}",
        "",
        "Validation:",
    ]
    for name, metrics in report["validation"].items():
        lines.append(
            f"  {name}: thr={metrics['threshold']:.2f} f1={metrics['f1']:.3f} "
            f"prec={metrics['precision']:.3f} cov={metrics['coverage']:.3f} "
            f"ret5={metrics['selected_ret5_avg']:+.4f}%"
        )
    lines.extend(
        [
            "",
            "Test comparison:",
            f"  rules only: cov={report['test_baseline_rules']['coverage']:.3f} "
            f"wr={report['test_baseline_rules']['selected_win_rate']:.3f} "
            f"ret5={report['test_baseline_rules']['selected_ret5_avg']:+.4f}%",
            f"  rules + ML: cov={report['test_rules_plus_ml']['coverage']:.3f} "
            f"wr={report['test_rules_plus_ml']['selected_win_rate']:.3f} "
            f"ret5={report['test_rules_plus_ml']['selected_ret5_avg']:+.4f}%",
            f"  delta: ret5={report['improvement_delta']['ret5_avg_delta']:+.4f}% "
            f"wr={report['improvement_delta']['win_rate_delta']:+.4f} "
            f"cov={report['improvement_delta']['coverage_delta']:+.4f}",
            "",
            "Top features:",
        ]
    )
    for item in report["top_feature_importance"]:
        lines.append(f"  {item['feature']}: {item['importance']:+.6f}")
    lines.append("")
    if report.get("segment_reports"):
        lines.append("Segment models:")
        for key, seg in sorted(report["segment_reports"].items()):
            delta = seg["delta"]
            lines.append(
                f"  {key}: rows={seg['rows_total']} ret5_delta={delta['ret5_avg_delta']:+.4f}% "
                f"wr_delta={delta['win_rate_delta']:+.4f} cov_delta={delta['coverage_delta']:+.4f}"
            )
        lines.append("")
    lines.append("Suggestions:")
    for item in report["suggestions"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_live_model_payload(report: dict) -> dict:
    payload = dict(report.get("model_payload", {}))
    segment_payloads = report.get("segment_model_payloads", {})
    if isinstance(segment_payloads, dict) and segment_payloads:
        payload["segment_model_payloads"] = segment_payloads
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate a baseline ML filter for crypto bot signals")
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
