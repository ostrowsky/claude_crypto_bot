"""Fast-reversal model inference (RM-3 Step 4 — guard + bandit wiring).

Loads the trained logistic regression model from `fast_reversal_model.json`
and exposes `predict_proba()` to compute `proba_fast_reversal` for a candidate.

If the model file is missing or malformed, `predict_proba()` returns None and
callers should treat that as "no signal" (do not block, do not penalise).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "fast_reversal_model.json"

_pylog = logging.getLogger("fast_reversal_model")
_LOCK = threading.RLock()
_MODEL_CACHE: Optional[Dict[str, Any]] = None
_MODEL_LOADED_MTIME: float = 0.0


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


def _load_model() -> Optional[Dict[str, Any]]:
    """Load model lazily; hot-reload if file mtime changed."""
    global _MODEL_CACHE, _MODEL_LOADED_MTIME
    with _LOCK:
        if not MODEL_FILE.exists():
            return None
        try:
            mtime = MODEL_FILE.stat().st_mtime
        except OSError:
            return None
        if _MODEL_CACHE is not None and mtime == _MODEL_LOADED_MTIME:
            return _MODEL_CACHE
        try:
            data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _pylog.warning("fast_reversal_model load failed: %s", e)
            return None
        if not all(k in data for k in ("weights", "bias", "means", "stds", "feature_names")):
            _pylog.warning("fast_reversal_model missing required fields")
            return None
        _MODEL_CACHE = data
        _MODEL_LOADED_MTIME = mtime
        return data


def is_available() -> bool:
    """True if a usable model is on disk."""
    return _load_model() is not None


def predict_proba(features: Dict[str, Any]) -> Optional[float]:
    """
    Predict P(fast_reversal) for a candidate.

    `features` is a dict keyed by FEATURE_NAMES values (vol_x, slope, adx,
    rsi, macd_hist_norm, atr_pct, candidate_score, ml_proba,
    forecast_return_pct, today_change_pct, btc_vs_ema50). Missing keys are
    treated as 0.0.

    Returns probability in [0, 1], or None if no model is loaded.
    """
    model = _load_model()
    if model is None:
        return None

    feature_names: List[str] = model["feature_names"]
    weights: List[float] = model["weights"]
    bias: float = float(model["bias"])
    means: List[float] = model["means"]
    stds: List[float] = model["stds"]

    if not (len(weights) == len(feature_names) == len(means) == len(stds)):
        return None

    z = bias
    for j, fname in enumerate(feature_names):
        v = _safe_float(features.get(fname))
        std = stds[j] if stds[j] > 1e-10 else 1.0
        x_norm = (v - means[j]) / std
        z += weights[j] * x_norm

    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def model_metadata() -> Optional[Dict[str, Any]]:
    """Return loaded model metadata (AUC, n_train, etc.) or None."""
    model = _load_model()
    if model is None:
        return None
    return {
        "model_type": model.get("model_type"),
        "auc_train": model.get("auc_train"),
        "auc_val": model.get("auc_val"),
        "trained_on": model.get("trained_on"),
        "label_positive_rate": model.get("label_positive_rate"),
    }
