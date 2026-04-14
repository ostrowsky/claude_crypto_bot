from __future__ import annotations

"""
Top Gainer Classifier — predicts which coins will be EOD top gainers.

This is the PRIMARY success metric model:
  "Bot gives early BUY signals on coins that end up in top gainers by 24:00"

Architecture:
  - Gradient boosting (CatBoost/XGBoost) ensemble
  - Rich feature set: morning momentum, order flow, derivatives, cross-asset
  - Daily label: binary (coin in top 20 gainers by 24h change at 00:00 UTC)
  - Multi-threshold output: P(top5), P(top10), P(top20), P(top50)
  - Confidence-weighted BUY score bonus

Training data:
  - Historical top gainer lists from Binance /ticker/24hr at 00:00 UTC
  - Features computed at various hours (2:00, 4:00, 6:00, ... UTC) = multi-snapshot
  - Backfill 6-12 months for stable model

Usage:
    model = TopGainerModel()
    model.load("top_gainer_model.json")
    # Every monitoring cycle:
    features = model.compute_features(sym, klines, indicators, order_flow, derivatives)
    prediction = model.predict(features)
    if prediction.is_likely_top_gainer:
        score_bonus = prediction.score_bonus  # add to entry score
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = Path(__file__).resolve().parent / "top_gainer_model.json"
DATASET_FILE = Path(__file__).resolve().parent / "top_gainer_dataset.jsonl"


@dataclass
class TopGainerPrediction:
    """Prediction output for a single symbol."""
    symbol: str
    timestamp_ms: int

    # Core predictions
    prob_top5: float          # P(in top 5 gainers by EOD)
    prob_top10: float         # P(in top 10)
    prob_top20: float         # P(in top 20)
    prob_top50: float         # P(in top 50)

    # Derived
    expected_eod_return: float  # Expected EOD return (%)
    confidence: float           # Model confidence [0, 1]

    # Feature importance (top 5 for interpretability)
    top_features: Dict[str, float] = field(default_factory=dict)

    @property
    def is_likely_top_gainer(self) -> bool:
        """High-confidence top 20 prediction."""
        return self.prob_top20 > 0.35 and self.confidence > 0.5

    @property
    def score_bonus(self) -> float:
        """Score bonus to add to entry decision."""
        if self.prob_top5 > 0.3:
            return 15.0
        if self.prob_top10 > 0.3:
            return 10.0
        if self.prob_top20 > 0.35:
            return 6.0
        if self.prob_top50 > 0.4:
            return 3.0
        return 0.0

    @property
    def position_size_mult(self) -> float:
        """Position size multiplier based on conviction."""
        if self.prob_top5 > 0.3:
            return 1.5
        if self.prob_top10 > 0.3:
            return 1.3
        if self.prob_top20 > 0.35:
            return 1.15
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "prob_top5": round(self.prob_top5, 4),
            "prob_top10": round(self.prob_top10, 4),
            "prob_top20": round(self.prob_top20, 4),
            "prob_top50": round(self.prob_top50, 4),
            "expected_eod_return": round(self.expected_eod_return, 2),
            "confidence": round(self.confidence, 3),
            "is_likely": self.is_likely_top_gainer,
            "score_bonus": self.score_bonus,
        }


# ── Feature definitions ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Morning momentum (computed at signal time)
    "tg_return_1h",               # Return last 1 hour (%)
    "tg_return_4h",               # Return last 4 hours (%)
    "tg_return_since_open",       # Return since 00:00 UTC (%)
    "tg_volume_ratio_1h",         # Current 1h vol / avg 1h vol (20-day)
    "tg_volume_ratio_4h",         # Current 4h vol / avg 4h vol
    "tg_volume_acceleration",     # vol_ratio_1h / vol_ratio_4h (accel > 1 = fresh)

    # Price structure
    "tg_price_vs_ema20_pct",      # Distance from EMA20 (%)
    "tg_price_vs_ema50_pct",
    "tg_price_vs_ema200_pct",
    "tg_ema20_slope",             # EMA20 slope (% per bar)
    "tg_adx",
    "tg_rsi",
    "tg_atr_pct",                 # ATR as % of price (volatility normalized)
    "tg_daily_range_pct",         # High-low range today (%)
    "tg_range_position",          # Where in today's range: (close - low) / (high - low)

    # Cross-asset context
    "tg_btc_return_1h",           # BTC return last 1h (market context)
    "tg_btc_return_4h",
    "tg_vs_btc_1h",              # Coin return - BTC return (relative strength)
    "tg_vs_btc_4h",
    "tg_sector_avg_return",       # Avg return of same sector coins (if available)

    # Order flow (from OrderFlowAnalyzer)
    "tg_of_cvd_pct_5m",          # CVD % (5-min window)
    "tg_of_imbalance_5m",        # Buy/sell imbalance
    "tg_of_large_trade_ratio_5m", # Large trade volume ratio
    "tg_of_breakout_signal_5m",   # Order flow breakout signal

    # Derivatives (from DerivativesData)
    "tg_funding_rate",
    "tg_oi_change_1h",
    "tg_oi_change_4h",
    "tg_ls_ratio",
    "tg_liq_total_1h",
    "tg_funding_flip",

    # Historical gainer patterns
    "tg_was_top_gainer_yesterday", # 1 if was top gainer yesterday
    "tg_top_gainer_count_7d",      # How many times top gainer in last 7 days
    "tg_avg_daily_return_7d",      # Average daily return (% )
    "tg_max_daily_return_7d",      # Max daily return in last 7 days

    # Temporal
    "tg_hour_utc",                 # Hour of day (0-23)
    "tg_hour_sin",                 # sin(2π × hour/24)
    "tg_hour_cos",                 # cos(2π × hour/24)
    "tg_day_of_week",              # 0=Mon, 6=Sun
    "tg_is_weekend",               # 1 if Sat/Sun
]


class TopGainerModel:
    """
    Gradient boosting model for top gainer prediction.

    Uses tree-based ensemble (stored as JSON for portability).
    Supports CatBoost serialized model or custom decision tree ensemble.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_payload: Optional[dict] = None
        self._feature_names: List[str] = list(FEATURE_NAMES)
        self._thresholds: Dict[str, float] = {
            "top5": 0.15,
            "top10": 0.20,
            "top20": 0.30,
            "top50": 0.40,
        }
        self._cb_models: Dict[str, Any] = {}  # tier -> loaded CatBoost model
        if model_path:
            self.load(model_path)

    def load(self, path: str) -> bool:
        """Load trained model from JSON."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._model_payload = data
            if "feature_names" in data:
                self._feature_names = data["feature_names"]
            if "thresholds" in data:
                self._thresholds.update(data["thresholds"])
            log.info("TopGainerModel loaded from %s", path)
            return True
        except Exception:
            log.warning("TopGainerModel: no model at %s, using heuristic", path)
            return False

    def compute_features(
        self,
        symbol: str,
        klines_1h: Optional[np.ndarray],
        klines_15m: Optional[np.ndarray],
        features_1h: Optional[dict],
        features_15m: Optional[dict],
        order_flow_features: Optional[Dict[str, float]] = None,
        derivatives_features: Optional[Dict[str, float]] = None,
        btc_data: Optional[dict] = None,
        historical_gainer_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute all features for top gainer prediction.

        Aggregates data from multiple sources into a flat feature dict.
        """
        f: Dict[str, float] = {}
        now_ms = int(time.time() * 1000)

        # ── Morning momentum ─────────────────────────────────────────────────
        if klines_1h is not None and len(klines_1h) >= 4:
            close = klines_1h["c"].astype(float)
            vol = klines_1h["v"].astype(float)
            high = klines_1h["h"].astype(float)
            low = klines_1h["l"].astype(float)

            # Returns
            f["tg_return_1h"] = _pct_change(close, 1)
            f["tg_return_4h"] = _pct_change(close, 4)

            # Since open (approximate: first bar of last 24 bars)
            start_idx = max(0, len(close) - 24)
            f["tg_return_since_open"] = (
                (close[-1] - close[start_idx]) / close[start_idx] * 100.0
                if close[start_idx] > 0 else 0.0
            )

            # Volume ratios
            if len(vol) >= 21:
                avg_vol_20 = float(np.mean(vol[-21:-1]))
                f["tg_volume_ratio_1h"] = vol[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
                avg_vol_4 = float(np.mean(vol[-5:-1]))
                f["tg_volume_ratio_4h"] = (
                    float(np.sum(vol[-4:])) / (avg_vol_4 * 4) if avg_vol_4 > 0 else 1.0
                )
            else:
                f["tg_volume_ratio_1h"] = 1.0
                f["tg_volume_ratio_4h"] = 1.0

            f["tg_volume_acceleration"] = (
                f["tg_volume_ratio_1h"] / f["tg_volume_ratio_4h"]
                if f["tg_volume_ratio_4h"] > 0 else 1.0
            )

            # Daily range
            last_24_high = float(np.max(high[-24:])) if len(high) >= 24 else float(high[-1])
            last_24_low = float(np.min(low[-24:])) if len(low) >= 24 else float(low[-1])
            f["tg_daily_range_pct"] = (
                (last_24_high - last_24_low) / last_24_low * 100.0
                if last_24_low > 0 else 0.0
            )
            f["tg_range_position"] = (
                (close[-1] - last_24_low) / (last_24_high - last_24_low)
                if (last_24_high - last_24_low) > 0 else 0.5
            )
        else:
            for key in ["tg_return_1h", "tg_return_4h", "tg_return_since_open",
                        "tg_volume_ratio_1h", "tg_volume_ratio_4h",
                        "tg_volume_acceleration", "tg_daily_range_pct", "tg_range_position"]:
                f[key] = 0.0

        # ── Price structure from indicators ──────────────────────────────────
        feat = features_1h or features_15m or {}
        i = -1  # last bar
        if klines_1h is not None:
            i = len(klines_1h) - 1

        f["tg_price_vs_ema20_pct"] = _feat_pct(feat, "ema_fast", klines_1h, i)
        f["tg_price_vs_ema50_pct"] = _feat_pct(feat, "ema_slow", klines_1h, i)
        f["tg_price_vs_ema200_pct"] = _feat_pct(feat, "ema200", klines_1h, i)
        f["tg_ema20_slope"] = _feat_val(feat, "slope", i, 0.0)
        f["tg_adx"] = _feat_val(feat, "adx", i, 25.0)
        f["tg_rsi"] = _feat_val(feat, "rsi", i, 50.0)

        atr = _feat_val(feat, "atr", i, 0.0)
        price = float(klines_1h["c"][i]) if klines_1h is not None and i >= 0 else 1.0
        f["tg_atr_pct"] = atr / price * 100.0 if price > 0 else 0.0

        # ── Cross-asset context ──────────────────────────────────────────────
        if btc_data:
            f["tg_btc_return_1h"] = btc_data.get("btc_return_1h", 0.0)
            f["tg_btc_return_4h"] = btc_data.get("btc_return_4h", 0.0)
            f["tg_vs_btc_1h"] = f.get("tg_return_1h", 0.0) - f["tg_btc_return_1h"]
            f["tg_vs_btc_4h"] = f.get("tg_return_4h", 0.0) - f["tg_btc_return_4h"]
        else:
            f["tg_btc_return_1h"] = 0.0
            f["tg_btc_return_4h"] = 0.0
            f["tg_vs_btc_1h"] = 0.0
            f["tg_vs_btc_4h"] = 0.0
        f["tg_sector_avg_return"] = 0.0  # TODO: sector classification

        # ── Order flow features ──────────────────────────────────────────────
        of = order_flow_features or {}
        f["tg_of_cvd_pct_5m"] = of.get("of_cvd_pct_5m", 0.0)
        f["tg_of_imbalance_5m"] = of.get("of_imbalance_5m", 0.5)
        f["tg_of_large_trade_ratio_5m"] = of.get("of_large_trade_ratio_5m", 0.0)
        f["tg_of_breakout_signal_5m"] = of.get("of_breakout_5m", 0.0)

        # ── Derivatives features ─────────────────────────────────────────────
        dd = derivatives_features or {}
        f["tg_funding_rate"] = dd.get("deriv_funding_rate", 0.0)
        f["tg_oi_change_1h"] = dd.get("deriv_oi_change_1h", 0.0)
        f["tg_oi_change_4h"] = dd.get("deriv_oi_change_4h", 0.0)
        f["tg_ls_ratio"] = dd.get("deriv_ls_ratio", 1.0)
        f["tg_liq_total_1h"] = dd.get("deriv_liq_total_1h", 0.0)
        f["tg_funding_flip"] = dd.get("deriv_funding_flip", 0.0)

        # ── Historical gainer stats ──────────────────────────────────────────
        gs = historical_gainer_stats or {}
        f["tg_was_top_gainer_yesterday"] = gs.get("was_top_gainer_yesterday", 0.0)
        f["tg_top_gainer_count_7d"] = gs.get("top_gainer_count_7d", 0.0)
        f["tg_avg_daily_return_7d"] = gs.get("avg_daily_return_7d", 0.0)
        f["tg_max_daily_return_7d"] = gs.get("max_daily_return_7d", 0.0)

        # ── Temporal ─────────────────────────────────────────────────────────
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        hour = now.hour + now.minute / 60.0
        f["tg_hour_utc"] = float(now.hour)
        f["tg_hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        f["tg_hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        f["tg_day_of_week"] = float(now.weekday())
        f["tg_is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0

        return f

    def predict(self, features: Dict[str, float]) -> TopGainerPrediction:
        """
        Predict top gainer probability.

        If trained model available: use it.
        Otherwise: heuristic based on momentum + volume + order flow.
        """
        if self._model_payload and "model_type" in self._model_payload:
            return self._predict_model(features)
        return self._predict_heuristic(features)

    def predict_batch(
        self,
        features_batch: Dict[str, Dict[str, float]],
    ) -> Dict[str, TopGainerPrediction]:
        """Predict for multiple symbols. Returns {symbol: prediction}."""
        return {sym: self.predict(f) for sym, f in features_batch.items()}

    def _predict_model(self, features: Dict[str, float]) -> TopGainerPrediction:
        """Prediction using trained CatBoost tier models."""
        try:
            payload = self._model_payload
            tier_models = payload.get("tier_models", {}) if payload else {}
            if not tier_models:
                return self._predict_heuristic(features)

            x = np.array([[features.get(fn, 0.0) for fn in self._feature_names]])
            probs: Dict[str, float] = {}

            for tier in ["top5", "top10", "top20", "top50"]:
                tm = tier_models.get(tier, {})
                if not tm:
                    probs[tier] = 0.0
                    continue

                # Lazy-load CatBoost model
                if tier not in self._cb_models:
                    model_json_str = tm.get("model_json")
                    stump_data = tm if tm.get("model_type") == "stump_ensemble" else None
                    if model_json_str and isinstance(model_json_str, str):
                        try:
                            import tempfile, os
                            from catboost import CatBoostClassifier
                            with tempfile.NamedTemporaryFile(
                                suffix=".json", delete=False, mode="w", encoding="utf-8"
                            ) as tf:
                                tf.write(model_json_str)
                                tmp_path = tf.name
                            try:
                                cb = CatBoostClassifier()
                                cb.load_model(tmp_path, format="json")
                                self._cb_models[tier] = ("catboost", cb)
                            finally:
                                os.unlink(tmp_path)
                        except Exception:
                            self._cb_models[tier] = ("stump", stump_data)
                    else:
                        self._cb_models[tier] = ("stump", stump_data)

                kind, model_obj = self._cb_models.get(tier, ("none", None))
                if kind == "catboost" and model_obj is not None:
                    p = float(model_obj.predict_proba(x)[0, 1])
                elif kind == "stump" and model_obj:
                    p = self._stump_predict(model_obj, x[0])
                else:
                    p = 0.0
                probs[tier] = p

            return TopGainerPrediction(
                symbol=features.get("symbol", ""),
                timestamp_ms=int(time.time() * 1000),
                prob_top5=probs.get("top5", 0.0),
                prob_top10=probs.get("top10", 0.0),
                prob_top20=probs.get("top20", 0.0),
                prob_top50=probs.get("top50", 0.0),
                expected_eod_return=0.0,
                confidence=float(max(probs.values())) if probs else 0.0,
            )
        except Exception:
            pass
        return self._predict_heuristic(features)

    def _stump_predict(self, payload: dict, x: np.ndarray) -> float:
        """Run stump ensemble prediction."""
        try:
            pred = float(payload.get("init_pred", 0.0))
            lr = float(payload.get("learning_rate", 0.1))
            for stump in payload.get("stumps", []):
                f = stump["feature"]
                thr = stump["threshold"]
                if x[f] <= thr:
                    pred += stump["left_value"]
                else:
                    pred += stump["right_value"]
            return float(1.0 / (1.0 + np.exp(-pred)))
        except Exception:
            return 0.0

    def _predict_heuristic(self, features: Dict[str, float]) -> TopGainerPrediction:
        """
        Heuristic prediction when no trained model available.

        Combines momentum, volume, and order flow signals into
        a rough probability estimate. Designed to be better than random
        while the model is being trained.
        """
        score = 0.0
        max_score = 100.0

        # ── Momentum (40% weight) ────────────────────────────────────────────
        ret_1h = features.get("tg_return_1h", 0.0)
        ret_4h = features.get("tg_return_4h", 0.0)
        ret_open = features.get("tg_return_since_open", 0.0)

        # Early positive momentum is key signal
        if ret_open > 3.0:
            score += 15.0
        elif ret_open > 1.5:
            score += 10.0
        elif ret_open > 0.5:
            score += 5.0

        # Accelerating momentum (1h > 4h/4 = acceleration)
        if ret_1h > 0 and ret_4h > 0:
            accel = ret_1h / (ret_4h / 4.0) if ret_4h > 0.1 else 2.0
            if accel > 1.5:
                score += 10.0
            elif accel > 1.0:
                score += 5.0

        # Relative strength vs BTC
        vs_btc = features.get("tg_vs_btc_4h", 0.0)
        if vs_btc > 2.0:
            score += 10.0
        elif vs_btc > 1.0:
            score += 5.0

        # ── Volume (25% weight) ──────────────────────────────────────────────
        vol_ratio = features.get("tg_volume_ratio_1h", 1.0)
        vol_accel = features.get("tg_volume_acceleration", 1.0)

        if vol_ratio > 3.0:
            score += 10.0
        elif vol_ratio > 2.0:
            score += 6.0
        elif vol_ratio > 1.5:
            score += 3.0

        if vol_accel > 1.5:
            score += 5.0  # Volume is accelerating

        # ── Order flow (20% weight) ──────────────────────────────────────────
        cvd_pct = features.get("tg_of_cvd_pct_5m", 0.0)
        imbalance = features.get("tg_of_imbalance_5m", 0.5)
        large_trades = features.get("tg_of_large_trade_ratio_5m", 0.0)
        breakout_sig = features.get("tg_of_breakout_signal_5m", 0.0)

        if cvd_pct > 5.0 and imbalance > 0.55:
            score += 8.0
        elif cvd_pct > 2.0 and imbalance > 0.52:
            score += 4.0

        if large_trades > 0.3:
            score += 5.0
        if breakout_sig > 0.5:
            score += 7.0

        # ── Derivatives (15% weight) ─────────────────────────────────────────
        oi_change = features.get("tg_oi_change_1h", 0.0)
        funding_flip = features.get("tg_funding_flip", 0.0)
        ls_ratio = features.get("tg_ls_ratio", 1.0)

        if oi_change > 5.0:
            score += 5.0  # New money entering
        if funding_flip > 0.5:
            score += 5.0  # Sentiment shift
        if ls_ratio < 0.8 and ret_open > 1.0:
            score += 5.0  # Short squeeze potential

        # ── Normalize to probabilities ───────────────────────────────────────
        raw_prob = min(1.0, score / max_score)

        # Different thresholds for different tiers
        prob_top50 = raw_prob
        prob_top20 = raw_prob * 0.6
        prob_top10 = raw_prob * 0.35
        prob_top5 = raw_prob * 0.15

        expected_return = ret_open + (score / max_score * 5.0)  # rough estimate

        return TopGainerPrediction(
            symbol=features.get("symbol", ""),
            timestamp_ms=int(time.time() * 1000),
            prob_top5=prob_top5,
            prob_top10=prob_top10,
            prob_top20=prob_top20,
            prob_top50=prob_top50,
            expected_eod_return=expected_return,
            confidence=min(1.0, score / 50.0),  # confident if score > 50
        )

    # ── Dataset management ───────────────────────────────────────────────────

    def log_training_sample(
        self,
        symbol: str,
        features: Dict[str, float],
        label_top5: bool,
        label_top10: bool,
        label_top20: bool,
        label_top50: bool,
        eod_return_pct: float,
    ) -> None:
        """Log a training sample to dataset file."""
        record = {
            "ts": int(time.time() * 1000),
            "symbol": symbol,
            "features": features,
            "label_top5": int(label_top5),
            "label_top10": int(label_top10),
            "label_top20": int(label_top20),
            "label_top50": int(label_top50),
            "eod_return_pct": round(eod_return_pct, 4),
        }
        try:
            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            log.exception("Failed to log top gainer training sample")

    async def collect_daily_labels(self, session: "aiohttp.ClientSession") -> Dict[str, float]:
        """
        Fetch 24hr ticker from Binance and return {symbol: 24h_change_%}.
        Call at 00:00 UTC to get EOD labels.
        """
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

            results = {}
            for t in data:
                sym = str(t.get("symbol", ""))
                if sym.endswith("USDT"):
                    results[sym] = float(t.get("priceChangePercent", 0))
            return results
        except Exception:
            return {}


# ── Helper functions ─────────────────────────────────────────────────────────

def _pct_change(arr: np.ndarray, periods: int) -> float:
    """% change over last N periods."""
    if len(arr) <= periods or arr[-periods - 1] == 0:
        return 0.0
    return (arr[-1] - arr[-periods - 1]) / arr[-periods - 1] * 100.0


def _feat_val(feat: dict, key: str, i: int, default: float) -> float:
    """Get feature value safely."""
    arr = feat.get(key)
    if arr is not None and 0 <= i < len(arr) and np.isfinite(arr[i]):
        return float(arr[i])
    # Try negative index
    if arr is not None and i < 0 and abs(i) <= len(arr):
        val = arr[i]
        if np.isfinite(val):
            return float(val)
    return default


def _feat_pct(feat: dict, key: str, klines: Optional[np.ndarray], i: int) -> float:
    """Get price vs feature as % difference."""
    if klines is None or i < 0 and abs(i) > len(klines):
        return 0.0
    price = float(klines["c"][i]) if i >= 0 else float(klines["c"][i])
    val = _feat_val(feat, key, i, price)
    return (price - val) / val * 100.0 if val > 0 else 0.0
