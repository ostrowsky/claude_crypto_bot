from __future__ import annotations

"""
Multi-Horizon Signal Framework — separate models and logic per timeframe.

Horizons:
  - scalp     (1m-15m)  : Order flow + micro impulse + spread dynamics
  - intraday  (15m-4h)  : Existing strategy modes + ML gating (current bot)
  - daily     (4h-24h)  : TFT + sentiment + funding cycle
  - swing     (1d-7d)   : On-chain + macro regime + sector rotation
  - position  (1w-4w)   : Fundamental + tokenomics + narrative

Each horizon has:
  - Its own signal generation logic
  - Independent entry/exit parameters
  - Separate position slots and sizing
  - Cascading signal: longer horizon confirms shorter horizon direction

Architecture:
  - HorizonManager orchestrates all horizons
  - Each horizon is a HorizonStrategy subclass
  - Cascading confirmation: swing direction → daily timing → intraday entry
  - Portfolio allocation: configurable % per horizon

Usage:
    mgr = HorizonManager(ws_manager, derivatives, regime_detector)
    signals = await mgr.scan_all("BTCUSDT")
    # signals = {horizon: HorizonSignal, ...}
    composite = mgr.composite_signal("BTCUSDT")
    # composite.direction, composite.confidence, composite.entry_horizon
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class Direction(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class Horizon(Enum):
    SCALP = "scalp"           # 1m - 15m
    INTRADAY = "intraday"     # 15m - 4h
    DAILY = "daily"           # 4h - 24h
    SWING = "swing"           # 1d - 7d
    POSITION = "position"     # 1w - 4w


# ── Configuration per horizon ────────────────────────────────────────────────

HORIZON_CONFIG = {
    Horizon.SCALP: {
        "timeframes": ["1m", "5m"],
        "max_hold_minutes": 60,
        "position_pct": 0.15,          # 15% of portfolio
        "max_positions": 3,
        "min_confidence": 0.65,
        "data_sources": ["order_flow", "book_ticker", "1m_klines"],
        "atr_trail_mult": 0.8,
        "score_weight_in_composite": 0.10,
    },
    Horizon.INTRADAY: {
        "timeframes": ["15m", "1h"],
        "max_hold_minutes": 720,       # 12 hours
        "position_pct": 0.35,          # 35%
        "max_positions": 6,
        "min_confidence": 0.55,
        "data_sources": ["15m_klines", "1h_klines", "order_flow", "derivatives"],
        "atr_trail_mult": 1.0,
        "score_weight_in_composite": 0.30,
    },
    Horizon.DAILY: {
        "timeframes": ["1h", "4h"],
        "max_hold_minutes": 2880,      # 48 hours
        "position_pct": 0.25,          # 25%
        "max_positions": 4,
        "min_confidence": 0.60,
        "data_sources": ["1h_klines", "4h_klines", "derivatives", "regime"],
        "atr_trail_mult": 1.5,
        "score_weight_in_composite": 0.30,
    },
    Horizon.SWING: {
        "timeframes": ["4h", "1d"],
        "max_hold_minutes": 10080,     # 7 days
        "position_pct": 0.15,          # 15%
        "max_positions": 3,
        "min_confidence": 0.65,
        "data_sources": ["4h_klines", "1d_klines", "regime", "on_chain"],
        "atr_trail_mult": 2.5,
        "score_weight_in_composite": 0.20,
    },
    Horizon.POSITION: {
        "timeframes": ["1d", "1w"],
        "max_hold_minutes": 43200,     # 30 days
        "position_pct": 0.10,          # 10%
        "max_positions": 2,
        "min_confidence": 0.70,
        "data_sources": ["1d_klines", "1w_klines", "regime", "fundamental"],
        "atr_trail_mult": 3.5,
        "score_weight_in_composite": 0.10,
    },
}


@dataclass
class HorizonSignal:
    """Signal from a single horizon analysis."""
    horizon: Horizon
    symbol: str
    direction: Direction
    confidence: float              # [0, 1]
    score: float                   # Raw score (like existing entry_score)
    timestamp_ms: int

    # Signal details
    entry_mode: str = ""           # e.g., "trend", "breakout", "order_flow_surge"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    expected_return_pct: float = 0.0
    max_hold_minutes: int = 0

    # Data that led to signal
    features: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        config = HORIZON_CONFIG.get(self.horizon, {})
        return (
            self.direction in (Direction.BUY, Direction.STRONG_BUY)
            and self.confidence >= config.get("min_confidence", 0.6)
        )

    @property
    def risk_reward(self) -> float:
        if self.entry_price > 0 and self.stop_loss > 0 and self.take_profit > 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            return reward / risk if risk > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon.value,
            "symbol": self.symbol,
            "direction": self.direction.name,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 1),
            "entry_mode": self.entry_mode,
            "expected_return_pct": round(self.expected_return_pct, 2),
            "risk_reward": round(self.risk_reward, 2),
            "reasons": self.reasons,
        }


@dataclass
class CompositeSignal:
    """Aggregated signal across all horizons."""
    symbol: str
    timestamp_ms: int
    direction: Direction
    confidence: float
    score: float
    horizon_signals: Dict[str, HorizonSignal] = field(default_factory=dict)

    # Cascade quality
    alignment_count: int = 0       # How many horizons agree on direction
    alignment_pct: float = 0.0     # % of horizons aligned
    best_entry_horizon: str = ""   # Recommended horizon for entry
    cascade_bonus: float = 0.0     # Score bonus from multi-horizon alignment

    @property
    def is_high_conviction(self) -> bool:
        return self.alignment_count >= 3 and self.confidence > 0.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.name,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 1),
            "alignment": f"{self.alignment_count}/{len(self.horizon_signals)}",
            "best_entry": self.best_entry_horizon,
            "cascade_bonus": round(self.cascade_bonus, 1),
            "horizons": {k: v.to_dict() for k, v in self.horizon_signals.items()},
        }


# ── Horizon strategy implementations ────────────────────────────────────────

class ScalpStrategy:
    """
    Scalp horizon (1m-15m): order flow driven.

    Entry signals:
      - Order flow surge: CVD spike + large trades + trade velocity
      - Spread compression: tightening spread → imminent move
      - Book imbalance: heavy bid side → upward pressure
      - Micro breakout: 1m price breaks above 5m resistance with volume
    """

    def analyze(
        self,
        symbol: str,
        klines_1m: Optional[np.ndarray],
        order_flow_metrics: Optional[dict],
        book_ticker: Optional[dict],
    ) -> HorizonSignal:
        """Analyze scalp opportunity from real-time data."""
        now_ms = int(time.time() * 1000)
        signal = HorizonSignal(
            horizon=Horizon.SCALP,
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            score=0.0,
            timestamp_ms=now_ms,
        )

        if not order_flow_metrics:
            return signal

        score = 0.0
        reasons = []

        # ── Order flow surge ─────────────────────────────────────────────────
        of_1m = order_flow_metrics.get("1m")
        if of_1m:
            if of_1m.cvd_pct > 8.0 and of_1m.imbalance > 0.58:
                score += 25.0
                reasons.append(f"OF surge: CVD {of_1m.cvd_pct:.1f}%, imbalance {of_1m.imbalance:.2f}")

            if of_1m.large_trade_ratio > 0.4 and of_1m.large_buy_volume > of_1m.large_sell_volume:
                score += 15.0
                reasons.append(f"Large buy trades: {of_1m.large_trade_count}")

            if of_1m.breakout_signal > 0.5:
                score += 20.0
                reasons.append(f"OF breakout signal: {of_1m.breakout_signal:.2f}")

            if of_1m.trades_per_sec > 5.0:
                score += 10.0
                reasons.append(f"High velocity: {of_1m.trades_per_sec:.1f} tps")

        # ── 1m price action ──────────────────────────────────────────────────
        if klines_1m is not None and len(klines_1m) >= 10:
            close = klines_1m["c"].astype(float)
            high = klines_1m["h"].astype(float)
            vol = klines_1m["v"].astype(float)

            # Micro breakout: price above 5-bar high with volume
            recent_high = float(np.max(high[-6:-1]))
            if close[-1] > recent_high and vol[-1] > np.mean(vol[-10:]) * 2:
                score += 15.0
                reasons.append("Micro breakout above 5-bar high")

            # Quick momentum: 3 consecutive up bars with increasing volume
            if (close[-1] > close[-2] > close[-3] and
                    vol[-1] > vol[-2] > vol[-3]):
                score += 10.0
                reasons.append("3-bar momentum acceleration")

        # ── Compute signal ───────────────────────────────────────────────────
        max_score = 95.0
        confidence = min(1.0, score / max_score)

        if score >= 40:
            signal.direction = Direction.STRONG_BUY if score >= 65 else Direction.BUY
            signal.confidence = confidence
            signal.score = score
            signal.entry_mode = "order_flow_surge" if score >= 50 else "micro_breakout"
            signal.reasons = reasons
            signal.max_hold_minutes = HORIZON_CONFIG[Horizon.SCALP]["max_hold_minutes"]

        return signal


class IntradayStrategy:
    """
    Intraday horizon (15m-4h): existing strategy + enhanced with new data.

    This wraps the existing strategy.py logic but enhances entry/exit
    decisions with order flow, derivatives, and regime data.
    """

    def analyze(
        self,
        symbol: str,
        existing_signal: Optional[Dict[str, Any]],
        order_flow_features: Optional[Dict[str, float]],
        derivatives_features: Optional[Dict[str, float]],
        regime_state: Optional[Any],
        top_gainer_prediction: Optional[Any],
    ) -> HorizonSignal:
        """Wrap existing bot signal with enhanced context."""
        now_ms = int(time.time() * 1000)
        signal = HorizonSignal(
            horizon=Horizon.INTRADAY,
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            score=0.0,
            timestamp_ms=now_ms,
        )

        if not existing_signal:
            return signal

        # Start from existing bot's analysis
        base_score = existing_signal.get("score", 0.0)
        mode = existing_signal.get("mode", "")
        is_buy = existing_signal.get("is_buy", False)

        if not is_buy:
            return signal

        score = base_score
        reasons = [f"Base: {mode} score={base_score:.0f}"]

        # ── Enhance with order flow ──────────────────────────────────────────
        if order_flow_features:
            of_cvd = order_flow_features.get("of_cvd_pct_5m", 0.0)
            of_imbalance = order_flow_features.get("of_imbalance_5m", 0.5)

            if of_cvd > 3.0 and of_imbalance > 0.55:
                bonus = min(8.0, of_cvd)
                score += bonus
                reasons.append(f"+{bonus:.0f} order flow confirmation")
            elif of_cvd < -3.0 and of_imbalance < 0.45:
                score -= 6.0
                reasons.append("-6 order flow divergence (selling pressure)")

        # ── Enhance with derivatives ─────────────────────────────────────────
        if derivatives_features:
            oi_change = derivatives_features.get("deriv_oi_change_1h", 0.0)
            funding = derivatives_features.get("deriv_funding_rate", 0.0)
            funding_flip = derivatives_features.get("deriv_funding_flip", 0.0)

            if oi_change > 5.0:
                score += 5.0
                reasons.append("+5 OI expanding (new money)")
            if funding_flip > 0.5:
                score += 4.0
                reasons.append("+4 funding flip (sentiment shift)")
            if funding > 0.001:
                score -= 3.0
                reasons.append("-3 high funding (crowded long)")

        # ── Regime adjustment ────────────────────────────────────────────────
        if regime_state:
            regime_bonus = getattr(regime_state, "score_bonus", 0.0)
            if regime_bonus != 0:
                score += regime_bonus
                reasons.append(f"{regime_bonus:+.0f} regime ({regime_state})")

            allowed = getattr(regime_state, "allowed_entry_modes", ())
            if allowed and mode not in allowed:
                score -= 15.0
                reasons.append(f"-15 mode '{mode}' not allowed in {regime_state.name}")

        # ── Top gainer boost ─────────────────────────────────────────────────
        if top_gainer_prediction:
            tg_bonus = getattr(top_gainer_prediction, "score_bonus", 0.0)
            if tg_bonus > 0:
                score += tg_bonus
                reasons.append(f"+{tg_bonus:.0f} top gainer prediction")

        # ── Compute final signal ─────────────────────────────────────────────
        confidence = min(1.0, max(0.0, (score - 40) / 40.0))

        if score >= 50:
            signal.direction = Direction.STRONG_BUY if score >= 65 else Direction.BUY
            signal.confidence = confidence
            signal.score = score
            signal.entry_mode = mode
            signal.reasons = reasons
            signal.max_hold_minutes = HORIZON_CONFIG[Horizon.INTRADAY]["max_hold_minutes"]

        return signal


class DailyStrategy:
    """
    Daily horizon (4h-24h): trend-following with macro context.

    Entry signals:
      - 4h trend confirmation (EMA stack + ADX > 20)
      - Positive funding cycle (funding low → rising)
      - OI expansion with price (genuine trend)
      - Strong regime (bull or recovery)
    """

    def analyze(
        self,
        symbol: str,
        klines_4h: Optional[np.ndarray],
        features_4h: Optional[dict],
        derivatives_features: Optional[Dict[str, float]],
        regime_state: Optional[Any],
    ) -> HorizonSignal:
        """Analyze daily opportunity."""
        now_ms = int(time.time() * 1000)
        signal = HorizonSignal(
            horizon=Horizon.DAILY,
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            score=0.0,
            timestamp_ms=now_ms,
        )

        if klines_4h is None or len(klines_4h) < 30 or features_4h is None:
            return signal

        i = len(klines_4h) - 1
        close = float(klines_4h["c"][i])

        score = 0.0
        reasons = []

        # ── 4h trend structure ───────────────────────────────────────────────
        ema20 = _safe(features_4h, "ema_fast", i, close)
        ema50 = _safe(features_4h, "ema_slow", i, close)
        ema200 = _safe(features_4h, "ema200", i, close)
        adx = _safe(features_4h, "adx", i, 0.0)
        rsi = _safe(features_4h, "rsi", i, 50.0)
        slope = _safe(features_4h, "slope", i, 0.0)

        if close > ema20 > ema50:
            score += 15.0
            reasons.append("4h EMA stack bullish")
            if close > ema200:
                score += 5.0
                reasons.append("Above 4h EMA200")

        if adx > 20 and slope > 0.05:
            score += 10.0
            reasons.append(f"4h ADX={adx:.0f}, slope={slope:.2f}")

        if 45 < rsi < 70:
            score += 5.0
            reasons.append(f"4h RSI={rsi:.0f} (healthy)")

        # ── Derivatives ──────────────────────────────────────────────────────
        if derivatives_features:
            oi_4h = derivatives_features.get("deriv_oi_change_4h", 0.0)
            if oi_4h > 3.0 and close > ema20:
                score += 8.0
                reasons.append(f"+8 OI expanding {oi_4h:.1f}% with uptrend")

            funding = derivatives_features.get("deriv_funding_rate", 0.0)
            if -0.0005 < funding < 0.0003:
                score += 5.0
                reasons.append("Funding neutral-low (room for longs)")

        # ── Regime ───────────────────────────────────────────────────────────
        if regime_state:
            if regime_state.name in ("strong_bull", "weak_bull"):
                score += 8.0
                reasons.append(f"Regime: {regime_state}")
            elif regime_state.name in ("strong_bear", "volatile_chop"):
                score -= 20.0
                reasons.append(f"Regime unfavorable: {regime_state}")

        confidence = min(1.0, max(0.0, (score - 20) / 40.0))
        if score >= 30:
            signal.direction = Direction.STRONG_BUY if score >= 50 else Direction.BUY
            signal.confidence = confidence
            signal.score = score
            signal.entry_mode = "daily_trend"
            signal.reasons = reasons
            signal.max_hold_minutes = HORIZON_CONFIG[Horizon.DAILY]["max_hold_minutes"]

        return signal


class SwingStrategy:
    """
    Swing horizon (1d-7d): macro trend following.

    Entry signals:
      - 1d EMA stack alignment
      - Regime = strong_bull or recovery
      - Weekly momentum positive
      - OI growing steadily over days
    """

    def analyze(
        self,
        symbol: str,
        klines_1d: Optional[np.ndarray],
        features_1d: Optional[dict],
        regime_state: Optional[Any],
    ) -> HorizonSignal:
        """Analyze swing opportunity."""
        now_ms = int(time.time() * 1000)
        signal = HorizonSignal(
            horizon=Horizon.SWING,
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            score=0.0,
            timestamp_ms=now_ms,
        )

        if klines_1d is None or len(klines_1d) < 30 or features_1d is None:
            return signal

        i = len(klines_1d) - 1
        close = float(klines_1d["c"][i])

        score = 0.0
        reasons = []

        ema20 = _safe(features_1d, "ema_fast", i, close)
        ema50 = _safe(features_1d, "ema_slow", i, close)
        adx = _safe(features_1d, "adx", i, 0.0)
        rsi = _safe(features_1d, "rsi", i, 50.0)

        if close > ema20 > ema50:
            score += 20.0
            reasons.append("1d EMA stack bullish")

        if adx > 20:
            score += 10.0
            reasons.append(f"1d ADX={adx:.0f}")

        if 40 < rsi < 65:
            score += 10.0
            reasons.append(f"1d RSI={rsi:.0f} (not overbought)")

        # Weekly momentum
        if len(klines_1d) >= 7:
            week_return = (close - float(klines_1d["c"][i - 7])) / float(klines_1d["c"][i - 7]) * 100
            if week_return > 0:
                score += min(10.0, week_return * 2)
                reasons.append(f"Week return +{week_return:.1f}%")

        if regime_state and regime_state.name == "strong_bull":
            score += 10.0
            reasons.append("Strong bull regime")

        confidence = min(1.0, max(0.0, (score - 25) / 35.0))
        if score >= 35:
            signal.direction = Direction.STRONG_BUY if score >= 55 else Direction.BUY
            signal.confidence = confidence
            signal.score = score
            signal.entry_mode = "swing_trend"
            signal.reasons = reasons
            signal.max_hold_minutes = HORIZON_CONFIG[Horizon.SWING]["max_hold_minutes"]

        return signal


# ── Horizon Manager ──────────────────────────────────────────────────────────

class HorizonManager:
    """
    Orchestrates all horizon strategies and produces composite signals.

    Cascade logic:
      1. Swing/position horizons set the direction (macro bias)
      2. Daily horizon confirms the timing (is today a good entry day?)
      3. Intraday horizon finds the entry point (existing bot + enhancements)
      4. Scalp horizon can accelerate entry (order flow spike)

    Score bonus for multi-horizon alignment.
    """

    def __init__(self):
        self.scalp = ScalpStrategy()
        self.intraday = IntradayStrategy()
        self.daily = DailyStrategy()
        self.swing = SwingStrategy()
        self._last_signals: Dict[str, Dict[str, HorizonSignal]] = {}

    def analyze_all(
        self,
        symbol: str,
        *,
        klines_1m: Optional[np.ndarray] = None,
        klines_15m: Optional[np.ndarray] = None,
        klines_1h: Optional[np.ndarray] = None,
        klines_4h: Optional[np.ndarray] = None,
        klines_1d: Optional[np.ndarray] = None,
        features_15m: Optional[dict] = None,
        features_1h: Optional[dict] = None,
        features_4h: Optional[dict] = None,
        features_1d: Optional[dict] = None,
        order_flow_metrics: Optional[dict] = None,
        order_flow_features: Optional[Dict[str, float]] = None,
        derivatives_features: Optional[Dict[str, float]] = None,
        regime_state: Optional[Any] = None,
        top_gainer_prediction: Optional[Any] = None,
        existing_intraday_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, HorizonSignal]:
        """
        Run all horizon analyses for a symbol.
        Returns {horizon_name: HorizonSignal}.
        """
        signals = {}

        # Scalp
        signals["scalp"] = self.scalp.analyze(
            symbol, klines_1m, order_flow_metrics, None,
        )

        # Intraday (wraps existing bot logic)
        signals["intraday"] = self.intraday.analyze(
            symbol, existing_intraday_signal,
            order_flow_features, derivatives_features,
            regime_state, top_gainer_prediction,
        )

        # Daily
        signals["daily"] = self.daily.analyze(
            symbol, klines_4h, features_4h,
            derivatives_features, regime_state,
        )

        # Swing
        signals["swing"] = self.swing.analyze(
            symbol, klines_1d, features_1d, regime_state,
        )

        self._last_signals[symbol] = signals
        return signals

    def composite_signal(self, symbol: str) -> CompositeSignal:
        """
        Compute composite signal from all horizons.
        Multi-horizon alignment gets a score bonus.
        """
        signals = self._last_signals.get(symbol, {})
        now_ms = int(time.time() * 1000)

        if not signals:
            return CompositeSignal(
                symbol=symbol, timestamp_ms=now_ms,
                direction=Direction.NEUTRAL, confidence=0.0, score=0.0,
            )

        # Count bullish horizons
        bullish = []
        for name, sig in signals.items():
            if sig.direction in (Direction.BUY, Direction.STRONG_BUY):
                bullish.append(name)

        alignment = len(bullish)
        total = len(signals)
        alignment_pct = alignment / total if total > 0 else 0.0

        # Weighted score
        weighted_score = 0.0
        total_weight = 0.0
        for name, sig in signals.items():
            horizon = sig.horizon
            config = HORIZON_CONFIG.get(horizon, {})
            weight = config.get("score_weight_in_composite", 0.2)
            if sig.direction in (Direction.BUY, Direction.STRONG_BUY):
                weighted_score += sig.score * weight
            elif sig.direction in (Direction.SELL, Direction.STRONG_SELL):
                weighted_score -= sig.score * weight * 0.5
            total_weight += weight

        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Cascade bonus: alignment across horizons
        cascade_bonus = 0.0
        if alignment >= 4:
            cascade_bonus = 15.0
        elif alignment >= 3:
            cascade_bonus = 10.0
        elif alignment >= 2:
            cascade_bonus = 5.0

        final_score = avg_score + cascade_bonus

        # Determine direction
        if alignment >= 3 and avg_score > 30:
            direction = Direction.STRONG_BUY
        elif alignment >= 2 and avg_score > 20:
            direction = Direction.BUY
        else:
            direction = Direction.NEUTRAL

        confidence = min(1.0, max(0.0, alignment_pct * 0.6 + (avg_score / 80) * 0.4))

        # Best entry horizon: prefer intraday, but scalp if very strong
        best_entry = "intraday"
        scalp_sig = signals.get("scalp")
        if scalp_sig and scalp_sig.score > 60:
            best_entry = "scalp"

        return CompositeSignal(
            symbol=symbol,
            timestamp_ms=now_ms,
            direction=direction,
            confidence=confidence,
            score=final_score,
            horizon_signals=signals,
            alignment_count=alignment,
            alignment_pct=alignment_pct,
            best_entry_horizon=best_entry,
            cascade_bonus=cascade_bonus,
        )

    def get_entry_bonus(self, symbol: str) -> float:
        """
        Get score bonus from multi-horizon alignment.
        Add this to existing entry score in monitor.py.
        """
        composite = self.composite_signal(symbol)
        return composite.cascade_bonus


def _safe(feat: dict, key: str, i: int, default: float) -> float:
    arr = feat.get(key)
    if arr is not None and 0 <= i < len(arr) and np.isfinite(arr[i]):
        return float(arr[i])
    return default
