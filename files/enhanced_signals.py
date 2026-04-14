from __future__ import annotations

"""
Enhanced Signals Integration Layer.

Bridges the new data modules (WebSocket, order flow, derivatives, regime,
top gainer model, multi-horizon) with the existing monitor.py entry flow.

This module is imported by monitor.py and provides:
  - EnhancedContext: per-tick context enriched with all new data sources
  - score_adjustment(): computes total score bonus/penalty from new signals
  - should_block_entry(): regime/order-flow based entry gating
  - startup/shutdown helpers for all new background services

Design principle: additive integration. The existing rule-based pipeline
remains primary. New signals add score bonuses/penalties and optional vetoes.
The monitor calls `enhance_entry_decision()` AFTER its existing score
computation and BEFORE the final entry/block decision.

Usage in monitor.py:
    from enhanced_signals import EnhancedSignals

    # At startup:
    enhanced = EnhancedSignals()
    await enhanced.start(symbols=watchlist)

    # In _poll_coin, after computing candidate_score:
    ctx = enhanced.get_context(sym)
    adj = enhanced.score_adjustment(sym, tf, mode, candidate_score, feat, data, i)
    candidate_score += adj.total_bonus
    if adj.block_reason:
        # log and return
        ...
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import config

log = logging.getLogger(__name__)


@dataclass
class ScoreAdjustment:
    """Result of enhanced signal scoring."""
    # Individual bonuses/penalties
    order_flow_bonus: float = 0.0
    derivatives_bonus: float = 0.0
    regime_bonus: float = 0.0
    top_gainer_bonus: float = 0.0
    multi_horizon_bonus: float = 0.0

    # Aggregate
    total_bonus: float = 0.0

    # Optional block
    block_reason: str = ""
    is_blocked: bool = False

    # Metadata for logging
    details: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "of_bonus": round(self.order_flow_bonus, 2),
            "deriv_bonus": round(self.derivatives_bonus, 2),
            "regime_bonus": round(self.regime_bonus, 2),
            "tg_bonus": round(self.top_gainer_bonus, 2),
            "mh_bonus": round(self.multi_horizon_bonus, 2),
            "total": round(self.total_bonus, 2),
            "blocked": self.is_blocked,
            "block_reason": self.block_reason,
        }


class EnhancedSignals:
    """
    Manages all enhanced signal sources and provides score adjustments.

    Lifecycle:
      1. await enhanced.start(symbols)  — start WS, derivatives, etc.
      2. enhanced.score_adjustment(...)  — called per entry candidate
      3. await enhanced.stop()           — graceful shutdown
    """

    def __init__(self):
        self._ws_manager = None
        self._order_flow = None
        self._derivatives = None
        self._regime_detector = None
        self._top_gainer_model = None
        self._horizon_manager = None
        self._started = False
        self._current_regime = None

    async def start(self, symbols: List[str]) -> None:
        """Initialize and start all enhanced data sources."""
        if self._started:
            return

        syms = [s.upper() for s in symbols]
        log.info("EnhancedSignals: starting for %d symbols", len(syms))

        # ── WebSocket Manager ────────────────────────────────────────────────
        if getattr(config, "WS_ENABLED", True):
            try:
                from ws_manager import WSManager
                self._ws_manager = WSManager(
                    symbols=syms,
                    enable_trades=getattr(config, "WS_ENABLE_TRADES", True),
                    enable_book=getattr(config, "WS_ENABLE_BOOK", True),
                    enable_klines=getattr(config, "WS_ENABLE_KLINES_1M", True),
                )
                await self._ws_manager.start()
                log.info("EnhancedSignals: WebSocket started")
            except Exception:
                log.exception("EnhancedSignals: WebSocket start failed")
                self._ws_manager = None

        # ── Order Flow Analyzer ──────────────────────────────────────────────
        if getattr(config, "ORDER_FLOW_ENABLED", True) and self._ws_manager:
            try:
                from order_flow import OrderFlowAnalyzer
                self._order_flow = OrderFlowAnalyzer(self._ws_manager)
                log.info("EnhancedSignals: OrderFlow ready")
            except Exception:
                log.exception("EnhancedSignals: OrderFlow init failed")

        # ── Derivatives Data ─────────────────────────────────────────────────
        if getattr(config, "DERIVATIVES_ENABLED", True):
            try:
                from derivatives_data import DerivativesData
                self._derivatives = DerivativesData()
                await self._derivatives.start_background_fetch(
                    symbols=syms,
                    funding_interval=getattr(config, "DERIVATIVES_FUNDING_INTERVAL", 60),
                    oi_interval=getattr(config, "DERIVATIVES_OI_INTERVAL", 60),
                    ls_interval=getattr(config, "DERIVATIVES_LS_INTERVAL", 300),
                )
                log.info("EnhancedSignals: Derivatives started")
            except Exception:
                log.exception("EnhancedSignals: Derivatives start failed")
                self._derivatives = None

        # ── Regime Detector ──────────────────────────────────────────────────
        if getattr(config, "REGIME_DETECTOR_ENABLED", True):
            try:
                from regime_detector import RegimeDetector
                model_file = getattr(config, "REGIME_HMM_MODEL_FILE", "regime_model.json")
                self._regime_detector = RegimeDetector(model_path=model_file)
                log.info("EnhancedSignals: RegimeDetector ready")
            except Exception:
                log.exception("EnhancedSignals: RegimeDetector init failed")

        # ── Top Gainer Model ─────────────────────────────────────────────────
        if getattr(config, "TOP_GAINER_MODEL_ENABLED", True):
            try:
                from top_gainer_model import TopGainerModel
                model_file = getattr(config, "TOP_GAINER_MODEL_FILE", "top_gainer_model.json")
                self._top_gainer_model = TopGainerModel(model_path=model_file)
                log.info("EnhancedSignals: TopGainerModel ready")
            except Exception:
                log.exception("EnhancedSignals: TopGainerModel init failed")

        # ── Multi-Horizon Manager ────────────────────────────────────────────
        if getattr(config, "MULTI_HORIZON_ENABLED", True):
            try:
                from multi_horizon import HorizonManager
                self._horizon_manager = HorizonManager()
                log.info("EnhancedSignals: HorizonManager ready")
            except Exception:
                log.exception("EnhancedSignals: HorizonManager init failed")

        self._started = True
        log.info("EnhancedSignals: all modules started")

    async def stop(self) -> None:
        """Stop all background services."""
        if self._ws_manager:
            await self._ws_manager.stop()
        if self._derivatives:
            await self._derivatives.stop()
        self._started = False
        log.info("EnhancedSignals: stopped")

    # ── Main API ─────────────────────────────────────────────────────────────

    def score_adjustment(
        self,
        sym: str,
        tf: str,
        mode: str,
        current_score: float,
        feat: dict,
        data: np.ndarray,
        i: int,
        *,
        is_bull_day: bool = False,
        report: Optional[Any] = None,
    ) -> ScoreAdjustment:
        """
        Compute total score adjustment from all enhanced signals.

        Called after existing candidate_score is computed but before
        final entry/block decision.
        """
        adj = ScoreAdjustment()

        # ── Order Flow ───────────────────────────────────────────────────────
        if self._order_flow and getattr(config, "ORDER_FLOW_SCORE_ENABLED", True):
            try:
                adj.order_flow_bonus = self._compute_order_flow_bonus(sym, mode)
                adj.details["order_flow"] = self._get_order_flow_summary(sym)
            except Exception:
                log.debug("OrderFlow scoring failed for %s", sym)

        # ── Derivatives ──────────────────────────────────────────────────────
        if self._derivatives and getattr(config, "DERIVATIVES_ENABLED", True):
            try:
                adj.derivatives_bonus = self._compute_derivatives_bonus(sym, feat, i)
                adj.details["derivatives"] = self._get_derivatives_summary(sym)
            except Exception:
                log.debug("Derivatives scoring failed for %s", sym)

        # ── Regime ───────────────────────────────────────────────────────────
        if self._regime_detector and getattr(config, "REGIME_ENTRY_GATING_ENABLED", True):
            try:
                regime_result = self._compute_regime_bonus(sym, mode, feat, data, i)
                adj.regime_bonus = regime_result[0]
                if regime_result[1]:
                    adj.block_reason = regime_result[1]
                    adj.is_blocked = True
                adj.details["regime"] = self._get_regime_summary()
            except Exception:
                log.debug("Regime scoring failed for %s", sym)

        # ── Top Gainer ───────────────────────────────────────────────────────
        if self._top_gainer_model and getattr(config, "TOP_GAINER_MODEL_ENABLED", True):
            try:
                adj.top_gainer_bonus = self._compute_top_gainer_bonus(
                    sym, feat, data, i, report
                )
                adj.details["top_gainer"] = self._get_top_gainer_summary(sym)
            except Exception:
                log.debug("TopGainer scoring failed for %s", sym)

        # ── Multi-Horizon Cascade ────────────────────────────────────────────
        if self._horizon_manager and getattr(config, "MULTI_HORIZON_CASCADE_BONUS_ENABLED", True):
            try:
                adj.multi_horizon_bonus = self._horizon_manager.get_entry_bonus(sym)
            except Exception:
                log.debug("MultiHorizon scoring failed for %s", sym)

        # ── Total ────────────────────────────────────────────────────────────
        adj.total_bonus = (
            adj.order_flow_bonus
            + adj.derivatives_bonus
            + adj.regime_bonus
            + adj.top_gainer_bonus
            + adj.multi_horizon_bonus
        )

        return adj

    def get_order_flow_features(self, sym: str) -> Dict[str, float]:
        """Get order flow features for ML models."""
        if not self._order_flow:
            return {}
        try:
            windows = getattr(config, "ORDER_FLOW_WINDOWS", (1, 5, 15))
            return self._order_flow.compute_all_features(sym, windows=windows)
        except Exception:
            return {}

    def get_derivatives_features(self, sym: str) -> Dict[str, float]:
        """Get derivatives features for ML models."""
        if not self._derivatives:
            return {}
        try:
            return self._derivatives.get_features(sym)
        except Exception:
            return {}

    def get_regime_state(self):
        """Get current regime state."""
        return self._current_regime

    def update_regime(self, btc_data: np.ndarray, btc_features: dict) -> None:
        """Update regime from BTC data (call once per monitoring tick)."""
        if self._regime_detector:
            try:
                self._current_regime = self._regime_detector.detect(
                    btc_data, btc_features
                )
            except Exception:
                log.debug("Regime update failed")

    @property
    def ws_stats(self) -> Dict[str, Any]:
        """WebSocket connection stats."""
        return self._ws_manager.stats if self._ws_manager else {}

    # ── Private scoring methods ──────────────────────────────────────────────

    def _compute_order_flow_bonus(self, sym: str, mode: str) -> float:
        """Order flow score bonus/penalty."""
        metrics = self._order_flow.compute(sym, window_minutes=5)
        if not metrics:
            return 0.0

        bonus = 0.0

        # Bullish confirmation
        cvd_min = getattr(config, "ORDER_FLOW_BULL_CVD_MIN", 3.0)
        imb_min = getattr(config, "ORDER_FLOW_BULL_IMBALANCE_MIN", 0.55)
        if metrics.cvd_pct > cvd_min and metrics.imbalance > imb_min:
            bonus += getattr(config, "ORDER_FLOW_BULL_BONUS", 8.0)

        # Bearish divergence (selling pressure despite BUY signal)
        cvd_max = getattr(config, "ORDER_FLOW_BEAR_CVD_MAX", -3.0)
        if metrics.cvd_pct < cvd_max and metrics.imbalance < 0.45:
            bonus -= getattr(config, "ORDER_FLOW_BEAR_PENALTY", 6.0)

        # Breakout signal from order flow
        if metrics.breakout_signal > 0.5:
            bonus += getattr(config, "ORDER_FLOW_BREAKOUT_BONUS", 10.0)

        return bonus

    def _compute_derivatives_bonus(
        self, sym: str, feat: dict, i: int,
    ) -> float:
        """Derivatives score bonus/penalty."""
        snap = self._derivatives.get_snapshot(sym)
        if not snap:
            return 0.0

        bonus = 0.0

        # OI expansion + uptrend
        if snap.oi and snap.oi.oi_change_1h_pct > 5.0:
            # Confirm uptrend: check price vs EMA
            ema = feat.get("ema_fast")
            price_arr = feat.get("close") if "close" in feat else None
            if ema is not None and i < len(ema):
                bonus += getattr(config, "DERIV_OI_EXPANSION_BONUS", 5.0)

        # Funding flip
        if snap.funding:
            if (len(snap.funding_history) >= 4
                    and snap.funding_history[-1].rate > 0
                    and snap.funding_history[-4].rate < 0):
                bonus += getattr(config, "DERIV_FUNDING_FLIP_BONUS", 4.0)

            # High funding penalty (crowded long)
            if snap.funding.rate > 0.001:
                bonus -= getattr(config, "DERIV_HIGH_FUNDING_PENALTY", 3.0)

        # Short squeeze potential
        if snap.ls_ratio and snap.ls_ratio.ls_ratio < 0.8:
            bonus += getattr(config, "DERIV_SHORT_SQUEEZE_BONUS", 5.0)

        return bonus

    def _compute_regime_bonus(
        self, sym: str, mode: str, feat: dict, data: np.ndarray, i: int,
    ) -> tuple:
        """Returns (bonus, block_reason_or_empty)."""
        if not self._current_regime:
            return (0.0, "")

        regime = self._current_regime
        bonus = regime.score_bonus

        # Block in risk-off regimes
        if regime.is_risk_off and getattr(config, "REGIME_RISK_OFF_BLOCK", True):
            return (bonus, f"regime {regime.name}: risk-off, no entries")

        # Check if mode is allowed in current regime
        allowed = regime.allowed_entry_modes
        if allowed and mode not in allowed:
            return (bonus - 10.0, f"mode '{mode}' not allowed in regime {regime.name}")

        return (bonus, "")

    def _compute_top_gainer_bonus(
        self,
        sym: str,
        feat: dict,
        data: np.ndarray,
        i: int,
        report: Optional[Any],
    ) -> float:
        """Top gainer prediction score bonus."""
        if not self._top_gainer_model:
            return 0.0

        # Compute features
        of_features = self.get_order_flow_features(sym)
        dd_features = self.get_derivatives_features(sym)

        features = self._top_gainer_model.compute_features(
            symbol=sym,
            klines_1h=data,  # approximate: use whatever TF data we have
            klines_15m=None,
            features_1h=feat,
            features_15m=None,
            order_flow_features=of_features,
            derivatives_features=dd_features,
        )

        prediction = self._top_gainer_model.predict(features)
        return prediction.score_bonus

    def _get_order_flow_summary(self, sym: str) -> Dict[str, Any]:
        """Summary for logging."""
        if not self._order_flow:
            return {}
        m = self._order_flow.compute(sym, window_minutes=5)
        if not m:
            return {}
        return {
            "cvd_pct": round(m.cvd_pct, 2),
            "imbalance": round(m.imbalance, 3),
            "large_trades": m.large_trade_count,
            "tps": round(m.trades_per_sec, 1),
            "breakout": round(m.breakout_signal, 2),
        }

    def _get_derivatives_summary(self, sym: str) -> Dict[str, Any]:
        """Summary for logging."""
        if not self._derivatives:
            return {}
        snap = self._derivatives.get_snapshot(sym)
        if not snap:
            return {}
        result = {}
        if snap.funding:
            result["funding"] = round(snap.funding.rate, 6)
        if snap.oi:
            result["oi_change_1h"] = round(snap.oi.oi_change_1h_pct, 2)
        if snap.ls_ratio:
            result["ls_ratio"] = round(snap.ls_ratio.ls_ratio, 2)
        result["liq_1h"] = round(snap.liq_buy_volume_1h + snap.liq_sell_volume_1h, 0)
        return result

    def _get_regime_summary(self) -> Dict[str, Any]:
        """Summary for logging."""
        if not self._current_regime:
            return {}
        return {
            "name": self._current_regime.name,
            "prob": round(self._current_regime.probability, 2),
            "age_bars": self._current_regime.regime_age_bars,
        }

    def _get_top_gainer_summary(self, sym: str) -> Dict[str, Any]:
        """Summary for logging."""
        return {}  # filled during compute_top_gainer_bonus


# ── Singleton ────────────────────────────────────────────────────────────────
_enhanced: Optional[EnhancedSignals] = None


def get_enhanced_signals() -> Optional[EnhancedSignals]:
    """Get global instance (None if not started)."""
    return _enhanced


async def start_enhanced_signals(symbols: List[str]) -> EnhancedSignals:
    """Start global enhanced signals instance."""
    global _enhanced
    if _enhanced is None:
        _enhanced = EnhancedSignals()
    if not _enhanced._started:
        await _enhanced.start(symbols)
    return _enhanced


async def stop_enhanced_signals() -> None:
    """Stop global instance."""
    global _enhanced
    if _enhanced:
        await _enhanced.stop()
        _enhanced = None
