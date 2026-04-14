from __future__ import annotations

"""
Order Flow Analyzer — real-time trade flow metrics.

Computes from raw aggTrade stream (via WSManager):
  - CVD (Cumulative Volume Delta) = Σ(buy_vol - sell_vol)
  - Trade imbalance ratio         = buy_count / total_count
  - Large trade detection         = trades > K × σ from mean size
  - Volume-weighted price (VWAP)  = Σ(price × qty) / Σ(qty)
  - Buy/sell pressure ratio       = buy_vol / sell_vol
  - Trade velocity                = trades per second (rolling)

These are LEADING indicators — they precede price movement by 1-30 minutes.

Usage:
    from ws_manager import WSManager
    from order_flow import OrderFlowAnalyzer

    analyzer = OrderFlowAnalyzer(ws_manager)
    metrics = analyzer.compute("BTCUSDT", window_minutes=5)
    # metrics.cvd, metrics.imbalance, metrics.large_trade_count, ...
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OrderFlowMetrics:
    """Order flow metrics for a symbol over a time window."""
    symbol: str
    window_minutes: int
    timestamp_ms: int

    # Core metrics
    cvd: float                    # Cumulative Volume Delta (+ = buying pressure)
    cvd_pct: float                # CVD as % of total volume
    imbalance: float              # Buy ratio: [0, 1], >0.55 = buying pressure
    buy_volume: float             # Total buy volume (base asset)
    sell_volume: float            # Total sell volume (base asset)
    total_volume: float           # Total volume
    trade_count: int              # Number of trades in window
    trades_per_sec: float         # Trade velocity

    # Large trades
    large_trade_count: int        # Trades > 2σ above mean
    large_buy_volume: float       # Volume from large buys
    large_sell_volume: float      # Volume from large sells
    large_trade_ratio: float      # large_vol / total_vol

    # Price-based
    vwap: float                   # Volume-weighted average price
    price_vs_vwap_pct: float      # Current price vs VWAP (%)

    # Derived signals
    absorption_signal: float      # Price flat but high volume = absorption
    exhaustion_signal: float      # Large CVD but price didn't move = exhaustion
    breakout_signal: float        # CVD + large trades + velocity spike

    @property
    def is_bullish(self) -> bool:
        return self.cvd > 0 and self.imbalance > 0.52

    @property
    def is_strong_bullish(self) -> bool:
        return self.cvd_pct > 5.0 and self.imbalance > 0.55 and self.large_trade_ratio > 0.3

    def to_features(self) -> Dict[str, float]:
        """Convert to flat feature dict for ML models."""
        return {
            "of_cvd_pct": self.cvd_pct,
            "of_imbalance": self.imbalance,
            "of_trades_per_sec": self.trades_per_sec,
            "of_large_trade_count": float(self.large_trade_count),
            "of_large_trade_ratio": self.large_trade_ratio,
            "of_large_buy_ratio": (
                self.large_buy_volume / (self.large_buy_volume + self.large_sell_volume)
                if (self.large_buy_volume + self.large_sell_volume) > 0 else 0.5
            ),
            "of_price_vs_vwap_pct": self.price_vs_vwap_pct,
            "of_absorption": self.absorption_signal,
            "of_exhaustion": self.exhaustion_signal,
            "of_breakout": self.breakout_signal,
            "of_buy_sell_pressure": (
                self.buy_volume / self.sell_volume
                if self.sell_volume > 0 else 2.0
            ),
        }


class OrderFlowAnalyzer:
    """
    Computes order flow metrics from WSManager trade buffers.

    No state beyond what WSManager holds — purely computational.
    Call compute() on each monitoring tick or on-demand.
    """

    def __init__(self, ws_manager):
        self._ws = ws_manager
        # Cache for σ computation (per-symbol rolling stats)
        self._trade_size_stats: Dict[str, tuple] = {}  # sym → (mean, std, update_ts)

    def compute(
        self,
        symbol: str,
        window_minutes: int = 5,
        large_trade_sigma: float = 2.0,
    ) -> Optional[OrderFlowMetrics]:
        """
        Compute order flow metrics for symbol over last `window_minutes`.

        Returns None if insufficient data (< 10 trades).
        """
        trades = self._ws.get_trades(symbol, limit=TRADE_BUFFER_LIMIT)
        if len(trades) < 10:
            return None

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_minutes * 60 * 1000

        # Filter to window
        window_trades = [t for t in trades if t.t >= cutoff_ms]
        if len(window_trades) < 5:
            return None

        # ── Core metrics ─────────────────────────────────────────────────────
        prices = np.array([t.price for t in window_trades])
        qtys = np.array([t.qty for t in window_trades])
        is_buy = np.array([t.is_buyer for t in window_trades])

        buy_mask = is_buy
        sell_mask = ~is_buy

        buy_vol = float(np.sum(qtys[buy_mask])) if buy_mask.any() else 0.0
        sell_vol = float(np.sum(qtys[sell_mask])) if sell_mask.any() else 0.0
        total_vol = buy_vol + sell_vol
        cvd = buy_vol - sell_vol
        cvd_pct = (cvd / total_vol * 100.0) if total_vol > 0 else 0.0

        n_trades = len(window_trades)
        n_buys = int(buy_mask.sum())
        imbalance = n_buys / n_trades if n_trades > 0 else 0.5

        time_span_s = max(1, (window_trades[-1].t - window_trades[0].t) / 1000.0)
        trades_per_sec = n_trades / time_span_s

        # ── VWAP ─────────────────────────────────────────────────────────────
        notional = prices * qtys
        vwap = float(np.sum(notional) / np.sum(qtys)) if np.sum(qtys) > 0 else float(prices[-1])
        current_price = float(prices[-1])
        price_vs_vwap = (current_price - vwap) / vwap * 100.0 if vwap > 0 else 0.0

        # ── Large trade detection ────────────────────────────────────────────
        trade_sizes = prices * qtys  # notional size
        mean_size = float(np.mean(trade_sizes))
        std_size = float(np.std(trade_sizes))
        threshold = mean_size + large_trade_sigma * std_size if std_size > 0 else mean_size * 3

        large_mask = trade_sizes > threshold
        large_count = int(large_mask.sum())
        large_buy_vol = float(np.sum(qtys[large_mask & buy_mask])) if (large_mask & buy_mask).any() else 0.0
        large_sell_vol = float(np.sum(qtys[large_mask & sell_mask])) if (large_mask & sell_mask).any() else 0.0
        large_vol = large_buy_vol + large_sell_vol
        large_ratio = large_vol / total_vol if total_vol > 0 else 0.0

        # Update rolling stats cache
        self._trade_size_stats[symbol] = (mean_size, std_size, now_ms)

        # ── Derived signals ──────────────────────────────────────────────────
        price_change_pct = abs(
            (float(prices[-1]) - float(prices[0])) / float(prices[0]) * 100.0
        ) if prices[0] > 0 else 0.0

        # Absorption: high volume but price didn't move much
        # (someone is absorbing selling/buying pressure)
        absorption = 0.0
        if total_vol > 0 and price_change_pct < 0.1 and trades_per_sec > 2.0:
            absorption = min(1.0, trades_per_sec / 10.0)

        # Exhaustion: large CVD but price barely moved
        # (buyers/sellers running out of steam)
        exhaustion = 0.0
        if abs(cvd_pct) > 10.0 and price_change_pct < 0.2:
            exhaustion = min(1.0, abs(cvd_pct) / 30.0)

        # Breakout: CVD + large trades + velocity all spiking
        breakout = 0.0
        if (cvd_pct > 3.0 and large_count >= 2
                and trades_per_sec > 3.0 and price_change_pct > 0.3):
            breakout = min(1.0, (cvd_pct / 10.0 + large_ratio + trades_per_sec / 10.0) / 3.0)

        return OrderFlowMetrics(
            symbol=symbol,
            window_minutes=window_minutes,
            timestamp_ms=now_ms,
            cvd=cvd,
            cvd_pct=cvd_pct,
            imbalance=imbalance,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            total_volume=total_vol,
            trade_count=n_trades,
            trades_per_sec=trades_per_sec,
            large_trade_count=large_count,
            large_buy_volume=large_buy_vol,
            large_sell_volume=large_sell_vol,
            large_trade_ratio=large_ratio,
            vwap=vwap,
            price_vs_vwap_pct=price_vs_vwap,
            absorption_signal=absorption,
            exhaustion_signal=exhaustion,
            breakout_signal=breakout,
        )

    def compute_multi_window(
        self,
        symbol: str,
        windows: tuple = (1, 5, 15, 60),
    ) -> Dict[str, OrderFlowMetrics]:
        """Compute metrics for multiple time windows. Returns {f"{w}m": metrics}."""
        results = {}
        for w in windows:
            m = self.compute(symbol, window_minutes=w)
            if m:
                results[f"{w}m"] = m
        return results

    def compute_all_features(
        self,
        symbol: str,
        windows: tuple = (1, 5, 15),
    ) -> Dict[str, float]:
        """
        Flat feature dict for ML models across all windows.
        Keys: of_{window}m_{metric_name}
        """
        features: Dict[str, float] = {}
        for w in windows:
            m = self.compute(symbol, window_minutes=w)
            if m:
                for k, v in m.to_features().items():
                    features[f"{k}_{w}m"] = v
        return features

    def get_cvd_series(
        self,
        symbol: str,
        bucket_seconds: int = 60,
        lookback_minutes: int = 60,
    ) -> Optional[np.ndarray]:
        """
        Get CVD time series bucketed by `bucket_seconds`.
        Returns array of shape (N, 3): [timestamp_ms, cvd_bucket, cumulative_cvd].
        Useful for plotting and trend detection.
        """
        trades = self._ws.get_trades(symbol, limit=TRADE_BUFFER_LIMIT)
        if len(trades) < 10:
            return None

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - lookback_minutes * 60 * 1000
        window_trades = [t for t in trades if t.t >= cutoff_ms]
        if not window_trades:
            return None

        bucket_ms = bucket_seconds * 1000
        start_ms = (window_trades[0].t // bucket_ms) * bucket_ms
        end_ms = now_ms
        n_buckets = max(1, int((end_ms - start_ms) / bucket_ms) + 1)

        result = np.zeros((n_buckets, 3))
        for i in range(n_buckets):
            result[i, 0] = start_ms + i * bucket_ms

        for t in window_trades:
            idx = int((t.t - start_ms) / bucket_ms)
            if 0 <= idx < n_buckets:
                delta = t.qty if t.is_buyer else -t.qty
                result[idx, 1] += delta

        # Cumulative
        result[:, 2] = np.cumsum(result[:, 1])
        return result


# Buffer limit for fetching trades from WSManager
TRADE_BUFFER_LIMIT = 5000
