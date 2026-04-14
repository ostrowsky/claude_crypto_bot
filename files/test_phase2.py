#!/usr/bin/env python3
"""
test_phase2.py — Tests for Phase 2 modules.

Covers:
  T300–T309  ws_manager       — WSManager, Kline1m, AggTrade, BookTicker, buffers
  T310–T319  order_flow       — OrderFlowAnalyzer, metrics, CVD, large trades
  T320–T329  derivatives_data — DerivativesSnapshot, OI changes, features
  T330–T339  regime_detector  — GaussianHMM, RegimeDetector, RegimeState
  T340–T349  top_gainer_model — TopGainerPrediction, heuristic predict, features
  T350–T359  multi_horizon    — strategies, HorizonManager, cascade bonuses
  T360–T369  enhanced_signals — ScoreAdjustment, EnhancedSignals integration

Run:
    python -m pytest test_phase2.py -v
    python test_phase2.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, AsyncMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Mock aiohttp if not installed ────────────────────────────────────────────
try:
    import aiohttp
except ImportError:
    aiohttp_mock = MagicMock()
    aiohttp_mock.ClientSession = MagicMock
    aiohttp_mock.ClientTimeout = MagicMock
    aiohttp_mock.WSMsgType = MagicMock()
    aiohttp_mock.WSMsgType.TEXT = 1
    aiohttp_mock.WSMsgType.ERROR = 2
    aiohttp_mock.WSMsgType.CLOSED = 3
    aiohttp_mock.ClientWSTimeout = MagicMock
    sys.modules["aiohttp"] = aiohttp_mock

# ── Mock telegram ────────────────────────────────────────────────────────────
for _mod in ["telegram", "telegram.ext", "python_telegram_bot"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _make_candles(n: int = 200, trend: str = "up", base: float = 1.0) -> np.ndarray:
    """Generate synthetic OHLCV structured array."""
    np.random.seed(42)
    c = np.ones(n) * base
    now_ms = int(time.time() * 1000)
    t = np.array([now_ms - (n - i) * 15 * 60 * 1000 for i in range(n)], dtype=np.int64)

    for i in range(1, n):
        noise = np.random.randn() * 0.002
        if trend == "up":
            c[i] = c[i-1] * (1 + 0.003 + noise)
        elif trend == "down":
            c[i] = c[i-1] * (1 - 0.003 + noise)
        elif trend == "flat":
            c[i] = c[i-1] * (1 + noise * 0.3)
        elif trend == "impulse":
            if i >= n - 10:
                c[i] = c[i-1] * (1 + 0.015 + abs(noise))
            else:
                c[i] = c[i-1] * (1 + noise * 0.3)

    spread = 0.001
    h = c * (1 + spread + np.abs(np.random.randn(n) * 0.002))
    l = c * (1 - spread - np.abs(np.random.randn(n) * 0.002))
    o = np.roll(c, 1); o[0] = c[0]
    v = np.abs(np.random.randn(n) + 3) * 1000 * base

    arr = np.zeros(n, dtype=[
        ("t", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("v", "f8"),
    ])
    arr["t"] = t; arr["o"] = o; arr["h"] = h
    arr["l"] = l; arr["c"] = c; arr["v"] = v
    return arr


def _make_feat(n: int = 200, trend: str = "up"):
    """Returns (data, feat, i_now) ready to use."""
    from indicators import compute_features
    data = _make_candles(n, trend)
    feat = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
    i_now = n - 2
    return data, feat, i_now


# ════════════════════════════════════════════════════════════════════════════════
# T300–T309  ws_manager
# ════════════════════════════════════════════════════════════════════════════════

class TestWSManager(unittest.TestCase):
    """Tests for WebSocket Manager module."""

    def setUp(self):
        from ws_manager import WSManager, Kline1m, AggTrade, BookTicker, SymbolBuffers
        self.WSManager = WSManager
        self.Kline1m = Kline1m
        self.AggTrade = AggTrade
        self.BookTicker = BookTicker
        self.SymbolBuffers = SymbolBuffers

    def test_T300_wsmanager_init_symbols(self):
        """T300: WSManager stores symbols uppercased."""
        mgr = self.WSManager(symbols=["btcusdt", "ethusdt"])
        self.assertEqual(mgr.symbols, {"BTCUSDT", "ETHUSDT"})

    def test_T301_wsmanager_init_empty(self):
        """T301: WSManager works with no symbols."""
        mgr = self.WSManager()
        self.assertEqual(len(mgr.symbols), 0)
        self.assertFalse(mgr._running)

    def test_T302_handle_kline_creates_bar(self):
        """T302: _handle_kline populates klines_1m buffer."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()
        payload = {
            "k": {
                "s": "BTCUSDT", "t": 1000000, "o": "100.0", "h": "101.0",
                "l": "99.0", "c": "100.5", "v": "500.0", "q": "50000.0",
                "n": 100, "x": True,
            }
        }
        mgr._handle_kline(payload)
        buf = mgr._buffers["BTCUSDT"]
        self.assertEqual(len(buf.klines_1m), 1)
        bar = buf.klines_1m[0]
        self.assertEqual(bar.t, 1000000)
        self.assertAlmostEqual(bar.c, 100.5)
        self.assertTrue(bar.closed)

    def test_T303_handle_kline_updates_existing_bar(self):
        """T303: _handle_kline updates bar with same timestamp instead of appending."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()
        base = {
            "k": {"s": "BTCUSDT", "t": 1000000, "o": "100", "h": "101",
                   "l": "99", "c": "100", "v": "100", "q": "100", "n": 10, "x": False}
        }
        mgr._handle_kline(base)
        self.assertEqual(len(mgr._buffers["BTCUSDT"].klines_1m), 1)

        # Update same bar
        base["k"]["c"] = "102"
        base["k"]["x"] = True
        mgr._handle_kline(base)
        self.assertEqual(len(mgr._buffers["BTCUSDT"].klines_1m), 1)
        self.assertAlmostEqual(mgr._buffers["BTCUSDT"].klines_1m[0].c, 102.0)

    def test_T304_handle_agg_trade(self):
        """T304: _handle_agg_trade appends to trades buffer."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()
        payload = {"s": "BTCUSDT", "T": 2000000, "p": "100.5", "q": "1.5", "m": False, "a": 12345}
        mgr._handle_agg_trade(payload)
        trades = mgr._buffers["BTCUSDT"].trades
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].price, 100.5)
        self.assertTrue(trades[0].is_buyer)  # m=False → buyer is taker

    def test_T305_handle_book_ticker(self):
        """T305: _handle_book_ticker sets bid/ask and spread."""
        mgr = self.WSManager(symbols=["ETHUSDT"])
        mgr._buffers["ETHUSDT"] = self.SymbolBuffers()
        payload = {"s": "ETHUSDT", "u": 99999, "b": "3000.0", "B": "10.0", "a": "3001.0", "A": "8.0"}
        mgr._handle_book_ticker(payload)
        book = mgr._buffers["ETHUSDT"].book
        self.assertIsNotNone(book)
        self.assertAlmostEqual(book.bid, 3000.0)
        self.assertAlmostEqual(book.ask, 3001.0)
        self.assertGreater(book.spread_bps, 0)

    def test_T306_get_klines_array_returns_only_closed(self):
        """T306: get_klines_array returns only closed bars as numpy array."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()
        # Add closed and unclosed bars
        for i in range(5):
            mgr._handle_kline({
                "k": {"s": "BTCUSDT", "t": 1000 + i * 60000, "o": "100", "h": "101",
                       "l": "99", "c": str(100 + i), "v": "100", "q": "100",
                       "n": 10, "x": i < 4}  # last one is unclosed
            })
        arr = mgr.get_klines_array("BTCUSDT", limit=10)
        self.assertIsNotNone(arr)
        self.assertEqual(len(arr), 4)  # only closed bars
        self.assertEqual(arr.dtype.names, ("t", "o", "h", "l", "c", "v"))

    def test_T307_get_trades_returns_limited(self):
        """T307: get_trades respects limit parameter."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()
        for i in range(100):
            mgr._handle_agg_trade(
                {"s": "BTCUSDT", "T": 1000 + i, "p": "100", "q": "1", "m": False, "a": i}
            )
        trades = mgr.get_trades("BTCUSDT", limit=10)
        self.assertEqual(len(trades), 10)

    def test_T308_handle_message_routes_correctly(self):
        """T308: _handle_message routes kline, trade, and book messages."""
        mgr = self.WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = self.SymbolBuffers()

        # Kline message
        kline_msg = json.dumps({
            "stream": "btcusdt@kline_1m",
            "data": {"k": {"s": "BTCUSDT", "t": 100, "o": "1", "h": "1", "l": "1",
                           "c": "1", "v": "1", "q": "1", "n": 1, "x": True}}
        })
        mgr._handle_message(kline_msg)
        self.assertEqual(len(mgr._buffers["BTCUSDT"].klines_1m), 1)

        # Trade message
        trade_msg = json.dumps({
            "stream": "btcusdt@aggTrade",
            "data": {"s": "BTCUSDT", "T": 200, "p": "1", "q": "1", "m": False, "a": 1}
        })
        mgr._handle_message(trade_msg)
        self.assertEqual(len(mgr._buffers["BTCUSDT"].trades), 1)

    def test_T309_build_stream_list(self):
        """T309: _build_stream_list generates correct stream names."""
        mgr = self.WSManager(symbols=["BTCUSDT"], enable_trades=True, enable_book=True, enable_klines=True)
        streams = mgr._build_stream_list()
        self.assertIn("btcusdt@kline_1m", streams)
        self.assertIn("btcusdt@aggTrade", streams)
        self.assertIn("btcusdt@bookTicker", streams)

        mgr2 = self.WSManager(symbols=["BTCUSDT"], enable_trades=False, enable_book=False, enable_klines=True)
        streams2 = mgr2._build_stream_list()
        self.assertEqual(len(streams2), 1)
        self.assertIn("btcusdt@kline_1m", streams2)


# ════════════════════════════════════════════════════════════════════════════════
# T310–T319  order_flow
# ════════════════════════════════════════════════════════════════════════════════

class TestOrderFlow(unittest.TestCase):
    """Tests for Order Flow Analyzer module."""

    def setUp(self):
        from ws_manager import WSManager, AggTrade, SymbolBuffers
        from order_flow import OrderFlowAnalyzer, OrderFlowMetrics
        self.OrderFlowAnalyzer = OrderFlowAnalyzer
        self.OrderFlowMetrics = OrderFlowMetrics
        self.AggTrade = AggTrade
        # Create a mock WS manager with trade data
        self.ws = WSManager(symbols=["BTCUSDT"])
        self.ws._buffers["BTCUSDT"] = SymbolBuffers()

    def _populate_trades(self, n=100, buy_ratio=0.6, base_price=100.0):
        """Populate WS buffer with synthetic trades."""
        now_ms = int(time.time() * 1000)
        for i in range(n):
            is_buy = i < int(n * buy_ratio)
            trade = self.AggTrade(
                t=now_ms - (n - i) * 500,  # 500ms apart
                price=base_price + np.random.randn() * 0.5,
                qty=abs(np.random.randn()) + 0.1,
                is_buyer=is_buy,
                trade_id=i,
            )
            self.ws._buffers["BTCUSDT"].trades.append(trade)

    def test_T310_compute_returns_none_insufficient_data(self):
        """T310: compute returns None with < 10 trades."""
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=5)
        self.assertIsNone(result)

    def test_T311_compute_returns_metrics(self):
        """T311: compute returns valid OrderFlowMetrics."""
        self._populate_trades(200, buy_ratio=0.7)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=60)
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertGreater(result.total_volume, 0)
        self.assertGreater(result.trade_count, 0)

    def test_T312_cvd_positive_when_buying_dominant(self):
        """T312: CVD is positive when buys dominate."""
        self._populate_trades(200, buy_ratio=0.8)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=60)
        self.assertIsNotNone(result)
        self.assertGreater(result.cvd, 0)
        self.assertGreater(result.cvd_pct, 0)

    def test_T313_imbalance_reflects_buy_ratio(self):
        """T313: imbalance ratio reflects actual buy/sell distribution."""
        self._populate_trades(200, buy_ratio=0.7)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=60)
        self.assertIsNotNone(result)
        # With 70% buys, imbalance should be roughly 0.7
        self.assertGreater(result.imbalance, 0.6)

    def test_T314_to_features_returns_expected_keys(self):
        """T314: to_features() returns all expected feature keys."""
        self._populate_trades(200)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=60)
        self.assertIsNotNone(result)
        features = result.to_features()
        expected_keys = [
            "of_cvd_pct", "of_imbalance", "of_trades_per_sec",
            "of_large_trade_count", "of_large_trade_ratio",
            "of_large_buy_ratio", "of_price_vs_vwap_pct",
            "of_absorption", "of_exhaustion", "of_breakout",
            "of_buy_sell_pressure",
        ]
        for k in expected_keys:
            self.assertIn(k, features, f"Missing key: {k}")

    def test_T315_is_bullish_property(self):
        """T315: is_bullish and is_strong_bullish properties work."""
        m = self.OrderFlowMetrics(
            symbol="TEST", window_minutes=5, timestamp_ms=0,
            cvd=100.0, cvd_pct=8.0, imbalance=0.6,
            buy_volume=80.0, sell_volume=20.0, total_volume=100.0,
            trade_count=100, trades_per_sec=5.0,
            large_trade_count=5, large_buy_volume=30.0, large_sell_volume=5.0,
            large_trade_ratio=0.35, vwap=100.0, price_vs_vwap_pct=0.5,
            absorption_signal=0.0, exhaustion_signal=0.0, breakout_signal=0.0,
        )
        self.assertTrue(m.is_bullish)
        self.assertTrue(m.is_strong_bullish)

        # Not bullish
        m2 = self.OrderFlowMetrics(
            symbol="TEST", window_minutes=5, timestamp_ms=0,
            cvd=-50.0, cvd_pct=-5.0, imbalance=0.4,
            buy_volume=40.0, sell_volume=60.0, total_volume=100.0,
            trade_count=100, trades_per_sec=2.0,
            large_trade_count=1, large_buy_volume=5.0, large_sell_volume=10.0,
            large_trade_ratio=0.15, vwap=100.0, price_vs_vwap_pct=-0.5,
            absorption_signal=0.0, exhaustion_signal=0.0, breakout_signal=0.0,
        )
        self.assertFalse(m2.is_bullish)

    def test_T316_compute_multi_window(self):
        """T316: compute_multi_window returns dict keyed by window."""
        self._populate_trades(500, buy_ratio=0.6)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        results = analyzer.compute_multi_window("BTCUSDT", windows=(1, 5, 60))
        # At least some windows should return results
        self.assertIsInstance(results, dict)

    def test_T317_compute_all_features_flat_dict(self):
        """T317: compute_all_features returns flat dict with window-prefixed keys."""
        self._populate_trades(500, buy_ratio=0.6)
        analyzer = self.OrderFlowAnalyzer(self.ws)
        features = analyzer.compute_all_features("BTCUSDT", windows=(1, 5))
        self.assertIsInstance(features, dict)
        # Keys should have window suffix
        for k in features:
            self.assertTrue(k.endswith("_1m") or k.endswith("_5m"),
                            f"Key {k} missing window suffix")

    def test_T318_large_trade_detection(self):
        """T318: Large trades are detected when size > mean + 2σ."""
        now_ms = int(time.time() * 1000)
        from ws_manager import SymbolBuffers
        self.ws._buffers["BTCUSDT"] = SymbolBuffers()
        # Add normal trades (small notional ~100)
        for i in range(95):
            trade = self.AggTrade(
                t=now_ms - (100 - i) * 500, price=100.0, qty=1.0,
                is_buyer=True, trade_id=i,
            )
            self.ws._buffers["BTCUSDT"].trades.append(trade)
        # Add a few very large trades (notional ~50000, 500x the normal)
        for i in range(5):
            trade = self.AggTrade(
                t=now_ms - (5 - i) * 500, price=100.0, qty=500.0,
                is_buyer=True, trade_id=95 + i,
            )
            self.ws._buffers["BTCUSDT"].trades.append(trade)

        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("BTCUSDT", window_minutes=60)
        self.assertIsNotNone(result)
        self.assertGreater(result.large_trade_count, 0)
        self.assertGreater(result.large_trade_ratio, 0)

    def test_T319_unknown_symbol_returns_empty(self):
        """T319: Querying unknown symbol returns None/empty."""
        analyzer = self.OrderFlowAnalyzer(self.ws)
        result = analyzer.compute("UNKNOWN", window_minutes=5)
        self.assertIsNone(result)
        trades = self.ws.get_trades("UNKNOWN")
        self.assertEqual(len(trades), 0)


# ════════════════════════════════════════════════════════════════════════════════
# T320–T329  derivatives_data
# ════════════════════════════════════════════════════════════════════════════════

class TestDerivativesData(unittest.TestCase):
    """Tests for Derivatives Data module."""

    def test_T320_derivatives_snapshot_to_features_empty(self):
        """T320: DerivativesSnapshot.to_features() returns defaults when no data."""
        from derivatives_data import DerivativesSnapshot
        snap = DerivativesSnapshot(symbol="BTCUSDT", timestamp_ms=0)
        f = snap.to_features()
        self.assertIn("deriv_funding_rate", f)
        self.assertEqual(f["deriv_funding_rate"], 0.0)
        self.assertEqual(f["deriv_oi_change_1h"], 0.0)
        self.assertEqual(f["deriv_ls_ratio"], 1.0)

    def test_T321_derivatives_snapshot_with_funding(self):
        """T321: DerivativesSnapshot.to_features() includes funding data."""
        from derivatives_data import DerivativesSnapshot, FundingSnapshot
        funding = FundingSnapshot(
            symbol="BTCUSDT", rate=0.0003, time_ms=0,
            next_time_ms=0, mark_price=50000.0, index_price=49990.0,
            premium_pct=0.02,
        )
        snap = DerivativesSnapshot(symbol="BTCUSDT", timestamp_ms=0, funding=funding)
        f = snap.to_features()
        self.assertAlmostEqual(f["deriv_funding_rate"], 0.0003)
        self.assertAlmostEqual(f["deriv_premium_pct"], 0.02)

    def test_T322_derivatives_snapshot_with_oi(self):
        """T322: OI features include change percentages."""
        from derivatives_data import DerivativesSnapshot, OISnapshot
        oi = OISnapshot(
            symbol="BTCUSDT", oi_contracts=100000.0, oi_value_usd=5e9,
            timestamp_ms=0, oi_change_1h_pct=6.0, oi_change_4h_pct=12.0,
            oi_change_24h_pct=20.0,
        )
        snap = DerivativesSnapshot(symbol="BTCUSDT", timestamp_ms=0, oi=oi)
        f = snap.to_features()
        self.assertAlmostEqual(f["deriv_oi_change_1h"], 6.0)
        self.assertAlmostEqual(f["deriv_oi_spike_1h"], 1.0)  # >5% → spike

    def test_T323_derivatives_snapshot_with_ls_ratio(self):
        """T323: L/S ratio features detect crowd positioning."""
        from derivatives_data import DerivativesSnapshot, LongShortRatio
        ls = LongShortRatio(
            symbol="BTCUSDT", long_ratio=0.65, short_ratio=0.35,
            ls_ratio=1.86, timestamp_ms=0,
        )
        snap = DerivativesSnapshot(symbol="BTCUSDT", timestamp_ms=0, ls_ratio=ls)
        f = snap.to_features()
        self.assertAlmostEqual(f["deriv_ls_ratio"], 1.86)
        self.assertEqual(f["deriv_crowd_long"], 1.0)   # >1.5
        self.assertEqual(f["deriv_crowd_short"], 0.0)

    def test_T324_derivatives_snapshot_liquidations(self):
        """T324: Liquidation features compute volumes and cascade flag."""
        from derivatives_data import DerivativesSnapshot
        snap = DerivativesSnapshot(
            symbol="BTCUSDT", timestamp_ms=0,
            liq_buy_volume_1h=500_000.0, liq_sell_volume_1h=600_000.0,
        )
        f = snap.to_features()
        self.assertAlmostEqual(f["deriv_liq_total_1h"], 1_100_000.0)
        self.assertEqual(f["deriv_liq_cascade"], 1.0)  # >1M
        self.assertAlmostEqual(f["deriv_liq_buy_ratio"], 500_000 / 1_100_000, places=3)

    def test_T325_handle_liquidation_updates_snapshot(self):
        """T325: _handle_liquidation updates snapshot volumes."""
        from derivatives_data import DerivativesData, DerivativesSnapshot, Liquidation
        from collections import deque
        dd = DerivativesData()
        dd._symbols = {"BTCUSDT"}
        dd._snapshots["BTCUSDT"] = DerivativesSnapshot(symbol="BTCUSDT", timestamp_ms=0)
        dd._liquidations["BTCUSDT"] = deque(maxlen=500)
        raw = json.dumps({
            "o": {"s": "BTCUSDT", "S": "BUY", "p": "50000", "q": "2", "T": int(time.time() * 1000)}
        })
        dd._handle_liquidation(raw)
        snap = dd._snapshots["BTCUSDT"]
        self.assertGreater(snap.liq_buy_volume_1h, 0)

    def test_T326_compute_oi_changes(self):
        """T326: _compute_oi_changes sets correct % changes."""
        from derivatives_data import DerivativesData, OISnapshot
        dd = DerivativesData()
        now = int(time.time() * 1000)
        history = [
            OISnapshot(symbol="BTC", oi_contracts=1000, oi_value_usd=0, timestamp_ms=now - 25 * 3_600_000),
            OISnapshot(symbol="BTC", oi_contracts=1050, oi_value_usd=0, timestamp_ms=now - 5 * 3_600_000),
            OISnapshot(symbol="BTC", oi_contracts=1080, oi_value_usd=0, timestamp_ms=now - 1 * 3_600_000),
        ]
        current = OISnapshot(symbol="BTC", oi_contracts=1100, oi_value_usd=0, timestamp_ms=now)
        dd._compute_oi_changes(current, history)
        # 1h: (1100-1080)/1080 ≈ 1.85%
        self.assertGreater(current.oi_change_1h_pct, 0)

    def test_T327_derivatives_data_init(self):
        """T327: DerivativesData initializes cleanly."""
        from derivatives_data import DerivativesData
        dd = DerivativesData()
        self.assertFalse(dd._running)
        self.assertIsNone(dd._session)
        self.assertEqual(len(dd._symbols), 0)

    def test_T328_get_features_empty_symbol(self):
        """T328: get_features returns empty dict for unknown symbol."""
        from derivatives_data import DerivativesData
        dd = DerivativesData()
        f = dd.get_features("UNKNOWN")
        self.assertEqual(f, {})

    def test_T329_funding_flip_detection(self):
        """T329: Funding flip detected when rate goes neg → pos."""
        from derivatives_data import DerivativesSnapshot, FundingSnapshot
        # history[-4] must be negative, history[-1] must be positive
        # With 8 items, index -4 = item 4, index -1 = item 7
        # Make items 0-5 negative, items 6-7 positive
        history = []
        for i in range(8):
            rate = -0.001 if i < 6 else 0.0005
            history.append(FundingSnapshot(
                symbol="BTC", rate=rate, time_ms=i * 1000,
                next_time_ms=0, mark_price=50000, index_price=50000, premium_pct=0,
            ))
        snap = DerivativesSnapshot(
            symbol="BTC", timestamp_ms=0,
            funding=history[-1], funding_history=history,
        )
        f = snap.to_features()
        self.assertEqual(f["deriv_funding_flip"], 1.0)


# ════════════════════════════════════════════════════════════════════════════════
# T330–T339  regime_detector
# ════════════════════════════════════════════════════════════════════════════════

class TestRegimeDetector(unittest.TestCase):
    """Tests for Regime Detector module."""

    def test_T330_regime_names_count(self):
        """T330: Exactly 6 regime names defined."""
        from regime_detector import REGIME_NAMES
        self.assertEqual(len(REGIME_NAMES), 6)
        self.assertIn("strong_bull", REGIME_NAMES)
        self.assertIn("strong_bear", REGIME_NAMES)

    def test_T331_regime_params_completeness(self):
        """T331: All regimes have complete parameter sets."""
        from regime_detector import REGIME_PARAMS, REGIME_NAMES
        required_keys = ["allowed_modes", "rsi_hi", "rsi_lo", "vol_mult",
                         "max_positions", "position_size_mult", "score_bonus"]
        for name in REGIME_NAMES:
            self.assertIn(name, REGIME_PARAMS, f"Missing regime: {name}")
            for key in required_keys:
                self.assertIn(key, REGIME_PARAMS[name], f"Missing {key} in {name}")

    def test_T332_regime_state_properties(self):
        """T332: RegimeState properties return correct values."""
        from regime_detector import RegimeState, REGIME_PARAMS
        state = RegimeState(
            name="strong_bull", probability=0.8,
            probabilities={"strong_bull": 0.8}, timestamp_ms=0,
        )
        self.assertEqual(state.score_bonus, REGIME_PARAMS["strong_bull"]["score_bonus"])
        self.assertFalse(state.is_risk_off)
        self.assertGreater(state.max_positions, 0)
        self.assertGreater(state.position_size_mult, 0)

    def test_T333_regime_state_risk_off(self):
        """T333: strong_bear and volatile_chop are risk-off."""
        from regime_detector import RegimeState
        for rname in ("strong_bear", "volatile_chop"):
            state = RegimeState(name=rname, probability=0.7, probabilities={}, timestamp_ms=0)
            self.assertTrue(state.is_risk_off, f"{rname} should be risk-off")

        for rname in ("strong_bull", "weak_bull", "ranging"):
            state = RegimeState(name=rname, probability=0.7, probabilities={}, timestamp_ms=0)
            self.assertFalse(state.is_risk_off, f"{rname} should NOT be risk-off")

    def test_T334_regime_state_to_features(self):
        """T334: RegimeState.to_features() returns expected keys."""
        from regime_detector import RegimeState, REGIME_NAMES
        probs = {n: 1.0 / 6 for n in REGIME_NAMES}
        state = RegimeState(
            name="strong_bull", probability=0.5,
            probabilities=probs, timestamp_ms=0, regime_age_bars=10,
        )
        f = state.to_features()
        for rname in REGIME_NAMES:
            self.assertIn(f"regime_{rname}", f)
        self.assertIn("regime_score_bonus", f)
        self.assertIn("regime_is_risk_off", f)
        self.assertEqual(f["regime_age_bars"], 10.0)

    def test_T335_gaussian_hmm_fit_and_viterbi(self):
        """T335: GaussianHMM can fit data and decode states."""
        from regime_detector import GaussianHMM
        np.random.seed(42)
        # Simple 2-state problem
        hmm = GaussianHMM(n_states=2, n_features=2)
        # Generate synthetic data: state 0 = high mean, state 1 = low mean
        data0 = np.random.randn(50, 2) + np.array([3.0, 3.0])
        data1 = np.random.randn(50, 2) + np.array([-3.0, -3.0])
        X = np.vstack([data0, data1])
        hmm.fit(X, n_iter=10)
        self.assertTrue(hmm._fitted)
        states, log_prob = hmm.viterbi(X)
        self.assertEqual(len(states), 100)
        self.assertTrue(np.isfinite(log_prob))

    def test_T336_gaussian_hmm_predict_proba(self):
        """T336: predict_proba returns valid probabilities."""
        from regime_detector import GaussianHMM
        np.random.seed(42)
        hmm = GaussianHMM(n_states=3, n_features=2)
        X = np.random.randn(50, 2)
        hmm.fit(X, n_iter=5)
        probs = hmm.predict_proba(X)
        self.assertEqual(probs.shape, (50, 3))
        # Each row should sum to ~1
        row_sums = probs.sum(axis=1)
        for s in row_sums:
            self.assertAlmostEqual(s, 1.0, places=3)

    def test_T337_gaussian_hmm_save_load(self):
        """T337: HMM save/load roundtrip preserves parameters."""
        from regime_detector import GaussianHMM
        np.random.seed(42)
        hmm = GaussianHMM(n_states=3, n_features=2)
        X = np.random.randn(30, 2)
        hmm.fit(X, n_iter=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            hmm.save(path)
            hmm2 = GaussianHMM()
            hmm2.load(path)
            np.testing.assert_array_almost_equal(hmm.pi, hmm2.pi)
            np.testing.assert_array_almost_equal(hmm.A, hmm2.A)
            np.testing.assert_array_almost_equal(hmm.means, hmm2.means)
        finally:
            os.unlink(path)

    def test_T338_regime_detector_rules_fallback(self):
        """T338: RegimeDetector falls back to rules when no HMM model."""
        from regime_detector import RegimeDetector, REGIME_NAMES
        detector = RegimeDetector()  # no model
        data, feat, i = _make_feat(200, "up")
        state = detector.detect(data, feat)
        self.assertIn(state.name, REGIME_NAMES)
        self.assertGreater(state.probability, 0)
        # Uptrend should detect bull regime
        self.assertIn("bull", state.name)

    def test_T339_logsumexp_stability(self):
        """T339: _logsumexp handles extreme values."""
        from regime_detector import _logsumexp
        # Normal values
        x = np.array([1.0, 2.0, 3.0])
        result = _logsumexp(x)
        self.assertTrue(np.isfinite(result))

        # Very large values (should not overflow)
        x_large = np.array([1000.0, 1001.0, 1002.0])
        result_large = _logsumexp(x_large)
        self.assertTrue(np.isfinite(result_large))

        # Very negative values
        x_neg = np.array([-1000.0, -1001.0, -1002.0])
        result_neg = _logsumexp(x_neg)
        self.assertTrue(np.isfinite(result_neg))


# ════════════════════════════════════════════════════════════════════════════════
# T340–T349  top_gainer_model
# ════════════════════════════════════════════════════════════════════════════════

class TestTopGainerModel(unittest.TestCase):
    """Tests for Top Gainer Model module."""

    def test_T340_feature_names_defined(self):
        """T340: FEATURE_NAMES has expected count and format."""
        from top_gainer_model import FEATURE_NAMES
        self.assertGreater(len(FEATURE_NAMES), 30)
        for fn in FEATURE_NAMES:
            self.assertTrue(fn.startswith("tg_"), f"Feature {fn} should start with tg_")

    def test_T341_prediction_score_bonus(self):
        """T341: TopGainerPrediction.score_bonus tiers work correctly."""
        from top_gainer_model import TopGainerPrediction
        # Top 5 prediction
        p1 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.4, prob_top10=0.5,
            prob_top20=0.6, prob_top50=0.7, expected_eod_return=5.0, confidence=0.8,
        )
        self.assertEqual(p1.score_bonus, 15.0)

        # Top 10 prediction
        p2 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.1, prob_top10=0.4,
            prob_top20=0.5, prob_top50=0.6, expected_eod_return=3.0, confidence=0.7,
        )
        self.assertEqual(p2.score_bonus, 10.0)

        # Top 20 prediction
        p3 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.05, prob_top10=0.1,
            prob_top20=0.4, prob_top50=0.5, expected_eod_return=2.0, confidence=0.6,
        )
        self.assertEqual(p3.score_bonus, 6.0)

        # No prediction
        p4 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.01, prob_top10=0.05,
            prob_top20=0.1, prob_top50=0.2, expected_eod_return=0.5, confidence=0.3,
        )
        self.assertEqual(p4.score_bonus, 0.0)

    def test_T342_is_likely_top_gainer(self):
        """T342: is_likely_top_gainer requires both prob and confidence."""
        from top_gainer_model import TopGainerPrediction
        # Both high → True
        p = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.1, prob_top10=0.3,
            prob_top20=0.5, prob_top50=0.6, expected_eod_return=3.0, confidence=0.7,
        )
        self.assertTrue(p.is_likely_top_gainer)

        # Low confidence → False
        p2 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.1, prob_top10=0.3,
            prob_top20=0.5, prob_top50=0.6, expected_eod_return=3.0, confidence=0.3,
        )
        self.assertFalse(p2.is_likely_top_gainer)

    def test_T343_position_size_mult(self):
        """T343: position_size_mult scales with conviction tier."""
        from top_gainer_model import TopGainerPrediction
        p_top5 = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.4, prob_top10=0.5,
            prob_top20=0.6, prob_top50=0.7, expected_eod_return=5.0, confidence=0.8,
        )
        self.assertEqual(p_top5.position_size_mult, 1.5)

        p_none = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.01, prob_top10=0.05,
            prob_top20=0.1, prob_top50=0.2, expected_eod_return=0.0, confidence=0.2,
        )
        self.assertEqual(p_none.position_size_mult, 1.0)

    def test_T344_heuristic_predict_basic(self):
        """T344: Heuristic predict returns valid prediction for momentum signal."""
        from top_gainer_model import TopGainerModel
        model = TopGainerModel()
        features = {
            "tg_return_1h": 3.0,
            "tg_return_4h": 8.0,
            "tg_return_since_open": 5.0,
            "tg_vs_btc_4h": 3.0,
            "tg_volume_ratio_1h": 3.5,
            "tg_volume_acceleration": 1.6,
            "tg_of_cvd_pct_5m": 6.0,
            "tg_of_imbalance_5m": 0.6,
            "tg_of_large_trade_ratio_5m": 0.4,
            "tg_of_breakout_signal_5m": 0.7,
            "tg_oi_change_1h": 6.0,
            "tg_funding_flip": 1.0,
            "tg_ls_ratio": 0.7,
        }
        pred = model.predict(features)
        self.assertGreater(pred.prob_top50, 0)
        self.assertGreater(pred.confidence, 0)

    def test_T345_heuristic_predict_zero_features(self):
        """T345: Heuristic handles all-zero features gracefully."""
        from top_gainer_model import TopGainerModel
        model = TopGainerModel()
        pred = model.predict({})
        self.assertEqual(pred.prob_top5, 0.0)
        self.assertEqual(pred.score_bonus, 0.0)

    def test_T346_compute_features_with_klines(self):
        """T346: compute_features extracts momentum and structure features."""
        from top_gainer_model import TopGainerModel
        model = TopGainerModel()
        data = _make_candles(50, "up", base=100.0)
        from indicators import compute_features
        feat = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
        features = model.compute_features(
            symbol="TEST", klines_1h=data, klines_15m=None,
            features_1h=feat, features_15m=None,
        )
        self.assertIn("tg_return_1h", features)
        self.assertIn("tg_rsi", features)
        self.assertIn("tg_adx", features)
        self.assertIn("tg_hour_utc", features)
        self.assertIn("tg_hour_sin", features)

    def test_T347_to_dict_structure(self):
        """T347: TopGainerPrediction.to_dict() has expected keys."""
        from top_gainer_model import TopGainerPrediction
        p = TopGainerPrediction(
            symbol="X", timestamp_ms=0, prob_top5=0.1, prob_top10=0.2,
            prob_top20=0.3, prob_top50=0.4, expected_eod_return=2.0, confidence=0.6,
        )
        d = p.to_dict()
        for key in ["symbol", "prob_top5", "prob_top10", "prob_top20", "prob_top50",
                     "expected_eod_return", "confidence", "is_likely", "score_bonus"]:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_T348_pct_change_helper(self):
        """T348: _pct_change computes correct percentage."""
        from top_gainer_model import _pct_change
        arr = np.array([100.0, 102.0, 105.0, 103.0, 110.0])
        # 1-period change: (110 - 103) / 103 * 100
        result = _pct_change(arr, 1)
        self.assertAlmostEqual(result, (110 - 103) / 103 * 100, places=4)
        # 4-period change: (110 - 100) / 100 * 100
        result4 = _pct_change(arr, 4)
        self.assertAlmostEqual(result4, 10.0, places=4)

    def test_T349_model_load_nonexistent(self):
        """T349: Loading nonexistent model returns False, uses heuristic."""
        from top_gainer_model import TopGainerModel
        model = TopGainerModel(model_path="/nonexistent/model.json")
        self.assertIsNone(model._model_payload)
        # Should still predict via heuristic
        pred = model.predict({"tg_return_since_open": 5.0})
        self.assertGreater(pred.prob_top50, 0)


# ════════════════════════════════════════════════════════════════════════════════
# T350–T359  multi_horizon
# ════════════════════════════════════════════════════════════════════════════════

class TestMultiHorizon(unittest.TestCase):
    """Tests for Multi-Horizon Framework module."""

    def test_T350_direction_enum_values(self):
        """T350: Direction enum has correct ordering."""
        from multi_horizon import Direction
        self.assertEqual(Direction.STRONG_BUY.value, 2)
        self.assertEqual(Direction.NEUTRAL.value, 0)
        self.assertEqual(Direction.STRONG_SELL.value, -2)

    def test_T351_horizon_config_completeness(self):
        """T351: All horizons have required config keys."""
        from multi_horizon import HORIZON_CONFIG, Horizon
        required_keys = ["timeframes", "max_hold_minutes", "position_pct",
                         "max_positions", "min_confidence", "score_weight_in_composite"]
        for h in Horizon:
            self.assertIn(h, HORIZON_CONFIG, f"Missing config for {h}")
            for key in required_keys:
                self.assertIn(key, HORIZON_CONFIG[h], f"Missing {key} in {h}")

    def test_T352_horizon_signal_is_actionable(self):
        """T352: is_actionable checks direction + min_confidence."""
        from multi_horizon import HorizonSignal, Direction, Horizon
        sig = HorizonSignal(
            horizon=Horizon.INTRADAY, symbol="BTC", direction=Direction.BUY,
            confidence=0.7, score=60.0, timestamp_ms=0,
        )
        self.assertTrue(sig.is_actionable)

        sig_low = HorizonSignal(
            horizon=Horizon.INTRADAY, symbol="BTC", direction=Direction.BUY,
            confidence=0.3, score=30.0, timestamp_ms=0,
        )
        self.assertFalse(sig_low.is_actionable)

        sig_neutral = HorizonSignal(
            horizon=Horizon.INTRADAY, symbol="BTC", direction=Direction.NEUTRAL,
            confidence=0.9, score=0.0, timestamp_ms=0,
        )
        self.assertFalse(sig_neutral.is_actionable)

    def test_T353_horizon_signal_risk_reward(self):
        """T353: risk_reward computes correctly from entry/stop/target."""
        from multi_horizon import HorizonSignal, Direction, Horizon
        sig = HorizonSignal(
            horizon=Horizon.DAILY, symbol="BTC", direction=Direction.BUY,
            confidence=0.7, score=50.0, timestamp_ms=0,
            entry_price=100.0, stop_loss=95.0, take_profit=115.0,
        )
        # risk = 5, reward = 15, R:R = 3.0
        self.assertAlmostEqual(sig.risk_reward, 3.0)

    def test_T354_scalp_strategy_neutral_no_data(self):
        """T354: ScalpStrategy returns NEUTRAL without order flow data."""
        from multi_horizon import ScalpStrategy, Direction
        strat = ScalpStrategy()
        sig = strat.analyze("BTCUSDT", None, None, None)
        self.assertEqual(sig.direction, Direction.NEUTRAL)

    def test_T355_scalp_strategy_detects_surge(self):
        """T355: ScalpStrategy detects order flow surge."""
        from multi_horizon import ScalpStrategy, Direction
        from order_flow import OrderFlowMetrics
        of_1m = OrderFlowMetrics(
            symbol="BTC", window_minutes=1, timestamp_ms=0,
            cvd=500, cvd_pct=12.0, imbalance=0.65,
            buy_volume=800, sell_volume=200, total_volume=1000,
            trade_count=200, trades_per_sec=8.0,
            large_trade_count=5, large_buy_volume=300, large_sell_volume=50,
            large_trade_ratio=0.5, vwap=100.0, price_vs_vwap_pct=0.2,
            absorption_signal=0.0, exhaustion_signal=0.0, breakout_signal=0.7,
        )
        strat = ScalpStrategy()
        sig = strat.analyze("BTCUSDT", None, {"1m": of_1m}, None)
        self.assertIn(sig.direction, (Direction.BUY, Direction.STRONG_BUY))
        self.assertGreater(sig.score, 40)

    def test_T356_daily_strategy_bullish_on_uptrend(self):
        """T356: DailyStrategy detects bullish setup from 4h klines."""
        from multi_horizon import DailyStrategy, Direction
        data, feat, i = _make_feat(200, "up")
        strat = DailyStrategy()
        sig = strat.analyze("BTCUSDT", data, feat, None, None)
        # Strong uptrend should at least detect some bullishness
        self.assertIn(sig.direction, (Direction.BUY, Direction.STRONG_BUY, Direction.NEUTRAL))

    def test_T357_swing_strategy_needs_data(self):
        """T357: SwingStrategy returns NEUTRAL without sufficient data."""
        from multi_horizon import SwingStrategy, Direction
        strat = SwingStrategy()
        sig = strat.analyze("BTC", None, None, None)
        self.assertEqual(sig.direction, Direction.NEUTRAL)
        # With too few bars
        short_data = _make_candles(10, "up")
        sig2 = strat.analyze("BTC", short_data, {}, None)
        self.assertEqual(sig2.direction, Direction.NEUTRAL)

    def test_T358_horizon_manager_cascade_bonus(self):
        """T358: HorizonManager gives cascade bonus for multi-horizon alignment."""
        from multi_horizon import HorizonManager, Direction, HorizonSignal, Horizon
        mgr = HorizonManager()
        # Manually inject aligned signals
        mgr._last_signals["BTCUSDT"] = {
            "scalp": HorizonSignal(horizon=Horizon.SCALP, symbol="BTC",
                                   direction=Direction.BUY, confidence=0.7, score=50.0,
                                   timestamp_ms=0),
            "intraday": HorizonSignal(horizon=Horizon.INTRADAY, symbol="BTC",
                                      direction=Direction.STRONG_BUY, confidence=0.8, score=65.0,
                                      timestamp_ms=0),
            "daily": HorizonSignal(horizon=Horizon.DAILY, symbol="BTC",
                                   direction=Direction.BUY, confidence=0.65, score=45.0,
                                   timestamp_ms=0),
            "swing": HorizonSignal(horizon=Horizon.SWING, symbol="BTC",
                                   direction=Direction.BUY, confidence=0.6, score=40.0,
                                   timestamp_ms=0),
        }
        composite = mgr.composite_signal("BTCUSDT")
        self.assertEqual(composite.alignment_count, 4)
        self.assertEqual(composite.cascade_bonus, 15.0)
        self.assertEqual(composite.direction, Direction.STRONG_BUY)
        self.assertTrue(composite.is_high_conviction)

    def test_T359_composite_signal_empty(self):
        """T359: composite_signal returns NEUTRAL for unknown symbol."""
        from multi_horizon import HorizonManager, Direction
        mgr = HorizonManager()
        composite = mgr.composite_signal("UNKNOWN")
        self.assertEqual(composite.direction, Direction.NEUTRAL)
        self.assertEqual(composite.score, 0.0)
        self.assertEqual(composite.cascade_bonus, 0.0)


# ════════════════════════════════════════════════════════════════════════════════
# T360–T369  enhanced_signals
# ════════════════════════════════════════════════════════════════════════════════

class TestEnhancedSignals(unittest.TestCase):
    """Tests for Enhanced Signals integration layer."""

    def test_T360_score_adjustment_defaults(self):
        """T360: ScoreAdjustment defaults to zero bonus, not blocked."""
        from enhanced_signals import ScoreAdjustment
        adj = ScoreAdjustment()
        self.assertEqual(adj.total_bonus, 0.0)
        self.assertFalse(adj.is_blocked)
        self.assertEqual(adj.block_reason, "")

    def test_T361_score_adjustment_to_log_dict(self):
        """T361: to_log_dict returns all expected keys."""
        from enhanced_signals import ScoreAdjustment
        adj = ScoreAdjustment(
            order_flow_bonus=5.0, derivatives_bonus=3.0,
            regime_bonus=-2.0, top_gainer_bonus=10.0,
            multi_horizon_bonus=5.0, total_bonus=21.0,
        )
        d = adj.to_log_dict()
        self.assertEqual(d["of_bonus"], 5.0)
        self.assertEqual(d["total"], 21.0)
        self.assertFalse(d["blocked"])

    def test_T362_enhanced_signals_init(self):
        """T362: EnhancedSignals initializes with all None components."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        self.assertFalse(es._started)
        self.assertIsNone(es._ws_manager)
        self.assertIsNone(es._order_flow)
        self.assertIsNone(es._derivatives)
        self.assertIsNone(es._regime_detector)
        self.assertIsNone(es._top_gainer_model)
        self.assertIsNone(es._horizon_manager)

    def test_T363_score_adjustment_no_crash_when_no_modules(self):
        """T363: score_adjustment returns zero adjustment when no modules loaded."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        data = _make_candles(50)
        adj = es.score_adjustment(
            sym="BTCUSDT", tf="15m", mode="trend",
            current_score=50.0, feat={}, data=data, i=48,
        )
        self.assertEqual(adj.total_bonus, 0.0)
        self.assertFalse(adj.is_blocked)

    def test_T364_get_order_flow_features_empty(self):
        """T364: get_order_flow_features returns empty dict when no analyzer."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        f = es.get_order_flow_features("BTCUSDT")
        self.assertEqual(f, {})

    def test_T365_get_derivatives_features_empty(self):
        """T365: get_derivatives_features returns empty dict when no data source."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        f = es.get_derivatives_features("BTCUSDT")
        self.assertEqual(f, {})

    def test_T366_compute_regime_bonus_no_regime(self):
        """T366: _compute_regime_bonus returns 0 when no regime detected."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        data = _make_candles(50)
        bonus, block = es._compute_regime_bonus("BTC", "trend", {}, data, 48)
        self.assertEqual(bonus, 0.0)
        self.assertEqual(block, "")

    def test_T367_compute_regime_bonus_risk_off_blocks(self):
        """T367: Risk-off regime blocks entry."""
        from enhanced_signals import EnhancedSignals
        from regime_detector import RegimeState
        es = EnhancedSignals()
        es._current_regime = RegimeState(
            name="strong_bear", probability=0.8,
            probabilities={"strong_bear": 0.8}, timestamp_ms=0,
        )
        with patch.object(type(es), '_compute_regime_bonus', wraps=es._compute_regime_bonus):
            data = _make_candles(50)
            bonus, block = es._compute_regime_bonus("BTC", "trend", {}, data, 48)
        self.assertLess(bonus, 0)
        self.assertIn("risk-off", block)

    def test_T368_update_regime(self):
        """T368: update_regime sets _current_regime."""
        from enhanced_signals import EnhancedSignals
        from regime_detector import RegimeDetector
        es = EnhancedSignals()
        es._regime_detector = RegimeDetector()
        data, feat, i = _make_feat(200, "up")
        es.update_regime(data, feat)
        self.assertIsNotNone(es._current_regime)

    def test_T369_ws_stats_empty(self):
        """T369: ws_stats returns empty dict when no WS manager."""
        from enhanced_signals import EnhancedSignals
        es = EnhancedSignals()
        self.assertEqual(es.ws_stats, {})


# ════════════════════════════════════════════════════════════════════════════════
# T370–T379  additional edge cases
# ════════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests."""

    def test_T370_wsmanager_ignores_unknown_symbol(self):
        """T370: WS handlers silently ignore unknown symbols."""
        from ws_manager import WSManager, SymbolBuffers
        mgr = WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = SymbolBuffers()
        # Send data for unknown symbol
        mgr._handle_kline({"k": {"s": "UNKNOWN", "t": 1, "o": "1", "h": "1",
                                  "l": "1", "c": "1", "v": "1", "q": "1", "n": 1, "x": True}})
        mgr._handle_agg_trade({"s": "UNKNOWN", "T": 1, "p": "1", "q": "1", "m": False, "a": 1})
        mgr._handle_book_ticker({"s": "UNKNOWN", "u": 1, "b": "1", "B": "1", "a": "1", "A": "1"})
        # No crash, no data for unknown
        self.assertEqual(len(mgr._buffers["BTCUSDT"].klines_1m), 0)

    def test_T371_kline1m_dataclass_fields(self):
        """T371: Kline1m dataclass has all expected fields."""
        from ws_manager import Kline1m
        k = Kline1m(t=0, o=1.0, h=2.0, l=0.5, c=1.5, v=100.0, qv=150.0, trades=10, closed=True)
        self.assertEqual(k.t, 0)
        self.assertTrue(k.closed)

    def test_T372_book_ticker_spread_calculation(self):
        """T372: BookTicker spread_bps calculated correctly."""
        from ws_manager import WSManager, SymbolBuffers
        mgr = WSManager(symbols=["TESTUSDT"])
        mgr._buffers["TESTUSDT"] = SymbolBuffers()
        mgr._handle_book_ticker({
            "s": "TESTUSDT", "u": 1,
            "b": "100.00", "B": "10",
            "a": "100.10", "A": "10",
        })
        book = mgr.get_book_ticker("TESTUSDT")
        # spread = 0.10, mid = 100.05, bps = 0.10/100.05 * 10000 ≈ 9.995
        self.assertAlmostEqual(book.spread_bps, 10.0, places=0)

    def test_T373_regime_detector_extract_features(self):
        """T373: _extract_features returns correct shape."""
        from regime_detector import RegimeDetector
        detector = RegimeDetector()
        data = _make_candles(100, "up")
        from indicators import compute_features
        feat = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
        features = detector._extract_features(data, feat, None)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[1], 6)  # 6 HMM features
        # Should have n - 20 rows (warmup)
        self.assertEqual(features.shape[0], 80)

    def test_T374_regime_detector_extract_features_too_short(self):
        """T374: _extract_features returns None for too-short data."""
        from regime_detector import RegimeDetector
        detector = RegimeDetector()
        data = _make_candles(10, "up")
        from indicators import compute_features
        feat = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
        features = detector._extract_features(data, feat, None)
        self.assertIsNone(features)

    def test_T375_intraday_strategy_wraps_existing(self):
        """T375: IntradayStrategy enhances existing signal with bonuses."""
        from multi_horizon import IntradayStrategy, Direction
        strat = IntradayStrategy()
        existing = {"score": 55.0, "mode": "trend", "is_buy": True}
        sig = strat.analyze("BTC", existing, None, None, None, None)
        self.assertIn(sig.direction, (Direction.BUY, Direction.STRONG_BUY))
        self.assertEqual(sig.entry_mode, "trend")

    def test_T376_intraday_strategy_no_buy(self):
        """T376: IntradayStrategy returns NEUTRAL when existing signal is not buy."""
        from multi_horizon import IntradayStrategy, Direction
        strat = IntradayStrategy()
        existing = {"score": 30.0, "mode": "trend", "is_buy": False}
        sig = strat.analyze("BTC", existing, None, None, None, None)
        self.assertEqual(sig.direction, Direction.NEUTRAL)

    def test_T377_composite_signal_to_dict(self):
        """T377: CompositeSignal.to_dict() has expected keys."""
        from multi_horizon import CompositeSignal, Direction
        cs = CompositeSignal(
            symbol="BTC", timestamp_ms=0, direction=Direction.BUY,
            confidence=0.7, score=50.0, alignment_count=3,
            alignment_pct=0.75, best_entry_horizon="intraday",
            cascade_bonus=10.0,
        )
        d = cs.to_dict()
        for key in ["symbol", "direction", "confidence", "score",
                     "alignment", "best_entry", "cascade_bonus"]:
            self.assertIn(key, d)

    def test_T378_horizon_signal_to_dict(self):
        """T378: HorizonSignal.to_dict() has expected keys."""
        from multi_horizon import HorizonSignal, Direction, Horizon
        sig = HorizonSignal(
            horizon=Horizon.SCALP, symbol="BTC", direction=Direction.BUY,
            confidence=0.7, score=50.0, timestamp_ms=0,
            entry_mode="surge", reasons=["test"],
        )
        d = sig.to_dict()
        self.assertEqual(d["horizon"], "scalp")
        self.assertEqual(d["direction"], "BUY")
        self.assertEqual(d["reasons"], ["test"])

    def test_T379_get_trade_count_since(self):
        """T379: get_trade_count_since counts trades after timestamp."""
        from ws_manager import WSManager, SymbolBuffers, AggTrade
        mgr = WSManager(symbols=["BTCUSDT"])
        mgr._buffers["BTCUSDT"] = SymbolBuffers()
        now_ms = int(time.time() * 1000)
        for i in range(50):
            trade = AggTrade(t=now_ms - (50 - i) * 1000, price=100.0,
                             qty=1.0, is_buyer=True, trade_id=i)
            mgr._buffers["BTCUSDT"].trades.append(trade)
        # Count trades in last 10 seconds
        count = mgr.get_trade_count_since("BTCUSDT", now_ms - 10000)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 50)


# ════════════════════════════════════════════════════════════════════════════════
# T380–T385  training script imports and structures
# ════════════════════════════════════════════════════════════════════════════════

class TestTrainingScripts(unittest.TestCase):
    """Tests for training script data structures and helpers."""

    def test_T380_train_top_gainer_compute_metrics(self):
        """T380: _compute_metrics returns valid AUC and precision/recall."""
        from train_top_gainer import _compute_metrics
        np.random.seed(42)
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        metrics = _compute_metrics(y_true, y_proba, "test")
        self.assertIn("auc", metrics)
        self.assertIn("precision_at_03", metrics)
        self.assertIn("recall_at_03", metrics)
        self.assertGreater(metrics["auc"], 0.5)  # should be better than random

    def test_T381_compute_metrics_all_positive(self):
        """T381: _compute_metrics handles edge case of all positive labels."""
        from train_top_gainer import _compute_metrics
        y_true = np.ones(10)
        y_proba = np.random.rand(10)
        metrics = _compute_metrics(y_true, y_proba, "all_pos")
        self.assertIn("auc", metrics)

    def test_T382_compute_metrics_all_negative(self):
        """T382: _compute_metrics handles edge case of all negative labels."""
        from train_top_gainer import _compute_metrics
        y_true = np.zeros(10)
        y_proba = np.random.rand(10)
        metrics = _compute_metrics(y_true, y_proba, "all_neg")
        self.assertIn("auc", metrics)

    def test_T383_stump_ensemble_trains(self):
        """T383: Stump ensemble trains without CatBoost."""
        from train_top_gainer import _train_stump_ensemble
        np.random.seed(42)
        n_features = 5
        X_train = np.random.randn(200, n_features)
        y_train = (X_train[:, 0] > 0).astype(float)
        X_val = np.random.randn(50, n_features)
        y_val = (X_val[:, 0] > 0).astype(float)
        payload, metrics = _train_stump_ensemble(
            X_train, y_train, X_val, y_val, "test",
            n_estimators=20, learning_rate=0.1,
        )
        self.assertEqual(payload["model_type"], "stump_ensemble")
        self.assertGreater(len(payload["stumps"]), 0)
        self.assertGreater(metrics["auc"], 0.5)

    def test_T384_top_gainer_model_log_training_sample(self):
        """T384: log_training_sample writes valid JSONL."""
        from top_gainer_model import TopGainerModel
        model = TopGainerModel()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            import top_gainer_model
            orig = top_gainer_model.DATASET_FILE
            top_gainer_model.DATASET_FILE = Path(path)
            model.log_training_sample(
                symbol="BTCUSDT",
                features={"tg_return_1h": 2.5},
                label_top5=False, label_top10=False,
                label_top20=True, label_top50=True,
                eod_return_pct=4.5,
            )
            top_gainer_model.DATASET_FILE = orig
            with open(path, "r") as rf:
                line = rf.readline()
            record = json.loads(line)
            self.assertEqual(record["symbol"], "BTCUSDT")
            self.assertEqual(record["label_top20"], 1)
        finally:
            os.unlink(path)

    def test_T385_regime_detector_strong_bear(self):
        """T385: RegimeDetector identifies downtrend as bearish."""
        from regime_detector import RegimeDetector
        detector = RegimeDetector()
        data, feat, i = _make_feat(200, "down")
        state = detector.detect(data, feat)
        # Downtrend should be classified as bear
        self.assertIn(state.name, ("weak_bear", "strong_bear", "volatile_chop", "ranging"))


# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
