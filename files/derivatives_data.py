from __future__ import annotations

"""
Derivatives Data Pipeline — funding rate, open interest, liquidations.

Fetches from Binance Futures API:
  - Funding rate history    → sentiment indicator (positive = longs pay shorts)
  - Open interest           → total outstanding contracts (OI spike = conviction)
  - Liquidation stream      → forced closures (cascade potential)
  - Long/Short ratio        → crowd positioning

These are LEADING indicators for spot price:
  - Funding flip (neg → pos) often precedes spot rally
  - OI spike + price up     = genuine trend (new money entering)
  - OI spike + price flat   = squeeze building
  - Liquidation cascade     = volatility explosion imminent

Usage:
    dd = DerivativesData()
    await dd.start_background_fetch(symbols=["BTCUSDT", "ETHUSDT"])
    snapshot = dd.get_snapshot("BTCUSDT")
    features = dd.get_features("BTCUSDT")
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

import aiohttp
import numpy as np

log = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.binance.com"
FUTURES_WS = "wss://fstream.binance.com"


@dataclass
class FundingSnapshot:
    """Funding rate data point."""
    symbol: str
    rate: float               # Current funding rate (e.g., 0.0001 = 0.01%)
    time_ms: int              # Funding timestamp
    next_time_ms: int         # Next funding time
    mark_price: float
    index_price: float
    premium_pct: float        # (mark - index) / index × 100


@dataclass
class OISnapshot:
    """Open interest data point."""
    symbol: str
    oi_contracts: float       # OI in contracts
    oi_value_usd: float       # OI in USD
    timestamp_ms: int

    # Derived (needs history)
    oi_change_1h_pct: float = 0.0
    oi_change_4h_pct: float = 0.0
    oi_change_24h_pct: float = 0.0


@dataclass
class LongShortRatio:
    """Top trader long/short ratio."""
    symbol: str
    long_ratio: float         # Long account ratio
    short_ratio: float        # Short account ratio
    ls_ratio: float           # Long/Short ratio (>1 = more longs)
    timestamp_ms: int


@dataclass
class Liquidation:
    """Liquidation event."""
    symbol: str
    side: str                 # "BUY" (short liq) or "SELL" (long liq)
    price: float
    qty: float
    notional_usd: float
    timestamp_ms: int


@dataclass
class DerivativesSnapshot:
    """Complete derivatives data for a symbol."""
    symbol: str
    timestamp_ms: int

    # Latest funding
    funding: Optional[FundingSnapshot] = None
    funding_history: list = field(default_factory=list)  # Last 8 periods (24h)

    # Open interest
    oi: Optional[OISnapshot] = None
    oi_history: list = field(default_factory=list)  # Last 24 data points (5m intervals)

    # Long/Short
    ls_ratio: Optional[LongShortRatio] = None

    # Liquidations (last 1h)
    liquidations: list = field(default_factory=list)
    liq_buy_volume_1h: float = 0.0   # Short liquidation volume
    liq_sell_volume_1h: float = 0.0  # Long liquidation volume

    def to_features(self) -> Dict[str, float]:
        """Convert to flat feature dict for ML models."""
        f: Dict[str, float] = {}

        # Funding
        if self.funding:
            f["deriv_funding_rate"] = self.funding.rate
            f["deriv_premium_pct"] = self.funding.premium_pct
            # Funding trend (avg of last 3 vs last 8)
            if len(self.funding_history) >= 3:
                recent_3 = np.mean([x.rate for x in self.funding_history[-3:]])
                f["deriv_funding_trend_3"] = float(recent_3)
            if len(self.funding_history) >= 8:
                all_8 = np.mean([x.rate for x in self.funding_history])
                f["deriv_funding_trend_8"] = float(all_8)
                f["deriv_funding_flip"] = 1.0 if (
                    self.funding_history[-1].rate > 0 and self.funding_history[-4].rate < 0
                ) else 0.0
        else:
            f["deriv_funding_rate"] = 0.0
            f["deriv_premium_pct"] = 0.0

        # OI
        if self.oi:
            f["deriv_oi_change_1h"] = self.oi.oi_change_1h_pct
            f["deriv_oi_change_4h"] = self.oi.oi_change_4h_pct
            f["deriv_oi_change_24h"] = self.oi.oi_change_24h_pct
            # OI spike detection
            f["deriv_oi_spike_1h"] = 1.0 if abs(self.oi.oi_change_1h_pct) > 5.0 else 0.0
        else:
            f["deriv_oi_change_1h"] = 0.0
            f["deriv_oi_change_4h"] = 0.0
            f["deriv_oi_change_24h"] = 0.0
            f["deriv_oi_spike_1h"] = 0.0

        # Long/Short ratio
        if self.ls_ratio:
            f["deriv_ls_ratio"] = self.ls_ratio.ls_ratio
            f["deriv_crowd_long"] = 1.0 if self.ls_ratio.ls_ratio > 1.5 else 0.0
            f["deriv_crowd_short"] = 1.0 if self.ls_ratio.ls_ratio < 0.67 else 0.0
        else:
            f["deriv_ls_ratio"] = 1.0
            f["deriv_crowd_long"] = 0.0
            f["deriv_crowd_short"] = 0.0

        # Liquidations
        total_liq = self.liq_buy_volume_1h + self.liq_sell_volume_1h
        f["deriv_liq_total_1h"] = total_liq
        f["deriv_liq_buy_ratio"] = (
            self.liq_buy_volume_1h / total_liq if total_liq > 0 else 0.5
        )
        f["deriv_liq_cascade"] = 1.0 if total_liq > 1_000_000 else 0.0

        return f


class DerivativesData:
    """
    Fetches and caches derivatives data from Binance Futures.

    Runs a background task that polls every N seconds:
      - Funding rate:   every 60s (changes every 8h)
      - Open interest:  every 60s
      - L/S ratio:      every 300s
      - Liquidations:   via WebSocket (real-time)
    """

    def __init__(self):
        self._symbols: Set[str] = set()
        self._snapshots: Dict[str, DerivativesSnapshot] = {}
        self._oi_history: Dict[str, Deque[OISnapshot]] = {}  # 24h ring buffer
        self._funding_history: Dict[str, Deque[FundingSnapshot]] = {}
        self._liquidations: Dict[str, Deque[Liquidation]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._liq_ws_task: Optional[asyncio.Task] = None

    async def start_background_fetch(
        self,
        symbols: List[str],
        funding_interval: int = 60,
        oi_interval: int = 60,
        ls_interval: int = 300,
    ) -> None:
        """Start background data fetching for given symbols."""
        if self._running:
            return
        self._running = True
        self._symbols = {s.upper() for s in symbols}
        self._session = aiohttp.ClientSession()

        for sym in self._symbols:
            self._snapshots[sym] = DerivativesSnapshot(
                symbol=sym, timestamp_ms=int(time.time() * 1000)
            )
            self._oi_history[sym] = deque(maxlen=288)  # 24h @ 5min intervals
            self._funding_history[sym] = deque(maxlen=24)  # 3 days of 8h funding
            self._liquidations[sym] = deque(maxlen=500)

        # Start polling tasks
        self._tasks.append(asyncio.create_task(
            self._poll_loop(self._fetch_all_funding, funding_interval, "funding")
        ))
        self._tasks.append(asyncio.create_task(
            self._poll_loop(self._fetch_all_oi, oi_interval, "oi")
        ))
        self._tasks.append(asyncio.create_task(
            self._poll_loop(self._fetch_all_ls_ratio, ls_interval, "ls_ratio")
        ))
        # Liquidation WS
        self._liq_ws_task = asyncio.create_task(self._liq_ws_loop())
        self._tasks.append(self._liq_ws_task)

        # Initial fetch
        await self._fetch_all_funding()
        await self._fetch_all_oi()
        log.info("DerivativesData started for %d symbols", len(self._symbols))

    async def stop(self) -> None:
        """Stop all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._session:
            await self._session.close()
            self._session = None
        log.info("DerivativesData stopped")

    def get_snapshot(self, symbol: str) -> Optional[DerivativesSnapshot]:
        """Get current derivatives snapshot for symbol."""
        return self._snapshots.get(symbol.upper())

    def get_features(self, symbol: str) -> Dict[str, float]:
        """Get flat feature dict for ML models."""
        snap = self.get_snapshot(symbol)
        return snap.to_features() if snap else {}

    def get_all_features(self) -> Dict[str, Dict[str, float]]:
        """Get features for all symbols."""
        return {sym: snap.to_features() for sym, snap in self._snapshots.items()}

    # ── Polling infrastructure ───────────────────────────────────────────────

    async def _poll_loop(self, fn, interval: int, name: str) -> None:
        """Generic polling loop with error handling."""
        while self._running:
            try:
                await fn()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("DerivativesData %s fetch error", name)
            await asyncio.sleep(interval)

    # ── Funding rate ─────────────────────────────────────────────────────────

    async def _fetch_all_funding(self) -> None:
        """Fetch funding rate for all symbols."""
        if not self._session:
            return
        tasks = [self._fetch_funding(sym) for sym in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_funding(self, symbol: str) -> None:
        """Fetch funding rate and mark price for one symbol."""
        try:
            url = f"{FUTURES_BASE}/fapi/v1/premiumIndex"
            async with self._session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            mark = float(data.get("markPrice", 0))
            index_p = float(data.get("indexPrice", 0))
            premium = (mark - index_p) / index_p * 100.0 if index_p > 0 else 0.0

            snap = FundingSnapshot(
                symbol=symbol,
                rate=float(data.get("lastFundingRate", 0)),
                time_ms=int(data.get("time", 0)),
                next_time_ms=int(data.get("nextFundingTime", 0)),
                mark_price=mark,
                index_price=index_p,
                premium_pct=premium,
            )

            self._funding_history[symbol].append(snap)
            ds = self._snapshots.get(symbol)
            if ds:
                ds.funding = snap
                ds.funding_history = list(self._funding_history[symbol])
                ds.timestamp_ms = int(time.time() * 1000)

        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

    # ── Open interest ────────────────────────────────────────────────────────

    async def _fetch_all_oi(self) -> None:
        """Fetch OI for all symbols."""
        if not self._session:
            return
        tasks = [self._fetch_oi(sym) for sym in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_oi(self, symbol: str) -> None:
        """Fetch open interest for one symbol."""
        try:
            url = f"{FUTURES_BASE}/fapi/v1/openInterest"
            async with self._session.get(url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            oi_val = float(data.get("openInterest", 0))

            # Get notional value from ticker
            mark = 0.0
            ds = self._snapshots.get(symbol)
            if ds and ds.funding:
                mark = ds.funding.mark_price
            oi_usd = oi_val * mark if mark > 0 else 0.0

            snap = OISnapshot(
                symbol=symbol,
                oi_contracts=oi_val,
                oi_value_usd=oi_usd,
                timestamp_ms=int(data.get("time", int(time.time() * 1000))),
            )

            # Compute changes from history
            history = self._oi_history.get(symbol)
            if history:
                self._compute_oi_changes(snap, list(history))

            self._oi_history[symbol].append(snap)
            if ds:
                ds.oi = snap
                ds.oi_history = list(self._oi_history[symbol])[-24:]
                ds.timestamp_ms = int(time.time() * 1000)

        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

    def _compute_oi_changes(self, current: OISnapshot, history: List[OISnapshot]) -> None:
        """Compute OI % changes from historical data."""
        now = current.timestamp_ms
        for h in reversed(history):
            age_h = (now - h.timestamp_ms) / 3_600_000.0
            if h.oi_contracts > 0:
                change = (current.oi_contracts - h.oi_contracts) / h.oi_contracts * 100.0
                if age_h >= 0.9 and current.oi_change_1h_pct == 0.0:
                    current.oi_change_1h_pct = change
                if age_h >= 3.5 and current.oi_change_4h_pct == 0.0:
                    current.oi_change_4h_pct = change
                if age_h >= 23.0 and current.oi_change_24h_pct == 0.0:
                    current.oi_change_24h_pct = change
                    break

    # ── Long/Short ratio ─────────────────────────────────────────────────────

    async def _fetch_all_ls_ratio(self) -> None:
        """Fetch L/S ratio for all symbols."""
        if not self._session:
            return
        tasks = [self._fetch_ls_ratio(sym) for sym in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_ls_ratio(self, symbol: str) -> None:
        """Fetch top trader long/short ratio."""
        try:
            url = f"{FUTURES_BASE}/futures/data/topLongShortAccountRatio"
            params = {"symbol": symbol, "period": "5m", "limit": 1}
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            if not data:
                return
            entry = data[0]
            snap = LongShortRatio(
                symbol=symbol,
                long_ratio=float(entry.get("longAccount", 0.5)),
                short_ratio=float(entry.get("shortAccount", 0.5)),
                ls_ratio=float(entry.get("longShortRatio", 1.0)),
                timestamp_ms=int(entry.get("timestamp", 0)),
            )

            ds = self._snapshots.get(symbol)
            if ds:
                ds.ls_ratio = snap
                ds.timestamp_ms = int(time.time() * 1000)

        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

    # ── Liquidation WebSocket ────────────────────────────────────────────────

    async def _liq_ws_loop(self) -> None:
        """WebSocket for all-market liquidation stream."""
        url = f"{FUTURES_WS}/ws/!forceOrder@arr"
        backoff = 1.0

        while self._running:
            try:
                session = aiohttp.ClientSession()
                async with session.ws_connect(url, heartbeat=20) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self._handle_liquidation(msg.data)
                            except Exception:
                                pass
                        elif msg.type in (aiohttp.WSMsgType.ERROR,
                                          aiohttp.WSMsgType.CLOSED):
                            break
                await session.close()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                pass
            except asyncio.CancelledError:
                break

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_liquidation(self, raw: str) -> None:
        """Process liquidation event."""
        data = json.loads(raw)
        o = data.get("o", {})
        sym = str(o.get("s", "")).upper()
        if sym not in self._symbols:
            return

        price = float(o.get("p", 0))
        qty = float(o.get("q", 0))
        liq = Liquidation(
            symbol=sym,
            side=str(o.get("S", "")),
            price=price,
            qty=qty,
            notional_usd=price * qty,
            timestamp_ms=int(o.get("T", int(time.time() * 1000))),
        )

        self._liquidations[sym].append(liq)

        # Update snapshot
        ds = self._snapshots.get(sym)
        if ds:
            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - 3_600_000  # 1h
            recent = [l for l in self._liquidations[sym] if l.timestamp_ms >= cutoff_ms]
            ds.liquidations = recent
            ds.liq_buy_volume_1h = sum(l.notional_usd for l in recent if l.side == "BUY")
            ds.liq_sell_volume_1h = sum(l.notional_usd for l in recent if l.side == "SELL")


# ── Singleton ────────────────────────────────────────────────────────────────
_global_dd: Optional[DerivativesData] = None


async def get_derivatives_data(symbols: Optional[List[str]] = None) -> DerivativesData:
    """Get or create global DerivativesData instance."""
    global _global_dd
    if _global_dd is None:
        _global_dd = DerivativesData()
        if symbols:
            await _global_dd.start_background_fetch(symbols)
    return _global_dd


async def stop_derivatives_data() -> None:
    """Stop global DerivativesData."""
    global _global_dd
    if _global_dd:
        await _global_dd.stop()
        _global_dd = None
