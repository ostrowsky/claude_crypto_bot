from __future__ import annotations

"""
WebSocket Data Manager — real-time Binance streams.

Replaces REST polling with sub-second data delivery:
  - 1m klines (mini-ticker)   → 1-minute candles for fast impulse detection
  - aggTrade stream           → individual trades for order-flow analysis
  - bookTicker                → best bid/ask for spread monitoring

Architecture:
  - Single shared WS connection per stream type (combined streams)
  - Ring buffers per symbol (configurable depth)
  - Async generator interface for consumers
  - Auto-reconnect with exponential backoff

Usage:
    mgr = WSManager(symbols=["BTCUSDT", "ETHUSDT"])
    await mgr.start()
    # Access latest data:
    bars_1m = mgr.get_klines("BTCUSDT", limit=60)
    trades  = mgr.get_trades("BTCUSDT", limit=500)
    book    = mgr.get_book_ticker("BTCUSDT")
    ...
    await mgr.stop()
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

# ── Binance WS endpoints ─────────────────────────────────────────────────────
WS_BASE = "wss://stream.binance.com:9443"
WS_COMBINED = f"{WS_BASE}/stream"
# Max streams per connection (Binance limit = 1024, we stay conservative)
MAX_STREAMS_PER_CONN = 200

# ── Ring buffer sizes ────────────────────────────────────────────────────────
KLINE_1M_BUFFER = 1440       # 24 hours of 1m bars
TRADE_BUFFER = 5000          # last 5000 trades per symbol
BOOK_TICKER_BUFFER = 1       # only latest


@dataclass
class Kline1m:
    """Single 1-minute kline bar."""
    t: int              # open time (ms)
    o: float            # open
    h: float            # high
    l: float            # low
    c: float            # close
    v: float            # volume (base asset)
    qv: float           # quote volume
    trades: int         # number of trades
    closed: bool        # is this bar final?


@dataclass
class AggTrade:
    """Single aggregated trade."""
    t: int              # timestamp (ms)
    price: float
    qty: float
    is_buyer: bool      # True = buyer is maker (market sell hit bid)
    trade_id: int


@dataclass
class BookTicker:
    """Best bid/ask snapshot."""
    t: int              # update time (ms)
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    spread_bps: float   # spread in basis points


@dataclass
class SymbolBuffers:
    """Per-symbol ring buffers for all stream types."""
    klines_1m: Deque[Kline1m] = field(default_factory=lambda: deque(maxlen=KLINE_1M_BUFFER))
    trades: Deque[AggTrade] = field(default_factory=lambda: deque(maxlen=TRADE_BUFFER))
    book: Optional[BookTicker] = None
    # Derived: 1m numpy array (rebuilt on demand, cached)
    _kline_array_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _kline_cache_len: int = 0


class WSManager:
    """
    Manages Binance WebSocket connections for real-time data.

    Streams per symbol:
      - {sym}@kline_1m      → 1-minute candles
      - {sym}@aggTrade       → individual trades
      - {sym}@bookTicker     → best bid/ask

    Auto-reconnects on disconnect. Thread-safe via asyncio locks.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        enable_trades: bool = True,
        enable_book: bool = True,
        enable_klines: bool = True,
    ):
        self._symbols: Set[str] = set()
        if symbols:
            self._symbols = {s.upper() for s in symbols}
        self._enable_trades = enable_trades
        self._enable_book = enable_book
        self._enable_klines = enable_klines

        self._buffers: Dict[str, SymbolBuffers] = {}
        self._sessions: List[aiohttp.ClientSession] = []
        self._ws_connections: List[aiohttp.ClientWebSocketResponse] = []
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

        # Stats
        self._msg_count = 0
        self._last_msg_ts = 0.0
        self._reconnect_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all WebSocket connections."""
        if self._running:
            return
        self._running = True

        for sym in self._symbols:
            if sym not in self._buffers:
                self._buffers[sym] = SymbolBuffers()

        # Build stream names
        streams = self._build_stream_list()
        if not streams:
            log.warning("WSManager: no streams to subscribe")
            return

        # Split into chunks (Binance limit)
        chunks = [streams[i:i + MAX_STREAMS_PER_CONN]
                  for i in range(0, len(streams), MAX_STREAMS_PER_CONN)]

        for chunk in chunks:
            task = asyncio.create_task(self._ws_loop(chunk))
            self._tasks.append(task)

        log.info("WSManager started: %d symbols, %d streams, %d connections",
                 len(self._symbols), len(streams), len(chunks))

    async def stop(self) -> None:
        """Gracefully stop all connections."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for ws in self._ws_connections:
            await ws.close()
        for session in self._sessions:
            await session.close()
        self._tasks.clear()
        self._ws_connections.clear()
        self._sessions.clear()
        log.info("WSManager stopped")

    async def add_symbols(self, symbols: List[str]) -> None:
        """Hot-add symbols (reconnects affected connections)."""
        new_syms = {s.upper() for s in symbols} - self._symbols
        if not new_syms:
            return
        async with self._lock:
            self._symbols.update(new_syms)
            for sym in new_syms:
                self._buffers[sym] = SymbolBuffers()
        # Restart to pick up new streams
        if self._running:
            await self.stop()
            await self.start()

    def get_klines_array(self, symbol: str, limit: int = 60) -> Optional[np.ndarray]:
        """
        Get 1m klines as numpy structured array (same format as REST klines).
        Returns None if no data available.
        """
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        if not buf or not buf.klines_1m:
            return None

        # Only use closed bars
        closed = [k for k in buf.klines_1m if k.closed]
        if not closed:
            return None

        n = min(limit, len(closed))
        bars = list(closed)[-n:]

        arr = np.zeros(n, dtype=[
            ("t", "i8"), ("o", "f8"), ("h", "f8"),
            ("l", "f8"), ("c", "f8"), ("v", "f8"),
        ])
        for idx, k in enumerate(bars):
            arr[idx] = (k.t, k.o, k.h, k.l, k.c, k.v)
        return arr

    def get_current_bar(self, symbol: str) -> Optional[Kline1m]:
        """Get the current (unclosed) 1m bar, or the latest closed."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        if not buf or not buf.klines_1m:
            return None
        return buf.klines_1m[-1]

    def get_trades(self, symbol: str, limit: int = 500) -> List[AggTrade]:
        """Get recent trades (newest last)."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        if not buf:
            return []
        trades = list(buf.trades)
        return trades[-limit:]

    def get_book_ticker(self, symbol: str) -> Optional[BookTicker]:
        """Get latest bid/ask."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        return buf.book if buf else None

    def get_trade_count_since(self, symbol: str, since_ms: int) -> int:
        """Count trades since timestamp."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        if not buf:
            return 0
        count = 0
        for t in reversed(buf.trades):
            if t.t < since_ms:
                break
            count += 1
        return count

    @property
    def symbols(self) -> Set[str]:
        return set(self._symbols)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "symbols": len(self._symbols),
            "messages": self._msg_count,
            "last_msg_age_s": time.time() - self._last_msg_ts if self._last_msg_ts else None,
            "reconnects": self._reconnect_count,
            "connections": len(self._ws_connections),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_stream_list(self) -> List[str]:
        streams = []
        for sym in sorted(self._symbols):
            s = sym.lower()
            if self._enable_klines:
                streams.append(f"{s}@kline_1m")
            if self._enable_trades:
                streams.append(f"{s}@aggTrade")
            if self._enable_book:
                streams.append(f"{s}@bookTicker")
        return streams

    async def _ws_loop(self, streams: List[str]) -> None:
        """Main WS loop with auto-reconnect."""
        backoff = 1.0
        url = f"{WS_COMBINED}?streams={'/'.join(streams)}"

        while self._running:
            session = aiohttp.ClientSession()
            self._sessions.append(session)
            try:
                async with session.ws_connect(
                    url,
                    heartbeat=20,
                    max_msg_size=0,  # unlimited
                    timeout=aiohttp.ClientWSTimeout(ws_close=10),
                ) as ws:
                    self._ws_connections.append(ws)
                    backoff = 1.0  # reset on successful connect
                    log.info("WS connected: %d streams", len(streams))

                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self._handle_message(msg.data)
                            except Exception:
                                log.exception("WS message handler error")
                        elif msg.type in (aiohttp.WSMsgType.ERROR,
                                          aiohttp.WSMsgType.CLOSED):
                            break

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                log.warning("WS connection error: %s", e)
            except asyncio.CancelledError:
                break
            finally:
                if ws in self._ws_connections:
                    self._ws_connections.remove(ws)
                await session.close()
                if session in self._sessions:
                    self._sessions.remove(session)

            if self._running:
                self._reconnect_count += 1
                log.info("WS reconnecting in %.1fs (attempt %d)",
                         backoff, self._reconnect_count)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _handle_message(self, raw: str) -> None:
        """Route incoming WS message to appropriate handler."""
        data = json.loads(raw)
        self._msg_count += 1
        self._last_msg_ts = time.time()

        # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
        stream = data.get("stream", "")
        payload = data.get("data", data)

        if "@kline_" in stream:
            self._handle_kline(payload)
        elif "@aggTrade" in stream:
            self._handle_agg_trade(payload)
        elif "@bookTicker" in stream:
            self._handle_book_ticker(payload)

    def _handle_kline(self, data: dict) -> None:
        """Process kline stream message."""
        k = data.get("k", {})
        sym = str(k.get("s", "")).upper()
        if sym not in self._buffers:
            return

        bar = Kline1m(
            t=int(k.get("t", 0)),
            o=float(k.get("o", 0)),
            h=float(k.get("h", 0)),
            l=float(k.get("l", 0)),
            c=float(k.get("c", 0)),
            v=float(k.get("v", 0)),
            qv=float(k.get("q", 0)),
            trades=int(k.get("n", 0)),
            closed=bool(k.get("x", False)),
        )

        buf = self._buffers[sym]
        # Update or append
        if buf.klines_1m and buf.klines_1m[-1].t == bar.t:
            buf.klines_1m[-1] = bar
        else:
            buf.klines_1m.append(bar)

        # Invalidate numpy cache
        buf._kline_array_cache = None

    def _handle_agg_trade(self, data: dict) -> None:
        """Process aggTrade stream message."""
        sym = str(data.get("s", "")).upper()
        if sym not in self._buffers:
            return

        trade = AggTrade(
            t=int(data.get("T", data.get("t", 0))),
            price=float(data.get("p", 0)),
            qty=float(data.get("q", 0)),
            is_buyer=not bool(data.get("m", False)),  # m=True means buyer is maker
            trade_id=int(data.get("a", 0)),
        )
        self._buffers[sym].trades.append(trade)

    def _handle_book_ticker(self, data: dict) -> None:
        """Process bookTicker stream message."""
        sym = str(data.get("s", "")).upper()
        if sym not in self._buffers:
            return

        bid = float(data.get("b", 0))
        ask = float(data.get("a", 0))
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 1.0
        spread_bps = (ask - bid) / mid * 10000.0 if mid > 0 else 0.0

        self._buffers[sym].book = BookTicker(
            t=int(data.get("u", 0)),
            bid=bid,
            bid_qty=float(data.get("B", 0)),
            ask=ask,
            ask_qty=float(data.get("A", 0)),
            spread_bps=spread_bps,
        )


# ── Convenience: global singleton ────────────────────────────────────────────
_global_ws: Optional[WSManager] = None


async def get_ws_manager(symbols: Optional[List[str]] = None) -> WSManager:
    """Get or create global WSManager instance."""
    global _global_ws
    if _global_ws is None:
        _global_ws = WSManager(symbols=symbols)
        await _global_ws.start()
    elif symbols:
        await _global_ws.add_symbols(symbols)
    return _global_ws


async def stop_ws_manager() -> None:
    """Stop global WSManager."""
    global _global_ws
    if _global_ws:
        await _global_ws.stop()
        _global_ws = None
