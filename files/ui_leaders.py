"""Today's watchlist leaders by daily MOVE, with why each one is or is not held.

Why this exists
---------------
On 2026-08-20 the operator's status panel read "Открытых сигналов: 0" while three
watchlist coins sat in Binance's daily top-20 (XRP, ORDI, ENA, all +19-21% on the
day). The panel was accurate and useless: it reported an empty portfolio without
saying what the market was doing or why nothing had been taken. Diagnosing that
took a full log excavation.

This module puts the answer on the panel: the day's biggest movers from the
watchlist, whether each is held, and if not, which gate last rejected it.

MOVE, not change
----------------
Ranked by `high / open - 1` for the rolling 24h window, not by `priceChangePercent`.
The project's target is the day's largest MOVE — a coin that ran +20% and gave it
all back belongs on this list, because the bot's job was to catch the run. This is
the same definition used in the trend-start work and in durable memory.

Threading
---------
The render path has a hard deadline and the loop already reports lag warnings, so
nothing here may block it. A daemon thread refreshes on its own cadence and
`get_cached()` returns the last completed snapshot instantly, or an empty tuple
before the first refresh lands. No network call ever happens on the UI path.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

SPOT_TICKER = "https://api.binance.com/api/v3/ticker/24hr"
REFRESH_SEC = 60.0
HTTP_TIMEOUT = 20.0
DEFAULT_LIMIT = 8

# Last gate to reject each symbol: sym -> (unix_ts, reason_code, human_reason).
# Written from botlog.log_blocked, which every gate funnels through, so this
# stays a single in-memory dict update and never touches disk.
_LAST_BLOCK: dict = {}
_BLOCK_LOCK = threading.Lock()

_leaders: tuple = ()
_leaders_at: float = 0.0
_LEAD_LOCK = threading.Lock()
_thread: Optional[threading.Thread] = None


@dataclass(frozen=True)
class Leader:
    sym: str
    move_pct: float          # high vs open over the rolling 24h window
    change_pct: float        # close vs open, for contrast
    last: float
    held: bool
    block_reason: str        # "" when held or never seen


def note_block(sym: str, reason_code: Optional[str], reason: Optional[str]) -> None:
    """Record the most recent rejection for a symbol. Called from botlog."""
    if not sym:
        return
    with _BLOCK_LOCK:
        _LAST_BLOCK[str(sym)] = (time.time(), str(reason_code or ""), str(reason or ""))


def last_block(sym: str, max_age_sec: float = 3600.0) -> str:
    """Human-readable reason the symbol was last rejected, '' if none/stale."""
    with _BLOCK_LOCK:
        rec = _LAST_BLOCK.get(str(sym))
    if not rec:
        return ""
    ts, code, reason = rec
    if time.time() - ts > max_age_sec:
        return ""
    return code or reason


def _fetch_tickers() -> list:
    with urllib.request.urlopen(SPOT_TICKER, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode())


def compute_leaders(watchlist: Sequence[str],
                    held_syms: Sequence[str],
                    limit: int = DEFAULT_LIMIT,
                    tickers: Optional[list] = None) -> tuple:
    """Rank the watchlist by 24h MOVE. Pure given `tickers`, so it is testable."""
    wl = {str(s) for s in watchlist}
    held = {str(s) for s in held_syms}
    rows = []
    for r in (tickers if tickers is not None else _fetch_tickers()):
        sym = r.get("symbol")
        if sym not in wl:
            continue
        try:
            op = float(r["openPrice"])
            hi = float(r["highPrice"])
            last = float(r["lastPrice"])
            chg = float(r["priceChangePercent"])
        except (KeyError, TypeError, ValueError):
            continue
        if op <= 0:
            continue
        rows.append(Leader(
            sym=sym,
            move_pct=(hi / op - 1.0) * 100.0,
            change_pct=chg,
            last=last,
            held=sym in held,
            block_reason="" if sym in held else last_block(sym),
        ))
    rows.sort(key=lambda x: -x.move_pct)
    return tuple(rows[:max(1, limit)])


def get_cached() -> tuple:
    """Instant, never blocks. Empty before the first refresh completes."""
    with _LEAD_LOCK:
        return _leaders


def cache_age_sec() -> float:
    with _LEAD_LOCK:
        return float("inf") if not _leaders_at else time.time() - _leaders_at


def refresh_once(watchlist_fn: Callable[[], Sequence[str]],
                 held_fn: Callable[[], Sequence[str]],
                 limit: int = DEFAULT_LIMIT) -> tuple:
    global _leaders, _leaders_at
    rows = compute_leaders(watchlist_fn(), held_fn(), limit=limit)
    with _LEAD_LOCK:
        _leaders = rows
        _leaders_at = time.time()
    return rows


def start(watchlist_fn: Callable[[], Sequence[str]],
          held_fn: Callable[[], Sequence[str]],
          limit: int = DEFAULT_LIMIT,
          interval: float = REFRESH_SEC) -> None:
    """Start the background refresher once. Safe to call repeatedly."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return

    def _loop():
        while True:
            try:
                refresh_once(watchlist_fn, held_fn, limit=limit)
            except Exception:
                # A ticker outage must never take the UI thread down; the panel
                # shows a staleness note instead.
                pass
            time.sleep(interval)

    _thread = threading.Thread(target=_loop, name="ui-leaders", daemon=True)
    _thread.start()


def render(rows: Sequence[Leader], age_sec: float = 0.0) -> str:
    """Telegram-ready block. Never raises; an empty list has its own message."""
    if not rows:
        return "_лидеры дня ещё не загружены_"
    out = []
    for r in rows:
        mark = "✅" if r.held else "▫️"
        tail = ""
        if r.held:
            tail = " — *в портфеле*"
        elif r.block_reason:
            tail = " — отклонён: `%s`" % r.block_reason[:34]
        out.append("%s *%s* +%.1f%% (закр %+.1f%%)%s"
                   % (mark, r.sym.replace("USDT", ""), r.move_pct, r.change_pct, tail))
    if age_sec > 3 * REFRESH_SEC:
        out.append("_данные устарели на %.0f мин_" % (age_sec / 60.0))
    return "\n".join(out)
