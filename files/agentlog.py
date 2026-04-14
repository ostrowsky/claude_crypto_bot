from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as _np


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return super().default(obj)


LOG_FILE = Path("agent_events.jsonl")
_pylog = logging.getLogger("agentlog")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(event: Dict[str, Any]) -> None:
    event.setdefault("ts", _now())
    event.setdefault("source", "market_agent")
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, cls=_SafeEncoder) + "\n")
    except Exception as exc:
        _pylog.warning("agentlog write error: %s", exc)


def log_analysis(n_scanned: int, n_entries: int, n_open_positions: int) -> None:
    _write(
        {
            "event": "analysis_done",
            "n_scanned": n_scanned,
            "n_entries": n_entries,
            "n_open_positions": n_open_positions,
        }
    )


def log_entry(
    sym: str,
    tf: str,
    mode: str,
    price: float,
    ema20: float,
    slope: float,
    rsi: float,
    adx: float,
    vol_x: float,
    macd_hist: float,
    daily_range: float,
    trail_k: float,
    max_hold_bars: int,
    forecast_return_pct: float = 0.0,
    today_change_pct: float = 0.0,
) -> None:
    _write(
        {
            "event": "entry",
            "sym": sym,
            "tf": tf,
            "mode": mode,
            "price": price,
            "ema20": round(ema20, 8),
            "slope_pct": round(slope, 3),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "vol_x": round(vol_x, 2),
            "macd_hist": round(macd_hist, 8),
            "daily_range": round(daily_range, 2),
            "trail_k": trail_k,
            "max_hold_bars": max_hold_bars,
            "forecast_return_pct": round(float(forecast_return_pct), 4),
            "today_change_pct": round(float(today_change_pct), 4),
        }
    )


def log_exit(
    sym: str,
    tf: str,
    mode: str,
    entry_price: float,
    exit_price: float,
    reason: str,
    bars_held: int,
    trail_k: float,
) -> None:
    pnl = round((exit_price / entry_price - 1.0) * 100.0, 3) if entry_price > 0 else 0.0
    _write(
        {
            "event": "exit",
            "sym": sym,
            "tf": tf,
            "mode": mode,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl,
            "reason": reason,
            "bars_held": bars_held,
            "trail_k": trail_k,
        }
    )


def log_blocked(
    sym: str,
    tf: str,
    price: float,
    reason: str,
    *,
    signal_type: str = "buy",
    rsi: Optional[float] = None,
    adx: Optional[float] = None,
    vol_x: Optional[float] = None,
    daily_range: Optional[float] = None,
) -> None:
    rec: Dict[str, Any] = {
        "event": "blocked",
        "sym": sym,
        "tf": tf,
        "signal_type": signal_type,
        "price": price,
        "reason": reason,
    }
    if rsi is not None:
        rec["rsi"] = round(rsi, 1)
    if adx is not None:
        rec["adx"] = round(adx, 1)
    if vol_x is not None:
        rec["vol_x"] = round(vol_x, 2)
    if daily_range is not None:
        rec["daily_range"] = round(daily_range, 2)
    _write(rec)
