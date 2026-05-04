"""Signal-efficiency evaluator for claude_crypto_bot.

Reads bot_events.jsonl + price history, labels ground-truth uptrends with a
ZigZag-style swing detector, matches BUY/SELL signals to trends, and writes
a structured feedback report.

Run with --help for the full CLI. The script is deliberately self-contained:
the only third-party deps are pandas + requests (both already in the bot's
requirements.txt).

Author's note: every assumption about input schema is gated behind
NORMALIZE_EVENT() so you can swap it out if your bot logs differently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import uuid

# Force UTF-8 stdout/stderr (Windows cp1251 default chokes on Unicode arrows etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:
    sys.stderr.write("This script needs pandas. Run: pip install pandas requests\n")
    raise

try:
    import requests
except ImportError:
    requests = None  # only needed if we have to fetch klines from Binance


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Event:
    ts: pd.Timestamp
    symbol: str
    action: str  # 'BUY' / 'SELL' / 'HOLD' / other (ignored)
    price: float
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trend:
    symbol: str
    start_idx: int
    start_ts: pd.Timestamp
    start_price: float
    end_idx: int
    end_ts: pd.Timestamp
    end_price: float

    @property
    def gain_pct(self) -> float:
        return (self.end_price - self.start_price) / self.start_price * 100

    @property
    def duration_bars(self) -> int:
        return self.end_idx - self.start_idx


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

# Map from "field we want" -> list of "candidate field names in the wild".
# Edit this if your event schema differs and you don't want to use --schema-map.
DEFAULT_FIELD_ALIASES = {
    "ts":       ["ts", "timestamp", "time", "datetime"],
    "symbol":   ["symbol", "pair", "ticker", "asset"],
    "action":   ["action", "event", "signal", "signal_type", "side", "type"],
    "price":    ["price", "close", "entry_price", "px"],
    "confidence": ["confidence", "score", "prob", "probability", "conf"],
}


def _coerce_ts(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        # heuristic: ms vs s
        if value > 10**12:
            return pd.Timestamp(value, unit="ms", tz="UTC")
        return pd.Timestamp(value, unit="s", tz="UTC")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse timestamp: {value!r}")
    return ts


def normalize_event(raw: dict[str, Any], aliases: dict[str, list[str]]) -> Event | None:
    """Convert a raw JSONL line into an Event, or None if it isn't a signal we care about."""
    def get(key: str) -> Any:
        for cand in aliases[key]:
            if cand in raw and raw[cand] is not None:
                return raw[cand]
        return None

    action = get("action")
    if action is None:
        return None
    action = str(action).upper().strip()
    if action not in {"BUY", "SELL"}:
        return None  # silently drop HOLDs / SCAN / etc.

    ts = get("ts")
    symbol = get("symbol")
    price = get("price")
    if ts is None or symbol is None or price is None:
        return None

    return Event(
        ts=_coerce_ts(ts),
        symbol=str(symbol).upper(),
        action=action,
        price=float(price),
        confidence=float(get("confidence")) if get("confidence") is not None else None,
        raw=raw,
    )


def load_events(path: Path, aliases: dict[str, list[str]]) -> list[Event]:
    events: list[Event] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            ev = normalize_event(raw, aliases)
            if ev is not None:
                events.append(ev)
            else:
                skipped += 1
    events.sort(key=lambda e: (e.symbol, e.ts))
    sys.stderr.write(f"[loader] events loaded: {len(events)}, skipped: {skipped}\n")
    return events


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

TF_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440,
}


def fetch_klines_from_binance(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Bar]:
    """Last-resort price loader. Public Binance REST, no API key needed."""
    if requests is None:
        raise RuntimeError("requests not installed; cannot fetch klines")
    url = "https://api.binance.com/api/v3/klines"
    bars: list[Bar] = []
    cur = start
    interval = timeframe
    while cur < end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(cur.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        for k in chunk:
            bars.append(Bar(
                ts=pd.Timestamp(k[0], unit="ms", tz="UTC"),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))
        last_ts = pd.Timestamp(chunk[-1][0], unit="ms", tz="UTC")
        if last_ts <= cur:
            break
        cur = last_ts + pd.Timedelta(minutes=TF_TO_MINUTES[interval])
        if len(chunk) < 1000:
            break
    return bars


def load_price_history(
    project_root: Path,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Bar]:
    """Try local cache first, fall back to live fetch."""
    candidates = [
        project_root / "history" / f"{symbol}_{timeframe}.parquet",
        project_root / "history" / f"{symbol}_{timeframe}.csv",
        project_root / "data" / f"{symbol}_{timeframe}.parquet",
        project_root / "data" / f"{symbol}_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            df = _normalize_klines_df(df)
            df = df[(df["ts"] >= start) & (df["ts"] <= end)]
            return [Bar(ts=row.ts, open=row.open, high=row.high, low=row.low,
                        close=row.close, volume=row.volume) for row in df.itertuples()]
    sys.stderr.write(f"[loader] no local cache for {symbol} {timeframe}; "
                     f"fetching from Binance...\n")
    return fetch_klines_from_binance(symbol, timeframe, start, end)


def _normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for want in ["ts", "open", "high", "low", "close", "volume"]:
        for cand in [want, want.capitalize(), want.upper(),
                     want + "_time" if want == "ts" else want]:
            if cand in df.columns:
                rename_map[cand] = want
                break
    df = df.rename(columns=rename_map)
    if "ts" not in df.columns:
        # try a 'time' or 'date' column
        for c in ["time", "date", "datetime", "timestamp"]:
            if c in df.columns:
                df = df.rename(columns={c: "ts"})
                break
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Trend labeling (ZigZag-style swing detection)
# ---------------------------------------------------------------------------

def detect_trends(
    bars: list[Bar],
    swing_threshold_pct: float,
    max_intratrend_drawdown_pct: float,
    min_trend_duration_bars: int,
    symbol: str,
) -> list[Trend]:
    """See references/trend_definition.md for the algorithm and rationale."""
    if len(bars) < 2:
        return []

    direction = "up"
    swing_low_idx, swing_low_price = 0, bars[0].low
    swing_high_idx, swing_high_price = 0, bars[0].high
    trends: list[Trend] = []

    for i in range(1, len(bars)):
        bar = bars[i]
        if direction == "up":
            if bar.high > swing_high_price:
                swing_high_idx, swing_high_price = i, bar.high
                continue
            drawdown_pct = (swing_high_price - bar.low) / swing_high_price * 100
            if drawdown_pct >= max_intratrend_drawdown_pct:
                gain_pct = (swing_high_price - swing_low_price) / swing_low_price * 100
                duration = swing_high_idx - swing_low_idx
                if gain_pct >= swing_threshold_pct and duration >= min_trend_duration_bars:
                    trends.append(Trend(
                        symbol=symbol,
                        start_idx=swing_low_idx,
                        start_ts=bars[swing_low_idx].ts,
                        start_price=swing_low_price,
                        end_idx=swing_high_idx,
                        end_ts=bars[swing_high_idx].ts,
                        end_price=swing_high_price,
                    ))
                direction = "down"
                swing_low_idx, swing_low_price = i, bar.low
        else:
            if bar.low < swing_low_price:
                swing_low_idx, swing_low_price = i, bar.low
                continue
            rebound_pct = (bar.high - swing_low_price) / swing_low_price * 100
            if rebound_pct >= max_intratrend_drawdown_pct:
                direction = "up"
                swing_high_idx, swing_high_price = i, bar.high

    # Open trend at end of window: if we're still tracking an up-swing and
    # the unrealized gain already exceeds the threshold, emit it so live
    # evaluation captures the most recent move. The swing high may not be
    # final (price could continue), but for grading the bot's behavior so
    # far this is the right thing to do.
    if direction == "up":
        gain_pct = (swing_high_price - swing_low_price) / swing_low_price * 100
        duration = swing_high_idx - swing_low_idx
        if gain_pct >= swing_threshold_pct and duration >= min_trend_duration_bars:
            trends.append(Trend(
                symbol=symbol,
                start_idx=swing_low_idx,
                start_ts=bars[swing_low_idx].ts,
                start_price=swing_low_price,
                end_idx=swing_high_idx,
                end_ts=bars[swing_high_idx].ts,
                end_price=swing_high_price,
            ))
    return trends


# ---------------------------------------------------------------------------
# Signal -> trend matching
# ---------------------------------------------------------------------------

@dataclass
class TradeVerdict:
    trade_id: str
    symbol: str
    buy_event: Event
    sell_event: Event | None
    matched_trend: Trend | None
    buy_bar_idx: int
    sell_bar_idx: int | None
    fee_pct: float

    def asdict(self) -> dict:
        out = {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "buy_signal": _event_to_dict(self.buy_event),
            "sell_signal": _event_to_dict(self.sell_event) if self.sell_event else None,
            "matched_trend": _trend_to_dict(self.matched_trend) if self.matched_trend else None,
            "buy_lateness_bars": self.buy_lateness_bars,
            "buy_lateness_pct_of_move": self.buy_lateness_pct_of_move,
            "sell_lateness_bars": self.sell_lateness_bars,
            "sell_lateness_pct_of_move": self.sell_lateness_pct_of_move,
            "captured_pnl_pct": self.captured_pnl_pct,
            "available_pnl_pct": self.available_pnl_pct,
            "capture_ratio": self.capture_ratio,
            "verdict": self.verdict,
            "verdict_summary": self.verdict_summary,
        }
        return out

    @property
    def buy_lateness_bars(self) -> int | None:
        if not self.matched_trend:
            return None
        return self.buy_bar_idx - self.matched_trend.start_idx

    @property
    def buy_lateness_pct_of_move(self) -> float | None:
        if not self.matched_trend:
            return None
        t = self.matched_trend
        denom = t.end_price - t.start_price
        if denom <= 0:
            return None
        return (self.buy_event.price - t.start_price) / denom * 100

    @property
    def sell_lateness_bars(self) -> int | None:
        if not self.matched_trend or self.sell_bar_idx is None:
            return None
        return self.sell_bar_idx - self.matched_trend.end_idx

    @property
    def sell_lateness_pct_of_move(self) -> float | None:
        if not self.matched_trend or not self.sell_event:
            return None
        t = self.matched_trend
        denom = t.end_price - self.buy_event.price
        if denom <= 0:
            return None
        if (self.sell_bar_idx or 0) > t.end_idx:
            return (t.end_price - self.sell_event.price) / denom * 100
        return -(t.end_price - self.sell_event.price) / denom * 100

    @property
    def captured_pnl_pct(self) -> float | None:
        if not self.sell_event:
            return None
        buy = self.buy_event.price * (1 + self.fee_pct / 100)
        sell = self.sell_event.price * (1 - self.fee_pct / 100)
        return (sell - buy) / buy * 100

    @property
    def available_pnl_pct(self) -> float | None:
        if not self.matched_trend:
            return None
        t = self.matched_trend
        return (t.end_price - t.start_price) / t.start_price * 100

    @property
    def capture_ratio(self) -> float | None:
        if self.captured_pnl_pct is None or self.available_pnl_pct in (None, 0):
            return None
        return self.captured_pnl_pct / self.available_pnl_pct

    @property
    def verdict(self) -> str:
        if self.captured_pnl_pct is not None and self.captured_pnl_pct < 0:
            return "losing_trade"
        bl = self.buy_lateness_bars
        sl = self.sell_lateness_bars
        if bl is None or sl is None:
            return "incomplete"
        if bl <= 1 and -1 <= sl <= 1:
            return "optimal"
        if bl > 1 and -1 <= sl <= 1:
            return "late_entry_optimal_exit"
        if bl <= 1 and sl > 1:
            return "optimal_entry_late_exit"
        if bl <= 1 and sl < -1:
            return "optimal_entry_premature_exit"
        if bl > 1 and sl > 1:
            return "late_entry_late_exit"
        if bl > 1 and sl < -1:
            return "late_entry_premature_exit"
        return "incomplete"

    @property
    def verdict_summary(self) -> str:
        bl = self.buy_lateness_bars
        blpct = self.buy_lateness_pct_of_move
        slpct = self.sell_lateness_pct_of_move
        cr = self.capture_ratio
        if bl is None or cr is None:
            return "Incomplete trade — no matched trend or no sell."
        parts = []
        if bl > 1:
            parts.append(f"Bought {bl} bars late ({blpct:.1f}% of the move was already gone)")
        elif bl <= 0:
            parts.append("Bought at or before the swing low (good)")
        else:
            parts.append("Bought roughly at the swing low (good)")
        if slpct is not None:
            if slpct > 5:
                parts.append(f"sold late, giving back {slpct:.1f}% of unrealized peak")
            elif slpct < -5:
                parts.append(f"sold prematurely, missing {-slpct:.1f}% of remaining upside")
            else:
                parts.append("sold near the peak (good)")
        parts.append(f"net captured {cr*100:.0f}% of available profit")
        return ". ".join(parts) + "."


def _event_to_dict(ev: Event | None) -> dict | None:
    if ev is None:
        return None
    return {
        "ts": ev.ts.isoformat(),
        "price": ev.price,
        "confidence": ev.confidence,
    }


def _trend_to_dict(t: Trend | None) -> dict | None:
    if t is None:
        return None
    return {
        "true_start_ts": t.start_ts.isoformat(),
        "true_start_price": t.start_price,
        "true_end_ts": t.end_ts.isoformat(),
        "true_end_price": t.end_price,
        "gain_pct": t.gain_pct,
        "duration_bars": t.duration_bars,
    }


def match_signals_to_trends(
    events: list[Event],
    trends: list[Trend],
    bars: list[Bar],
    fee_pct: float,
    swing_threshold_pct: float,
    max_intratrend_drawdown_pct: float,
) -> tuple[list[TradeVerdict], list[Trend], list[Event]]:
    """Trade-pair-centric matching.

    Step 1: Pair BUY/SELL events into round-trip trades (trader-style: a BUY
            followed by the next SELL of the same symbol). Unpaired BUYs are
            still treated as trades with sell_event=None.
    Step 2: For each trade, find the trend whose window [start, end] best
            overlaps the trade. A trade is matched if it sits substantially
            inside a trend; otherwise it is a false-positive candidate.
    Step 3: A trade not matched to any trend is confirmed false-positive only
            if no swing_threshold_pct upward move occurred inside its lifetime.
    Step 4: Trends with no matched trade in their window are 'missed'.
    """
    if not bars:
        return [], list(trends), [e for e in events if e.action == "BUY"]

    def nearest_bar_idx(ts: pd.Timestamp) -> int:
        lo, hi = 0, len(bars) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if bars[mid].ts < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # Step 1 — pair BUYs and SELLs in chronological order
    by_symbol: dict[str, list[Event]] = {}
    for e in events:
        by_symbol.setdefault(e.symbol, []).append(e)
    for lst in by_symbol.values():
        lst.sort(key=lambda x: x.ts)

    @dataclass
    class TradePair:
        symbol: str
        buy: Event
        sell: Event | None

    pairs: list[TradePair] = []
    for sym, evs in by_symbol.items():
        in_position = False
        cur_buy = None
        for e in evs:
            if e.action == "BUY":
                if in_position:
                    # consecutive BUY without SELL — close out the previous as
                    # an open trade and start a new one.
                    pairs.append(TradePair(sym, cur_buy, None))
                cur_buy = e
                in_position = True
            elif e.action == "SELL":
                if in_position and cur_buy is not None:
                    pairs.append(TradePair(sym, cur_buy, e))
                    cur_buy = None
                    in_position = False
                # SELL without BUY is ignored (the bot may have shorted; out of scope here)
        if in_position and cur_buy is not None:
            pairs.append(TradePair(sym, cur_buy, None))  # still-open trade

    # Step 2 — match each trade pair to its best-fitting trend
    used_trends: set[int] = set()
    verdicts: list[TradeVerdict] = []
    unmatched_pairs: list[TradePair] = []

    for pair in pairs:
        candidates = []
        for i, t in enumerate(trends):
            if i in used_trends or t.symbol != pair.symbol:
                continue
            # how much of the pair's lifetime overlaps the trend window?
            buy_ts = pair.buy.ts
            sell_ts = pair.sell.ts if pair.sell else t.end_ts
            overlap_start = max(buy_ts, t.start_ts)
            overlap_end = min(sell_ts, t.end_ts)
            if overlap_end <= overlap_start:
                # no overlap at all — but maybe BUY is just before the swing low?
                # accept up to 3 bars of slack on either side
                tf_min = TF_TO_MINUTES.get(_infer_tf(bars), 5)
                slack = pd.Timedelta(minutes=3 * tf_min)
                if (buy_ts >= t.start_ts - slack and buy_ts <= t.end_ts + slack):
                    candidates.append((i, t, 0.0))  # tiebreak by closeness
                continue
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            candidates.append((i, t, overlap_seconds))

        if not candidates:
            unmatched_pairs.append(pair)
            continue

        # pick the trend with the largest overlap; ties broken by closer to BUY
        candidates.sort(key=lambda x: (-x[2], abs((x[1].start_ts - pair.buy.ts).total_seconds())))
        i, t, _ = candidates[0]
        used_trends.add(i)

        verdicts.append(TradeVerdict(
            trade_id=f"t_{len(verdicts)+1:03d}",
            symbol=pair.symbol,
            buy_event=pair.buy,
            sell_event=pair.sell,
            matched_trend=t,
            buy_bar_idx=nearest_bar_idx(pair.buy.ts),
            sell_bar_idx=nearest_bar_idx(pair.sell.ts) if pair.sell else None,
            fee_pct=fee_pct,
        ))

    # Step 3 — confirm false positives by forward-looking price check
    false_positives: list[Event] = []
    for pair in unmatched_pairs:
        buy_idx = nearest_bar_idx(pair.buy.ts)
        # look forward up to 2 * min_trend_duration_bars; if price never rose
        # by swing_threshold_pct before dropping by max_intratrend_drawdown_pct,
        # this is a real false positive.
        lookahead = 12  # ~min_trend_duration_bars * 4; conservative
        confirmed_false = True
        peak = pair.buy.price
        for k in range(buy_idx + 1, min(buy_idx + 1 + lookahead, len(bars))):
            peak = max(peak, bars[k].high)
            up_pct = (peak - pair.buy.price) / pair.buy.price * 100
            dd_pct = (peak - bars[k].low) / peak * 100 if peak else 0
            if up_pct >= swing_threshold_pct:
                confirmed_false = False
                break
            if dd_pct >= max_intratrend_drawdown_pct and up_pct < swing_threshold_pct:
                # decisively turned down before reaching threshold
                break
        if confirmed_false:
            false_positives.append(pair.buy)
        else:
            # the BUY did catch a real move, just one the labeler didn't keep
            # (e.g., gain just below swing_threshold_pct). Record it as a
            # verdict with no matched_trend so capture metrics are skipped
            # but it doesn't pollute the false-positive count.
            verdicts.append(TradeVerdict(
                trade_id=f"t_{len(verdicts)+1:03d}",
                symbol=pair.symbol,
                buy_event=pair.buy,
                sell_event=pair.sell,
                matched_trend=None,
                buy_bar_idx=buy_idx,
                sell_bar_idx=nearest_bar_idx(pair.sell.ts) if pair.sell else None,
                fee_pct=fee_pct,
            ))

    # Step 4 — trends with no matched trade are missed
    missed = [t for i, t in enumerate(trends) if i not in used_trends]
    return verdicts, missed, false_positives


def _infer_tf(bars: list[Bar]) -> str:
    if len(bars) < 2:
        return "5m"
    delta_min = (bars[1].ts - bars[0].ts).total_seconds() / 60
    for k, v in TF_TO_MINUTES.items():
        if abs(v - delta_min) < 0.5:
            return k
    return "5m"


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def median_or_none(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(vals) if vals else None


def percentile_or_none(xs: Iterable[float | None], p: float) -> float | None:
    vals = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def build_summary(
    verdicts: list[TradeVerdict],
    missed: list[Trend],
    false_positives: list[Event],
    total_buys: int,
    total_sells: int,
    bars_by_symbol: dict[str, list[Bar]],
) -> dict:
    total_trends = len(verdicts) + len(missed)
    completed = [v for v in verdicts if v.sell_event is not None]
    capture_ratios = [v.capture_ratio for v in completed]
    captured_pnls = [v.captured_pnl_pct for v in completed if v.captured_pnl_pct is not None]
    wins = [p for p in captured_pnls if p > 0]
    losses = [p for p in captured_pnls if p <= 0]

    # buy-and-hold across symbols, equal-weight average
    bh_returns = []
    for sym, bars in bars_by_symbol.items():
        if len(bars) >= 2:
            bh_returns.append((bars[-1].close - bars[0].close) / bars[0].close * 100)
    bh_pnl = sum(bh_returns) / len(bh_returns) if bh_returns else 0.0

    total_pnl = sum(captured_pnls) if captured_pnls else 0.0

    # max drawdown of equity curve (each trade equal-weight)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in captured_pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # sell-lateness categorization
    too_late = sum(1 for v in verdicts if v.sell_lateness_bars is not None and v.sell_lateness_bars > 1)
    premature = sum(1 for v in verdicts if v.sell_lateness_bars is not None and v.sell_lateness_bars < -1)
    optimal_exit = sum(1 for v in verdicts if v.sell_lateness_bars is not None and -1 <= v.sell_lateness_bars <= 1)

    return {
        "total_trends_in_period": total_trends,
        "trends_caught_with_buy": len(verdicts),
        "trends_missed": len(missed),
        "miss_rate": (len(missed) / total_trends) if total_trends else 0.0,
        "total_buy_signals": total_buys,
        "buys_into_real_trend": len(verdicts),
        "false_positive_buys": len(false_positives),
        "false_positive_rate": (len(false_positives) / total_buys) if total_buys else 0.0,
        "total_sell_signals": total_sells,
        "sell_lateness_too_late_count": too_late,
        "sell_lateness_premature_count": premature,
        "sell_lateness_optimal_count": optimal_exit,
        "median_buy_lateness_bars": median_or_none([v.buy_lateness_bars for v in verdicts]),
        "median_buy_lateness_pct_of_move": median_or_none([v.buy_lateness_pct_of_move for v in verdicts]),
        "median_sell_lateness_bars": median_or_none([v.sell_lateness_bars for v in verdicts]),
        "median_capture_ratio": median_or_none(capture_ratios),
        "p25_capture_ratio": percentile_or_none(capture_ratios, 0.25),
        "p75_capture_ratio": percentile_or_none(capture_ratios, 0.75),
        "total_realized_pnl_pct": total_pnl,
        "buy_and_hold_pnl_pct": bh_pnl,
        "alpha_vs_buy_and_hold_pct": total_pnl - bh_pnl,
        "win_rate": (len(wins) / len(captured_pnls)) if captured_pnls else None,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "max_drawdown_pct": max_dd,
    }


def detect_patterns(verdicts: list[TradeVerdict], summary: dict) -> list[dict]:
    patterns = []
    mbl = summary.get("median_buy_lateness_bars")
    if mbl is not None and mbl >= 3:
        late_count = sum(1 for v in verdicts if (v.buy_lateness_bars or 0) >= 3)
        evidence = (f"Median BUY lateness = {mbl:.0f} bars; {late_count} of {len(verdicts)} "
                    f"caught trends had lateness >= 3 bars.")
        impact = summary.get("median_buy_lateness_pct_of_move") or 0
        patterns.append({
            "pattern_id": "consistent_late_entry",
            "evidence": evidence,
            "estimated_impact_pct": impact,
            "recommendation": (
                "Investigate the entry confirmation chain. The most common culprits "
                "are ADX/ATR/RSI confirmation thresholds set too high, or a "
                "multi-bar smoothing that adds latency. Try lowering ADX threshold "
                "or replacing slow confirmations with a fast trend trigger "
                "(5-bar EMA slope, breakout of 10-bar high)."
            ),
        })
    mfp = summary.get("false_positive_rate")
    if mfp is not None and mfp >= 0.3:
        patterns.append({
            "pattern_id": "high_false_positive_rate",
            "evidence": f"False-positive BUY rate = {mfp*100:.0f}%.",
            "estimated_impact_pct": -mfp * 2,
            "recommendation": (
                "Add a regime filter: suppress BUY when 20-bar range < 2x ATR "
                "(ranging market). Alternatively, raise the ML ranker threshold "
                "via the acceptance gate in config.py."
            ),
        })
    msl = summary.get("median_sell_lateness_bars")
    if msl is not None and msl >= 3:
        patterns.append({
            "pattern_id": "consistent_late_exit",
            "evidence": f"Median SELL lateness = {msl:.0f} bars after the swing high.",
            "estimated_impact_pct": -3.0,
            "recommendation": (
                "Tighten the trailing-stop or add a momentum-fade trigger. "
                "E.g., exit when RSI crosses below 60 from above, or when 3 "
                "consecutive bars close below the 5-bar EMA."
            ),
        })
    elif msl is not None and msl <= -2:
        patterns.append({
            "pattern_id": "consistent_premature_exit",
            "evidence": f"Median SELL lateness = {msl:.0f} bars (early).",
            "estimated_impact_pct": -2.5,
            "recommendation": (
                "Loosen the exit. The current logic likely fires on the first "
                "pullback. Require 2 consecutive bearish closes or a larger "
                "ATR-multiple stop."
            ),
        })
    miss = summary.get("miss_rate")
    if miss is not None and miss >= 0.5:
        patterns.append({
            "pattern_id": "high_miss_rate",
            "evidence": f"Missed {miss*100:.0f}% of sustainable uptrends in the window.",
            "estimated_impact_pct": -miss * 5,
            "recommendation": (
                "The entry filter is too restrictive overall. Consider running "
                "the ML candidate ranker shadow report and lowering the "
                "acceptance-gate AUC threshold by 0.02."
            ),
        })
    return patterns


def build_coaching_examples(verdicts: list[TradeVerdict], missed: list[Trend]) -> list[dict]:
    """Pick 3 instructive cases: one well-caught, one late-but-profitable, one missed."""
    examples = []
    if verdicts:
        good = [v for v in verdicts if v.verdict == "optimal" and v.captured_pnl_pct]
        if good:
            v = max(good, key=lambda x: x.captured_pnl_pct)
            examples.append(_make_example(v, "Example — A trend the bot caught well"))
        late = [v for v in verdicts if v.verdict.startswith("late_entry") and v.captured_pnl_pct and v.captured_pnl_pct > 0]
        if late:
            v = max(late, key=lambda x: (x.buy_lateness_bars or 0))
            examples.append(_make_example(v, "Example — Profitable but entered late"))
    if missed:
        biggest = max(missed, key=lambda t: t.gain_pct)
        examples.append({
            "title": "Example — A trend the bot missed entirely",
            "narrative": (
                f"On {biggest.start_ts.isoformat()}, {biggest.symbol} moved from "
                f"{biggest.start_price:.4f} to {biggest.end_price:.4f} over "
                f"{biggest.duration_bars} bars ({biggest.gain_pct:.2f}% gain). "
                f"No BUY signal was emitted in the [{biggest.start_ts.isoformat()}, "
                f"{biggest.end_ts.isoformat()}] window. Investigate the indicator "
                f"snapshot at {biggest.start_ts.isoformat()} to see which filter "
                f"was blocking entry."
            ),
        })
    return examples


def _make_example(v: TradeVerdict, title: str) -> dict:
    t = v.matched_trend
    bl = v.buy_lateness_bars or 0
    blpct = v.buy_lateness_pct_of_move or 0
    cr = (v.capture_ratio or 0) * 100
    narrative = (
        f"On {t.start_ts.isoformat()}, {v.symbol} moved from {t.start_price:.4f} to "
        f"{t.end_price:.4f} over {t.duration_bars} bars ({t.gain_pct:.2f}% gain). "
        f"The bot bought at {v.buy_event.price:.4f} ({bl} bars after the swing low, "
        f"{blpct:.1f}% of the move already gone)"
    )
    if v.sell_event:
        slpct = v.sell_lateness_pct_of_move or 0
        narrative += (
            f" and sold at {v.sell_event.price:.4f}. "
            f"Net captured {cr:.0f}% of available profit "
            f"(sell lateness: {slpct:+.1f}% of move)."
        )
    else:
        narrative += " and never sold within the trend window."
    return {"title": title, "narrative": narrative}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    s = report["summary"]
    cfg = report["config"]
    lines: list[str] = []
    sym = ", ".join(cfg["symbols"])
    lines.append(f"# Signal Efficiency Report — {sym}, "
                 f"{cfg['window_start']} → {cfg['window_end']}")
    lines.append("")

    # Executive summary
    cr = s.get("median_capture_ratio")
    bl = s.get("median_buy_lateness_pct_of_move")
    miss = s.get("miss_rate")
    alpha = s.get("alpha_vs_buy_and_hold_pct")

    cr_str = f"{cr*100:.0f}%" if cr is not None else "n/a"
    bl_str = f"{bl:.1f}%" if bl is not None else "n/a"
    miss_str = f"{miss*100:.0f}%" if miss is not None else "n/a"
    alpha_str = f"{alpha:+.2f}%" if alpha is not None else "n/a"

    verdict_word = "underperforming"
    if cr is not None and cr >= 0.6 and (alpha or 0) > 0:
        verdict_word = "performing well"
    elif cr is not None and cr >= 0.4:
        verdict_word = "marginally profitable"

    lines.append("## Executive summary")
    lines.append(
        f"Across the window, the bot caught {s['trends_caught_with_buy']} of "
        f"{s['total_trends_in_period']} sustainable uptrends (miss rate {miss_str}), "
        f"captured a median {cr_str} of available profit per trade, and entered "
        f"a median {bl_str} into each move. Alpha vs. buy-and-hold: {alpha_str}. "
        f"Overall the bot is **{verdict_word}**."
    )
    lines.append("")

    # Headline metrics
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value | What good looks like |")
    lines.append("|---|---|---|")
    lines.append(f"| Median BUY lateness | {bl_str} of move | < 25% |")
    lines.append(f"| Median capture ratio | {cr_str} | > 60% |")
    lines.append(f"| Miss rate | {miss_str} | < 50% |")
    lines.append(f"| False-positive rate | "
                 f"{s['false_positive_rate']*100:.0f}% | < 30% |")
    lines.append(f"| Alpha vs buy-and-hold | {alpha_str} | > 0 |")
    lines.append(f"| Win rate | "
                 f"{(s['win_rate'] or 0)*100:.0f}% | > 55% |")
    pf = s.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf else "n/a"
    lines.append(f"| Profit factor | {pf_str} | > 1.5 |")
    lines.append("")

    # Wins / losses
    lines.append("## What the bot did well")
    well = []
    if cr is not None and cr >= 0.6:
        well.append(f"Captured a median {cr_str} of available profit — solid efficiency once it enters.")
    if (s.get("win_rate") or 0) >= 0.6:
        well.append(f"Win rate of {(s['win_rate'] or 0)*100:.0f}% — most trades are profitable.")
    if (alpha or 0) > 0:
        well.append(f"Beat buy-and-hold by {alpha_str}.")
    if not well:
        well.append("Nothing notable in this window.")
    for w in well:
        lines.append(f"- {w}")
    lines.append("")

    lines.append("## What the bot did wrong")
    if report["patterns"]:
        for p in sorted(report["patterns"], key=lambda x: -abs(x.get("estimated_impact_pct") or 0)):
            lines.append(f"- **{p['pattern_id']}** — {p['evidence']}")
    else:
        lines.append("- No systemic issues detected in this window.")
    lines.append("")

    # Coaching
    lines.append("## Coaching examples")
    for ex in report.get("coaching_examples", []):
        lines.append(f"### {ex['title']}")
        lines.append(ex['narrative'])
        lines.append("")

    # Recommendations
    lines.append("## Recommended adjustments")
    if report["patterns"]:
        for i, p in enumerate(report["patterns"], 1):
            lines.append(f"{i}. {p['recommendation']}")
    else:
        lines.append("No changes recommended for this window.")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append(
        f"Trends were labeled with a ZigZag-style swing detector configured for "
        f"`swing_threshold_pct={cfg['swing_threshold_pct']}`, "
        f"`max_intratrend_drawdown_pct={cfg['max_intratrend_drawdown_pct']}`, "
        f"`min_trend_duration_bars={cfg['min_trend_duration_bars']}`. "
        f"Fees of {cfg['fee_pct']}% per side were applied to realized P&L. "
        f"Re-run with different thresholds to test sensitivity. See "
        f"`references/trend_definition.md` for the full algorithm."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate claude_crypto_bot signals.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--events-file", type=Path, default=None,
                   help="Path to bot_events.jsonl (default: <project-root>/bot_events.jsonl)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to evaluate (default: all in events)")
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--swing-threshold-pct", type=float, default=3.0)
    p.add_argument("--max-intratrend-drawdown-pct", type=float, default=1.5)
    p.add_argument("--min-trend-duration-bars", type=int, default=3)
    p.add_argument("--fee-pct", type=float, default=0.075)
    p.add_argument("--output-dir", type=Path, default=Path("./evaluation_output"))
    p.add_argument("--append-to-rl-memory", action="store_true")
    p.add_argument("--rl-memory-path", type=Path, default=None)
    p.add_argument("--schema-map", type=str, default=None,
                   help="JSON dict overriding the field-alias map.")
    args = p.parse_args()

    aliases = dict(DEFAULT_FIELD_ALIASES)
    if args.schema_map:
        override = json.loads(args.schema_map)
        for k, v in override.items():
            aliases[k] = v if isinstance(v, list) else [v]

    events_path = args.events_file or (args.project_root / "bot_events.jsonl")
    if not events_path.exists():
        sys.stderr.write(f"events file not found: {events_path}\n")
        return 2

    events = load_events(events_path, aliases)
    if not events:
        sys.stderr.write("no events parsed; check schema with --schema-map\n")
        return 3

    # filter by window
    end = max(e.ts for e in events)
    start = end - pd.Timedelta(days=args.window_days)
    events = [e for e in events if start <= e.ts <= end]

    symbols = args.symbols or sorted({e.symbol for e in events})

    bars_by_symbol: dict[str, list[Bar]] = {}
    all_trends: list[Trend] = []
    for sym in symbols:
        bars = load_price_history(args.project_root, sym, args.timeframe, start, end)
        bars_by_symbol[sym] = bars
        trends = detect_trends(
            bars,
            swing_threshold_pct=args.swing_threshold_pct,
            max_intratrend_drawdown_pct=args.max_intratrend_drawdown_pct,
            min_trend_duration_bars=args.min_trend_duration_bars,
            symbol=sym,
        )
        all_trends.extend(trends)

    # split events by symbol, match each
    all_verdicts: list[TradeVerdict] = []
    all_missed: list[Trend] = []
    all_false_positives: list[Event] = []
    for sym in symbols:
        sym_events = [e for e in events if e.symbol == sym]
        sym_trends = [t for t in all_trends if t.symbol == sym]
        sym_bars = bars_by_symbol[sym]
        verdicts, missed, fps = match_signals_to_trends(
            sym_events, sym_trends, sym_bars,
            fee_pct=args.fee_pct,
            swing_threshold_pct=args.swing_threshold_pct,
            max_intratrend_drawdown_pct=args.max_intratrend_drawdown_pct,
        )
        all_verdicts.extend(verdicts)
        all_missed.extend(missed)
        all_false_positives.extend(fps)

    total_buys = sum(1 for e in events if e.action == "BUY")
    total_sells = sum(1 for e in events if e.action == "SELL")
    summary = build_summary(all_verdicts, all_missed, all_false_positives,
                            total_buys, total_sells, bars_by_symbol)
    patterns = detect_patterns(all_verdicts, summary)
    examples = build_coaching_examples(all_verdicts, all_missed)

    run_id = (f"eval_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
              f"_{'-'.join(symbols)}_{args.timeframe}")
    report = {
        "evaluation_run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "symbols": symbols,
            "timeframe": args.timeframe,
            "swing_threshold_pct": args.swing_threshold_pct,
            "max_intratrend_drawdown_pct": args.max_intratrend_drawdown_pct,
            "min_trend_duration_bars": args.min_trend_duration_bars,
            "fee_pct": args.fee_pct,
            "labeler_version": "swing_v1",
        },
        "summary": summary,
        "trade_verdicts": [v.asdict() for v in all_verdicts],
        "missed_opportunities": [_trend_to_dict(t) | {"symbol": t.symbol} for t in all_missed],
        "false_positives": [
            {"buy_signal_ts": ev.ts.isoformat(), "symbol": ev.symbol,
             "buy_price": ev.price, "confidence": ev.confidence}
            for ev in all_false_positives
        ],
        "patterns": patterns,
        "coaching_examples": examples,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )

    # RL memory append
    if args.append_to_rl_memory:
        rl_path = args.rl_memory_path or (args.project_root / "rl_memory.jsonl")
        record = {
            "type": "evaluation_feedback",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "summary": {
                "miss_rate": summary["miss_rate"],
                "false_positive_rate": summary["false_positive_rate"],
                "median_buy_lateness_bars": summary["median_buy_lateness_bars"],
                "median_capture_ratio": summary["median_capture_ratio"],
                "alpha_vs_buy_and_hold_pct": summary["alpha_vs_buy_and_hold_pct"],
            },
            "top_pattern": patterns[0]["pattern_id"] if patterns else None,
            "top_recommendation": patterns[0]["recommendation"] if patterns else None,
            "report_path": str((args.output_dir / "report.json").resolve()),
        }
        with rl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # one-screen stdout summary
    print(f"=== {run_id} ===")
    print(f"Window: {start.date()} → {end.date()}  Symbols: {', '.join(symbols)}  TF: {args.timeframe}")
    print(f"Trends in period: {summary['total_trends_in_period']}  "
          f"Caught: {summary['trends_caught_with_buy']}  Missed: {summary['trends_missed']}  "
          f"(miss rate: {summary['miss_rate']*100:.0f}%)")
    print(f"BUY signals: {total_buys}  False positives: {summary['false_positive_buys']} "
          f"({summary['false_positive_rate']*100:.0f}%)")
    print(f"Median BUY lateness: "
          f"{summary['median_buy_lateness_bars']} bars  "
          f"({(summary['median_buy_lateness_pct_of_move'] or 0):.1f}% of move)")
    print(f"Median capture ratio: "
          f"{(summary['median_capture_ratio'] or 0)*100:.1f}%")
    print(f"Realized P&L: {summary['total_realized_pnl_pct']:.2f}%  "
          f"Buy-and-hold: {summary['buy_and_hold_pnl_pct']:.2f}%  "
          f"Alpha: {summary['alpha_vs_buy_and_hold_pct']:+.2f}%")
    print(f"Wrote: {args.output_dir / 'report.json'}")
    print(f"Wrote: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
