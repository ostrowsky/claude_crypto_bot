"""ZigZag-based sustainable-uptrend labeler.

Extracted from skills/signal-efficiency-evaluator/scripts/evaluate_signals.py
(detect_trends function) per signal-evaluator-integration-spec.md Phase B.

Why standalone module:
- The bot needs the same labeler for label_sustained_uptrend backfill
  (top_gainer_dataset.jsonl) and for EX1 metric refactor (Phase D).
- Skill keeps its copy intact; this module is the bot's internal version
  used by retraining and metric scripts.

Public API:
    detect_uptrends(bars, swing_pct=4.0, max_drawdown_pct=2.0,
                    min_duration_bars=4) -> list[UpTrend]

`bars` is anything with attributes/keys: ts, open, high, low, close, volume.
Accepts list of dicts, dataclass instances, or numpy structured arrays.

Spec: docs/specs/features/signal-evaluator-integration-spec.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple


class UpTrend(NamedTuple):
    """Single sustainable uptrend identified by ZigZag.

    Attributes:
        start_idx: index in source bars where swing-low occurred
        start_ts:  timestamp of swing-low (whatever ts type was passed)
        start_price: swing-low price (low of bar)
        end_idx: index of swing-high
        end_ts: timestamp of swing-high
        end_price: swing-high price (high of bar)
    """
    symbol: str
    start_idx: int
    start_ts: Any
    start_price: float
    end_idx: int
    end_ts: Any
    end_price: float

    @property
    def gain_pct(self) -> float:
        return (self.end_price - self.start_price) / self.start_price * 100

    @property
    def duration_bars(self) -> int:
        return self.end_idx - self.start_idx


def _bar_field(bar: Any, name: str) -> float:
    """Polymorphic accessor: dict-like, dataclass-like, or numpy."""
    if isinstance(bar, dict):
        return float(bar[name])
    return float(getattr(bar, name))


def _bar_ts(bar: Any) -> Any:
    if isinstance(bar, dict):
        return bar.get("ts") or bar.get("timestamp")
    return getattr(bar, "ts", None) or getattr(bar, "timestamp", None)


def detect_uptrends(
    bars: Iterable[Any],
    *,
    symbol: str = "",
    swing_pct: float = 4.0,
    max_drawdown_pct: float = 2.0,
    min_duration_bars: int = 4,
) -> list[UpTrend]:
    """Identify all sustainable uptrends in a price series via ZigZag.

    Algorithm:
      Walk forward; track current swing-low (when in 'down' state) or
      swing-high (when 'up'). Trend ends when counter-move from extreme
      reaches max_drawdown_pct. Emit (low, high) pair if the move was
      >= swing_pct and lasted >= min_duration_bars.

    Defaults are tuned for 15m/1h crypto majors:
      - 4% swing: skips noise, catches meaningful moves
      - 2% drawdown: roughly 2 ATR for typical pairs
      - 4 bar minimum: filters single-spike fakeouts

    For other timeframes:
      - 5m: swing 2.5%, drawdown 1.0%
      - 1h: swing 5-7%, drawdown 2.5%
      - 1d: swing 12%, drawdown 5%

    Args:
        bars: iterable of objects with ts/open/high/low/close attributes
              (dict, dataclass, namedtuple all work).
        symbol: optional label stamped on each UpTrend.
        swing_pct: minimum gain to register an uptrend.
        max_drawdown_pct: counter-move that breaks the trend.
        min_duration_bars: minimum length of an uptrend in bars.

    Returns:
        List of UpTrend records sorted by start_idx ascending.
    """
    bars_list = list(bars)
    if len(bars_list) < 2:
        return []

    direction = "up"
    swing_low_idx, swing_low_price = 0, _bar_field(bars_list[0], "low")
    swing_high_idx, swing_high_price = 0, _bar_field(bars_list[0], "high")
    trends: list[UpTrend] = []

    for i in range(1, len(bars_list)):
        bar = bars_list[i]
        bar_high = _bar_field(bar, "high")
        bar_low = _bar_field(bar, "low")

        if direction == "up":
            if bar_high > swing_high_price:
                swing_high_idx, swing_high_price = i, bar_high
                continue
            drawdown_pct = (swing_high_price - bar_low) / swing_high_price * 100
            if drawdown_pct >= max_drawdown_pct:
                gain_pct = (swing_high_price - swing_low_price) / swing_low_price * 100
                duration = swing_high_idx - swing_low_idx
                if gain_pct >= swing_pct and duration >= min_duration_bars:
                    trends.append(UpTrend(
                        symbol=symbol,
                        start_idx=swing_low_idx,
                        start_ts=_bar_ts(bars_list[swing_low_idx]),
                        start_price=swing_low_price,
                        end_idx=swing_high_idx,
                        end_ts=_bar_ts(bars_list[swing_high_idx]),
                        end_price=swing_high_price,
                    ))
                direction = "down"
                swing_low_idx, swing_low_price = i, bar_low
        else:
            if bar_low < swing_low_price:
                swing_low_idx, swing_low_price = i, bar_low
                continue
            rebound_pct = (bar_high - swing_low_price) / swing_low_price * 100
            if rebound_pct >= max_drawdown_pct:
                direction = "up"
                swing_high_idx, swing_high_price = i, bar_high

    # Open trend at end of window: emit if gain already exceeds threshold.
    # Live evaluation should grade the trend so far even if not yet completed.
    if direction == "up":
        gain_pct = (swing_high_price - swing_low_price) / swing_low_price * 100
        duration = swing_high_idx - swing_low_idx
        if gain_pct >= swing_pct and duration >= min_duration_bars:
            trends.append(UpTrend(
                symbol=symbol,
                start_idx=swing_low_idx,
                start_ts=_bar_ts(bars_list[swing_low_idx]),
                start_price=swing_low_price,
                end_idx=swing_high_idx,
                end_ts=_bar_ts(bars_list[swing_high_idx]),
                end_price=swing_high_price,
            ))
    return trends


def find_containing_trend(
    trends: list[UpTrend],
    entry_ts: Any,
    exit_ts: Any | None = None,
) -> UpTrend | None:
    """Return the UpTrend whose [start_ts, end_ts] interval best contains
    the trade. If exit_ts is None, find first trend covering entry_ts only.

    Used by EX1 refactor (Phase D): for each paired (entry, exit), find
    the matching ZigZag trend and use its gain_pct as `potential`.
    """
    candidates = []
    for t in trends:
        # entry must be within or just before the trend
        if entry_ts > t.end_ts:
            continue
        if exit_ts is not None and exit_ts < t.start_ts:
            continue
        candidates.append(t)
    if not candidates:
        return None
    # Prefer trend whose start is closest BEFORE entry (we entered into it)
    candidates.sort(key=lambda t: abs((entry_ts - t.start_ts).total_seconds()
                                       if hasattr(entry_ts, 'timestamp') else 0))
    return candidates[0]
