"""Leader exit mode (X-7b): hold a position whose coin leads the day on a wide trail.

WHY (operator, 2026-09-27: "catch 1-3 leaders and hold them as long as needed")
The bot buys 57% of rockets before +10% but keeps ~8% of the move: soft exits
(WEAK, RSI overbought, micro-weakness) and a tight ATR trail close it while the
coin keeps running (+8.4% median after the exit). Backtest X-7b on every real
15m trade since 2026-03-01 (docs/specs/features/leader-exit-x6-spec.md):
rocket-days +1.18 pp per trade [+0.75, +1.68], positive 7/7 months; winner-days
+1.07 pp; other trades -0.09 pp; all trades +0.026 pp [-0.030, +0.087].

RULE (identical to the backtest)
  switch  15m position; at a closed bar at or after the entry bar the coin is
          rank <= LEADER_EXIT_RANK_MAX (5 since X-9c; X-7b had 3) by return since
          the UTC open across the watchlist, at >= LEADER_EXIT_MIN_RET (+3%; X-7b
          +5%) -> leader mode: the stop
          is RESET to close - max(k*ATR, LEADER_EXIT_FLOOR_PCT*close) (it may go
          down: the point is to give the leader room) and only ratchets up after
  hold    every other exit (WEAK, RSI, EMA, micro-weakness, profit-lock, time,
          rotation, replacement) is skipped for the position
  exit    close < stop, or the coin is out of the day's top-LEADER_EXIT_LOST_RANK
          (10) at least LEADER_EXIT_LOST_MIN_BARS (4) bars after the switch, or
          LEADER_EXIT_MAX_HOLD_BARS (7 days) after the switch

Ranks come from leader_alert.latest_ranks() (recomputed on every closed 15m bar
for the whole watchlist). Stale or missing ranks never switch a position and
never trigger the lost-lead exit; the wide trail still applies.
"""
from __future__ import annotations

from typing import Optional, Tuple

import config

BAR_MS = 15 * 60 * 1000


def enabled() -> bool:
    return bool(getattr(config, "LEADER_EXIT_ENABLED", False))


def params() -> dict:
    return {
        "rank_max": int(getattr(config, "LEADER_EXIT_RANK_MAX", 3)),
        "min_ret": float(getattr(config, "LEADER_EXIT_MIN_RET", 0.05)),
        "floor": float(getattr(config, "LEADER_EXIT_FLOOR_PCT", 0.08)),
        "lost_rank": int(getattr(config, "LEADER_EXIT_LOST_RANK", 10)),
        "lost_min_bars": int(getattr(config, "LEADER_EXIT_LOST_MIN_BARS", 4)),
        "max_hold_bars": int(getattr(config, "LEADER_EXIT_MAX_HOLD_BARS", 96 * 7)),
        "tfs": tuple(getattr(config, "LEADER_EXIT_TF", ("15m",))),
    }


def wide_buffer(price: float, k: float, atr: float, floor: float) -> float:
    return max(k * atr if atr > 0 else 0.0, floor * price)


def _rank_of(sym: str, ranks, bar_ts: int, max_lag_bars: int = 1) -> Optional[Tuple[int, float]]:
    """(rank, ret) of `sym` if the ranking is fresh enough for the bar being judged."""
    if not ranks:
        return None
    rbar, table = ranks
    if rbar is None or rbar < bar_ts - max_lag_bars * BAR_MS:
        return None
    return table.get(sym)


def step(pos, *, sym: str, tf: str, close: float, atr: float, bar_ts: int, ranks) -> Tuple[str, str]:
    """One poll of an open position. Returns (action, reason):

    'none'    not a leader position -> the normal exit logic runs
    'switch'  just entered leader mode this bar (stop reset) -> skip other exits
    'hold'    in leader mode, still held -> skip other exits
    'exit'    in leader mode, close the position with `reason`
    """
    p = params()
    if not getattr(pos, "leader_mode", False):
        if tf not in p["tfs"]:
            return "none", ""
        rr = _rank_of(sym, ranks, bar_ts)
        if rr is None:
            return "none", ""
        rbar = ranks[0]
        if rbar < int(pos.entry_ts):
            return "none", ""
        rank, ret = rr
        if rank <= p["rank_max"] and ret >= p["min_ret"]:
            pos.leader_mode = True
            pos.leader_since_ts = int(bar_ts)
            pos.trail_stop = close - wide_buffer(close, float(pos.trail_k), atr, p["floor"])
            return "switch", "монета №%d дня, %+.1f%% от открытия" % (rank, 100 * ret)
        return "none", ""

    new = close - wide_buffer(close, float(pos.trail_k), atr, p["floor"])
    if new > pos.trail_stop:
        pos.trail_stop = new
    if pos.trail_stop > 0 and close < pos.trail_stop:
        return "exit", "режим лидера: широкий трейл пробит (стоп %.6g)" % pos.trail_stop
    held_bars = (int(bar_ts) - int(getattr(pos, "leader_since_ts", bar_ts))) // BAR_MS
    if held_bars >= p["max_hold_bars"]:
        return "exit", "режим лидера: 7 дней удержания"
    if held_bars >= p["lost_min_bars"]:
        rr = _rank_of(sym, ranks, bar_ts)
        if rr is not None and rr[0] > p["lost_rank"]:
            return "exit", "режим лидера: монета выпала из топ-%d дня (место №%d)" % (p["lost_rank"], rr[0])
    return "hold", ""


def switch_message(sym: str, tf: str, reason: str, stop: float, close: float) -> str:
    return ("🏁 *{sym}* `[{tf}]` — позиция переведена в *режим лидера*: {reason}.\n"
            "Держим широким стопом `{stop:.6g}` ({dist:+.1f}% от цены); выход — при пробое стопа "
            "или когда монета выпадет из топ-10 дня. Обычные сигналы «слабости» для неё отключены.").format(
        sym=sym, tf=tf, reason=reason, stop=stop, dist=(stop / close - 1) * 100 if close > 0 else 0.0)
