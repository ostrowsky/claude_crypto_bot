"""Leader-of-the-day INFO alert: a coin that has held a top-3 day lead for 2 hours.

WHY (operator, 2026-09-27)
The goal is 1-3 day-leaders, not a stream of buy signals. The leader-mode
research (docs/specs/features/leader-mode-spec.md, 457 days x 101 coins) found
that a coin holding rank <= 3 by return since the UTC open, at >= +7.5%, for 2
consecutive hours finishes the day in the watchlist top-3 ~70% of the time and
in the global top-20 ~72-77% -- but entering at that moment loses on average
(-0.7..-1.2% per trade). With this module's own semantics (first qualifying bar
per coin, max 3 a day, 2025-07..2026-09): 850 alerts, 69% end the day in the
watchlist top-3 (train 68% / test 70%), 77% in the global top-20; from the alert
price the median further rise is +2.7% and the median close of the day -0.8%.
So this is information for the operator, never a buy signal: it says WHICH coin
leads, not WHEN to enter.

RULE (identical to _backtest_leader_mode_v2.py, variant "L-1 persist 2h")
  ret_k  = close of 15m bar k / open of the 00:00 UTC bar - 1, closed bars only
  rank_k = rank of ret_k among watchlist coins that have bar k (1 = best)
  alert when rank_k <= LEADER_ALERT_RANK_MAX and ret_k >= LEADER_ALERT_MIN_RET
  on each of the last LEADER_ALERT_HOLD_BARS closed bars of the current UTC day;
  at most once per coin per day, at most LEADER_ALERT_MAX_PER_DAY a day.

Every alert is logged (bot_events.jsonl, event "leader_alert") so the live hit
rate can be read against the backtest's ~70%. Fails closed: any error means no
alert, never a crash of the monitor loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

log = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parent.parent / ".runtime" / "leader_alerts.json"
BAR_MS = 15 * 60 * 1000

# Ranking of the whole watchlist at the latest closed 15m bar of the UTC day,
# refreshed by refresh() on every new bar. leader_exit.py reads it.
_LATEST: Dict[str, object] = {"bar_ts": None, "ranks": {}}


def latest_ranks() -> Optional[Tuple[int, Dict[str, Tuple[int, float]]]]:
    """(bar_ts, {sym: (rank, return since the UTC open)}) or None before the first refresh."""
    if _LATEST["bar_ts"] is None:
        return None
    return int(_LATEST["bar_ts"]), dict(_LATEST["ranks"])


def rank_last_bar(day: Dict[str, Tuple[float, Dict[int, float]]]) -> Tuple[Optional[int], Dict[str, Tuple[int, float]]]:
    """Rank every coin by return since the UTC open at the latest closed bar any coin has."""
    bars = {t for _, closes in day.values() for t in closes}
    if not bars:
        return None, {}
    last = max(bars)
    rets = {s: closes[last] / op - 1 for s, (op, closes) in day.items() if op > 0 and last in closes}
    order = sorted(rets, key=lambda s: -rets[s])
    return last, {s: (k + 1, rets[s]) for k, s in enumerate(order)}


def enabled() -> bool:
    return bool(getattr(config, "LEADER_ALERT_ENABLED", False))


def params() -> Tuple[int, float, int, int]:
    return (int(getattr(config, "LEADER_ALERT_RANK_MAX", 3)),
            float(getattr(config, "LEADER_ALERT_MIN_RET", 0.075)),
            int(getattr(config, "LEADER_ALERT_HOLD_BARS", 8)),
            int(getattr(config, "LEADER_ALERT_MAX_PER_DAY", 3)))


def find_leaders(day: Dict[str, Tuple[float, Dict[int, float]]], rank_max: int, min_ret: float,
                 hold_bars: int) -> List[dict]:
    """day: sym -> (open of the 00:00 bar, {bar_open_ms: close}) for CLOSED bars of today.

    Returns the coins meeting the rule at the latest closed bar, best rank first.
    """
    bars = sorted({t for _, closes in day.values() for t in closes})
    if len(bars) < hold_bars:
        return []
    ranks: Dict[int, Dict[str, Tuple[int, float]]] = {}
    for t in bars[-hold_bars:]:
        rets = {s: closes[t] / op - 1 for s, (op, closes) in day.items() if op > 0 and t in closes}
        order = sorted(rets, key=lambda s: -rets[s])
        ranks[t] = {s: (k + 1, rets[s]) for k, s in enumerate(order)}
    out = []
    for s in day:
        seq = [ranks[t].get(s) for t in bars[-hold_bars:]]
        if all(x is not None and x[0] <= rank_max and x[1] >= min_ret for x in seq):
            out.append({"sym": s, "rank": seq[-1][0], "ret": seq[-1][1], "bar_ts": bars[-1],
                        "price": day[s][1][bars[-1]]})
    return sorted(out, key=lambda x: x["rank"])


def _load_state(today: str) -> dict:
    try:
        st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if st.get("day") == today:
            return st
    except Exception:
        pass
    return {"day": today, "alerted": {}}


def _save_state(st: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("leader_alert: state save failed: %s", e)


async def _fetch_today(session, sym: str, day_ms: int, now_ms: int, sem) -> Optional[Tuple[float, Dict[int, float]]]:
    url = f"{config.BINANCE_REST}/api/v3/klines"
    import aiohttp
    async with sem:
        try:
            async with session.get(url, params={"symbol": sym, "interval": "15m", "startTime": day_ms, "limit": 100},
                                   timeout=aiohttp.ClientTimeout(total=20)) as r:
                r.raise_for_status()
                js = await r.json()
        except Exception:
            return None
    if not isinstance(js, list) or not js or int(js[0][0]) != day_ms:
        return None                      # no 00:00 bar -> no day open -> not ranked
    closes = {int(k[0]): float(k[4]) for k in js if int(k[6]) < now_ms}
    return float(js[0][1]), closes


def format_message(x: dict, hold_bars: int) -> str:
    return ("ℹ️ *Лидер дня: {sym}*\n"
            "Держится в топ-{rmax} роста вотчлиста {hours:g} ч подряд: *{ret:+.1f}%* от открытия суток (00:00 UTC), "
            "сейчас место №{rank}, цена {price:g}.\n"
            "Это информация, *не сигнал на покупку*. По истории (450 дней) 69% таких монет заканчивают "
            "день в топ-3 вотчлиста; от цены алерта в медиане ещё +2.7% вверх, а к закрытию суток −0.8%.").format(
        sym=x["sym"], rmax=params()[0], hours=hold_bars / 4, ret=100 * x["ret"], rank=x["rank"], price=x["price"])


def ranking_needed() -> bool:
    try:
        import leader_exit
        return enabled() or leader_exit.enabled()
    except Exception:
        return enabled()


async def run_once(session, send, watchlist: List[str], now: Optional[datetime] = None) -> List[dict]:
    """Call on each new closed 15m bar: refresh the watchlist ranking (read by
    leader_exit.py) and send / log any new leader alerts."""
    if not ranking_needed():
        return []
    rank_max, min_ret, hold_bars, max_day = params()
    now = now or datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    day_ms = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    today = now.strftime("%Y-%m-%d")
    sem = asyncio.Semaphore(int(getattr(config, "LEADER_ALERT_FETCH_CONCURRENCY", 8)))
    got = await asyncio.gather(*(_fetch_today(session, s, day_ms, now_ms, sem) for s in watchlist))
    day = {s: g for s, g in zip(watchlist, got) if g is not None and g[1]}
    if len(day) < 20:
        log.info("leader_alert: only %d coins with today's bars, skipped", len(day))
        return []
    bar, ranks = rank_last_bar(day)
    if bar is not None:
        _LATEST["bar_ts"], _LATEST["ranks"] = bar, ranks
    if not enabled():
        return []
    st = _load_state(today)
    if len(st["alerted"]) >= max_day:
        return []
    sent = []
    for x in find_leaders(day, rank_max, min_ret, hold_bars):
        if x["sym"] in st["alerted"] or len(st["alerted"]) >= max_day:
            continue
        try:
            await send(format_message(x, hold_bars))
        except Exception as e:
            log.warning("leader_alert: send failed for %s: %s", x["sym"], e)
            continue
        st["alerted"][x["sym"]] = {"bar_ts": x["bar_ts"], "rank": x["rank"], "ret": round(x["ret"], 5), "price": x["price"]}
        _save_state(st)
        try:
            import botlog
            botlog.log_leader_alert(x["sym"], x["price"], x["rank"], x["ret"], hold_bars, x["bar_ts"], len(day))
        except Exception as e:
            log.warning("leader_alert: botlog failed: %s", e)
        log.info("LEADER ALERT %s rank %d ret %+.2f%% (%d coins ranked)", x["sym"], x["rank"], 100 * x["ret"], len(day))
        sent.append(x)
    return sent
