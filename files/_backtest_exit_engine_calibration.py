"""P0 hypothesis П-1: calibrate the exit counterfactual against the live trail.

The engine used on 2026-09-09..25 (_backtest_weak_exit_above_breakeven.replay)
differs from monitor.py in three systematic ways:
  anchor    it trails the running HIGH with a width fixed from ATR at the exit
            bar; live trails the CLOSE: stop = max(stop, close - max(k*ATR_now,
            floor*close)), recomputed every closed bar
  fill      it exits intrabar AT the stop when low <= stop; live exits at the
            bar CLOSE once close < stop -- the engine gets a better price on
            every breach
  extras    live also tightens (profit-lock k 1.4/1.2 + floor, trend-hold weak
            k 1.4, EXIT_RL TIGHTEN k*0.75); the engine has none

This file implements the live rule WITHOUT the tighteners (`live_trail`) and
checks it where it must agree with reality: exits that the bot actually took on
the ATR trail. Then it re-runs the exit-class comparison with the calibrated
engine, split by day-leader rank, so the audit's exit hypotheses (X-1..X-4) can
be judged on a trustworthy counterfactual.
"""
import collections
import io
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import indicators as I  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _backtest_weak_exit_above_breakeven as W  # noqa: E402

FLOORS = {"impulse_speed": "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", "strong_trend": "TRAIL_MIN_BUFFER_PCT_STRONG_TREND",
          "impulse": "TRAIL_MIN_BUFFER_PCT_IMPULSE", "trend": "TRAIL_MIN_BUFFER_PCT_TREND",
          "alignment": "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", "retest": "TRAIL_MIN_BUFFER_PCT_RETEST",
          "breakout": "TRAIL_MIN_BUFFER_PCT_BREAKOUT"}


def floor_pct(mode):
    """monitor._trail_min_buffer_pct, as a fraction of price."""
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    return float(getattr(cfg, FLOORS.get(mode, "TRAIL_MIN_BUFFER_PCT_DEFAULT"), 0.0))


def buffer(price, k, atr, mode):
    """monitor._compute_trail_buffer."""
    a = k * atr if atr > 0 else 0.0
    f = floor_pct(mode) * price if price > 0 else 0.0
    return max(a, f)


def live_trail(c, atr, i_entry, entry_price, k, mode, i_from, max_bars=96, stop0=None):
    """The live ATR-trail rule, tighteners excluded.

    Stop starts at entry - buffer(entry ATR) and ratchets up with close - buffer
    on every closed bar; exit on the first bar whose CLOSE is below the stop, at
    that close. Bars before i_from only move the stop (the position was still
    open live). Returns (exit_index, pnl%) or None.
    """
    if not (0 <= i_entry < len(c)) or entry_price <= 0:
        return None
    a0 = atr[i_entry] if np.isfinite(atr[i_entry]) else 0.0
    stop = stop0 if stop0 is not None else entry_price - buffer(entry_price, k, a0, mode)
    last = min(len(c) - 1, i_from + max_bars)
    for j in range(i_entry + 1, last + 1):
        cj = c[j]
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, cj - buffer(cj, k, aj, mode))
        if j >= i_from and stop > 0 and cj < stop:
            return j, (cj / entry_price - 1) * 100
    if last <= i_from:
        return None
    return last, (c[last] / entry_price - 1) * 100


def cls(reason):
    r = reason or ""
    for key, name in (("RSI перекуплен", "RSI overbought"), ("RSI дивергенция", "WEAK RSI divergence"),
                      ("объёмное истощение", "WEAK volume exhaustion"), ("EMA-веер", "WEAK EMA fan"),
                      ("quality recheck failed - price_edge", "WEAK recheck price_edge"),
                      ("quality recheck failed - MACD", "WEAK recheck MACD"),
                      ("micro-weakness", "micro-weakness after profit-lock"), ("ATR-трейл", "ATR trail"),
                      ("первое закрытие ниже EMA", "1st close < EMA20 (loss)"),
                      ("закрытия подряд ниже EMA", "2 closes < EMA20"), ("разворачивается вниз", "EMA20 turns down"),
                      ("время", "time max hold"), ("ниже EMA", "price < EMA20")):
        if key in r:
            return name
    return re.sub(r"[\d.]+", "#", r)[:34]


wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
wl = wl if isinstance(wl, list) else wl.get("symbols", wl)
S = {}
ret = collections.defaultdict(dict)
for sym in wl:
    b = TD.bars_15m(sym)
    if not b:
        continue
    h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
    atr = I._atr(h, l, c, cfg.ATR_PERIOD)
    S[sym] = (b, {x[0]: i for i, x in enumerate(b)}, c, atr)
    day, op = None, None
    for x in b:
        if x[0].date() != day:
            day, op = x[0].date(), x[1]
        if x[0] >= datetime(2026, 3, 1, tzinfo=timezone.utc):
            ret[x[0]][sym] = (x[4] / op - 1) * 100

exits = []
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if b'"exit"' not in raw:
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("event") != "exit" or e.get("tf") != "15m" or e.get("sym") not in S:
            continue
        if not all(isinstance(e.get(k), (int, float)) for k in ("entry_price", "exit_price", "pnl_pct", "bars_held")):
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        # the live decision is taken on the last CLOSED bar at poll time
        bk = d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)
        b, idx, c, atr = S[e["sym"]]
        i = idx.get(bk)
        if i is None:
            continue
        i_close = i - 1                       # last closed bar when the exit was sent
        i_entry = i_close - int(e["bars_held"])
        if i_entry < 20:
            continue
        e.update(_i=i_close, _ie=i_entry, _cls=cls(e.get("reason")), _m=d.strftime("%Y-%m"), _bk=bk)
        exits.append(e)
print("15m exits: %d" % len(exits))

# ---------------- 1. calibration on ATR-trail exits ----------------
atr_ex = [e for e in exits if e["_cls"] == "ATR trail"]
match, off, dpnl_new, dpnl_old = 0, [], [], []
for e in atr_ex:
    b, idx, c, atr = S[e["sym"]]
    r = live_trail(c, atr, e["_ie"], float(e["entry_price"]), float(e.get("trail_k") or 2.0), e.get("mode"), e["_ie"] + 1)
    if r is None:
        continue
    j, p = r
    off.append(j - e["_i"])
    match += abs(j - e["_i"]) <= 1
    dpnl_new.append(p - float(e["pnl_pct"]))
    old = W.replay(b, e["_i"] + 1, e["_ie"], float(e["entry_price"]), float(e.get("trail_k") or 2.0), e.get("mode"), cfg, 96)
    if old is not None:
        dpnl_old.append(old - float(e["pnl_pct"]))
off.sort(); dn = sorted(dpnl_new)
print("\n=== 1. CALIBRATION on %d exits the bot took on the ATR trail ===" % len(atr_ex))
print("  live-rule replay from ENTRY: exit bar within +-1 of the real one: %d of %d (%.0f%%)" % (match, len(off), 100.0 * match / max(1, len(off))))
print("  exit-bar offset (replay - real): p10 %+d  p25 %+d  median %+d  p75 %+d  p90 %+d" % (
    off[len(off) // 10], off[len(off) // 4], off[len(off) // 2], off[3 * len(off) // 4], off[9 * len(off) // 10]))
print("  pnl replay - real: mean %+.3f%%  median %+.3f%%" % (sum(dn) / len(dn), dn[len(dn) // 2]))
print("  old engine (high-anchored, intrabar fill), same exits: mean %+.3f%%" % (sum(dpnl_old) / max(1, len(dpnl_old))))
early = sum(1 for x in off if x < -1); late = sum(1 for x in off if x > 1)
print("  replay exits EARLIER than real: %d, LATER: %d  (later = the live stop was tighter than the plain rule)" % (early, late))

# ---------------- 2. exit classes vs the calibrated trail ----------------
rnd = random.Random(4)


def ci(d):
    if len(d) < 20:
        return float("nan"), float("nan")
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(1500))
    return bs[37], bs[-38]


rows = []
for e in exits:
    b, idx, c, atr = S[e["sym"]]
    # continue the plain live trail from the entry; the exit is ignored, so the
    # counterfactual position is whatever the plain trail would have done after it
    r = live_trail(c, atr, e["_ie"], float(e["entry_price"]), float(e.get("trail_k") or 2.0), e.get("mode"), e["_i"] + 1)
    if r is None:
        continue
    cs = ret.get(e["_bk"], {})
    me = cs.get(e["sym"])
    rank = (1 + sum(1 for v in cs.values() if v > me)) if me is not None and len(cs) > 50 else None
    rows.append({"cls": e["_cls"], "pnl": float(e["pnl_pct"]), "cf": r[1], "rank": rank, "m": e["_m"]})


def line(name, v):
    if len(v) < 20:
        return
    d = [x["cf"] - x["pnl"] for x in v]
    lo, hi = ci(d)
    print("  %-34s n=%5d  trail-actual mean %+6.2f%% [%+.2f, %+.2f]  trail better %3.0f%%%s" % (
        name, len(v), sum(d) / len(d), lo, hi, 100.0 * sum(x > 0 for x in d) / len(d),
        "   <-- excludes 0" if (lo > 0 or hi < 0) else ""))


print("\n=== 2. EVERY EXIT CLASS vs the CALIBRATED plain trail (%d exits) ===" % len(rows))
by = collections.defaultdict(list)
for x in rows:
    by[x["cls"]].append(x)
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    line(k, v)
print("\n  split by day-leader rank at the exit bar (return since UTC open, watchlist):")
for lo_r, hi_r, name in ((1, 5, "rank 1-5"), (6, 20, "rank 6-20"), (21, 999, "rank 21+")):
    v = [x for x in rows if x["rank"] is not None and lo_r <= x["rank"] <= hi_r]
    line("ALL CLASSES " + name, v)
    for k in ("WEAK RSI divergence", "ATR trail", "RSI overbought", "time max hold", "1st close < EMA20 (loss)"):
        line("   " + k, [x for x in v if x["cls"] == k])
lead = collections.defaultdict(list); rest = collections.defaultdict(list)
for x in rows:
    if x["rank"] is None:
        continue
    (lead if x["rank"] <= 5 else rest)[x["m"]].append(x["cf"] - x["pnl"])
print("\n  by month, leaders (1-5) minus the rest (21+ and 6-20):")
pos = 0
for m in sorted(lead):
    if rest.get(m):
        dlt = sum(lead[m]) / len(lead[m]) - sum(rest[m]) / len(rest[m])
        pos += dlt > 0
        print("    %s  leaders %+6.2f%% [%3d]   rest %+6.2f%% [%4d]   diff %+6.2f" % (
            m, sum(lead[m]) / len(lead[m]), len(lead[m]), sum(rest[m]) / len(rest[m]), len(rest[m]), dlt))
print("  months with leaders > rest: %d of %d" % (pos, len(lead)))
json.dump(rows, io.open(FILES.parent / ".runtime/backtests/p0_exit_rows.json", "w", encoding="utf-8"))
