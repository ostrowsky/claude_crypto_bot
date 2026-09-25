"""Every exit class vs the ATR trail, split by whether the coin was a day-leader.

For every 15m exit (max period) the counterfactual is the same as in
_backtest_weak_exit_above_breakeven: keep the position and let the ATR trail
decide (trail_k from the trade, mode floor, stop tested before the peak absorbs
the bar). "Leader" is observable in real time: the coin's return since the UTC
open at the exit bar, ranked across the watchlist at that same bar.
"""
import collections
import io
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _backtest_weak_exit_above_breakeven as W  # noqa: E402

wl = json.load(io.open(FILES / "watchlist.json", encoding="utf-8"))
wl = wl if isinstance(wl, list) else wl.get("symbols", wl)

# cross-section: return since UTC open for every watchlist coin at every 15m bar
ret = collections.defaultdict(dict)      # ts -> {sym: ret%}
idx_of = {}
for sym in wl:
    b = TD.bars_15m(sym)
    if not b:
        continue
    idx_of[sym] = (b, {x[0]: i for i, x in enumerate(b)})
    day_open = None
    cur_day = None
    for x in b:
        d = x[0].date()
        if d != cur_day:
            cur_day, day_open = d, x[1]
        if day_open and x[0] >= datetime(2026, 3, 1, tzinfo=timezone.utc):
            ret[x[0]][sym] = (x[4] / day_open - 1) * 100
print("cross-section built: %d symbols, %d bar timestamps" % (len(idx_of), len(ret)))


def cls(reason):
    r = reason or ""
    for k, name in (("RSI перекуплен", "RSI overbought"), ("RSI дивергенция", "WEAK RSI divergence"),
                    ("объёмное истощение", "WEAK volume exhaustion"), ("EMA-веер", "WEAK EMA fan"),
                    ("quality recheck failed - price_edge", "WEAK recheck price_edge"),
                    ("quality recheck failed - MACD", "WEAK recheck MACD"),
                    ("micro-weakness", "micro-weakness after profit-lock"),
                    ("ATR-трейл", "ATR trail"), ("первое закрытие ниже EMA", "1st close < EMA20 (loss)"),
                    ("закрытия подряд ниже EMA", "2 closes < EMA20"), ("разворачивается вниз", "EMA20 turns down"),
                    ("время", "time max hold"), ("Цена ниже EMA", "price < EMA20"), ("ниже EMA", "price < EMA20")):
        if k in r:
            return name
    return re.sub(r"[\d.]+", "#", r)[:34]


rows = []
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if b'"exit"' not in raw:
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("event") != "exit" or e.get("tf") != "15m" or e.get("sym") not in idx_of:
            continue
        if not all(isinstance(e.get(k), (int, float)) for k in ("entry_price", "pnl_pct", "bars_held")):
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        bk = d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)
        bars, idx = idx_of[e["sym"]]
        i = idx.get(bk)
        if i is None:
            continue
        cf = W.replay(bars, i, i - int(e["bars_held"]), float(e["entry_price"]),
                      float(e.get("trail_k") or 2.0), e.get("mode"), cfg, 96)
        if cf is None:
            continue
        cs = ret.get(bk, {})
        me = cs.get(e["sym"])
        rank = (1 + sum(1 for v in cs.values() if v > me)) if me is not None and len(cs) > 50 else None
        rows.append({"cls": cls(e.get("reason")), "pnl": float(e["pnl_pct"]), "cf": cf, "rank": rank,
                     "ret_open": me, "m": d.strftime("%Y-%m")})
print("15m exits replayed: %d" % len(rows))

rnd = random.Random(4)


def ci(d):
    if len(d) < 20:
        return float("nan"), float("nan")
    b = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(1500))
    return b[37], b[-38]


def line(name, v):
    if not v:
        return
    d = [x["cf"] - x["pnl"] for x in v]
    lo, hi = ci(d)
    p = sorted(x["pnl"] for x in v)
    print("  %-34s n=%5d  actual med %+6.2f%%  trail-actual mean %+6.2f%% [%+.2f, %+.2f]  trail better %3.0f%%%s" % (
        name, len(v), p[len(p) // 2], sum(d) / len(d), lo, hi, 100.0 * sum(x > 0 for x in d) / len(d),
        "   <-- CI excludes 0" if (lo > 0 or hi < 0) else ""))


print("\n=== ALL 15m EXITS BY CLASS: keep the exit vs hand it to the ATR trail ===")
by = collections.defaultdict(list)
for r in rows:
    by[r["cls"]].append(r)
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    if len(v) >= 20:
        line(k, v)

print("\n=== THE SAME, SPLIT BY DAY-LEADER RANK AT THE EXIT BAR (return since UTC open, watchlist) ===")
for lo_r, hi_r, name in ((1, 5, "rank 1-5"), (6, 20, "rank 6-20"), (21, 999, "rank 21+")):
    print(" %s" % name)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        vv = [x for x in v if x["rank"] is not None and lo_r <= x["rank"] <= hi_r]
        if len(vv) >= 20:
            line("   " + k, vv)
    allv = [x for x in rows if x["rank"] is not None and lo_r <= x["rank"] <= hi_r]
    line("   ALL CLASSES", allv)

print("\nleader rank 1-5, all classes, by month (trail-actual mean):")
bm = collections.defaultdict(list)
for x in rows:
    if x["rank"] is not None and x["rank"] <= 5:
        bm[x["m"]].append(x["cf"] - x["pnl"])
for m in sorted(bm):
    print("  %s n=%4d %+6.2f%%" % (m, len(bm[m]), sum(bm[m]) / len(bm[m])))
json.dump(rows, io.open(FILES.parent / ".runtime/backtests/audit_exit_rows.json", "w", encoding="utf-8"))
