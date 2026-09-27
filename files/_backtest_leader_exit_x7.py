"""X-7: switch the exit EARLIER than X-6 -- three variants fixed before the run.

X-6 (leader-exit-x6-spec.md) was safe but touched 2.4% of trades: the bot had
already exited 97.6% of the time before leadership was confirmed (2h). X-7:

  X-7a  confirmation after 1h (4 closed bars in the day's top-3 at >= +7.5%),
        then exit on losing the day's top-10 (>= 1h after) with an 8% floor
  X-7b  switch at the first closed bar with rank <= 3 at >= +5% (no hold), same exit
  X-7c  do not widen the stop; if the REAL exit was a soft one (WEAK, RSI
        overbought, micro-weakness, EMA20 exits) while the coin was in the day's
        top-5, keep holding on the plain P-1 trail with the entry's k and mode
        floor, and exit at the first close where it is out of the top-5
Same counterfactual rules as X-6 otherwise; same criterion: per-trade
non-inferiority over all 15m trades + capture on rocket / winner days.
Spec: docs/specs/features/leader-exit-x6-spec.md (section X-7)
"""
import collections
import io
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import immutable_labels as IL  # noqa: E402
import indicators as I  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

START = "2026-03-01"
STEP = timedelta(minutes=15)
CAP = 96 * 7

wl = E.load_watchlist()
raw = {s: TD.bars_15m(s) for s in wl}
raw = {s: b for s, b in raw.items() if len(b) > 2000}
syms = sorted(raw)
col = {s: j for j, s in enumerate(syms)}
t0 = min(b[0][0] for b in raw.values()).replace(hour=0, minute=0)
t1 = max(b[-1][0] for b in raw.values())
N = int((t1 - t0) / STEP) + 1
S = len(syms)
O, H, L, C = (np.full((N, S), np.nan) for _ in range(4))
for j, s in enumerate(syms):
    for (t, o, h, l, c, v) in raw[s]:
        k = int((t - t0) / STEP)
        O[k, j], H[k, j], L[k, j], C[k, j] = o, h, l, c
ATR = np.column_stack([I._atr(np.nan_to_num(H[:, j], nan=np.nanmean(H[:, j])),
                              np.nan_to_num(L[:, j], nan=np.nanmean(L[:, j])),
                              np.nan_to_num(C[:, j], nan=np.nanmean(C[:, j])), cfg.ATR_PERIOD) for j in range(S)])
days = N // 96
N2 = days * 96
RET = C[:N2] / np.repeat(O[::96][:days], 96, axis=0) - 1
RANK = np.full((N2, S), 999, dtype=int)
for i in range(N2):
    row = RET[i]
    ok = np.isfinite(row)
    if ok.any():
        order = np.argsort(-np.where(ok, row, -9))
        RANK[i, order] = np.arange(1, S + 1)
        RANK[i, ~ok] = 999
def lead_matrix(rank_max, min_ret, bars):
    """True at the close of bar i when the coin held rank <= rank_max at >= min_ret for `bars` closed bars of the day."""
    m = np.zeros((N2, S), dtype=bool)
    for d in range(days):
        streak = np.zeros(S, dtype=int)
        for i in range(d * 96, (d + 1) * 96):
            cond = (RANK[i] <= rank_max) & (RET[i] >= min_ret)
            streak = np.where(cond, streak + 1, 0)
            m[i] = streak >= bars
    return m


LEAD_A = lead_matrix(3, 0.075, 4)
LEAD_B = lead_matrix(3, 0.05, 1)
print("grid %s .. %s, %d coins" % (t0.date(), t1.date(), S))


def idx_of(dt):
    return int((dt - t0) / STEP)


# real 15m entries with their exits
ent = collections.defaultdict(list)
exs = collections.defaultdict(list)
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw_ in fh:
        if b'"entry"' not in raw_ and b'"exit"' not in raw_:
            continue
        try:
            e = json.loads(raw_.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("tf") != "15m" or str(e.get("ts", "")) < START or e.get("sym") not in col:
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        (ent if e["event"] == "entry" else exs)[e["sym"]].append((d, e))
trades = []
for sym, es in ent.items():
    xs = sorted(exs.get(sym, []), key=lambda x: x[0])
    for d, e in sorted(es, key=lambda x: x[0]):
        x = next((x for x in xs if x[0] > d), None)
        if x is None or not isinstance(x[1].get("pnl_pct"), (int, float)):
            continue
        ie = idx_of(d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0)) - 1
        ix = idx_of(x[0].replace(minute=x[0].minute // 15 * 15, second=0, microsecond=0)) - 1
        if ie < 100 or ix <= ie or ix >= N2:
            continue
        trades.append({"sym": sym, "j": col[sym], "ie": ie, "ix": ix, "ep": float(e["price"]), "real": float(x[1]["pnl_pct"]),
                       "k": float(e.get("trail_k") or 2.0), "mode": e.get("mode") or e.get("signal_mode"), "day": d.strftime("%Y-%m-%d"), "reason": str(x[1].get("reason") or "")})
print("real 15m trades with an exit since %s: %d" % (START, len(trades)))


FLOORS = {"impulse_speed": "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", "strong_trend": "TRAIL_MIN_BUFFER_PCT_STRONG_TREND",
          "impulse": "TRAIL_MIN_BUFFER_PCT_IMPULSE", "trend": "TRAIL_MIN_BUFFER_PCT_TREND",
          "alignment": "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", "retest": "TRAIL_MIN_BUFFER_PCT_RETEST",
          "breakout": "TRAIL_MIN_BUFFER_PCT_BREAKOUT"}


def floor_pct(mode):
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    return float(getattr(cfg, FLOORS.get(mode, "TRAIL_MIN_BUFFER_PCT_DEFAULT"), 0.0))


def soft(reason):
    r = reason or ""
    return ("WEAK" in r or "перекуплен" in r or "micro-weakness" in r or "EMA20" in r) and "ATR" not in r


def wide_from(t, s, floor=0.08, lost_lead=True):
    j, ep, k = t["j"], t["ep"], t["k"]
    cs = C[s, j]
    stop = cs - max(k * ATR[s, j], floor * cs)
    last = min(N2 - 1, s + CAP)
    for q in range(s + 1, last + 1):
        cq = C[q, j]
        if not np.isfinite(cq):
            continue
        if np.isfinite(ATR[q, j]):
            stop = max(stop, cq - max(k * ATR[q, j], floor * cq))
        if cq < stop or (lost_lead and q - s >= 4 and RANK[q, j] > 10):
            return (cq / ep - 1) * 100
    q = last
    while q > s and not np.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100


def variant(t, name):
    j, ie, ix = t["j"], t["ie"], t["ix"]
    if name in ("X-7a", "X-7b"):
        m = LEAD_A if name == "X-7a" else LEAD_B
        lead = np.where(m[ie:ix, j])[0]
        if len(lead) == 0:
            return t["real"], False
        return wide_from(t, ie + int(lead[0])), True
    # X-7c: soft real exit while in the day's top-5 -> hold on the plain trail until out of the top-5
    if not soft(t["reason"]) or RANK[ix, j] > 5:
        return t["real"], False
    ep, k, fl = t["ep"], t["k"], floor_pct(t["mode"])
    a0 = ATR[ie, j] if np.isfinite(ATR[ie, j]) else 0.0
    stop = ep - max(k * a0, fl * ep)
    last = min(N2 - 1, ix + CAP)
    for q in range(ie + 1, last + 1):
        cq = C[q, j]
        if not np.isfinite(cq):
            continue
        if np.isfinite(ATR[q, j]):
            stop = max(stop, cq - max(k * ATR[q, j], fl * cq))
        if q <= ix:
            continue
        if cq < stop or RANK[q, j] > 5:
            return (cq / ep - 1) * 100, True
    return (C[last, j] / ep - 1) * 100, True


# goal-side labels: rocket days and immutable winners
rocket = {(r["day"], r["sym"]) for r in map(json.loads, io.open(FILES.parent / ".runtime/backtests/rocket_events.jsonl", encoding="utf-8"))
          if r["trig"] == 0.025 and r["rocket"]}
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
win = set(win)
for t in trades:
    d0 = idx_of(datetime.strptime(t["day"], "%Y-%m-%d").replace(tzinfo=timezone.utc))
    op = O[d0, t["j"]]
    t["move"] = (np.nanmax(H[d0:d0 + 96, t["j"]]) / op - 1) * 100 if op > 0 else np.nan
    t["rocket"] = (t["day"], t["sym"]) in rocket
    t["winner"] = (t["day"], t["sym"]) in win

rnd = random.Random(6)
for name in ("X-7a", "X-7b", "X-7c"):
    diffs, touched = [], []
    for t in trades:
        v, hit = variant(t, name)
        t[name] = v
        diffs.append(v - t["real"])
        if hit:
            touched.append(t)
    d = np.array(diffs)
    bs = sorted(np.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(1000))
    print("\n=== %s ===" % name)
    print("  all 15m trades n=%d: real mean %+.3f%% -> variant %+.3f%%, diff %+.3f pp [%+.3f, %+.3f] -> %s" % (
        len(trades), np.mean([t["real"] for t in trades]), np.mean([t[name] for t in trades]), d.mean(), bs[25], bs[-26],
        "NON-INFERIOR" if bs[25] >= -0.10 else "FAILS -0.10 pp"))
    if touched:
        tr = np.array([t["real"] for t in touched]); tx = np.array([t[name] for t in touched])
        print("  trades the variant changed: %d (%.1f%%): real mean %+.2f%% median %+.2f%% -> variant mean %+.2f%% median %+.2f%%, better in %.0f%%" % (
            len(touched), 100 * len(touched) / len(trades), tr.mean(), np.median(tr), tx.mean(), np.median(tx), 100 * np.mean(tx > tr)))
    for gname, key in (("rocket-days", "rocket"), ("immutable top-20 winner-days", "winner")):
        g = [t for t in trades if t[key] and np.isfinite(t["move"]) and t["move"] > 0]
        if g:
            cr = np.median([t["real"] / t["move"] for t in g]) * 100
            cx = np.median([t[name] / t["move"] for t in g]) * 100
            print("  %-28s n=%4d  median capture real %4.0f%% -> variant %4.0f%% | mean pnl real %+.2f%% -> variant %+.2f%%" % (
                gname, len(g), cr, cx, np.mean([t["real"] for t in g]), np.mean([t[name] for t in g])))
    bym = collections.defaultdict(list)
    for t in trades:
        bym[t["day"][:7]].append(t[name] - t["real"])
    print("  by month (variant - real, mean pp): " + "  ".join("%s %+.2f" % (m[5:], np.mean(v)) for m, v in sorted(bym.items())))
json.dump([{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in t.items()} for t in trades],
          io.open(FILES.parent / ".runtime/backtests/leader_exit_x7_trades.json", "w", encoding="utf-8"), default=str)
