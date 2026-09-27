"""X-6: once a held coin becomes a CONFIRMED day-leader, hold it on a wide trail.

WHY
_audit_rocket_blockers.py: the bot buys 57% of rockets before +10%, but the
median trade is +1.3% on a 14.1% day move and the coin rises another +8.4%
after the exit (ATR trail, WEAK RSI divergence, RSI overbought, micro-weakness).
A wide exit for everyone failed before (it costs more on ordinary coins); but a
leader is now identifiable in real time (leader_alert.py: 8 closed 15m bars in
the day's top-3 at >= +7.5% -> ~70% real leaders). X-6 widens the exit ONLY
for those positions.

COUNTERFACTUAL, per real 15m entry of the bot (2026-03-01 on, with its exit)
  - if the coin was not a confirmed leader at any closed bar between the entry
    and the real exit: X-6 = the real pnl (nothing changes)
  - else, from the first confirmed-leader bar s (< real exit): the position is
    managed only by a wide close-anchored trail -- stop reset to
    close_s - max(k*ATR, F*close_s) and ratcheted up (P-1 engine) -- WEAK /
    RSI / EMA / time exits off; cap 7 days
Pre-registered variants (fixed before running, all printed):
  W8   floor F = 8%
  W12  floor F = 12%
  LL   exit when the coin drops out of the day's top-10 (>= 1h after s), 8% floor as a safety net
Criterion: per-trade non-inferiority over ALL 15m entries (lower 95% bound of
X-6 - real >= -0.10 pp) and the goal side -- capture (pnl / day move) on
rocket-days and immutable top-20 winner-days.
Spec: docs/specs/features/leader-exit-x6-spec.md
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
LEAD_RANK, LEAD_RET, LEAD_BARS = 3, 0.075, 8

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
LEAD = np.zeros((N2, S), dtype=bool)          # confirmed leader at the close of bar i
for d in range(days):
    streak = np.zeros(S, dtype=int)
    for i in range(d * 96, (d + 1) * 96):
        cond = (RANK[i] <= LEAD_RANK) & (RET[i] >= LEAD_RET)
        streak = np.where(cond, streak + 1, 0)
        LEAD[i] = streak >= LEAD_BARS
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
                       "k": float(e.get("trail_k") or 2.0), "day": d.strftime("%Y-%m-%d"), "reason": str(x[1].get("reason"))[:40]})
print("real 15m trades with an exit since %s: %d" % (START, len(trades)))


def x6(t, floor, lost_lead):
    j, ie, ix = t["j"], t["ie"], t["ix"]
    lead = np.where(LEAD[ie:ix, j])[0]          # confirmed at a closed bar before the real exit
    if len(lead) == 0:
        return t["real"], False
    s = ie + int(lead[0])
    ep = t["ep"]
    cs = C[s, j]
    k = t["k"]
    stop = cs - max(k * ATR[s, j], floor * cs)
    last = min(N2 - 1, s + CAP)
    for q in range(s + 1, last + 1):
        cq = C[q, j]
        if not np.isfinite(cq):
            continue
        if np.isfinite(ATR[q, j]):
            stop = max(stop, cq - max(k * ATR[q, j], floor * cq))
        if cq < stop:
            return (cq / ep - 1) * 100, True
        if lost_lead and q - s >= 4 and RANK[q, j] > 10:
            return (cq / ep - 1) * 100, True
    q = last
    while q > s and not np.isfinite(C[q, j]):
        q -= 1
    return (C[q, j] / ep - 1) * 100, True


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
for name, floor, ll in (("W8  floor 8%", 0.08, False), ("W12 floor 12%", 0.12, False), ("LL  lost lead + 8%", 0.08, True)):
    diffs, touched = [], []
    for t in trades:
        v, hit = x6(t, floor, ll)
        t[name] = v
        diffs.append(v - t["real"])
        if hit:
            touched.append(t)
    d = np.array(diffs)
    bs = sorted(np.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(1000))
    print("\n=== %s ===" % name)
    print("  all 15m trades n=%d: real mean %+.3f%% -> X-6 %+.3f%%, diff %+.3f pp [%+.3f, %+.3f] -> %s" % (
        len(trades), np.mean([t["real"] for t in trades]), np.mean([t[name] for t in trades]), d.mean(), bs[25], bs[-26],
        "NON-INFERIOR" if bs[25] >= -0.10 else "FAILS -0.10 pp"))
    if touched:
        tr = np.array([t["real"] for t in touched]); tx = np.array([t[name] for t in touched])
        print("  trades that became a confirmed leader before the real exit: %d (%.1f%%): real mean %+.2f%% median %+.2f%% -> X-6 mean %+.2f%% median %+.2f%%, better in %.0f%%" % (
            len(touched), 100 * len(touched) / len(trades), tr.mean(), np.median(tr), tx.mean(), np.median(tx), 100 * np.mean(tx > tr)))
    for gname, key in (("rocket-days", "rocket"), ("immutable top-20 winner-days", "winner")):
        g = [t for t in trades if t[key] and np.isfinite(t["move"]) and t["move"] > 0]
        if g:
            cr = np.median([t["real"] / t["move"] for t in g]) * 100
            cx = np.median([t[name] / t["move"] for t in g]) * 100
            print("  %-28s n=%4d  median capture real %4.0f%% -> X-6 %4.0f%% | mean pnl real %+.2f%% -> X-6 %+.2f%%" % (
                gname, len(g), cr, cx, np.mean([t["real"] for t in g]), np.mean([t[name] for t in g])))
    bym = collections.defaultdict(list)
    for t in trades:
        bym[t["day"][:7]].append(t[name] - t["real"])
    print("  by month (X-6 - real, mean pp): " + "  ".join("%s %+.2f" % (m[5:], np.mean(v)) for m, v in sorted(bym.items())))
json.dump([{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in t.items()} for t in trades],
          io.open(FILES.parent / ".runtime/backtests/leader_exit_x6_trades.json", "w", encoding="utf-8"), default=str)
