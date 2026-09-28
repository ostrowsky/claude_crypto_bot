"""X-8 and X-9: make the rockets the bot already touches land in leader mode.

Operator, 2026-09-28 (HBAR: bought at +2.3% at 06:16, sold +1.3% at 08:15 on a
WEAK exit at +3.6% of the day -- below X-7b's +5% switch -- then +17% at 09:00
inside the cooldown): "such rockets must end up among the leaders".
Baseline = X-7b as live since 2026-09-27 (leader-exit-x6-spec.md). Variants
fixed before this run:

  X-9a  switch to leader mode at rank <= 3 and >= +3%     (X-7b: +5%)
  X-9b  switch at rank <= 3 and >= +2.5%
  X-9c  switch at rank <= 5 and >= +3%
  X-8   cooldown override: after a real exit the variant did not switch, if
        within the next COOLDOWN_BARS closed bars the coin is rank <= 3 at >= +5%
        and the bot has not re-entered it, re-enter at that close directly in
        leader mode (wide trail 8% + exit on losing the day's top-10, 7-day cap)
  X-8 on top of X-7b and on top of X-9a

Criterion: per-trade non-inferiority against X-7b (lower 95% bound of the
difference >= -0.10 pp over all trades, added X-8 trades included), mean trade
on rocket / winner days, and the operator's requirement measured directly:
coverage = share of rocket-days on which the coin was held in leader mode.
Spec: docs/specs/features/leader-exit-x6-spec.md (section X-8 / X-9)
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


LEAD_9A = lead_matrix(3, 0.03, 1)
LEAD_9B = lead_matrix(3, 0.025, 1)
LEAD_9C = lead_matrix(5, 0.03, 1)
COOL = int(getattr(cfg, "COOLDOWN_BARS", 8))


def switched(t, m):
    lead = np.where(m[t["ie"]:t["ix"], t["j"]])[0]
    return None if len(lead) == 0 else t["ie"] + int(lead[0])


def managed(t, m):
    s = switched(t, m)
    if s is None:
        return t["real"], None
    return wide_from(t, s), s


real_entries = collections.defaultdict(list)
for t in trades:
    real_entries[t["j"]].append(t["ie"])

VARS = {"X-7b (live)": LEAD_B, "X-9a top3 +3%": LEAD_9A, "X-9b top3 +2.5%": LEAD_9B, "X-9c top5 +3%": LEAD_9C}
res = {}
for name, m in VARS.items():
    vals, sw = [], []
    for t in trades:
        v, s = managed(t, m)
        vals.append(v)
        sw.append(s)
    res[name] = (vals, sw)


def x8_added(m_base):
    """Re-entries in leader mode after exits the base variant did not switch."""
    added = []
    for t, s in zip(trades, res[m_base][1]):
        if s is not None:
            continue
        j, ix = t["j"], t["ix"]
        win_end = min(N2 - 1, ix + COOL)
        lead = np.where(LEAD_B[ix + 1:win_end + 1, j])[0]
        if len(lead) == 0:
            continue
        q = ix + 1 + int(lead[0])
        if any(ix < e <= q for e in real_entries[j]):
            continue
        tt = dict(t, ep=C[q, j], ie=q)
        pnl = wide_from(tt, q)
        day = (t0 + q * STEP).strftime("%Y-%m-%d")
        added.append({"day": day, "sym": t["sym"], "pnl": pnl, "rocket": (day, t["sym"]) in rocket,
                      "winner": (day, t["sym"]) in win})
    return added


rnd = random.Random(12)


def ci(d):
    bs = sorted(np.mean(rnd.choices(d, k=len(d))) for _ in range(1000))
    return bs[25], bs[-26]


base_vals = res["X-7b (live)"][0]
full, _, _ = E.load_uptime(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc))
rocket_days = sorted(k for k in rocket if k[0] >= START and k[0] in full)


def coverage(sw, added):
    held = set()
    for t, s in zip(trades, sw):
        if s is not None:
            held.add(((t0 + s * STEP).strftime("%Y-%m-%d"), t["sym"]))
    for a in added:
        held.add((a["day"], a["sym"]))
    return sum(1 for k in rocket_days if k in held)


print("rocket-days since %s with the bot up all day: %d" % (START, len(rocket_days)))
print("%-18s %8s %30s %20s %20s %22s" % ("variant", "switched", "all trades vs X-7b (pp)", "rocket-day mean", "winner-day mean", "rockets in leader mode"))
for name, (vals, sw) in res.items():
    d = [v - b for v, b in zip(vals, base_vals)]
    lo, hi = ci(d)
    rk = [v for v, t in zip(vals, trades) if t["rocket"]]
    wn = [v for v, t in zip(vals, trades) if t["winner"]]
    cov = coverage(sw, [])
    print("%-18s %8d   %+.3f [%+.3f, %+.3f] %s   %+6.2f%% (n=%d)   %+6.2f%% (n=%d)    %4d = %4.1f%%" % (
        name, sum(1 for s in sw if s is not None), np.mean(d), lo, hi, "ok " if lo >= -0.10 else "BAD",
        np.mean(rk), len(rk), np.mean(wn), len(wn), cov, 100 * cov / len(rocket_days)))
    bym = collections.defaultdict(list)
    for v, b, t in zip(vals, base_vals, trades):
        if t["rocket"]:
            bym[t["day"][:7]].append(v - b)
    dr = [v - b for v, b, t in zip(vals, base_vals, trades) if t["rocket"]]
    dw = [v - b for v, b, t in zip(vals, base_vals, trades) if t["winner"]]
    if any(dr):
        l1, h1 = ci(dr)
        l2, h2 = ci(dw)
        print("      vs X-7b: rocket-days %+.2f pp [%+.2f, %+.2f], winner-days %+.2f pp [%+.2f, %+.2f]" % (np.mean(dr), l1, h1, np.mean(dw), l2, h2))
    print("      rocket-day diff vs X-7b by month: " + "  ".join("%s %+.2f" % (m[5:], np.mean(x)) for m, x in sorted(bym.items())))
ndays = max(1, len({t["day"] for t in trades}))
for base in ("X-7b (live)", "X-9a top3 +3%"):
    added = x8_added(base)
    vals = res[base][0]
    comb = vals + [a["pnl"] for a in added]
    bs = sorted(np.mean(rnd.choices(comb, k=len(comb))) - np.mean(rnd.choices(base_vals, k=len(base_vals))) for _ in range(1000))
    ap = [a["pnl"] for a in added]
    cov = coverage(res[base][1], added)
    rk = [a["pnl"] for a in added if a["rocket"]]
    print("")
    print("X-8 on top of %s: %d re-entries (%.2f/day); mean %+.2f%% median %+.2f%%; on rocket-days %d (mean %s)" % (
        base, len(added), len(added) / ndays, np.mean(ap) if ap else float("nan"), np.median(ap) if ap else float("nan"),
        len(rk), ("%+.2f%%" % np.mean(rk)) if rk else "-"))
    print("   combined mean - X-7b mean: %+.3f pp [%+.3f, %+.3f] %s | rockets in leader mode %d = %.1f%%" % (
        np.mean(comb) - np.mean(base_vals), bs[25], bs[-26], "ok" if bs[25] >= -0.10 else "BAD", cov, 100 * cov / len(rocket_days)))
    bym = collections.defaultdict(list)
    for a in added:
        bym[a["day"][:7]].append(a["pnl"])
    print("   re-entries by month: " + "  ".join("%s %+.2f(n=%d)" % (m[5:], np.mean(v), len(v)) for m, v in sorted(bym.items())))
