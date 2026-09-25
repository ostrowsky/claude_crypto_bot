"""Would re-entering DURING the post-exit cooldown have paid?

WHY THIS EXISTS

ATOM 2026-09-06..09 was entered at 1.559, 1.621, 1.687 and 1.965 -- each exit
followed by a cooldown (COOLDOWN_BARS = 19) and a re-entry higher up. QNT
2026-09-24 exited at +5% on RSI and never came back. The exit rules themselves
were measured and kept (WEAK, RSI). What was never measured is the cooldown: when
an entry rule fires again inside the 19 bars after an exit, is that entry worth
taking?

METHOD

  exits      every 15m exit in bot_events.jsonl (the bot's own trades, TH-06)
  candidate  the FIRST bar inside (exit, exit + 19 x 15m] where a live entry
             rule fires under today's caps -- the FIRE rows of the strategy
             replay (_backtest_lateness_caps.py, .runtime/backtests/lateness_rows.jsonl)
  outcome    ATR trail from that bar's close (trail_k 2.0, the rule's mode floor,
             96-bar cap) on the repaired long 15m store; plus 4h peak and trough
  control    every other FIRE row of the same coins that is NOT inside any
             post-exit cooldown -- what a rule firing is worth in general
  split      by exit class: in profit (pnl > 0) vs at a loss, and WEAK/RSI exits
             separately, because a cooldown after a broken trend should help and
             one after a profitable exit might not.

SCOPE: rule layer only -- live, a re-entry would still face ml_zone,
trend_quality, chop, correlation and MAX_OPEN. Stated, not hidden.
VERDICT 2026-09-25: REFUTED. The cooldown stays. Do not re-test without new evidence.

4 066 15m exits, 2026-03-03 .. 2026-09-25; in 1 395 (34%) an entry rule fired
again inside the 19-bar cooldown, typically +1.42% above the exit price (94% of
the time above it). Against 24 993 rule firings outside any cooldown (trailed
mean -0.00%, median -0.34%):

    group                      n     trailed mean   minus control, 95% CI
    all cooldown re-entries  1 394      +0.04%      [-0.07, +0.16]
    after an exit in profit    607      +0.16%      [-0.04, +0.37]
    after an exit at a loss    787      -0.05%      [-0.19, +0.08]
    after a WEAK exit          400      +0.13%      [-0.08, +0.37]
    after an ATR-trail exit    585      +0.02%      [-0.16, +0.21]

No group excludes zero, medians are below the control's, and the months split
3 positive / 4 negative. A re-entry inside the cooldown is worth what any rule
firing is worth -- nothing -- so lifting the cooldown would add entries without
edge. The ATOM stair-step (in at 1.559, 1.621, 1.687, 1.965) is the market, not
a lost edge of the cooldown.
"""
from __future__ import annotations

import bisect
import collections
import io
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _backtest_weak_exit_above_breakeven as W  # noqa: E402

ROWS = FILES.parent / ".runtime" / "backtests" / "lateness_rows.jsonl"
CD = timedelta(minutes=15 * int(getattr(cfg, "COOLDOWN_BARS", 19)))

fire = collections.defaultdict(list)
for l in io.open(ROWS, encoding="utf-8"):
    r = json.loads(l)
    if r["band"] == "FIRE":
        r["dt"] = datetime.fromisoformat(r["ts"])
        fire[r["sym"]].append(r)
for v in fire.values():
    v.sort(key=lambda r: r["dt"])

exits = []
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if b'"exit"' not in raw:
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if e.get("event") != "exit" or e.get("tf") != "15m" or e.get("sym") not in fire:
            continue
        if not isinstance(e.get("pnl_pct"), (int, float)):
            continue
        d = datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        e["dt"] = d
        exits.append(e)
print("15m exits with a replayed coin: %d  (%s .. %s)" % (len(exits), min(e["dt"] for e in exits).date(), max(e["dt"] for e in exits).date()))

in_cd = set()
cands = []
for e in exits:
    rs = fire[e["sym"]]
    ts = [r["dt"] for r in rs]
    k = bisect.bisect_right(ts, e["dt"])
    first = None
    while k < len(rs) and rs[k]["dt"] <= e["dt"] + CD:
        in_cd.add(id(rs[k]))
        if first is None:
            first = rs[k]
        k += 1
    if first is not None:
        cands.append((e, first))
print("exits followed by a rule firing inside the cooldown: %d (%.0f%%)" % (len(cands), 100.0 * len(cands) / len(exits)))

_B = {}


def outcome(sym, dt, rule):
    if sym not in _B:
        b = TD.bars_15m(sym)
        _B[sym] = (b, {x[0]: i for i, x in enumerate(b)})
    bars, idx = _B[sym]
    i = idx.get(dt)
    if i is None or i + 16 >= len(bars):
        return None
    px = bars[i][4]
    fut = bars[i + 1:i + 17]
    mode = rule if rule in ("trend", "impulse", "alignment", "impulse_speed") else "trend"
    t = W.replay(bars, i, i, px, 2.0, mode, cfg, 96)
    if t is None:
        return None
    return {"trail": t, "peak": (max(b[2] for b in fut) / px - 1) * 100, "trough": (min(b[3] for b in fut) / px - 1) * 100}


def s(vals):
    v = [x for x in vals if x is not None]
    n = len(v)
    if not n:
        return None
    t = sorted(x["trail"] for x in v)
    return {"n": n, "mean": sum(t) / n, "med": t[n // 2], "win": sum(x > 0 for x in t) / n,
            "peak": sum(x["peak"] for x in v) / n, "trough": sum(x["trough"] for x in v) / n, "t": t}


def show(name, st):
    if not st:
        print("  %-34s n=0" % name)
        return
    print("  %-34s n=%6d  trailed mean %+6.2f%%  median %+6.2f%%  >0 %3.0f%%  | 4h peak %5.2f%%  trough %6.2f%%" % (
        name, st["n"], st["mean"], st["med"], 100 * st["win"], st["peak"], st["trough"]))


cand_out = [(e, r, outcome(r["sym"], r["dt"], r["rule"])) for e, r in cands]
ctrl_rows = [r for rs in fire.values() for r in rs if id(r) not in in_cd]
rnd = random.Random(9)
ctrl_rows = rnd.sample(ctrl_rows, min(25000, len(ctrl_rows)))
ctrl = s([outcome(r["sym"], r["dt"], r["rule"]) for r in ctrl_rows])

print("\n=== RE-ENTRY INSIDE THE COOLDOWN vs A RULE FIRING ANYWHERE ELSE ===")
show("control: FIRE outside cooldowns", ctrl)
show("all cooldown re-entries", s([o for _, _, o in cand_out]))
weak = lambda e: str(e.get("reason") or "").startswith("\u26a0") or "weak" in str(e.get("reason") or "").lower()
rsi = lambda e: "RSI перекуплен" in str(e.get("reason") or "")
groups = [
    ("after an exit IN PROFIT", lambda e: e["pnl_pct"] > 0),
    ("after an exit AT A LOSS", lambda e: e["pnl_pct"] <= 0),
    ("after a WEAK exit", weak),
    ("after an RSI-overbought exit", rsi),
    ("after an ATR-trail exit", lambda e: "ATR" in str(e.get("reason") or "")),
]
res = {}
for name, f in groups:
    st = s([o for e, _, o in cand_out if f(e)])
    res[name] = st
    show(name, st)


def boot(a, b, reps=1000):
    o = sorted(sum(rnd.choice(a) for _ in a) / len(a) - sum(rnd.choice(b) for _ in b) / len(b) for _ in range(reps))
    return o[int(.025 * reps)], o[int(.975 * reps)]


print("\nminus control, 95% bootstrap interval:")
for name, st in [("all cooldown re-entries", s([o for _, _, o in cand_out]))] + list(res.items()):
    if st and st["n"] >= 30:
        lo, hi = boot(st["t"], ctrl["t"])
        print("  %-34s %+6.2f%%   [%+.2f, %+.2f]  %s" % (name, st["mean"] - ctrl["mean"], lo, hi,
                                                     "zero inside" if lo <= 0 <= hi else "EXCLUDES zero"))

print("\nby month, all cooldown re-entries (trailed mean [n]) vs control:")
bm = collections.defaultdict(list)
for e, r, o in cand_out:
    if o:
        bm[r["dt"].strftime("%Y-%m")].append(o["trail"])
for m in sorted(bm):
    v = bm[m]
    print("  %s  %+6.2f%% [%4d]" % (m, sum(v) / len(v), len(v)))

gaps = []
for e, r in cands:
    b = _B.get(r["sym"])
    if b and r["dt"] in b[1] and e.get("exit_price"):
        gaps.append((b[0][b[1][r["dt"]]][4] / e["exit_price"] - 1) * 100)
gaps.sort()
print("re-entry price vs the exit price: median %+.2f%%, p25 %+.2f%%, p75 %+.2f%% (n=%d)" % (
    gaps[len(gaps) // 2], gaps[len(gaps) // 4], gaps[3 * len(gaps) // 4], len(gaps)))
