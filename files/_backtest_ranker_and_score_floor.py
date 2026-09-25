"""E-3 and a finding on the way: what the candidate ranker and the entry-score floor still decide.

1. Ranker components against the outcome, by month, over every row that logged
   them (entries + blocked): AUC of final_score / top_gainer_prob / quality_proba
   vs the 4h forward peak >= 3% (pipeline_replay_validator.forward_peak), rows
   deduplicated per coin-hour. These are live scores graded on later prices, so
   they are out-of-sample by construction.
2. The entry-score floor. Since 2026-06 REGIME_SOFT_GATE_ENABLED turns a
   below-floor candidate into a soft pass that "the bandit decides"; since
   2026-09-07 the bandit gate is off, so the floor blocks nothing. Entries below
   and above the floor are compared on the calibrated engine (live_trail, P-1)
   and on their real exits, by month, and by their share of early entries into
   immutable top-20 winner-days.
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
import label_store as LS  # noqa: E402
import pipeline_replay_validator as V  # noqa: E402
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

BAR = timedelta(minutes=15)
FLOORS = {"impulse_speed": "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", "strong_trend": "TRAIL_MIN_BUFFER_PCT_STRONG_TREND",
          "impulse": "TRAIL_MIN_BUFFER_PCT_IMPULSE", "trend": "TRAIL_MIN_BUFFER_PCT_TREND",
          "alignment": "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", "retest": "TRAIL_MIN_BUFFER_PCT_RETEST",
          "breakout": "TRAIL_MIN_BUFFER_PCT_BREAKOUT"}


def floor_pct(mode):
    if not getattr(cfg, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    return float(getattr(cfg, FLOORS.get(mode, "TRAIL_MIN_BUFFER_PCT_DEFAULT"), 0.0))


def live_trail(c, atr, ie, k, fl, max_bars=96):
    """P-1 calibrated live trail: close-anchored ratchet, exit at the first close below the stop."""
    ep = c[ie]
    a0 = atr[ie] if np.isfinite(atr[ie]) else 0.0
    stop = ep - max(k * a0, fl * ep)
    last = min(len(c) - 1, ie + max_bars)
    if last <= ie:
        return None
    for j in range(ie + 1, last + 1):
        aj = atr[j] if np.isfinite(atr[j]) else 0.0
        if aj > 0:
            stop = max(stop, c[j] - max(k * aj, fl * c[j]))
        if c[j] < stop:
            return j, (c[j] / ep - 1) * 100
    return last, (c[last] / ep - 1) * 100


def auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows, seen = [], set()
entries, exits = [], {}
with io.open(FILES / "bot_events.jsonl", "rb") as fh:
    for raw in fh:
        if not (b'"entry"' in raw or b'"exit"' in raw or b"ranker_final_score" in raw):
            continue
        try:
            e = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        ev = e.get("event")
        ts = str(e.get("ts", ""))
        if not ts:
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if ev == "entry":
            entries.append((d, e))
        elif ev == "exit":
            exits.setdefault((e.get("sym"), e.get("tf")), []).append((d, e))
        if ev in ("entry", "blocked") and isinstance(e.get("ranker_final_score"), (int, float)):
            k = (e.get("sym"), d.strftime("%Y-%m-%d %H"))
            if k in seen:
                continue
            seen.add(k)
            rows.append((d, e))

# ---------------- 1. ranker AUC by month ----------------
print("=== 1. RANKER components vs 4h forward peak >= 3% (coin-hour rows, entries + blocked) ===")
by_m = collections.defaultdict(list)
for d, e in rows:
    pk = V.forward_peak(e["sym"], str(e.get("tf")), d, float(e.get("price") or 0))
    if pk is None:
        continue
    by_m[d.strftime("%Y-%m")].append((pk >= 3.0, e.get("ranker_final_score"), e.get("ranker_top_gainer_prob"), e.get("ranker_quality_proba")))
print("  month     n    base   final_score  top_gainer_prob  quality_proba")
allr = []
for m in sorted(by_m):
    v = by_m[m]
    allr += v
    out = []
    for j in (1, 2, 3):
        sub = [(x[0], x[j]) for x in v if isinstance(x[j], (int, float))]
        out.append(auc([a for a, _ in sub], [b for _, b in sub]) if len(sub) > 50 else float("nan"))
    print("  %s %6d  %5.1f%%     %.3f         %.3f          %.3f" % (m, len(v), 100 * np.mean([x[0] for x in v]), *out))
out = []
for j in (1, 2, 3):
    sub = [(x[0], x[j]) for x in allr if isinstance(x[j], (int, float))]
    out.append(auc([a for a, _ in sub], [b for _, b in sub]))
print("  ALL     %6d  %5.1f%%     %.3f         %.3f          %.3f" % (len(allr), 100 * np.mean([x[0] for x in allr]), *out))

# ---------------- 2. entries below vs above the floor ----------------
print("\n=== 2. ENTRY-SCORE FLOOR: entries below vs at/above it (15m, calibrated engine + real exits) ===")


def real_pnl(d, e):
    for xd, x in exits.get((e.get("sym"), e.get("tf")), ()):
        if xd > d and isinstance(x.get("pnl_pct"), (int, float)):
            return float(x["pnl_pct"])
    return None


bars = {}
grp = collections.defaultdict(list)
for d, e in entries:
    if e.get("tf") != "15m" or d < datetime(2026, 6, 1, tzinfo=timezone.utc):
        continue
    cs, fl = e.get("candidate_score"), e.get("score_floor")
    if not (isinstance(cs, (int, float)) and isinstance(fl, (int, float))):
        continue
    sym = e["sym"]
    if sym not in bars:
        b = TD.bars_15m(sym)
        if not b:
            bars[sym] = None
            continue
        h = np.array([x[2] for x in b]); l = np.array([x[3] for x in b]); c = np.array([x[4] for x in b])
        bars[sym] = ({x[0]: i for i, x in enumerate(b)}, c, I._atr(h, l, c, cfg.ATR_PERIOD))
    if bars[sym] is None:
        continue
    idx, c, atr = bars[sym]
    i = idx.get(d.replace(minute=d.minute // 15 * 15, second=0, microsecond=0) - BAR)
    if i is None:
        continue
    k = e.get("trail_k")
    r = live_trail(c, atr, i, float(k) if isinstance(k, (int, float)) else 2.0, floor_pct(e.get("mode")))
    if not r:
        continue
    side = "below" if cs < fl else "above"
    grp[(side, d.strftime("%Y-%m"))].append((r[1], real_pnl(d, e), cs - fl))
    grp[(side, "ALL")].append((r[1], real_pnl(d, e), cs - fl))
rnd = random.Random(3)
print("  month    below: n  engine   real      above: n  engine   real     below-above engine [95% CI]")
for m in sorted({k[1] for k in grp if k[1] != "ALL"}) + ["ALL"]:
    b, a = grp.get(("below", m), []), grp.get(("above", m), [])
    if not b or not a:
        continue
    be, ae = [x[0] for x in b], [x[0] for x in a]
    br = [x[1] for x in b if x[1] is not None]
    ar = [x[1] for x in a if x[1] is not None]
    bs = sorted(np.mean([rnd.choice(be) for _ in be]) - np.mean([rnd.choice(ae) for _ in ae]) for _ in range(1000))
    print("  %-7s  %5d %+6.2f%% %+6.2f%%    %5d %+6.2f%% %+6.2f%%     %+.2f [%+.2f, %+.2f]" % (
        m, len(b), np.mean(be), np.mean(br) if br else float("nan"), len(a), np.mean(ae), np.mean(ar) if ar else float("nan"),
        np.mean(be) - np.mean(ae), bs[25], bs[-26]))
below = grp.get(("below", "ALL"), [])
if below:
    gaps = sorted(x[2] for x in below)
    print("  how far below the floor (score - floor): median %.1f, p10 %.1f" % (gaps[len(gaps) // 2], gaps[len(gaps) // 10]))

# goal: early entries into winner-days, by side of the floor
wl = E.load_watchlist()
full, _, _ = E.load_uptime(datetime(2026, 6, 1, tzinfo=timezone.utc))
win, _ = IL.winners_by_day(top_n=20, watchlist=wl, rank_before_filter=True)
dl = LS.intraday_deadlines()
W = {k for k in win if k[0] >= "2026-06-01" and k[0] in full and k in dl and dl[k][1] is not None}
hit_days = collections.defaultdict(set)
for d, e in entries:
    key = (d.strftime("%Y-%m-%d"), e.get("sym"))
    if key not in W:
        continue
    op, dd = dl[key]
    if not (op <= d < dd):
        continue
    cs, fl = e.get("candidate_score"), e.get("score_floor")
    side = "below" if isinstance(cs, (int, float)) and isinstance(fl, (int, float)) and cs < fl else "above/unknown"
    hit_days[side].add(key)
only_below = hit_days["below"] - hit_days["above/unknown"]
print("\n  winner-days since 2026-06 (bot up, crossing known): %d; entered before the crossing: %d" % (
    len(W), len(hit_days["below"] | hit_days["above/unknown"])))
print("  ... of which ONLY by below-floor entries: %d  (a working floor would have lost them)" % len(only_below))
