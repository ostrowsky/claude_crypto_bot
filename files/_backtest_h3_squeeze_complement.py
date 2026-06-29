"""H3 + complementarity: does a volatility-SQUEEZE (Bollinger-width in a low
percentile = pre-breakout coil) flag silent-miss top-20 BEFORE the move, and does
it catch a DIFFERENT subset than decoupling (H1)? Decoupling catches high-vol
idiosyncratic movers; squeeze should catch the quiet accumulators — if
complementary, the scan-promotion rule should be a UNION (decoupling OR squeeze).

Mirrors H1: recompute at the pre-move bar from klines (no lookahead), compare
silent-miss vs baseline, and report the UNION recall vs decoupling alone.
Read-only.  pyembed\python.exe files\_backtest_h3_squeeze_complement.py
"""
from __future__ import annotations
import csv, io, json, sys, random
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config
from decoupling_signal import scores_from_rets

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
WIN_1H = int(getattr(config, "DECOUPLING_WINDOW_BARS", 168))
VOLQ = float(getattr(config, "DECOUPLING_VOL_PCTILE_MIN", 0.66))
CORRMAX = float(getattr(config, "DECOUPLING_CORR_MAX", 0.60))
BB_PCTILE_MAX = 0.25   # squeeze = BB width in bottom 25% of its trailing range
BB_LOOK = 96           # ~1d of 15m bars for the percentile window

K = {}      # 1h closes (for decoupling)
K15 = {}    # 15m closes (for squeeze BB width)
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]
    if sym not in WL:
        continue
    with io.open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ts15 = []; cl15 = []
    for r in rows:
        try:
            ts15.append(int(datetime.fromisoformat(r["ts"]).timestamp()*1000)); cl15.append(float(r["close"]))
        except Exception:
            continue
    if len(cl15) > WIN_1H*4 + 5:
        K15[sym] = (np.array(ts15), np.array(cl15))
        K[sym] = (np.array(ts15[::4]), np.array(cl15[::4]))
print(f"series loaded: {len(K)} watchlist syms")


def logrets_before(sym, cutoff):
    d = K.get(sym)
    if d is None: return None
    ts, cl = d; j = int(np.searchsorted(ts, cutoff))
    if j < WIN_1H + 1: return None
    seg = cl[j-WIN_1H-1:j]
    if len(seg) < 20 or np.any(seg <= 0): return None
    return list(np.diff(np.log(seg)))


def dec_scores_at(cutoff):
    rmap = {s: r for s in K if (r := logrets_before(s, cutoff))}
    return scores_from_rets(rmap, vol_q=VOLQ, corr_max=CORRMAX)


def squeeze_at(sym, cutoff):
    """True if current BB-width is in the bottom BB_PCTILE_MAX of its recent range."""
    d = K15.get(sym)
    if d is None: return None
    ts, cl = d; j = int(np.searchsorted(ts, cutoff))
    if j < BB_LOOK + 21: return None
    widths = []
    for t in range(j-BB_LOOK, j):
        w = cl[t-20:t]
        m = w.mean()
        if m > 0: widths.append(w.std()/m)
    if len(widths) < 20: return None
    cur = widths[-1]; rank = sum(1 for x in widths if x <= cur)/len(widths)
    return rank <= BB_PCTILE_MAX


# top-20 + silent-miss (reuse funnel logic)
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    top.add((datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d"), e.get("symbol")))
scanned = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    sym = e.get("sym") or e.get("symbol")
    try: d = datetime.fromisoformat(e.get("ts", "").replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except: continue
    if sym in WL: scanned.add((d, sym))
silent = sorted(k for k in top if k not in scanned)


def cut(d): return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)


def evalg(pairs):
    us = dflag = sflag = uflag = 0
    for d, sym in pairs:
        c = cut(d); sc = dec_scores_at(c).get(sym); sq = squeeze_at(sym, c)
        if sc is None or sq is None: continue
        us += 1
        df = bool(sc["flag"]); dflag += df; sflag += sq; uflag += (df or sq)
    return us, dflag, sflag, uflag


us_s, d_s, sq_s, u_s = evalg(silent)
random.seed(1); days = sorted({d for d, _ in top}); base = []
for _ in range(400):
    dd = random.choice(days); ss = random.choice(list(K))
    if (dd, ss) not in top: base.append((dd, ss))
us_b, d_b, sq_b, u_b = evalg(base)
print("=" * 64)
print(f"{'group':<20}{'n':>5}{'decoup%':>9}{'squeeze%':>10}{'UNION%':>8}")
for name, us, df, sf, uf in [("silent-miss", us_s, d_s, sq_s, u_s),
                             ("baseline", us_b, d_b, sq_b, u_b)]:
    if us:
        print(f"{name:<20}{us:>5}{100*df/us:>8.0f}%{100*sf/us:>9.0f}%{100*uf/us:>7.0f}%")
print("=" * 64)
print("UNION% > decoup% on silent-miss => squeeze is complementary (adds recall);")
print("promotion rule should be decoupling OR squeeze. If squeeze ~ baseline, drop it.")
