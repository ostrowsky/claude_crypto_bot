"""H1 validation: would DECOUPLING-based scan promotion catch the silent-miss
top-20 earlier? Silent misses (top-20 winners the bot logged NO event for, ~23%)
are the biggest earliness lever and the auto-pipeline is blind to them.

decoupling_score is logged only on entry events, so silent misses have none —
recompute it directly from klines via decoupling_signal.scores_from_rets over the
DECOUPLING_WINDOW (7d 1h) ending BEFORE the move day (no lookahead). Then ask:
what fraction of silent-miss top-20 would have flagged (=> been promoted to scan
earlier)? Compare vs entered top-20 and a baseline flag rate (scan-load proxy).

H1 holds if silent-miss flag-recall is high AND baseline flag rate stays a
manageable scan set. Read-only, max period.
  pyembed\python.exe files\_backtest_h1_decoupling_promotion.py
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
H = 3600_000

# ---- 1h closes per WL sym from 15m klines ----
K = {}
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]
    if sym not in WL:
        continue
    ts = []; cl = []
    with io.open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows[::4]:
        try:
            ts.append(int(datetime.fromisoformat(r["ts"]).timestamp()*1000)); cl.append(float(r["close"]))
        except Exception:
            continue
    if len(cl) > WIN_1H + 5:
        K[sym] = (np.array(ts), np.array(cl))
print(f"1h series loaded: {len(K)} watchlist syms")


def logrets_before(sym, cutoff_ms):
    d = K.get(sym)
    if d is None:
        return None
    ts, cl = d
    j = int(np.searchsorted(ts, cutoff_ms))
    if j < WIN_1H + 1:
        return None
    seg = cl[j-WIN_1H-1:j]
    if len(seg) < 20 or np.any(seg <= 0):
        return None
    return list(np.diff(np.log(seg)))


def scores_at(cutoff_ms):
    rmap = {}
    for s in K:
        r = logrets_before(s, cutoff_ms)
        if r:
            rmap[s] = r
    return scores_from_rets(rmap, vol_q=VOLQ, corr_max=CORRMAX)


# ---- top-20 winners + scanned/entered sets ----
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    top.add((d, e.get("symbol")))

scanned = set(); entered = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    sym = e.get("sym") or e.get("symbol")
    try: d = datetime.fromisoformat(e.get("ts", "").replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except: continue
    if sym in WL:
        scanned.add((d, sym))
        if e.get("event") == "entry":
            entered.add((d, sym))

silent = sorted(k for k in top if k not in scanned)
ent = sorted(k for k in top if k in entered)
print(f"top-20 total={len(top)}  silent-miss={len(silent)}  entered={len(ent)}")


def day_cut(d):
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()*1000)


def eval_group(pairs):
    flagged = 0; scores = []; usable = 0
    for d, sym in pairs:
        sc = scores_at(day_cut(d)).get(sym)
        if sc is None:
            continue
        usable += 1
        scores.append(sc["decoupling_score"])
        if sc["flag"]:
            flagged += 1
    return usable, flagged, (np.mean(scores) if scores else 0.0)


us_s, fl_s, sc_s = eval_group(silent)
us_e, fl_e, sc_e = eval_group(ent)

# baseline: random (day, WL-sym) non-top20 pairs
random.seed(1)
all_days = sorted({d for d, _ in top})
base_pairs = []
for _ in range(400):
    d = random.choice(all_days); s = random.choice(list(K))
    if (d, s) not in top:
        base_pairs.append((d, s))
us_b, fl_b, sc_b = eval_group(base_pairs)

print("=" * 64)
print(f"{'group':<22}{'n_usable':>9}{'flag%':>8}{'mean_score':>12}")
print(f"{'silent-miss top-20':<22}{us_s:>9}{100*fl_s/max(1,us_s):>7.0f}%{sc_s:>12.3f}")
print(f"{'entered top-20':<22}{us_e:>9}{100*fl_e/max(1,us_e):>7.0f}%{sc_e:>12.3f}")
print(f"{'baseline non-top20':<22}{us_b:>9}{100*fl_b/max(1,us_b):>7.0f}%{sc_b:>12.3f}")
print("=" * 64)
lift = (fl_s/max(1,us_s)) / max(0.01, fl_b/max(1,us_b))
print(f"silent-miss flag-recall = {100*fl_s/max(1,us_s):.0f}%  (baseline {100*fl_b/max(1,us_b):.0f}%, lift x{lift:.2f})")
print("H1 holds if silent-miss flag-recall is meaningfully > baseline (promotion")
print("would catch them earlier) at a baseline flag rate that keeps scan-load sane.")
