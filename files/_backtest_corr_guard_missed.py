"""Validate (max period) the dead correlation entry-guard: reconstruct the
bot's open-position set over time and count how often check_entry SHOULD have
blocked an entry (candidate correlated rho>=threshold with >= CORR_MAX_PER_CLUSTER
already-open positions) but didn't — because the matrix never includes the
candidate, so the guard is a structural no-op (0 blocks all-time).

Method (read-only): walk bot_events entry/exit chronologically to maintain the
open set; for each entry compute Pearson rho (log-returns over a 48h window,
matching CORR_GUARD_WINDOW_BARS*tf) between the candidate and each open position
from history klines; flag if the cluster (rho>=threshold) already has
>= max_cluster members. ASCII-only.
    pyembed\python.exe files\_backtest_corr_guard_missed.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
WINDOW = timedelta(hours=48)        # CORR_GUARD: 1h * 48 bars
try:
    import config as _c
    THRESH = float(getattr(_c, "CORR_GUARD_THRESHOLD", 0.65))
    MAXCL = int(getattr(_c, "CORR_MAX_PER_CLUSTER", 2))
except Exception:
    THRESH, MAXCL = 0.65, 2
CUT = datetime.now(timezone.utc) - timedelta(days=30)

WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
_K = {}
for s in WL:
    p = HIST / f"{s}_15m.csv"
    if not p.exists():
        continue
    ts, cl = [], []
    with io.open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts.append(datetime.fromisoformat(r["ts"]).timestamp()); cl.append(float(r["close"]))
            except Exception:
                continue
    if len(cl) > 50:
        _K[s] = (np.array(ts), np.array(cl))


def logret_window(sym, t_end):
    d = _K.get(sym)
    if d is None:
        return None
    ts, cl = d
    t1 = t_end.timestamp(); t0 = (t_end - WINDOW).timestamp()
    m = (ts >= t0) & (ts <= t1)
    c = cl[m]
    if len(c) < 20:
        return None
    return np.diff(np.log(c))


def rho(a, b):
    n = min(len(a), len(b))
    if n < 20:
        return None
    a, b = a[-n:], b[-n:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# walk events
events = []
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln:
        continue
    try: e = json.loads(ln)
    except: continue
    if e.get("event") not in ("entry", "exit"):
        continue
    sym = e.get("sym") or e.get("symbol") or ""
    try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
    except: continue
    events.append((dt, e.get("event"), sym))
events.sort()

open_set = {}     # sym -> entry_dt
n_entries = 0; n_eval = 0; should_block = 0; cluster_sizes = []
examples = []
for dt, et, sym in events:
    if et == "exit":
        open_set.pop(sym, None)
        continue
    # entry
    if dt >= CUT and sym in _K:
        n_entries += 1
        cand = logret_window(sym, dt)
        opens = [o for o in open_set if o != sym]
        if cand is not None and opens:
            n_eval += 1
            cluster = []
            for o in opens:
                ow = logret_window(o, dt)
                if ow is None:
                    continue
                r = rho(cand, ow)
                if r is not None and r >= THRESH:
                    cluster.append((o, r))
            if len(cluster) >= MAXCL:
                should_block += 1
                cluster_sizes.append(len(cluster))
                if len(examples) < 8:
                    examples.append((str(dt)[:16], sym, len(cluster),
                                     ",".join(f"{o}:{r:.2f}" for o, r in cluster[:3])))
    open_set[sym] = dt

print("=" * 70)
print(f"Correlation entry-guard MISSED-block validation (30d, thr={THRESH}, max_cluster={MAXCL})")
print("=" * 70)
print(f"klines loaded for {len(_K)}/{len(WL)} watchlist syms")
print(f"entries in window: {n_entries}  evaluable (had open peers + klines): {n_eval}")
print(f"entries the guard SHOULD have blocked: {should_block} "
      f"({100*should_block/max(1,n_eval):.0f}% of evaluable)")
if cluster_sizes:
    print(f"  median cluster size at those entries: {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
print("\nexamples (time, candidate, cluster_size, top peers rho):")
for x in examples:
    print("  ", x)
print("\nEach 'should-block' = an entry that joined >=2 already-open positions it")
print("was rho>=thr correlated with -> the dead guard let the cluster grow, so a")
print("BTC dump hits all of them at once. Bear-day caveat: live threshold is 0.99")
print("(guard off) on down days — the worst case is also the least protected.")
