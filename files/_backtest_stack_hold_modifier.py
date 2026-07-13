"""POL-type exit modifier: hold longer (wider trail + longer max-hold) ONLY when
the coin has a confirmed BULLISH 1d STACK at entry (close > MA7 > MA25 on daily
closes, computed through the PREVIOUS day — no lookahead). The unconditional
version of this lever was OOS-null (_backtest_premature_trail.py); hypothesis:
conditioning on daily structure (a fixed rule, not a trained proba — no leakage)
selects the subgroup where holding pays (multi-day grinds like POL) and leaves
everything else on the current tight policy.

Replay real TAKE entries over 15m klines (rolling ~31d window): per entry,
simulate trail k=2.0/48bars (current proxy) vs k=4.0/96bars (hold variant),
grouped by stack-at-entry. Decision rule to evaluate: WIDE if stack else CURRENT.
  pyembed\python.exe files\_backtest_stack_hold_modifier.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
K_CUR, H_CUR = 2.0, 48
K_WIDE, H_WIDE = 4.0, 96

# ---- daily closes (420d cache) -> per-sym {date: (ma7, ma25, close)} ----
daily = json.load(io.open(ROOT/"files"/"_daily_closes_cache.json", encoding="utf-8"))
dstack = {}
for sym, rows in daily.items():
    dates = [datetime.fromtimestamp(r[0]/1000, tz=timezone.utc).strftime("%Y-%m-%d") for r in rows]
    c = np.array([r[1] for r in rows])
    m = {}
    for i in range(25, len(c)):
        ma7 = float(np.mean(c[i-6:i+1])); ma25 = float(np.mean(c[i-24:i+1]))
        m[dates[i]] = bool(c[i] > ma7 > ma25)
    dstack[sym] = m


def stack_at_entry(sym, entry_dt):
    """Stack condition using the last COMPLETED day before the entry (no lookahead)."""
    d = (entry_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return dstack.get(sym, {}).get(d)


# ---- 15m klines ----
K = {}
for p in HIST.glob("*_15m.csv"):
    sym = p.name[:-8]
    ts = []; hi = []; lo = []; cl = []
    with io.open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                ts.append(int(datetime.fromisoformat(r["ts"]).timestamp()*1000))
                hi.append(float(r["high"])); lo.append(float(r["low"])); cl.append(float(r["close"]))
            except Exception:
                continue
    if len(cl) > 100:
        K[sym] = (np.array(ts), np.array(hi), np.array(lo), np.array(cl))


def replay(sym, bar_ts, atr_pct, k, maxh):
    d = K.get(sym)
    if d is None or not atr_pct or atr_pct <= 0:
        return None
    ts, hi, lo, cl = d
    j = int(np.searchsorted(ts, bar_ts))
    if j >= len(cl) - 4 or j < 1:
        return None
    entry = cl[j]
    if entry <= 0:
        return None
    buf = atr_pct/100.0 * k
    peak = entry; end = min(j + maxh, len(cl) - 1); realized = None
    for t in range(j+1, end+1):
        peak = max(peak, hi[t])
        if lo[t] <= peak * (1 - buf):
            realized = (peak*(1-buf) - entry)/entry*100.0
            break
    if realized is None:
        realized = (cl[end] - entry)/entry*100.0
    pot = (float(np.max(hi[j+1:end+1])) - entry)/entry*100.0
    return realized, pot


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


rows = []
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"take"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    if (e.get("decision", {}) or {}).get("action") != "take": continue
    sym = e.get("sym"); bar_ts = e.get("bar_ts")
    atr = _f((e.get("f", {}) or {}).get("atr_pct"))
    if not sym or not bar_ts or atr is None: continue
    entry_dt = datetime.fromtimestamp(bar_ts/1000, tz=timezone.utc)
    st = stack_at_entry(sym, entry_dt)
    if st is None: continue
    cur = replay(sym, bar_ts, atr, K_CUR, H_CUR)
    wide = replay(sym, bar_ts, atr, K_WIDE, H_WIDE)
    if cur is None or wide is None: continue
    rows.append((bool(st), cur[0], wide[0], cur[1], wide[1]))

n = len(rows)
st_rows = [r for r in rows if r[0]]; no_rows = [r for r in rows if not r[0]]
print(f"replayed takes={n}  stack-at-entry: {len(st_rows)} ({100*len(st_rows)/max(1,n):.0f}%)  "
      f"no-stack: {len(no_rows)}")
print(f"policy: CURRENT k={K_CUR}/{H_CUR}b   WIDE k={K_WIDE}/{H_WIDE}b")
print("-" * 68)
print(f"{'group':<12}{'n':>6}{'cur_real%':>11}{'wide_real%':>12}{'delta':>8}{'cur_win%':>9}{'wide_win%':>10}")
for name, g in [("STACK", st_rows), ("no-stack", no_rows)]:
    if not g: continue
    cr = np.mean([r[1] for r in g]); wr = np.mean([r[2] for r in g])
    cw = 100*np.mean([r[1] > 0 for r in g]); ww = 100*np.mean([r[2] > 0 for r in g])
    print(f"{name:<12}{len(g):>6}{cr:>+11.3f}{wr:>+12.3f}{wr-cr:>+8.3f}{cw:>9.0f}{ww:>10.0f}")
print("-" * 68)
base = np.mean([r[1] for r in rows])
mixed = np.mean([(r[2] if r[0] else r[1]) for r in rows])
print(f"ALL-CURRENT baseline: {base:+.3f}%   RULE (wide if stack else current): {mixed:+.3f}%"
      f"   delta {mixed-base:+.3f}%")
print("\nRULE is worth wiring only if delta > 0 driven by the STACK group (wide")
print("beats current specifically there); if wide loses in-stack too, the POL-type")
print("hold lever is refuted even in its narrow conditioned form.")

# VERDICT (2026-07-13, 523 replayed takes, ~31d klines window):
# REFUTED — even the narrow conditioned form. In-stack entries (n=48): wide trail
# makes it WORSE (-0.504% -> -1.171%/trade); rule overall -0.061% vs baseline.
# Deep retraces inside daily uptrends get fully caught by the wide stop; POL-type
# multi-week runners are too rare to pay for it. Side-finding: entries INTO an
# already-confirmed 1d stack underperform (-0.50% vs +0.16%, win 23% vs 38%) —
# confirmed daily uptrend at entry = late chase (consistent with the refuted 1d
# alert: fresh daily stacks mean-revert). Do not wire the hold modifier.
