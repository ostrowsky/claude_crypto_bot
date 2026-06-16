"""Exit-policy backtest for impulse_speed entries (the EX1 capture leak).

Morning-report finding (2026-06-01): top-20 capture is held down by
`atr_trail` prematurely stopping impulse_speed winners — they get knocked
out at a small/large LOSS on a deep retrace, then the coin runs +200..+470%.
`time_max_hold` exits captured ~20x more (EX1 +0.044 vs atr_trail -0.002).

This replays the price path AFTER each real impulse_speed entry (klines from
history/<sym>_15m.csv, backfill 35d) under several exit policies and asks:
  Does a wider stop / time-hold / partial+runner raise WINNER capture
  WITHOUT blowing up the LOSERS? (At entry the bot can't tell them apart, so
  the honest test is net per-trade pnl across ALL impulse_speed entries.)

Policies (long-only, pnl% = (exit-entry)/entry*100):
  ACTUAL      - what the bot really realised (from bot_events)
  TRAIL_1.5/3/5/8 - trailing stop, fixed % buffer from running peak
  HOLD_24H/48H    - ignore trail, exit at close ~24/48h after entry
  PARTIAL     - 50% off at +25% (lock), 50% runner trail 8% to horizon

Read-only. Run from repo root:
    pyembed\python.exe files\_backtest_exit_policy_impulse.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "history"
NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=35)
HORIZON_BARS = 192          # 48h on 15m
BAR_MIN = 15
MODE_FILTER = "impulse_speed"


def _norm_pct(v):
    if v is None: return None
    try: v = float(v)
    except: return None
    return v if abs(v) > 5 else v * 100


# ---- klines cache ---------------------------------------------------------
_KCACHE: dict[str, list[dict]] = {}
def klines(sym: str, tf="15m"):
    if sym in _KCACHE: return _KCACHE[sym]
    path = HISTORY / f"{sym}_{tf}.csv"
    bars = []
    if path.exists():
        with io.open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    bars.append({
                        "ts": datetime.fromisoformat(row["ts"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    })
                except Exception:
                    continue
    _KCACHE[sym] = bars
    return bars


def _fwd(sym, entry_dt):
    """Bars at/after entry, capped to HORIZON_BARS."""
    bars = klines(sym)
    fwd = [b for b in bars if b["ts"] >= entry_dt]
    return fwd[:HORIZON_BARS]


# ---- exit policies: return realised pnl% given entry price + forward bars --
def p_trail(entry, fwd, buf):
    if not fwd: return None
    peak = entry; stop = peak * (1 - buf)
    for b in fwd:
        if b["low"] <= stop:
            return (stop - entry) / entry * 100
        if b["high"] > peak:
            peak = b["high"]; stop = peak * (1 - buf)
    return (fwd[-1]["close"] - entry) / entry * 100


def p_hold(entry, fwd, hours):
    if not fwd: return None
    n = min(len(fwd) - 1, int(hours * 60 / BAR_MIN))
    return (fwd[n]["close"] - entry) / entry * 100


def p_partial(entry, fwd, take_at=0.25, runner_buf=0.08):
    """50% off at +take_at, 50% runner trail runner_buf to horizon."""
    if not fwd: return None
    target = entry * (1 + take_at)
    half1 = None
    peak = entry; stop = peak * (1 - runner_buf)
    for b in fwd:
        # runner stop check first (conservative)
        if b["low"] <= stop and half1 is None:
            # both halves never reached target -> whole exits at stop
            return (stop - entry) / entry * 100
        if half1 is None and b["high"] >= target:
            half1 = (target - entry) / entry * 100
        if half1 is not None and b["low"] <= stop:
            r = (stop - entry) / entry * 100
            return 0.5 * half1 + 0.5 * r
        if b["high"] > peak:
            peak = b["high"]; stop = peak * (1 - runner_buf)
    last = (fwd[-1]["close"] - entry) / entry * 100
    if half1 is None:
        return last
    return 0.5 * half1 + 0.5 * last


POLICIES = {
    "TRAIL_1.5%": lambda e, f, a: p_trail(e, f, 0.015),
    "TRAIL_3%":   lambda e, f, a: p_trail(e, f, 0.03),
    "TRAIL_5%":   lambda e, f, a: p_trail(e, f, 0.05),
    "TRAIL_8%":   lambda e, f, a: p_trail(e, f, 0.08),
    "HOLD_24H":   lambda e, f, a: p_hold(e, f, 24),
    "HOLD_48H":   lambda e, f, a: p_hold(e, f, 48),
    "PARTIAL_25/8": lambda e, f, a: p_partial(e, f),
}


# ---- top20 set + potential proxy -----------------------------------------
top20 = set(); potential = {}
with io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8") as f:
    for ln in f:
        try: e = json.loads(ln)
        except: continue
        ts_ms = e.get("ts")
        if not ts_ms: continue
        dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        if dt < CUT: continue
        sym = e.get("symbol"); d = dt.strftime("%Y-%m-%d")
        if e.get("label_top20") == 1: top20.add((d, sym))
        feat = e.get("features") or {}
        cands = []
        eod = _norm_pct(e.get("eod_return_pct"))
        if eod is not None: cands.append(eod)
        for k in ("tg_return_4h", "tg_return_since_open", "tg_return_1h"):
            v = _norm_pct(feat.get(k))
            if v is not None: cands.append(v)
        if cands:
            cur = potential.get((d, sym))
            potential[(d, sym)] = max(cur, max(cands)) if cur is not None else max(cands)


# ---- pair impulse_speed entries with their real exit ----------------------
open_t = {}; trades = []
with io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if '"event"' not in ln: continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event", "")
        if ev not in ("entry", "exit"): continue
        try: dt = datetime.fromisoformat((e.get("ts","")).replace("Z","+00:00"))
        except: continue
        if dt < CUT: continue
        sym = e.get("sym") or e.get("symbol") or ""
        if not sym: continue
        if ev == "entry":
            open_t[sym] = {"d": dt.strftime("%Y-%m-%d"),
                           "price": float(e.get("price") or e.get("entry_price") or 0),
                           "mode": e.get("mode","?"), "tf": e.get("tf","?"),
                           "dt": dt}
        else:
            ent = open_t.pop(sym, None)
            if not ent: continue
            ex_p = float(e.get("exit_price") or e.get("price") or 0)
            if ent["price"] <= 0 or ex_p <= 0: continue
            if ent["mode"] != MODE_FILTER: continue
            trades.append({
                "d": ent["d"], "sym": sym, "tf": ent["tf"], "dt": ent["dt"],
                "entry": ent["price"],
                "actual_pnl": (ex_p - ent["price"]) / ent["price"] * 100,
                "is_top20": (ent["d"], sym) in top20,
                "potential": potential.get((ent["d"], sym)),
            })


# ---- simulate -------------------------------------------------------------
sim = []        # rows with all policy pnls
no_klines = 0
for t in trades:
    fwd = _fwd(t["sym"], t["dt"])
    if not fwd:
        no_klines += 1
        continue
    row = {"is_top20": t["is_top20"], "potential": t["potential"],
           "ACTUAL": t["actual_pnl"]}
    for name, fn in POLICIES.items():
        row[name] = fn(t["entry"], fwd, None)
    sim.append(row)


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals: return None
    n = len(vals); vals_s = sorted(vals)
    mean = sum(vals)/n; med = vals_s[n//2]
    win = sum(1 for v in vals if v > 0)/n*100
    return {"n": n, "mean": mean, "median": med, "win": win}


def cap(rows, key):
    """mean capture = pnl/potential over winners with potential>0."""
    cs = []
    for r in rows:
        p = r.get("potential"); v = r.get(key)
        if p and p > 0 and v is not None:
            cs.append(max(-0.5, min(1.5, v / p)))
    return sum(cs)/len(cs) if cs else None


winners = [r for r in sim if r["is_top20"]]
losers  = [r for r in sim if not r["is_top20"]]

print("="*78)
print(f"Exit-policy backtest — mode={MODE_FILTER}, 35d, horizon={HORIZON_BARS}b/48h")
print("="*78)
print(f"impulse_speed exits paired: {len(trades)}  simulated (klines found): "
      f"{len(sim)}  (skipped no-klines: {no_klines})")
print(f"  winners(top20)={len(winners)}  losers(non-top20)={len(losers)}")
print()
cols = ["ACTUAL"] + list(POLICIES.keys())
hdr = f"{'policy':<13}" + "".join(f"{c:>9}" for c in
      ["W_mean","W_win%","W_cap","L_mean","L_win%","ALL_mean"])
print(hdr); print("-"*len(hdr))
for c in cols:
    w = agg(winners, c); l = agg(losers, c); a = agg(sim, c); wc = cap(winners, c)
    def f(x, p="{:+.2f}"): return p.format(x) if x is not None else "  -"
    print(f"{c:<13}"
          f"{f(w['mean']) if w else '  -':>9}"
          f"{f(w['win'],'{:.0f}') if w else '  -':>9}"
          f"{f(wc,'{:+.3f}') if wc is not None else '  -':>9}"
          f"{f(l['mean']) if l else '  -':>9}"
          f"{f(l['win'],'{:.0f}') if l else '  -':>9}"
          f"{f(a['mean']) if a else '  -':>9}")

print()
print("Read: W_*=winners(top20), L_*=losers(non-top20), ALL_mean=net per-trade")
print("over EVERY impulse_speed entry (what the bot actually faces at decision).")
print("A policy is worth deploying only if W_cap/W_mean rises AND ALL_mean does")
print("NOT fall vs ACTUAL — i.e. winner upside not paid for by loser blow-ups.")
