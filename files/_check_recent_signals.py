"""
Diagnose recent entries (last 7 days):
  - Which were 'single-pump-bar' entries? (vol_x >= 10)
  - Their ML metrics at entry time (from candidate ranker outputs if stored)
  - Their outcomes (paired exit PnL, or current MTM)
  - Test proposed gate: BLOCK if vol_x >= VX_THR AND ML-signals weak
    where ML-weak = (rank < 0 OR ev < 0 OR tg_proba < 0.30)

Goals:
  1. Confirm BNT-like trades get blocked (no benefit, big drawdown risk).
  2. Confirm DOGS-like trades are KEPT (positive ML + pump = real winner).
  3. Quantify: how many recent entries fall into each bucket, avg PnL.
"""
from __future__ import annotations
import json, io, sys, math, urllib.request
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

CUTOFF_DAYS = 7
cutoff = (datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)).strftime("%Y-%m-%d")

# ── Load entries+exits ──────────────────────────────────────────────────
entries, exits_ = [], []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        et = e.get("event")
        if et == "entry": entries.append(e)
        elif et == "exit": exits_.append(e)
entries.sort(key=lambda e: e.get("ts", ""))
exits_.sort(key=lambda e: e.get("ts", ""))

# Peek one entry to see ALL keys/features (to find rank/EV/TG storage)
print("=== ENTRY EVENT SCHEMA (all keys seen in last 50 recent entries) ===")
recent = [e for e in entries if e.get("ts", "") >= cutoff]
all_keys = set()
for e in recent[-50:]: all_keys |= set(e.keys())
print(f"Seen keys: {sorted(all_keys)}")
# Sample one with vol_x:
for e in recent[-50:]:
    if (e.get("vol_x") or 0) >= 5:
        print(f"\nSample high-vol entry:\n{json.dumps(e, indent=2, ensure_ascii=False)[:2000]}")
        break

# Pair trades
ent_q = defaultdict(list)
for e in entries: ent_q[e.get("sym", "")].append(e)
trades = []
for x in exits_:
    sym = x.get("sym", "")
    q = ent_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    trades.append({
        "sym": sym, "tf": ent.get("tf"), "mode": ent.get("mode"),
        "vol_x": float(ent.get("vol_x") or 0),
        "rsi": ent.get("rsi"), "adx": ent.get("adx"),
        "slope_pct": ent.get("slope_pct"), "macd_hist": ent.get("macd_hist"),
        "score": ent.get("score"), "rank": ent.get("rank"),
        "ev": ent.get("ev"), "tg_proba": ent.get("tg_proba"),
        "quality": ent.get("quality"),
        "pnl": float(pnl) if pnl is not None else None,
        "bars": x.get("bars_held") or 0,
        "reason": x.get("reason", ""),
        "entry_ts": ent.get("ts", ""), "exit_ts": x.get("ts", ""),
        "entry_price": float(ent.get("price") or 0),
        "closed": True,
    })

# Remaining open entries (no matched exit)
open_entries = []
for sym, q in ent_q.items():
    for ent in q:
        if ent.get("ts", "") >= cutoff:
            open_entries.append({
                "sym": sym, "tf": ent.get("tf"), "mode": ent.get("mode"),
                "vol_x": float(ent.get("vol_x") or 0),
                "score": ent.get("score"), "rank": ent.get("rank"),
                "ev": ent.get("ev"), "tg_proba": ent.get("tg_proba"),
                "entry_ts": ent.get("ts", ""),
                "entry_price": float(ent.get("price") or 0),
                "closed": False,
            })

# ── Recent 7-day trades ────────────────────────────────────────────────
recent_closed = [t for t in trades if t["entry_ts"] >= cutoff and t["pnl"] is not None]
print(f"\n=== LAST {CUTOFF_DAYS} DAYS ===")
print(f"Closed trades: {len(recent_closed)}  |  Open positions: {len(open_entries)}")

def vol_bucket(v):
    if v is None: return "?"
    if v < 2: return "<2"
    if v < 5: return "2-5"
    if v < 10: return "5-10"
    if v < 20: return "10-20"
    return ">=20"

print("\n=== PnL by vol_x bucket (closed trades last 7d) ===")
by_vol = defaultdict(list)
for t in recent_closed: by_vol[vol_bucket(t["vol_x"])].append(t)
for k in ["<2","2-5","5-10","10-20",">=20","?"]:
    lst = by_vol.get(k, [])
    if not lst: continue
    n = len(lst); pnls = [t["pnl"] for t in lst]
    wins = sum(1 for p in pnls if p>0); avg = sum(pnls)/n
    print(f"  vol_x {k:>6s}  n={n:>3d}  win={wins/n*100:>5.1f}%  avg={avg:+.3f}%  sum={sum(pnls):+6.2f}%")

# ── Fetch current MTM for open positions ──────────────────────────────
print("\n=== OPEN POSITIONS (last 7d) — current MTM ===")
def fetch_price(sym):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
        with urllib.request.urlopen(url, timeout=6) as r:
            return float(json.load(r)["price"])
    except: return None

for o in open_entries:
    cur = fetch_price(o["sym"])
    if cur is None or o["entry_price"] <= 0:
        print(f"  {o['sym']:12s} {o['tf']:>3s} {o['mode']:>14s}  price unavail")
        continue
    pct = (cur - o["entry_price"])/o["entry_price"]*100
    o["mtm_pct"] = pct
    ml_str = ""
    for k in ["score","rank","ev","tg_proba"]:
        v = o.get(k)
        ml_str += f" {k}={v}" if v is not None else ""
    vs = f"vx={o['vol_x']:.1f}" if o["vol_x"] else ""
    print(f"  {o['sym']:12s} {o['tf']:>3s} {o['mode']:>14s}  entry={o['entry_price']:.6g}  now={cur:.6g}  MTM={pct:+6.2f}%  {vs}{ml_str}")

# ── Propose gate: pump-bar + weak ML ──────────────────────────────────
print("\n=== PROPOSED GATE backtest (historical) ===")
print("Rule: BLOCK if vol_x >= VX_THR AND (rank<0 OR ev<0 OR tg_proba<0.30)")
print("      (any missing ML field treated as 'unknown, do not block')\n")

def ml_weak(t):
    r, ev, tg = t.get("rank"), t.get("ev"), t.get("tg_proba")
    # Only block if we have DATA and at least one is clearly bad
    if r is not None and r < 0: return True
    if ev is not None and ev < 0: return True
    if tg is not None and tg < 0.30: return True
    return False

# Use ALL closed trades (not just 7d) if rank/ev/tg are present
with_ml = [t for t in trades if t["pnl"] is not None and
           (t.get("rank") is not None or t.get("ev") is not None or t.get("tg_proba") is not None)]
print(f"Trades with ML metadata stored: {len(with_ml)}  (of {len(trades)} total)")

if not with_ml:
    print("  ⚠️  No rank/ev/tg_proba stored on entry events.")
    print("  ⚠️  Gate cannot be backtested from bot_events.jsonl alone.")
    print("  ⚠️  Need to either:")
    print("      (a) add rank/ev/tg_proba to entry payload going forward, or")
    print("      (b) pull from critic_dataset.jsonl (which has decision metadata)")

# Fall back: bucket by vol_x ranges only
for VX in [5, 10, 15, 20]:
    hi = [t for t in recent_closed if t["vol_x"] >= VX]
    lo = [t for t in recent_closed if t["vol_x"] < VX]
    if not hi: continue
    hi_n = len(hi); hi_avg = sum(t["pnl"] for t in hi)/hi_n
    hi_win = sum(1 for t in hi if t["pnl"]>0)/hi_n*100
    lo_n = len(lo); lo_avg = sum(t["pnl"] for t in lo)/lo_n if lo_n else 0
    lo_win = sum(1 for t in lo if t["pnl"]>0)/lo_n*100 if lo_n else 0
    print(f"  VX>={VX:>2}: high-vol n={hi_n}  win={hi_win:.1f}%  avg={hi_avg:+.3f}%   |   "
          f"rest n={lo_n}  win={lo_win:.1f}%  avg={lo_avg:+.3f}%")

# ── Try critic_dataset.jsonl for richer metadata ──────────────────────
print("\n=== Checking critic_dataset.jsonl for ML metadata ===")
try:
    cd_samples = []
    with io.open(FILES / "critic_dataset.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i > 5000: break
            if not ln.strip(): continue
            try: e = json.loads(ln)
            except: continue
            cd_samples.append(e)
            if len(cd_samples) >= 3: break
    if cd_samples:
        print(f"critic_dataset first sample keys: {list(cd_samples[0].keys())}")
        print(json.dumps(cd_samples[0], indent=2, ensure_ascii=False)[:1500])
except Exception as e:
    print(f"  can't read: {e}")
