"""
Check correctness of today's ALIGNMENT signals (KSM, HBAR, BTC) and reassess
the mode's historical usefulness.

Steps:
  1. Scan bot_events.jsonl for today's alignment entries on the 3 symbols.
  2. Find their matching exits (if any) to compute actual PnL.
  3. Fetch current Binance price to compute MTM for still-open positions.
  4. Aggregate alignment-mode historical stats:
     - per-tf win%, avg pnl, median pnl, bars_held distribution
     - compare against other modes on same tf
     - look at failure modes (how often does it exit red, at what bar)
  5. Specifically check: signals fired near local top? (RSI at entry vs peak,
     bars from 20-bar high, upper-wick last-2-bars share)
"""
from __future__ import annotations
import json, io, sys, urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

TODAY = "2026-04-19"
TARGETS = {"KSMUSDT", "HBARUSDT", "BTCUSDT"}

# ── 1. Scan today's events ────────────────────────────────────────────────
print(f"=== 1) TODAY'S ALIGNMENT SIGNALS ({TODAY}) ===")
todays_entries = []
todays_exits = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        ts = e.get("ts", "")
        if not ts.startswith(TODAY): continue
        sym = e.get("sym", "")
        if sym not in TARGETS: continue
        et = e.get("event")
        if et == "entry" and e.get("mode") == "alignment":
            todays_entries.append(e)
        elif et == "exit":
            todays_exits.append(e)

print(f"Found {len(todays_entries)} alignment entries, {len(todays_exits)} exits on targets today\n")
for e in todays_entries:
    print(f"  ENTRY {e.get('ts')}  {e.get('sym')}  tf={e.get('tf')}  "
          f"price={e.get('price')}  mode={e.get('mode')}")
for x in todays_exits:
    print(f"  EXIT  {x.get('ts')}  {x.get('sym')}  pnl={x.get('pnl_pct')}  "
          f"bars={x.get('bars_held')}  reason={x.get('reason')}")

# ── 2. Fetch current Binance prices ───────────────────────────────────────
print("\n=== 2) CURRENT MTM (Binance spot price) ===")
def fetch_price(sym):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
        with urllib.request.urlopen(url, timeout=6) as r:
            return float(json.load(r)["price"])
    except Exception as e:
        return None

entry_prices = {}
for e in todays_entries:
    sym = e.get("sym")
    entry_prices.setdefault(sym, []).append(float(e.get("price", 0)))

for sym, prices in entry_prices.items():
    cur = fetch_price(sym)
    if cur is None:
        print(f"  {sym}: price fetch FAILED")
        continue
    for ep in prices:
        pct = (cur - ep) / ep * 100
        verdict = "✓ PROFIT" if pct > 0 else "✗ LOSS  "
        print(f"  {sym}: entry={ep:.6g}  now={cur:.6g}  {verdict} {pct:+.2f}%")

# ── 3. Historical alignment stats (pair entries↔exits) ────────────────────
print("\n=== 3) HISTORICAL ALIGNMENT PERFORMANCE ===")
entries_all = []
exits_all = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        et = e.get("event")
        if et == "entry": entries_all.append(e)
        elif et == "exit": exits_all.append(e)

entries_all.sort(key=lambda e: e.get("ts", ""))
exits_all.sort(key=lambda e: e.get("ts", ""))

ent_q = defaultdict(list)
for e in entries_all: ent_q[e.get("sym", "")].append(e)

trades = []
for x in exits_all:
    sym = x.get("sym", "")
    q = ent_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    trades.append({
        "sym": sym,
        "mode": ent.get("mode"),
        "tf": ent.get("tf"),
        "pnl": float(pnl),
        "bars": x.get("bars_held") or 0,
        "reason": x.get("reason", ""),
        "entry_ts": ent.get("ts", ""),
        "rsi": (ent.get("features") or {}).get("rsi"),
        "adx": (ent.get("features") or {}).get("adx"),
        "slope_pct": (ent.get("features") or {}).get("slope_pct"),
    })
print(f"Paired {len(trades)} total trades")

def agg(lst, name):
    if not lst:
        print(f"  {name:45s} n=0"); return
    n = len(lst)
    pnls = [t["pnl"] for t in lst]
    wins = sum(1 for p in pnls if p > 0)
    tot = sum(pnls); avg = tot/n
    med = sorted(pnls)[n//2]
    losers = [p for p in pnls if p <= 0]
    winners = [p for p in pnls if p > 0]
    avg_w = sum(winners)/len(winners) if winners else 0
    avg_l = sum(losers)/len(losers) if losers else 0
    print(f"  {name:45s} n={n:>4d}  win={wins/n*100:>5.1f}%  avg={avg:+.3f}%  "
          f"med={med:+.2f}%  sum={tot:+7.1f}%  W̄={avg_w:+.2f}  L̄={avg_l:+.2f}")

print("\nBy (mode, tf):")
groups = defaultdict(list)
for t in trades:
    groups[(t["mode"], t["tf"])].append(t)
for key in sorted(groups.keys(), key=lambda k: -len(groups[k])):
    mode, tf = key
    agg(groups[key], f"{mode}|{tf}")

# Alignment-focused deep dive
print("\n=== 4) ALIGNMENT DEEP DIVE ===")
align = [t for t in trades if t["mode"] == "alignment"]
print(f"Total alignment trades: {len(align)}")

for tf in ["15m", "1h"]:
    sub = [t for t in align if t["tf"] == tf]
    if not sub: continue
    print(f"\n  --- alignment|{tf} (n={len(sub)}) ---")
    agg(sub, "overall")

    # Bars-held distribution
    bars = Counter()
    for t in sub:
        b = t["bars"]
        if b <= 2: bars["≤2"] += 1
        elif b <= 5: bars["3-5"] += 1
        elif b <= 10: bars["6-10"] += 1
        elif b <= 20: bars["11-20"] += 1
        else: bars[">20"] += 1
    print(f"    bars_held dist: {dict(bars)}")

    # Reason for exit
    reasons = Counter(t["reason"] for t in sub)
    print(f"    exit reasons:  {dict(reasons.most_common(6))}")

    # Early-bar losers (bars ≤ 3): share and avg
    early_losers = [t for t in sub if t["bars"] <= 3 and t["pnl"] < 0]
    print(f"    early losers (bars≤3 & pnl<0): {len(early_losers)}/{len(sub)} "
          f"({len(early_losers)/len(sub)*100:.1f}%)")
    if early_losers:
        print(f"      their avg pnl: {sum(t['pnl'] for t in early_losers)/len(early_losers):+.2f}%")

    # RSI-at-entry bucketing (is signal firing at RSI peaks?)
    rsi_buckets = {"<50": [], "50-60": [], "60-70": [], "70-80": [], ">80": []}
    for t in sub:
        r = t.get("rsi")
        if r is None: continue
        if r < 50: rsi_buckets["<50"].append(t)
        elif r < 60: rsi_buckets["50-60"].append(t)
        elif r < 70: rsi_buckets["60-70"].append(t)
        elif r < 80: rsi_buckets["70-80"].append(t)
        else: rsi_buckets[">80"].append(t)
    print(f"    RSI@entry buckets:")
    for k, v in rsi_buckets.items():
        if not v: continue
        pnls = [t["pnl"] for t in v]
        wr = sum(1 for p in pnls if p>0)/len(v)*100
        print(f"      RSI {k:>6s}  n={len(v):>3d}  win={wr:>5.1f}%  avg={sum(pnls)/len(v):+.3f}%")

    # Last-7-day slice
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    recent = [t for t in sub if t["entry_ts"] >= cutoff]
    if recent:
        agg(recent, f"last 7d (>={cutoff})")

    # Last-24h slice
    cutoff24 = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")
    rec24 = [t for t in sub if t["entry_ts"] >= cutoff24]
    if rec24:
        agg(rec24, f"last 24h")

# Cross-reference: what % of alignment signals hit stop at the very first
# pullback (bars_held ≤ 4 and pnl in [-3, -1]% — exactly the pattern of
# "signal fired at local top")
print("\n=== 5) 'FIRED AT LOCAL TOP' PATTERN ===")
for tf in ["15m", "1h"]:
    sub = [t for t in align if t["tf"] == tf]
    if not sub: continue
    top_pat = [t for t in sub if t["bars"] <= 4 and -5 <= t["pnl"] <= -0.5]
    if sub:
        print(f"  {tf}: {len(top_pat)}/{len(sub)} ({len(top_pat)/len(sub)*100:.1f}%) "
              f"match 'fired-at-top' pattern (bars≤4, pnl in [-5,-0.5]%)")
        if top_pat:
            avg_b = sum(t["bars"] for t in top_pat)/len(top_pat)
            avg_p = sum(t["pnl"] for t in top_pat)/len(top_pat)
            print(f"    their avg bars={avg_b:.1f}, avg pnl={avg_p:+.2f}%")
