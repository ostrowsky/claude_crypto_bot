"""
Fast check: are fast-exit alignment/trend 15m trades concentrated in
low-ATR (low-volatility) entries where trail_stop catches up in 1 bar?

Hypothesis: ATR/price < 0.25% → stop width < 0.5% → normal noise kills the position.

Uses klines API to compute ATR at entry bar for recent fast exits.
Also checks what % of ALL entries would be blocked by ATR threshold.
"""
from __future__ import annotations
import json, io, sys, math, time, urllib.request
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

MODES_CHECK = {"alignment", "trend", "strong_trend"}
TRAIL_K_DEFAULT = 2.0

# ── Load pairs ────────────────────────────────────────────────────────────
ent_q: dict[str, list] = defaultdict(list)
exits_raw: list = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event")
        if ev == "entry": ent_q[e.get("sym","")].append(e)
        elif ev == "exit": exits_raw.append(e)

for k in ent_q: ent_q[k].sort(key=lambda e: e.get("ts",""))
exits_raw.sort(key=lambda e: e.get("ts",""))

trades = []
for x in exits_raw:
    sym = x.get("sym","")
    q = ent_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    mode = ent.get("mode","")
    if mode not in MODES_CHECK: continue
    if ent.get("tf") != "15m": continue
    bars = int(x.get("bars_held") or 0)
    trades.append({
        "sym": sym, "mode": mode,
        "entry_ts": ent.get("ts",""),
        "entry_price": float(ent.get("price") or 0),
        "trail_k": float(ent.get("trail_k") or TRAIL_K_DEFAULT),
        "pnl": float(pnl), "bars": bars,
        "fast": bars <= 3,
    })

fast = [t for t in trades if t["fast"]]
slow = [t for t in trades if not t["fast"]]
print(f"Total 15m alignment/trend trades: {len(trades)}  fast={len(fast)}  slow={len(slow)}")

# ── Fetch klines & compute ATR ────────────────────────────────────────────
def parse_ts(s):
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        return int(dt.timestamp()*1000)
    except: return 0

def fetch_prev_bars(sym, entry_ms, n=20):
    url = (f"https://fapi.binance.com/fapi/v1/klines"
           f"?symbol={sym}&interval=15m&endTime={entry_ms}&limit={n}")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.load(r)
        return [{"h":float(k[2]),"l":float(k[3]),"c":float(k[4])} for k in data]
    except: return None

def compute_atr(bars, period=14):
    if not bars or len(bars) < period: return None
    trs = []
    for i in range(1, len(bars)):
        trs.append(max(bars[i]["h"]-bars[i]["l"],
                       abs(bars[i]["h"]-bars[i-1]["c"]),
                       abs(bars[i]["l"]-bars[i-1]["c"])))
    if len(trs) < period: return None
    a = sum(trs[-period:]) / period
    return a

# Sample: last 60 fast + 30 slow
sample = fast[-60:] + slow[-30:]
print(f"Fetching ATR for {len(sample)} trades...")

enriched = []
for i, t in enumerate(sample):
    start_ms = parse_ts(t["entry_ts"])
    if start_ms == 0: continue
    bars = fetch_prev_bars(t["sym"], start_ms, 20)
    if not bars: continue
    atr = compute_atr(bars)
    if atr is None or t["entry_price"] == 0: continue
    atr_pct = atr / t["entry_price"] * 100
    stop_width_pct = atr_pct * t["trail_k"]
    t["atr_pct"] = atr_pct
    t["stop_width_pct"] = stop_width_pct
    enriched.append(t)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(sample)}  ok={len(enriched)}")
    time.sleep(0.05)

print(f"Enriched {len(enriched)} trades\n")

fast_e = [t for t in enriched if t["fast"]]
slow_e = [t for t in enriched if not t["fast"]]

# ── ATR/price distribution ────────────────────────────────────────────────
def atr_stats(lst, label):
    if not lst:
        print(f"  {label}: n=0"); return
    atrs = [t["atr_pct"] for t in lst]
    sw   = [t["stop_width_pct"] for t in lst]
    avg_a = sum(atrs)/len(atrs)
    avg_s = sum(sw)/len(sw)
    pcts = sorted(atrs)
    n = len(pcts)
    p25 = pcts[n//4]; p50 = pcts[n//2]; p75 = pcts[3*n//4]
    tiny = sum(1 for a in atrs if a < 0.25)
    print(f"  {label:38s} n={n}  ATR/price: avg={avg_a:.3f}%  "
          f"p25={p25:.3f}%  p50={p50:.3f}%  p75={p75:.3f}%  "
          f"<0.25%: {tiny}/{n}({tiny/n*100:.0f}%)  stop_w_avg={avg_s:.3f}%")

print("=== ATR/price distribution: fast vs slow exits ===")
atr_stats(fast_e, "FAST exits (≤3 bars)")
atr_stats(slow_e, "SLOW exits (>3 bars)")

# ── Threshold sweep ───────────────────────────────────────────────────────
print(f"\n=== Threshold sweep: block if ATR/price < X% ===")
print(f"  {'Threshold':>10}  {'Fast blocked':>12}  {'Slow blocked':>12}  "
      f"{'Fast%blk':>9}  {'Slow%blk':>9}")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    fb = sum(1 for t in fast_e if t["atr_pct"] < thr)
    sb = sum(1 for t in slow_e if t["atr_pct"] < thr)
    fp = fb/len(fast_e)*100 if fast_e else 0
    sp = sb/len(slow_e)*100 if slow_e else 0
    print(f"  ATR < {thr:.2f}%  "
          f"  fast_blk={fb:>3d}/{len(fast_e):<3d}({fp:>4.0f}%)  "
          f"  slow_blk={sb:>3d}/{len(slow_e):<3d}({sp:>4.0f}%)")

# ── P&L by ATR bucket ─────────────────────────────────────────────────────
print(f"\n=== P&L by ATR/price bucket (all enriched) ===")
buckets = [(0,0.15),(0.15,0.25),(0.25,0.40),(0.40,0.70),(0.70,99)]
for lo, hi in buckets:
    sub = [t for t in enriched if lo <= t["atr_pct"] < hi]
    if not sub: continue
    n = len(sub); fast_n = sum(1 for t in sub if t["fast"])
    pnls = [t["pnl"] for t in sub]
    avg = sum(pnls)/n; wins = sum(1 for p in pnls if p>0)
    print(f"  ATR [{lo:.2f},{hi:.2f})%  n={n:>3d}  fast={fast_n}/{n}  "
          f"win={wins/n*100:.0f}%  avg={avg:+.3f}%")

# ── Concrete: ENS and SEI ATR ─────────────────────────────────────────────
print(f"\n=== Recent signal coins: estimated ATR ===")
for sym in ["ENSUSDT", "SEIUSDT", "POLUSDT"]:
    bars = fetch_prev_bars(sym, int(datetime.now(timezone.utc).timestamp()*1000), 20)
    if bars:
        atr = compute_atr(bars)
        price = bars[-1]["c"]
        if atr and price:
            print(f"  {sym:14s}  price={price:.6g}  ATR={atr:.6g}  "
                  f"ATR/price={atr/price*100:.3f}%  stop@ATR×2={atr*2/price*100:.3f}%")
    time.sleep(0.05)
