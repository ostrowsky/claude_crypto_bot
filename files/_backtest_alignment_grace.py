"""
Backtest: grace period for alignment 15m fast exits.

Hypothesis: if we hold a FIXED initial stop for the first N bars
(no trailing), then switch to ATR trailing — fast exits are avoided
and the trade has time to develop.

Method:
  1. Load all fast alignment 15m exits (bars_held ≤ 3) from bot_events
  2. For each, fetch 25 forward bars from Binance klines API
  3. Reconstruct ATR, compute initial stop = entry - trail_k × ATR
  4. Simulate three regimes:
       Baseline: trail from bar 1 (current behavior)
       Grace-3:  fixed stop for bars 1-3, trail from bar 4
       Grace-4:  fixed stop for bars 1-4, trail from bar 5
  5. Compare avg P&L, win%, survival rate

Trailing stop rule (approximation of bot logic):
  stop_new = max(stop_old, close_bar - trail_k × ATR)
  exit if close_bar < stop_new
"""
from __future__ import annotations
import json, io, sys, math, time, urllib.request
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

TRAIL_K_DEFAULT = 2.0
LIMIT_BARS = 25  # fetch this many bars after entry

# ── 1. Load fast alignment exits ─────────────────────────────────────────
entries_q: dict[str, list[dict]] = defaultdict(list)
exits_raw: list[dict] = []

with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        ev = e.get("event")
        if ev == "entry":
            entries_q[e.get("sym","")].append(e)
        elif ev == "exit":
            exits_raw.append(e)

entries_q = {k: sorted(v, key=lambda e: e.get("ts","")) for k, v in entries_q.items()}
exits_raw.sort(key=lambda e: e.get("ts",""))

fast_trades = []
for x in exits_raw:
    sym = x.get("sym","")
    q = entries_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    bars = x.get("bars_held") or 0
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    if ent.get("mode") != "alignment": continue
    if ent.get("tf") != "15m": continue
    if int(bars) > 3: continue
    fast_trades.append({
        "sym": sym,
        "entry_ts": ent.get("ts",""),
        "entry_price": float(ent.get("price") or 0),
        "trail_k": float(ent.get("trail_k") or TRAIL_K_DEFAULT),
        "pnl_orig": float(pnl),
        "bars_orig": int(bars),
    })

print(f"Fast alignment 15m exits (≤3 bars): {len(fast_trades)}")

# ── 2. Fetch klines ──────────────────────────────────────────────────────
def parse_ts(s: str) -> int:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except: return 0

def fetch_klines(sym: str, start_ms: int, limit: int) -> list | None:
    url = (f"https://fapi.binance.com/fapi/v1/klines"
           f"?symbol={sym}&interval=15m&startTime={start_ms}&limit={limit}")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.load(r)
        return [{"o": float(k[1]), "h": float(k[2]),
                 "l": float(k[3]), "c": float(k[4])} for k in data]
    except: return None

def atr_series(bars: list, period: int = 14) -> list:
    trs = []
    for i, b in enumerate(bars):
        if i == 0: trs.append(b["h"] - b["l"]); continue
        pc = bars[i-1]["c"]
        trs.append(max(b["h"]-b["l"], abs(b["h"]-pc), abs(b["l"]-pc)))
    if len(trs) < period:
        return [None] * len(trs)
    out = [None] * (period - 1)
    a = sum(trs[:period]) / period
    out.append(a)
    for i in range(period, len(trs)):
        a = (a * (period-1) + trs[i]) / period
        out.append(a)
    return out

# ── 3. Simulate trailing stop ────────────────────────────────────────────
def simulate_trail(bars: list, entry_price: float, trail_k: float,
                   grace_bars: int = 0) -> dict:
    """
    Simulate trailing stop from bar 0 (entry bar).
    grace_bars: for bars 0..grace_bars-1, use fixed initial stop.
    Returns: {"exit_bar": int, "exit_price": float, "pnl_pct": float}
    """
    atrs = atr_series(bars)

    # Compute initial ATR at entry (last available before entry)
    init_atr = None
    for a in reversed(atrs[:15]):
        if a is not None:
            init_atr = a
            break
    if init_atr is None:
        init_atr = (bars[0]["h"] - bars[0]["l"]) if bars else 0.001

    initial_stop = entry_price - trail_k * init_atr
    stop = initial_stop
    max_close = entry_price

    for i, bar in enumerate(bars):
        c = bar["c"]
        atr_now = atrs[i] if atrs[i] is not None else init_atr

        if i >= grace_bars:
            # trailing: update stop to trail behind max close
            max_close = max(max_close, c)
            trail_stop = max_close - trail_k * atr_now
            stop = max(stop, trail_stop)

        # check exit: use close (conservative; real bot uses intra-bar)
        if c <= stop:
            pnl = (c - entry_price) / entry_price * 100
            return {"exit_bar": i, "exit_price": c, "pnl_pct": pnl, "stop": stop}

    # no exit within window — mark as "still open" at last close
    c = bars[-1]["c"]
    pnl = (c - entry_price) / entry_price * 100
    return {"exit_bar": len(bars), "exit_price": c, "pnl_pct": pnl, "stop": stop}

# ── 4. Main loop ──────────────────────────────────────────────────────────
print(f"Fetching klines for {len(fast_trades)} trades...")
enriched = []; failed = 0

for i, t in enumerate(fast_trades):
    if not t["entry_ts"] or not t["entry_price"]: failed += 1; continue
    start_ms = parse_ts(t["entry_ts"])
    if start_ms == 0: failed += 1; continue

    bars = fetch_klines(t["sym"], start_ms, LIMIT_BARS)
    if not bars or len(bars) < 5: failed += 1; continue

    t["bars_data"] = bars
    enriched.append(t)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(fast_trades)}  ok={len(enriched)}  failed={failed}")
    time.sleep(0.05)

print(f"Enriched {len(enriched)} / {len(fast_trades)} (failed {failed})\n")

# ── 5. Simulate and compare ───────────────────────────────────────────────
def summarize(results: list, label: str):
    if not results:
        print(f"  {label:45s} n=0"); return None
    n = len(results)
    pnls = [r["pnl_pct"] for r in results]
    wins = sum(1 for p in pnls if p > 0)
    avg = sum(pnls) / n
    sd = (sum((p-avg)**2 for p in pnls)/n)**0.5 if n > 1 else 0
    sh = avg/sd*math.sqrt(n) if sd > 0 else 0
    print(f"  {label:45s} n={n:>4d}  win={wins/n*100:>5.1f}%  "
          f"avg={avg:+.4f}%  sharpe={sh:+.2f}")
    return {"n": n, "avg": avg, "win": wins/n*100, "sh": sh}

print("=== SIMULATION RESULTS ===\n")

for grace in [0, 3, 4, 5]:
    label = "baseline (grace=0)" if grace == 0 else f"grace={grace} bars"
    results = []
    for t in enriched:
        r = simulate_trail(t["bars_data"], t["entry_price"],
                           t["trail_k"], grace_bars=grace)
        results.append(r)
    summarize(results, label)

# ── 6. Survival and bar distribution ──────────────────────────────────────
print(f"\n=== EXIT BAR distribution per regime ===")
for grace in [0, 3, 4]:
    label = f"grace={grace}"
    exits_by_bar: dict[int, int] = defaultdict(int)
    for t in enriched:
        r = simulate_trail(t["bars_data"], t["entry_price"], t["trail_k"], grace_bars=grace)
        eb = min(r["exit_bar"], LIMIT_BARS)
        bucket = eb if eb <= 10 else (15 if eb <= 15 else (20 if eb <= 20 else 25))
        exits_by_bar[bucket] += 1
    dist = ", ".join(f"bar{k}:{v}" for k, v in sorted(exits_by_bar.items()))
    print(f"  {label}: {dist}")

# ── 7. Worst-case analysis: grace=3 vs baseline, tail comparison ──────────
print(f"\n=== PERCENTILE P&L comparison (baseline vs grace=3) ===")
base_pnls = sorted(
    simulate_trail(t["bars_data"], t["entry_price"], t["trail_k"], 0)["pnl_pct"]
    for t in enriched
)
g3_pnls = sorted(
    simulate_trail(t["bars_data"], t["entry_price"], t["trail_k"], 3)["pnl_pct"]
    for t in enriched
)
n = len(base_pnls)
for pct in [10, 25, 50, 75, 90]:
    idx = int(n * pct / 100)
    print(f"  p{pct:>2d}: baseline={base_pnls[idx]:+.3f}%  grace3={g3_pnls[idx]:+.3f}%  "
          f"delta={g3_pnls[idx]-base_pnls[idx]:+.3f}%")

# ── 8. Slow trades check: does grace hurt already-profitable slow exits? ──
print(f"\n=== SANITY: slow trades (bars_orig>3) — does grace=3 change them? ===")
# reload slow trades
entries_q2: dict[str, list[dict]] = defaultdict(list)
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") == "entry":
            entries_q2[e.get("sym","")].append(e)
entries_q2 = {k: sorted(v, key=lambda e: e.get("ts","")) for k, v in entries_q2.items()}

slow_sample = []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("event") != "exit": continue
        sym = e.get("sym","")
        q = entries_q2.get(sym, [])
        if not q: continue
        ent = q.pop(0)
        bars = e.get("bars_held") or 0
        pnl = e.get("pnl_pct")
        if pnl is None: continue
        if ent.get("mode") != "alignment" or ent.get("tf") != "15m": continue
        if int(bars) <= 3: continue
        slow_sample.append({
            "sym": sym, "entry_ts": ent.get("ts",""),
            "entry_price": float(ent.get("price") or 0),
            "trail_k": float(ent.get("trail_k") or TRAIL_K_DEFAULT),
            "pnl_orig": float(pnl),
        })
        if len(slow_sample) >= 40: break

print(f"  Loading {len(slow_sample)} slow alignment trades for sanity check...")
slow_enriched = []
for t in slow_sample[:30]:
    start_ms = parse_ts(t["entry_ts"])
    if start_ms == 0: continue
    bars = fetch_klines(t["sym"], start_ms, LIMIT_BARS)
    if not bars or len(bars) < 5: continue
    t["bars_data"] = bars
    slow_enriched.append(t)
    time.sleep(0.05)

print(f"  Slow enriched: {len(slow_enriched)}")
base_slow = [simulate_trail(t["bars_data"], t["entry_price"], t["trail_k"], 0) for t in slow_enriched]
g3_slow   = [simulate_trail(t["bars_data"], t["entry_price"], t["trail_k"], 3) for t in slow_enriched]
summarize(base_slow, "slow: baseline (grace=0)")
summarize(g3_slow,   "slow: grace=3")
