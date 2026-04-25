"""
Backtest 4 proposed ALIGNMENT input gates:

  G1 MACD-slope:     macd_hist[-1] >= macd_hist[-3]            (moment not fading)
  G2 RSI anti-div:   NOT (close[-1]>close[-5] AND rsi[-1] < rsi[-5]-3)
  G3 Upper-wick:     last-2-bar mean (high-max(O,C))/(H-L) <= 0.50
  G4 Near-top:       (high_20bar - close) / atr >= 0.30

Fetch OHLCV from Binance Futures klines API per alignment trade entry
(±60 bars before entry). Recompute MACD/RSI/ATR, apply gates, compare
KEPT vs baseline (no gate).
"""
from __future__ import annotations
import json, io, sys, math, time, urllib.request
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

TF_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000}
INTERVAL_MAP = {"15m": "15m", "1h": "1h"}

# ── 1. Pair alignment trades ──────────────────────────────────────────────
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

ent_q = defaultdict(list)
for e in entries: ent_q[e.get("sym", "")].append(e)

trades = []
for x in exits_:
    sym = x.get("sym", "")
    q = ent_q.get(sym, [])
    if not q: continue
    ent = q.pop(0)
    pnl = x.get("pnl_pct")
    if pnl is None: continue
    if ent.get("mode") != "alignment": continue
    if ent.get("tf") not in TF_MS: continue
    trades.append({
        "sym": sym, "tf": ent["tf"], "mode": ent["mode"],
        "pnl": float(pnl), "bars": x.get("bars_held") or 0,
        "entry_ts": ent.get("ts", ""), "entry_price": float(ent.get("price") or 0),
    })
print(f"Alignment trades to backtest: {len(trades)}")

# ── 2. Fetch klines per trade ─────────────────────────────────────────────
def parse_ts(s: str) -> int:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception: return 0

def fetch_klines(sym: str, tf: str, end_ms: int, limit: int = 60) -> list | None:
    iv = INTERVAL_MAP[tf]
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={iv}&endTime={end_ms}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.load(r)
        return [{"ts": k[0], "o": float(k[1]), "h": float(k[2]),
                 "l": float(k[3]), "c": float(k[4]), "v": float(k[5])} for k in data]
    except Exception as e:
        return None

# ── 3. Indicator recomputation ────────────────────────────────────────────
def ema(vals, period):
    if not vals: return []
    k = 2 / (period + 1); out = [vals[0]]
    for v in vals[1:]: out.append(v * k + out[-1] * (1 - k))
    return out

def rsi(closes, period=14):
    if len(closes) < period + 1: return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    avg_g = sum(gains[:period]) / period; avg_l = sum(losses[:period]) / period
    out = [None] * period
    rs = avg_g / avg_l if avg_l > 0 else 100
    out.append(100 - 100/(1+rs))
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period-1) + gains[i]) / period
        avg_l = (avg_l * (period-1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l > 0 else 100
        out.append(100 - 100/(1+rs))
    return out

def macd_hist_series(closes):
    e12 = ema(closes, 12); e26 = ema(closes, 26)
    macd = [a-b for a,b in zip(e12, e26)]
    sig = ema(macd, 9)
    return [m - s for m, s in zip(macd, sig)]

def atr_series(bars, period=14):
    trs = []
    for i, b in enumerate(bars):
        if i == 0: trs.append(b["h"] - b["l"]); continue
        pc = bars[i-1]["c"]
        tr = max(b["h"]-b["l"], abs(b["h"]-pc), abs(b["l"]-pc))
        trs.append(tr)
    out = [None]*(period-1)
    if len(trs) < period: return out + [None]*(len(trs)-(period-1))
    a = sum(trs[:period])/period; out.append(a)
    for i in range(period, len(trs)):
        a = (a*(period-1) + trs[i])/period; out.append(a)
    return out

# ── 4. Gates ──────────────────────────────────────────────────────────────
def evaluate_gates(bars: list) -> dict | None:
    if not bars or len(bars) < 30: return None
    closes = [b["c"] for b in bars]
    highs  = [b["h"] for b in bars]
    opens  = [b["o"] for b in bars]

    mh = macd_hist_series(closes)
    rs = rsi(closes)
    at = atr_series(bars)
    if not mh or len(mh) < 4 or not rs or len(rs) < 6: return None

    # G1: MACD slope not fading
    g1 = mh[-1] >= mh[-3]

    # G2: RSI anti-divergence (NOT divergence)
    div = (closes[-1] > closes[-5]) and (rs[-1] is not None and rs[-5] is not None
           and rs[-1] < rs[-5] - 3)
    g2 = not div

    # G3: mean upper-wick share of last 2 bars
    def wick_share(b):
        rng = b["h"] - b["l"]
        if rng <= 0: return 0
        body_top = max(b["o"], b["c"])
        return (b["h"] - body_top) / rng
    ws = (wick_share(bars[-1]) + wick_share(bars[-2])) / 2
    g3 = ws <= 0.50

    # G4: distance to 20-bar high in ATRs
    h20 = max(highs[-20:])
    atr_last = at[-1] if at and at[-1] else None
    if atr_last and atr_last > 0:
        dist_atr = (h20 - closes[-1]) / atr_last
        g4 = dist_atr >= 0.30
    else:
        g4 = True  # can't evaluate -> allow

    return {"g1": g1, "g2": g2, "g3": g3, "g4": g4,
            "macd_slope_ok": g1, "rsi_div": div,
            "upper_wick_avg": ws, "dist_to_high_atr":
                ((h20 - closes[-1]) / atr_last) if atr_last else None}

# ── 5. Main loop ──────────────────────────────────────────────────────────
print("Fetching klines (this may take ~1-2 min)...")
enriched = []; fetched = 0; failed = 0
for i, t in enumerate(trades):
    entry_ms = parse_ts(t["entry_ts"])
    if entry_ms == 0: failed += 1; continue
    end_ms = entry_ms - 1  # bars ending BEFORE entry bar
    bars = fetch_klines(t["sym"], t["tf"], end_ms, limit=60)
    if not bars: failed += 1; continue
    gates = evaluate_gates(bars)
    if not gates: failed += 1; continue
    t.update(gates); enriched.append(t); fetched += 1
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(trades)}  ok={fetched} failed={failed}")
    time.sleep(0.05)  # be nice to binance
print(f"Done. Enriched {fetched}/{len(trades)} trades (failed {failed})\n")

# ── 6. Analysis ───────────────────────────────────────────────────────────
def summarize(lst, name):
    if not lst:
        print(f"  {name:50s} n=0"); return None
    n = len(lst); pnls = [t["pnl"] for t in lst]
    wins = sum(1 for p in pnls if p > 0); tot = sum(pnls); avg = tot/n
    sd = (sum((p-avg)**2 for p in pnls)/n)**0.5 if n>1 else 0
    sh = (avg/sd*math.sqrt(n)) if sd>0 else 0
    print(f"  {name:50s} n={n:>4d}  win={wins/n*100:>5.1f}%  avg={avg:+.3f}%  "
          f"sum={tot:+7.1f}%  sharpe={sh:+.2f}")
    return {"n": n, "win": wins/n*100, "avg": avg, "sum": tot, "sharpe": sh}

for tf in ["15m", "1h"]:
    sub = [t for t in enriched if t["tf"] == tf]
    if not sub: continue
    print(f"=== alignment | {tf} (n={len(sub)}) ===")
    base = summarize(sub, "baseline (all alignment)")

    for gname, pred in [("G1 MACD-slope",   lambda t: t["g1"]),
                        ("G2 RSI anti-div", lambda t: t["g2"]),
                        ("G3 Upper-wick",   lambda t: t["g3"]),
                        ("G4 Dist-to-high", lambda t: t["g4"])]:
        kept = [t for t in sub if pred(t)]
        blk  = [t for t in sub if not pred(t)]
        summarize(kept, f"KEPT  by {gname}")
        summarize(blk,  f"      BLOCKED")

    # Combined gate (all 4 pass)
    kept_all = [t for t in sub if t["g1"] and t["g2"] and t["g3"] and t["g4"]]
    blk_all  = [t for t in sub if not (t["g1"] and t["g2"] and t["g3"] and t["g4"])]
    print()
    summarize(kept_all, "ALL 4 gates pass")
    summarize(blk_all,  "    at least one FAILED")

    # Pairs (2-gate combos) - which pair is strongest?
    print("  pair combos (KEPT = both gates pass):")
    gkeys = ["g1","g2","g3","g4"]
    for a in range(4):
        for b in range(a+1, 4):
            kept = [t for t in sub if t[gkeys[a]] and t[gkeys[b]]]
            st = summarize(kept, f"    {gkeys[a]}+{gkeys[b]}")
            if st and base:
                d = st["avg"] - base["avg"]
                blocked = len(sub) - st["n"]
                print(f"          delta_avg={d:+.3f}%  blocked={blocked}/{len(sub)}")
    print()

# ── 7. "Fired at top" pattern coverage ────────────────────────────────────
print("=== 'fired-at-top' pattern coverage ===")
for tf in ["15m", "1h"]:
    sub = [t for t in enriched if t["tf"] == tf]
    top_pat = [t for t in sub if t["bars"] <= 4 and -5 <= t["pnl"] <= -0.5]
    if not top_pat: continue
    # How many of these would each gate catch?
    for gname, pred in [("G1", lambda t: not t["g1"]),
                        ("G2", lambda t: not t["g2"]),
                        ("G3", lambda t: not t["g3"]),
                        ("G4", lambda t: not t["g4"]),
                        ("ANY of 4", lambda t: not (t["g1"] and t["g2"] and t["g3"] and t["g4"]))]:
        caught = [t for t in top_pat if pred(t)]
        print(f"  {tf} {gname:10s} catches {len(caught)}/{len(top_pat)} "
              f"({len(caught)/len(top_pat)*100:.1f}%) of fired-at-top losers")
