"""Diagnose why BONKUSDT has no recent signals."""
from __future__ import annotations
import json, io, sys, urllib.request
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

TODAY = datetime.now(timezone.utc).date().isoformat()
YESTERDAY = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
SYM = "BONKUSDT"

entries, exits_, blocked = [], [], []
with io.open(FILES / "bot_events.jsonl", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except: continue
        if e.get("sym") != SYM: continue
        ts = e.get("ts","")
        ev = e.get("event")
        if ev == "entry":   entries.append(e)
        elif ev == "exit":  exits_.append(e)
        elif ev == "blocked": blocked.append(e)

recent_entries  = [e for e in entries  if e.get("ts","") >= YESTERDAY]
recent_exits    = [e for e in exits_   if e.get("ts","") >= YESTERDAY]
recent_blocked  = [e for e in blocked  if e.get("ts","") >= YESTERDAY]
today_blocked   = [e for e in blocked  if e.get("ts","") >= TODAY]
today_entries   = [e for e in entries  if e.get("ts","") >= TODAY]

print(f"=== BONKUSDT — last 48h ===")
print(f"Entries:  {len(recent_entries)}")
print(f"Exits:    {len(recent_exits)}")
print(f"Blocked:  {len(recent_blocked)}")

# Is there a current open position?
open_pos = len(recent_entries) - len(recent_exits)
print(f"Currently open: {max(0, open_pos)} position(s)")

print(f"\n=== Today's entries ===")
for e in today_entries:
    print(f"  {e.get('ts','')}  mode={e.get('mode')}  tf={e.get('tf')}  "
          f"price={e.get('price')}  bars={e.get('bars_held','?')}")

print(f"\n=== Today's blocks by reason_code ===")
by_rc = Counter(e.get("reason_code","?") for e in today_blocked)
for rc, n in by_rc.most_common(10):
    print(f"  {n:>4d}  {rc}")

print(f"\n=== Today's blocks by (reason_code, tf, mode) ===")
by_trio = Counter(
    (e.get("reason_code","?"), e.get("tf","?"), e.get("mode","?"))
    for e in today_blocked
)
for (rc, tf, mode), n in by_trio.most_common(15):
    print(f"  {n:>4d}  {rc:>28s}  {tf:>4s}  {mode}")

print(f"\n=== Last 5 block reasons (full text) ===")
for e in today_blocked[-5:]:
    print(f"  {e.get('ts','')[:16]}  [{e.get('tf')}]  {e.get('mode')}  "
          f"rc={e.get('reason_code')}  reason={str(e.get('reason',''))[:120]}")

# Fetch current market data
print(f"\n=== Current BONKUSDT market data ===")
try:
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BONKUSDT"
    with urllib.request.urlopen(url, timeout=8) as r:
        t = json.load(r)
    pct = float(t.get("priceChangePercent", 0))
    price = float(t.get("lastPrice", 0))
    vol = float(t.get("quoteVolume", 0))
    high = float(t.get("highPrice", 0))
    low  = float(t.get("lowPrice", 0))
    daily_range = (high - low) / low * 100 if low else 0
    print(f"  Price:        {price:.8g}")
    print(f"  24h change:   {pct:+.2f}%")
    print(f"  Daily range:  {daily_range:.2f}%")
    print(f"  Vol (24h):    ${vol/1e6:.1f}M")
except Exception as ex:
    print(f"  fetch error: {ex}")

# Fetch 15m klines for indicator estimate
print(f"\n=== BONKUSDT 15m indicators (last 50 bars) ===")
try:
    url = "https://fapi.binance.com/fapi/v1/klines?symbol=BONKUSDT&interval=15m&limit=50"
    with urllib.request.urlopen(url, timeout=8) as r:
        klines = json.load(r)
    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    vols   = [float(k[5]) for k in klines]

    # EMA
    def ema(vals, p):
        k = 2/(p+1); out = [vals[0]]
        for v in vals[1:]: out.append(v*k + out[-1]*(1-k))
        return out
    e20 = ema(closes, 20); e50 = ema(closes, 50)

    # Slope (20-bar EMA, last 3 bars)
    slope = (e20[-1]/e20[-4] - 1)*100 if e20[-4] else 0

    # RSI
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i]-closes[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    p=14; ag=sum(gains[:p])/p; al=sum(losses[:p])/p
    for i in range(p, len(gains)):
        ag=(ag*(p-1)+gains[i])/p; al=(al*(p-1)+losses[i])/p
    rs = ag/al if al else 100; rsi = 100-100/(1+rs)

    # ADX
    def adx_simple(highs, lows, closes, p=14):
        dms_p, dms_m, trs = [], [], []
        for i in range(1, len(closes)):
            h, l, ph, pl, pc = highs[i],lows[i],highs[i-1],lows[i-1],closes[i-1]
            dm_p = max(h-ph, 0) if (h-ph) > (pl-l) else 0
            dm_m = max(pl-l, 0) if (pl-l) > (h-ph) else 0
            tr = max(h-l, abs(h-pc), abs(l-pc))
            dms_p.append(dm_p); dms_m.append(dm_m); trs.append(tr)
        if len(trs) < p: return None
        atr=sum(trs[:p])/p; sdp=sum(dms_p[:p])/p; sdm=sum(dms_m[:p])/p
        for i in range(p, len(trs)):
            atr=(atr*(p-1)+trs[i])/p; sdp=(sdp*(p-1)+dms_p[i])/p; sdm=(sdm*(p-1)+dms_m[i])/p
        di_p=100*sdp/atr if atr else 0; di_m=100*sdm/atr if atr else 0
        dx = 100*abs(di_p-di_m)/(di_p+di_m) if (di_p+di_m) else 0
        return dx
    adx = adx_simple(highs, lows, closes)

    # Vol×
    avg_vol = sum(vols[-21:-1])/20 if len(vols)>=21 else 1
    vol_x = vols[-1]/avg_vol if avg_vol else 0

    # ATR
    trs2 = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            for i in range(1, len(closes))]
    atr_v = sum(trs2[-14:])/14

    price_now = closes[-1]
    ema20_now = e20[-1]; ema50_now = e50[-1]
    print(f"  close={price_now:.8g}  EMA20={ema20_now:.8g}  EMA50={ema50_now:.8g}")
    print(f"  slope={slope:+.3f}%  ADX={adx:.1f}  RSI={rsi:.1f}  vol_x={vol_x:.2f}")
    print(f"  ATR={atr_v:.8g}  ATR/price={atr_v/price_now*100:.3f}%  stop@ATR×2={atr_v*2/price_now*100:.3f}%")
    print(f"  close>EMA20: {price_now>ema20_now}  EMA20>EMA50: {ema20_now>ema50_now}")

    # Check alignment conditions manually
    print(f"\n=== Manual alignment gate check ===")
    price_edge = (price_now - ema20_now) / ema20_now * 100
    ema_sep = (ema20_now - ema50_now) / ema50_now * 100
    print(f"  price_edge from EMA20: {price_edge:.3f}%  (max 2.0%)")
    print(f"  EMA sep (EMA20-EMA50)/EMA50: {ema_sep:.3f}%  (min 0.05%)")
    print(f"  slope: {slope:+.3f}%  (min 0.05%)")
    print(f"  RSI: {rsi:.1f}  (range [45,82] nonbull [50,66])")
    print(f"  ADX: {adx:.1f}  (min 15, nonbull min 20)")
    print(f"  vol_x: {vol_x:.2f}  (min 0.8, nonbull min 1.0)")

    # Check impulse_speed conditions
    print(f"\n=== Manual impulse_speed (15m) gate check ===")
    print(f"  ADX: {adx:.1f}  (min 15)")
    print(f"  RSI: {rsi:.1f}  (max 76)")
    print(f"  daily_range approx: see 24h above")

except Exception as ex:
    print(f"  klines error: {ex}")

print(f"\n=== Cooldown check ===")
# Last exit for BONK
bonk_exits_sorted = sorted(exits_, key=lambda e: e.get("ts",""))
if bonk_exits_sorted:
    last_exit = bonk_exits_sorted[-1]
    print(f"  Last exit: {last_exit.get('ts','')}  reason={str(last_exit.get('exit_reason',''))[:80]}")
    print(f"  PnL: {last_exit.get('pnl_pct','?')}%  bars={last_exit.get('bars_held','?')}")
else:
    print("  No exits found")
