"""What separates overbought moves that KEEP RUNNING from those that reverse?

Motivating case: C98USDT 2026-08-07 — the bot exited at RSI 87.6 with +6.3% and
the coin ran another ~16%. Previous attempts to fix this were unconditional
("hold longer", "wider trail", "premature-exit model") and were all refuted, so
this study asks the conditional question instead: at the moment RSI goes extreme,
which measurable features predict continuation?

Event: the first 1h bar of an overbought episode (RSI14 crosses >= RSI_HOT after
being below it). That mirrors the bot's exit trigger.

Outcome (path-dependent, first touch — no lookahead): starting next bar, does
price gain >= CONT_PCT before drawing down >= DD_PCT within HORIZON bars?
  CONTINUATION = the up-target is hit first
  REVERSAL     = the drawdown is hit first (or neither within the horizon)

Features are computed ONLY from bars up to and including the event bar:
  vol_x        relative volume vs its own 20-bar mean
  vol_accel    last-3-bar volume vs the previous 20
  ext_ma25     % above MA25 (extension)
  ext_ma99     % above MA99
  rsi          the RSI value itself
  bars_hot     how long RSI has already been >= 70 (persistence)
  green_run    consecutive up bars
  body_ratio   |close-open| / (high-low) of the event bar (conviction)
  upper_wick   upper wick share of the bar range (selling into strength)
  macd_hist_n  MACD histogram / close, %
  new_high_n   is close the highest of the last 168 bars (7d breakout)
  btc_up       BTC 1h close above its own MA25 (market regime)
  trades_x     trade count vs its 20-bar mean

For each feature the script reports the continuation rate in the bottom/middle/top
tercile plus the lift of the top tercile over the base rate, so a feature only
counts if it separates outcomes materially.

Read-only.  pyembed\python.exe files\_backtest_overbought_continuation.py
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
H = json.load(io.open(ROOT/"files"/"_hourly_ohlcv.json", encoding="utf-8"))

RSI_HOT = 80.0        # overbought trigger (the bot exited C98 at 87.6)
CONT_PCT = 5.0        # continuation target from the event close
DD_PCT = 3.0          # drawdown that counts as "reversed"
HORIZON = 24          # bars (1h) to resolve the race


def rsi14(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    k = 2.0/(n+1)
    au = np.empty_like(up); ad = np.empty_like(dn)
    au[0], ad[0] = up[0], dn[0]
    for i in range(1, len(up)):
        au[i] = up[i]*k + au[i-1]*(1-k)
        ad[i] = dn[i]*k + ad[i-1]*(1-k)
    return 100 - 100/(1 + au/np.maximum(ad, 1e-12))


def ema(a, n):
    k = 2.0/(n+1); out = np.empty_like(a); out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i]*k + out[i-1]*(1-k)
    return out


def ma(a, n):
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        cs = np.cumsum(np.insert(a, 0, 0.0))
        out[n-1:] = (cs[n:] - cs[:-n]) / n
    return out


# BTC regime reference
btc = H.get("BTCUSDT")
btc_up_at = {}
if btc:
    bt = np.array([r[0] for r in btc]); bc = np.array([r[4] for r in btc])
    bma = ma(bc, 25)
    for i in range(len(bt)):
        if not np.isnan(bma[i]):
            btc_up_at[int(bt[i])] = bool(bc[i] > bma[i])

FEATS = ["vol_x", "vol_accel", "ext_ma25", "ext_ma99", "rsi", "bars_hot",
         "green_run", "body_ratio", "upper_wick", "macd_hist_n", "new_high_n",
         "btc_up", "trades_x"]
rows = []

for sym, k in H.items():
    if len(k) < 200:
        continue
    t = np.array([r[0] for r in k]); o = np.array([r[1] for r in k])
    hi = np.array([r[2] for r in k]); lo = np.array([r[3] for r in k])
    c = np.array([r[4] for r in k]); v = np.array([r[5] for r in k])
    tr = np.array([r[6] for r in k], dtype=float)
    r = rsi14(c)
    m25, m99 = ma(c, 25), ma(c, 99)
    vma = ma(v, 20); tma = ma(tr, 20)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)

    for i in range(120, len(c) - HORIZON - 1):
        if not (r[i] >= RSI_HOT and r[i-1] < RSI_HOT):
            continue                       # first bar of the episode only
        if np.isnan(m25[i]) or np.isnan(m99[i]) or np.isnan(vma[i]) or vma[i] <= 0:
            continue
        entry = c[i]
        if entry <= 0:
            continue
        up_t = entry * (1 + CONT_PCT/100.0)
        dn_t = entry * (1 - DD_PCT/100.0)
        outcome = 0
        for j in range(i+1, i+1+HORIZON):
            hit_up = hi[j] >= up_t
            hit_dn = lo[j] <= dn_t
            if hit_up and hit_dn:
                outcome = 0                # ambiguous bar -> count as reversal
                break
            if hit_up:
                outcome = 1; break
            if hit_dn:
                outcome = 0; break

        bars_hot = 0
        while bars_hot < 48 and r[i-bars_hot] >= 70:
            bars_hot += 1
        green = 0
        while green < 24 and c[i-green] > o[i-green]:
            green += 1
        rng = max(hi[i] - lo[i], 1e-12)
        rows.append({
            "sym": sym, "outcome": outcome,
            "vol_x": v[i]/vma[i],
            "vol_accel": float(np.mean(v[i-2:i+1]) / max(np.mean(v[i-22:i-2]), 1e-12)),
            "ext_ma25": (entry/m25[i]-1)*100,
            "ext_ma99": (entry/m99[i]-1)*100,
            "rsi": r[i],
            "bars_hot": bars_hot,
            "green_run": green,
            "body_ratio": abs(c[i]-o[i])/rng,
            "upper_wick": (hi[i]-max(o[i], c[i]))/rng,
            "macd_hist_n": hist[i]/entry*100,
            "new_high_n": 1.0 if c[i] >= np.max(hi[max(0, i-168):i]) else 0.0,
            "btc_up": 1.0 if btc_up_at.get(int(t[i]), False) else 0.0,
            "trades_x": tr[i]/max(tma[i], 1e-12) if not np.isnan(tma[i]) else 1.0,
        })

n = len(rows)
if n < 50:
    print(f"only {n} overbought episodes — not enough"); sys.exit(0)
base = sum(x["outcome"] for x in rows)/n
print("=" * 78)
print(f"Overbought continuation study  ·  {len(H)} symbols, {n} episodes "
      f"(RSI14 1h crossing >= {RSI_HOT:.0f})")
print(f"Outcome: +{CONT_PCT:.0f}% before -{DD_PCT:.0f}% within {HORIZON}h "
      f"(first touch)   BASE RATE = {100*base:.1f}%")
print("=" * 78)
print(f"{'feature':<13}{'low third':>12}{'mid':>8}{'high third':>12}"
      f"{'lift(top)':>11}{'spread':>9}")
res = []
for f in FEATS:
    vals = np.array([x[f] for x in rows])
    out = np.array([x["outcome"] for x in rows], dtype=float)
    if len(set(vals.tolist())) <= 2:                    # binary feature
        lo_r = out[vals == 0].mean() if (vals == 0).any() else float("nan")
        hi_r = out[vals == 1].mean() if (vals == 1).any() else float("nan")
        print(f"{f:<13}{100*lo_r:>11.1f}%{'-':>8}{100*hi_r:>11.1f}%"
              f"{hi_r/base:>11.2f}{100*(hi_r-lo_r):>8.1f}pp")
        res.append((abs(hi_r-lo_r), f, hi_r/base))
        continue
    q1, q2 = np.quantile(vals, [1/3, 2/3])
    lo_r = out[vals <= q1].mean(); mid = out[(vals > q1) & (vals <= q2)].mean()
    hi_r = out[vals > q2].mean()
    print(f"{f:<13}{100*lo_r:>11.1f}%{100*mid:>7.1f}%{100*hi_r:>11.1f}%"
          f"{hi_r/base:>11.2f}{100*(hi_r-lo_r):>8.1f}pp")
    res.append((abs(hi_r-lo_r), f, hi_r/base))

print("-" * 78)
res.sort(reverse=True)
print("strongest separators:", ", ".join(f"{f} ({100*d:.0f}pp)" for d, f, _ in res[:4]))
print("\nA feature is useful only if the spread is large AND the top tercile beats")
print("the base rate — otherwise it just re-labels the same coin flip.")

# RESULT (2026-08-07, 105 symbols x ~41d of 1h bars, 920 overbought episodes):
#   BASE continuation rate = 14.0%  (+5% before -3% within 24h)
#
#   feature        low third -> high third   lift
#   ext_ma25          8.8%  ->  19.2%        1.37   <- strongest
#   ext_ma99          9.4%  ->  18.6%        1.32
#   vol_accel        10.1%  ->  18.6%        1.32
#   vol_x            11.1%  ->  16.9%        1.21
#   rsi              15.3%  ->  15.3%        1.09   <- NO separation at all
#   bars_hot         15.2%  ->  14.6%        1.04
#   green_run        13.8%  ->  14.4%        1.03
#   upper_wick       15.3%  ->  14.7%        1.05
#   btc_up           17.3%  ->  13.1%        0.94   <- market-wide moves continue LESS
#
# Combinations:
#   ext top decile (> ~8% over MA25)      n=92  28.3%  lift 2.02  <- best
#   ext top third AND vol_accel top third n=150 20.0%  lift 1.43
#   ext top decile AND vol_accel top dec. n=32  18.8%  lift 1.34  (does NOT stack)
#   ext bottom third AND vol_accel bottom n=135  7.4%  lift 0.53  (reliable reversal)
#
# CONCLUSIONS
# 1. How overbought the RSI is carries NO information about continuation - the
#    bot's exit trigger is not the problem, and tuning the RSI threshold is
#    pointless.
# 2. The only real discriminator is EXTENSION above MA25: the most extended
#    decile continues twice as often as the base rate. Volume helps on its own
#    but adds nothing on top of extension.
# 3. Even so, "hold" stays EV-negative: at the best-conditioned 28.3%,
#    0.283*5 - 0.717*3 = -0.76% per episode (base rate: -1.88%). This is the
#    mechanism behind the three earlier refutations of holding longer - the
#    continuation odds never cover the drawdown.
# 4. Market context runs the other way: overbought moves while BTC is BELOW its
#    MA25 continue MORE (17.3% vs 13.1%) - idiosyncratic strength beats
#    market-wide strength, consistent with the decoupling finding.
#
# => Do NOT relax the overbought exit. The useful direction is the mirror image:
#    weak-extension overbought (7.4% continuation) is a near-certain reversal and
#    is worth avoiding on the ENTRY side, not chasing on the exit side.
