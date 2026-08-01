"""How much noise would a SOFTER scan-promotion criterion cost, and how many
missed rockets would it recover?

Context: POL-type coins sit in a blind spot — too calm for the momentum modes
(discovery only promotes signal_now=True) and too market-correlated for the
decoupling promoter (which wants corr 0.1-0.3). The only remaining lever is to
promote coins into the SCAN on a much softer 1h condition, paying with noise.
This quantifies that trade honestly.

Criteria (evaluated at each 1h bar close, fires only BEFORE that day's high —
no lookahead):
  S1  close > MA7  AND  MACD hist > 0        (the "soft" idea)
  S2  close > MA7 > MA25                      (weak 1h stack)
  S3  close > MA7  AND  RSI14 > 55
  S4  close > MA7  AND  MACD hist > 0  AND  RSI14 > 55   (tightened S1)

Metrics per criterion:
  promotions/day        — scan load (coin-days per day)
  extra/day             — beyond coins the bot already scanned that day
  top20 recall          — share of watchlist top-20 (day,sym) flagged before peak
  silent-miss recall    — same, restricted to top-20 the bot NEVER scanned
  precision             — share of promotions that were top-20 that day
  POL                   — days POLUSDT would have been promoted

Read-only.  pyembed\python.exe files\_backtest_soft_promotion.py
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
H = json.load(io.open(ROOT/"files"/"_hourly_cache.json", encoding="utf-8"))

# Data hygiene: the cache is spot klines; a few watchlist pairs are no longer
# traded there, so their "last 1000 bars" are from 2024 and would pollute both
# the load and the recall figures. Keep only symbols whose latest bar is recent,
# then evaluate on the window they actually share.
_now_ms = datetime.now(timezone.utc).timestamp() * 1000
_stale = [s for s, r in H.items() if not r or (_now_ms - r[-1][0]) > 3*86400_000]
for s in _stale:
    H.pop(s, None)
_starts = [r[0][0] for r in H.values()]
WINDOW_START = datetime.fromtimestamp(max(_starts)/1000, tz=timezone.utc).strftime("%Y-%m-%d")
print(f"[data] usable symbols: {len(H)} (dropped {len(_stale)} stale: "
      f"{', '.join(sorted(_stale)[:6])}{'...' if len(_stale) > 6 else ''})")


def ema(a, n):
    k = 2.0/(n+1); out = np.empty_like(a); out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i]*k + out[i-1]*(1-k)
    return out


def rsi14(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = ema(up, n); ad = ema(dn, n)
    rs = np.divide(au, np.maximum(ad, 1e-12))
    return 100 - 100/(1+rs)


def ma(c, n):
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        cs = np.cumsum(np.insert(c, 0, 0.0))
        out[n-1:] = (cs[n:] - cs[:-n]) / n
    return out


# ---- top-20 (day,sym) and what the bot actually scanned ----
top = set()
for ln in io.open(ROOT/"files"/"top_gainer_dataset.jsonl", encoding="utf-8", errors="replace"):
    if '"label_top20"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    ts = e.get("ts")
    if not ts or e.get("label_top20") != 1 or e.get("symbol") not in WL: continue
    top.add((datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d"), e.get("symbol")))

scanned = set()
for ln in io.open(ROOT/"files"/"bot_events.jsonl", encoding="utf-8", errors="replace"):
    if '"event"' not in ln: continue
    try: e = json.loads(ln)
    except: continue
    s = e.get("sym") or e.get("symbol")
    try: d = datetime.fromisoformat(str(e.get("ts", "")).replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except: continue
    if s in WL: scanned.add((d, s))

CRITERIA = ["S1 MA7+MACD", "S2 stack1h", "S3 MA7+RSI55", "S4 MA7+MACD+RSI55"]
fires = {c: set() for c in CRITERIA}       # (day, sym)
days_seen = set()
pol_days = defaultdict(list)

for sym, rows in H.items():
    if sym not in WL or len(rows) < 60:
        continue
    ts = np.array([r[0] for r in rows])
    hi = np.array([r[1] for r in rows]); cl = np.array([r[3] for r in rows])
    dates = [datetime.fromtimestamp(t/1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts]
    m7, m25 = ma(cl, 7), ma(cl, 25)
    e12, e26 = ema(cl, 12), ema(cl, 26)
    macd = e12 - e26; hist = macd - ema(macd, 9)
    r = rsi14(cl)

    by_day = defaultdict(list)
    for i, d in enumerate(dates):
        by_day[d].append(i)

    for d, idx in by_day.items():
        if len(idx) < 6 or d < WINDOW_START:   # partial day / outside shared window
            continue
        days_seen.add(d)
        peak_i = idx[int(np.argmax(hi[idx]))]
        for i in idx:
            if i > peak_i or i < 26:      # only before the day's high; need MA history
                continue
            c_ = cl[i]
            cond = {
                "S1 MA7+MACD":       c_ > m7[i] and hist[i] > 0,
                "S2 stack1h":        c_ > m7[i] > m25[i],
                "S3 MA7+RSI55":      c_ > m7[i] and r[i] > 55,
                "S4 MA7+MACD+RSI55": c_ > m7[i] and hist[i] > 0 and r[i] > 55,
            }
            for name, ok in cond.items():
                if ok:
                    fires[name].add((d, sym))
                    if sym == "POLUSDT":
                        pol_days[name].append(d)

nd = len(days_seen)
top_in_window = {k for k in top if k[0] in days_seen}
silent = {k for k in top_in_window if k not in scanned}
already = {k for k in scanned if k[0] in days_seen}

print("=" * 78)
print(f"Soft scan-promotion trade-off  |  window {min(days_seen)}..{max(days_seen)} "
      f"({nd} days, {len(WL)} coins)")
print(f"watchlist top-20 in window: {len(top_in_window)}   of them NEVER scanned "
      f"(silent): {len(silent)}   bot currently scans ~{len(already)/max(1,nd):.0f} coin-days/day")
print("=" * 78)
print(f"{'criterion':<20}{'promo/day':>10}{'extra/day':>10}{'top20 rec':>11}"
      f"{'silent rec':>12}{'precision':>11}")
for c in CRITERIA:
    f = fires[c]
    extra = {k for k in f if k not in scanned}
    t_rec = len(f & top_in_window) / max(1, len(top_in_window)) * 100
    s_rec = len(f & silent) / max(1, len(silent)) * 100
    prec = len(f & top_in_window) / max(1, len(f)) * 100
    print(f"{c:<20}{len(f)/nd:>10.0f}{len(extra)/nd:>10.0f}{t_rec:>10.0f}%"
          f"{s_rec:>11.0f}%{prec:>10.1f}%")
print("-" * 78)
for c in CRITERIA:
    dd = sorted(set(pol_days[c]))
    print(f"POL would be promoted on {len(dd):>2} of {nd} days by {c}"
          f"  {('e.g. ' + ', '.join(dd[-3:])) if dd else ''}")
print("\nRead: 'extra/day' is the added scan load (coins/day the bot does not watch")
print("today). 'silent rec' is the payoff — share of currently invisible top-20 the")
print("criterion would surface. 'precision' shows how much of the added load is noise.")

# RESULT (2026-08-01, 42d x 94 tradable symbols, 1h bars, fires only before the
# day's high):
#   criterion            promo/day  extra/day  top20 rec  silent rec  precision
#   S1 MA7+MACD                 70         32        86%         74%       3.3%
#   S2 close>MA7>MA25           59         23        90%         84%       4.1%   <- best
#   S3 MA7+RSI55                71         32        93%         87%       3.5%
#   S4 MA7+MACD+RSI55           67         28        86%         74%       3.4%
#
# Baseline: the bot scans ~43 coin-days/day and sees 74 of 112 top-20 (66%);
# 38 top-20 (34%) are never scanned at all. Of the top-20 it DOES scan, 58%
# convert into an actual entry.
#
# S2 costs +23 coins/day of scan load and would surface 84% of the invisible
# top-20 -> at the observed 58% conversion, ~18 extra caught top-20 per 42 days
# (~13/month, vs ~31/month caught today).
#
# HONEST CAVEATS: (1) the 58% conversion is measured on coins that reached the
# scan through momentum, i.e. exactly the ones the gates like — silent-miss coins
# are calmer by construction, so their conversion is likely LOWER (POL: 403 blocks
# and 3 short entries during the 17 days it WAS scanned). Treat ~13/month as an
# upper bound. (2) +53% scan load on a bot already logging 135 event-loop-lag
# warnings/day (up to 51s) and with a history of silent deaths — roll out capped,
# not wide open.
