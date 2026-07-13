"""Validate a 1d "trend start" info-alert (POL-type slow grinds are structurally
invisible to the bot's 15m/1h momentum modes — ADX ~20, slope <0.7%, range <4%).

Detector on daily bars (aggregated from 15m klines), evaluated at day close (no
lookahead): ALERT when
    close > MA7 > MA25         (bullish daily stack)
AND the stack is FRESH (condition false on any of the prior 2 days)
AND close > close[1]           (confirming up day).

Report over the full kline history x watchlist:
  - alerts/day (channel noise budget)
  - forward returns after alert (+7d, +14d), win rates
  - how many alerts precede a >=10% 7d move (useful) vs <2% (noise)
  - POLUSDT case: which date it would have alerted (trend began 2026-07-01)
Read-only.  pyembed\python.exe files\_backtest_daily_trend_alert.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))


def daily_bars(path):
    """15m csv -> list of (date_str, close). Uses last 15m close of each UTC day."""
    days = {}
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                t = datetime.fromisoformat(r["ts"]); c = float(r["close"])
            except Exception:
                continue
            days[t.strftime("%Y-%m-%d")] = c   # ordered file -> last wins
    return sorted(days.items())


def ma(a, n, i):
    return float(np.mean(a[i-n+1:i+1])) if i >= n-1 else None


alerts = []          # (date, sym, fwd7, fwd14)
all_days = set()
pol_alerts = []
for p in sorted(HIST.glob("*_15m.csv")):
    sym = p.name[:-8]
    if sym not in WL:
        continue
    bars = daily_bars(p)
    if len(bars) < 30:
        continue
    dates = [b[0] for b in bars]; closes = np.array([b[1] for b in bars])
    all_days.update(dates)

    def stack(i):
        m7 = ma(closes, 7, i); m25 = ma(closes, 25, i)
        return (m7 is not None and m25 is not None
                and closes[i] > m7 > m25)

    for i in range(26, len(bars)):
        if not (stack(i) and closes[i] > closes[i-1]):
            continue
        if stack(i-1) or stack(i-2):     # not fresh
            continue
        f7 = (closes[i+7]/closes[i]-1)*100 if i+7 < len(bars) else None
        f14 = (closes[i+14]/closes[i]-1)*100 if i+14 < len(bars) else None
        alerts.append((dates[i], sym, f7, f14))
        if sym == "POLUSDT":
            pol_alerts.append(dates[i])

n_days = len(all_days)
f7s = [a[2] for a in alerts if a[2] is not None]
f14s = [a[3] for a in alerts if a[3] is not None]
big7 = sum(1 for x in f7s if x >= 10)
mid7 = sum(1 for x in f7s if 2 <= x < 10)
noise7 = sum(1 for x in f7s if x < 2)
print("=" * 66)
print(f"1d trend-start alert  ·  {len(alerts)} alerts over {n_days} days "
      f"({len(alerts)/max(1,n_days):.1f}/day across ~{len(WL)} coins)")
print(f"forward +7d : mean {np.mean(f7s):+.2f}%  median {np.median(f7s):+.2f}%  "
      f"win {100*sum(1 for x in f7s if x>0)/max(1,len(f7s)):.0f}%  (n={len(f7s)})")
print(f"forward +14d: mean {np.mean(f14s):+.2f}%  median {np.median(f14s):+.2f}%  "
      f"win {100*sum(1 for x in f14s if x>0)/max(1,len(f14s)):.0f}%  (n={len(f14s)})")
print(f"alert quality (+7d): >=10% move: {big7} ({100*big7/max(1,len(f7s)):.0f}%)   "
      f"2-10%: {mid7} ({100*mid7/max(1,len(f7s)):.0f}%)   <2% (noise): {noise7} "
      f"({100*noise7/max(1,len(f7s)):.0f}%)")
print(f"\nPOLUSDT alert dates: {pol_alerts}  (trend began 2026-07-01; bot's first"
      f"\n  actual entry was 2026-07-07 — measure the earliness gain)")
print("=" * 66)
print("USEFUL if: alerts/day is a sane channel load (~<5), forward +7d clearly")
print("positive, and POL would have been flagged near 07-02..03.")

# VERDICT (2026-07-13, 420d x 105 coins via Binance 1d klines, no lookahead):
# REFUTED. All rule variants (MA7>MA25 fresh / 5d above rising MA7 / steady
# 6-25%-per-7d grind) produce ~0.7-1.3 alerts/day with NEGATIVE forward returns
# (+7d mean -1.6..-2.0%, win 34-38%, 68-75% noise<2%). Fresh daily up-stacks on
# alts mostly mean-revert; POL's 13-day grind is survivorship. Best rule would
# have flagged POL only 2 days before the bot's actual first entry (07-05 vs
# 07-07) while spamming a losing signal class. Do NOT build the 1d alert tier.
