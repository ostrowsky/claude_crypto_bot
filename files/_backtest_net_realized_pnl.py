"""Daily metric: realized per-trade PnL, GROSS vs NET-of-fees.

Backtests/realized figures were gross. Validated 2026-06-17 that round-trip
fees (Binance USDT-M taker ~0.05%/side = config.FEE_ROUNDTRIP_PCT) are material
(net per-trade ~ -0.10pp vs gross) and that the bot's average trade is
net-negative after fees. Per honest-measurement (CLAUDE.md s0.3) we surface
BOTH so the gap is never hidden: gross stays comparable to history, net tells
the truth a channel follower actually experiences.

Source: critic_dataset.jsonl trade_exit_pnl on watchlist take entries, last N
days (the tradeable universe, consistent with the canonical NS fix).

ASCII-only. Emits one METRIC_JSON line for report_metrics_daily.py.
    pyembed\python.exe files\_backtest_net_realized_pnl.py
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DAYS = 14
CUT = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime("%Y-%m-%d")

try:
    import config as _cfg
    FEE = float(getattr(_cfg, "FEE_ROUNDTRIP_PCT", 0.10))
except Exception:
    FEE = 0.10

try:
    WL = set(json.load(io.open(ROOT/"files"/"watchlist.json", encoding="utf-8")))
except Exception:
    WL = set()


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


pnls = []
for ln in io.open(ROOT/"files"/"critic_dataset.jsonl", encoding="utf-8", errors="replace"):
    if "trade_exit_pnl" not in ln:
        continue
    try: e = json.loads(ln)
    except: continue
    if str(e.get("ts_signal", ""))[:10] < CUT:
        continue
    if WL and e.get("sym") not in WL:
        continue
    if str((e.get("decision", {}) or {}).get("action", "")) != "take":
        continue
    p = _f((e.get("labels", {}) or {}).get("trade_exit_pnl"))
    if p is not None:
        pnls.append(p)

n = len(pnls)
if n == 0:
    print("no resolved trades in window")
    print("METRIC_JSON:" + json.dumps({"metric": "NET_realized_pnl", "n": 0}))
    sys.exit(0)

gross_avg = sum(pnls) / n
net_avg = gross_avg - FEE
gross_sum = sum(pnls)
net_sum = gross_sum - FEE * n
gross_win = sum(1 for p in pnls if p > 0) / n * 100
net_win = sum(1 for p in pnls if p - FEE > 0) / n * 100

print("=" * 60)
print(f"Net realized PnL (watchlist take entries, {DAYS}d, n={n})")
print("=" * 60)
print(f"  fee_roundtrip   = {FEE:.2f}%")
print(f"  avg/trade GROSS = {gross_avg:+.3f}%   NET = {net_avg:+.3f}%")
print(f"  sum       GROSS = {gross_sum:+.1f}%    NET = {net_sum:+.1f}%")
print(f"  win%      GROSS = {gross_win:.1f}     NET = {net_win:.1f}")

metric = {
    "metric": "NET_realized_pnl",
    "n": n,
    "fee_roundtrip_pct": FEE,
    "avg_pnl_gross_pct": round(gross_avg, 4),
    "avg_pnl_net_pct": round(net_avg, 4),
    "sum_pnl_gross_pct": round(gross_sum, 2),
    "sum_pnl_net_pct": round(net_sum, 2),
    "win_pct_gross": round(gross_win, 1),
    "win_pct_net": round(net_win, 1),
}
print("METRIC_JSON:" + json.dumps(metric))
