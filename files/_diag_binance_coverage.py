"""Check how many USDT perp futures exist on Binance vs our watchlist."""
from __future__ import annotations
import json, io, sys, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
FILES = Path(__file__).resolve().parent

# Binance exchange info
with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10) as r:
    info = json.load(r)
syms_usdt = [
    s["symbol"] for s in info["symbols"]
    if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s["contractType"] == "PERPETUAL"
]
print(f"Binance USDT-M perp futures (TRADING, PERPETUAL): {len(syms_usdt)}")

# 24h tickers for volume
with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10) as r:
    tickers = json.load(r)
usdt_t = [t for t in tickers if t["symbol"].endswith("USDT")]
usdt_t.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
print(f"With 24h data: {len(usdt_t)}")

print("\nTop 15 by 24h volume:")
for t in usdt_t[:15]:
    vol = float(t["quoteVolume"]) / 1e6
    chg = float(t["priceChangePercent"])
    print(f"  {t['symbol']:20s}  vol={vol:>8.0f}M  chg={chg:+.1f}%")

# Watchlist overlap
wl = json.loads((FILES / "watchlist.json").read_text(encoding="utf-8"))
wl_set = set(wl)
all_set = set(syms_usdt)
top100_set = set(t["symbol"] for t in usdt_t[:100])
top200_set = set(t["symbol"] for t in usdt_t[:200])
top300_set = set(t["symbol"] for t in usdt_t[:300])

print(f"\nWatchlist: {len(wl_set)} symbols")
print(f"Binance all perps: {len(all_set)}")
print(f"WL overlap with all: {len(wl_set & all_set)}")
print(f"WL overlap with top100 vol: {len(wl_set & top100_set)}")
print(f"WL overlap with top200 vol: {len(wl_set & top200_set)}")
print(f"WL NOT in top200: {sorted(wl_set - top200_set)}")

# How many top-100 coins NOT in watchlist
top100_not_wl = top100_set - wl_set
print(f"\nTop 100 coins NOT in watchlist: {len(top100_not_wl)}")
print(sorted(top100_not_wl)[:20])

# What fraction of daily top-gainers are outside watchlist?
# Simulate: who would be in top-20 today among ALL vs watchlist
print("\n=== Daily top-20 gainer distribution ===")
top20_all = [t["symbol"] for t in usdt_t[:len(usdt_t)]]
# Sort by priceChangePercent
usdt_t_sorted = sorted(usdt_t, key=lambda t: float(t["priceChangePercent"]), reverse=True)
top20_today = [t["symbol"] for t in usdt_t_sorted[:20]]
top20_today_wl = [s for s in top20_today if s in wl_set]
print(f"Today's top-20 gainers (all Binance): {top20_today}")
print(f"Of those, in watchlist: {len(top20_today_wl)} / 20")
print(f"WL top-20: {top20_today_wl}")

# Potential missed if watchlist were larger
missed = [s for s in top20_today if s not in wl_set]
print(f"Missed (not in WL): {missed}")
