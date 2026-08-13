"""Extend the local 1h kline cache so backtests run on the max available period.

The existing `_hourly_ohlcv.json` holds 1000 bars per symbol (~42 days), and 11
of the 105 symbols have gappy series that silently poison a time split. This
pulls a long contiguous window per watchlist symbol from Binance futures and
writes `_hourly_ohlcv_long.json` (list of [openTime, o, h, l, c, volume]).

Read-only against the repo; network fetch only.
    pyembed\python.exe files\_fetch_hourly_history.py [days]
"""
from __future__ import annotations
import io, json, sys, time
from pathlib import Path
from urllib.request import urlopen

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NEED = DAYS * 24
API = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval=1h&limit=1500&endTime={e}"

WL = json.load(io.open(ROOT/"watchlist.json", encoding="utf-8"))
out: dict[str, list] = {}
now_ms = int(time.time()*1000)

for n, sym in enumerate(WL, 1):
    bars: dict[int, list] = {}
    end = now_ms
    for _ in range(NEED//1500 + 1):
        try:
            with urlopen(API.format(s=sym, e=end), timeout=20) as r:
                chunk = json.load(r)
        except Exception as exc:
            print(f"  {sym}: {exc}"); break
        if not chunk:
            break
        for k in chunk:
            bars[int(k[0])] = [int(k[0]), float(k[1]), float(k[2]),
                               float(k[3]), float(k[4]), float(k[5])]
        end = int(chunk[0][0]) - 1
        if len(bars) >= NEED:
            break
        time.sleep(0.12)
    if bars:
        out[sym] = [bars[t] for t in sorted(bars)][-NEED:]
    print(f"[{n:>3}/{len(WL)}] {sym:<14} {len(out.get(sym, []))} баров", flush=True)

dst = ROOT/"_hourly_ohlcv_long.json"
json.dump(out, io.open(dst, "w", encoding="utf-8"))
lens = sorted(len(v) for v in out.values())
print(f"\n{len(out)} монет -> {dst.name} · баров: min {lens[0]} медиана "
      f"{lens[len(lens)//2]} max {lens[-1]} (~{lens[len(lens)//2]/24:.0f} дней)")
