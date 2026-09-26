"""Fetch taker-buy base volume for every watchlist coin's 15m history (Binance spot klines field 9).

The long 15m store keeps OHLCV only; the taker-buy share -- how much of the
volume was aggressive buying -- is the one order-flow measure Binance serves
for the full history, so it is the only flow hypothesis testable on 455 days.
Output: .runtime/backtests/taker_15m/<SYM>.json  {open_time_ms: [volume, taker_buy_volume]}
Resumable: coins already fetched to the store's last bar are skipped.
"""
import asyncio
import io
import json
import sys
from pathlib import Path

import aiohttp

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _backtest_trend_start_detector as TD  # noqa: E402
import _compute_early_capture as E  # noqa: E402

OUT = FILES.parent / ".runtime" / "backtests" / "taker_15m"
URL = "https://api.binance.com/api/v3/klines"


async def one(session, sym, sem):
    b = TD.bars_15m(sym)
    if len(b) < 2000:
        return sym, 0
    start = int(b[0][0].timestamp() * 1000)
    end = int(b[-1][0].timestamp() * 1000) + 1
    fp = OUT / (sym + ".json")
    if fp.exists():
        have = json.loads(fp.read_text(encoding="utf-8"))
        if have and max(map(int, have)) >= end - 1:
            return sym, len(have)
    rows = {}
    cur = start
    async with sem:
        while cur < end:
            params = {"symbol": sym, "interval": "15m", "startTime": cur, "endTime": end, "limit": 1000}
            for attempt in range(5):
                try:
                    async with session.get(URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
                        if r.status == 429 or r.status == 418:
                            await asyncio.sleep(30)
                            continue
                        r.raise_for_status()
                        js = await r.json()
                        break
                except Exception:
                    await asyncio.sleep(2 + attempt * 3)
            else:
                break
            if not js:
                break
            for k in js:
                rows[int(k[0])] = [float(k[5]), float(k[9])]
            nxt = int(js[-1][0]) + 900_000
            if nxt <= cur:
                break
            cur = nxt
            await asyncio.sleep(0.05)
    fp.write_text(json.dumps(rows), encoding="utf-8")
    return sym, len(rows)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*(one(s, sym, sem) for sym in E.load_watchlist()))
    print("coins:", len(res), "bars:", sum(n for _, n in res), "empty:", [s for s, n in res if n == 0])


asyncio.run(main())
