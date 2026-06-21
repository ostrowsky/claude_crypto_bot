"""Backfill 90d × 15m klines for watchlist + a few external peers.

Finer timeframe to test sub-hour BTC→alt lead-lag (1h was too coarse,
absorbed the propagation). 90d × 15m = 8640 bars → ~9 pages/coin.
~110 coins × 9 ≈ 1000 calls.

Writes history/<sym>_15m_90d.csv.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_FILE = ROOT / "files" / "watchlist.json"
HISTORY_DIR = ROOT / "history"
INTER_REQ_DELAY = 0.04
DAYS = 90
TF_MS = 15 * 60 * 1000  # 15m

# external peers worth tracking as leading-indicator candidates
EXTRA = ["NOTUSDT", "XAUTUSDT", "RAYUSDT", "BNSOLUSDT", "STGUSDT",
         "1000SATSUSDT", "CVXUSDT", "PIXELUSDT", "SUNUSDT", "ARKMUSDT",
         "MOVRUSDT", "GALAUSDT", "RONINUSDT", "PEPEUSDT"]


async def fetch_chunk(session, sym, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": sym, "interval": "15m",
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    try:
        async with session.get(url, params=params) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None


async def fetch_all(session, sym):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - DAYS * 24 * 3600 * 1000
    bars = []
    cur = start_ms
    while cur < end_ms:
        chunk = await fetch_chunk(session, sym, cur, end_ms)
        if not chunk:
            break
        bars.extend(chunk)
        nxt = int(chunk[-1][0]) + TF_MS
        if nxt <= cur or len(chunk) < 1000:
            break
        cur = nxt
        await asyncio.sleep(INTER_REQ_DELAY)
    return bars


async def main():
    import aiohttp
    wl = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    syms = sorted(set(wl) | set(EXTRA))
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[backfill15m] {len(syms)} symbols, 15m × {DAYS}d")
    cached = empty = fail = 0
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sym in enumerate(syms, 1):
            bars = await fetch_all(session, sym)
            if not bars:
                (empty if bars == [] else fail).__class__  # noop
                if bars is None: fail += 1
                else: empty += 1
                continue
            seen = {int(r[0]): r for r in bars}
            rows = [seen[k] for k in sorted(seen)]
            try:
                with io.open(HISTORY_DIR / f"{sym}_15m_90d.csv", "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for r in rows:
                        ts_iso = datetime.fromtimestamp(int(r[0]) / 1000, tz=timezone.utc).isoformat()
                        f.write(f"{ts_iso},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
                cached += 1
            except Exception as e:
                print(f"  write fail {sym}: {e}"); fail += 1; continue
            if i % 20 == 0 or i == len(syms):
                print(f"  [{i}/{len(syms)}] cached={cached} empty={empty} fail={fail} ({time.time()-t0:.0f}s)")
            await asyncio.sleep(INTER_REQ_DELAY)
    print(f"\n[backfill15m] done in {time.time()-t0:.0f}s cached={cached} empty={empty} fail={fail}")


if __name__ == "__main__":
    asyncio.run(main())
