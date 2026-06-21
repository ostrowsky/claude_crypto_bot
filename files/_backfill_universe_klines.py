"""Backfill 1h klines × 30 days for the full Binance USDT universe.

Used by extended correlation-cluster analysis (universe vs watchlist).
Writes to history/<sym>_1h.csv.

At 1h × 30d = 720 bars, fits in one Binance API call per coin.
427 coins × ~150 ms = ~1 minute total.
"""
from __future__ import annotations

import asyncio
import csv
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
UNIVERSE_FILE = ROOT / ".runtime" / "binance_universe.json"
HISTORY_DIR = ROOT / "history"

# Throttle: Binance allows 1200 req-weight/min. Klines = 1 weight each.
# 427 coins × 1 weight = ~430, well under limit. Add light delay to be polite.
INTER_REQ_DELAY = 0.05


async def fetch_one(session, sym: str, days: int = 30, tf: str = "1h"):
    """Fetch single coin klines for 30 d × 1 h. Returns list of bars or [] on error."""
    url = "https://api.binance.com/api/v3/klines"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    params = {"symbol": sym, "interval": tf,
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    try:
        async with session.get(url, params=params) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception:
        return None


async def main():
    import aiohttp

    if not UNIVERSE_FILE.exists():
        print("[ERR] run _fetch_binance_universe.py first")
        return

    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    syms = data["symbols"]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[backfill] universe size: {len(syms)} symbols, 1h × 30d each")

    cached = empty = fail = 0
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sym in enumerate(syms, 1):
            out_path = HISTORY_DIR / f"{sym}_1h.csv"
            rows = await fetch_one(session, sym)
            if rows is None:
                fail += 1
                continue
            if not rows:
                empty += 1
                continue
            try:
                with io.open(out_path, "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for row in rows:
                        ts_ms = int(row[0])
                        ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
                        f.write(f"{ts_iso},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
                cached += 1
            except Exception as e:
                print(f"  [{i}/{len(syms)}] write fail {sym}: {e}")
                fail += 1
                continue
            if i % 50 == 0 or i == len(syms):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(syms)}] cached={cached} empty={empty} fail={fail}  ({elapsed:.0f}s)")
            await asyncio.sleep(INTER_REQ_DELAY)

    print(f"\n[backfill] done in {time.time()-t0:.0f}s")
    print(f"  cached: {cached}")
    print(f"  empty:  {empty}")
    print(f"  fail:   {fail}")


if __name__ == "__main__":
    asyncio.run(main())
