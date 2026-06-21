"""Backfill 365d × 1h klines for the full Binance USDT universe (paginated).

8760 bars per coin > 1000 API limit, so we page in 1000-bar chunks
(9 calls/coin). 427 coins × 9 ≈ 3800 calls; at 40ms throttle ≈ 3 min.

Writes history/<sym>_1h_365d.csv (separate from the 30d cache so the
short-window cluster script keeps working).
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
UNIVERSE_FILE = ROOT / ".runtime" / "binance_universe.json"
HISTORY_DIR = ROOT / "history"
INTER_REQ_DELAY = 0.04
DAYS = 365
TF_MS = 3600 * 1000  # 1h


async def fetch_chunk(session, sym, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": sym, "interval": "1h",
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
    start_ms = end_ms - DAYS * 24 * TF_MS
    bars = []
    cur = start_ms
    while cur < end_ms:
        chunk = await fetch_chunk(session, sym, cur, end_ms)
        if not chunk:
            break
        bars.extend(chunk)
        last_open = int(chunk[-1][0])
        nxt = last_open + TF_MS
        if nxt <= cur or len(chunk) < 1000:
            break
        cur = nxt
        await asyncio.sleep(INTER_REQ_DELAY)
    return bars


async def main():
    import aiohttp
    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    syms = data["symbols"]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[backfill365] {len(syms)} symbols, 1h × {DAYS}d")

    cached = empty = fail = 0
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sym in enumerate(syms, 1):
            out_path = HISTORY_DIR / f"{sym}_1h_365d.csv"
            bars = await fetch_all(session, sym)
            if bars is None:
                fail += 1; continue
            if not bars:
                empty += 1; continue
            # dedup by open ts
            seen = {}
            for row in bars:
                seen[int(row[0])] = row
            rows = [seen[k] for k in sorted(seen)]
            try:
                with io.open(out_path, "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for row in rows:
                        ts_iso = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).isoformat()
                        f.write(f"{ts_iso},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
                cached += 1
            except Exception as e:
                print(f"  write fail {sym}: {e}"); fail += 1; continue
            if i % 25 == 0 or i == len(syms):
                print(f"  [{i}/{len(syms)}] cached={cached} empty={empty} fail={fail}  ({time.time()-t0:.0f}s)")
            await asyncio.sleep(INTER_REQ_DELAY)

    print(f"\n[backfill365] done in {time.time()-t0:.0f}s  cached={cached} empty={empty} fail={fail}")


if __name__ == "__main__":
    asyncio.run(main())
