"""Backfill 419d x 15m klines for the watchlist, to ask the earliness question
at a resolution that can answer it.

On 1h bars the median caught +20% trend runs 7 hours, so its opening hour is one
observation in seven and has almost no shape to describe. Every attempt to make
the detector fire earlier failed identically on that grid: the start label at
window 1/2/3/6h left `into%` flat at 40-46%, and forcing RSI < 65 bought
into% 32% at the cost of three quarters of all catches. At 15m the same trend
carries 28 observations and its first half hour becomes describable.

419 days matches the 1h experiments exactly so the two are comparable (TH-04).
40 224 bars per symbol > the 1000-bar API limit, so we page in 1000-bar chunks:
~41 calls x 99 symbols ~ 4060 calls, ~2 weight each against a 6000/min budget.

Writes history/<sym>_15m_419d.csv, a new name so the 30d `_15m` and 90d
`_15m_90d` caches keep working untouched. Resumable: a symbol whose file already
covers the window is skipped, so an interrupted run costs only what it had not
yet fetched.
"""
from __future__ import annotations

import asyncio
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY_DIR = ROOT / "history"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

INTER_REQ_DELAY = 0.06   # ~2000 weight/min against a 6000/min budget
DAYS = 419
TF_MS = 900 * 1000       # 15m
INTERVAL = "15m"
EXPECT = DAYS * 24 * 4   # 40 224 bars


def already_complete(path: Path) -> bool:
    """A file counts as done only if it is long enough AND ends recently.

    Length alone would accept a truncated run that stopped early, and recency
    alone would accept a file holding only the last week.
    """
    if not path.exists():
        return False
    try:
        with io.open(path, encoding="utf-8") as fh:
            lines = sum(1 for _ in fh) - 1
        if lines < EXPECT * 0.90:
            return False
        with io.open(path, encoding="utf-8") as fh:
            last = fh.readlines()[-1]
        ts = datetime.fromisoformat(last.split(",")[0])
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return age_h < 48
    except Exception:
        return False


async def fetch_chunk(session, sym, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": sym, "interval": INTERVAL,
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    for attempt in range(3):
        try:
            async with session.get(url, params=params) as r:
                if r.status == 429:
                    await asyncio.sleep(5.0 * (attempt + 1))
                    continue
                if r.status != 200:
                    return None
                return await r.json()
        except Exception:
            await asyncio.sleep(1.0 * (attempt + 1))
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
    import _diag_uptrend_population as UP

    syms = UP.watchlist()
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    print("[15m/419d] %d watchlist symbols, expecting ~%d bars each" % (len(syms), EXPECT))

    done = skipped = empty = fail = 0
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sym in enumerate(syms, 1):
            out_path = HISTORY_DIR / ("%s_15m_419d.csv" % sym)
            if already_complete(out_path):
                skipped += 1
                continue
            bars = await fetch_all(session, sym)
            if not bars:
                empty += 1
                print("  no data: %s" % sym)
                continue
            seen = {}
            for row in bars:
                seen[int(row[0])] = row
            rows = [seen[k] for k in sorted(seen)]
            try:
                tmp = out_path.with_suffix(".csv.part")
                with io.open(tmp, "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for row in rows:
                        ts_iso = datetime.fromtimestamp(
                            int(row[0]) / 1000, tz=timezone.utc).isoformat()
                        f.write("%s,%s,%s,%s,%s,%s\n"
                                % (ts_iso, row[1], row[2], row[3], row[4], row[5]))
                tmp.replace(out_path)   # atomic: no half file is ever readable
                done += 1
            except Exception as e:
                print("  write fail %s: %s" % (sym, e))
                fail += 1
                continue
            if i % 10 == 0 or i == len(syms):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                left = (len(syms) - i) / rate / 60 if rate > 0 else 0
                print("  [%d/%d] done=%d skip=%d empty=%d fail=%d  %.0fs  ~%.0fm left"
                      % (i, len(syms), done, skipped, empty, fail, el, left))

    print("\n[15m/419d] finished in %.0fs  done=%d skipped=%d empty=%d fail=%d"
          % (time.time() - t0, done, skipped, empty, fail))
    tot = sum(1 for _ in HISTORY_DIR.glob("*_15m_419d.csv"))
    print("[15m/419d] %d symbols now hold a 419-day 15m cache" % tot)


if __name__ == "__main__":
    asyncio.run(main())
