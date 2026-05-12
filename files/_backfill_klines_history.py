"""Backfill 30 d × all watchlist coins × <tf> klines into history/<sym>_<tf>.csv.

Used by:
  - _backtest_ex1_realized_potential.py --use-zigzag (Phase D EX1 honest mode)
  - _run_signal_evaluator.py (skill needs cache, no `requests` available)
  - _backfill_sustained_uptrend (future Phase B continuation)

Idempotent: skips symbols where cache file already covers the requested
window. Run weekly to keep cache fresh.

Usage:
  pyembed/python.exe files/_backfill_klines_history.py
  pyembed/python.exe files/_backfill_klines_history.py --days 60 --tf 15m
  pyembed/python.exe files/_backfill_klines_history.py --skip-existing
"""
from __future__ import annotations
import argparse, asyncio, io, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
WATCHLIST = ROOT / "files" / "watchlist.json"


def load_watchlist() -> list[str]:
    with io.open(WATCHLIST, encoding="utf-8") as f:
        return json.load(f)


def cache_covers_window(path: Path, start: datetime, end: datetime,
                        max_age_h: float = 6.0) -> bool:
    """Return True if cache file exists, covers the window, and is fresh."""
    if not path.exists(): return False
    try:
        # Cheap: read first and last lines, parse ts
        with io.open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 3: return False
        # First data line index is 1 (0 is header)
        first_ts_str = lines[1].split(",", 1)[0]
        last_ts_str = lines[-1].split(",", 1)[0]
        first_ts = datetime.fromisoformat(first_ts_str)
        last_ts = datetime.fromisoformat(last_ts_str)
        if first_ts > start: return False  # cache starts too late
        if (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600 > max_age_h:
            return False  # cache stale
        return True
    except Exception:
        return False


async def fetch_paginated(session, symbol: str, tf: str,
                          start_ms: int, end_ms: int) -> list[list]:
    """Fetch klines with pagination (Binance API max 1500/call).
    Returns raw list-of-lists (each: [ts, o, h, l, c, v, ...]).
    """
    url = "https://api.binance.com/api/v3/klines"
    out = []
    cur = start_ms
    LIMIT = 1000  # Binance spot max per call
    bar_ms = {"1m":60_000,"5m":300_000,"15m":900_000,"30m":1_800_000,
              "1h":3_600_000,"4h":14_400_000,"1d":86_400_000}.get(tf, 900_000)
    while cur < end_ms:
        params = {"symbol": symbol, "interval": tf,
                  "startTime": cur, "endTime": end_ms, "limit": LIMIT}
        try:
            import aiohttp as _a
            async with session.get(url, params=params,
                                   timeout=_a.ClientTimeout(total=30)) as r:
                r.raise_for_status()
                js = await r.json()
        except Exception as e:
            print(f"    page fetch fail at {cur}: {e}")
            break
        if not isinstance(js, list) or not js:
            break
        out.extend(js)
        last_ts = int(js[-1][0])
        if last_ts <= cur:
            break
        # Continue from next bar after last fetched
        next_start = last_ts + bar_ms
        if next_start >= end_ms:
            break
        cur = next_start
    return out


async def backfill(symbols: list[str], tf: str, days: int,
                   skip_existing: bool = False) -> tuple[int, int]:
    """Returns (cached_count, skipped_count)."""
    import aiohttp

    HISTORY_DIR.mkdir(exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    cached = skipped = 0
    print(f"[backfill] {len(symbols)} symbols × {days}d × {tf} → {HISTORY_DIR}")

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, sym in enumerate(symbols, 1):
            cache_path = HISTORY_DIR / f"{sym}_{tf}.csv"
            if skip_existing and cache_covers_window(cache_path, start, end):
                skipped += 1
                if i % 10 == 0:
                    print(f"  [{i:3d}/{len(symbols)}] skipped (fresh): {sym}")
                continue
            try:
                rows = await fetch_paginated(session, sym, tf, start_ms, end_ms)
                if not rows:
                    print(f"  [{i:3d}/{len(symbols)}] EMPTY: {sym}")
                    continue
                with io.open(cache_path, "w", encoding="utf-8") as f:
                    f.write("ts,open,high,low,close,volume\n")
                    for row in rows:
                        ts_ms = int(row[0])
                        ts_iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()
                        f.write(f"{ts_iso},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
                cached += 1
                if i <= 5 or i % 10 == 0 or i == len(symbols):
                    print(f"  [{i:3d}/{len(symbols)}] cached: {sym} ({len(rows)} bars)")
            except Exception as e:
                print(f"  [{i:3d}/{len(symbols)}] FAIL {sym}: {e}")

    print(f"[backfill] done: {cached} cached, {skipped} skipped")
    return cached, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--tf", type=str, default="15m")
    p.add_argument("--symbols", type=str, default=None,
                   help="comma-separated subset; default = full watchlist")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip if cache already covers the window and is < 6h old")
    args = p.parse_args()

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        syms = load_watchlist()

    asyncio.run(backfill(syms, args.tf, args.days, args.skip_existing))


if __name__ == "__main__":
    main()
