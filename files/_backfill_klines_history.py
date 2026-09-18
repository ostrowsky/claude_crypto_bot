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


LONG_SUFFIX = {"15m": "15m_419d", "1h": "1h_365d"}
STEP = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1)}


def extend_long_store(sym: str, tf: str = "15m") -> tuple[int, str]:
    """Append bars from the rolling <sym>_15m.csv onto the long <sym>_15m_419d.csv.

    Added 2026-09-18. The long file was a one-off 419-day backfill that ended
    2026-08-20, while this task rewrites a separate ROLLING 30-day file. Every
    15m reader that wants history (the backtests, and the peak TRAINING label in
    ml_signal_model) reads the long file -- so from 08-20 on they silently saw
    nothing: the peak label resolved 48% of August's 15m rows and 0% of
    September's, and those rows fell back to the old inverted ret_5>0 label.
    Extending the long file daily makes it a growing store and fixes both.

    Append-only: bars at or before the long file's last timestamp are never
    rewritten, so earlier results computed on it stay reproducible. A gap
    between the two files is reported, not papered over -- it means the rolling
    window has moved past the long file's end and the bars in between must be
    re-fetched (Binance serves historical klines, so nothing is lost for good).
    Returns (bars appended, status).
    """
    long_p = HISTORY_DIR / f"{sym}_{LONG_SUFFIX[tf]}.csv"
    roll_p = HISTORY_DIR / f"{sym}_{tf}.csv"
    if not long_p.exists() or not roll_p.exists():
        return 0, "missing"
    long_lines = io.open(long_p, encoding="utf-8").read().splitlines()
    if len(long_lines) < 2:
        return 0, "empty"
    last_ts = long_lines[-1].split(",", 1)[0]
    roll_lines = io.open(roll_p, encoding="utf-8").read().splitlines()[1:]
    new = [ln for ln in roll_lines if ln and ln.split(",", 1)[0] > last_ts]
    if not new:
        return 0, "current"
    status = "ok"
    try:
        gap = (datetime.fromisoformat(new[0].split(",", 1)[0])
               - datetime.fromisoformat(last_ts))
        if gap > STEP[tf]:
            status = "GAP %s" % gap
    except ValueError:
        status = "unparsed"
    tmp = long_p.with_name(long_p.name + ".part")
    with io.open(tmp, "w", encoding="utf-8", newline=chr(10)) as f:
        f.write(chr(10).join(long_lines + new) + chr(10))
    tmp.replace(long_p)
    return len(new), status


def extend_all(symbols: list[str], tf: str = "15m") -> None:
    added = gaps = 0
    for sym in symbols:
        n, st = extend_long_store(sym, tf)
        added += n
        if st.startswith("GAP"):
            gaps += 1
            print(f"  [extend] {sym}: {st} before the appended bars -- re-fetch that span")
    print(f"[extend] long {tf} store: {added} bars appended across {len(symbols)} "
          f"symbols, {gaps} with a gap")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--tf", type=str, default="15m")
    p.add_argument("--symbols", type=str, default=None,
                   help="comma-separated subset; default = full watchlist")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip if cache already covers the window and is < 6h old")
    p.add_argument("--extend-only", action="store_true",
                   help="only append the rolling file onto the long store (15m, 1h)")
    args = p.parse_args()

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        syms = load_watchlist()

    if not args.extend_only:
        asyncio.run(backfill(syms, args.tf, args.days, args.skip_existing))
    if args.tf in LONG_SUFFIX:
        extend_all(syms, args.tf)


if __name__ == "__main__":
    main()
