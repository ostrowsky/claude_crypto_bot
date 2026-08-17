"""Fetch daily klines for the global USDT universe and fill the label store.

The original `label_topN` ranks ALL USDT pairs from `/api/v3/ticker/24hr` at the
moment of the snapshot. The store held only watchlist symbols, so it could
reproduce "top-N within the watchlist" and nothing global — which collapsed the
tiers (top20 and top50 came out byte-identical) and kept the North Star's two
values on incomparable denominators.

This fills the gap at daily resolution: one request per symbol returns 1000
days, so the whole universe costs ~750 requests. Intraday timing is NOT
reconstructed — see `label_store.build_day_record_daily`.

Existing (symbol, day) records are never touched: hourly records are richer, and
the store forbids overwriting in any case.

    pyembed\\python.exe files\\build_global_labels.py --days 240
    pyembed\\python.exe files\\build_global_labels.py --days 240 --dry-run

Spec: docs/specs/features/global-label-universe-spec.md
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import label_store as LS  # noqa: E402

BINANCE = "https://api.binance.com"
# A Binance pair is ASCII upper-alnum. The dataset contains at least one row
# whose `symbol` is not, and feeding it into a URL raised UnicodeEncodeError
# 500 symbols into a 734-symbol run — one bad row killed the whole build.
VALID_SYMBOL = re.compile(r"^[A-Z0-9]{2,20}USDT$")
DATASET = HERE / "top_gainer_dataset.jsonl"
PAUSE_S = 0.12                     # ~8 req/s, well inside the weight budget
MAX_RETRIES = 3


def _get(url: str, *, timeout: int = 30):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {MAX_RETRIES}: {url} — {last}")


def live_universe() -> set[str]:
    return {t["symbol"] for t in _get(f"{BINANCE}/api/v3/ticker/24hr")
            if t["symbol"].endswith("USDT")}


def dataset_universe(path: Path = DATASET) -> set[str]:
    """Symbols the dataset itself saw.

    The live ticker list has no delisted pairs, so a past day's global ranking
    would silently omit coins that were live then. These recover the ones the
    exchange still serves klines for; the rest are unrecoverable (TH-05).
    """
    out: set[str] = set()
    if not path.exists():
        return out
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            sym = json.loads(line).get("symbol")
        except json.JSONDecodeError:
            continue
        if isinstance(sym, str) and VALID_SYMBOL.match(sym):
            out.add(sym)
    return out


def daily_bars(symbol: str, start_ms: int) -> list[list]:
    url = (f"{BINANCE}/api/v3/klines?symbol={symbol}&interval=1d"
           f"&startTime={start_ms}&limit=1000")
    try:
        raw = _get(url)
    except RuntimeError:
        return []
    # [open_ts, open, high, low, close, volume, close_ts, ...]
    return [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]),
             float(b[5])] for b in raw]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-symbols", type=int, default=0,
                    help="cap the universe, for a smoke run")
    args = ap.parse_args(argv)

    cut = datetime.now(timezone.utc) - timedelta(days=args.days)
    start_ms = int(cut.replace(hour=0, minute=0, second=0,
                               microsecond=0).timestamp() * 1000)

    live = live_universe()
    seen = dataset_universe()
    universe = sorted(live | seen)
    if args.limit_symbols:
        universe = universe[:args.limit_symbols]

    print(f"universe: {len(universe)} symbols "
          f"({len(live)} live, {len(seen - live)} delisted-or-absent from dataset)")
    print(f"window:   {cut.strftime('%Y-%m-%d')} .. today  ({args.days}d)")
    if args.dry_run:
        print("dry run — no fetch, no write")
        return 0

    store = LS.LabelStore()
    existing = {(r["symbol"], r["utc_day"]) for r in store.records()}
    print(f"store already holds {len(existing)} (symbol, day) records\n")

    resolved: list[str] = []
    failed: list[str] = []
    written = skipped = 0
    t0 = time.time()

    for i, sym in enumerate(universe, 1):
        try:
            bars = daily_bars(sym, start_ms)
        except Exception as exc:                  # noqa: BLE001 — see below
            # Deliberately broad: this loop is a long unattended fetch, and the
            # cost of one unexpected symbol must be that symbol, not the run.
            print(f"  [{sym}] skipped: {type(exc).__name__}: {exc}")
            bars = []
        if not bars:
            failed.append(sym)
        else:
            resolved.append(sym)
            for bar in bars:
                key = (sym, LS._utc_day(bar[0]))
                if key in existing:
                    skipped += 1          # hourly record wins: it is richer
                    continue
                try:
                    rec = LS.build_day_record_daily(
                        sym, bar[0], bar,
                        provenance={"source": "binance_klines_1d",
                                    "fetched_at": int(time.time() * 1000)})
                except ValueError:
                    continue
                if not rec["complete"]:
                    continue              # today's unfinished day is not a label
                if store.put(rec):
                    written += 1
                    existing.add(key)
        if i % 50 == 0 or i == len(universe):
            print(f"  {i}/{len(universe)}  written={written} "
                  f"skipped={skipped} failed={len(failed)}  "
                  f"{time.time()-t0:.0f}s")
        time.sleep(PAUSE_S)

    summary = LS.summarise_universe_build(resolved=resolved, failed=failed)
    summary.update({"records_written": written,
                    "records_skipped_existing": skipped,
                    "window_days": args.days})
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
