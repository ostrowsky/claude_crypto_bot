"""Backfill 419d of funding rates for the watchlist.

Funding is the ONLY alternative data source Binance serves deep enough to meet
this project's max-period standard. Measured against the API on 2026-08-20:

    funding rate            reaches 2025-06-26  (420d)  8h steps
    open interest           30 days, HTTP 400 beyond    5m/15m/1h/4h
    taker buy/sell volume   30 days, HTTP 400 beyond    1h
    long/short ratios       30 days, HTTP 400 beyond    1h
    order book depth        no history at all           live snapshot only

So funding can be validated now on the same 419-day window as the klines, and
everything else has to be collected forward before it can ever be tested.

Why funding is worth asking about at all: price and volume were shown unable to
separate the START of a strong trend from its MIDDLE at 1h or 15m (five
attacks, all negative -- see trend-start-detector-spec.md). Funding measures
something those cannot: crowd positioning. The hypothesis it makes testable is
that a trend's opening hours run on neutral or negative funding (nobody is
positioned yet) while its middle runs on elevated funding (the crowd has piled
in). If true, that is exactly the missing start-vs-middle discriminator.

Writes history/<sym>_funding.csv. Resumable: a symbol whose file already
reaches back far enough and ends recently is skipped.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = ROOT / "history"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DAYS = 419
STEP_MS = 8 * 3600 * 1000
PAGE = 1000


def fetch(sym, start_ms, end_ms):
    url = ("https://fapi.binance.com/fapi/v1/fundingRate?"
           + urllib.parse.urlencode(dict(symbol=sym, limit=PAGE,
                                         startTime=start_ms, endTime=end_ms)))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def already_ok(path, oldest_needed_ms):
    if not path.exists():
        return False
    try:
        with io.open(path, encoding="utf-8") as fh:
            rows = fh.readlines()
        if len(rows) < 100:
            return False
        first = datetime.fromisoformat(rows[1].split(",")[0])
        last = datetime.fromisoformat(rows[-1].split(",")[0])
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return first.timestamp() * 1000 <= oldest_needed_ms + 3 * 86400000 and age_h < 48
    except Exception:
        return False


def main():
    import _diag_uptrend_population as UP

    syms = UP.watchlist()
    HISTORY.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    start = now - DAYS * 86400000
    print("[funding] %d symbols, %d days, 8h steps" % (len(syms), DAYS))

    done = skipped = empty = 0
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        out = HISTORY / ("%s_funding.csv" % sym)
        if already_ok(out, start):
            skipped += 1
            continue
        rows = {}
        cur = start
        while cur < now:
            chunk = fetch(sym, cur, min(now, cur + PAGE * STEP_MS))
            if not chunk:
                break
            for r in chunk:
                try:
                    rows[int(r["fundingTime"])] = (float(r["fundingRate"]),
                                                   float(r.get("markPrice") or 0.0))
                except (KeyError, ValueError, TypeError):
                    continue
            last = max(int(r["fundingTime"]) for r in chunk)
            nxt = last + STEP_MS
            if nxt <= cur or len(chunk) < PAGE:
                break
            cur = nxt
            time.sleep(0.12)
        if not rows:
            empty += 1
            print("  no data: %s" % sym)
            continue
        tmp = out.with_suffix(".csv.part")
        with io.open(tmp, "w", encoding="utf-8") as fh:
            fh.write("ts,funding_rate,mark_price\n")
            for k in sorted(rows):
                fr, mp = rows[k]
                fh.write("%s,%.10f,%.10f\n"
                         % (datetime.fromtimestamp(k / 1000, tz=timezone.utc).isoformat(),
                            fr, mp))
        tmp.replace(out)
        done += 1
        if i % 20 == 0 or i == len(syms):
            print("  [%d/%d] done=%d skip=%d empty=%d  (%.0fs)"
                  % (i, len(syms), done, skipped, empty, time.time() - t0))
        time.sleep(0.12)

    print("\n[funding] finished in %.0fs  done=%d skipped=%d empty=%d"
          % (time.time() - t0, done, skipped, empty))
    n = sum(1 for _ in HISTORY.glob("*_funding.csv"))
    print("[funding] %d symbols now hold a funding history" % n)


if __name__ == "__main__":
    main()
