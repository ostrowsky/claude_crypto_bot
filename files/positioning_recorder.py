"""Record what derivatives positioning says, then come back and check it.

WHY THIS EXISTS

Open interest, taker flow and long/short positioning describe something price
cannot: who is positioned and which side is paying. Binance serves them for
**30 days only**, which holds roughly two dozen +20% moves across the watchlist
-- far too few to tell a real relationship from a coincidence. That is why every
statement made from these numbers so far has been labelled a mechanism rather
than an edge.

There is exactly one way out of that, and it is not cleverness: start writing
them down. A snapshot taken today can be scored in six hours and never expires
after that. In three months this file holds a window the API will never give
back, and the question "does positioning predict anything on THIS watchlist"
becomes answerable instead of arguable.

WHAT IT DOES

    snapshot   one row per watchlist coin: price, the day's move, extension,
               OI over 1h/4h/24h, taker ratio and its trend, retail and top-account
               positioning, funding, and the flow class derived from OI against
               price (longs opening / shorts opening / short covering / ...).
    resolve    revisits rows older than --horizon hours and writes what actually
               happened: peak gain, close-to-close return, and whether the coin
               reached the day's top-50 by MOVE across the whole futures universe.
    report     scores resolved rows by flow class, ALWAYS against the all-coin
               baseline for the same snapshots. A class that "wins 40% of the
               time" means nothing until the pool's own rate sits beside it.

HONEST LIMITS, stated here so they travel with the data

    One day is an anecdote. On a day when the whole market rises, every class
    will look good; on a falling day, none will. Only the SPREAD between classes
    carries information, and only once there are enough days for that spread to
    be distinguishable from noise. The report prints the sample size next to
    every number for exactly this reason and refuses to draw conclusions below
    --min-n rows.

    Positioning is frequently wrong. Every contract counted here has someone on
    the other side who read the same data and concluded the opposite.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FAPI = "https://fapi.binance.com"
STORE = HERE / "positioning_history.jsonl"
PAUSE = 0.05


def get(path, **kw):
    url = FAPI + path + ("?" + urllib.parse.urlencode(kw) if kw else "")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def change(rows, key, back):
    if not rows or len(rows) <= back:
        return None
    try:
        a, b = float(rows[-1 - back][key]), float(rows[-1][key])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return (b / a - 1.0) * 100.0 if a > 0 else None


def classify(d_oi, d_px):
    """OI against price. Neither alone says which side the money took."""
    if d_oi is None or d_px is None:
        return "unknown"
    if d_oi > 0.5 and d_px > 0.3:
        return "longs_opening"
    if d_oi > 0.5 and d_px < -0.3:
        return "shorts_opening"
    if d_oi < -0.5 and d_px > 0.3:
        return "short_covering"
    if d_oi < -0.5 and d_px < -0.3:
        return "longs_closing"
    return "flat"


def append(rows):
    with io.open(STORE, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load():
    if not STORE.exists():
        return []
    out = []
    with io.open(STORE, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def cmd_snapshot(args):
    import _diag_uptrend_population as UP

    wl = UP.watchlist()
    tick = {r["symbol"]: r for r in (get("/fapi/v1/ticker/24hr") or [])}
    now = datetime.now(timezone.utc)
    rows = []
    for i, s in enumerate(wl, 1):
        t = tick.get(s)
        if not t:
            continue
        try:
            op, hi, lo = float(t["openPrice"]), float(t["highPrice"]), float(t["lowPrice"])
            la, chg = float(t["lastPrice"]), float(t["priceChangePercent"])
        except (KeyError, TypeError, ValueError):
            continue
        if op <= 0 or hi <= lo:
            continue

        oi = get("/futures/data/openInterestHist", symbol=s, period="1h", limit=24)
        time.sleep(PAUSE)
        kl = get("/fapi/v1/klines", symbol=s, interval="1h", limit=30)
        time.sleep(PAUSE)
        tk = get("/futures/data/takerlongshortRatio", symbol=s, period="1h", limit=6)
        time.sleep(PAUSE)
        gl = get("/futures/data/globalLongShortAccountRatio", symbol=s, period="1h", limit=1)
        time.sleep(PAUSE)
        tp = get("/futures/data/topLongShortPositionRatio", symbol=s, period="1h", limit=1)
        time.sleep(PAUSE)
        fr = get("/fapi/v1/fundingRate", symbol=s, limit=1)
        time.sleep(PAUSE)

        closes = [float(k[4]) for k in kl] if kl else []
        ma25 = sum(closes[-25:]) / 25 if len(closes) >= 25 else None
        d_px4 = ((closes[-1] / closes[-5] - 1) * 100) if len(closes) >= 5 else None
        tkv = []
        for r in (tk or []):
            try:
                tkv.append(float(r["buySellRatio"]))
            except (KeyError, TypeError, ValueError):
                pass
        d_oi4 = change(oi, "sumOpenInterest", 4)

        rows.append({
            "ts": now.isoformat(), "sym": s, "price": la,
            "move": (hi / op - 1) * 100, "chg": chg,
            "rngpos": (la - lo) / (hi - lo) * 100,
            "vs_ma25": ((la / ma25 - 1) * 100) if ma25 else None,
            "oi_1h": change(oi, "sumOpenInterest", 1),
            "oi_4h": d_oi4,
            "oi_24h": change(oi, "sumOpenInterest", 23),
            "px_4h": d_px4,
            "taker": tkv[-1] if tkv else None,
            "taker_trend": (tkv[-1] / (sum(tkv) / len(tkv))) if tkv else None,
            "retail": (float(gl[-1]["longShortRatio"]) if gl else None),
            "top": (float(tp[-1]["longShortRatio"]) if tp else None),
            "funding_bp": (float(fr[-1]["fundingRate"]) * 10000 if fr else None),
            "flow": classify(d_oi4, d_px4),
            "resolved": False,
        })
        if i % 25 == 0:
            print("  %d/%d" % (i, len(wl)))

    append(rows)
    by = defaultdict(int)
    for r in rows:
        by[r["flow"]] += 1
    print("snapshot written: %d coins at %s" % (len(rows), now.strftime("%Y-%m-%d %H:%M UTC")))
    print("  flow classes:", dict(by))
    print("  store now holds %d rows" % len(load()))


def cmd_resolve(args):
    rows = load()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.horizon)
    todo = [r for r in rows if not r.get("resolved")
            and datetime.fromisoformat(r["ts"]) <= cutoff]
    if not todo:
        print("nothing due: no unresolved rows older than %dh" % args.horizon)
        return

    # The day's top-50 by MOVE across the whole futures universe -- the operator's
    # own yardstick, and the reason this is not just a return calculation.
    tick = get("/fapi/v1/ticker/24hr") or []
    universe = []
    for t in tick:
        s = t.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        try:
            op, hi = float(t["openPrice"]), float(t["highPrice"])
        except (KeyError, TypeError, ValueError):
            continue
        if op > 0:
            universe.append((s, (hi / op - 1) * 100))
    universe.sort(key=lambda x: -x[1])
    top50 = {s for s, _ in universe[:50]}

    print("resolving %d rows older than %dh" % (len(todo), args.horizon))
    by_sym = defaultdict(list)
    for r in todo:
        by_sym[r["sym"]].append(r)

    for i, (sym, rs) in enumerate(sorted(by_sym.items()), 1):
        oldest = min(datetime.fromisoformat(r["ts"]) for r in rs)
        kl = get("/fapi/v1/klines", symbol=sym, interval="15m",
                 startTime=int(oldest.timestamp() * 1000), limit=200)
        time.sleep(PAUSE)
        if not kl:
            continue
        for r in rs:
            t0 = datetime.fromisoformat(r["ts"])
            end = t0 + timedelta(hours=args.horizon)
            seg = [k for k in kl
                   if t0.timestamp() * 1000 <= int(k[0]) <= end.timestamp() * 1000]
            if len(seg) < 2:
                continue
            px = float(r["price"])
            peak = max(float(k[2]) for k in seg)
            last = float(seg[-1][4])
            r["out_peak_pct"] = (peak / px - 1) * 100
            r["out_close_pct"] = (last / px - 1) * 100
            r["out_top50"] = sym in top50
            r["resolved"] = True
        if i % 25 == 0:
            print("  %d/%d symbols" % (i, len(by_sym)))

    with io.open(STORE, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = sum(1 for r in rows if r.get("resolved"))
    print("resolved rows in store: %d of %d" % (done, len(rows)))


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def cmd_report(args):
    rows = [r for r in load() if r.get("resolved") and r.get("out_peak_pct") is not None]
    if not rows:
        print("no resolved rows yet -- run `resolve` after the horizon has passed")
        return
    days = sorted({r["ts"][:10] for r in rows})
    print("resolved rows: %d over %d day(s): %s .. %s"
          % (len(rows), len(days), days[0], days[-1]))

    pool = [r["out_peak_pct"] for r in rows]
    base3 = 100.0 * sum(1 for x in pool if x > 3) / len(pool)
    base5 = 100.0 * sum(1 for x in pool if x > 5) / len(pool)
    base50 = 100.0 * sum(1 for r in rows if r.get("out_top50")) / len(rows)

    print()
    print("=" * 92)
    print("OUTCOME BY FLOW CLASS — every number sits beside the all-coin pool")
    print("=" * 92)
    print("%-22s%7s%11s%11s%9s%9s%10s" % (
        "class", "n", "peak med", "peak p75", ">3%", ">5%", "top50"))
    print("-" * 92)
    print("%-22s%7d%10.2f%%%10.2f%%%8.0f%%%8.0f%%%9.0f%%" % (
        "ALL COINS (pool)", len(pool), q(pool, .5), q(pool, .75), base3, base5, base50))
    print("-" * 92)

    groups = defaultdict(list)
    for r in rows:
        groups[r.get("flow", "unknown")].append(r)
    # the specific claim made on 2026-08-21: money betting up AND buyers pressing
    groups["longs_opening + taker>1.1"] = [
        r for r in rows if r.get("flow") == "longs_opening"
        and (r.get("taker") or 0) > 1.1]

    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(v) < args.min_n:
            print("%-22s%7d   (below --min-n %d, not scored)" % (k[:21], len(v), args.min_n))
            continue
        p = [x["out_peak_pct"] for x in v]
        s3 = 100.0 * sum(1 for x in p if x > 3) / len(p)
        s5 = 100.0 * sum(1 for x in p if x > 5) / len(p)
        t50 = 100.0 * sum(1 for x in v if x.get("out_top50")) / len(v)
        flag = "  <-- beats the pool" if s3 > base3 * 1.2 else ""
        print("%-22s%7d%10.2f%%%10.2f%%%8.0f%%%8.0f%%%9.0f%%%s" % (
            k[:21], len(v), q(p, .5), q(p, .75), s3, s5, t50, flag))

    print()
    print("READ THIS")
    print("  Only the SPREAD between a class and the pool carries information.")
    print("  On a day the whole market rises every class looks strong, and on a")
    print("  falling day none does -- which is why the pool row is printed first.")
    if len(days) < 20:
        print("  %d day(s) of history. This is an anecdote, not evidence. The"
              % len(days))
        print("  earliest an honest answer becomes possible is a few dozen days.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot").set_defaults(fn=cmd_snapshot)
    r = sub.add_parser("resolve")
    r.add_argument("--horizon", type=int, default=8)
    r.set_defaults(fn=cmd_resolve)
    p = sub.add_parser("report")
    p.add_argument("--min-n", type=int, default=30)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
