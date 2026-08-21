"""Watchlist screen from data the bot does not use: OI, taker flow, positioning.

The bot decides on price and volume. Everything here comes from a different
place -- who is positioned, how heavily, and which side is paying to hold. None
of it appears in any feature the bot computes, so this is an independent read
rather than a second opinion from the same evidence.

WHAT IS PULLED, per watchlist symbol (Binance USDT-M futures, public endpoints)

    24h ticker              move (high vs open), change (close vs open)
    openInterestHist 1h     open interest over the last 24h -> growth
    takerlongshortRatio 1h  taker buy volume vs sell volume, the aggressor side
    globalLongShortAccountRatio  how retail accounts are positioned
    topLongShortPositionRatio    how the largest accounts are positioned
    fundingRate             the last few settlements

WHAT THE COLUMNS MEAN, and why each is here

    OI 24h      new money entering or leaving the contract. Price up on rising OI
                is fresh positioning; price up on FALLING OI is shorts closing,
                which ends when they are done.
    taker       >1 means market buyers are lifting offers rather than sitting on
                bids. It is the one column here that describes aggression rather
                than inventory.
    top/retail  when the largest accounts lean the opposite way from retail, the
                crowd is on the other side of the people with the most at stake.
    funding     what longs pay shorts. High positive funding is a crowded long
                and a standing cost; negative funding with price rising means the
                move is happening against the positioning.

WHAT THIS IS NOT

    Not a prediction and not a recommendation. Every relationship above is a
    plausible mechanism, not a measured edge on this watchlist -- funding was the
    only one of these testable over the full 419-day window and it did NOT help
    the detector (AUC 0.606 on the narrow question, no improvement in practice).
    The 30-day API limit on the rest is exactly why they remain untested.
    Read this as a description of the current state, not a forecast.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FAPI = "https://fapi.binance.com"
PAUSE = 0.06


def get(path, **kw):
    url = FAPI + path + ("?" + urllib.parse.urlencode(kw) if kw else "")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def pct_change(series, key):
    if not series or len(series) < 2:
        return None
    try:
        a, b = float(series[0][key]), float(series[-1][key])
    except (KeyError, TypeError, ValueError):
        return None
    return (b / a - 1.0) * 100.0 if a > 0 else None


def main():
    import _diag_uptrend_population as UP

    wl = set(UP.watchlist())
    tick = get("/fapi/v1/ticker/24hr") or []
    rows = {}
    for r in tick:
        s = r.get("symbol")
        if s not in wl:
            continue
        try:
            op, hi, la = float(r["openPrice"]), float(r["highPrice"]), float(r["lastPrice"])
            ch, qv = float(r["priceChangePercent"]), float(r["quoteVolume"])
        except (KeyError, TypeError, ValueError):
            continue
        if op <= 0:
            continue
        rows[s] = {
            "sym": s, "move": (hi / op - 1) * 100, "chg": ch, "last": la,
            "qvol": qv, "off_high": (la / hi - 1) * 100 if hi else 0.0,
        }

    print("watchlist symbols with futures data: %d" % len(rows))
    print("pulling open interest, taker flow, positioning and funding ...")
    for i, s in enumerate(sorted(rows), 1):
        d = rows[s]
        oi = get("/futures/data/openInterestHist", symbol=s, period="1h", limit=24)
        time.sleep(PAUSE)
        d["oi24"] = pct_change(oi, "sumOpenInterest")
        d["oi_usd"] = (float(oi[-1]["sumOpenInterestValue"]) if oi else None)

        tk = get("/futures/data/takerlongshortRatio", symbol=s, period="1h", limit=6)
        time.sleep(PAUSE)
        try:
            d["taker"] = float(tk[-1]["buySellRatio"]) if tk else None
        except (KeyError, TypeError, ValueError):
            d["taker"] = None

        gl = get("/futures/data/globalLongShortAccountRatio", symbol=s, period="1h", limit=2)
        time.sleep(PAUSE)
        try:
            d["retail"] = float(gl[-1]["longShortRatio"]) if gl else None
        except (KeyError, TypeError, ValueError):
            d["retail"] = None

        tp = get("/futures/data/topLongShortPositionRatio", symbol=s, period="1h", limit=2)
        time.sleep(PAUSE)
        try:
            d["top"] = float(tp[-1]["longShortRatio"]) if tp else None
        except (KeyError, TypeError, ValueError):
            d["top"] = None

        fr = get("/fapi/v1/fundingRate", symbol=s, limit=3)
        time.sleep(PAUSE)
        try:
            d["fund"] = float(fr[-1]["fundingRate"]) * 10000 if fr else None
        except (KeyError, TypeError, ValueError):
            d["fund"] = None

        if i % 20 == 0:
            print("  %d/%d" % (i, len(rows)))

    have = [d for d in rows.values() if d.get("oi24") is not None and d.get("taker")]
    print("complete rows: %d" % len(have))

    def show(title, key, reverse=True, n=12, note=""):
        print()
        print("=" * 104)
        print(title)
        if note:
            print(note)
        print("=" * 104)
        print("%-11s%9s%9s%10s%9s%9s%9s%9s%11s" % (
            "sym", "move%", "chg%", "OI 24h%", "taker", "retail", "top", "fund bp", "OI $m"))
        print("-" * 104)
        for d in sorted(have, key=lambda x: (x.get(key) if x.get(key) is not None else -9e9),
                        reverse=reverse)[:n]:
            print("%-11s%8.1f%%%8.1f%%%9.1f%%%9.2f%9s%9s%9s%10.0f" % (
                d["sym"].replace("USDT", ""), d["move"], d["chg"],
                d["oi24"] if d["oi24"] is not None else float("nan"),
                d["taker"] or 0,
                ("%.2f" % d["retail"]) if d["retail"] else "-",
                ("%.2f" % d["top"]) if d["top"] else "-",
                ("%+.1f" % d["fund"]) if d["fund"] is not None else "-",
                (d["oi_usd"] or 0) / 1e6))

    show("OPEN INTEREST GROWTH over 24h — new money entering the contract",
         "oi24", note="Price rising on rising OI is fresh positioning. Rising on "
                      "FALLING OI is shorts covering, which stops when they finish.")

    quiet = [d for d in have if d["move"] < 8.0]
    if quiet:
        print()
        print("=" * 104)
        print("OI BUILDING WHILE PRICE HAS NOT MOVED YET (move < 8% today)")
        print("The one pattern in this file that price cannot show by construction:")
        print("positioning ahead of a move has not reached the chart yet. Untested "
              "on this watchlist -- 30 days of API history cannot validate it.")
        print("=" * 104)
        print("%-11s%9s%9s%10s%9s%9s%9s%9s%11s" % (
            "sym", "move%", "chg%", "OI 24h%", "taker", "retail", "top", "fund bp", "OI $m"))
        print("-" * 104)
        for d in sorted(quiet, key=lambda x: -(x["oi24"] or -9e9))[:12]:
            print("%-11s%8.1f%%%8.1f%%%9.1f%%%9.2f%9s%9s%9s%10.0f" % (
                d["sym"].replace("USDT", ""), d["move"], d["chg"], d["oi24"],
                d["taker"] or 0,
                ("%.2f" % d["retail"]) if d["retail"] else "-",
                ("%.2f" % d["top"]) if d["top"] else "-",
                ("%+.1f" % d["fund"]) if d["fund"] is not None else "-",
                (d["oi_usd"] or 0) / 1e6))

    show("TAKER AGGRESSION — buyers lifting offers rather than resting bids",
         "taker", n=10)

    print()
    print("READ THIS")
    print("  Nothing above is a forecast. Funding is the only one of these columns")
    print("  that could be tested over the full 419-day window, and it did NOT help")
    print("  the detector; the rest are capped at 30 days of history by the API,")
    print("  which is why they are described as mechanisms rather than as edges.")


if __name__ == "__main__":
    main()
