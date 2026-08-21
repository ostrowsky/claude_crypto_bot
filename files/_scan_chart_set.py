"""Full external read on a named set of coins, for comparing them against each other.

Built for the 20 charts the operator screenshotted on 2026-08-21. All of them
look alike on price -- third day of a rally, RSI 72-88, price far above MA25 --
so the chart cannot separate them. Everything below comes from outside the price
series: who is positioned, how that changed, and which side is paying.

THE COLUMNS, and what each is FOR

    move / chg     the day's MOVE (high vs open) against its close change. A big
                   gap means the coin ran and gave it back.
    range pos      where the last price sits between the day's low and high.
                   95% means it is at the highs; 40% means it has already faded.
    vs MA25        extension. The bot's own trend_quality guard rejects above
                   ~3-4% for a reason: entries that far out have no room.
    OI 1h/4h/24h   open interest across three windows. 24h alone cannot tell
                   whether money is arriving NOW or arrived yesterday and is
                   leaving -- the three together can.
    taker          taker buy volume / sell volume, last hour. >1 = buyers are
                   lifting offers rather than resting bids.
    tk trend       that ratio now against its 6h average. Rising = aggression
                   building; falling = the buying is tiring.
    top / retail   largest accounts vs all accounts, long/short. When they
                   diverge, the crowd is opposite the people with the most at
                   stake.
    fund           funding in basis points. High positive = crowded long paying
                   to stay; negative while price rises = the move is happening
                   against the positioning.

WHAT THIS IS NOT

    Not a ranking, not a recommendation, and not a forecast. These are
    measurements of the present. The only one of them testable over this
    project's full 419-day window was funding, and it did NOT improve the
    detector. The rest are capped at 30 days of API history, which is far too
    little to validate anything -- they are shown because they describe
    something price does not, not because they are known to predict.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

FAPI = "https://fapi.binance.com"
PAUSE = 0.05

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "LINK", "SUI", "UNI", "BCH", "CRV",
         "POL", "INJ", "SEI", "ZRX", "STRK", "TIA", "WIF", "AXL",
         # added 2026-08-21 on the operator's second batch of charts
         "AAVE", "GLM", "DOGE", "NEAR",
         # SHIB and BONK trade as 1000SHIB / 1000BONK on USD-M futures
         "1000SHIB", "1000BONK"]


def get(path, **kw):
    url = FAPI + path + ("?" + urllib.parse.urlencode(kw) if kw else "")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def series_change(rows, key, back):
    """Percent change of `key` from `back` samples ago to the latest."""
    if not rows or len(rows) <= back:
        return None
    try:
        a = float(rows[-1 - back][key])
        b = float(rows[-1][key])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return (b / a - 1.0) * 100.0 if a > 0 else None


def ema(vals, n):
    if not vals:
        return None
    k = 2.0 / (n + 1.0)
    out = vals[0]
    for v in vals[1:]:
        out = v * k + out * (1 - k)
    return out


def main():
    syms = [c + "USDT" for c in COINS]
    tick = {r["symbol"]: r for r in (get("/fapi/v1/ticker/24hr") or [])}

    rows = []
    for i, s in enumerate(syms, 1):
        t = tick.get(s)
        if not t:
            print("no ticker for %s" % s)
            continue
        try:
            op, hi, lo = float(t["openPrice"]), float(t["highPrice"]), float(t["lowPrice"])
            la, chg = float(t["lastPrice"]), float(t["priceChangePercent"])
        except (KeyError, TypeError, ValueError):
            continue
        if op <= 0 or hi <= lo:
            continue

        kl = get("/fapi/v1/klines", symbol=s, interval="1h", limit=120)
        time.sleep(PAUSE)
        closes = [float(k[4]) for k in kl] if kl else []
        ma25 = sum(closes[-25:]) / 25 if len(closes) >= 25 else None

        oi = get("/futures/data/openInterestHist", symbol=s, period="1h", limit=24)
        time.sleep(PAUSE)
        tk = get("/futures/data/takerlongshortRatio", symbol=s, period="1h", limit=6)
        time.sleep(PAUSE)
        gl = get("/futures/data/globalLongShortAccountRatio", symbol=s, period="1h", limit=1)
        time.sleep(PAUSE)
        tp = get("/futures/data/topLongShortPositionRatio", symbol=s, period="1h", limit=1)
        time.sleep(PAUSE)
        fr = get("/fapi/v1/fundingRate", symbol=s, limit=1)
        time.sleep(PAUSE)

        tkv = []
        if tk:
            for r in tk:
                try:
                    tkv.append(float(r["buySellRatio"]))
                except (KeyError, TypeError, ValueError):
                    pass

        rows.append({
            "sym": s.replace("USDT", ""),
            "move": (hi / op - 1) * 100,
            "chg": chg,
            "rpos": (la - lo) / (hi - lo) * 100,
            "vs25": ((la / ma25 - 1) * 100) if ma25 else None,
            "oi1": series_change(oi, "sumOpenInterest", 1),
            "oi4": series_change(oi, "sumOpenInterest", 4),
            "oi24": series_change(oi, "sumOpenInterest", 23),
            "taker": tkv[-1] if tkv else None,
            "tktr": (tkv[-1] / (sum(tkv) / len(tkv))) if tkv else None,
            "retail": (float(gl[-1]["longShortRatio"]) if gl else None),
            "top": (float(tp[-1]["longShortRatio"]) if tp else None),
            "fund": (float(fr[-1]["fundingRate"]) * 10000 if fr else None),
        })
        if i % 5 == 0:
            print("  %d/%d" % (i, len(syms)))

    f = lambda v, s="%6.1f": (s % v) if v is not None else "     -"

    print()
    print("=" * 120)
    print("THE 20 CHARTS, MEASURED FROM OUTSIDE THE PRICE SERIES")
    print("=" * 120)
    print("%-6s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s" % (
        "coin", "move%", "chg%", "rngpos", "vsMA25", "OI 1h", "OI 4h", "OI 24h",
        "taker", "tk trd", "retail", "top", "fund"))
    print("-" * 120)
    for r in sorted(rows, key=lambda x: -(x["oi24"] if x["oi24"] is not None else -9e9)):
        print("%-6s%8s%8s%7s%%%8s%8s%8s%8s%8s%8s%8s%8s%8s" % (
            r["sym"], f(r["move"]), f(r["chg"]), f(r["rpos"], "%5.0f"),
            f(r["vs25"]), f(r["oi1"], "%6.2f"), f(r["oi4"], "%6.2f"),
            f(r["oi24"]), f(r["taker"], "%6.2f"), f(r["tktr"], "%6.2f"),
            f(r["retail"], "%6.2f"), f(r["top"], "%6.2f"), f(r["fund"], "%+6.1f")))

    print()
    print("=" * 120)
    print("WHAT SEPARATES THEM")
    print("=" * 120)

    def pick(label, keyfn, n=4, fmt="%.1f"):
        v = [r for r in rows if keyfn(r) is not None]
        if not v:
            return
        v.sort(key=lambda r: -keyfn(r))
        print("%-34s %s" % (label, "  ".join(
            "%s %s" % (r["sym"], fmt % keyfn(r)) for r in v[:n])))

    pick("new money arriving in the last hour", lambda r: r["oi1"], fmt="%+.2f%%")
    pick("new money over 24h", lambda r: r["oi24"], fmt="%+.1f%%")
    pick("buyers most aggressive right now", lambda r: r["taker"], fmt="%.2f")
    pick("aggression building vs its own 6h", lambda r: r["tktr"], fmt="%.2f")
    pick("largest accounts most long", lambda r: r["top"], fmt="%.2f")
    print()
    ext = [r for r in rows if r["vs25"] is not None]
    ext.sort(key=lambda r: r["vs25"])
    print("%-34s %s" % ("least extended above MA25", "  ".join(
        "%s %+.1f%%" % (r["sym"], r["vs25"]) for r in ext[:4])))
    faded = [r for r in rows if r["rpos"] is not None]
    faded.sort(key=lambda r: r["rpos"])
    print("%-34s %s" % ("furthest off the day's high", "  ".join(
        "%s %.0f%%" % (r["sym"], r["rpos"]) for r in faded[:4])))

    print()
    print("READ THIS")
    print("  A ranking is not present on purpose. Funding was the only column here")
    print("  testable over the full 419-day window and it did not help the")
    print("  detector; the rest are capped at 30 days of API history, which holds")
    print("  ~two dozen +20% moves -- too few for any of it to be called an edge.")


if __name__ == "__main__":
    main()
