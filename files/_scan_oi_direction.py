"""Is the new money betting UP or DOWN? Open interest read against price.

The operator's question, and the reason the previous table was not enough:
open interest rising says money arrived, not which side it took. Every contract
has a long and a short, so OI alone is direction-blind.

Price over the same window resolves it. The standard reading, and the one used
here:

    OI up   + price up    LONGS OPENING     new money betting on a rise
    OI up   + price down  SHORTS OPENING    new money betting on a fall
    OI down + price up    SHORT COVERING    the rise is people closing bets
                                            against it, which ends when they
                                            finish -- not new demand
    OI down + price down  LONGS CLOSING     holders leaving

Taker ratio is carried alongside as a second opinion: it says which side was in
a hurry. Longs opening while sellers are the aggressors is a contradiction worth
seeing rather than averaging away.

The window is 4h, not 24h. Over a full day a coin can do all four in sequence
and the net tells you nothing about where it is now.

NOT A FORECAST. "Money is positioning for a rise" is a statement about what has
already happened, not about what will. Positioning is frequently wrong -- that is
why the other side of every one of these contracts exists.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

FAPI = "https://fapi.binance.com"
PAUSE = 0.05
WINDOW = 4          # hours

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "LINK", "SUI", "UNI", "BCH", "CRV",
         "POL", "INJ", "SEI", "ZRX", "STRK", "TIA", "WIF", "AXL",
         "AAVE", "GLM", "DOGE", "NEAR", "1000SHIB", "1000BONK"]


def get(path, **kw):
    url = FAPI + path + ("?" + urllib.parse.urlencode(kw) if kw else "")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def main():
    rows = []
    for i, c in enumerate(COINS, 1):
        s = c + "USDT"
        oi = get("/futures/data/openInterestHist", symbol=s, period="1h",
                 limit=WINDOW + 1)
        time.sleep(PAUSE)
        kl = get("/fapi/v1/klines", symbol=s, interval="1h", limit=WINDOW + 1)
        time.sleep(PAUSE)
        tk = get("/futures/data/takerlongshortRatio", symbol=s, period="1h", limit=1)
        time.sleep(PAUSE)
        if not oi or not kl or len(oi) < 2 or len(kl) < 2:
            print("no data for %s" % s)
            continue
        try:
            oi_a, oi_b = float(oi[0]["sumOpenInterest"]), float(oi[-1]["sumOpenInterest"])
            px_a, px_b = float(kl[0][4]), float(kl[-1][4])
            oi_usd = float(oi[-1]["sumOpenInterestValue"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if oi_a <= 0 or px_a <= 0:
            continue
        d_oi = (oi_b / oi_a - 1) * 100
        d_px = (px_b / px_a - 1) * 100
        taker = None
        try:
            taker = float(tk[-1]["buySellRatio"]) if tk else None
        except (KeyError, TypeError, ValueError):
            pass

        if d_oi > 0.5 and d_px > 0.3:
            kind = "LONGS OPENING"
        elif d_oi > 0.5 and d_px < -0.3:
            kind = "SHORTS OPENING"
        elif d_oi < -0.5 and d_px > 0.3:
            kind = "SHORT COVERING"
        elif d_oi < -0.5 and d_px < -0.3:
            kind = "LONGS CLOSING"
        else:
            kind = "no clear flow"

        rows.append({"sym": c, "d_oi": d_oi, "d_px": d_px,
                     "taker": taker, "kind": kind, "oi_usd": oi_usd})
        if i % 6 == 0:
            print("  %d/%d" % (i, len(COINS)))

    order = ["LONGS OPENING", "SHORT COVERING", "no clear flow",
             "LONGS CLOSING", "SHORTS OPENING"]
    titles = {
        "LONGS OPENING": ("NEW MONEY BETTING ON A RISE",
                          "OI up and price up over %dh: positions opened into "
                          "strength." % WINDOW),
        "SHORTS OPENING": ("NEW MONEY BETTING ON A FALL",
                           "OI up while price fell: the arriving money took the "
                           "other side."),
        "SHORT COVERING": ("RISING ON CLOSURES, NOT ON NEW DEMAND",
                           "OI down while price rose: bets against it are being "
                           "closed. That buying stops when they are done."),
        "LONGS CLOSING": ("HOLDERS LEAVING", "OI down and price down."),
        "no clear flow": ("NOTHING DECISIVE", "Neither moved enough to read."),
    }

    for k in order:
        sub = [r for r in rows if r["kind"] == k]
        if not sub:
            continue
        title, note = titles[k]
        sub.sort(key=lambda r: -r["d_oi"])
        print()
        print("=" * 84)
        print(title)
        print(note)
        print("=" * 84)
        print("%-10s%12s%12s%10s%14s" % (
            "coin", "money %dh" % WINDOW, "price %dh" % WINDOW, "taker", "OI now $m"))
        print("-" * 84)
        for r in sub:
            warn = ""
            if k == "LONGS OPENING" and r["taker"] is not None and r["taker"] < 0.95:
                warn = "  <-- but sellers are the aggressors"
            print("%-10s%11.2f%%%11.2f%%%10s%13.0f%s" % (
                r["sym"], r["d_oi"], r["d_px"],
                ("%.2f" % r["taker"]) if r["taker"] is not None else "-",
                r["oi_usd"] / 1e6, warn))

    print()
    print("HOW TO READ THIS")
    print("  'Betting on a rise' describes what already happened -- money took a")
    print("  side. It does not say the side is right. Every one of these contracts")
    print("  has someone on the other end who thinks the opposite, and over 30 days")
    print("  of available history there is no way to check which group tends to win.")


if __name__ == "__main__":
    main()
