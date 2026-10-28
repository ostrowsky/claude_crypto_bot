"""LA -- exchange announcements as information before the price (listing-announcements-la-1028-spec.md §5).

Written BEFORE announcements.py. Offline: pages come from fakes; the title fixtures are REAL titles read on 2026-10-28
from Binance CMS (catalogs 48 / 161), OKX and Bybit (spec §5.3).

Interface the tests fix:
  parse_binance(article, catalog) / parse_okx(item) / parse_bybit(item) -> record
      {source, id, catalog, title, url, t_pub (ms UTC), raw}
  AnnouncementStore(root).append(records, now) -> added (idempotent by (source, id); adds t_seen) / .load(source)
  drop_migrated(records, min_cluster)  bulk re-posts (>= min_cluster records of one source in one minute) are dropped
  backfill(fetch_page, parse, since_ms, max_pages) -> records, paging until a page is empty or older than since_ms
  classify(record) -> event type;  extract_tickers(title, etype) -> [base tickers]
  to_spot(tickers, etype, spot_syms) -> [spot symbols]  (perp prefixes 1000 / 1M handled)
  tradable(sym, t_pub, first_bar_ms) -> bool   (a closed 15m bar exists before t_pub)
  entry_bar_open(t_pub, delay_ms) -> open ms of the first bar opening at or after t_pub + delay
  pre_return(arr, t_pub, hours) -> return over the `hours` before t_pub from bars closed by t_pub
  control_time(...) / control_coin(...)  random controls without events, reproducible by seed
"""
import json
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR = 15 * 60 * 1000
HOUR = 3_600_000


def ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ------------------------------------------------------------------ real titles (2026-10-28)
BN_PERP = [
    ("Binance Futures Will Launch USDⓈ-Margined CTUSDT Perpetual Contract (2026-10-01)", ["CT"]),
    ("Binance Futures Will Launch USDⓈ-Margined BRUSDT and PLUMEUSDT Perpetual Contracts With up to 25x Leverage", ["BR", "PLUME"]),
    ("Binance Futures Will Launch USDⓈ-Margined BMTUSDT and MUBARAKUSDT Perpetual Contracts With up to 25x Leverag", ["BMT", "MUBARAK"]),
    ("Binance Futures Will Launch USDⓈ-Margined VICUSDT Perpetual Contract With Up To 75x Leverage", ["VIC"]),
    ("Binance Futures Will Launch USDⓈ-M LQTY Perpetual Contract with Up to 20X Leverage", ["LQTY"]),
    ("Binance Futures Will Launch USDⓈ-M TRU Perpetual Contract with Up to 20X Leverage", ["TRU"]),
    ("Binance Futures Will Launch USDⓈ-M CKB Perpetual Contracts with Up to 20X Leverage", ["CKB"]),
    ("Binance Futures Will Launch USDⓈ-M SSV Perpetual Contracts with Up to 20X Leverage", ["SSV"]),
    ("Binance Futures to Relaunch USDⓈ-M BNX Perpetual Contracts with Up to 20X Leverage", ["BNX"]),
    ("Binance Futures Will Launch UNI USDT-Margined Perpetual Contract With Up to 50x Leverage", ["UNI"]),
]
BN_SPOT = [
    ("Binance Will List Hyperliquid (HYPE) with Seed Tag Applied", ["HYPE"]),
    ("Binance Will List Centrifuge (CFG) with Seed Tag Applied", ["CFG"]),
    ("Binance Will List Katana (KAT) with Seed Tag Applied", ["KAT"]),
    ("Binance Lists Avalanche (AVAX), Deposits Are Now Open", ["AVAX"]),
    ("Binance Lists Cream Finance (CREAM)", ["CREAM"]),
    ("Binance Will Continue to List RedStone (RED)", ["RED"]),
    ("Binance Exchange Adds GoPro (GPROB) and Reddit (RDDTB) bStocks Trading Pairs on Binance Spot/Convert - 2026-09-16",
     ["GPROB", "RDDTB"]),
    ("Binance Exchange Adds CEA Industries (BNCB) bStocks Trading Pair on Binance Spot/Convert - 2026-09-14", ["BNCB"]),
    ("Introducing Nillion (NIL) on Binance Launchpool! Farm NIL by Locking BNB, FDUSD, and USDC", ["NIL"]),
    ("Introducing Bubblemaps (BMT) on Binance HODLer Airdrops! Earn BMT With Retroactive BNB Simple Earn Subscriptions", ["BMT"]),
]
BN_MARGIN = [
    ("Binance Adds ALCX, DF & More Tokens on Cross Margin and DF on Isolated Margin", ["ALCX", "DF"]),
    ("Binance Adds LQTY, MDT, QUICK & More Pairs on Cross Margin & Isolated Margin", ["LQTY", "MDT", "QUICK"]),
    ("Binance Adds BNX, YFI & More Pairs on Cross Margin & Isolated Margin", ["BNX", "YFI"]),
    ("Binance Margin Will Add New Pairs - 2026-09-23", []),
]
BN_OTHER = [
    "Binance Will Add Hyperliquid (HYPE) on Earn, Buy Crypto, Convert, VIP Loan & Margin",
    "Binance Will Add Bubblemaps (BMT) on Earn, Buy Crypto, Convert & Margin",
    "Binance Will Add StraitsX USD (XUSD) and Four (FORM) on Earn, Buy Crypto, Margin, Convert & Futures",
    "Binance Futures Will List USDⓈ-M & COIN-M Quarterly 0326 Delivery Contracts",
    "Binance Options Will List LINK Contract",
    "Binance Futures Will Increase Maximum Leverage on BNB Contracts to 75x",
    "Binance Will Add 5 bStocks Tokenized Securities as Collateral Asset - 2026-10-28",
    "Notice on New Trading Pairs & Trading Bots Services on Binance Spot - 2026-03-10",
    "Binance Futures Will Launch Multiple TradFi USDⓈ-Margined Perpetual Contracts (2026-10-26)",
    "Binance Futures Will Launch OURAUSDT USDⓈ-Margined Perpetual Contract Pre-IPO Trading (2026-09-23)",
]
BN_DELIST = [
    ("Binance Will Delist EPIC, QUICK on 2026-10-28", ["EPIC", "QUICK"]),
    ("Binance Will Delist A2Z, FORTH, HOOK, IDEX, LRC, NTRN, RDNT, SXP on 2026-04-01",
     ["A2Z", "FORTH", "HOOK", "IDEX", "LRC", "NTRN", "RDNT", "SXP"]),
    ("Binance Will Delist ANT, MULTI, VAI, XMR on 2024-02-20", ["ANT", "MULTI", "VAI", "XMR"]),
]
BN161_OTHER = [
    "Notice of Removal of Spot Trading Pairs - 2026-10-30",
    "Binance Futures Will Delist PTBUSDT and MITOUSDT USDⓈ-M Perpetual Contracts (2026-10-23)",
    "Binance Alpha Will Remove PTB, AA, OWL, DORA, ELIZAOS, MEMES, 哭哭马  (2026-10-23)",
    "Binance Margin Will Delist UTK on 2026-03-30",
]
OKX_SPOT = [
    ("OKX to list WET (HumidiFi) for spot trading", ["WET"]),
    ("OKX to list NIGHT (Midnight) for spot trading", ["NIGHT"]),
    ("OKX to list Dash (DASH) for spot trading", ["DASH"]),
    ("OKX will launch BREV/USDⓈ for spot trading", ["BREV"]),
    ("OKX will launch NIGHT/USDⓈ for spot trading", ["NIGHT"]),
    ("OKX will launch DASH/USDⓈ for spot trading", ["DASH"]),
]
OKX_PERP = [
    ("OKX to list perpetual futures for BTW crypto", ["BTW"]),
    ("OKX to list perpetual futures for PIPPIN, RLS crypto", ["PIPPIN", "RLS"]),
    ("OKX to list perpetual futures for TRUTH crypto", ["TRUTH"]),
    ("OKX to list perpetual futures for RAVE crypto", ["RAVE"]),
    ("OKX to list pre-market perpetual futures for LIT (Lighter) crypto", ["LIT"]),
    ("OKX to list MEWUSD X-Perp", ["MEW"]),
    ("OKX to list BATUSD X-Perp", ["BAT"]),
]
OKX_OTHER = [
    "OKX to list TradFi perpetual futures for CEG, BAC, IBKR and MPWR",
    "OKX to list ASMLUSD, ONUSD and ALABUSD Equity X-Perps",
    "OKX to list Unified Tokenized Stocks xMARA, xMCD and more for spot trading",
    "OKX to support VELODROME crypto migration",
    "OKX to enable margin trading, Simple Earn and Flexible Loan for 2Z, ZORA, VIRTUAL crypto",
]
BYBIT_SPOT = [
    ("Bybit to List Plasma (XPL) on Spot", ["XPL"]),
    ("Bybit to List Ripple USD (RLUSD) on Spot", ["RLUSD"]),
    ("Bybit to List 0G  (0G) on Spot", ["0G"]),
    ("Bybit to List Anoma (XAN) on Spot", ["XAN"]),
    ("Bybit to List Aster (ASTER)  on Spot", ["ASTER"]),
]
BYBIT_PERP = [
    ("New Listing :  FLUIDUSDT Perpetual Contract, with up to 50x leverage", ["FLUID"]),
    ("New Listing : BLESSUSDT Perpetual Contract in Innovation Zone, with up to 25x leverage", ["BLESS"]),
    ("New Listing : HEMIUSDT Perpetual Contract in Innovation Zone, with up to 25x leverage", ["HEMI"]),
    ("New listing: AXISUSDT Perpetual Contract in Innovation Zone, with up to 20x leverage", ["AXIS"]),
    ("New listing: Listing of DNAUSDT on Bybit Perpetual on Oct 16, 2026, 11:45AM UTC", ["DNA"]),
    ("Listing of ASTERUSDT on Bybit Perpetual Pre-Market on Sep 19, 2025, 9:00AM UTC", ["ASTER"]),
]
BYBIT_OTHER = [
    "New listing: Listing of EIDOUSDT on Bybit TradFi Perpetual on Oct 27, 2026, 10:30AM UTC",
    "PIEVERSE Token Splash— Grab a share of the 11000 PIEVERSE prize pool .",
    "🔥 Listing of XPL on Convert & Bybit Savings",
    "Explore USDE/USDT and USDE/USDC for Spot margin trading",
    "Converting 0GUSDT Perpetual Pre-Market Trading to Standard Perpetual Contract on Bybit Derivatives",
    "Postponement of Anoma (XAN) Spot listing",
]


def rec(source, title, catalog=None, t="2025-01-01 00:00"):
    return {"source": source, "id": title[:20], "catalog": catalog, "title": title, "url": "", "t_pub": ms(t), "raw": {}}


# ------------------------------------------------------------------ 1. store, parsing, paging
class TestParseAndStore(unittest.TestCase):
    def test_parsers(self):
        import announcements as AN
        b = AN.parse_binance({"code": "abc", "title": "Binance Will List X (X)", "releaseDate": 1700000000000}, 48)
        self.assertEqual((b["source"], b["id"], b["catalog"], b["t_pub"]), ("binance", "abc", 48, 1700000000000))
        self.assertIn("abc", b["url"])
        o = AN.parse_okx({"title": "OKX to list X (X) for spot trading", "pTime": "1700000000000", "url": "https://okx/x"})
        self.assertEqual((o["source"], o["t_pub"], o["id"]), ("okx", 1700000000000, "https://okx/x"))
        y = AN.parse_bybit({"title": "Bybit to List X (X) on Spot", "publishTime": 1700000000000, "dateTimestamp": 1,
                            "url": "https://bybit/x"})
        self.assertEqual((y["source"], y["t_pub"]), ("bybit", 1700000000000))
        y2 = AN.parse_bybit({"title": "Bybit to List X (X) on Spot", "dateTimestamp": 1700000001000, "url": "https://bybit/y"})
        self.assertEqual(y2["t_pub"], 1700000001000)                       # publishTime missing on old pages

    def test_store_idempotent_and_t_seen(self):
        import announcements as AN
        with tempfile.TemporaryDirectory() as d:
            st = AN.AnnouncementStore(Path(d))
            r = [rec("binance", "A (A)"), rec("binance", "B (B)")]
            now = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
            self.assertEqual(st.append(r, now), 2)
            self.assertEqual(st.append(r, now), 0)
            rows = st.load("binance")
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(x["t_seen"] == int(now.timestamp() * 1000) for x in rows))
            self.assertEqual(rows[0]["t_pub"], ms("2025-01-01 00:00"))      # t_pub stays the source's time

    def test_drop_migrated_bulk(self):
        import announcements as AN
        bulk = [rec("okx", f"OKX Launches T{k}", t="2023-03-03 07:53") for k in range(25)]
        normal = [rec("okx", "OKX to list Dash (DASH) for spot trading", t="2025-11-25 09:00")]
        out = AN.drop_migrated(bulk + normal, min_cluster=20)
        self.assertEqual([r["title"] for r in out], [normal[0]["title"]])

    def test_backfill_pages_until_since(self):
        import announcements as AN
        pages = {1: [{"code": "a", "title": "x", "releaseDate": ms("2025-05-01 00:00")}],
                 2: [{"code": "b", "title": "y", "releaseDate": ms("2024-01-01 00:00")}],
                 3: [{"code": "c", "title": "z", "releaseDate": ms("2022-06-01 00:00")}],
                 4: [{"code": "d", "title": "w", "releaseDate": ms("2021-06-01 00:00")}]}
        calls = []

        def fetch(p):
            calls.append(p)
            return pages.get(p, [])
        out = AN.backfill(fetch, lambda a: AN.parse_binance(a, 48), since_ms=ms("2023-01-01 00:00"), max_pages=50)
        self.assertEqual([r["id"] for r in out], ["a", "b"])                 # c is older than since
        self.assertEqual(calls, [1, 2, 3])                                   # stops after the first page older than since
        self.assertEqual(AN.backfill(lambda p: [], lambda a: a, since_ms=0, max_pages=5), [])


# ------------------------------------------------------------------ 2-3. classification and tickers (real titles)
class TestClassify(unittest.TestCase):
    def check(self, source, catalog, cases, etype):
        import announcements as AN
        for c in cases:
            title, tick = (c, None) if isinstance(c, str) else c
            r = rec(source, title, catalog)
            self.assertEqual(AN.classify(r), etype, title)
            if tick is not None:
                self.assertEqual(AN.extract_tickers(title, etype), tick, title)

    def test_binance(self):
        self.check("binance", 48, BN_PERP, "BN_PERP_NEW")
        self.check("binance", 48, BN_SPOT, "BN_SPOT_NEW")
        self.check("binance", 48, BN_MARGIN, "BN_MARGIN")
        self.check("binance", 48, BN_OTHER, "BN_OTHER")
        self.check("binance", 161, BN_DELIST, "BN_DELIST")
        self.check("binance", 161, BN161_OTHER, "BN_OTHER")

    def test_okx(self):
        self.check("okx", None, OKX_SPOT, "OKX_SPOT_NEW")
        self.check("okx", None, OKX_PERP, "OKX_PERP_NEW")
        self.check("okx", None, OKX_OTHER, "OKX_OTHER")

    def test_bybit(self):
        self.check("bybit", None, BYBIT_SPOT, "BYBIT_SPOT_NEW")
        self.check("bybit", None, BYBIT_PERP, "BYBIT_PERP_NEW")
        self.check("bybit", None, BYBIT_OTHER, "BYBIT_OTHER")

    def test_no_ticker_and_dates_are_not_tickers(self):
        import announcements as AN
        self.assertEqual(AN.extract_tickers("Binance Margin Will Add New Pairs - 2026-09-23", "BN_MARGIN"), [])
        self.assertEqual(AN.extract_tickers("Binance Will List 牛来 (牛来) with Seed Tag Applied", "BN_SPOT_NEW"), [])
        self.assertNotIn("2026", AN.extract_tickers("Binance Futures Will Launch USDⓈ-Margined CTUSDT Perpetual Contract "
                                                    "(2026-10-01)", "BN_PERP_NEW"))

    def test_to_spot(self):
        import announcements as AN
        spot = {"PEPEUSDT", "BMTUSDT", "UNIUSDT", "BONKUSDT"}
        self.assertEqual(AN.to_spot(["BMT", "MUBARAK"], "BN_PERP_NEW", spot), ["BMTUSDT"])
        self.assertEqual(AN.to_spot(["1000PEPE"], "BN_PERP_NEW", spot), ["PEPEUSDT"])
        self.assertEqual(AN.to_spot(["1000BONK"], "BYBIT_PERP_NEW", spot), ["BONKUSDT"])
        self.assertEqual(AN.to_spot(["UNI"], "OKX_SPOT_NEW", spot), ["UNIUSDT"])
        self.assertEqual(AN.to_spot([], "BN_OTHER", spot), [])


# ------------------------------------------------------------------ 4-7. tradability, entry time, leak, controls
class TestTiming(unittest.TestCase):
    def test_tradable_needs_a_closed_bar_before_t_pub(self):
        import announcements as AN
        t = ms("2025-03-17 11:31")
        self.assertTrue(AN.tradable("BMTUSDT", t, {"BMTUSDT": t - 2 * BAR}))
        self.assertFalse(AN.tradable("BMTUSDT", t, {"BMTUSDT": t - 5 * 60 * 1000}))   # bar still open at t_pub
        self.assertFalse(AN.tradable("NEWUSDT", t, {}))

    def test_entry_bar_never_the_publication_bar(self):
        import announcements as AN
        self.assertEqual(AN.entry_bar_open(ms("2026-10-28 03:00"), 60_000), ms("2026-10-28 03:15"))
        self.assertEqual(AN.entry_bar_open(ms("2026-10-28 03:07"), 60_000), ms("2026-10-28 03:15"))
        self.assertEqual(AN.entry_bar_open(ms("2026-10-28 03:14") + 30_000, 60_000), ms("2026-10-28 03:30"))

    def test_pre_return_uses_bars_closed_by_t_pub(self):
        import announcements as AN
        t0 = ms("2025-01-01 00:00")
        n = 200
        c = np.array([100.0 * (1.001 ** i) for i in range(n)])
        arr = {"t": np.arange(n, dtype=np.int64) * BAR + t0, "c": c}
        t_pub = t0 + 120 * BAR + 7 * 60 * 1000              # inside bar 120 (still open)
        r4 = AN.pre_return(arr, t_pub, 4)
        self.assertAlmostEqual(r4, c[119] / c[119 - 16] - 1, places=12)   # bar 119 closed at bar 120's open
        c2 = c.copy()
        c2[120:] *= 3
        self.assertAlmostEqual(AN.pre_return({"t": arr["t"], "c": c2}, t_pub, 4), r4, places=12)

    def test_controls_reproducible_and_away_from_events(self):
        import announcements as AN
        t = ms("2025-03-17 11:31")
        ev = {"BMTUSDT": [t, ms("2025-03-25 00:00")]}
        a = AN.control_time("BMTUSDT", t, ev, random.Random(5))
        b = AN.control_time("BMTUSDT", t, ev, random.Random(5))
        self.assertEqual(a, b)
        self.assertEqual(datetime.fromtimestamp(a / 1000, timezone.utc).month, 3)
        self.assertTrue(all(abs(a - x) >= 48 * HOUR for x in ev["BMTUSDT"]))
        coins = ["AUSDT", "BUSDT", "CUSDT", "BMTUSDT"]
        ev2 = {"AUSDT": [t + HOUR], "BMTUSDT": [t]}
        picks = {AN.control_coin(t, coins, ev2, random.Random(s)) for s in range(40)}
        self.assertTrue(picks <= {"BUSDT", "CUSDT"})
        self.assertEqual(AN.control_coin(t, coins, ev2, random.Random(9)), AN.control_coin(t, coins, ev2, random.Random(9)))


# ------------------------------------------------------------------ 8. isolation
class TestIsolation(unittest.TestCase):
    def test_public_hosts_only(self):
        import announcements as AN
        import exchange_client as X
        for h in AN.HOSTS.values():
            host = h.split("//")[1]
            self.assertIn(host, X.PUBLIC_HOSTS, host)
            self.assertNotIn(host, X.ALLOWED_HOSTS, host)
        src = (HERE / "announcements.py").read_text(encoding="utf-8")
        for bad in ("signed=True", "place_order(", "account(", '"' + '.runtime' + '"', "BOT_FILES /"):
            self.assertNotIn(bad, src, bad)
        import trader_config as TC
        self.assertEqual(AN.ROOT, TC.RUNTIME / "announcements")


if __name__ == "__main__":
    unittest.main(verbosity=2)
