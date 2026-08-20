"""Guards for the status panel: live flags, and the day's leaders.

The panel this replaces said "Открытых сигналов: 0" on a day when three watchlist
coins sat in Binance's daily top-20. It was accurate and useless — it reported an
empty portfolio without saying what the market did or why nothing was taken, and
finding that out took a full log excavation.

Two properties matter and both are easy to lose silently: the panel must never
block the render path, and the version line must describe what is RUNNING rather
than what git happens to point at.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ui_leaders as U  # noqa: E402

BOT_SRC = (HERE / "bot.py").read_text(encoding="utf-8")
LOG_SRC = (HERE / "botlog.py").read_text(encoding="utf-8")

TICK = [
    {"symbol": "XRPUSDT", "openPrice": "1.00", "highPrice": "1.21",
     "lastPrice": "1.13", "priceChangePercent": "13.0"},
    {"symbol": "ORDIUSDT", "openPrice": "3.60", "highPrice": "4.45",
     "lastPrice": "4.07", "priceChangePercent": "11.0"},
    {"symbol": "BTCUSDT", "openPrice": "70000", "highPrice": "72000",
     "lastPrice": "71000", "priceChangePercent": "1.4"},
    {"symbol": "NOTINWLUSDT", "openPrice": "1", "highPrice": "9",
     "lastPrice": "9", "priceChangePercent": "800"},
]
WL = ["XRPUSDT", "ORDIUSDT", "BTCUSDT"]


class TestRankingIsByMoveNotClose(unittest.TestCase):
    """The target is the day's largest MOVE. A coin that ran +25% and gave it
    all back belongs at the top — catching that run was the bot's job."""

    def test_it_ranks_by_high_over_open(self):
        rows = U.compute_leaders(WL, [], limit=5, tickers=TICK)
        self.assertEqual([r.sym for r in rows], ["ORDIUSDT", "XRPUSDT", "BTCUSDT"])
        self.assertAlmostEqual(rows[0].move_pct, (4.45 / 3.60 - 1) * 100, places=6)

    def test_close_change_is_shown_but_does_not_order(self):
        # ORDI closed +11.0% against XRP's +13.0% and still ranks first on move.
        rows = U.compute_leaders(WL, [], limit=5, tickers=TICK)
        self.assertGreater(rows[1].change_pct, rows[0].change_pct)

    def test_non_watchlist_symbols_are_excluded(self):
        # The watchlist is immutable; a +800% coin the operator cannot trade is
        # noise on this panel, not a finding.
        rows = U.compute_leaders(WL, [], limit=9, tickers=TICK)
        self.assertNotIn("NOTINWLUSDT", [r.sym for r in rows])

    def test_a_zero_open_price_is_skipped_not_divided_by(self):
        bad = TICK + [{"symbol": "ZEROUSDT", "openPrice": "0", "highPrice": "1",
                       "lastPrice": "1", "priceChangePercent": "0"}]
        rows = U.compute_leaders(WL + ["ZEROUSDT"], [], limit=9, tickers=bad)
        self.assertNotIn("ZEROUSDT", [r.sym for r in rows])


class TestItSaysWhyACoinIsNotHeld(unittest.TestCase):
    """The question that cost an evening was 'it's in the top-20, why isn't it
    in the portfolio'. The panel answers it inline."""

    def setUp(self):
        U._LAST_BLOCK.clear()

    def test_held_coins_are_marked_and_carry_no_reason(self):
        rows = U.compute_leaders(WL, ["XRPUSDT"], limit=5, tickers=TICK)
        xrp = next(r for r in rows if r.sym == "XRPUSDT")
        self.assertTrue(xrp.held)
        self.assertEqual(xrp.block_reason, "")

    def test_a_rejected_coin_carries_the_gate_that_rejected_it(self):
        U.note_block("ORDIUSDT", "ml_zone", "ML proba 0.03 outside zone")
        rows = U.compute_leaders(WL, [], limit=5, tickers=TICK)
        self.assertEqual(next(r for r in rows if r.sym == "ORDIUSDT").block_reason,
                         "ml_zone")

    def test_stale_rejections_are_dropped(self):
        # An hour-old reason would describe a market that no longer exists.
        U.note_block("ORDIUSDT", "ml_zone", "x")
        U._LAST_BLOCK["ORDIUSDT"] = (time.time() - 7200, "ml_zone", "x")
        self.assertEqual(U.last_block("ORDIUSDT"), "")

    def test_the_recorder_is_hooked_into_the_single_choke_point(self):
        # Every gate funnels through log_blocked; hooking anywhere else would
        # cover some gates and silently miss others.
        self.assertIn("ui_leaders.note_block(sym, reason_code, reason)", LOG_SRC)


class TestTheRenderPathNeverBlocks(unittest.TestCase):
    def test_get_cached_returns_immediately_before_any_refresh(self):
        t0 = time.time()
        U.get_cached()
        self.assertLess(time.time() - t0, 0.05)

    def test_the_menu_reads_the_cache_rather_than_fetching(self):
        self.assertIn("ui_leaders.get_cached()", BOT_SRC)
        self.assertNotIn("ui_leaders.compute_leaders(", BOT_SRC)

    def test_the_refresher_runs_on_its_own_thread_not_the_5s_keeper(self):
        # The snapshot keeper ticks every 5s and this makes an HTTP call.
        self.assertIn("ui_leaders.start(", BOT_SRC)
        self.assertIn("UI_LEADERS_REFRESH_SEC", BOT_SRC)

    def test_an_empty_cache_renders_a_message_rather_than_raising(self):
        self.assertIn("ещё не загружены", U.render(()))

    def test_stale_data_is_labelled(self):
        rows = U.compute_leaders(WL, [], limit=2, tickers=TICK)
        self.assertIn("устарели", U.render(rows, age_sec=10 * U.REFRESH_SEC))


class TestTheVersionLineDescribesWhatIsRunning(unittest.TestCase):
    """git HEAD read `a041af9 · 08-19` for hours while the process ran a config
    edited one minute before startup, because the commits went through a separate
    worktree. The banner now prints the flags that decide behaviour."""

    def test_the_flags_line_exists_and_reads_the_live_config(self):
        self.assertIn("def _live_flags_line()", BOT_SRC)
        for flag in ("ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE",
                     "ML_GENERAL_HARD_BLOCK_MIN", "BANDIT_ENABLED",
                     "ROTATION_ENABLED"):
            self.assertIn(flag, BOT_SRC)

    def test_the_config_mtime_is_shown(self):
        self.assertIn("def _config_stamp()", BOT_SRC)
        self.assertIn("getmtime", BOT_SRC)

    def test_the_menu_actually_renders_both(self):
        self.assertIn("_live_flags_line()", BOT_SRC.split("def _main_menu_text")[1])
        self.assertIn("_config_stamp()", BOT_SRC.split("def _main_menu_text")[1])

    def test_it_degrades_instead_of_breaking_the_menu(self):
        # A missing attribute must not take the whole panel down.
        self.assertIn('return "flags unavailable"', BOT_SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
