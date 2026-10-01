"""Dead pairs removed from the watchlist; daily liveness check (watchlist-dead-1001-spec.md)."""
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import config
import watchlist_liveness as WL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEAD = {"TONUSDT", "MKRUSDT", "LRCUSDT", "MDTUSDT", "OXTUSDT", "BAKEUSDT", "PYRUSDT", "TRUUSDT", "SNTUSDT"}
OLD = {"RNDRUSDT", "EOSUSDT", "ACAUSDT"}
NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


class TestCleanup(unittest.TestCase):
    def test_watchlist_has_no_dead_pair(self):
        wl = json.loads((HERE / "watchlist.json").read_text(encoding="utf-8"))
        self.assertFalse(DEAD & set(wl))
        self.assertEqual(len(wl), len(set(wl)))
        self.assertEqual(len(wl), 93)

    def test_backup_keeps_the_old_list(self):
        old = json.loads((HERE / "watchlist.pre_dead_cleanup_1001.json").read_text(encoding="utf-8"))
        self.assertTrue(DEAD <= set(old))
        self.assertEqual(len(old), 102)

    def test_fallback_list_is_clean_too(self):
        self.assertFalse((DEAD | OLD) & set(config.DEFAULT_WATCHLIST))


class TestLiveness(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(WL.classify("BREAK", NOW, NOW), "dead")
        self.assertEqual(WL.classify("TRADING", NOW - timedelta(hours=1), NOW), "live")
        self.assertEqual(WL.classify("TRADING", NOW - timedelta(days=3), NOW), "dead")
        self.assertEqual(WL.classify(None, None, NOW), "unknown")
        self.assertEqual(WL.classify("TRADING", None, NOW), "unknown")

    def test_exchange_outage_is_never_dead(self):
        def boom(url, timeout=20):
            raise OSError("down")
        r = WL.check(["AUSDT"], now=NOW, get=boom)
        self.assertFalse(r["available"])
        self.assertNotIn("dead", r)

    def test_check_finds_a_delisted_pair(self):
        ts = int((NOW - timedelta(minutes=15)).timestamp() * 1000)

        def fake(url, timeout=20):
            if "exchangeInfo" in url:
                return {"symbols": [{"symbol": "AUSDT", "status": "TRADING"}, {"symbol": "BUSDT", "status": "BREAK"}]}
            return [[ts, "1", "1", "1", "1"]]
        r = WL.check(["AUSDT", "BUSDT"], now=NOW, get=fake)
        self.assertEqual([d["symbol"] for d in r["dead"]], ["BUSDT"])

    def test_report_step_and_daily_run(self):
        import bot_health_report as H
        sc = {"north_star": {"status": "verified"}, "portfolio_alpha": {"value": -1.0},
              "realized_potential": {"value": 0.6, "target": 0.5}, "signal_precision": {}, "message_rate": {}}
        with mock.patch.object(WL, "load_latest", return_value={"available": True, "dead": [{"symbol": "BUSDT"}]}):
            ids = {s["id"]: s for s in H.derive_next_steps(sc, {"evaluation_scope": "out_of_sample_time_holdout"},
                                                         {}, date(2026, 10, 1))}
        self.assertIn("BUSDT", ids["remove_dead_watchlist_pairs"]["evidence"])
        src = (HERE / "pipeline_run.py").read_text(encoding="utf-8")
        self.assertLess(src.index("watchlist_liveness.py"), src.index('run_step("notify"'))

    def test_documented(self):
        self.assertTrue((ROOT / "docs/specs/features/watchlist-dead-1001-spec.md").exists())
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("watchlist-dead-1001-spec.md", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
