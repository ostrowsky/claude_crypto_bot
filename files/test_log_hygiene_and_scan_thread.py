"""Token redaction in logs and the market-scan analysis thread (2026-09-28)."""
import asyncio
import io
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

TOKEN = "1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"


def _bot_filter_class():
    """Build _RedactSecrets from bot.py source without importing bot.py (it starts the app state)."""
    src = (HERE / "bot.py").read_text(encoding="utf-8")
    start = src.index("class _RedactSecrets(logging.Filter):")
    end = src.index("def _install_log_hygiene()")
    ns = {"logging": logging}
    exec(src[start:end], ns)
    return ns["_RedactSecrets"]


class TestRedaction(unittest.TestCase):
    def setUp(self):
        self.buf = io.StringIO()
        self.h = logging.StreamHandler(self.buf)
        self.h.setFormatter(logging.Formatter("%(message)s"))
        self.h.addFilter(_bot_filter_class()([TOKEN]))
        self.log = logging.getLogger("test_redact")
        self.log.addHandler(self.h)
        self.log.setLevel(logging.INFO)
        self.log.propagate = False

    def tearDown(self):
        self.log.removeHandler(self.h)

    def test_message_and_args(self):
        self.log.info("POST https://api.telegram.org/bot%s/sendMessage", TOKEN)
        self.log.info("plain https://api.telegram.org/bot" + TOKEN + "/getUpdates")
        out = self.buf.getvalue()
        self.assertNotIn(TOKEN, out)
        self.assertEqual(out.count("<TELEGRAM_TOKEN>"), 2)

    def test_traceback(self):
        try:
            raise RuntimeError("url https://api.telegram.org/bot%s/x" % TOKEN)
        except RuntimeError:
            self.log.exception("failed")
        self.assertNotIn(TOKEN, self.buf.getvalue())

    def test_short_or_empty_secret_is_ignored(self):
        f = _bot_filter_class()(["", "abc"])
        rec = logging.LogRecord("x", logging.INFO, "", 0, "abc text", (), None)
        self.assertTrue(f.filter(rec))
        self.assertEqual(rec.getMessage(), "abc text")


class TestWiring(unittest.TestCase):
    def test_bot_installs_it_and_quiets_httpx(self):
        src = (HERE / "bot.py").read_text(encoding="utf-8")
        self.assertIn("_install_log_hygiene()", src)
        self.assertIn('logging.getLogger("httpx").setLevel', src)
        self.assertEqual(config.HTTPX_LOG_LEVEL, "WARNING")

    def test_market_scan_analysis_runs_in_a_thread(self):
        src = (HERE / "strategy.py").read_text(encoding="utf-8")
        self.assertIn("await asyncio.to_thread(analyze_coin, sym, tf, data, from_scan=from_scan)", src)
        self.assertIs(config.ANALYSIS_IN_THREAD, True)

    def test_thread_path_returns_the_same_reports(self):
        import strategy

        async def fake_fetch(session, sym, tf):
            return {"sym": sym, "tf": tf}

        def fake_analyze(sym, tf, data, from_scan=False):
            return strategy.CoinReport(symbol=sym, tf=tf, today_signals=[], today_confirmed=False, signal_now=False,
                                       today_accuracy={}, best_horizon=0, best_accuracy=0.0, in_play=False, note="x")

        out = {}
        for flag in (False, True):
            with mock.patch.object(strategy, "fetch_klines", fake_fetch), \
                 mock.patch.object(strategy, "analyze_coin", fake_analyze), \
                 mock.patch.object(config, "ANALYSIS_IN_THREAD", flag):
                in_play, skipped = asyncio.run(strategy._run_analysis(["AUSDT", "BUSDT"]))
            out[flag] = sorted((r.symbol, r.tf) for r in in_play + skipped)
        self.assertEqual(out[False], out[True])
        self.assertEqual(len(out[True]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
