"""Hang fix of the A0 recorder (demo-connection-a-1010-spec.md, «Исправление 2026-10-10»).

Written from the spec BEFORE the code (operator rule 2026-10-10: spec -> tests -> code).
"""
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import exchange_client as X
import microstructure_recorder as M


class SlowBody(io.BytesIO):
    """A response that keeps sending: every read() returns a little data after a pause."""

    def __init__(self, chunks, pause):
        super().__init__()
        self._chunks, self._pause = list(chunks), pause
        self.headers = {}

    def read(self, n=-1):
        out = b""
        while self._chunks:
            time.sleep(self._pause)
            out += self._chunks.pop(0)
        return out

    def read1(self, n=-1):
        if not self._chunks:
            return b""
        time.sleep(self._pause)
        return self._chunks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestRequestDeadline(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(X.REQUEST_DEADLINE_SEC, 30)

    def test_slow_trickle_is_cut_by_total_deadline(self):
        """Requirement 1: a body that keeps arriving must not outlive the deadline."""
        body = [b"{"] + [b" " * 10] * 50 + [b"}"]
        c = X.ExchangeClient(key="k", secret="s", opener=lambda req, timeout=None: SlowBody(body, 0.05))
        old = X.REQUEST_DEADLINE_SEC
        X.REQUEST_DEADLINE_SEC = 0.5
        try:
            t0 = time.time()
            with self.assertRaises(X.RequestDeadline):
                c.server_time()
            self.assertLess(time.time() - t0, 1.5)
        finally:
            X.REQUEST_DEADLINE_SEC = old

    def test_fast_response_passes(self):
        c = X.ExchangeClient(key="k", secret="s",
                             opener=lambda req, timeout=None: SlowBody([b'{"serverTime": 5}'], 0.0))
        self.assertEqual(c.server_time(), 5)


class TestExchangeInfoDiskCache(unittest.TestCase):
    def _client(self, calls):
        def opener(req, timeout=None):
            calls.append(req.full_url)
            return SlowBody([json.dumps({"symbols": [], "rateLimits": []}).encode()], 0.0)
        return X.ExchangeClient(key="", secret="", opener=opener)

    def test_fresh_cache_means_no_request(self):
        """Requirement 2: a cache younger than a day is used and the network is not touched."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "exchange_info.json"
            p.write_text(json.dumps({"fetched_at": time.time() - 3600, "data": {"symbols": [{"symbol": "AUSDT"}]}}),
                         encoding="utf-8")
            calls = []
            x = M.cached_exchange_info(self._client(calls), path=p)
            self.assertEqual(calls, [])
            self.assertEqual(x["symbols"][0]["symbol"], "AUSDT")

    def test_stale_or_missing_or_broken_cache_fetches_and_rewrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "exchange_info.json"
            for content in (None, "{broken", json.dumps({"fetched_at": time.time() - 2 * 86400, "data": {}})):
                if content is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_text(content, encoding="utf-8")
                calls = []
                M.cached_exchange_info(self._client(calls), path=p)
                self.assertEqual(len(calls), 1)
                saved = json.loads(p.read_text(encoding="utf-8"))
                self.assertGreater(saved["fetched_at"], time.time() - 60)

    def test_max_age_constant(self):
        self.assertEqual(M.EXCHANGE_INFO_MAX_AGE_SEC, 86400)

    def test_snapshot_uses_the_cache(self):
        src = (Path(M.__file__)).read_text(encoding="utf-8")
        self.assertIn("cached_exchange_info(client)", src)
        self.assertNotIn("xinfo = client.exchange_info()", src)


class TestWatchdogsAndOrder(unittest.TestCase):
    def test_watchdogs(self):
        here = Path(M.__file__).parent
        self.assertIn("dump_traceback_later(200, exit=True)", (here / "microstructure_recorder.py").read_text(encoding="utf-8"))
        self.assertIn("dump_traceback_later(60, exit=True)", (here / "equity_snapshots.py").read_text(encoding="utf-8"))

    def test_cmd_runs_equity_first(self):
        cmd = (Path(M.__file__).parent / "run_recorder.cmd").read_text(encoding="utf-8")
        self.assertLess(cmd.index("equity_snapshots.py"), cmd.index("microstructure_recorder.py snapshot"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUniverseUsesCache(unittest.TestCase):
    def test_universe_main_reads_the_daily_cache(self):
        src = (Path(M.__file__).parent / "trading_universe.py").read_text(encoding="utf-8")
        self.assertIn("cached_exchange_info(", src)
        self.assertNotIn("u = build(c.exchange_info())", src)


class TestSchedulerTask(unittest.TestCase):
    """Requirement 5: the scheduled task must not run below normal priority (CPU starvation)."""

    def test_task_priority_normal(self):
        import subprocess
        import sys
        if sys.platform != "win32":
            self.skipTest("Windows Task Scheduler only")
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-ScheduledTask -TaskName CryptoBot_MicrostructureRecorder).Settings.Priority"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not r.stdout.strip():
            self.skipTest("task not registered on this machine")
        self.assertLessEqual(int(r.stdout.strip()), 6)

    def test_tasks_run_without_a_console_window(self):
        """2026-10-13: both 5-minute tasks ran `cmd.exe /c ...` as Interactive and opened an empty
        console window every 5 minutes (all output goes to the log). They must start headless."""
        import subprocess
        import sys
        if sys.platform != "win32":
            self.skipTest("Windows Task Scheduler only")
        for name in ("CryptoBot_MicrostructureRecorder", "CryptoBot_TraderShadow"):
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"$a=(Get-ScheduledTask -TaskName {name}).Actions[0]; $a.Execute; $a.Arguments"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not r.stdout.strip():
                self.skipTest(f"{name} not registered on this machine")
            execute, _, args = r.stdout.strip().partition("\n")
            self.assertTrue(execute.strip().lower().endswith("conhost.exe"), (name, execute))
            self.assertTrue(args.strip().startswith("--headless "), (name, args))


class TestSnapshotRecord(unittest.TestCase):
    def test_elapsed_is_in_the_written_line(self):
        """The written record carries elapsed_sec (it was set after the line was written)."""
        src = Path(M.__file__).read_text(encoding="utf-8")
        i_el, i_wr = src.index('rec["elapsed_sec"]'), src.index('fh.write(json.dumps(rec')
        self.assertLess(i_el, i_wr)


class TestNetworkErrorsAreExchangeErrors(unittest.TestCase):
    """Bug 2026-10-11/12: TimeoutError / handshake timeouts escaped _request (not an ExchangeError), so a
    single network hiccup killed the whole snapshot (3 lost snapshots in 2 days)."""

    def test_timeout_and_urlerror_become_exchange_error(self):
        import urllib.error
        for exc in (TimeoutError("The read operation timed out"), urllib.error.URLError("handshake timed out"),
                    ConnectionResetError("reset")):
            def opener(req, timeout=None, _e=exc):
                raise _e
            c = X.ExchangeClient(key="", secret="", opener=opener)
            with self.assertRaises(X.ExchangeError) as cm:
                c.server_time()
            self.assertIsNone(cm.exception.status)
            self.assertEqual(cm.exception.code, "network")

    def test_snapshot_survives_network_errors(self):
        class DeadClient:
            used_weight = {}

            def book_tickers(self, base=None):
                raise X.ExchangeError(None, "network", "timed out", "/api/v3/ticker/bookTicker")

            def depth(self, *a, **k):
                raise X.ExchangeError(None, "network", "timed out", "/api/v3/depth")

            def tickers_24h(self, base=None):
                raise X.ExchangeError(None, "network", "timed out", "/api/v3/ticker/24hr")
        old_cache, old_dir, old_state = M.cached_exchange_info, M.OUT_DIR, M.STATE
        with tempfile.TemporaryDirectory() as d:
            M.cached_exchange_info = lambda client, path=None, max_age=None: {"symbols": [], "rateLimits": []}
            M.OUT_DIR, M.STATE = Path(d), Path(d) / "_state.json"
            try:
                rec = M.snapshot(DeadClient())
            finally:
                M.cached_exchange_info, M.OUT_DIR, M.STATE = old_cache, old_dir, old_state
        self.assertIn("error", rec["hosts"]["live"])
        self.assertIn("error", rec["hosts"]["demo"])
