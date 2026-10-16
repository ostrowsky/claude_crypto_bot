"""Trader audit log -- tests written from trader-audit-log-1012-spec.md §7 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import io
import json
import os
import stat
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exchange_client as X
import fill_model as FM
import trading_policy as TP

HERE = Path(__file__).resolve().parent
KEY, SECRET = "KEY_abcdefgh12345678", "SECRET_zyxw98765432"


class FakeResp(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(json.dumps(body).encode())
        self.headers = headers or {"x-mbx-used-weight-1m": "42"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def rows_in(root):
    out = []
    for f in sorted(Path(root).rglob("*.jsonl")):
        out += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def writer(root, kind="exchange", process="test"):
    import audit as A
    return A.AuditWriter(kind, process, root=Path(root))


class TestExchangeLog(unittest.TestCase):
    """§7.1–7.3."""

    def test_one_line_per_request_with_fields(self):
        with tempfile.TemporaryDirectory() as d:
            c = X.ExchangeClient(key=KEY, secret=SECRET, opener=lambda req, timeout=None: FakeResp(
                {"serverTime": 1} if "/time" in req.full_url else {"balances": [{"asset": "USDT", "free": "5", "locked": "0"},
                                                                                {"asset": "BNB", "free": "0", "locked": "0"}],
                                                                   "canTrade": True}), audit=writer(d))
            c.account()
            rows = rows_in(d)
            paths = [r["path"] for r in rows]
            self.assertEqual(paths.count("/api/v3/account"), 1)
            acc = next(r for r in rows if r["path"] == "/api/v3/account")
            for k in ("schema", "ts", "process", "run_id", "host", "method", "params", "signed", "status", "ok",
                      "latency_ms", "bytes", "used_weight_1m", "retry"):
                self.assertIn(k, acc)
            self.assertEqual(acc["schema"], "audit.exchange.v1")
            self.assertTrue(acc["signed"] and acc["ok"])
            self.assertEqual(acc["used_weight_1m"], 42)
            self.assertIn("body", acc)                                       # account body is kept (summarised)
            self.assertEqual([b["asset"] for b in acc["body"]["balances"]], ["USDT"])

    def test_errors_are_logged(self):
        def http_err(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                         io.BytesIO(json.dumps({"code": -1013, "msg": "Filter failure: LOT_SIZE"}).encode()))

        def net_err(req, timeout=None):
            raise TimeoutError("The read operation timed out")
        with tempfile.TemporaryDirectory() as d:
            for op in (http_err, net_err):
                c = X.ExchangeClient(key="", secret="", opener=op, audit=writer(d))
                with self.assertRaises(X.ExchangeError):
                    c.book_tickers()
            rows = rows_in(d)
            self.assertEqual([r["error_code"] for r in rows], [-1013, "network"])
            self.assertTrue(all(r["ok"] is False for r in rows))

    def test_retry_after_1021_is_its_own_line(self):
        n = {"k": 0}

        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1_000_000_000_000})
            n["k"] += 1
            if n["k"] == 1:
                raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                             io.BytesIO(json.dumps({"code": -1021, "msg": "ts"}).encode()))
            return FakeResp({"balances": [], "canTrade": True})
        with tempfile.TemporaryDirectory() as d:
            X.ExchangeClient(key=KEY, secret=SECRET, opener=opener, audit=writer(d)).account()
            acc = [r for r in rows_in(d) if r["path"] == "/api/v3/account"]
            self.assertEqual([r["retry"] for r in acc], [False, True])

    def test_no_secrets_anywhere(self):
        def opener(req, timeout=None):
            if "/time" in req.full_url:
                return FakeResp({"serverTime": 1})
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                         io.BytesIO(json.dumps({"code": -2015, "msg": f"Invalid key {KEY}"}).encode()))
        with tempfile.TemporaryDirectory() as d:
            c = X.ExchangeClient(key=KEY, secret=SECRET, opener=opener, audit=writer(d))
            with self.assertRaises(X.ExchangeError):
                c.account()
            text = "".join(f.read_text(encoding="utf-8") for f in Path(d).rglob("*.jsonl"))
            for bad in (KEY, SECRET, "signature", "X-MBX-APIKEY", "apiKey"):
                self.assertNotIn(bad, text, bad)

    def test_bodies_only_for_money_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            c = X.ExchangeClient(key="", secret="", opener=lambda req, timeout=None: FakeResp({"symbols": [1, 2, 3]}),
                                 audit=writer(d))
            c.exchange_info(max_age_sec=0)
            c.depth("BTCUSDT")
            for r in rows_in(d):
                self.assertNotIn("body", r)
                self.assertGreater(r["bytes"], 0)


class TestRobustness(unittest.TestCase):
    """§7.4–7.5."""

    def test_unwritable_audit_does_not_break_request(self):
        import audit as A
        w = A.AuditWriter("exchange", "test", root=Path("Z:/definitely/not/writable/\0bad"))
        c = X.ExchangeClient(key="", secret="", opener=lambda req, timeout=None: FakeResp({"serverTime": 7}), audit=w)
        self.assertEqual(c.server_time(), 7)
        self.assertGreaterEqual(w.failures, 1)

    def test_daily_rotation_utc(self):
        import audit as A
        with tempfile.TemporaryDirectory() as d:
            w = A.AuditWriter("decisions", "test", root=Path(d))
            w.write({"x": 1}, now=datetime(2026, 10, 12, 23, 59, 59, tzinfo=timezone.utc))
            w.write({"x": 2}, now=datetime(2026, 10, 13, 0, 0, 1, tzinfo=timezone.utc))
            files = sorted(p.name for p in (Path(d) / "decisions").glob("*.jsonl"))
            self.assertEqual(files, ["2026-10-12.jsonl", "2026-10-13.jsonl"])


class TestExplainedDecision(unittest.TestCase):
    """§7.6."""

    def test_every_reason_code(self):
        base = TP.CHAMPION
        cases = [
            (base, {"kind": "blocked"}, {}, "not_bot_entry"),
            (replace(base, admission=TP.Admission(kind="all_candidates", modes=("trend",))), {"mode": "x"}, {}, "mode_filtered"),
            (replace(base, admission=TP.Admission(kind="all_candidates", min_score=0.5)), {}, {}, "score_missing"),
            (replace(base, admission=TP.Admission(kind="all_candidates", min_score=0.5)), {"ml_proba": 0.1}, {}, "score_below_min"),
            (replace(base, admission=TP.Admission(kind="all_candidates", random_frac=0.0)), {"sym": "A", "ts": "t"}, {}, "random_excluded"),
            (replace(base, admission=TP.Admission(kind="all_candidates", max_per_day=1)), {}, {"day_count": 1}, "daily_cap"),
            (replace(base, admission=TP.Admission(kind="all_candidates")), {}, {"open": 10}, "slots_full"),
            (replace(base, admission=TP.Admission(kind="all_candidates"), risk=TP.Risk(one_per_symbol=True)), {"sym": "A"},
             {"open_syms": {"A"}}, "symbol_held"),
            (replace(base, admission=TP.Admission(kind="all_candidates"), risk=TP.Risk(daily_loss_limit_pct=2)), {},
             {"day_return_pct": -3}, "daily_loss_limit"),
            (replace(base, admission=TP.Admission(kind="all_candidates"), risk=TP.Risk(cooldown_bars=5)), {"sym": "A"},
             {"cooldown_until": {"A": 10}, "bar": 5}, "cooldown"),
            (replace(base, admission=TP.Admission(kind="all_candidates"), sizing=TP.Sizing(kind="risk")), {}, {}, "no_stop_distance"),
            (replace(base, admission=TP.Admission(kind="all_candidates"), sizing=TP.Sizing(max_position_pct=0.0)), {}, {}, "zero_size"),
        ]
        for pol, cand, st, code in cases:
            frac, why = TP.decide_entry_explained(pol, cand, st)
            self.assertIsNone(frac, code)
            self.assertEqual(why, code)
            self.assertIsNone(TP.decide_entry(pol, cand, st))
        frac, why = TP.decide_entry_explained(base, {"kind": "entry"}, {})
        self.assertEqual((frac, why), (0.1, "enter"))
        self.assertEqual(TP.decide_entry(base, {"kind": "entry"}, {}), 0.1)


class TestShadowDecisions(unittest.TestCase):
    """§7.7–7.8."""

    def test_one_decision_per_book_and_exits(self):
        import shadow_portfolio as S
        from test_shadow_portfolio import two_sym_grid, grid_events, X4 as X4P, T0 as ST0, STEP as SSTEP
        g = two_sym_grid()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            evs = grid_events(g, n_trades=20, seed=11)
            p.write_text("".join(json.dumps(r) + "\n" for r in evs), encoding="utf-8")
            w = writer(d, "decisions", "shadow")
            eng = S.ShadowEngine({"champion": TP.CHAMPION, "x4_floor8": X4P}, FM.model("taker", 1),
                                 floor_of=lambda m: 0.0, bars=S.GridBars(g), decisions=w)
            eng.process(S.EventTail(p, {}).read_new(), now=ST0 + (g.N - 1) * SSTEP)
            rows = rows_in(Path(d) / "decisions")            # not the ev.jsonl input that sits in the same dir
            n_entries = sum(1 for e in evs if e["event"] == "entry")
            ent = [r for r in rows if r["kind"] == "entry_decision"]
            self.assertEqual(len(ent), 2 * n_entries)
            self.assertTrue(all(r["action"] in ("enter", "skip") and r["reason"] for r in ent))
            n_closed = len(eng.trades("champion")) + len(eng.trades("x4_floor8"))
            ex = [r for r in rows if r["kind"] == "exit_decision"]
            self.assertEqual(len(ex), n_closed)
            self.assertTrue(all("price_after_costs" in r and "reason" in r for r in ex))
            marks = [(r["book"], r["day"]) for r in rows if r["kind"] == "day_mark"]
            self.assertEqual(len(marks), len(set(marks)))
            ids = [r["decision_id"] for r in ent]
            self.assertEqual(len(ids), len(set(ids)))                          # different per book and event

    def test_decision_id_deterministic(self):
        import audit as A
        a = A.decision_id("champion", "AUSDT", "15m", "2026-10-12T10:00:00+00:00", "entry_decision")
        b = A.decision_id("champion", "AUSDT", "15m", "2026-10-12T10:00:00+00:00", "entry_decision")
        c = A.decision_id("x4_floor8", "AUSDT", "15m", "2026-10-12T10:00:00+00:00", "entry_decision")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestReport(unittest.TestCase):
    """§7.9."""

    def test_summary(self):
        import audit_report as AR
        ex = [{"path": "/api/v3/depth", "ok": True, "latency_ms": v} for v in (10, 20, 30, 40)] + \
             [{"path": "/api/v3/depth", "ok": False, "latency_ms": 100, "error_code": "network"}]
        dec = [{"kind": "entry_decision", "book": "x4", "action": "skip", "reason": "slots_full"}] * 3 + \
              [{"kind": "entry_decision", "book": "x4", "action": "enter", "reason": "enter"}]
        s = AR.summarize(ex, dec)
        self.assertAlmostEqual(s["exchange"]["/api/v3/depth"]["error_share"], 0.2)
        self.assertEqual(s["exchange"]["/api/v3/depth"]["p50_ms"], 30)
        self.assertEqual(s["decisions"]["x4"]["skip_reasons"]["slots_full"], 3)
        self.assertEqual(s["decisions"]["x4"]["enter"], 1)


class TestBoundary(unittest.TestCase):
    """§7.10."""

    def test_audit_root_in_trader_runtime(self):
        import audit as A
        import trader_config as TC
        self.assertEqual(A.ROOT, TC.RUNTIME / "audit")
        self.assertTrue(TC.AUDIT_EXCHANGE_ENABLED and TC.AUDIT_DECISIONS_ENABLED)
        src = (HERE / "audit.py").read_text(encoding="utf-8")
        self.assertIn('"a"', src)
        self.assertNotIn('"w"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
