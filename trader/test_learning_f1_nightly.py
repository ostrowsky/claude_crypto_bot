"""F1 nightly build -- tests written from trader-learning-loop-f-1020-spec.md "F1 nightly -- контракт" BEFORE the
code (operator rule 2026-10-10: spec -> tests -> code). Synthetic journals, offline."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
T0 = datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)
EV = "2026-10-21T00:45:10+00:00"                        # the bot's entry event (paper book key time)
KEY = f"AAAUSDT|15m|{EV}"
KEY2 = "BBBUSDT|15m|2026-10-21T01:00:05+00:00"


def iso(dt):
    return dt.isoformat()


def micro_line(ts, sym="AAAUSDT", mid=2.0, spread=6.0, demo_spread=9.0, div=1.5, depth=True):
    d = {"ts": iso(ts),
         "hosts": {"live": {"book": {sym: [mid, spread], "BTCUSDT": [80000.0, 0.01]}},
                   "demo": {"book": {sym: [mid * 1.0001, demo_spread]}}},
         "divergence_bps": {sym: div},
         "depth": {"live": ({sym: {"buy": {"100": 3.0, "1000": 7.5}, "sell": {"100": 3.1, "1000": 8.0}}} if depth else {}),
                   "demo": {}},
         "vol24h": {"live": {sym: [1.5e6, 9000]}, "demo": {}}}
    return json.dumps(d)


def journal():
    """Executor + shadow rows for one day, as the audit journal writes them."""
    e = lambda **kw: {"schema": "audit.decision.v1", "process": "executor", "book": "exec_x4", "mode": "demo", **kw}  # noqa: E731
    s = lambda **kw: {"schema": "audit.decision.v1", "process": "shadow", "book": "x4_floor8", **kw}                # noqa: E731
    return [
        s(kind="entry_decision", action="enter", sym="AAAUSDT", tf="15m", event_ts=EV, ts=iso(T0 + timedelta(minutes=46)),
          candidate={"price": 2.0, "mode": "trend", "trail_k": 2.0, "ml_proba": 0.31, "candidate_score": 55.0,
                     "ranker_final_score": -0.4}),
        e(kind="real_entry", action="enter", key=KEY, sym="AAAUSDT", ts=iso(T0 + timedelta(minutes=46)),
          decision_id="d-enter-A", fill_price=2.004, paper_price=2.0, slippage_bps=20.0, latency_sec=50.0, cost=20.0),
        s(kind="entry_decision", action="enter", sym="BBBUSDT", tf="15m", event_ts=KEY2.split("|")[2],
          ts=iso(T0 + timedelta(minutes=61)), candidate={"price": 5.0, "mode": "impulse_speed", "trail_k": 2.5}),
        e(kind="real_entry", action="skip", key=KEY2, sym="BBBUSDT", ts=iso(T0 + timedelta(minutes=61)),
          decision_id="d-skip-B", reason="exposure_cap"),
        s(kind="exit_decision", sym="AAAUSDT", tf="15m", event_ts=EV, ts=iso(T0 + timedelta(hours=6)), ret_pct=-3.0,
          reason="trail"),
        s(kind="exit_decision", sym="BBBUSDT", tf="15m", event_ts=KEY2.split("|")[2], ts=iso(T0 + timedelta(hours=7)),
          ret_pct=4.0, reason="time"),
        e(kind="real_exit", action="exit", key=KEY, sym="AAAUSDT", ts=iso(T0 + timedelta(hours=6, minutes=1)),
          decision_id="d-exit-A", reason="trail", pnl=-0.7, ret_pct=-3.5),
    ]


class TestMicroFeatures(unittest.TestCase):
    def test_features_of_one_snapshot(self):
        import trade_dataset as TD
        f = TD.micro_features(json.loads(micro_line(T0)), "AAAUSDT")
        self.assertEqual(f, {"spread_bps_live": 6.0, "spread_bps_demo": 9.0, "divergence_bps": 1.5,
                             "depth_buy_1000_bps": 7.5, "depth_sell_1000_bps": 8.0, "qvol24h_live": 1.5e6})

    def test_missing_parts_are_none_and_absent_pair_is_none(self):
        import trade_dataset as TD
        f = TD.micro_features(json.loads(micro_line(T0, depth=False)), "AAAUSDT")
        self.assertIsNone(f["depth_buy_1000_bps"])
        self.assertIsNone(TD.micro_features(json.loads(micro_line(T0)), "ZZZUSDT"))

    def test_load_only_wanted_pairs_skips_broken_lines(self):
        import trade_dataset as TD
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2026-10-21.jsonl"
            p.write_text(micro_line(T0) + "\n{broken\n" + micro_line(T0 + timedelta(minutes=5), spread=4.0) + "\n",
                         encoding="utf-8")
            snaps = TD.load_micro([p], {"AAAUSDT"})
            self.assertEqual(set(snaps), {"AAAUSDT"})
            self.assertEqual([round(f["spread_bps_live"], 1) for _, f in snaps["AAAUSDT"]], [6.0, 4.0])


class TestShadowIndex(unittest.TestCase):
    def test_enter_and_exit_by_sym_tf_event_second(self):
        import trade_dataset as TD
        idx = TD.shadow_index(journal())
        a = idx[("AAAUSDT", "15m", EV[:19])]
        self.assertAlmostEqual(a["ret_pct"], -3.0)
        self.assertAlmostEqual(a["features"]["ml_proba"], 0.31)
        self.assertEqual(a["features"]["mode"], "trend")

    def test_open_shadow_has_no_ret(self):
        import trade_dataset as TD
        idx = TD.shadow_index([r for r in journal() if not (r.get("kind") == "exit_decision" and r["sym"] == "AAAUSDT")])
        self.assertIsNone(idx[("AAAUSDT", "15m", EV[:19])]["ret_pct"])

    def test_other_books_ignored(self):
        import trade_dataset as TD
        rows = [dict(r, book="champion") if r.get("process") == "shadow" else r for r in journal()]
        self.assertEqual(TD.shadow_index(rows), {})


class TestNightly(unittest.TestCase):
    def snaps(self):
        t = T0 + timedelta(minutes=46)
        return {"AAAUSDT": [(t - timedelta(minutes=9), {"spread_bps_live": 9.0}),
                            (t - timedelta(minutes=2), {"spread_bps_live": 5.0}),
                            (t + timedelta(minutes=1), {"spread_bps_live": 99.0})]}

    def test_rows_get_micro_and_bot_features_known_at_entry(self):
        import trade_dataset as TD
        out = TD.nightly(journal(), self.snaps())
        a = next(r for r in out["rows"] if r["decision_id"] == "d-enter-A")
        self.assertEqual(a["micro"]["spread_bps_live"], 5.0)            # latest <= entry, 120 s old
        self.assertAlmostEqual(a["micro"]["age_sec"], 120.0)
        self.assertAlmostEqual(a["bot"]["candidate_score"], 55.0)
        self.assertAlmostEqual(a["outcome"]["ret_pct"], -3.5)
        b = next(r for r in out["rows"] if r["decision_id"] == "d-skip-B")
        self.assertIsNone(b["micro"])                                   # no snapshot for this pair

    def test_counterfactual_for_skip_from_shadow(self):
        import trade_dataset as TD
        out = TD.nightly(journal(), {})
        cf = [r for r in out["rows"] if r["source"] == "replay"]
        self.assertEqual(len(cf), 1)
        self.assertEqual(cf[0]["key"], KEY2)
        self.assertAlmostEqual(cf[0]["outcome"]["ret_pct"], 4.0)

    def test_live_vs_replay_from_shadow(self):
        import trade_dataset as TD
        out = TD.nightly(journal(), {})
        lv = out["live_vs_replay"]
        self.assertEqual(lv["n"], 1)
        self.assertAlmostEqual(lv["mean_diff_pct"], -0.5)               # live -3.5 vs shadow -3.0
        self.assertEqual(lv["no_replay"], 0)

    def test_live_vs_replay_split_by_mode(self):
        """Paper (C1) and demo (C2) fills are different executions: never one number (spec §0 rule 3)."""
        import trade_dataset as TD
        j = journal()
        paper = [dict(r, mode="paper", key=r["key"].replace("AAAUSDT", "CCCUSDT"), sym="CCCUSDT",
                      decision_id=r["decision_id"] + "-p")
                 for r in j if r.get("process") == "executor" and r.get("key") == KEY]
        shadow_c = [dict(r, sym="CCCUSDT", ret_pct=1.0) for r in j if r.get("process") == "shadow" and r.get("sym") == "AAAUSDT"]
        out = TD.nightly(j + paper + shadow_c, {})
        bm = out["live_vs_replay"]["by_mode"]
        self.assertEqual(bm["demo"]["n"], 1)
        self.assertAlmostEqual(bm["demo"]["mean_diff_pct"], -0.5)
        self.assertEqual(bm["paper"]["n"], 1)
        self.assertAlmostEqual(bm["paper"]["mean_diff_pct"], -4.5)          # live -3.5 vs shadow +1.0

    def test_unmatched_live_trade_counted(self):
        import trade_dataset as TD
        rows = [r for r in journal() if r.get("process") != "shadow"]
        out = TD.nightly(rows, {})
        self.assertEqual(out["live_vs_replay"]["n"], 0)
        self.assertEqual(out["live_vs_replay"]["no_replay"], 1)


class TestBuildCli(unittest.TestCase):
    def setup_dirs(self, d):
        dec, mic, out = Path(d) / "decisions", Path(d) / "micro", Path(d) / "learning"
        dec.mkdir()
        mic.mkdir()
        (dec / "2026-10-21.jsonl").write_text("\n".join(json.dumps(r) for r in journal()) + "\n", encoding="utf-8")
        (dec / "2026-10-10.jsonl").write_text(json.dumps(dict(journal()[1], decision_id="old-day")) + "\n", encoding="utf-8")
        (mic / "2026-10-21.jsonl").write_text(micro_line(T0 + timedelta(minutes=44)) + "\n", encoding="utf-8")
        return dec, mic, out

    def test_build_writes_dataset_and_reports_and_is_idempotent(self):
        import trade_dataset as TD
        with tempfile.TemporaryDirectory() as d:
            dec, mic, out = self.setup_dirs(d)
            now = T0 + timedelta(days=1, hours=0, minutes=20)
            s1 = TD.build(days=3, now=now, decisions_dir=dec, micro_dir=mic, out_dir=out)
            self.assertEqual(s1["appended"], 3)                         # enter A (with outcome), skip B, replay B
            self.assertTrue((out / "trades.jsonl").exists())
            self.assertTrue((out / "live_vs_replay.json").exists())
            self.assertTrue((out / "BUILD_SUMMARY.json").exists())
            ids = [json.loads(x)["decision_id"] for x in (out / "trades.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("old-day", ids)                            # outside the 3-day window
            a = [json.loads(x) for x in (out / "trades.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(x)["decision_id"] == "d-enter-A"][0]
            self.assertAlmostEqual(a["micro"]["spread_bps_live"], 6.0)
            s2 = TD.build(days=3, now=now, decisions_dir=dec, micro_dir=mic, out_dir=out)
            self.assertEqual(s2["appended"], 0)

    def test_runs_as_a_script_under_the_embedded_python(self):
        """Bug 2026-10-20: the first real run failed with "No module named 'trader_config'" -- the embedded
        Python does not put the script's folder on sys.path, and the tests import the module in-process."""
        import subprocess
        import sys
        r = subprocess.run([sys.executable, str(HERE / "trade_dataset.py"), "--help"], capture_output=True, text=True,
                           cwd=str(HERE.parent))
        self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_run_daily_calls_build(self):
        src = (HERE / "run_daily.cmd").read_text(encoding="utf-8")
        self.assertIn("trade_dataset.py build", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
