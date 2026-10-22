"""Live vs replay on C2 -- reproducing tests BEFORE the fix (operator rule: a bug fix starts with a reproducing test).

Bug 2026-10-21: the F1 live-vs-replay check matched 1 of 10 closed demo trades. Its replay was the E7 shadow book
x4_floor8, whose 10 slots were full since 10-12, while the executor's own paper book started empty at the C2 switch
(10-20): 12 of 18 demo entries were 'slots_full' in the shadow. The right per-trade replay is the executor's OWN
paper book (same engine, same entry, no foreign slots) -- but its exit result was never written to the journal.
Fix: the paper exit result travels with the exit action and is logged as paper_ret_pct on the executor's exit row
(or on the 'already_closed' skip row when the exchange stop closed the trade first); F1 prefers it to the shadow.
"""
import types
import unittest
from datetime import datetime, timedelta, timezone

from test_executor import NOW, closed, make

KEY = "ABCUSDT|15m|2026-10-14T00:45:00+00:00"


def exit_action(key, reason="trail", paper_ret=-2.0):
    sym, tf, ts = key.split("|")
    return {"type": "exit", "key": key, "sym": sym, "tf": tf, "ts": ts, "reason": reason, "paper_ret_pct": paper_ret}


class TestExecutorLogsPaperResult(unittest.TestCase):
    def test_policy_exit_row_carries_paper_ret(self):
        E, paper, ex, w, exe = make()
        key = paper.enter()
        exe.cycle([], NOW)
        paper.queue.append(exit_action(key, paper_ret=-2.0))
        exe.cycle([], NOW + timedelta(minutes=20))
        self.assertIsNotNone(closed(exe, key))
        row = [r for r in w.kind("real_exit") if r.get("key") == key and r.get("action") == "exit"][-1]
        self.assertAlmostEqual(row["paper_ret_pct"], -2.0)

    def test_already_closed_skip_carries_paper_ret(self):
        """The exchange stop closed it first; the paper exit that comes later still reports its result."""
        E, paper, ex, w, exe = make()
        key = paper.enter()
        exe.cycle([], NOW)
        p = exe.positions[key]
        exe._closed_by_exchange(p, {"cummulativeQuoteQty": "18.0", "executedQty": "9.99", "fills": []}, NOW)
        paper.queue.append(exit_action(key, paper_ret=-7.5))
        exe.cycle([], NOW + timedelta(minutes=20))
        row = [r for r in w.kind("real_exit") if r.get("key") == key and r.get("action") == "skip"][-1]
        self.assertEqual(row["reason"], "already_closed")
        self.assertAlmostEqual(row["paper_ret_pct"], -7.5)


class TestPaperBookForwardsResult(unittest.TestCase):
    def test_exit_action_has_paper_ret(self):
        import executor as E
        pb = E.PaperBook.__new__(E.PaperBook)
        ledger = [{"type": "exit", "sym": "ABCUSDT", "tf": "15m", "entry_ts": "2026-10-14T00:45:00+00:00",
                   "exit_time": "2026-10-14T05:00:00+00:00", "reason": "trail", "ret_pct": -3.25}]
        pb.engine = types.SimpleNamespace(process=lambda events, now: None, ledger=ledger,
                                          bars=types.SimpleNamespace(last_closed=lambda t: None))
        pb._n, pb._last_bar, pb.name = 0, None, "x4_floor8"
        acts = pb.process([], NOW)
        self.assertEqual(len(acts), 1)
        self.assertAlmostEqual(acts[0]["paper_ret_pct"], -3.25)


class TestDatasetPrefersOwnPaper(unittest.TestCase):
    def rows(self, skip_closed=False):
        e = lambda **kw: {"schema": "audit.decision.v1", "process": "executor", "book": "exec_x4", "mode": "demo", **kw}  # noqa: E731
        t = datetime(2026, 10, 21, 1, tzinfo=timezone.utc)
        out = [e(kind="real_entry", action="enter", key=KEY, sym="ABCUSDT", ts=t.isoformat(), decision_id="d1",
                 fill_price=2.0, paper_price=2.0)]
        if skip_closed:
            out += [e(kind="real_exit", action="exit", key=KEY, sym="ABCUSDT", ts=(t + timedelta(hours=1)).isoformat(),
                      reason="exchange_stop", pnl=-2.0, ret_pct=-10.0),
                    e(kind="real_exit", action="skip", key=KEY, sym="ABCUSDT", ts=(t + timedelta(hours=2)).isoformat(),
                      reason="already_closed", paper_ret_pct=-19.0)]
        else:
            out += [e(kind="real_exit", action="exit", key=KEY, sym="ABCUSDT", ts=(t + timedelta(hours=3)).isoformat(),
                      reason="trail", pnl=-0.6, ret_pct=-3.0, paper_ret_pct=-2.5)]
        return out

    def test_own_paper_is_the_replay_without_any_shadow_row(self):
        import trade_dataset as TD
        lv = TD.nightly(self.rows(), {})["live_vs_replay"]
        self.assertEqual(lv["n"], 1)
        self.assertAlmostEqual(lv["mean_diff_pct"], -0.5)            # live -3.0 vs own paper -2.5
        self.assertEqual(lv["rows"][0]["replay_source"], "own_paper")

    def test_paper_result_from_the_already_closed_row(self):
        import trade_dataset as TD
        lv = TD.nightly(self.rows(skip_closed=True), {})["live_vs_replay"]
        self.assertEqual(lv["n"], 1)
        self.assertAlmostEqual(lv["mean_diff_pct"], 9.0)             # the exchange stop saved 9 pp vs the paper trail

    def test_shadow_still_the_fallback(self):
        import trade_dataset as TD
        rows = [dict(r) for r in self.rows()]
        rows[-1].pop("paper_ret_pct")
        rows.append({"book": "x4_floor8", "kind": "exit_decision", "sym": "ABCUSDT", "tf": "15m",
                     "event_ts": "2026-10-14T00:45:00+00:00", "ret_pct": -1.0, "reason": "trail"})
        lv = TD.nightly(rows, {})["live_vs_replay"]
        self.assertEqual(lv["n"], 1)
        self.assertEqual(lv["rows"][0]["replay_source"], "shadow")
        self.assertAlmostEqual(lv["mean_diff_pct"], -2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
