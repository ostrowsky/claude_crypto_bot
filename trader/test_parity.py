"""Demo vs paper parity (demo-paper-parity-1029-spec.md §6). Written BEFORE parity.py. Synthetic journal rows.

Interface the tests fix:
  pairs(decisions) -> [{key, state ('full' | 'open' | 'no_paper'), d_pp, gap_usdt, entry_pp, fee_pp, rest_pp, exit_day, ...}]
  coverage(decisions, day) -> {paper_entries, entered, skips: {reason: n}, no_real_position, exchange_stops, open, no_paper}
  windows(pairs, day) -> {7: agg, 30: agg, 'all': agg}, each with 'verdict'
  block(decisions, day) -> list of report lines
"""
import unittest


def ent(key, ts, slip=8.0, cost=20.0, fee=0.015, est=False, mode="demo", book="exec_x4"):
    return {"kind": "real_entry", "book": book, "mode": mode, "action": "enter", "key": key, "sym": key.split("|")[0],
            "ts": ts, "cost": cost, "slippage_bps": slip, "latency_sec": 30.0, "fee_usdt": fee, "fee_estimated": est}


def ex(key, ts, ret, paper, pnl=None, reason="time", mode="demo"):
    r = {"kind": "real_exit", "book": "exec_x4", "mode": mode, "action": "exit", "key": key, "sym": key.split("|")[0],
         "ts": ts, "reason": reason, "ret_pct": ret, "pnl": pnl if pnl is not None else ret / 100 * 20.0}
    if paper is not None:
        r["paper_ret_pct"] = paper
    return r


def skip_exit(key, ts, reason, paper=None):
    r = {"kind": "real_exit", "book": "exec_x4", "mode": "demo", "action": "skip", "key": key, "sym": key.split("|")[0],
         "ts": ts, "reason": reason}
    if paper is not None:
        r["paper_ret_pct"] = paper
    return r


class TestPairs(unittest.TestCase):
    def test_states_and_money(self):
        import parity as P
        A, B, C, D = "AUSDT|15m|1", "BUSDT|15m|2", "CUSDT|15m|3", "DUSDT|15m|4"
        dec = [ent(A, "2026-10-20T01:00:00+00:00"), ex(A, "2026-10-20T05:00:00+00:00", 1.5, 1.0),
               ent(B, "2026-10-20T02:00:00+00:00"), ex(B, "2026-10-20T03:00:00+00:00", -3.0, None, reason="exchange_stop"),
               skip_exit(B, "2026-10-20T09:00:00+00:00", "already_closed", paper=-1.0),
               ent(C, "2026-10-20T04:00:00+00:00"),
               ent(D, "2026-10-20T06:00:00+00:00"), ex(D, "2026-10-20T07:00:00+00:00", 0.5, None),
               ent("EUSDT|15m|5", "2026-10-20T06:00:00+00:00", mode="paper"),
               ex("EUSDT|15m|5", "2026-10-20T07:00:00+00:00", 9.0, 1.0, mode="paper"),
               ent("FUSDT|15m|6", "2026-10-20T06:00:00+00:00", book="x4_floor8")]
        ps = {p["key"]: p for p in P.pairs(dec)}
        self.assertEqual(set(ps), {A, B, C, D})
        self.assertEqual(ps[A]["state"], "full")
        self.assertAlmostEqual(ps[A]["d_pp"], 0.5)
        self.assertAlmostEqual(ps[A]["gap_usdt"], 0.3 - 0.2)
        self.assertEqual(ps[A]["exit_day"], "2026-10-20")
        self.assertEqual(ps[B]["state"], "full")
        self.assertAlmostEqual(ps[B]["d_pp"], -2.0)
        self.assertEqual(ps[C]["state"], "open")
        self.assertEqual(ps[D]["state"], "no_paper")

    def test_decomposition(self):
        import parity as P
        A, B = "AUSDT|15m|1", "BUSDT|15m|2"
        dec = [ent(A, "2026-10-20T01:00:00+00:00", slip=8.0, fee=0.015), ex(A, "2026-10-20T05:00:00+00:00", 1.0, 0.5),
               ent(B, "2026-10-20T01:00:00+00:00", slip=28.0, fee=0.0), ex(B, "2026-10-20T05:00:00+00:00", 1.0, 0.5)]
        ps = {p["key"]: p for p in P.pairs(dec)}
        self.assertAlmostEqual(ps[A]["entry_pp"], 0.0)
        self.assertAlmostEqual(ps[A]["fee_pp"], 0.05)
        self.assertAlmostEqual(ps[A]["rest_pp"], 0.45)
        self.assertAlmostEqual(ps[B]["entry_pp"], -0.20)
        self.assertAlmostEqual(ps[B]["fee_pp"], 0.0)
        self.assertAlmostEqual(ps[B]["rest_pp"], 0.70)


class TestCoverageWindowsBlock(unittest.TestCase):
    def test_coverage(self):
        import parity as P
        day = "2026-10-20"
        dec = [ent("AUSDT|15m|1", day + "T01:00:00+00:00"),
               {"kind": "real_entry", "book": "exec_x4", "mode": "demo", "action": "skip", "reason": "asset_held",
                "key": "AUSDT|1h|9", "ts": day + "T01:05:00+00:00"},
               skip_exit("ZUSDT|15m|0", day + "T02:00:00+00:00", "no_real_position"),
               {"kind": "execution_divergence", "book": "exec_x4", "mode": "demo", "key": "AUSDT|15m|1",
                "ts": day + "T03:00:00+00:00", "reason": "exchange_stop"}]
        c = P.coverage(dec, day)
        self.assertEqual(c["paper_entries"], 2)
        self.assertEqual(c["entered"], 1)
        self.assertEqual(c["skips"], {"asset_held": 1})
        self.assertEqual(c["no_real_position"], 1)
        self.assertEqual(c["exchange_stops"], 1)

    def test_windows_too_early(self):
        import parity as P
        dec = []
        for i in range(5):
            k = f"S{i}USDT|15m|{i}"
            dec += [ent(k, f"2026-10-2{i}T01:00:00+00:00"), ex(k, f"2026-10-2{i}T05:00:00+00:00", 1.0, 0.0)]
        w = P.windows(P.pairs(dec), "2026-10-24")
        self.assertEqual(w["all"]["n_days"], 5)
        self.assertAlmostEqual(w["all"]["mean_pct"], 1.0)
        self.assertIn("рано судить", w["all"]["verdict"])

    def test_block(self):
        import parity as P
        day = "2026-10-20"
        dec = [ent("AUSDT|15m|1", day + "T01:00:00+00:00"), ex("AUSDT|15m|1", day + "T05:00:00+00:00", 3.0, 1.0)]
        L = P.block(dec, day)
        self.assertTrue(L[0].startswith("6. Демо против бумаги"))
        txt = "\n".join(L)
        self.assertIn("AUSDT", txt)                         # |d| = 2 > 1 pp: an outlier line
        self.assertIn("рано судить", txt)
        self.assertIn("нет демо-сделок", "\n".join(P.block([], day)))

    def test_report_section(self):
        import trader_report as TR
        txt = TR.build("2026-10-20", {}, parity=["6. Демо против бумаги (x)", "  нет демо-сделок"])
        self.assertIn("6. Демо против бумаги", txt)
        self.assertNotIn("6. Демо против бумаги", TR.build("2026-10-20", {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
