"""E7 shadow portfolios -- tests written from e7-shadow-portfolio-1012-spec.md §7 BEFORE the code
(operator rule 2026-10-10: spec -> tests -> code)."""
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

import execution_replay as R
import fill_model as FM
import trading_policy as TP
from test_execution_replay import FakeGrid, T0, STEP

HERE = Path(__file__).resolve().parent
X4 = replace(TP.CHAMPION, name="x4_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def walk(n, seed=3):
    rnd = random.Random(seed)
    p, out = 100.0, []
    for _ in range(n):
        p *= 1 + rnd.gauss(0.0003, 0.006)
        out.append(p)
    return out


def write_events(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def ts_of(ix, minute=1):
    return (T0 + (ix + 1) * STEP + timedelta(minutes=minute)).isoformat()


def grid_events(g, n_trades=14, seed=5):
    """Entries/exits of two symbols, some overlapping, one 1h trade."""
    rnd = random.Random(seed)
    rows, ie, free = [], 120, {}                  # free: (sym, tf) -> first bar after its last exit
    for i in range(n_trades):
        sym = g.syms[i % len(g.syms)]
        j = g.col[sym]
        tf = "1h" if i == 4 else "15m"
        ie = max(ie, free.get((sym, tf), 0))      # the bot never holds two positions in one sym/tf
        ix = ie + rnd.randint(3, 30)
        if ix >= g.N - 100:                       # keep every trade inside the grid
            break
        free[(sym, tf)] = ix + 1
        rows.append({"event": "entry", "sym": sym, "tf": tf, "ts": ts_of(ie), "price": float(g.C[ie, j]),
                     "mode": "trend", "trail_k": 2.0})
        rows.append({"event": "exit", "sym": sym, "tf": tf, "ts": ts_of(ix, 2), "exit_price": float(g.C[ix, j]),
                     "pnl_pct": (g.C[ix, j] / g.C[ie, j] - 1) * 100, "reason": "trail"})
        ie += rnd.randint(2, 25)
    return rows


def two_sym_grid():
    c1, c2 = walk(96 * 8, 3), walk(96 * 8, 9)
    g = FakeGrid(list(zip(c1, c2)), syms=("AUSDT", "BUSDT"))
    g.ATR = np.full(g.C.shape, 0.6)
    return g


class TestIncrementalExit(unittest.TestCase):
    """§7.2."""

    def test_chunked_feed_equals_exit_plan(self):
        import shadow_portfolio as S
        g = two_sym_grid()
        for ie, tf in ((150, "15m"), (200, "1h"), (300, "15m")):
            c = {"j": 0, "ie": ie, "tf": tf, "trail_k": 2.0, "mode": "trend", "price": float(g.C[ie, 0])}
            ref = TP.exit_plan(X4.__class__(exits=X4.exits), g, c, lambda m: 0.0)
            for chunk in (1, 3, 7):
                ex = S.P1Exit(k=2.0, floor=0.08, c0=float(g.C[ie, 0]), a0=float(g.ATR[ie, 0]), ie=ie,
                              max_bars=96, hourly=(tf == "1h"))
                got, q = None, ie + 1
                while got is None and q <= g.N - 1:
                    for qq in range(q, min(q + chunk, g.N)):
                        got = ex.feed(qq, float(g.C[qq, 0]), float(g.ATR[qq, 0]))
                        if got is not None:
                            break
                    q += chunk
                self.assertEqual(got[:3], ref[:3], (ie, tf, chunk))


class TestEventTail(unittest.TestCase):
    """§7.3."""

    def test_tail_semantics(self):
        import shadow_portfolio as S
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            rows = [{"event": "entry", "sym": "A", "tf": "15m", "ts": ts_of(i), "price": 1} for i in range(3)]
            write_events(p, rows)
            st = {}
            t = S.EventTail(p, st)
            self.assertEqual(len(t.read_new()), 3)
            self.assertEqual(t.read_new(), [])                         # nothing new -> nothing
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rows[0] | {"ts": ts_of(9)})[:-3])  # partial line, no newline
            self.assertEqual(t.read_new(), [])
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rows[0] | {"ts": ts_of(9)})[-3:] + "\n")
            self.assertEqual(len(t.read_new()), 1)
            write_events(p, rows[:2])                                  # file shrank (rotation)
            self.assertEqual(t.read_new(), [])                         # no duplicates of seen events


class TestParity(unittest.TestCase):
    """§7.1 / spec §5: same events + same bars -> the same trades and daily series as the replay."""

    def _replay(self, pol, g, path):
        trades, _ = R.load_events(since="2026-03-01", path=path)
        R.attach_grid(trades, g)
        return R.run(pol, FM.model("taker", 1), g, trades, floor_of=lambda m: 0.0)

    def test_parity_covers_trail_exits_and_many_days(self):
        """Strengthened 2026-10-12: the first parity case compared 2-3 days and only time exits."""
        import shadow_portfolio as S
        g = two_sym_grid()
        tight = replace(TP.CHAMPION, name="p1_tight", exits=TP.Exits(kind="p1_trail", floor_override=0.02))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            write_events(p, grid_events(g, n_trades=60, seed=11))
            for name, pol in (("champion", TP.CHAMPION), ("p1_tight", tight), ("x4_floor8", X4)):
                ref = self._replay(pol, g, p)
                eng = S.ShadowEngine({name: pol}, FM.model("taker", 1), floor_of=lambda m: 0.0, bars=S.GridBars(g))
                eng.process(S.EventTail(p, {}).read_new(), now=T0 + (g.N - 1) * STEP)
                mine = sorted(eng.trades(name), key=lambda t: t["entry_ts"])
                theirs = sorted(ref["trades"], key=lambda t: t["entry_ts"])
                self.assertEqual(len(mine), len(theirs), name)
                for a_, b_ in zip(mine, theirs):
                    self.assertEqual((a_["sym"], a_["entry_ts"], a_["reason"]), (b_["sym"], b_["entry_ts"], b_["reason"]))
                    self.assertAlmostEqual(a_["ret_pct"], b_["ret_pct"], places=9)
                ms, rs = dict(eng.series(name)), dict(ref["series"])
                common = sorted(set(ms) & set(rs))
                self.assertGreaterEqual(len(common), 5, name)
                for day in common:
                    self.assertAlmostEqual(ms[day], rs[day], places=9, msg=(name, day))
                if name == "p1_tight":
                    self.assertIn("trail", {t["reason"] for t in mine})

    def test_parity_champion_and_x4(self):
        import shadow_portfolio as S
        g = two_sym_grid()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.jsonl"
            write_events(p, grid_events(g))
            ref = {"champion": self._replay(TP.CHAMPION, g, p), "x4_floor8": self._replay(X4, g, p)}
            eng = S.ShadowEngine({"champion": TP.CHAMPION, "x4_floor8": X4}, FM.model("taker", 1),
                                 floor_of=lambda m: 0.0, bars=S.GridBars(g))
            tail = S.EventTail(p, {})
            eng.process(tail.read_new(), now=T0 + (g.N - 1) * STEP)
            for name, r in ref.items():
                mine = eng.trades(name)
                self.assertEqual(len(mine), len(r["trades"]), name)
                for a, b in zip(sorted(mine, key=lambda t: t["entry_ts"]), sorted(r["trades"], key=lambda t: t["entry_ts"])):
                    self.assertEqual((a["sym"], a["entry_ts"], a["reason"]), (b["sym"], b["entry_ts"], b["reason"]), name)
                    self.assertAlmostEqual(a["ret_pct"], b["ret_pct"], places=9)
                ms, rs = dict(eng.series(name)), dict(r["series"])
                common = sorted(set(ms) & set(rs))
                self.assertTrue(common, name)
                for day in common:
                    self.assertAlmostEqual(ms[day], rs[day], places=9, msg=(name, day))


class TestExitMatching(unittest.TestCase):
    """§7.4."""

    def test_orphan_exit_logged(self):
        import shadow_portfolio as S
        g = two_sym_grid()
        eng = S.ShadowEngine({"champion": TP.CHAMPION}, FM.model("taker", 1), floor_of=lambda m: 0.0, bars=S.GridBars(g))
        ev = [{"event": "exit", "sym": "AUSDT", "tf": "15m", "ts": T0 + 200 * STEP, "exit_price": 100.0, "pnl_pct": 1.0}]
        eng.process(ev, now=T0 + 210 * STEP)
        self.assertTrue(any(x["type"] == "orphan_exit" for x in eng.ledger))
        self.assertEqual(eng.trades("champion"), [])


class TestState(unittest.TestCase):
    """§7.5."""

    def test_atomic_save_and_corrupt_raises(self):
        import shadow_portfolio as S
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.json"
            S.save_state(p, {"a": 1})
            self.assertEqual(S.load_state(p), {"a": 1})
            self.assertFalse(p.with_suffix(".tmp").exists())
            p.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                S.load_state(p)
            self.assertIsNone(S.load_state(Path(d) / "missing.json"))


class TestDayMarkAndClosedBars(unittest.TestCase):
    """§7.6–7.7."""

    def test_one_mark_per_day(self):
        import shadow_portfolio as S
        g = two_sym_grid()
        eng = S.ShadowEngine({"champion": TP.CHAMPION}, FM.model("taker", 1), floor_of=lambda m: 0.0, bars=S.GridBars(g))
        eng.process([], now=T0 + 96 * 3 * STEP)
        eng.process([], now=T0 + (96 * 3 + 5) * STEP)
        days = [d for d, _ in eng.marks("champion")]
        self.assertEqual(len(days), len(set(days)))

    def test_closed_bars_only(self):
        import shadow_portfolio as S
        bars = [{"close_time": T0 + i * STEP} for i in range(1, 6)]
        self.assertEqual(len(S.closed_bars(bars, now=T0 + 3 * STEP + timedelta(seconds=1))), 3)


class TestReadout(unittest.TestCase):
    """§7.8."""

    def test_verdicts(self):
        import shadow_readout as SR
        rnd = random.Random(2)
        days = [f"2026-11-{d:02d}" for d in range(1, 29)] + [f"2026-12-{d:02d}" for d in range(1, 11)]
        p0 = [(d, rnd.gauss(-1.0, 0.5)) for d in days]
        better = [(d, v + 0.8 + rnd.gauss(0, 0.2)) for d, v in p0]
        worse = [(d, v - 0.8 + rnd.gauss(0, 0.2)) for d, v in p0]
        self.assertEqual(SR.verdict(p0[:10], better[:10], [0.1] * 200)["status"], "INCONCLUSIVE")   # < 28 days
        self.assertEqual(SR.verdict(p0, better, [-0.3] * 200)["status"], "KEEP_X4")
        self.assertEqual(SR.verdict(p0, worse, [-0.3] * 200)["status"], "REJECT_X4")
        prof = SR.verdict(p0, [(d, abs(v) * 0 + 0.5 + rnd.gauss(0, 0.1)) for d, v in p0], [0.4] * 300)
        self.assertTrue(prof["x4_profitable"])
        self.assertEqual(SR.verdict(p0, better, [-0.3] * 100)["status"], "INCONCLUSIVE")           # < 150 trades


class TestBoundary(unittest.TestCase):
    """§7.9."""

    def test_no_signed_requests_and_runtime_only(self):
        for name in ("shadow_portfolio.py", "shadow_readout.py"):
            src = (HERE / name).read_text(encoding="utf-8")
            for bad in ("account(", "test_order(", "place_order(", "place_oco(", '"' + '.runtime' + '"'):
                self.assertNotIn(bad, src, (name, bad))
        src = (HERE / "shadow_portfolio.py").read_text(encoding="utf-8")
        self.assertIn('open(self.path, "rb")', src)
        import trader_config as TC
        self.assertTrue(hasattr(TC, "SHADOW_PORTFOLIO_ENABLED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestLiveBars(unittest.TestCase):
    """Live kline provider: closed bars only, stable global index, ATR from the bot's own function."""

    def test_parse_klines(self):
        import shadow_portfolio as S
        from datetime import datetime, timezone
        base = datetime(2026, 10, 12, tzinfo=timezone.utc)
        ms = lambda d: int(d.timestamp() * 1000)  # noqa: E731
        rows = []
        for i in range(40):
            o = base + i * STEP
            rows.append([ms(o), "100", str(101 + i % 3), "99", str(100 + i * 0.1), "10", ms(o + STEP) - 1])

        class FakeClient:
            def klines(self, symbol, interval="15m", limit=500, base=None):
                return rows

        lb = S.LiveBars(FakeClient(), atr_period=14)
        now = base + 30 * STEP + timedelta(seconds=5)          # bars 0..29 closed, 30.. open
        lb.refresh(["AUSDT"], now)
        ix_last = lb.idx(base + 30 * STEP) - 1
        self.assertTrue(math.isfinite(lb.close("AUSDT", ix_last)))
        self.assertTrue(math.isnan(lb.close("AUSDT", ix_last + 1)))      # open bar not used
        self.assertEqual(lb.close_time(ix_last), base + 30 * STEP)
        self.assertTrue(math.isfinite(lb.atr("AUSDT", ix_last)))


class TestReadoutWindow(unittest.TestCase):
    """Spec §6: the window starts at the first FULL UTC day after the start (the start day is partial)."""

    def test_partial_start_day_excluded(self):
        import shadow_readout as SR
        marks = {"2026-10-12": 10000.0, "2026-10-13": 10100.0, "2026-10-14": 10050.0, "2026-10-15": 10080.0}
        s = SR.series_from_marks(marks, started_utc="2026-10-12T15:54:02+00:00")
        self.assertEqual([d for d, _ in s], ["2026-10-13", "2026-10-14"])
        s0 = SR.series_from_marks(marks, started_utc="2026-10-12T00:00:00+00:00")
        self.assertEqual(s0[0][0], "2026-10-12")
