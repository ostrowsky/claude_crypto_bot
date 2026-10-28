"""W1 -- shadow scanner of the whole Binance spot market (wide-scan-shadow-w1-1028-spec.md §6).

Written BEFORE wide_scan.py (operator rule 2026-10-10: spec -> tests -> code). Synthetic data, offline: no request
leaves the machine, the exchange is a fake `fetch(sym, start_ms) -> kline rows`.

Interface the tests fix (spec §2):
  BarStore(root)                    .load(sym) / .last_t(sym) / .append(sym, rows, now) -> added / .seed(sym, arr, n)
  ret_since_open(arr, i)            close[i] / open of the 00:00 UTC bar of that day - 1, None without that bar
  detect_leader(rets, flagged, top, min_ret) -> newly flagged syms
  detect_surge(arr, i)              quote volume >= SURGE_MULT x median of the previous SURGE_LOOKBACK bars and
                                    bar return >= SURGE_MIN_RET
  run_step(now, fetch, universe, root, gen_fn=None, enabled=True, watchlist=()) -> summary
  book_events(events)               only wide_gen entries feed the x4_wide book
  candidate_row(event, arr, i, rank, n_universe, in_watchlist) -> learning row (features known at bar i)
  label_outcomes(root, now) -> outcome rows appended (once per candidate, after 24 h)
  leaders_of_day(store, syms, day)  top-20 by close/open of the UTC day, None when the day has too little data
  early_seen(leaders, events, store, day) -> share flagged before +2.5% WITH the base rate beside it
"""
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR_MS = 15 * 60 * 1000
DAY0 = datetime(2026, 10, 20, tzinfo=timezone.utc)
D0_MS = int(DAY0.timestamp() * 1000)


def rows_for(closes, start_ms=D0_MS, vol=None, opens=None):
    """Binance kline rows [open_ms, o, h, l, c, v, close_ms, ...]."""
    out = []
    for i, c in enumerate(closes):
        o = opens[i] if opens else (closes[i - 1] if i else c)
        v = (vol or {}).get(i, 10.0)
        t = start_ms + i * BAR_MS
        out.append([t, str(o), str(max(o, c) * 1.001), str(min(o, c) * 0.999), str(c), str(v), t + BAR_MS - 1,
                    "0", 0, "0", "0", "0"])
    return out


def arr_for(closes, start_ms=D0_MS, vol=None, opens=None):
    r = rows_for(closes, start_ms, vol, opens)
    return {"t": np.array([x[0] for x in r], dtype=np.int64), "o": np.array([float(x[1]) for x in r]),
            "h": np.array([float(x[2]) for x in r]), "l": np.array([float(x[3]) for x in r]),
            "c": np.array([float(x[4]) for x in r]), "v": np.array([float(x[5]) for x in r])}


def at(i_bar, sec=5):
    """A moment just after bar i_bar (0-based from DAY0) closed."""
    return DAY0 + timedelta(milliseconds=(i_bar + 1) * BAR_MS, seconds=sec)


class FakeExchange:
    def __init__(self, series: dict):
        self.series = series                       # sym -> full rows
        self.calls = []

    def __call__(self, sym, start_ms):
        self.calls.append((sym, start_ms))
        if sym == "BADUSDT":
            raise RuntimeError("boom")
        return [r for r in self.series.get(sym, []) if r[0] >= start_ms][:1000]


# --------------------------------------------------------------------------- 1. bar store
class TestBarStore(unittest.TestCase):
    def test_append_no_duplicates_closed_only_gaps_kept(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            st = W.BarStore(Path(d))
            rows = rows_for([100 + i for i in range(10)])
            self.assertEqual(st.append("AUSDT", rows, at(7)), 8)            # bars 0..7 closed, 8 and 9 still open
            self.assertEqual(st.append("AUSDT", rows, at(7)), 0)            # same rows again: nothing new
            self.assertEqual(st.append("AUSDT", rows, at(9)), 2)
            a = st.load("AUSDT")
            self.assertEqual(len(a["t"]), 10)
            self.assertTrue(np.all(np.diff(a["t"]) == BAR_MS))
            gap = rows_for([200, 201], start_ms=D0_MS + 15 * BAR_MS)         # bars 10..14 missing on the exchange
            st.append("AUSDT", gap, at(16))
            a = st.load("AUSDT")
            self.assertEqual(len(a["t"]), 12)                                # the gap is not filled
            self.assertEqual(int(a["t"][-1]), D0_MS + 16 * BAR_MS)
            self.assertEqual(st.last_t("AUSDT"), D0_MS + 16 * BAR_MS)
            self.assertIsNone(st.last_t("NOPEUSDT"))

    def test_seed_takes_exactly_the_last_n(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            st = W.BarStore(Path(d))
            src = arr_for([float(i + 1) for i in range(50)])
            self.assertEqual(st.seed("AUSDT", src, 20), 20)
            a = st.load("AUSDT")
            self.assertEqual(len(a["t"]), 20)
            self.assertEqual(int(a["t"][0]), int(src["t"][30]))
            self.assertEqual(st.seed("AUSDT", src, 20), 0)                   # never re-seeds a non-empty store


# --------------------------------------------------------------------------- 4-5. detectors
class TestDetectors(unittest.TestCase):
    def test_ret_since_open(self):
        import wide_scan as W
        a = arr_for([100.0] * 5 + [110.0], opens=[100.0] * 6)
        self.assertAlmostEqual(W.ret_since_open(a, 5), 0.10)
        late = arr_for([100.0, 110.0], start_ms=D0_MS + 4 * BAR_MS, opens=[100.0, 100.0])
        self.assertIsNone(W.ret_since_open(late, 1))                         # the 00:00 bar of the day is missing

    def test_leader_top3_min5pct_once_a_day(self):
        import wide_scan as W
        rets = {"A": 0.20, "B": 0.10, "C": 0.06, "D": 0.04, "E": 0.03}
        self.assertEqual(W.detect_leader(rets, set(), top=3, min_ret=0.05), ["A", "B", "C"])
        self.assertEqual(W.detect_leader(rets, {"A", "B"}, top=3, min_ret=0.05), ["C"])
        rets2 = {"A": 0.20, "B": 0.10, "D": 0.04}
        self.assertEqual(W.detect_leader(rets2, set(), top=3, min_ret=0.05), ["A", "B"])   # D in top-3 but < +5%

    def test_surge(self):
        import wide_scan as W
        n = W.SURGE_LOOKBACK + 1
        closes = [100.0] * (n - 1) + [103.0]
        a = arr_for(closes, vol={n - 1: 10.0 * 6})
        self.assertTrue(W.detect_surge(a, n - 1))
        b = arr_for(closes, vol={n - 1: 10.0 * 3})                           # only 3x the volume
        self.assertFalse(W.detect_surge(b, n - 1))
        c = arr_for([100.0] * (n - 1) + [101.0], vol={n - 1: 60.0})           # +1% only
        self.assertFalse(W.detect_surge(c, n - 1))
        self.assertFalse(W.detect_surge(a, n - 5))                           # no future bar is read: earlier bar


# --------------------------------------------------------------------------- 2-3. step
def universe_series(n_bars=120):
    rnd = np.random.default_rng(1)
    out = {}
    for k, s in enumerate(("AUSDT", "BUSDT", "CUSDT", "DUSDT")):
        c = 100 * np.exp(np.cumsum(0.004 * rnd.standard_normal(n_bars)))
        if s == "AUSDT":
            c[100:] = c[99] * np.linspace(1.02, 1.30, n_bars - 100)          # a leader during the day
        out[s] = rows_for(list(c))
    return out


class TestStep(unittest.TestCase):
    def test_idempotent_in_the_same_bar_and_skips_pairs_without_new_bars(self):
        import wide_scan as W
        ex = FakeExchange(universe_series())
        calls = []

        def gen(sym, tf, arr, now=None, start_ms=None):
            calls.append((sym, tf, start_ms))
            return []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            s1 = W.run_step(at(110), ex, list(ex.series), root, gen_fn=gen)
            self.assertEqual(s1["evaluated"], 4)
            n_calls = len(calls)
            s2 = W.run_step(at(110, sec=200), ex, list(ex.series), root, gen_fn=gen)   # same bar, later
            self.assertTrue(s2["same_bar"])
            self.assertEqual(len(calls), n_calls)
            self.assertEqual(s2["evaluated"], 0)
            ev_lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if (root / "events.jsonl").exists() else []
            s3 = W.run_step(at(111), ex, list(ex.series), root, gen_fn=gen)
            self.assertEqual(s3["evaluated"], 4)
            ev2 = (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if (root / "events.jsonl").exists() else []
            self.assertGreaterEqual(len(ev2), len(ev_lines))

    def test_one_pair_error_does_not_stop_the_others(self):
        import wide_scan as W
        series = universe_series()
        ex = FakeExchange(series)
        with tempfile.TemporaryDirectory() as d:
            s = W.run_step(at(110), ex, list(series) + ["BADUSDT"], Path(d), gen_fn=lambda *a, **k: [])
            self.assertEqual(s["evaluated"], 4)
            self.assertIn("BADUSDT", s["errors"])

    def test_wide_gen_events_get_the_source_and_only_new_closed_bars(self):
        import wide_scan as W
        series = universe_series()
        ex = FakeExchange(series)
        seen = []

        def gen(sym, tf, arr, now=None, start_ms=None):
            seen.append((sym, tf, start_ms, int(arr["t"][-1])))
            if sym == "BUSDT" and tf == "15m":
                t_last = int(arr["t"][-1])
                return [{"event": "entry", "sym": sym, "tf": tf,
                         "ts": datetime.fromtimestamp((t_last + BAR_MS) / 1000, timezone.utc).isoformat(),
                         "price": float(arr["c"][-1]), "mode": "trend", "trail_k": 2.0, "source": "wide_gen"}]
            return []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            W.run_step(at(110), ex, list(series), root, gen_fn=gen)
            for sym, tf, start_ms, t_last in seen:
                self.assertLessEqual(t_last + (BAR_MS if tf == "15m" else 4 * BAR_MS), int(at(110).timestamp() * 1000))
            seen.clear()
            W.run_step(at(111), ex, list(series), root, gen_fn=gen)
            s15 = [x for x in seen if x[1] == "15m"]
            self.assertTrue(all(x[2] is not None and x[2] > D0_MS + 110 * BAR_MS for x in s15))  # only bars after the last step
            evs = [json.loads(x) for x in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            wg = [e for e in evs if e["source"] == "wide_gen"]
            self.assertTrue(wg)
            self.assertTrue(all(e["sym"] == "BUSDT" for e in wg))

    def test_1h_generated_only_when_a_new_hour_closed(self):
        """Speed (journal 10-28): the 1h generator runs only on steps where a new complete hour exists."""
        import wide_scan as W
        series = universe_series()
        ex = FakeExchange(series)
        tfs = []

        def gen(sym, tf, arr, now=None, start_ms=None):
            tfs.append((tf, int(arr["t"][-1])))
            return []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            W.run_step(at(103), ex, ["AUSDT"], root, gen_fn=gen)      # bars 0..103 closed: hour 100..103 complete
            tfs.clear()
            W.run_step(at(104), ex, ["AUSDT"], root, gen_fn=gen)      # bar 104 opens a new hour: no new complete hour
            self.assertEqual([t for t, _ in tfs], ["15m"])
            for i in (105, 106):
                W.run_step(at(i), ex, ["AUSDT"], root, gen_fn=gen)
            tfs.clear()
            W.run_step(at(107), ex, ["AUSDT"], root, gen_fn=gen)      # bars 104..107 complete the hour
            self.assertEqual(sorted(t for t, _ in tfs), ["15m", "1h"])

    def test_window_is_bounded(self):
        import wide_scan as W
        self.assertLessEqual(W.GEN_BARS_15M, 300)
        self.assertGreaterEqual(W.GEN_BARS_15M, 4 * 50)              # >= 4x the slowest EMA of the bot (50)

    def test_process_pool_equals_serial_with_the_real_generator(self):
        """Speed (journal 10-28): the default generator runs in a process pool; its events equal a serial run."""
        import candidate_gen as CG
        import wide_scan as W
        rnd = np.random.default_rng(7)
        series = {}
        for k in range(6):
            c = 100 * np.exp(np.cumsum(0.006 * rnd.standard_normal(400) + 0.0008))
            series[f"P{k}USDT"] = rows_for(list(c), vol={i: 10.0 + 40.0 * (i % 37 == 0) for i in range(400)})
        now = at(399)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            W.run_step(at(300), FakeExchange(series), list(series), Path(d1), gen_fn=CG.generate)
            W.run_step(at(300), FakeExchange(series), list(series), Path(d2), workers=3)
            for i in range(301, 400, 7):
                W.run_step(at(i), FakeExchange(series), list(series), Path(d1), gen_fn=CG.generate)
                W.run_step(at(i), FakeExchange(series), list(series), Path(d2), workers=3)
            read = lambda d: [json.loads(x) for x in (Path(d) / "events.jsonl").read_text(encoding="utf-8").splitlines()]                 if (Path(d) / "events.jsonl").exists() else []  # noqa: E731
            self.assertTrue([e for e in read(d1) if e["source"] == "wide_gen"])   # not an empty == empty
            self.assertEqual(read(d1), read(d2))
        _ = now

    def test_store_is_uncompressed_and_loads_back(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            st = W.BarStore(Path(d))
            st.append("AUSDT", rows_for([1.0, 2.0, 3.0]), at(2))
            with np.load(st.path("AUSDT")) as z:
                self.assertEqual(sorted(z.files), sorted(W.FIELDS))
            import zipfile
            with zipfile.ZipFile(st.path("AUSDT")) as zf:
                self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist()))
            self.assertEqual(list(W.BarStore(Path(d)).load("AUSDT")["c"]), [1.0, 2.0, 3.0])

    def test_default_generator_is_the_b2_generator(self):
        import candidate_gen as CG
        import wide_scan as W
        self.assertIs(W.default_gen(), CG.generate)

    def test_leader_event_written_once(self):
        import wide_scan as W
        series = universe_series()
        ex = FakeExchange(series)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(100, 115):
                W.run_step(at(i), ex, list(series), root, gen_fn=lambda *a, **k: [])
            evs = [json.loads(x) for x in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            lead = [e for e in evs if e["source"] == "leader"]
            self.assertEqual([e["sym"] for e in lead], ["AUSDT"])

    def test_disabled_does_nothing(self):
        import wide_scan as W
        ex = FakeExchange(universe_series())
        with tempfile.TemporaryDirectory() as d:
            s = W.run_step(at(110), ex, list(ex.series), Path(d), gen_fn=lambda *a, **k: [], enabled=False)
            self.assertEqual(ex.calls, [])
            self.assertEqual(list(Path(d).iterdir()), [])
            self.assertFalse(s["enabled"])


# --------------------------------------------------------------------------- 6. book
class TestBook(unittest.TestCase):
    def test_only_wide_gen_entries_and_x4_policy(self):
        import shadow_portfolio as SP
        import wide_scan as W
        evs = [{"event": "entry", "source": "wide_gen", "sym": "A"}, {"event": "entry", "source": "leader", "sym": "B"},
               {"event": "entry", "source": "surge", "sym": "C"}, {"event": "exit", "source": "wide_gen", "sym": "A"}]
        self.assertEqual([e["sym"] for e in W.book_events(evs)], ["A"])
        p = W.POLICIES["x4_wide"]
        self.assertEqual((p.exits, p.admission, p.risk), (SP.X4.exits, SP.X4.admission, SP.X4.risk))
        self.assertEqual(list(W.POLICIES), ["x4_wide"])


class TestOnePositionPerCoin(unittest.TestCase):
    """Defect 2026-10-28: x4_wide opened AXTIB, BB and BIGTIME twice (a 15m and a 1h entry on the same bar). The executor
    skips a second entry into a held coin (reason asset_held), so the book must not double either."""

    def test_filter_held_and_same_batch_duplicates(self):
        import wide_scan as W
        e = lambda s, tf, ts: {"event": "entry", "source": "wide_gen", "sym": s, "tf": tf, "ts": ts}  # noqa: E731
        evs = [e("A", "1h", "2026-10-28T14:00:00+00:00"), e("A", "15m", "2026-10-28T14:00:00+00:00"),
               e("B", "15m", "2026-10-28T14:00:00+00:00"), e("C", "15m", "2026-10-28T14:15:00+00:00"),
               e("C", "15m", "2026-10-28T14:00:00+00:00"), e("D", "15m", "2026-10-28T14:00:00+00:00")]
        kept, skipped = W.book_filter(evs, held={"B"})
        self.assertEqual([(x["sym"], x["tf"], x["ts"][11:16]) for x in kept],
                         [("A", "15m", "14:00"), ("C", "15m", "14:00"), ("D", "15m", "14:00")])
        self.assertEqual(sorted((x["sym"], r) for x, r in skipped),
                         [("A", "asset_held"), ("B", "asset_held"), ("C", "asset_held")])
        self.assertEqual([x["ts"] for x in kept], sorted(x["ts"] for x in kept))     # time order kept

    def test_reset_archives_and_starts_from_now(self):
        """Reset 2026-10-28: the old book is archived, not deleted; the new one reads events only after the reset."""
        import shadow_portfolio as SP
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "shadow").mkdir()
            (root / "shadow" / "state.json").write_text(json.dumps({"tail": {"offset": 10, "seen": []}, "books": {"x": 1}}),
                                                       encoding="utf-8")
            (root / "shadow" / "ledger.jsonl").write_text('{"type": "entry"}\n', encoding="utf-8")
            (root / "events.jsonl").write_text('{"event": "entry", "source": "wide_gen", "sym": "OLD"}\n', encoding="utf-8")
            info = W.reset_book(root, stamp="20261028T1700")
            self.assertTrue((root / "shadow" / "state.pre_reset_20261028T1700.json").exists())
            self.assertTrue((root / "shadow" / "ledger.pre_reset_20261028T1700.jsonl").exists())
            self.assertFalse((root / "shadow" / "ledger.jsonl").exists())
            st = SP.load_state(root / "shadow" / "state.json")
            self.assertIsNone(st["books"])
            self.assertEqual(st["tail"]["offset"], (root / "events.jsonl").stat().st_size)
            self.assertEqual(info["archived"], 2)
            with open(root / "events.jsonl", "a", encoding="utf-8") as fh:
                fh.write('{"event": "entry", "source": "wide_gen", "sym": "NEW", "tf": "15m", "ts": "t"}\n')
            new = SP.EventTail(root / "events.jsonl", st["tail"]).read_new()
            self.assertEqual([e["sym"] for e in new], ["NEW"])

    def test_book_step_applies_the_filter(self):
        src = (HERE / "wide_scan.py").read_text(encoding="utf-8")
        self.assertIn("book_filter(book_events(tail.read_new())", src)


# --------------------------------------------------------------------------- 7. learning rows
class TestLearning(unittest.TestCase):
    def test_row_has_no_future_features(self):
        import wide_scan as W
        closes = [100.0 + 0.1 * i for i in range(120)]
        a = arr_for(closes)
        b = arr_for(closes[:90] + [x * 2 for x in closes[90:]])
        ev = {"event": "entry", "source": "wide_gen", "sym": "AUSDT", "tf": "15m",
              "ts": datetime.fromtimestamp((int(a["t"][89]) + BAR_MS) / 1000, timezone.utc).isoformat(), "price": closes[89]}
        r1 = W.candidate_row(ev, a, 89, rank=3, n_universe=494, in_watchlist=False)
        r2 = W.candidate_row(ev, b, 89, rank=3, n_universe=494, in_watchlist=False)
        self.assertEqual(r1, r2)
        for k in ("ret_since_open", "rank", "ret_15m", "ret_1h", "qv_15m_ratio", "atr_pct", "in_watchlist", "key"):
            self.assertIn(k, r1)

    def test_outcomes_once_after_24h(self):
        import wide_scan as W
        n = 96 * 3
        closes = [100.0 * (1.001 ** i) for i in range(n)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = W.BarStore(root / "bars")
            st.append("AUSDT", rows_for(closes), at(n - 1))
            ev = {"event": "entry", "source": "wide_gen", "sym": "AUSDT", "tf": "15m",
                  "ts": datetime.fromtimestamp((D0_MS + 10 * BAR_MS) / 1000, timezone.utc).isoformat(), "price": closes[9]}
            a = st.load("AUSDT")
            W.append_candidates(root, [W.candidate_row(ev, a, 9, rank=1, n_universe=4, in_watchlist=False)])
            W.append_candidates(root, [W.candidate_row(ev, a, 9, rank=1, n_universe=4, in_watchlist=False)])   # idempotent
            self.assertEqual(W.label_outcomes(root, at(40)), 0)                  # 24 h not yet closed
            self.assertEqual(W.label_outcomes(root, at(n - 1)), 1)
            self.assertEqual(W.label_outcomes(root, at(n - 1)), 0)
            rows = [json.loads(x) for x in (root / "learning" / "wide_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["candidate", "outcome"])
            o = rows[1]
            self.assertAlmostEqual(o["fwd_24h"], closes[9 + 96] / closes[9] - 1, places=9)
            self.assertAlmostEqual(o["fwd_4h"], closes[9 + 16] / closes[9] - 1, places=9)
            for k in ("max_24h", "min_24h", "top20_day", "key"):
                self.assertIn(k, o)


# --------------------------------------------------------------------------- 8. leaders of the day
class TestLeaders(unittest.TestCase):
    def test_top20_by_close_over_open_and_no_data_days(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            st = W.BarStore(Path(d))
            syms = []
            for k in range(25):
                s = f"S{k:02d}USDT"
                syms.append(s)
                c = [100.0] * 95 + [100.0 + k]                               # day return = k %
                st.append(s, rows_for(c, opens=[100.0] * 96), at(95))
            top = W.leaders_of_day(st, syms, DAY0.strftime("%Y-%m-%d"))
            self.assertEqual(len(top), 20)
            self.assertEqual(top[0], "S24USDT")
            self.assertNotIn("S04USDT", top)
            self.assertIsNone(W.leaders_of_day(st, syms, (DAY0 + timedelta(days=1)).strftime("%Y-%m-%d")))   # no data

    def test_early_seen_carries_the_base_rate(self):
        import wide_scan as W
        with tempfile.TemporaryDirectory() as d:
            st = W.BarStore(Path(d))
            c = [100.0] * 10 + [101.0, 102.0, 103.0, 106.0] + [110.0] * 82
            st.append("AUSDT", rows_for(c, opens=[100.0] + c[:-1]), at(95))
            st.append("BUSDT", rows_for([100.0] * 96), at(95))
            ts_early = datetime.fromtimestamp((D0_MS + 12 * BAR_MS) / 1000, timezone.utc).isoformat()   # +2% so far
            evs = [{"source": "surge", "sym": "AUSDT", "ts": ts_early}, {"source": "leader", "sym": "BUSDT", "ts": ts_early}]
            r = W.early_seen(["AUSDT"], evs, st, DAY0.strftime("%Y-%m-%d"))
            self.assertEqual(r["n_leaders"], 1)
            self.assertEqual(r["flagged_before_cross"], 1)
            self.assertEqual(r["n_flagged_pairs"], 2)
            self.assertAlmostEqual(r["flagged_hit_share"], 0.5)
            self.assertIn("base_rate", r)
            self.assertIn("move_done_at_flag_median", r)


# --------------------------------------------------------------------------- 9-10. isolation, flag
class TestIsolation(unittest.TestCase):
    def test_no_signed_requests_no_bot_writes(self):
        src = (HERE / "wide_scan.py").read_text(encoding="utf-8")
        for bad in ("account(", "test_order(", "place_order(", "place_oco(", "signed=True", "BINANCE_DEMO_API_SECRET",
                    '"' + '.runtime' + '"', "BOT_FILES /"):
            self.assertNotIn(bad, src, bad)
        import trader_config as TC
        import wide_scan as W
        self.assertTrue(hasattr(TC, "WIDE_SCAN_ENABLED"))
        self.assertEqual(W.ROOT, TC.RUNTIME / "scan")
        self.assertTrue(str(W.LEARNING).startswith(str(TC.RUNTIME)))

    def test_audit_lines_whole_under_threads(self):
        import threading
        import wide_scan as W

        class Slow:
            def __init__(self):
                self.lines, self.busy = [], False

            def write(self, rec, now=None):
                assert not self.busy, "concurrent write"
                self.busy = True
                self.lines.append(rec)
                self.busy = False
                return True
        inner = Slow()
        w = W.LockedWriter(inner)
        ths = [threading.Thread(target=lambda: [w.write({"i": k}) for k in range(200)]) for _ in range(8)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertEqual(len(inner.lines), 1600)

    def test_public_host_only(self):
        import exchange_client as X
        import wide_scan as W
        self.assertIn(W.KLINES_HOST.split("//")[1], X.PUBLIC_HOSTS)
        self.assertNotIn(W.KLINES_HOST.split("//")[1], X.ALLOWED_HOSTS)      # the live public market, not the demo


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestDailyReportBlock(unittest.TestCase):
    """wide-scan-shadow-w1-1028-spec.md §3 (operator 2026-10-28: the W1 metrics go into the trader's daily report)."""

    def wide(self, **kw):
        r = {"day": "2026-10-29",
             "coverage": {"bars_share": 1.0, "pair_share": 0.998, "latency_p50": 150.0, "latency_p95": 260.0},
             "leaders": [f"L{k}USDT" for k in range(20)],
             "early_seen": {"n_leaders": 20, "n_flagged_leaders": 9, "flagged_before_cross": 2, "share_before_cross": 0.1,
                            "n_flagged_pairs": 60, "flagged_hit_share": 0.15, "n_pairs_day": 500, "base_rate": 0.04,
                            "move_done_at_flag_median": 0.55},
             "early_seen_outside_bot": {"n_leaders": 15, "n_flagged_leaders": 7, "flagged_before_cross": 1,
                                        "share_before_cross": 1 / 15, "n_flagged_pairs": 60, "flagged_hit_share": 0.12,
                                        "n_pairs_day": 500, "base_rate": 0.03, "move_done_at_flag_median": 0.6},
             "x4_wide_day": -0.12, "x4_floor8_day": -0.30, "disk_mb": 60.0}
        r.update(kw)
        return r

    def test_block_with_base_rate(self):
        import trader_report as TR
        txt = TR.build("2026-10-29", [], wide=self.wide())
        self.assertIn("Весь рынок", txt)
        for s in ("2 из 20", "база", "4.0%", "вне 93", "1 из 15", "x4_wide", "-0.120", "x4_floor8", "-0.300", "55%"):
            self.assertIn(s, txt, s)

    def test_no_scanner_data(self):
        import trader_report as TR
        txt = TR.build("2026-10-29", [], wide={"day": "2026-10-29", "coverage": None, "leaders": None})
        self.assertIn("Весь рынок", txt)
        self.assertIn("нет данных", txt)
        self.assertNotIn("из 20", txt)

    def test_report_main_passes_the_scanner(self):
        src = (HERE / "trader_report.py").read_text(encoding="utf-8")
        self.assertIn("wide_scan", src)
        self.assertIn("wide=", src)
