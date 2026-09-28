"""Leader exit mode (X-7b). Spec: docs/specs/features/leader-exit-x6-spec.md"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import leader_alert as LA  # noqa: E402
import leader_exit as LX  # noqa: E402
import rotation  # noqa: E402

BAR = 15 * 60 * 1000
T0 = 1_790_500_000_000 // BAR * BAR


def pos(**kw):
    d = dict(entry_ts=T0, trail_k=2.0, trail_stop=95.0, leader_mode=False, leader_since_ts=0,
             ranker_ev=-1.0, bars_elapsed=10, entry_price=100.0, symbol="AUSDT", tf="15m", signal_mode="trend")
    d.update(kw)
    return SimpleNamespace(**d)


def ranks(bar_ts, table):
    return bar_ts, table


class TestSwitch(unittest.TestCase):
    def test_switch_on_top3_at_plus5(self):
        p = pos()
        a, why = LX.step(p, sym="AUSDT", tf="15m", close=110.0, atr=1.0, bar_ts=T0 + 4 * BAR,
                         ranks=ranks(T0 + 4 * BAR, {"AUSDT": (2, 0.06)}))
        self.assertEqual(a, "switch")
        self.assertTrue(p.leader_mode)
        self.assertEqual(p.leader_since_ts, T0 + 4 * BAR)
        self.assertAlmostEqual(p.trail_stop, 110.0 - 8.8)      # 8% floor > 2*ATR
        self.assertIn("№2", why)

    def test_switch_may_lower_the_stop(self):
        p = pos(trail_stop=108.0)
        LX.step(p, sym="AUSDT", tf="15m", close=110.0, atr=1.0, bar_ts=T0 + BAR, ranks=ranks(T0 + BAR, {"AUSDT": (1, 0.10)}))
        self.assertLess(p.trail_stop, 108.0)

    def test_no_switch_when_not_a_leader(self):
        for rr in ((6, 0.10), (1, 0.029)):
            p = pos()
            a, _ = LX.step(p, sym="AUSDT", tf="15m", close=110.0, atr=1.0, bar_ts=T0 + BAR, ranks=ranks(T0 + BAR, {"AUSDT": rr}))
            self.assertEqual(a, "none")
            self.assertFalse(p.leader_mode)

    def test_only_15m(self):
        a, _ = LX.step(pos(tf="1h"), sym="AUSDT", tf="1h", close=110.0, atr=1.0, bar_ts=T0 + BAR,
                       ranks=ranks(T0 + BAR, {"AUSDT": (1, 0.2)}))
        self.assertEqual(a, "none")

    def test_stale_or_missing_ranks_never_switch(self):
        for r in (None, ranks(T0 - 3 * BAR, {"AUSDT": (1, 0.2)}), ranks(T0 + 5 * BAR, {})):
            a, _ = LX.step(pos(), sym="AUSDT", tf="15m", close=110.0, atr=1.0, bar_ts=T0 + 5 * BAR, ranks=r)
            self.assertEqual(a, "none")

    def test_ranks_older_than_the_entry_do_not_switch(self):
        a, _ = LX.step(pos(entry_ts=T0 + 2 * BAR), sym="AUSDT", tf="15m", close=110.0, atr=1.0, bar_ts=T0 + 2 * BAR,
                       ranks=ranks(T0 + BAR, {"AUSDT": (1, 0.2)}))
        self.assertEqual(a, "none")


class TestHoldAndExit(unittest.TestCase):
    def leader(self, **kw):
        return pos(leader_mode=True, leader_since_ts=T0, trail_stop=92.0, **kw)

    def test_hold_and_ratchet(self):
        p = self.leader()
        a, _ = LX.step(p, sym="AUSDT", tf="15m", close=120.0, atr=1.0, bar_ts=T0 + 2 * BAR, ranks=ranks(T0 + 2 * BAR, {"AUSDT": (5, 0.2)}))
        self.assertEqual(a, "hold")
        self.assertAlmostEqual(p.trail_stop, 120.0 - 9.6)

    def test_exit_on_the_wide_trail(self):
        p = self.leader()
        a, why = LX.step(p, sym="AUSDT", tf="15m", close=91.0, atr=1.0, bar_ts=T0 + 2 * BAR, ranks=None)
        self.assertEqual(a, "exit")
        self.assertIn("трейл", why)

    def test_lost_lead_waits_one_hour(self):
        p = self.leader()
        a, _ = LX.step(p, sym="AUSDT", tf="15m", close=100.0, atr=1.0, bar_ts=T0 + 3 * BAR, ranks=ranks(T0 + 3 * BAR, {"AUSDT": (15, 0.01)}))
        self.assertEqual(a, "hold")
        a, why = LX.step(p, sym="AUSDT", tf="15m", close=100.0, atr=1.0, bar_ts=T0 + 4 * BAR, ranks=ranks(T0 + 4 * BAR, {"AUSDT": (15, 0.01)}))
        self.assertEqual(a, "exit")
        self.assertIn("топ-10", why)

    def test_top10_is_still_held(self):
        p = self.leader()
        a, _ = LX.step(p, sym="AUSDT", tf="15m", close=100.0, atr=1.0, bar_ts=T0 + 8 * BAR, ranks=ranks(T0 + 8 * BAR, {"AUSDT": (10, 0.03)}))
        self.assertEqual(a, "hold")

    def test_stale_ranks_do_not_force_the_lost_lead_exit(self):
        p = self.leader()
        a, _ = LX.step(p, sym="AUSDT", tf="15m", close=100.0, atr=1.0, bar_ts=T0 + 8 * BAR, ranks=ranks(T0 + 2 * BAR, {"AUSDT": (50, 0.0)}))
        self.assertEqual(a, "hold")

    def test_seven_day_cap(self):
        p = self.leader()
        a, why = LX.step(p, sym="AUSDT", tf="15m", close=130.0, atr=1.0, bar_ts=T0 + 672 * BAR, ranks=None)
        self.assertEqual(a, "exit")
        self.assertIn("7 дней", why)


class TestRanking(unittest.TestCase):
    def test_rank_last_bar(self):
        day = {"A": (1.0, {T0: 1.0, T0 + BAR: 1.10}), "B": (1.0, {T0: 1.0, T0 + BAR: 1.20}), "C": (1.0, {T0: 1.0})}
        bar, table = LA.rank_last_bar(day)
        self.assertEqual(bar, T0 + BAR)
        self.assertEqual(table["B"][0], 1)
        self.assertEqual(table["A"][0], 2)
        self.assertNotIn("C", table)

    def test_ranking_needed_when_only_the_exit_is_on(self):
        with mock.patch.object(config, "LEADER_ALERT_ENABLED", False), mock.patch.object(config, "LEADER_EXIT_ENABLED", True):
            self.assertTrue(LA.ranking_needed())
        with mock.patch.object(config, "LEADER_ALERT_ENABLED", False), mock.patch.object(config, "LEADER_EXIT_ENABLED", False):
            self.assertFalse(LA.ranking_needed())


class TestProtection(unittest.TestCase):
    def test_rotation_never_evicts_a_leader(self):
        positions = {"AUSDT": pos(leader_mode=True), "BUSDT": pos(symbol="BUSDT", ranker_ev=-0.9)}
        cfg = SimpleNamespace(ROTATION_WEAK_EV_MAX=-0.40, ROTATION_WEAK_BARS_MIN=3, ROTATION_PROFIT_PROTECT_PCT=0.5)
        self.assertEqual(rotation.find_weakest_leg(positions, cfg, {"AUSDT": 99.0, "BUSDT": 99.0}), "BUSDT")

    def test_monitor_wiring(self):
        src = (HERE / "monitor.py").read_text(encoding="utf-8")
        body = src[src.index("async def _poll_coin("):]
        lx = body.index("leader_exit.step(")
        # the leader step runs before every other exit of the position
        for later in ("continuation_profit_lock_active = _continuation_profit_lock_active(", "# Выход по ATR-трейлу",
                      "check_exit_conditions(", "_fast_loss_ema_exit_reason("):
            self.assertLess(lx, body.index(later), later)
        rep = src[src.index("def _find_replaceable_position("):]
        self.assertIn('if getattr(pos, "leader_mode", False):', rep[:3000])
        self.assertIn('getattr(_prune_pos, "leader_mode", False)', src)
        self.assertIn("leader_alert.ranking_needed()", src)

    def test_persistence_round_trip(self):
        import monitor
        p = monitor.OpenPosition(symbol="AUSDT", tf="15m", entry_price=1.0, entry_bar=0, entry_ts=T0,
                                 entry_ema20=1.0, entry_slope=0.0, entry_adx=20.0, entry_rsi=50.0, entry_vol_x=1.0)
        p.leader_mode, p.leader_since_ts = True, T0 + BAR
        q = monitor._pos_from_dict(monitor._pos_to_dict(p))
        self.assertTrue(q.leader_mode)
        self.assertEqual(q.leader_since_ts, T0 + BAR)

    def test_flags(self):
        self.assertIs(config.LEADER_EXIT_ENABLED, True)
        self.assertEqual((config.LEADER_EXIT_RANK_MAX, config.LEADER_EXIT_MIN_RET, config.LEADER_EXIT_FLOOR_PCT,
                          config.LEADER_EXIT_LOST_RANK, config.LEADER_EXIT_LOST_MIN_BARS), (5, 0.03, 0.08, 10, 4))
        self.assertEqual(tuple(config.LEADER_EXIT_TF), ("15m",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
