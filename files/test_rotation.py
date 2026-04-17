"""
Unit-тесты для rotation.py — ML-gated portfolio rotation.
Запуск: python -m pytest test_rotation.py -v
     или: python test_rotation.py
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

import rotation


# ── Минимальная фейковая позиция для тестов ────────────────────────────────
@dataclass
class FakePos:
    symbol: str = ""
    entry_price: float = 100.0
    ranker_ev: float = 0.0
    bars_elapsed: int = 0
    trail_stop: float = 0.0

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price / self.entry_price - 1.0) * 100.0


# ── Дефолтный конфиг для тестов ─────────────────────────────────────────────
def make_config(**overrides):
    defaults = dict(
        ROTATION_ENABLED=True,
        ROTATION_ML_PROBA_MIN=0.62,
        ROTATION_WEAK_EV_MAX=-0.40,
        ROTATION_WEAK_BARS_MIN=3,
        ROTATION_PROFIT_PROTECT_PCT=0.5,
        ROTATION_MAX_PER_POLL=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Тесты ───────────────────────────────────────────────────────────────────
class TestFindWeakestLeg(unittest.TestCase):

    def test_picks_lowest_ev(self):
        positions = {
            "A": FakePos("A", ranker_ev=-0.50, bars_elapsed=5),
            "B": FakePos("B", ranker_ev=-0.70, bars_elapsed=5),  # слабее
            "C": FakePos("C", ranker_ev=-0.45, bars_elapsed=5),
        }
        last_prices = {"A": 100, "B": 100, "C": 100}
        weak = rotation.find_weakest_leg(positions, make_config(), last_prices)
        self.assertEqual(weak, "B")

    def test_skips_strong_ev(self):
        positions = {
            "A": FakePos("A", ranker_ev=-0.30, bars_elapsed=5),  # >-0.40 — сильная
            "B": FakePos("B", ranker_ev=-0.20, bars_elapsed=5),
        }
        weak = rotation.find_weakest_leg(positions, make_config(), {"A": 100, "B": 100})
        self.assertIsNone(weak)

    def test_skips_fresh_position(self):
        positions = {
            "A": FakePos("A", ranker_ev=-0.70, bars_elapsed=1),  # меньше WEAK_BARS_MIN=3
            "B": FakePos("B", ranker_ev=-0.50, bars_elapsed=5),
        }
        weak = rotation.find_weakest_leg(positions, make_config(), {"A": 100, "B": 100})
        self.assertEqual(weak, "B")  # B берём, A пропускаем как свежую

    def test_skips_profitable_position(self):
        # A — сильно убыточная по EV, но в плюсе по PnL → не трогать
        p_a = FakePos("A", ranker_ev=-0.80, bars_elapsed=5, entry_price=100.0)
        p_b = FakePos("B", ranker_ev=-0.50, bars_elapsed=5, entry_price=100.0)
        last_prices = {"A": 102.0, "B": 99.0}  # A: +2%, B: -1%
        weak = rotation.find_weakest_leg({"A": p_a, "B": p_b}, make_config(), last_prices)
        self.assertEqual(weak, "B")

    def test_empty_portfolio(self):
        self.assertIsNone(rotation.find_weakest_leg({}, make_config(), {}))

    def test_all_positions_strong(self):
        positions = {
            "A": FakePos("A", ranker_ev=+0.10, bars_elapsed=5),
            "B": FakePos("B", ranker_ev=-0.20, bars_elapsed=5),
            "C": FakePos("C", ranker_ev=+0.05, bars_elapsed=5),
        }
        weak = rotation.find_weakest_leg(positions, make_config(), {"A": 100, "B": 100, "C": 100})
        self.assertIsNone(weak)

    def test_missing_last_price_treated_as_zero_pnl(self):
        positions = {"A": FakePos("A", ranker_ev=-0.60, bars_elapsed=5)}
        # last_prices отсутствуют — pnl=0, что <= 0.5 → pass
        weak = rotation.find_weakest_leg(positions, make_config(), {})
        self.assertEqual(weak, "A")


class TestShouldRotate(unittest.TestCase):

    def setUp(self):
        self.positions = {
            "A": FakePos("A", ranker_ev=-0.70, bars_elapsed=5),
            "B": FakePos("B", ranker_ev=-0.30, bars_elapsed=5),
        }
        self.last_prices = {"A": 100, "B": 100}

    def test_disabled(self):
        cfg = make_config(ROTATION_ENABLED=False)
        d = rotation.should_rotate(0.70, self.positions, cfg, self.last_prices)
        self.assertFalse(d.allowed)
        self.assertIn("disabled", d.reason)

    def test_ml_proba_none(self):
        d = rotation.should_rotate(None, self.positions, make_config(), self.last_prices)
        self.assertFalse(d.allowed)
        self.assertIn("unknown", d.reason)

    def test_ml_proba_below_threshold(self):
        d = rotation.should_rotate(0.55, self.positions, make_config(), self.last_prices)
        self.assertFalse(d.allowed)
        self.assertIn("0.55", d.reason)

    def test_ml_proba_above_threshold_weak_exists(self):
        d = rotation.should_rotate(0.70, self.positions, make_config(), self.last_prices)
        self.assertTrue(d.allowed)
        self.assertEqual(d.weak_sym, "A")
        self.assertAlmostEqual(d.weak_ev, -0.70, places=2)

    def test_ml_proba_above_threshold_no_weak(self):
        strong_only = {
            "A": FakePos("A", ranker_ev=-0.20, bars_elapsed=5),
            "B": FakePos("B", ranker_ev=+0.10, bars_elapsed=5),
        }
        d = rotation.should_rotate(0.70, strong_only, make_config(), {"A": 100, "B": 100})
        self.assertFalse(d.allowed)
        self.assertIn("no weak leg", d.reason)

    def test_invalid_ml_proba(self):
        d = rotation.should_rotate("not-a-number", self.positions, make_config(), self.last_prices)
        self.assertFalse(d.allowed)
        self.assertIn("invalid", d.reason)

    def test_exact_threshold_accepted(self):
        d = rotation.should_rotate(0.62, self.positions, make_config(), self.last_prices)
        self.assertTrue(d.allowed)

    def test_empty_portfolio(self):
        d = rotation.should_rotate(0.70, {}, make_config(), {})
        self.assertFalse(d.allowed)
        self.assertIn("no weak leg", d.reason)


class TestEvictPosition(unittest.TestCase):

    def test_sets_trail_above_price(self):
        p = FakePos("A", entry_price=100.0, trail_stop=95.0)
        ok = rotation.evict_position(p, last_price=100.0)
        self.assertTrue(ok)
        self.assertGreater(p.trail_stop, 100.0)
        self.assertAlmostEqual(p.trail_stop, 100.1, places=4)

    def test_preserves_higher_existing_trail(self):
        p = FakePos("A", entry_price=100.0, trail_stop=101.5)
        ok = rotation.evict_position(p, last_price=100.0)
        self.assertTrue(ok)
        self.assertEqual(p.trail_stop, 101.5)

    def test_invalid_price(self):
        p = FakePos("A", entry_price=100.0, trail_stop=95.0)
        self.assertFalse(rotation.evict_position(p, last_price=None))
        self.assertFalse(rotation.evict_position(p, last_price=0.0))
        self.assertFalse(rotation.evict_position(p, last_price=-1.0))
        self.assertEqual(p.trail_stop, 95.0)  # не изменилось


class TestFormatMessage(unittest.TestCase):

    def test_contains_key_info(self):
        d = rotation.RotationDecision(
            allowed=True,
            weak_sym="METIS",
            weak_ev=-0.70,
            weak_bars=8,
            weak_pnl=-1.2,
            reason="evict METIS",
        )
        msg = rotation.format_rotation_message(
            sym="SOL", mode="impulse_speed", tf="1h",
            candidate_ml_proba=0.67, decision=d,
        )
        self.assertIn("METIS", msg)
        self.assertIn("SOL", msg)
        self.assertIn("impulse_speed", msg)
        self.assertIn("0.67", msg)
        self.assertIn("-0.70", msg)


class TestProfitProtectionEdgeCases(unittest.TestCase):

    def test_above_profit_protect_threshold(self):
        # +0.6% > 0.5% — защищённая позиция, не трогаем
        p = FakePos("A", ranker_ev=-0.80, bars_elapsed=5, entry_price=100.0)
        weak = rotation.find_weakest_leg({"A": p}, make_config(), {"A": 100.6})
        self.assertIsNone(weak)

    def test_just_below_profit_protect(self):
        # +0.3% < 0.5% → разрешаем вытеснение
        p = FakePos("A", ranker_ev=-0.80, bars_elapsed=5, entry_price=100.0)
        weak = rotation.find_weakest_leg({"A": p}, make_config(), {"A": 100.3})
        self.assertEqual(weak, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
