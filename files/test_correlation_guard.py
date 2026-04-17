"""
test_correlation_guard.py — Unit-тесты для correlation_guard.py

Покрывает:
  - CorrelationCache: get/set, is_fresh, mark_built
  - _pearson: базовые значения, нулевой std, короткий ряд
  - _log_returns: корректность формулы
  - _union_find_clusters: независимые, chain, polный граф
  - check_entry: allow, block, cluster size logic, bear mode (threshold=0.99)
  - marginal_score: нет штрафа без данных, штраф при высоком rho
  - prune_candidates: нет prune при малом кластере, prune при большом, profit protect
  - format_cluster_report: smoke test

Запуск: python -m pytest test_correlation_guard.py -v
"""

from __future__ import annotations

import math
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import patch, MagicMock

# Минимальный stub config чтобы не тащить весь config.py
import types
_cfg = types.ModuleType("config")
_cfg.CORR_GUARD_ENABLED = True
_cfg.CORR_GUARD_TF = "1h"
_cfg.CORR_GUARD_WINDOW_BARS = 48
_cfg.CORR_GUARD_THRESHOLD = 0.65
_cfg.CORR_GUARD_THRESHOLD_BULL = 0.60
_cfg.CORR_GUARD_THRESHOLD_BEAR = 0.99
_cfg.CORR_MAX_PER_CLUSTER = 2
_cfg.CORR_MARGINAL_WEIGHTING = True
_cfg.CORR_PRUNE_ENABLED = True
_cfg.CORR_PRUNE_PROFIT_PROTECT_PCT = 2.0
_cfg.CORR_CACHE_TTL_MIN = 15
sys.modules["config"] = _cfg

import correlation_guard as cg


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _FakePos:
    """Minimal OpenPosition stub."""
    symbol: str
    entry_price: float
    ranker_final_score: float = -0.5


def _make_cache(pairs: Dict[tuple, float]) -> cg.CorrelationCache:
    """Создаёт CorrelationCache с заранее заданными значениями."""
    cache = cg.CorrelationCache()
    for (a, b), rho in pairs.items():
        cache.set(a, b, rho)
    cache.mark_built([a for a, _ in pairs] + [b for _, b in pairs])
    return cache


# ── CorrelationCache ──────────────────────────────────────────────────────────

class TestCorrelationCache(unittest.TestCase):

    def test_get_set_symmetric(self):
        c = cg.CorrelationCache()
        c.set("AAA", "BBB", 0.75)
        self.assertAlmostEqual(c.get("AAA", "BBB"), 0.75)
        self.assertAlmostEqual(c.get("BBB", "AAA"), 0.75)  # symmetric

    def test_self_correlation(self):
        c = cg.CorrelationCache()
        self.assertEqual(c.get("AAA", "AAA"), 1.0)

    def test_missing_returns_none(self):
        c = cg.CorrelationCache()
        self.assertIsNone(c.get("X", "Y"))

    def test_is_fresh_ttl(self):
        c = cg.CorrelationCache()
        syms = ["A", "B"]
        c.mark_built(syms)
        self.assertTrue(c.is_fresh(syms))

    def test_is_fresh_different_syms(self):
        c = cg.CorrelationCache()
        c.mark_built(["A", "B"])
        self.assertFalse(c.is_fresh(["A", "B", "C"]))

    def test_is_fresh_expired(self):
        c = cg.CorrelationCache()
        c.mark_built(["A"])
        c.built_at = time.monotonic() - 999  # expired
        self.assertFalse(c.is_fresh(["A"]))


# ── Math primitives ───────────────────────────────────────────────────────────

class TestLogReturns(unittest.TestCase):

    def test_basic(self):
        closes = [100.0, 110.0, 99.0]
        rets = cg._log_returns(closes)
        self.assertEqual(len(rets), 2)
        self.assertAlmostEqual(rets[0], math.log(110 / 100), places=10)
        self.assertAlmostEqual(rets[1], math.log(99 / 110), places=10)

    def test_zero_close_gives_zero(self):
        closes = [100.0, 0.0, 110.0]
        rets = cg._log_returns(closes)
        self.assertEqual(rets[0], 0.0)   # 0-division guard → 0.0
        self.assertEqual(rets[1], 0.0)


class TestPearson(unittest.TestCase):

    def test_perfect_positive(self):
        a = [float(x) for x in range(20)]
        b = [float(x) * 2 for x in range(20)]
        self.assertAlmostEqual(cg._pearson(a, b), 1.0, places=5)

    def test_perfect_negative(self):
        a = [float(x) for x in range(20)]
        b = [float(-x) for x in range(20)]
        self.assertAlmostEqual(cg._pearson(a, b), -1.0, places=5)

    def test_uncorrelated(self):
        import random
        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(200)]
        b = [random.gauss(0, 1) for _ in range(200)]
        rho = cg._pearson(a, b)
        self.assertLess(abs(rho), 0.20)  # должно быть близко к 0

    def test_zero_std_returns_zero(self):
        a = [1.0] * 20  # нулевой std
        b = [float(x) for x in range(20)]
        self.assertEqual(cg._pearson(a, b), 0.0)

    def test_short_series_returns_zero(self):
        a = [1.0, 2.0, 3.0]   # < 5 элементов
        b = [1.0, 2.0, 3.0]
        self.assertEqual(cg._pearson(a, b), 0.0)


# ── Union-Find ────────────────────────────────────────────────────────────────

class TestUnionFind(unittest.TestCase):

    def test_all_independent(self):
        cache = cg.CorrelationCache()
        # Все rho = 0 (нет данных)
        syms = ["A", "B", "C"]
        clusters = cg._union_find_clusters(syms, cache, 0.65)
        # Все в разных кластерах
        self.assertEqual(len(set(clusters.values())), 3)

    def test_all_same_cluster(self):
        pairs = {("A", "B"): 0.80, ("A", "C"): 0.80, ("B", "C"): 0.80}
        cache = _make_cache(pairs)
        clusters = cg._union_find_clusters(["A", "B", "C"], cache, 0.65)
        self.assertEqual(len(set(clusters.values())), 1)

    def test_chain_cluster(self):
        # A-B коррелированы, B-C коррелированы → A/B/C в одном кластере (транзитивно)
        pairs = {("A", "B"): 0.70, ("B", "C"): 0.70, ("A", "C"): 0.40}
        cache = _make_cache(pairs)
        clusters = cg._union_find_clusters(["A", "B", "C"], cache, 0.65)
        self.assertEqual(len(set(clusters.values())), 1)

    def test_two_separate_clusters(self):
        pairs = {
            ("A", "B"): 0.75, ("A", "C"): 0.10,
            ("B", "C"): 0.10, ("C", "D"): 0.75,
        }
        cache = _make_cache(pairs)
        clusters = cg._union_find_clusters(["A", "B", "C", "D"], cache, 0.65)
        self.assertEqual(len(set(clusters.values())), 2)
        self.assertEqual(clusters["A"], clusters["B"])
        self.assertEqual(clusters["C"], clusters["D"])
        self.assertNotEqual(clusters["A"], clusters["C"])


# ── check_entry ───────────────────────────────────────────────────────────────

class TestCheckEntry(unittest.TestCase):

    def _positions(self, *syms):
        return {s: _FakePos(symbol=s, entry_price=1.0) for s in syms}

    def test_empty_portfolio_always_allowed(self):
        cache = cg.CorrelationCache()
        result = cg.check_entry("NEW", {}, cache)
        self.assertTrue(result.allowed)

    def test_no_correlation_data_allowed(self):
        # Нет данных в кэше → no block (unknown rho = no problem)
        cache = cg.CorrelationCache()
        pos = self._positions("EXISTING")
        result = cg.check_entry("NEW", pos, cache)
        self.assertTrue(result.allowed)

    def test_low_rho_allowed(self):
        pairs = {("NEW", "EXISTING"): 0.40}
        cache = _make_cache(pairs)
        pos = self._positions("EXISTING")
        result = cg.check_entry("NEW", pos, cache)
        self.assertTrue(result.allowed)

    def test_high_rho_one_existing_allowed(self):
        # 1 сильно коррелированная позиция — разрешено (cluster_size=1 < max=2)
        pairs = {("NEW", "EXIST1"): 0.80}
        cache = _make_cache(pairs)
        pos = self._positions("EXIST1")
        result = cg.check_entry("NEW", pos, cache)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.max_rho, 0.80)

    def test_high_rho_two_existing_blocked(self):
        # Два существующих с rho >= 0.65 → кластер уже полон (2/2) → блок
        pairs = {
            ("NEW", "EXIST1"): 0.75,
            ("NEW", "EXIST2"): 0.70,
        }
        cache = _make_cache(pairs)
        pos = self._positions("EXIST1", "EXIST2")
        result = cg.check_entry("NEW", pos, cache)
        self.assertFalse(result.allowed)
        self.assertEqual(result.cluster_size, 2)
        self.assertIn("corr_guard", result.reason)

    def test_bear_day_threshold_relaxed(self):
        # В медвежий день порог 0.99 → даже rho=0.90 не блокирует
        pairs = {
            ("NEW", "EXIST1"): 0.90,
            ("NEW", "EXIST2"): 0.90,
        }
        cache = _make_cache(pairs)
        pos = self._positions("EXIST1", "EXIST2")
        result = cg.check_entry("NEW", pos, cache, is_bear_day=True)
        self.assertTrue(result.allowed)

    def test_bull_day_threshold_stricter(self):
        # В бычий день порог 0.60 → rho=0.62 уже считается кластером
        pairs = {
            ("NEW", "EXIST1"): 0.62,
            ("NEW", "EXIST2"): 0.62,
        }
        cache = _make_cache(pairs)
        pos = self._positions("EXIST1", "EXIST2")
        result = cg.check_entry("NEW", pos, cache, is_bull_day=True)
        self.assertFalse(result.allowed)

    def test_disabled_guard_always_allowed(self):
        _cfg.CORR_GUARD_ENABLED = False
        pairs = {("NEW", "EXIST1"): 0.99, ("NEW", "EXIST2"): 0.99}
        cache = _make_cache(pairs)
        pos = self._positions("EXIST1", "EXIST2")
        result = cg.check_entry("NEW", pos, cache)
        self.assertTrue(result.allowed)
        _cfg.CORR_GUARD_ENABLED = True  # restore


# ── marginal_score ────────────────────────────────────────────────────────────

class TestMarginalScore(unittest.TestCase):

    def _pos(self, *syms):
        return {s: _FakePos(symbol=s, entry_price=1.0) for s in syms}

    def test_no_positions_no_change(self):
        cache = cg.CorrelationCache()
        ms = cg.marginal_score("NEW", {}, 100.0, cache)
        self.assertEqual(ms, 100.0)

    def test_no_cache_data_no_change(self):
        cache = cg.CorrelationCache()  # empty
        ms = cg.marginal_score("NEW", self._pos("EXIST"), 100.0, cache)
        self.assertEqual(ms, 100.0)

    def test_low_rho_below_threshold_no_change(self):
        pairs = {("NEW", "EXIST"): 0.25}
        cache = _make_cache(pairs)
        ms = cg.marginal_score("NEW", self._pos("EXIST"), 100.0, cache)
        self.assertEqual(ms, 100.0)  # < 0.30 → no penalty

    def test_high_rho_penalizes(self):
        pairs = {("NEW", "EXIST"): 0.75}
        cache = _make_cache(pairs)
        ms = cg.marginal_score("NEW", self._pos("EXIST"), 100.0, cache)
        self.assertAlmostEqual(ms, 100.0 * (1.0 - 0.75), places=5)

    def test_disabled_no_change(self):
        _cfg.CORR_MARGINAL_WEIGHTING = False
        pairs = {("NEW", "EXIST"): 0.99}
        cache = _make_cache(pairs)
        ms = cg.marginal_score("NEW", self._pos("EXIST"), 100.0, cache)
        self.assertEqual(ms, 100.0)
        _cfg.CORR_MARGINAL_WEIGHTING = True  # restore


# ── prune_candidates ──────────────────────────────────────────────────────────

class TestPruneCandidates(unittest.TestCase):

    def _make_positions(self, specs):
        """specs = [(sym, entry_price, ranker_final_score)]"""
        return {
            sym: _FakePos(symbol=sym, entry_price=ep, ranker_final_score=rfs)
            for sym, ep, rfs in specs
        }

    def test_no_prune_small_cluster(self):
        # Кластер из 2 (= max) → не закрываем
        pairs = {("A", "B"): 0.80}
        cache = _make_cache(pairs)
        pos = self._make_positions([("A", 1.0, -0.3), ("B", 1.0, -0.5)])
        result = cg.prune_candidates(pos, {"A": 1.0, "B": 1.0}, cache)
        self.assertEqual(len(result), 0)

    def test_prune_oversized_cluster(self):
        # Кластер из 3 (> max=2) → худший по ranker закрывается
        pairs = {
            ("A", "B"): 0.80, ("A", "C"): 0.80, ("B", "C"): 0.80,
        }
        cache = _make_cache(pairs)
        pos = self._make_positions([
            ("A", 1.0, -0.2),   # лучший ranker
            ("B", 1.0, -0.4),   # средний
            ("C", 1.0, -0.6),   # худший → к закрытию
        ])
        prices = {"A": 1.0, "B": 1.0, "C": 1.0}
        result = cg.prune_candidates(pos, prices, cache)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, "C")

    def test_prune_profit_protect(self):
        # C — худший по ranker, но прибыль > 2% → защита, не закрывать
        pairs = {
            ("A", "B"): 0.80, ("A", "C"): 0.80, ("B", "C"): 0.80,
        }
        cache = _make_cache(pairs)
        pos = self._make_positions([
            ("A", 1.0, -0.2),
            ("B", 1.0, -0.4),
            ("C", 1.0, -0.6),   # худший ranker, но +2.5% → защита
        ])
        # Цена C выросла на 2.5% от entry_price=1.0
        prices = {"A": 1.0, "B": 1.0, "C": 1.025}
        result = cg.prune_candidates(pos, prices, cache)
        # C защищён прибылью → pruned список пуст (или содержит B)
        pruned_syms = [r.symbol for r in result]
        self.assertNotIn("C", pruned_syms)

    def test_prune_disabled(self):
        _cfg.CORR_PRUNE_ENABLED = False
        pairs = {("A", "B"): 0.80, ("A", "C"): 0.80, ("B", "C"): 0.80}
        cache = _make_cache(pairs)
        pos = self._make_positions([("A", 1.0, -0.2), ("B", 1.0, -0.4), ("C", 1.0, -0.6)])
        result = cg.prune_candidates(pos, {"A": 1.0, "B": 1.0, "C": 1.0}, cache)
        self.assertEqual(len(result), 0)
        _cfg.CORR_PRUNE_ENABLED = True  # restore

    def test_prune_two_from_four(self):
        # Кластер из 4 → закрываем 2 худших, оставляем 2 лучших
        pairs = {
            ("A", "B"): 0.80, ("A", "C"): 0.80, ("A", "D"): 0.80,
            ("B", "C"): 0.80, ("B", "D"): 0.80, ("C", "D"): 0.80,
        }
        cache = _make_cache(pairs)
        pos = self._make_positions([
            ("A", 1.0, -0.1),
            ("B", 1.0, -0.3),
            ("C", 1.0, -0.5),  # к закрытию
            ("D", 1.0, -0.7),  # к закрытию
        ])
        prices = {s: 1.0 for s in "ABCD"}
        result = cg.prune_candidates(pos, prices, cache)
        self.assertEqual(len(result), 2)
        pruned = {r.symbol for r in result}
        self.assertEqual(pruned, {"C", "D"})


# ── get_or_create_cache ───────────────────────────────────────────────────────

class TestGetOrCreateCache(unittest.TestCase):

    def test_creates_new_cache(self):
        d = {}
        c = cg.get_or_create_cache(d)
        self.assertIsInstance(c, cg.CorrelationCache)
        self.assertIn("corr_cache", d)

    def test_returns_existing_cache(self):
        existing = cg.CorrelationCache()
        d = {"corr_cache": existing}
        c = cg.get_or_create_cache(d)
        self.assertIs(c, existing)


# ── format_cluster_report ─────────────────────────────────────────────────────

class TestFormatClusterReport(unittest.TestCase):

    def test_empty_positions(self):
        cache = cg.CorrelationCache()
        report = cg.format_cluster_report({}, cache)
        self.assertIn("позиций", report.lower())

    def test_single_position(self):
        pos = {"AAA": _FakePos("AAA", 1.0)}
        cache = cg.CorrelationCache()
        report = cg.format_cluster_report(pos, cache)
        self.assertIn("AAA", report)

    def test_clustered_positions(self):
        pairs = {("AAA", "BBB"): 0.80}
        cache = _make_cache(pairs)
        pos = {"AAA": _FakePos("AAA", 1.0), "BBB": _FakePos("BBB", 1.0)}
        report = cg.format_cluster_report(pos, cache)
        self.assertIn("AAA", report)
        self.assertIn("BBB", report)
        self.assertIn("Кластер", report)


# ── _effective_threshold ──────────────────────────────────────────────────────

class TestEffectiveThreshold(unittest.TestCase):

    def test_base(self):
        t = cg._effective_threshold()
        self.assertAlmostEqual(t, 0.65)

    def test_bull(self):
        t = cg._effective_threshold(is_bull_day=True)
        self.assertAlmostEqual(t, 0.60)

    def test_bear(self):
        t = cg._effective_threshold(is_bear_day=True)
        self.assertAlmostEqual(t, 0.99)

    def test_bull_bear_bear_wins(self):
        # bear overrides bull
        t = cg._effective_threshold(is_bull_day=True, is_bear_day=True)
        self.assertAlmostEqual(t, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
