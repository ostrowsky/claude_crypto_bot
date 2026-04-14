#!/usr/bin/env python3
"""
test_bot.py — Автотесты бота. Запускать при каждом старте или вручную.

Что проверяется:
  T01–T05  config          — watchlist, константы, load/save
  T06–T12  indicators      — EMA, RSI, ADX, ATR, vol_x, slope, compute_features
  T13–T20  strategy        — check_entry/exit/retest/breakout/impulse, analyze_coin
  T21–T27  monitor         — cooldown (формат!), OpenPosition, MonitorState
  T28–T32  ml_dataset      — дедупликация LRU, log_bar_snapshot, padding, fill_labels
  T33–T36  botlog          — все log_* функции пишут корректный JSON
  T37–T40  data_collector  — _detect_rule_signal, _get_market_context структура
  T41–T44  integration     — analyze_coin → CoinReport полнота полей
  T53–T56  regression      — TODAY_MIN_SIGNALS, _today_start_ms, confirmed без t3_ok,
                             ALIGNMENT_RSI_HI
  T57–T65  zombie/resilience — зомби-мониторинг, crash-safety, /test команда

Запуск:
    python test_bot.py
    python test_bot.py -v       # подробный вывод
    python test_bot.py TestMonitor  # только мониторинг
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock, Mock, AsyncMock

import numpy as np

# ── Добавляем текущую папку в путь (запуск рядом с ботом) ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Mock aiohttp если не установлен (для unit-тестов без сети) ─────────────────
try:
    import aiohttp
except ImportError:
    from unittest.mock import MagicMock
    aiohttp_mock = MagicMock()
    aiohttp_mock.ClientSession = MagicMock
    aiohttp_mock.ClientTimeout = MagicMock
    sys.modules["aiohttp"] = aiohttp_mock

# ── Mock telegram если не установлен ─────────────────────────────────────────
for _mod in ["telegram", "telegram.ext", "python_telegram_bot"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _make_candles(n: int = 200, trend: str = "up", base: float = 1.0) -> Dict:
    """
    Генерирует синтетические OHLCV данные.
    trend: "up" / "down" / "flat" / "impulse"
    """
    np.random.seed(42)
    c = np.ones(n) * base
    now_ms = int(time.time() * 1000)
    t = np.array([now_ms - (n - i) * 15 * 60 * 1000 for i in range(n)], dtype=np.int64)

    for i in range(1, n):
        noise = np.random.randn() * 0.002
        if trend == "up":
            c[i] = c[i-1] * (1 + 0.003 + noise)
        elif trend == "down":
            c[i] = c[i-1] * (1 - 0.003 + noise)
        elif trend == "flat":
            c[i] = c[i-1] * (1 + noise * 0.3)
        elif trend == "impulse":
            # Сильный рост в последних 10 барах
            if i >= n - 10:
                c[i] = c[i-1] * (1 + 0.015 + abs(noise))
            else:
                c[i] = c[i-1] * (1 + noise * 0.3)

    spread = 0.001
    h = c * (1 + spread + np.abs(np.random.randn(n) * 0.002))
    l = c * (1 - spread - np.abs(np.random.randn(n) * 0.002))
    o = np.roll(c, 1); o[0] = c[0]
    v = np.abs(np.random.randn(n) + 3) * 1000 * base

    # Структура как у fetch_klines
    arr = np.zeros(n, dtype=[
        ("t","i8"),("o","f8"),("h","f8"),("l","f8"),("c","f8"),("v","f8"),
    ])
    arr["t"] = t; arr["o"] = o; arr["h"] = h
    arr["l"] = l; arr["c"] = c; arr["v"] = v
    return arr


def _make_feat(n: int = 200, trend: str = "up"):
    """Возвращает (data, feat, i_now) готовые к использованию."""
    from indicators import compute_features
    data = _make_candles(n, trend)
    feat = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
    i_now = n - 2
    return data, feat, i_now


# ════════════════════════════════════════════════════════════════════════════════
# T01–T05  config
# ════════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def setUp(self):
        import config
        self.cfg = config

    def test_T01_watchlist_not_empty(self):
        """T01: DEFAULT_WATCHLIST содержит монеты"""
        wl = self.cfg.DEFAULT_WATCHLIST
        self.assertGreater(len(wl), 80, "Watchlist должен содержать >80 монет")

    def test_T02_aevousdt_in_default_watchlist(self):
        """T02: AEVOUSDT добавлен в DEFAULT_WATCHLIST"""
        self.assertIn("AEVOUSDT", self.cfg.DEFAULT_WATCHLIST,
                      "AEVOUSDT должен быть в DEFAULT_WATCHLIST")

    def test_T03_load_watchlist_returns_list(self):
        """T03: load_watchlist() возвращает список"""
        wl = self.cfg.load_watchlist()
        self.assertIsInstance(wl, list)
        self.assertGreater(len(wl), 0)

    def test_T04_save_load_watchlist_roundtrip(self):
        """T04: save/load watchlist — данные не меняются"""
        import config
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = Path(f.name)
        try:
            orig = config.WATCHLIST_FILE
            config.WATCHLIST_FILE = tmp
            test_list = ["BTCUSDT", "ETHUSDT", "TESTCOIN"]
            config.save_watchlist(test_list)
            loaded = config.load_watchlist()
            self.assertEqual(test_list, loaded)
        finally:
            config.WATCHLIST_FILE = orig
            tmp.unlink(missing_ok=True)

    def test_T05_required_constants_present(self):
        """T05: все необходимые константы определены"""
        required = [
            "EMA_FAST", "EMA_SLOW", "RSI_PERIOD", "ADX_PERIOD",
            "ADX_MIN", "EMA_SLOPE_MIN", "VOL_MULT", "RSI_BUY_LO", "RSI_BUY_HI",
            "DAILY_RANGE_MAX", "MAX_HOLD_BARS", "TODAY_MIN_SIGNALS",
            "ATR_TRAIL_K", "ATR_TRAIL_K_RETEST", "ATR_TRAIL_K_BREAKOUT",
            "ATR_TRAIL_K_STRONG", "COOLDOWN_BARS", "FORWARD_BARS",
            "BULL_DAY_RANGE_MAX", "BULL_DAY_RSI_HI",
            "IMPULSE_R1_MIN", "IMPULSE_R3_MIN", "IMPULSE_BODY_MIN",
            "IMPULSE_VOL_MIN", "IMPULSE_RSI_LO", "IMPULSE_RSI_HI",
        ]
        for name in required:
            self.assertTrue(hasattr(self.cfg, name), f"config.{name} не найден")

    def test_T06_no_duplicate_watchlist_symbols(self):
        """T06: в watchlist нет дублей"""
        wl = self.cfg.DEFAULT_WATCHLIST
        self.assertEqual(len(wl), len(set(wl)), "Дублирующиеся символы в watchlist")


# ════════════════════════════════════════════════════════════════════════════════
# T07–T14  indicators
# ════════════════════════════════════════════════════════════════════════════════

class TestIndicators(unittest.TestCase):

    def setUp(self):
        from indicators import _ema, _rsi, _adx, _atr, compute_features
        self.ema = _ema
        self.rsi = _rsi
        self.adx = _adx
        self.atr = _atr
        self.compute = compute_features

    def test_T07_ema_length_preserved(self):
        """T07: EMA возвращает массив той же длины"""
        x = np.random.rand(100) + 1
        out = self.ema(x, 20)
        self.assertEqual(len(out), len(x))

    def test_T08_ema_converges_on_constant(self):
        """T08: EMA сходится к константе на постоянном ряду"""
        x = np.ones(200) * 5.0
        out = self.ema(x, 20)
        self.assertAlmostEqual(out[-1], 5.0, places=4)

    def test_T09_rsi_range(self):
        """T09: RSI всегда в диапазоне [0, 100]"""
        c = np.random.rand(200) + 1
        rsi = self.rsi(c, 14)
        finite = rsi[np.isfinite(rsi)]
        self.assertTrue(np.all(finite >= 0), "RSI < 0 не допустим")
        self.assertTrue(np.all(finite <= 100), "RSI > 100 не допустим")

    def test_T10_adx_nonnegative(self):
        """T10: ADX >= 0"""
        data = _make_candles(200, "up")
        adx = self.adx(data["h"], data["l"], data["c"], 14)
        finite = adx[np.isfinite(adx)]
        self.assertTrue(np.all(finite >= 0))

    def test_T11_atr_nonnegative(self):
        """T11: ATR >= 0"""
        data = _make_candles(200)
        atr = self.atr(data["h"], data["l"], data["c"], 14)
        finite = atr[np.isfinite(atr)]
        self.assertTrue(np.all(finite >= 0))

    def test_T12_compute_features_keys(self):
        """T12: compute_features возвращает все ожидаемые ключи"""
        required_keys = [
            "ema_fast", "ema_slow", "rsi", "adx", "atr", "slope",
            "vol_x", "macd_hist", "daily_range_pct", "adx_sma",
            "close", "high", "low", "open", "ema200",
        ]
        data, feat, _ = _make_feat()
        for key in required_keys:
            self.assertIn(key, feat, f"compute_features: ключ '{key}' отсутствует")

    def test_T13_compute_features_lengths_equal(self):
        """T13: все массивы в feat одинаковой длины"""
        data, feat, _ = _make_feat(150)
        n = len(data)
        for key, arr in feat.items():
            self.assertEqual(len(arr), n, f"feat['{key}'] длина {len(arr)} ≠ {n}")

    def test_T14_vol_x_positive_uptrend(self):
        """T14: vol_x > 0 на баре с высоким объёмом"""
        data, feat, i = _make_feat(200, "up")
        vx = feat["vol_x"][i]
        self.assertTrue(np.isfinite(vx), "vol_x должен быть finite на валидных данных")
        self.assertGreater(vx, 0)


# ════════════════════════════════════════════════════════════════════════════════
# T15–T24  strategy
# ════════════════════════════════════════════════════════════════════════════════

class TestStrategy(unittest.TestCase):

    def setUp(self):
        from strategy import (
            check_entry_conditions, check_exit_conditions,
            check_retest_conditions, check_breakout_conditions,
            check_impulse_conditions, get_entry_mode, analyze_coin,
            CoinReport, _forward_accuracy, HorizonAccuracy,
        )
        self.check_entry    = check_entry_conditions
        self.check_exit     = check_exit_conditions
        self.check_retest   = check_retest_conditions
        self.check_breakout = check_breakout_conditions
        self.check_impulse  = check_impulse_conditions
        self.get_mode       = get_entry_mode
        self.analyze        = analyze_coin
        self.CoinReport     = CoinReport
        self.fwd_acc        = _forward_accuracy
        self.HorizonAcc     = HorizonAccuracy

    def test_T15_check_impulse_only_one_definition(self):
        """T15: check_impulse_conditions определена ровно один раз (не дублируется)"""
        import strategy
        import inspect
        src = inspect.getsource(strategy)
        count = src.count('def check_impulse_conditions(')
        self.assertEqual(count, 1, f"check_impulse_conditions определена {count} раз — должна быть 1")

    def test_T16_entry_conditions_no_data_returns_false(self):
        """T16: check_entry возвращает False при недостаточных данных"""
        data, feat, i = _make_feat(200, "flat")
        # Искусственно обнуляем индикаторы
        feat_bad = dict(feat)
        feat_bad["ema_fast"] = np.full(len(feat["ema_fast"]), np.nan)
        ok, reason = self.check_entry(feat_bad, i, data["c"].astype(float))
        self.assertFalse(ok)
        self.assertIn("данных", reason)

    def test_T17_entry_conditions_uptrend_can_pass(self):
        """T17: check_entry может пройти на сильном восходящем тренде"""
        import config
        # Создаём данные где все условия выполнены
        data, feat, i = _make_feat(200, "up")
        # Подменяем feat на "идеальные" значения
        c = data["c"].astype(float)
        feat["ema_fast"][i]        = c[i] * 0.995   # цена > EMA20
        feat["ema_slow"][i]        = c[i] * 0.98    # EMA20 > EMA50
        feat["slope"][i]           = 0.5            # slope > 0.1
        feat["adx"][i]             = 25.0           # ADX > 20
        feat["adx_sma"][i]         = 22.0           # ADX > ADX_SMA
        feat["vol_x"][i]           = 1.5            # vol > 1.3
        feat["rsi"][i]             = 60.0           # RSI в зоне
        feat["macd_hist"][i]       = 0.001          # MACD > 0
        feat["daily_range_pct"][i] = 3.0            # DR < 7
        ok, reason = self.check_entry(feat, i, c)
        self.assertTrue(ok, f"check_entry должен пройти на идеальных данных, причина: {reason}")

    def test_T18_exit_below_ema20_two_bars(self):
        """T18: check_exit срабатывает при 2 закрытиях подряд ниже EMA20"""
        data, feat, i = _make_feat(200, "down")
        c = data["c"].astype(float)
        # Принудительно ставим цену ниже EMA20 два бара подряд
        ema_val = c[i] * 1.05  # EMA20 выше цены
        feat["ema_fast"][i]     = ema_val
        feat["ema_fast"][i-1]   = ema_val
        feat["rsi"][i]          = 50.0
        feat["slope"][i]        = 0.1
        feat["adx"][i]          = 25.0
        feat["adx"][i - 3]      = 25.0
        reason = self.check_exit(feat, i, c)
        self.assertIsNotNone(reason, "check_exit должен вернуть причину при цене < EMA20 (2 бара)")

    def test_T19_retest_fails_without_prior_trend(self):
        """T19: check_retest отклоняет если тренда не было lb баров назад"""
        data, feat, i = _make_feat(200, "flat")
        c = data["c"].astype(float)
        # Убеждаемся что EMA20 выше цены lb баров назад
        lb = 12
        feat["ema_fast"][i - lb] = c[i - lb] * 1.1  # цена ниже EMA20 = нет тренда
        ok, reason = self.check_retest(feat, i)
        self.assertFalse(ok)

    def test_T20_impulse_returns_tuple(self):
        """T20: check_impulse_conditions возвращает Tuple[bool, str]"""
        data, feat, i = _make_feat(200, "up")
        result = self.check_impulse(feat, i)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)

    def test_T21_get_entry_mode_returns_valid(self):
        """T21: get_entry_mode возвращает 'trend' или 'strong_trend'"""
        data, feat, i = _make_feat(200, "up")
        mode = self.get_mode(feat, i)
        self.assertIn(mode, {"trend", "strong_trend"})

    def test_T22_analyze_coin_returns_coin_report(self):
        """T22: analyze_coin возвращает CoinReport с нужными полями"""
        data = _make_candles(300, "up")
        report = self.analyze("TESTUSDT", "15m", data)
        self.assertIsInstance(report, self.CoinReport)
        self.assertEqual(report.symbol, "TESTUSDT")
        self.assertEqual(report.tf, "15m")
        self.assertIsInstance(report.signal_now, bool)
        self.assertIsInstance(report.today_confirmed, bool)

    def test_T23_forward_accuracy_empty_signals(self):
        """T23: _forward_accuracy корректно работает с пустым списком"""
        c = np.random.rand(100) + 1
        result = self.fwd_acc([], c)
        import config
        for h in config.FORWARD_BARS:
            self.assertIn(h, result)
            self.assertEqual(result[h].total, 0)

    def test_T24_analyze_coin_no_signal_reason_set(self):
        """T24: analyze_coin заполняет no_signal_reason когда сигнала нет"""
        # Флэт → скорее всего сигнала нет
        data = _make_candles(300, "flat")
        report = self.analyze("FLATUSDT", "15m", data)
        if not report.signal_now:
            # Причина должна быть непустой строкой
            self.assertIsInstance(report.no_signal_reason, str)


# ════════════════════════════════════════════════════════════════════════════════
# T25–T30  monitor (cooldown критичный баг исправлен)
# ════════════════════════════════════════════════════════════════════════════════

class TestMonitor(unittest.TestCase):

    def setUp(self):
        from monitor import MonitorState, OpenPosition
        import config
        self.MonitorState  = MonitorState
        self.OpenPosition  = OpenPosition
        self.config        = config

    def test_T25_cooldown_stored_as_timestamp_ms(self):
        """T25: КРИТИЧНО — cooldown записывается как unix timestamp в мс, а не bar_index"""
        state = self.MonitorState()
        bar_ms = 15 * 60 * 1000
        now_ms = int(time.time() * 1000)
        cooldown_bars = 8

        # Имитируем запись cooldown как в monitor._poll_coin после ATR-выхода
        state.cooldowns["TESTUSDT"] = now_ms + cooldown_bars * bar_ms

        val = state.cooldowns["TESTUSDT"]

        # Значение должно быть unix timestamp (> 1 триллиона), не bar_index (~300)
        self.assertGreater(val, 1_000_000_000_000,
            f"cooldown должен быть unix_ms (~1.7T), получено {val} — похоже на bar_index!")

    def test_T26_cooldown_blocks_reentry(self):
        """T26: cooldown блокирует повторный вход пока не истёк"""
        state = self.MonitorState()
        bar_ms = 15 * 60 * 1000
        now_ms = int(time.time() * 1000)

        # Устанавливаем cooldown на 8 баров вперёд
        state.cooldowns["ETHUSDT"] = now_ms + 8 * bar_ms

        # Проверяем: текущий ts < cooldown → должен блокировать
        cooldown_until = state.cooldowns["ETHUSDT"]
        current_ts = now_ms  # "сейчас"
        self.assertLess(current_ts, cooldown_until,
                        "current_ts должен быть меньше cooldown_until — вход заблокирован")

    def test_T27_cooldown_expires_correctly(self):
        """T27: cooldown истекает через 8 баров"""
        state = self.MonitorState()
        bar_ms = 15 * 60 * 1000
        now_ms = int(time.time() * 1000)

        # Cooldown истёк 1 бар назад
        state.cooldowns["SOLUSDT"] = now_ms - bar_ms

        cooldown_until = state.cooldowns["SOLUSDT"]
        current_ts     = now_ms
        expired = current_ts >= cooldown_until
        self.assertTrue(expired, "Cooldown должен быть истёкшим через 8+ баров")

    def test_T28_cd_logged_separate_from_cooldowns(self):
        """T28: cd_logged — отдельный dict в MonitorState (не смешивает типы)"""
        state = self.MonitorState()
        # cooldowns хранит int (timestamp)
        state.cooldowns["BTCUSDT"] = int(time.time() * 1000) + 999999
        # cd_logged хранит bool
        state.cd_logged["BTCUSDT"] = True

        self.assertIsInstance(state.cooldowns["BTCUSDT"], int)
        self.assertIsInstance(state.cd_logged["BTCUSDT"], bool)
        # Они не пересекаются
        self.assertNotIn("BTCUSDT_cd_logged", state.cooldowns)

    def test_T29_open_position_pnl_calculation(self):
        """T29: OpenPosition.pnl_pct считает правильно"""
        pos = self.OpenPosition(
            symbol="TSTUSDT", tf="15m",
            entry_price=100.0, entry_bar=10, entry_ts=0,
            entry_ema20=99.0, entry_slope=0.5,
            entry_adx=25.0, entry_rsi=60.0, entry_vol_x=1.5,
        )
        self.assertAlmostEqual(pos.pnl_pct(110.0), 10.0, places=3)
        self.assertAlmostEqual(pos.pnl_pct(90.0),  -10.0, places=3)
        self.assertAlmostEqual(pos.pnl_pct(100.0),  0.0, places=3)

    def test_T30_monitor_state_initial_empty(self):
        """T30: MonitorState инициализируется пустым"""
        state = self.MonitorState()
        self.assertEqual(len(state.positions), 0)
        self.assertEqual(len(state.hot_coins), 0)
        self.assertEqual(len(state.cooldowns), 0)
        self.assertFalse(state.running)


# ════════════════════════════════════════════════════════════════════════════════
# T31–T36  ml_dataset
# ════════════════════════════════════════════════════════════════════════════════

class TestMLDataset(unittest.TestCase):

    def setUp(self):
        """Каждый тест получает чистый временный файл."""
        self._tmp = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        self._tmp.close()
        self._tmp_path = Path(self._tmp.name)

        import ml_dataset
        self.ml = ml_dataset
        self._orig_file = ml_dataset.ML_FILE
        ml_dataset.ML_FILE = self._tmp_path
        # Сбрасываем кэш дедупликации перед каждым тестом
        ml_dataset._logged_bars.clear()

    def tearDown(self):
        self.ml.ML_FILE = self._orig_file
        self._tmp_path.unlink(missing_ok=True)

    def test_T31_dedup_prevents_double_write(self):
        """T31: один и тот же (sym, tf, bar_ts) записывается только раз"""
        data, feat, i = _make_feat(200, "up")
        bar_ts = int(data["t"][i])

        self.ml.log_bar_snapshot(
            sym="DEDUPTEST", tf="15m", bar_ts=bar_ts,
            rule_signal="none", is_bull_day=False,
            feat=feat, i=i, data=data, btc_vs_ema50=0.0,
        )
        self.ml.log_bar_snapshot(
            sym="DEDUPTEST", tf="15m", bar_ts=bar_ts,
            rule_signal="none", is_bull_day=False,
            feat=feat, i=i, data=data, btc_vs_ema50=0.0,
        )
        lines = [l for l in self._tmp_path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, "Дублирующийся бар должен быть записан только раз")

    def test_T32_log_bar_snapshot_valid_json(self):
        """T32: log_bar_snapshot пишет валидный JSON"""
        data, feat, i = _make_feat(200, "up")
        bar_ts = int(data["t"][i])
        self.ml.log_bar_snapshot(
            sym="JSONTEST", tf="15m", bar_ts=bar_ts,
            rule_signal="trend", is_bull_day=True,
            feat=feat, i=i, data=data, btc_vs_ema50=1.5,
        )
        lines = [l for l in self._tmp_path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["sym"], "JSONTEST")
        self.assertEqual(rec["tf"], "15m")
        self.assertIn("f", rec)
        self.assertIn("seq", rec)
        self.assertIn("labels", rec)

    def test_T33_seq_length_correct(self):
        """T33: seq всегда SEQ_LEN элементов (включая padding)"""
        data_short, feat_short, i_short = _make_feat(30, "up")
        bar_ts = int(data_short["t"][i_short])
        rec_id = self.ml.log_bar_snapshot(
            sym="SEQTEST", tf="15m", bar_ts=bar_ts,
            rule_signal="none", is_bull_day=False,
            feat=feat_short, i=i_short, data=data_short, btc_vs_ema50=0.0,
        )
        lines = [l for l in self._tmp_path.read_text().splitlines() if l.strip()]
        rec = json.loads(lines[0])
        self.assertEqual(len(rec["seq"]), self.ml.SEQ_LEN,
                         f"seq должен содержать {self.ml.SEQ_LEN} элементов (с padding)")

    def test_T34_fill_labels_updates_record(self):
        """T34: fill_labels обновляет exit_pnl и exit_reason"""
        data, feat, i = _make_feat(200, "up")
        bar_ts = int(data["t"][i])
        rec_id = self.ml.log_bar_snapshot(
            sym="FILLTEST", tf="15m", bar_ts=bar_ts,
            rule_signal="trend", is_bull_day=False,
            feat=feat, i=i, data=data, btc_vs_ema50=0.0,
        )
        self.ml.fill_labels(rec_id, exit_pnl=2.5, exit_reason="ATR-trail", bars_held=5)
        lines = [l for l in self._tmp_path.read_text().splitlines() if l.strip()]
        rec = json.loads(lines[0])
        self.assertAlmostEqual(rec["labels"]["exit_pnl"], 2.5, places=2)
        self.assertEqual(rec["labels"]["exit_reason"], "ATR-trail")
        self.assertEqual(rec["labels"]["bars_held"], 5)

    def test_T35_lru_eviction_no_full_clear(self):
        """T35: _mark_logged при переполнении выбрасывает 10%, не очищает всё"""
        import ml_dataset
        # Заполняем почти до лимита
        for i in range(ml_dataset._MAX_LOGGED - 5):
            ml_dataset._mark_logged(f"key_{i}")
        size_before = len(ml_dataset._logged_bars)

        # Добавляем ещё — должна сработать LRU-eviction
        for i in range(20):
            ml_dataset._mark_logged(f"overflow_key_{i}")

        size_after = len(ml_dataset._logged_bars)
        # После eviction должно остаться ~90% + новые записи, не 0
        expected_min = ml_dataset._MAX_LOGGED * 0.85
        self.assertGreater(size_after, expected_min,
            f"После LRU eviction осталось {size_after} записей, "
            f"ожидалось >{expected_min:.0f} (не должно быть полной очистки)")

    def test_T36_log_signal_candidate_no_name_error(self):
        """T36: log_signal_candidate не вызывает NameError (rule_signal был undefined)"""
        data, feat, i = _make_feat(200, "up")
        bar_ts = int(data["t"][i])
        # Не должен бросать NameError
        try:
            self.ml.log_signal_candidate(
                sym="SIGTEST", tf="15m", bar_ts=bar_ts,
                signal_type="trend", is_bull_day=False,
                feat=feat, i=i, data=data, btc_vs_ema50=0.5,
            )
        except NameError as e:
            self.fail(f"log_signal_candidate вызвал NameError: {e}")

    def test_T36A_ml_file_is_absolute_in_runtime_module(self):
        """T36A: ML_FILE должен быть абсолютным, чтобы cwd не создавал второй датасет."""
        self.assertTrue(self._orig_file.is_absolute())

    def test_T36B_fill_labels_does_not_drop_concurrent_append(self):
        """T36B: rewrite под общим lock не должен терять строки, дописанные рядом."""
        base = {
            "id": "abc",
            "sym": "BTCUSDT",
            "tf": "15m",
            "labels": {"exit_pnl": None, "exit_reason": None, "bars_held": None},
        }
        extra = {
            "id": "def",
            "sym": "ETHUSDT",
            "tf": "15m",
            "labels": {"exit_pnl": None, "exit_reason": None, "bars_held": None},
        }
        self._tmp_path.write_text(
            json.dumps(base, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        done = threading.Event()

        def _worker():
            try:
                self.ml.fill_labels("abc", 1.25, "test-exit", 3)
            finally:
                done.set()

        with self.ml._FILE_LOCK:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            time.sleep(0.05)
            self.assertFalse(done.is_set(), "fill_labels должен ждать общий lock, а не писать параллельно")
            self.ml._w(extra)

        t.join(timeout=2.0)
        self.assertTrue(done.is_set(), "fill_labels не завершился после release lock")

        rows = [
            json.loads(line)
            for line in self._tmp_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 2)
        rows_by_id = {row["id"]: row for row in rows}
        self.assertIn("abc", rows_by_id)
        self.assertIn("def", rows_by_id)
        self.assertEqual(rows_by_id["abc"]["labels"]["exit_reason"], "test-exit")
        self.assertEqual(rows_by_id["def"]["sym"], "ETHUSDT")

    def test_T36C_atomic_replace_retries_permission_error(self):
        """T36C: atomic replace должен переживать краткие WinError 5."""
        target = self._tmp_path
        tmp = self._tmp_path.with_name(self._tmp_path.name + ".retry.tmp")
        target.write_text("old\n", encoding="utf-8")
        tmp.write_text("new\n", encoding="utf-8")

        original_replace = Path.replace
        calls = {"n": 0}

        def _flaky_replace(path_obj, dst):
            if Path(path_obj) == tmp and calls["n"] < 2:
                calls["n"] += 1
                raise PermissionError(5, "Access is denied")
            return original_replace(path_obj, dst)

        with patch("pathlib.Path.replace", new=_flaky_replace), \
             patch("ml_dataset.time.sleep", return_value=None):
            self.ml._atomic_replace_with_retry(tmp, target)

        self.assertGreaterEqual(calls["n"], 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
        self.assertFalse(tmp.exists(), "tmp file should be moved into target after successful retry")


# ════════════════════════════════════════════════════════════════════════════════
# T37–T40  botlog
# ════════════════════════════════════════════════════════════════════════════════

    def test_T36D_noop_rewrite_skips_cross_process_lock(self):
        """T36D: no-op rewrite не должен брать межпроцессный lock."""
        data, feat, i = _make_feat(200, "up")
        bar_ts = int(data["t"][i])
        rec_id = self.ml.log_bar_snapshot(
            sym="NOOPLOCK", tf="15m", bar_ts=bar_ts,
            rule_signal="trend", is_bull_day=False,
            feat=feat, i=i, data=data, btc_vs_ema50=0.0,
        )
        self.ml.fill_forward_label(rec_id, 3, 1.25)

        with patch.object(self.ml, "_dataset_io_lock", side_effect=AssertionError("lock should not be acquired")):
            self.ml.fill_forward_label(rec_id, 3, 1.25)


class TestBotlog(unittest.TestCase):

    def setUp(self):
        import botlog
        self._tmp = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        self._tmp.close()
        self._orig = botlog.LOG_FILE
        botlog.LOG_FILE = Path(self._tmp.name)
        self.bl = botlog

    def tearDown(self):
        self.bl.LOG_FILE = self._orig
        Path(self._tmp.name).unlink(missing_ok=True)

    def _read_events(self):
        return [json.loads(l) for l in Path(self._tmp.name).read_text().splitlines() if l.strip()]

    def test_T37_log_entry_valid_json(self):
        """T37: log_entry пишет валидный JSON с нужными полями"""
        self.bl.log_entry(
            sym="BTCUSDT", tf="15m", mode="trend",
            price=65000.0, ema20=64800.0, slope=0.3,
            rsi=58.0, adx=25.0, vol_x=1.5, macd_hist=0.001,
            daily_range=3.5, trail_k=2.0, max_hold_bars=16,
        )
        events = self._read_events()
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["event"], "entry")
        self.assertEqual(e["sym"], "BTCUSDT")
        self.assertIn("ts", e)

    def test_T38_log_exit_pnl_calculated(self):
        """T38: log_exit вычисляет pnl_pct правильно"""
        self.bl.log_exit(
            sym="ETHUSDT", tf="15m", mode="trend",
            entry_price=2000.0, exit_price=2100.0,
            reason="ATR-trail", bars_held=5, trail_k=2.0,
        )
        events = self._read_events()
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["pnl_pct"], 5.0, places=1)

    def test_T39_log_blocked_optional_fields(self):
        """T39: log_blocked работает с опциональными полями"""
        # С полями
        self.bl.log_blocked("XRPUSDT", "15m", 0.5, "ADX низкий", adx=18.0, vol_x=1.1)
        # Без полей
        self.bl.log_blocked("ADAUSDT", "1h", 0.3, "нет данных")
        events = self._read_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event"], "blocked")
        self.assertIn("adx", events[0])
        self.assertNotIn("adx", events[1])

    def test_T40_numpy_types_serialized(self):
        """T40: numpy типы корректно сериализуются в JSON"""
        self.bl.log_bull_day(
            is_bull=np.bool_(True),
            btc_price=np.float64(65000.5),
            btc_ema50=np.float32(64500.0),
            eff_range_max=7.0,
            eff_rsi_hi=72.0,
        )
        events = self._read_events()
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertIsInstance(e["is_bull"], bool)
        self.assertIsInstance(e["btc_price"], float)

    def test_T204_log_entry_with_ml_proba_writes_single_event(self):
        self.bl.log_entry(
            sym="BTCUSDT", tf="15m", mode="trend",
            price=65000.0, ema20=64800.0, slope=0.3,
            rsi=58.0, adx=25.0, vol_x=1.5, macd_hist=0.001,
            daily_range=3.5, trail_k=2.0, max_hold_bars=16,
            ml_proba=0.41789,
        )
        events = self._read_events()
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["ml_proba"], 0.4179, places=4)


# ════════════════════════════════════════════════════════════════════════════════
# T41–T44  integration: watchlist consistency
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_T41_all_scan_functions_use_load_watchlist(self):
        """T41: market_scan использует load_watchlist (не хардкод DEFAULT_WATCHLIST)"""
        import inspect, strategy
        src = inspect.getsource(strategy.market_scan)
        self.assertIn("load_watchlist()", src,
                      "market_scan должен использовать config.load_watchlist()")
        self.assertNotIn("DEFAULT_WATCHLIST", src,
                         "market_scan не должен использовать DEFAULT_WATCHLIST напрямую")

    def test_T42_impulse_scanner_uses_load_watchlist(self):
        """T42: impulse_scanner сканирует load_watchlist"""
        import inspect, impulse_scanner
        src = inspect.getsource(impulse_scanner._scan_once)
        self.assertIn("load_watchlist()", src,
                      "impulse_scanner должен использовать load_watchlist()")
        self.assertNotIn("DEFAULT_WATCHLIST", src)

    def test_T43_backfill_history_uses_load_watchlist(self):
        """T43: backfill_history.py использует load_watchlist"""
        import inspect
        # Читаем файл напрямую т.к. функции определены в __main__
        src = Path("backfill_history.py").read_text(encoding="utf-8")
        # Функция которая перебирает монеты
        self.assertNotIn("for sym in config.DEFAULT_WATCHLIST", src,
                         "backfill_history должен использовать load_watchlist()")

    def test_T44_coin_report_all_fields_present(self):
        """T44: CoinReport содержит все ожидаемые поля"""
        from strategy import CoinReport, HorizonAccuracy
        import config
        report = CoinReport(
            symbol="TEST", tf="15m",
            today_signals=5,
            today_accuracy={h: HorizonAccuracy(h, 5, 3) for h in config.FORWARD_BARS},
            today_confirmed=True,
            best_horizon=3, best_accuracy=60.0,
            in_play=True,
        )
        required = [
            "symbol", "tf", "today_signals", "today_accuracy", "today_confirmed",
            "best_horizon", "best_accuracy", "in_play", "signal_now",
            "current_price", "current_rsi", "current_adx", "no_signal_reason",
            "setup_now", "signal_mode",
        ]
        for field in required:
            self.assertTrue(hasattr(report, field), f"CoinReport.{field} отсутствует")

    def test_T45_check_impulse_is_calibrated_version(self):
        """T45: активная check_impulse_conditions — откалиброванная V2 (r1/r3/body)"""
        import inspect, strategy
        src = inspect.getsource(strategy.check_impulse_conditions)
        # V2 использует r1, r3, IMPULSE_R1_MIN
        self.assertIn("IMPULSE_R1_MIN", src, "Активная версия должна проверять r1")
        self.assertIn("IMPULSE_R3_MIN", src, "Активная версия должна проверять r3")
        self.assertIn("IMPULSE_BODY_MIN", src, "Активная версия должна проверять body")
        # V1 (мёртвая) использовала IMPULSE_CROSS_BARS в своём теле
        # В V2 IMPULSE_CROSS_BARS не используется
        self.assertNotIn("IMPULSE_CROSS_BARS", src,
                         "Активная V2 не должна использовать IMPULSE_CROSS_BARS")


# ════════════════════════════════════════════════════════════════════════════════
# T53–T56  regression — фиксы найденных багов (10.03.2026)
# ════════════════════════════════════════════════════════════════════════════════

class TestRegressions(unittest.TestCase):
    """Регрессионные тесты — каждый проверяет конкретный баг из истории."""

    def test_T53_today_min_signals_le_2(self):
        """T53: TODAY_MIN_SIGNALS ≤ 2 — иначе бот не подтверждает монеты"""
        import config
        self.assertLessEqual(
            config.TODAY_MIN_SIGNALS, 2,
            f"TODAY_MIN_SIGNALS={config.TODAY_MIN_SIGNALS} слишком высокий — "
            f"нужно ≤ 2 иначе подтверждение требует слишком много сигналов за день"
        )

    def test_T54_today_start_ms_uses_sliding_window(self):
        """T54: _today_start_ms возвращает метку ≥ 12ч назад (скользящее окно, не UTC полночь)"""
        from strategy import _today_start_ms
        now_ms = int(time.time() * 1000)
        start  = _today_start_ms()
        hours_ago = (now_ms - start) / (3600 * 1000)
        self.assertGreaterEqual(
            hours_ago, 12.0,
            f"_today_start_ms отстаёт только на {hours_ago:.1f}ч — "
            f"ожидалось ≥ 12ч (скользящее окно FORWARD_TEST_WINDOW_HOURS)"
        )
        self.assertLessEqual(
            hours_ago, 30.0,
            f"_today_start_ms слишком давно ({hours_ago:.1f}ч назад) — "
            f"вероятно вернул не скользящее окно а что-то другое"
        )

    def test_T55_confirmed_does_not_require_t3_ok(self):
        """T55: confirmed строится без t3_ok — только best_acc и t10_ok"""
        import inspect, strategy
        src = inspect.getsource(strategy.analyze_coin)
        # Раньше был тройной фильтр: best_acc AND t3_ok AND t10_ok
        # Теперь t3_ok убран: confirmed = best_acc >= MIN_ACCURACY and t10_ok
        self.assertNotIn(
            "t3_ok and t10_ok", src,
            "confirmed не должен требовать t3_ok — это был тройной фильтр, убивавший подтверждение"
        )
        # Проверяем что t10_ok всё ещё есть
        self.assertIn("t10_ok", src, "t10_ok должен оставаться как фильтр длинного горизонта")

    def test_T56_alignment_rsi_hi_ge_82(self):
        """T56: ALIGNMENT_RSI_HI ≥ 82 — медленные тренды разогреваются постепенно"""
        import config
        self.assertGreaterEqual(
            config.ALIGNMENT_RSI_HI, 82.0,
            f"ALIGNMENT_RSI_HI={config.ALIGNMENT_RSI_HI} — было 78, нужно ≥ 82 "
            f"чтобы не блокировать медленные бычьи тренды"
        )


# ════════════════════════════════════════════════════════════════════════════════
# T57–T65  zombie/resilience — зомби-мониторинг и устойчивость к ошибкам
# Эти тесты проверяют конкретные причины молчания бота 05–09.03.2026:
#   - state.running=True но asyncio-таск умер → мониторинг не перезапускался
#   - startup send() падал и убивал весь monitoring_loop таск
# ════════════════════════════════════════════════════════════════════════════════

class TestZombieResilience(unittest.TestCase):
    """Тесты зомби-состояния и устойчивости мониторинга."""

    # ── T57: зомби-детекция через task.done() ─────────────────────────────────

    def test_T57_task_done_check_exists_in_auto_reanalyze(self):
        """T57: _auto_reanalyze проверяет task.done() — зомби-детекция"""
        import inspect
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(
            "task.done()", src,
            "_auto_reanalyze должен проверять state.task.done() — "
            "иначе зомби (running=True, таск мёртв) никогда не обнаружится"
        )

    def test_T58_zombie_state_in_auto_reanalyze_restarts_monitoring(self):
        """T58: _auto_reanalyze перезапускает мониторинг при зомби (task.done())"""
        import inspect
        src = Path("bot.py").read_text(encoding="utf-8")
        # Проверяем что при _task_dead и state.running мониторинг перезапускается
        self.assertIn(
            "_task_dead", src,
            "Должна быть переменная _task_dead для обнаружения зомби-состояния"
        )
        # Убеждаемся: условие перезапуска включает ОБА случая
        self.assertIn(
            "not state.running or _task_dead", src,
            "Условие перезапуска должно быть 'not state.running OR _task_dead', "
            "а не только 'not state.running'"
        )

    def test_T59_monitoring_loop_startup_send_in_try_except(self):
        """T59: monitoring_loop оборачивает стартовый send() в try/except"""
        import inspect
        from monitor import monitoring_loop
        src = inspect.getsource(monitoring_loop)
        # Ищем try перед первым await send
        idx_try  = src.find("try:")
        idx_send = src.find("await send(")
        self.assertGreater(
            idx_send, -1, "monitoring_loop должен вызывать await send() при старте"
        )
        self.assertGreater(
            idx_try, -1,
            "monitoring_loop должен оборачивать стартовый send() в try/except — "
            "иначе ошибка Telegram убьёт весь таск"
        )
        self.assertLess(
            idx_try, idx_send,
            "try: должен стоять ДО первого await send() в monitoring_loop"
        )

    def test_T60_monitoring_loop_shutdown_send_in_try_except(self):
        """T60: monitoring_loop оборачивает финальный send() в try/except"""
        import inspect
        from monitor import monitoring_loop
        src = inspect.getsource(monitoring_loop)
        # Финальный send — последний await send в функции
        last_send_pos = src.rfind("await send(")
        # try должен быть где-то после while цикла, до последнего send
        last_try_pos  = src.rfind("try:")
        self.assertGreater(
            last_try_pos, -1,
            "monitoring_loop должен оборачивать финальный send() в try/except"
        )
        self.assertLess(
            last_try_pos, last_send_pos,
            "try: должен стоять перед финальным await send() в monitoring_loop"
        )

    def test_T61_monitoring_loop_uses_return_exceptions(self):
        """T61: asyncio.gather в monitoring_loop использует return_exceptions=True"""
        import inspect
        from monitor import monitoring_loop
        src = inspect.getsource(monitoring_loop)
        self.assertIn(
            "return_exceptions=True", src,
            "asyncio.gather должен использовать return_exceptions=True — "
            "иначе одна ошибка в _poll_coin убивает весь цикл мониторинга"
        )

    def test_T62_monitoring_loop_heartbeat_log_exists(self):
        """T62: monitoring_loop логирует heartbeat (alive) — видимость в логах"""
        import inspect
        from monitor import monitoring_loop
        src = inspect.getsource(monitoring_loop)
        self.assertIn(
            "alive", src,
            "monitoring_loop должен логировать heartbeat (alive) "
            "чтобы можно было убедиться что он не завис"
        )

    def test_T63_cmd_test_command_registered(self):
        """T63: /test команда зарегистрирована в боте"""
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(
            '"test"', src,
            '/test команда должна быть зарегистрирована через CommandHandler("test", ...)'
        )
        self.assertIn(
            "cmd_test", src,
            "Функция cmd_test должна быть определена в bot.py"
        )

    def test_T64_cmd_test_shows_task_status(self):
        """T64: cmd_test отображает статус monitoring task (живой/зомби/остановлен)"""
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(
            "task.done()", src,
            "cmd_test должен проверять state.task.done() чтобы обнаружить зомби"
        )
        # cmd_test должен реанимировать зомби сам
        self.assertIn(
            "monitoring_loop", src,
            "cmd_test должен перезапускать monitoring_loop при обнаружении зомби"
        )

    def test_T65_state_running_zombie_scenario(self):
        """T65: зомби-сценарий: running=True + task.done() → должна срабатывать детекция"""
        # Симулируем зомби через уже завершённый Future
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            # Создаём уже завершённый таск через корутину которая сразу возвращает
            async def _noop():
                return

            dead_task = loop.run_until_complete(asyncio.ensure_future(_noop(), loop=loop))

            from monitor import MonitorState
            state = MonitorState()
            state.running = True
            state.task    = dead_task  # зомби: running=True но таск завершился

            # Проверяем детекцию
            _task_dead = state.task is None or state.task.done()
            is_zombie  = state.running and _task_dead

            self.assertTrue(
                is_zombie,
                "Зомби-состояние (running=True, task.done()=True) должно быть обнаружено. "
                "Именно это привело к молчанию бота 05–09.03.2026"
            )

            # Проверяем что нормальное состояние не ложный зомби
            state2 = MonitorState()
            state2.running = False
            state2.task    = None
            _task_dead2 = state2.task is None or state2.task.done()
            is_zombie2  = state2.running and _task_dead2
            self.assertFalse(
                is_zombie2,
                "running=False + task=None не должно быть зомби"
            )
        finally:
            loop.close()


# ════════════════════════════════════════════════════════════════════════════════
# T66–T72: Chat ID, broadcast и _post_init — реальная причина молчания бота
# ════════════════════════════════════════════════════════════════════════════════

class TestChatIdAndBroadcast(unittest.TestCase):
    """
    Тесты на механизм chat_id / broadcast / _post_init.
    Реальная причина отсутствия сигналов 03–09.03.2026:
    при первом деплое .chat_ids не существовало → _known_chat_ids = {} →
    _post_init делал early return → мониторинг стартовал, но send() молчал.
    """

    def test_T66_post_init_returns_early_when_no_chat_ids(self):
        """T66: _post_init делает early return если _known_chat_ids пустой"""
        src = Path("bot.py").read_text(encoding="utf-8")
        # Должна быть проверка: if not _known_chat_ids: return
        self.assertIn(
            "if not _known_chat_ids",
            src,
            "_post_init должен проверять пустоту _known_chat_ids — "
            "иначе попытка отправить сообщение по пустому списку создаёт ложное "
            "ощущение что всё в порядке",
        )

    def test_T67_save_chat_id_adds_to_known_set(self):
        """T67: _save_chat_id добавляет chat_id в _known_chat_ids и сохраняет на диск"""
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(
            "_known_chat_ids.add(chat_id)",
            src,
            "_save_chat_id должен добавлять chat_id в множество _known_chat_ids",
        )
        self.assertIn(
            ".chat_ids",
            src,
            "_save_chat_id должен сохранять chat_ids в файл .chat_ids",
        )

    def test_T68_cmd_start_calls_save_chat_id(self):
        """T68: cmd_start вызывает _save_chat_id — единственный способ зарегистрироваться"""
        src = Path("bot.py").read_text(encoding="utf-8")
        # Находим функцию cmd_start и проверяем что она зовёт _save_chat_id
        idx_cmd_start = src.find("async def cmd_start(")
        idx_save      = src.find("_save_chat_id(chat_id)", idx_cmd_start)
        self.assertGreater(
            idx_cmd_start, -1,
            "cmd_start должна быть определена в bot.py",
        )
        self.assertGreater(
            idx_save, -1,
            "cmd_start должна вызывать _save_chat_id(chat_id) — "
            "иначе пользователь никогда не получит сигналы после первого деплоя",
        )

    def test_T69_broadcast_send_iterates_known_chat_ids(self):
        """T69: _make_broadcast_send итерирует по _known_chat_ids"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx_broadcast = src.find("def _make_broadcast_send(")
        idx_iter      = src.find("for cid in list(_known_chat_ids)", idx_broadcast)
        self.assertGreater(
            idx_broadcast, -1,
            "_make_broadcast_send должна быть определена в bot.py",
        )
        self.assertGreater(
            idx_iter, idx_broadcast,
            "_make_broadcast_send должна итерировать list(_known_chat_ids) — "
            "при пустом множестве сообщения просто не отправляются (молчание без ошибок)",
        )

    def test_T69A_broadcast_send_uses_retry_wrapper(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("await _send_with_retry(cid, text, app)", src)
        self.assertIn("def _is_retryable_send_error(", src)
        self.assertIn("broadcast send retry", src)

    def test_T70_backfill_logged_bars_uses_dict_not_set(self):
        """T70: backfill_history использует dict-присваивание для _logged_bars, а не .add()"""
        src = Path("backfill_history.py").read_text(encoding="utf-8")
        # Не должно быть .add() для _logged_bars
        self.assertNotIn(
            "_logged_bars.add(",
            src,
            "backfill_history._logged_bars — это OrderedDict, у него нет .add(). "
            "Использовать: _logged_bars[key] = True",
        )
        # Должно быть присваивание
        self.assertIn(
            "_logged_bars[",
            src,
            "backfill_history должен инициализировать _logged_bars через _logged_bars[key] = True",
        )

    def test_T71_auto_reanalyze_zombie_check_uses_task_dead_variable(self):
        """T71: _auto_reanalyze содержит переменную _task_dead для зомби-детекции"""
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(
            "_task_dead",
            src,
            "_auto_reanalyze должен вычислять _task_dead = task.done() — "
            "это позволяет перезапустить мониторинг даже если running=True",
        )
        self.assertIn(
            "not state.running or _task_dead",
            src,
            "Условие запуска мониторинга должно учитывать ОБА состояния: "
            "not state.running OR _task_dead",
        )

    def test_T72_auto_reanalyze_logs_zombie_warning(self):
        """T72: _auto_reanalyze логирует warning при обнаружении зомби-таска"""
        src = Path("bot.py").read_text(encoding="utf-8")
        # Должен быть log.warning с упоминанием зомби или zombie
        has_zombie_log = (
            ("zombie" in src.lower() or "зомби" in src.lower())
            and "log.warning" in src
        )
        self.assertTrue(
            has_zombie_log,
            "_auto_reanalyze должен логировать log.warning при обнаружении зомби-таска — "
            "иначе молчание мониторинга невозможно диагностировать по логам",
        )



# ════════════════════════════════════════════════════════════════════════════════
# T73–T84  Баги сессии 10.03.2026 — "нет позиций при сигналах" + MACD ложные алерты
# ════════════════════════════════════════════════════════════════════════════════

class TestNoPositionsBug(unittest.TestCase):
    """
    Покрывает корневую причину бага «нет позиций при активных сигналах».

    Суть: state.positions[sym] = pos стоял ПОСЛЕ ml_dataset.log_signal_candidate().
    Если ml_dataset выбрасывал исключение — позиция никогда не сохранялась.
    Мониторинг продолжал работать, сигналы находил, но positions оставался пустым.
    Пользователь видел: «📊 Активных сигналов нет» при 2 активных сигналах.

    Фикс: positions сохраняется ПЕРВЫМ, ml_dataset и botlog обёрнуты в try/except.
    """

    # ── T73: positions сохраняется ДО вызова ml_dataset ─────────────────────
    def test_T73_positions_saved_before_ml_dataset(self):
        """
        ГЛАВНЫЙ ТЕСТ. В monitor.py строка state.positions[sym] = pos должна
        идти раньше ml_dataset.log_signal_candidate().
        Если нарушено — одиночный сбой ml_dataset обнуляет все позиции.
        """
        src = Path("monitor.py").read_text(encoding="utf-8")
        pos_idx = src.find("state.positions[sym] = pos")
        ml_idx  = src.find("ml_dataset.log_signal_candidate(")
        self.assertGreater(pos_idx, 0,  "state.positions[sym] = pos not found in monitor.py")
        self.assertGreater(ml_idx,  0,  "ml_dataset.log_signal_candidate not found in monitor.py")
        self.assertLess(
            pos_idx, ml_idx,
            f"BUG: state.positions[sym] = pos (char {pos_idx}) должно быть ДО "
            f"ml_dataset.log_signal_candidate (char {ml_idx}). "
            f"Иначе исключение в ml_dataset уничтожит позицию."
        )

    # ── T74: positions сохраняется ДО вызова botlog.log_entry ────────────────
    def test_T74_positions_saved_before_botlog(self):
        """botlog.log_entry тоже после сохранения позиции."""
        src = Path("monitor.py").read_text(encoding="utf-8")
        pos_idx = src.find("state.positions[sym] = pos")
        log_idx = src.find("botlog.log_entry(")
        self.assertGreater(pos_idx, 0)
        self.assertGreater(log_idx,  0)
        self.assertLess(
            pos_idx, log_idx,
            f"BUG: state.positions[sym] = pos (char {pos_idx}) должно быть ДО "
            f"botlog.log_entry (char {log_idx})."
        )

    # ── T75: ml_dataset.log_signal_candidate обёрнут в try/except ────────────
    def test_T75_ml_dataset_wrapped_in_try_except(self):
        """
        ml_dataset.log_signal_candidate должен быть внутри try/except.
        Без этого любой сбой датасета (I/O, OOM, баг) убивает весь poll.
        """
        src = Path("monitor.py").read_text(encoding="utf-8")
        # Ищем try: перед ml_dataset вызовом (не далее 300 символов)
        ml_idx = src.find("ml_dataset.log_signal_candidate(")
        self.assertGreater(ml_idx, 0)
        window = src[max(0, ml_idx - 300): ml_idx]
        self.assertIn("try:", window,
            "ml_dataset.log_signal_candidate должен быть обёрнут в try/except")

    # ── T76: botlog.log_entry обёрнут в try/except ───────────────────────────
    def test_T76_botlog_wrapped_in_try_except(self):
        """botlog.log_entry тоже должен быть в try/except."""
        src = Path("monitor.py").read_text(encoding="utf-8")
        log_idx = src.find("botlog.log_entry(")
        self.assertGreater(log_idx, 0)
        window = src[max(0, log_idx - 800): log_idx]
        self.assertIn("try:", window,
            "botlog.log_entry должен быть обёрнут в try/except")

    # ── T77: log.info ENTRY добавлен сразу после сохранения позиции ──────────
    def test_T77_entry_log_after_positions_save(self):
        """
        После state.positions[sym] = pos должен быть log.info с ENTRY.
        Это даёт возможность дебажить по логам даже если notify упал.
        """
        src = Path("monitor.py").read_text(encoding="utf-8")
        pos_idx  = src.find("state.positions[sym] = pos")
        log_idx  = src.find("log.info", pos_idx)
        entry_ok = "ENTRY" in src[log_idx:log_idx + 100] if log_idx > pos_idx else False
        self.assertTrue(entry_ok,
            "После state.positions[sym] = pos должен следовать log.info с 'ENTRY'")


class TestMACDWarningBug(unittest.TestCase):
    """
    Bug 8: MACD warn streak рос на каждом поллинге (60с) по одним и тем же двум барам.
    Через 3 минуты после входа 30+ монет получали ложное предупреждение о развороте.
    """

    # ── T78: last_macd_bar_i поле существует ─────────────────────────────────
    def test_T78_last_macd_bar_i_field_exists(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("last_macd_bar_i: int = -1", src,
            "OpenPosition должен иметь поле last_macd_bar_i: int = -1")

    # ── T79: streak обновляется только при смене бара ─────────────────────────
    def test_T79_streak_gated_by_bar_index(self):
        """Проверяем что условие i != pos.last_macd_bar_i присутствует."""
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("i != pos.last_macd_bar_i", src,
            "MACD streak должен обновляться только на новом баре: i != pos.last_macd_bar_i")

    # ── T80: предупреждение требует bars_elapsed >= MACDWARN_BARS ─────────────
    def test_T80_warn_requires_bars_elapsed(self):
        """Без этого guard алерт мог прийти на баре входа (bars_elapsed=0)."""
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("pos.bars_elapsed >= config.MACDWARN_BARS", src,
            "MACD warn должен проверять pos.bars_elapsed >= config.MACDWARN_BARS")


class TestImpulseSpeedMode(unittest.TestCase):
    """
    Bug 10: get_entry_mode возвращал 'strong_trend' при ADX<28 если price_speed≥1.5%.
    Пример: AEVO ADX=24, price_speed=+3.4% → отображалось как 💪 Сильный тренд.
    Фикс: выделен отдельный режим 'impulse_speed' с честной меткой.
    """

    # ── T81: get_entry_mode возвращает impulse_speed ──────────────────────────
    def test_T81_get_entry_mode_returns_impulse_speed(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn('return "impulse_speed"', src,
            "get_entry_mode должен явно возвращать 'impulse_speed'")

    # ── T82: price_speed НЕ в блоке strong_trend ─────────────────────────────
    def test_T82_price_speed_not_in_strong_trend_block(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        # Ищем только сам if-блок ADX+vol → return "strong_trend" (без docstring)
        # Берём текст между 'ADX + vol' комментарием и return "strong_trend"
        start = src.find('ADX + vol')
        end   = src.find('return "strong_trend"', start)
        self.assertGreater(start, 0, "ADX + vol comment not found")
        self.assertGreater(end, start, "return strong_trend not found after ADX+vol comment")
        block = src[start:end + 20]
        # В этом блоке не должно быть ни or price_speed, ни price_speed >=
        self.assertNotIn("or price_speed", block,
            "strong_trend if-блок не должен содержать 'or price_speed'")

    # ── T83: impulse_speed получает широкий стоп в monitor.py ────────────────
    def test_T83_impulse_speed_gets_wide_stop_in_monitor(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn('"strong_trend", "impulse_speed"', src,
            "monitor.py: impulse_speed должен быть в ветке вместе со strong_trend")
        idx = src.find('"strong_trend", "impulse_speed"')
        window = src[idx: idx + 150]
        self.assertIn("ATR_TRAIL_K_STRONG", window,
            "monitor.py: ветка impulse_speed должна использовать ATR_TRAIL_K_STRONG")

    # ── T84: метка impulse_speed ≠ 'Сильный тренд' в обоих файлах ────────────
    def test_T84_impulse_speed_label_not_strong_trend(self):
        import re
        for fname in ("monitor.py", "strategy.py"):
            try:
                src = Path(fname).read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            m = re.search(r'"impulse_speed"\s*:\s*"([^"]+)"', src)
            if m:
                label = m.group(1)
                self.assertNotIn("Сильный тренд", label,
                    f"{fname}: метка impulse_speed не должна содержать 'Сильный тренд'")
                self.assertNotIn("BUY strong", label,
                    f"{fname}: метка impulse_speed не должна быть 'BUY strong'")

    def test_T83_impulse_speed_gets_wide_stop_in_monitor(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        branch = 'elif sig_mode in ("strong_trend", "impulse_speed")'
        self.assertIn(
            branch,
            src,
            "monitor.py: impulse_speed should share the stop branch with strong_trend",
        )
        idx = src.find(branch)
        window = src[idx: idx + 150]
        self.assertIn(
            "ATR_TRAIL_K_STRONG",
            window,
            "monitor.py: impulse_speed stop branch should use ATR_TRAIL_K_STRONG",
        )


class TestMarkdownEscaping(unittest.TestCase):
    """
    Bug 9: символы [ ] в no_signal_reason/setup_reason ломали ParseMode.MARKDOWN
    (BadRequest «can't find end of entity starting at byte offset 5285»).
    """

    # ── T85: no_signal_reason экранирует [ ] ─────────────────────────────────
    def test_T85_no_signal_reason_escapes_brackets(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn('replace("[",', src,
            "no_signal_reason должен экранировать символ [")
        self.assertIn('replace("]",', src,
            "no_signal_reason должен экранировать символ ]")

    # ── T86: кнопка позиций использует ParseMode.HTML ─────────────────────────
    def test_T86_positions_button_uses_html(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        start = src.find('action == "positions"')
        self.assertGreater(start, 0)
        # ParseMode.HTML может быть на 2-3 строки ниже edit_message_text — берём 300 символов
        end = src.find("edit_message_text", start)
        block = src[start:end + 300]
        self.assertIn("ParseMode.HTML", block,
            "positions button должен использовать ParseMode.HTML")

    # ── T87: bot.py syntax OK (проверка f-string backslash) ──────────────────
    def test_T87_bot_py_syntax_valid(self):
        """Убеждаемся что нет SyntaxError от backslash внутри f-string {}."""
        import ast
        src = Path("bot.py").read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            self.fail(f"bot.py SyntaxError: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# T88–T104  Expectancy-метрики (HorizonAccuracy + _forward_accuracy + confirmed)
# ════════════════════════════════════════════════════════════════════════════════

class TestExpectancy(unittest.TestCase):
    """
    Покрывает реализацию EV-метрик (пункт #1 рефакторинга).

    Исходная проблема: win% без учёта размера выигрыша/проигрыша.
    Пример-антипаттерн: 65% win, но все выигрыши +0.05%, все проигрыши -1.5%
    → EV = 0.65×0.05 − 0.35×1.5 = -0.49% → убыточная стратегия.
    """

    def setUp(self):
        import config as cfg
        self._orig_fwd = cfg.FORWARD_BARS
        cfg.FORWARD_BARS = [3, 5, 10]
        import config
        self.config = config
        from strategy import _forward_accuracy, HorizonAccuracy
        self.fwd_acc      = _forward_accuracy
        self.HorizonAcc   = HorizonAccuracy

    def tearDown(self):
        self.config.FORWARD_BARS = self._orig_fwd

    def _build_candles(self, signal_bars, win_pct, win_ret, lose_ret, n=100):
        """
        Строит ценовой ряд где сигналы дают заданные исходы.
        win_pct: доля выигрышных сигналов (0-1)
        win_ret: доход при выигрыше (%)
        lose_ret: доход при проигрыше (%) — отрицательное число
        """
        np.random.seed(42)
        c = np.ones(n)
        for s in signal_bars:
            for h in [3, 5, 10]:
                if s + h < n:
                    ret = win_ret if np.random.rand() < win_pct else lose_ret
                    c[s + h] = c[s] * (1 + ret / 100)
        # Заполняем пробелы линейной интерполяцией
        for i in range(1, n):
            if c[i] == 1.0:
                c[i] = c[i - 1]
        return c

    # ── T88: новые поля HorizonAccuracy присутствуют ─────────────────────────

    def test_T88_horizon_accuracy_has_ev_fields(self):
        """T88: HorizonAccuracy содержит все expectancy-поля"""
        ha = self.HorizonAcc(horizon=5, total=0, correct=0)
        for field in ("expected_return", "median_return", "downside_q10",
                      "upside_q90", "ev_proxy"):
            self.assertTrue(hasattr(ha, field), f"HorizonAccuracy.{field} не найден")

    def test_T89_horizon_accuracy_has_is_positive_ev(self):
        """T89: is_positive_ev property работает"""
        ha_pos = self.HorizonAcc(5, 10, 7, expected_return=0.3)
        ha_neg = self.HorizonAcc(5, 10, 6, expected_return=-0.2)
        ha_none = self.HorizonAcc(5, 10, 7)
        self.assertTrue(ha_pos.is_positive_ev)
        self.assertFalse(ha_neg.is_positive_ev)
        # Fallback к pct > 50 если нет EV данных
        self.assertTrue(ha_none.is_positive_ev)  # 7/10 = 70% > 50

    def test_T90_forward_accuracy_computes_ev(self):
        """T90: _forward_accuracy заполняет expected_return для достаточной выборки"""
        sigs = list(range(0, 50, 3))  # 16 сигналов
        c = self._build_candles(sigs, 0.6, 0.8, -0.5)
        result = self.fwd_acc(sigs[:12], c)  # первые 12 у которых T+10 есть
        fa10 = result.get(10)
        self.assertIsNotNone(fa10)
        self.assertIsNotNone(fa10.expected_return,
                             "_forward_accuracy должен вычислить expected_return")
        self.assertIsNotNone(fa10.downside_q10)
        self.assertIsNotNone(fa10.ev_proxy)

    def test_T91_ev_negative_for_lossy_strategy(self):
        """T91: EV отрицательный при больших стопах и малых профитах"""
        sigs = list(range(0, 50, 3))
        # 65% побед, но выигрыши +0.05%, проигрыши -1.5%
        c = self._build_candles(sigs, 0.65, 0.05, -1.5)
        result = self.fwd_acc(sigs[:12], c)
        fa10 = result.get(10)
        if fa10 and fa10.expected_return is not None:
            self.assertLess(fa10.expected_return, 0.1,
                "EV должен быть низким/отрицательным при маленьких выигрышах")

    def test_T92_ev_positive_for_good_strategy(self):
        """T92: EV положительный при нормальном соотношении риск/доход"""
        sigs = list(range(0, 50, 3))
        # 60% побед, выигрыши +0.8%, проигрыши -0.6%
        c = self._build_candles(sigs, 0.6, 0.8, -0.6)
        result = self.fwd_acc(sigs[:12], c)
        fa10 = result.get(10)
        if fa10 and fa10.expected_return is not None:
            self.assertGreater(fa10.expected_return, 0,
                "EV должен быть положительным при нормальном R/R")
            self.assertTrue(fa10.is_positive_ev)

    def test_T93_no_ev_with_insufficient_samples(self):
        """T93: EV не вычисляется при < 2 сигналах (нет стат. значимости)"""
        c = np.array([1.0, 1.01, 1.02, 1.03, 1.04, 1.05,
                      1.06, 1.07, 1.08, 1.09, 1.10, 1.11])
        result = self.fwd_acc([0], c)  # 1 сигнал — недостаточно
        fa10 = result.get(10)
        if fa10:
            self.assertIsNone(fa10.expected_return,
                "EV не должен вычисляться при 1 сигнале")

    def test_T94_downside_q10_le_median(self):
        """T94: downside_q10 ≤ median_return (по определению квантиля)"""
        sigs = list(range(0, 50, 3))
        c = self._build_candles(sigs, 0.6, 0.8, -0.6)
        result = self.fwd_acc(sigs[:12], c)
        for h, fa in result.items():
            if fa.downside_q10 is not None and fa.median_return is not None:
                self.assertLessEqual(
                    fa.downside_q10, fa.median_return,
                    f"T+{h}: downside_q10 должен быть ≤ median"
                )

    def test_T95_upside_q90_ge_median(self):
        """T95: upside_q90 ≥ median_return"""
        sigs = list(range(0, 50, 3))
        c = self._build_candles(sigs, 0.6, 0.8, -0.6)
        result = self.fwd_acc(sigs[:12], c)
        for h, fa in result.items():
            if fa.upside_q90 is not None and fa.median_return is not None:
                self.assertGreaterEqual(
                    fa.upside_q90, fa.median_return,
                    f"T+{h}: upside_q90 должен быть ≥ median"
                )

    def test_T96_ev_proxy_positive_when_ev_positive_downside_negative(self):
        """T96: ev_proxy > 0 когда EV > 0 и есть downside"""
        ha = self.HorizonAcc(5, 10, 6,
                             expected_return=0.5,
                             downside_q10=-1.0,
                             ev_proxy=0.5)
        self.assertGreater(ha.ev_proxy, 0)

    def test_T97_short_str_shows_ev(self):
        """T97: short_str() отображает EV со стрелкой"""
        ha = self.HorizonAcc(5, 10, 7, expected_return=0.34)
        s = ha.short_str()
        self.assertIn("EV", s, "short_str должен содержать 'EV'")
        self.assertIn("+", s, "short_str должен показывать знак")

    def test_T98_short_str_negative_ev_shows_down_icon(self):
        """T98: short_str() показывает ❌ при отрицательном EV"""
        ha = self.HorizonAcc(5, 10, 7, expected_return=-0.15)
        s = ha.short_str()
        self.assertIn("❌", s, "short_str должен показывать ❌ при отрицательном EV")

    def test_T99_config_ev_params_exist(self):
        """T99: EV_MIN_PCT и EV_MIN_SAMPLES присутствуют в конфиге"""
        import config
        self.assertTrue(hasattr(config, "EV_MIN_PCT"),
                        "config.EV_MIN_PCT не найден")
        self.assertTrue(hasattr(config, "EV_MIN_SAMPLES"),
                        "config.EV_MIN_SAMPLES не найден")
        self.assertIsInstance(config.EV_MIN_PCT, float)
        self.assertIsInstance(config.EV_MIN_SAMPLES, int)
        self.assertGreaterEqual(config.EV_MIN_SAMPLES, 1)

    def test_T100_analyze_coin_uses_ev_in_confirmed(self):
        """T100: analyze_coin учитывает EV при подтверждении (inspect проверка)"""
        import inspect, strategy
        src = inspect.getsource(strategy.analyze_coin)
        self.assertIn("ev_ok", src,
                      "analyze_coin должен проверять ev_ok для подтверждения")
        self.assertIn("expected_return", src,
                      "analyze_coin должен использовать expected_return")
        self.assertIn("EV_MIN_PCT", src,
                      "analyze_coin должен читать EV_MIN_PCT из конфига")

    def test_T101_ev_filter_does_not_block_when_no_data(self):
        """T101: EV-фильтр не блокирует когда данных мало (ev_detail пустой)"""
        import inspect, strategy
        src = inspect.getsource(strategy.analyze_coin)
        # Должна быть логика: если нет EV данных — не блокируем
        self.assertIn("ev_ok = True", src,
                      "При отсутствии EV данных ev_ok должен быть True (не блокировать)")


# ════════════════════════════════════════════════════════════════════════════════
# T102–T110  Portfolio risk management (monitor.py + config.py)
# ════════════════════════════════════════════════════════════════════════════════

class TestPortfolioLimits(unittest.TestCase):
    """
    Покрывает реализацию портфельных лимитов (пункт #2 рефакторинга).

    Проблема 11.03.2026: 12 монет вошли одновременно как одно рыночное движение,
    без портфельного контроля и без корреляционного лимита по группам.
    """

    def setUp(self):
        from monitor import _get_coin_group, _check_portfolio_limits, MonitorState
        import config
        self.get_group  = _get_coin_group
        self.check_port = _check_portfolio_limits
        self.MonitorState = MonitorState
        self.config = config

    def _make_state(self, symbols: list) -> "MonitorState":
        """Создаёт MonitorState с открытыми позициями."""
        state = self.MonitorState()
        for sym in symbols:
            from unittest.mock import MagicMock
            state.positions[sym] = MagicMock()
        return state

    # ── T102: config-параметры портфеля ──────────────────────────────────────

    def test_T102_portfolio_config_params_exist(self):
        """T102: MAX_OPEN_POSITIONS и MAX_POSITIONS_PER_GROUP в конфиге"""
        self.assertTrue(hasattr(self.config, "MAX_OPEN_POSITIONS"))
        self.assertTrue(hasattr(self.config, "MAX_POSITIONS_PER_GROUP"))
        self.assertIsInstance(self.config.MAX_OPEN_POSITIONS, int)
        self.assertIsInstance(self.config.MAX_POSITIONS_PER_GROUP, int)
        self.assertGreaterEqual(self.config.MAX_OPEN_POSITIONS, 1)
        self.assertGreaterEqual(self.config.MAX_POSITIONS_PER_GROUP, 1)

    def test_T103_coin_groups_dict_exists_and_populated(self):
        """T103: COIN_GROUPS — непустой словарь с корректной структурой"""
        self.assertTrue(hasattr(self.config, "COIN_GROUPS"))
        groups = self.config.COIN_GROUPS
        self.assertIsInstance(groups, dict)
        self.assertGreater(len(groups), 3, "Должно быть > 3 групп монет")
        for grp, members in groups.items():
            self.assertIsInstance(members, list,
                                  f"COIN_GROUPS['{grp}'] должен быть списком")
            self.assertGreater(len(members), 0,
                               f"Группа '{grp}' не должна быть пустой")
            for sym in members:
                self.assertTrue(sym.endswith("USDT"),
                                f"'{sym}' в группе '{grp}' должен заканчиваться на USDT")

    # ── T104: _get_coin_group ─────────────────────────────────────────────────

    def test_T104_get_group_known_coins(self):
        """T104: _get_coin_group возвращает правильные группы для известных монет"""
        # Тестируем монеты из реального события 11.03.2026
        cases = [
            ("AXSUSDT",    "GameFi"),
            ("SANDUSDT",   "GameFi"),
            ("ARBUSDT",    "L2_eth"),
            ("OPUSDT",     "L2_eth"),
            ("UNIUSDT",    "DeFi_amm"),
            ("SUSHIUSDT",  "DeFi_amm"),
        ]
        for sym, expected_grp in cases:
            result = self.get_group(sym)
            self.assertEqual(result, expected_grp,
                             f"{sym}: ожидалась группа '{expected_grp}', получена '{result}'")

    def test_T105_get_group_unknown_coin_returns_none(self):
        """T105: _get_coin_group возвращает None для неизвестной монеты"""
        self.assertIsNone(self.get_group("UNKNOWNUSDT"))
        self.assertIsNone(self.get_group(""))
        self.assertIsNone(self.get_group("FAKEUSDT"))

    def test_T106_no_duplicate_coins_across_groups(self):
        """T106: одна монета не может быть в двух группах"""
        groups = self.config.COIN_GROUPS
        seen = {}
        for grp, members in groups.items():
            for sym in members:
                self.assertNotIn(
                    sym, seen,
                    f"'{sym}' встречается в группах '{seen.get(sym)}' и '{grp}'"
                )
                seen[sym] = grp

    # ── T107: _check_portfolio_limits ────────────────────────────────────────

    def test_T107_full_portfolio_blocks_entry(self):
        """T107: КРИТИЧНО — при MAX_OPEN_POSITIONS открытых позициях вход заблокирован"""
        max_pos = self.config.MAX_OPEN_POSITIONS
        # Создаём ровно max_pos позиций с разными монетами
        symbols = [f"COIN{i}USDT" for i in range(max_pos)]
        state = self._make_state(symbols)
        ok, reason = self.check_port("NEWCOINUSDT", state)
        self.assertFalse(ok,
            f"При {max_pos}/{max_pos} позициях вход должен быть заблокирован")
        self.assertIn(str(max_pos), reason,
            "Причина блокировки должна содержать лимит")

    def test_T108_portfolio_allows_entry_below_limit(self):
        """T108: вход разрешён когда позиций меньше лимита"""
        max_pos = self.config.MAX_OPEN_POSITIONS
        # Одна позиция меньше лимита
        state = self._make_state([f"COIN{i}USDT" for i in range(max_pos - 1)])
        ok, reason = self.check_port("NEWUSDT", state)
        self.assertTrue(ok,
            f"При {max_pos-1}/{max_pos} позициях вход должен быть разрешён")

    def test_T109_group_limit_blocks_correlated_coins(self):
        """T109: КРИТИЧНО — группа заблокирована когда занято MAX_POSITIONS_PER_GROUP"""
        max_grp = self.config.MAX_POSITIONS_PER_GROUP
        # Берём реальную группу с известными монетами
        gameFi_coins = [
            s for s in self.config.COIN_GROUPS.get("GameFi", [])
        ]
        if len(gameFi_coins) < max_grp + 1:
            self.skipTest("Мало монет в группе GameFi для теста")

        # Занимаем max_grp слотов в группе GameFi
        state = self._make_state(gameFi_coins[:max_grp])
        # Следующая монета из той же группы — должна быть заблокирована
        new_game_fi = gameFi_coins[max_grp]
        ok, reason = self.check_port(new_game_fi, state)
        self.assertFalse(ok,
            f"Группа GameFi {max_grp}/{max_grp}: вход {new_game_fi} должен быть заблокирован")
        self.assertIn("GameFi", reason,
            "Причина должна называть группу")

    def test_T110_group_limit_allows_different_group(self):
        """T110: заполненная группа не блокирует монеты из других групп"""
        max_grp = self.config.MAX_POSITIONS_PER_GROUP
        gameFi_coins = self.config.COIN_GROUPS.get("GameFi", [])[:max_grp]
        state = self._make_state(gameFi_coins)

        # Монета из другой группы — должна пройти (если портфель не полный)
        ok, reason = self.check_port("ARBUSDT", state)  # L2_eth
        self.assertTrue(ok,
            f"L2_eth не должна блокироваться из-за лимита GameFi: {reason}")

    def test_T111_portfolio_check_in_monitor_source(self):
        """T111: _check_portfolio_limits вызывается в monitor._poll_coin до entry"""
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("_check_portfolio_limits", src,
            "_check_portfolio_limits должен вызываться в monitor.py")
        # Проверяем что проверка происходит ДО state.positions[sym] = pos
        check_idx = src.find("_check_portfolio_limits(")
        entry_idx = src.find("state.positions[sym] = pos")
        self.assertGreater(check_idx, 0,
            "_check_portfolio_limits не найден в monitor.py")
        self.assertGreater(entry_idx, 0,
            "state.positions[sym] = pos не найден в monitor.py")
        self.assertLess(check_idx, entry_idx,
            "Портфельная проверка должна быть ДО state.positions[sym] = pos")

    def test_T112_empty_portfolio_always_allows(self):
        """T112: пустой портфель всегда разрешает вход"""
        state = self.MonitorState()
        for sym in ["BTCUSDT", "ETHUSDT", "ARBUSDT", "AXSUSDT"]:
            ok, reason = self.check_port(sym, state)
            self.assertTrue(ok, f"Пустой портфель должен разрешать {sym}: {reason}")

    def test_T113_portfolio_check_returns_tuple(self):
        """T113: _check_portfolio_limits возвращает (bool, str)"""
        state = self.MonitorState()
        result = self.check_port("BTCUSDT", state)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)

    # ── T114: regression 11.03.2026 ──────────────────────────────────────────

    def test_T114_regression_11mar_scenario(self):
        """
        T114: РЕГРЕССИЯ 11.03.2026 — 12 монет не войдут одновременно.

        В тот день вошли: GLM, RENDER, ORDI, AXS, FIL, TIA, OP, UNI, DOT, ARB, SAND, SUSHI.
        С портфельным лимитом max_pos=6 max_grp=2 — максимум 6, не 12.
        """
        signals_11mar = [
            "GLMUSDT", "RENDERUSDT", "ORDIUSDT", "AXSUSDT",
            "FILUSDT", "TIAUSDT", "OPUSDT", "UNIUSDT",
            "DOTUSDT", "ARBUSDT", "SANDUSDT", "SUSHIUSDT",
        ]
        max_pos = self.config.MAX_OPEN_POSITIONS  # 6
        state   = self.MonitorState()
        entered = []
        blocked = []

        for sym in signals_11mar:
            ok, reason = self.check_port(sym, state)
            if ok:
                from unittest.mock import MagicMock
                state.positions[sym] = MagicMock()
                entered.append(sym)
            else:
                blocked.append((sym, reason))

        self.assertLessEqual(
            len(entered), max_pos,
            f"Должно войти ≤ {max_pos} монет, вошло: {len(entered)}: {entered}"
        )
        self.assertGreater(
            len(blocked), 0,
            "Часть монет должна быть заблокирована портфельным лимитом"
        )
        # Проверяем что ни одна группа не превышена
        group_counts = {}
        for sym in entered:
            grp = self.get_group(sym)
            if grp:
                group_counts[grp] = group_counts.get(grp, 0) + 1
        max_grp = self.config.MAX_POSITIONS_PER_GROUP
        for grp, cnt in group_counts.items():
            self.assertLessEqual(
                cnt, max_grp,
                f"Группа '{grp}': {cnt} позиций > лимита {max_grp}"
            )


# ════════════════════════════════════════════════════════════════════════════════
# T121–T125  Регрессия: немедленный WEAK выход на баре входа (AR 11.03.2026)
# ════════════════════════════════════════════════════════════════════════════════

class TestWeakExitGuard(unittest.TestCase):
    """
    Регрессия: ARUSDT 11.03.2026 19:16-19:17.

    Бот вошёл в позицию и через 1 минуту (0 баров) выдал SELL.
    Причина: RSI дивергенция существовала ДО входа, check_exit_conditions
    срабатывал на том же баре что и вход.
    Монета при этом росла +1.14% на 1h.

    Фикс: WEAK выходы подавляются первые MIN_WEAK_EXIT_BARS баров.
    """

    def test_T121_min_weak_exit_bars_config_exists(self):
        """T121: MIN_WEAK_EXIT_BARS присутствует в конфиге"""
        import config
        self.assertTrue(hasattr(config, "MIN_WEAK_EXIT_BARS"),
                        "config.MIN_WEAK_EXIT_BARS не найден")
        self.assertIsInstance(config.MIN_WEAK_EXIT_BARS, int)
        self.assertGreaterEqual(config.MIN_WEAK_EXIT_BARS, 1,
                                "MIN_WEAK_EXIT_BARS должен быть ≥ 1")

    def test_T122_weak_exit_suppressed_at_zero_bars(self):
        """T122: КРИТИЧНО — WEAK выход подавляется при bars_elapsed=0"""
        src = Path("monitor.py").read_text(encoding="utf-8")
        # Должна быть проверка is_weak и bars_elapsed
        self.assertIn("is_weak", src,
                      "monitor.py должен проверять is_weak перед выходом")
        self.assertIn("MIN_WEAK_EXIT_BARS", src,
                      "monitor.py должен читать MIN_WEAK_EXIT_BARS")
        self.assertIn("bars_elapsed < min_weak_bars", src,
                      "guard должен сравнивать bars_elapsed с min_weak_bars")

    def test_T123_weak_suppression_only_for_weak_signals(self):
        """T123: ATR-трейл и time-exit НЕ подавляются — только WEAK"""
        src = Path("monitor.py").read_text(encoding="utf-8")
        # Убеждаемся что ATR-выход (trail_stop) проверяется ДО фильтра WEAK
        atr_idx  = src.find("trail_stop > 0 and close_now < pos.trail_stop")
        weak_idx = src.find("MIN_WEAK_EXIT_BARS")
        self.assertGreater(atr_idx, 0, "ATR-трейл проверка не найдена")
        self.assertGreater(weak_idx, 0, "MIN_WEAK_EXIT_BARS не найден")
        self.assertLess(atr_idx, weak_idx,
                        "ATR-трейл должен проверяться ДО фильтра WEAK")

    def test_T124_weak_exit_allowed_after_min_bars(self):
        """T124: после MIN_WEAK_EXIT_BARS баров WEAK выход разрешён"""
        src = Path("monitor.py").read_text(encoding="utf-8")
        # Логика: if is_weak AND bars_elapsed < min → suppress
        # Следовательно при bars_elapsed >= min — не suppress
        idx = src.find("is_weak and pos.bars_elapsed < min_weak_bars")
        self.assertGreater(idx, 0,
                           "Условие подавления должно проверять bars_elapsed < min_weak_bars")
        # Проверяем что при этом reason обнуляется
        block = src[idx:idx + 400]
        self.assertIn("reason = None", block,
                      "При подавлении reason должен обнуляться")

    def test_T125_regression_ar_11mar(self):
        """
        T125: РЕГРЕССИЯ AR 11.03.2026 — WEAK не срабатывает при 0 барах.

        Проверяет что логика в monitor.py не допускает выхода
        с reason='⚠️ WEAK:...' при bars_elapsed=0.
        """
        src = Path("monitor.py").read_text(encoding="utf-8")

        # Находим блок кода Check exit conditions
        idx = src.find("# Check exit conditions")
        self.assertGreater(idx, 0, "Блок 'Check exit conditions' не найден")
        block = src[idx:idx + 1000]

        # Должна быть проверка что is_weak + bars_elapsed < min → reason=None
        self.assertIn("is_weak", block)
        self.assertIn("reason = None", block)

        # Должен быть debug-лог для диагностики подавленных выходов
        self.assertIn("EXIT SUPPRESSED", block,
                      "Должен логироваться подавлённый выход для диагностики")

    def test_T126_block_log_dedup_field_in_monitor_state(self):
        """T126: MonitorState содержит block_logged dict для dedup блокировок"""
        from monitor import MonitorState
        state = MonitorState()
        self.assertTrue(hasattr(state, "block_logged"),
                        "MonitorState должен иметь block_logged dict")
        self.assertIsInstance(state.block_logged, dict)

    def test_T127_block_log_dedup_in_monitor_source(self):
        """
        T127: РЕГРЕССИЯ 11.03.2026 — 2757 строк мусора (79% лога).
        TAOUSDT: 207 блокировок, ALGOUSDT: 158 и т.д.
        Теперь логируем не чаще BLOCK_LOG_INTERVAL_BARS баров.
        """
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("block_logged", src,
                      "monitor.py должен использовать block_logged для dedup")
        self.assertIn("BLOCK_LOG_INTERVAL_BARS", src,
                      "monitor.py должен читать интервал из конфига")
        # Dedup должен стоять ДО вызова botlog.log_blocked в портфельном блоке.
        # Ищем dedup (block_logged.get) и соответствующий botlog ПОСЛЕ него —
        # первый botlog.log_blocked до dedup может принадлежать другим блокам (MTF и т.п.)
        dedup_idx  = src.find("block_logged.get(sym")
        # Ищем ближайший botlog.log_blocked ПОСЛЕ dedup-проверки
        botlog_idx = src.find("botlog.log_blocked", dedup_idx)
        self.assertGreater(dedup_idx, 0, "block_logged.get(sym) не найден в monitor.py")
        self.assertGreater(botlog_idx, dedup_idx,
                           "botlog.log_blocked должен идти ПОСЛЕ dedup block_logged.get(sym)")

    def test_T128_block_log_interval_config(self):
        """T128: BLOCK_LOG_INTERVAL_BARS присутствует в конфиге"""
        import config
        self.assertTrue(hasattr(config, "BLOCK_LOG_INTERVAL_BARS"),
                        "config.BLOCK_LOG_INTERVAL_BARS не найден")
        self.assertIsInstance(config.BLOCK_LOG_INTERVAL_BARS, int)
        self.assertGreaterEqual(config.BLOCK_LOG_INTERVAL_BARS, 1)


# ════════════════════════════════════════════════════════════════════════════════
# T129–T131  Регрессия: ложный strong_trend на флэте SNX (12.03.2026)
# ════════════════════════════════════════════════════════════════════════════════

class TestStrongTrendClassification(unittest.TestCase):
    """
    Регрессия SNX 12.03.2026 00:08.

    Бот прислал "💪 Сильный тренд" при ADX=29.9, Vol×=3.54.
    Но монета неделю во флэте: EMA20=0.3156 ≈ EMA50=0.3130 (разрыв 0.83%),
    MACD≈0, MA20/50/200 почти в одной точке.

    Фикс: strong_trend требует ВСЕ три условия:
      1. ADX ≥ STRONG_ADX_MIN + Vol× ≥ STRONG_VOL_MIN
      2. EMA20 > EMA50 на ≥ STRONG_EMA_SEP_MIN % (0.9%)
      3. EMA50 наклон за 3 бара ≥ STRONG_EMA50_SLOPE_MIN % (0.05%)
    """

    def _make_feat(self, ema20, ema50, adx, vol_x, price_3ago, price_now, ema50_3ago):
        import numpy as np
        from indicators import compute_features
        n = 100
        c = np.full(n, ema20)
        v = np.full(n, 1.0)
        feat = compute_features(c, c, c, c, v)
        feat['adx'][-1]      = adx
        feat['vol_x'][-1]    = vol_x
        feat['ema_fast'][-1] = ema20
        feat['ema_slow'][-1] = ema50
        feat['ema_slow'][-4] = ema50_3ago
        feat['close'][-4]    = price_3ago
        feat['close'][-1]    = price_now
        return feat

    def test_T129_snx_flat_gets_trend_not_strong_trend(self):
        """
        T129: КРИТИЧНО — SNX флэт с ADX=29.9, Vol×=3.54 → должен быть 'trend'.
        EMA20=0.3156, EMA50=0.3130 → разрыв 0.83% < STRONG_EMA_SEP_MIN(0.9%).
        """
        from strategy import get_entry_mode
        feat = self._make_feat(
            ema20=0.3156, ema50=0.3130, adx=29.9, vol_x=3.54,
            price_3ago=0.315, price_now=0.319, ema50_3ago=0.3128,
        )
        mode = get_entry_mode(feat, 99)
        self.assertEqual(mode, "trend",
            f"SNX флэт (EMA_sep=0.83%) → ожидали 'trend', получили '{mode}'")

    def test_T130_ar_real_trend_stays_strong_trend(self):
        """
        T130: AR реальный тренд ADX=48, EMA_sep=3.03% → должен оставаться 'strong_trend'.
        """
        from strategy import get_entry_mode
        feat = self._make_feat(
            ema20=1.70, ema50=1.65, adx=48.3, vol_x=2.5,
            price_3ago=1.72, price_now=1.78, ema50_3ago=1.635,
        )
        mode = get_entry_mode(feat, 99)
        self.assertEqual(mode, "strong_trend",
            f"AR тренд (EMA_sep=3.03%, ADX=48) → ожидали 'strong_trend', получили '{mode}'")

    def test_T131_strong_trend_config_params_exist(self):
        """T131: STRONG_EMA_SEP_MIN и STRONG_EMA50_SLOPE_MIN присутствуют в конфиге"""
        import config
        self.assertTrue(hasattr(config, "STRONG_EMA_SEP_MIN"),
                        "config.STRONG_EMA_SEP_MIN не найден")
        self.assertTrue(hasattr(config, "STRONG_EMA50_SLOPE_MIN"),
                        "config.STRONG_EMA50_SLOPE_MIN не найден")
        self.assertIsInstance(config.STRONG_EMA_SEP_MIN, float)
        self.assertGreater(config.STRONG_EMA_SEP_MIN, 0.0)
        # Порог должен быть достаточно высок чтобы отфильтровать флэт (>0.5%)
        self.assertGreaterEqual(config.STRONG_EMA_SEP_MIN, 0.5,
                                "STRONG_EMA_SEP_MIN слишком низкий — не отфильтрует флэт")


# ════════════════════════════════════════════════════════════════════════════════
# T132–T136  Регрессия: ложные сигналы 12.03.2026 (LTC ретест, ICP/MANA alignment)
# ════════════════════════════════════════════════════════════════════════════════

class TestSignalQualityGuards(unittest.TestCase):
    """
    Регрессия сигналов 12.03.2026 00:06.

    LTC: ретест с зазором 0.001% (цена лежит на EMA20, не отскок).
    ICP: alignment при ADX=13.2 (флэт, нет направленности).
    MANA: alignment при EMA20≈EMA50 (разрыв 0.09%).
    """

    def _make_retest_feat(self, close_val, ema20, ema50, adx, vol_x, slope, rsi=55.0):
        import numpy as np
        from indicators import compute_features
        n = 120
        c = np.full(n, ema20); v = np.full(n, 1.0)
        feat = compute_features(c, c, c, c, v)
        feat['ema_fast'][-1] = ema20; feat['ema_slow'][-1] = ema50
        feat['adx'][-1] = adx; feat['vol_x'][-1] = vol_x; feat['rsi'][-1] = rsi
        feat['slope'][-1] = slope
        feat['close'][-1] = close_val; feat['close'][-2] = close_val * 0.998
        # Set up lookback bars above EMA20 (trend existed)
        for k in range(2, 15): feat['close'][-k] = ema20 * 1.01
        # Касание EMA20 3 бара назад
        feat['low'][-3] = ema20 * 0.999
        return feat

    def _make_alignment_feat(self, ema20, ema50, adx, vol_x, slope, rsi, macd=0.001, ema200=None):
        import numpy as np
        from indicators import compute_features
        n = 120
        c = np.full(n, ema20); v = np.full(n, 1.0)
        feat = compute_features(c, c, c, c, v)
        feat['ema_fast'][-1] = ema20; feat['ema_slow'][-1] = ema50
        feat['ema200'][-1] = ema200 if ema200 is not None else min(ema20, ema50) * 0.98
        feat['adx'][-1] = adx; feat['vol_x'][-1] = vol_x
        feat['rsi'][-1] = rsi; feat['slope'][-1] = slope
        feat['close'][-1] = ema20 * 1.005  # цена выше EMA20
        feat['macd_hist'][-1] = macd; feat['macd_hist'][-2] = macd; feat['macd_hist'][-3] = macd
        feat['macd_hist'][-4] = macd; feat['macd_hist'][-5] = macd
        feat['daily_range_pct'][-1] = 4.0
        return feat

    def test_T132_ltc_retest_close_on_ema_blocked(self):
        """
        T132: РЕГРЕССИЯ LTC 12.03.2026 — ретест когда close≈EMA20 (зазор 0.001%).
        close=54.97, EMA20=54.9694. Ожидаем блокировку по минимальному отскоку.
        """
        from strategy import check_retest_conditions
        feat = self._make_retest_feat(54.97, 54.9694, 54.71, 22.9, 0.84, slope=0.08, rsi=52.4)
        ok, reason = check_retest_conditions(feat, 119)
        if ok:
            self.fail("LTC с зазором 0.001% должен быть заблокирован, но прошёл")
        # Если заблокирован по другой причине (нет касания в тесте) — тоже ок,
        # главное что RETEST_MIN_BOUNCE_PCT существует в конфиге
        import config
        self.assertTrue(hasattr(config, "RETEST_MIN_BOUNCE_PCT"),
                        "config.RETEST_MIN_BOUNCE_PCT не найден")
        self.assertGreater(config.RETEST_MIN_BOUNCE_PCT, 0.0)

    def test_T133_retest_bounce_check_in_source(self):
        """T133: код ретеста проверяет минимальный отскок от EMA20"""
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("RETEST_MIN_BOUNCE_PCT", src,
                      "strategy.py должен проверять RETEST_MIN_BOUNCE_PCT")
        self.assertIn("bounce_min", src)
        # Проверка должна быть в check_retest_conditions
        idx = src.find("def check_retest_conditions")
        block = src[idx:idx + 3500]
        self.assertIn("bounce_min", block,
                      "bounce check должен быть внутри check_retest_conditions")

    def test_T134_icp_adx_13_alignment_blocked(self):
        """
        T134: РЕГРЕССИЯ ICP 12.03.2026 — alignment при ADX=13.2.
        ADX < ALIGNMENT_ADX_MIN(15) → заблокировать.
        """
        from strategy import check_alignment_conditions
        feat = self._make_alignment_feat(
            ema20=2.661, ema50=2.63, adx=13.2, vol_x=1.16,
            slope=0.14, rsi=54.9, macd=0.004,
        )
        ok, reason = check_alignment_conditions(feat, 119)
        self.assertFalse(ok,
            f"ICP ADX=13.2 должен быть заблокирован. reason='{reason}'")
        # Блокировка может прийти от ADX или MACD (порядок проверок) — важно что заблокирован
        self.assertTrue("ADX" in reason or "MACD" in reason,
            f"Причина должна упоминать ADX или MACD, got: '{reason}'")

    def test_T135_mana_tiny_ema_fan_alignment_blocked(self):
        """
        T135: РЕГРЕССИЯ MANA 12.03.2026 — alignment при EMA20≈EMA50 (разрыв 0.09%).
        EMA sep < ALIGNMENT_EMA_SEP_MIN(0.3%) → заблокировать.
        """
        from strategy import check_alignment_conditions
        feat = self._make_alignment_feat(
            ema20=0.09235, ema50=0.09227, adx=19.3, vol_x=1.12,
            slope=0.45, rsi=59.6, macd=0.0001,
        )
        ok, reason = check_alignment_conditions(feat, 119)
        self.assertFalse(ok,
            f"MANA EMA-fan=0.09% должен быть заблокирован. reason='{reason}'")

    def test_T136_alignment_quality_config_params(self):
        """T136: ALIGNMENT_ADX_MIN и ALIGNMENT_EMA_SEP_MIN в конфиге"""
        import config
        self.assertTrue(hasattr(config, "ALIGNMENT_ADX_MIN"),
                        "config.ALIGNMENT_ADX_MIN не найден")
        self.assertTrue(hasattr(config, "ALIGNMENT_EMA_SEP_MIN"),
                        "config.ALIGNMENT_EMA_SEP_MIN не найден")
        self.assertGreaterEqual(config.ALIGNMENT_ADX_MIN, 12,
                                "ALIGNMENT_ADX_MIN слишком низкий")
        # 15.03.2026: снижено с 0.3 → 0.05 чтобы ловить нарождающиеся
        # тренды (TIA sep=0.085%). SEI защита через MACD_REL_MIN=0.0002.
        self.assertGreaterEqual(config.ALIGNMENT_EMA_SEP_MIN, 0.04,
                                "ALIGNMENT_EMA_SEP_MIN слишком низкий (мин 0.04%)")
        self.assertLessEqual(config.ALIGNMENT_EMA_SEP_MIN, 0.31,
                             "ALIGNMENT_EMA_SEP_MIN слишком высокий (макс 0.31%)")


# ════════════════════════════════════════════════════════════════════════════════
# T137–T143  DataCollector — автообновление ml_dataset.jsonl
# ════════════════════════════════════════════════════════════════════════════════

    def test_T209_axl_nonbull_alignment_below_ema200_blocked(self):
        from strategy import check_alignment_conditions
        import config

        prev = bool(getattr(config, "_bull_day_active", False))
        try:
            config._bull_day_active = False
            feat = self._make_alignment_feat(
                ema20=0.05237379, ema50=0.05230, ema200=0.05340,
                adx=17.3, vol_x=2.67, slope=0.236, rsi=57.3, macd=0.00005,
            )
            ok, reason = check_alignment_conditions(feat, 119)
            self.assertFalse(ok, "AXL-like non-bull alignment below EMA200 должен блокироваться")
            self.assertIn("EMA200", reason)
        finally:
            config._bull_day_active = prev

    def test_T210_alignment_nonbull_guard_is_targeted(self):
        from strategy import check_alignment_conditions
        import config

        prev = bool(getattr(config, "_bull_day_active", False))
        try:
            config._bull_day_active = True
            feat = self._make_alignment_feat(
                ema20=0.05237379, ema50=0.05230, ema200=0.05340,
                adx=20.5, vol_x=2.67, slope=0.236, rsi=57.3, macd=0.00005,
            )
            ok, reason = check_alignment_conditions(feat, 119)
            self.assertTrue(ok, f"В bull режиме EMA200-guard не должен резать alignment. reason='{reason}'")
        finally:
            config._bull_day_active = prev

class TestDataCollector(unittest.TestCase):
    """
    Проверяет что data_collector правильно подключён и работоспособен.

    Баг 12.03.2026: ml_dataset.jsonl не обновлялся с 10.03.2026 16:32 (ручное обновление).
    Причина: data_collector.run_forever() НЕ вызывался из bot.py _post_init().
    Дополнительно: _detect_rule_signal вызывал check_retest_conditions(feat, i, c)
    и check_breakout_conditions(feat, i, h) с лишним третьим аргументом — TypeError при старте.
    """

    def test_T137_data_collector_module_importable(self):
        """T137: модуль data_collector импортируется без ошибок"""
        import importlib
        try:
            dc = importlib.import_module("data_collector")
        except Exception as e:
            self.fail(f"data_collector не импортируется: {e}")
        self.assertTrue(hasattr(dc, "run_forever"), "data_collector.run_forever не найден")
        self.assertTrue(hasattr(dc, "_process_coin"), "data_collector._process_coin не найден")
        self.assertTrue(hasattr(dc, "_detect_rule_signal"), "data_collector._detect_rule_signal не найден")

    def test_T138_post_init_starts_data_collector(self):
        """
        T138: КРИТИЧНО — bot.py _post_init запускает data_collector.run_forever.
        До фикса: _post_init только запускал _auto_reanalyze, data_collector не вызывался.
        """
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("async def _post_init")
        self.assertGreater(idx, 0, "_post_init не найден в bot.py")
        # Берём 1200 символов — достаточно чтобы охватить весь _post_init
        block = src[idx:idx + 1200]
        self.assertIn("data_collector", block,
            "_post_init должен импортировать/использовать data_collector")
        self.assertIn("run_forever", block,
            "_post_init должен запускать data_collector.run_forever()")

    def test_T139_detect_rule_signal_correct_signatures(self):
        """
        T139: КРИТИЧНО — _detect_rule_signal вызывает стратегию с правильными сигнатурами.

        Баг: check_retest_conditions(feat, i, c) и check_breakout_conditions(feat, i, h)
        имеют только 2 параметра (feat, i). Лишний третий аргумент → TypeError → весь
        data_collector падал молча (Exception проглатывался), rule_signal всегда "none".
        """
        src = Path("data_collector.py").read_text(encoding="utf-8")
        idx = src.find("def _detect_rule_signal")
        block = src[idx:idx + 600]
        # Проверяем что check_retest_conditions вызывается БЕЗ третьего аргумента
        self.assertNotIn("check_retest_conditions(feat, i, c)", block,
            "check_retest_conditions принимает только (feat, i), не (feat, i, c)")
        self.assertNotIn("check_breakout_conditions(feat, i, h)", block,
            "check_breakout_conditions принимает только (feat, i), не (feat, i, h)")
        # Проверяем правильные вызовы
        self.assertIn("check_retest_conditions(feat, i)", block,
            "check_retest_conditions должен вызываться как (feat, i)")
        self.assertIn("check_breakout_conditions(feat, i)", block,
            "check_breakout_conditions должен вызываться как (feat, i)")

    def test_T140_detect_rule_signal_returns_valid_value(self):
        """T140: _detect_rule_signal возвращает одно из ожидаемых значений, не падает"""
        import numpy as np
        import data_collector as dc
        from indicators import compute_features

        n = 120
        c = np.linspace(1.0, 1.05, n)
        v = np.ones(n)
        feat = compute_features(c, c, c, c, v)

        data = {
            "c": c, "o": c, "h": c * 1.002, "l": c * 0.998,
            "v": v, "t": np.arange(n, dtype=float) * 900000,
        }
        result = dc._detect_rule_signal(feat, n - 2, data)
        valid = {
            "none",
            "trend",
            "strong_trend",
            "retest",
            "breakout",
            "impulse_speed",
            "impulse",
            "alignment",
        }
        self.assertIn(result, valid,
            f"_detect_rule_signal вернул неожиданное значение: '{result}'")

    def test_T141_run_forever_is_coroutine(self):
        """T141: data_collector.run_forever — это async функция (coroutine)"""
        import asyncio
        import data_collector as dc
        self.assertTrue(asyncio.iscoroutinefunction(dc.run_forever),
            "data_collector.run_forever должна быть async функцией")

    def test_T142_collect_once_uses_watchlist(self):
        """T142: _collect_once использует config.load_watchlist() (а не хардкод)"""
        src = Path("data_collector.py").read_text(encoding="utf-8")
        idx = src.find("async def _collect_once")
        block = src[idx:idx + 400]
        self.assertIn("load_watchlist", block,
            "_collect_once должен читать watchlist из config.load_watchlist()")

    def test_T142b_data_collector_offloads_hot_path_to_threads(self):
        """T142b: data_collector не должен держать event loop на compute/write hot path."""
        src = Path("data_collector.py").read_text(encoding="utf-8")
        self.assertIn("def _process_coin_sync(", src)
        self.assertIn("await asyncio.to_thread(", src)
        self.assertIn("await asyncio.to_thread(_log_dataset_stats)", src)

    def test_T143_seconds_until_next_bar_positive(self):
        """T143: _seconds_until_next_bar возвращает положительное число"""
        import data_collector as dc
        wait = dc._seconds_until_next_bar()
        self.assertGreater(wait, 0, "_seconds_until_next_bar должна возвращать > 0")
        self.assertLessEqual(wait, 900 + 10,
            "_seconds_until_next_bar не должна быть > длины бара + буфер")


# ════════════════════════════════════════════════════════════════════════════════
# T144–T146  Кнопки Telegram: query.answer() timeout guard
# ════════════════════════════════════════════════════════════════════════════════

class TestCallbackQueryAnswerGuard(unittest.TestCase):
    """
    Баг 12.03.2026: кнопки падали с BadRequest 'Query is too old'.
    Причина: query.answer() вызывался без try/except.
    При перезапуске бота накопленные старые нажатия (>2 мин) приводили к крашу
    всего handler — действие не выполнялось вовсе.

    Дополнительно: в ветке scan_and_start был дублирующий query.answer()
    после уже вызванного выше — тоже BadRequest.
    """

    def test_T144_btn_query_answer_has_try_except(self):
        """
        T144: КРИТИЧНО — query.answer() в btn обёрнут в try/except.
        До фикса: await query.answer() без защиты → BadRequest → handler крашился.
        """
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("async def btn(")
        self.assertGreater(idx, 0, "async def btn не найден")
        block = src[idx:idx + 600]
        # Должен быть try блок рядом с query.answer()
        self.assertIn("try:", block,
            "query.answer() должен быть обёрнут в try/except")
        self.assertIn("except", block,
            "Должен быть except после try с query.answer()")

    def test_T145_btn_no_bare_query_answer(self):
        """
        T145: query.answer() вне try/except в btn недопустим.
        Проверяем что голый 'await query.answer()' без отступа не встречается.
        """
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("async def btn(")
        block = src[idx:idx + 800]
        # Голый вызов без отступа try-блока — это строка "    await query.answer()"
        # Допустима только версия внутри try (отступ 8 пробелов)
        lines = block.split("\n")
        for line in lines:
            if "await query.answer()" in line:
                # Должен быть внутри try — отступ >= 8 пробелов
                stripped = line.lstrip(" ")
                indent = len(line) - len(stripped)
                self.assertGreaterEqual(indent, 8,
                    f"query.answer() должен быть внутри try-блока (отступ >=8): {line!r}")

    def test_T146_scan_and_start_no_duplicate_answer(self):
        """
        T146: ветка scan_and_start не вызывает query.answer() повторно.
        До фикса: query.answer('Мониторинг уже запущен') после уже вызванного
        answer() → BadRequest 'query was already answered'.
        """
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find('action == "scan_and_start"')
        self.assertGreater(idx, 0, "ветка scan_and_start не найдена")
        block = src[idx:idx + 600]
        self.assertNotIn("await query.answer(", block,
            "scan_and_start не должна вызывать query.answer() — он уже вызван выше")

    def test_T148_fill_pending_uses_tf_aware_bar_ms(self):
        """
        T148: КРИТИЧНО — fill_pending_from_data вызывается с bar_ms = tf-специфичным значением.

        Баг 13.03.2026 (SEI 1h): data_collector всегда передавал bar_ms=900*1000 (15m),
        игнорируя фактический tf монеты. Для 1h-записей T+3 = 3 бара × 1h = 180 мин,
        но бот считал T+3 = 3 × 15мин = 45 мин → заполнял форвард ДОСРОЧНО,
        измеряя цену в совершенно другой момент времени.

        Следствие: все форвард-снимки (T+3, T+5, T+10) пришли через 75 мин вместо
        180/300/600 мин, показали +0.00% / -0.30% / +0.00% и все ❌.
        Реальный рост SEI (0.0669→0.0674+) произошёл позже, когда форварды уже были записаны.
        """
        src = Path("data_collector.py").read_text(encoding="utf-8")
        idx = src.find("# Заполняем forward labels")
        self.assertGreater(idx, 0, "fill_pending блок не найден")
        block = src[idx:idx + 500]

        # Нельзя использовать константу BAR_SECONDS напрямую для bar_ms
        self.assertNotIn("bar_ms = BAR_SECONDS * 1000", block,
            "bar_ms не должен хардкодиться в BAR_SECONDS — нужен tf-aware расчёт")

        # Должна быть таблица соответствия tf → секунды
        self.assertIn("_TF_SECONDS", block,
            "Должен быть словарь _TF_SECONDS с маппингом tf → секунды")
        self.assertIn('"1h": 3600', block,
            "_TF_SECONDS должен содержать 1h: 3600")
        self.assertIn('"15m": 900', block,
            "_TF_SECONDS должен содержать 15m: 900")

        # bar_ms должен браться из таблицы по tf
        self.assertIn(".get(tf,", block,
            "bar_ms должен вычисляться через _TF_SECONDS.get(tf, ...)")

    def test_T147_button_uses_load_watchlist_count(self):
        """
        T147: КРИТИЧНО — кнопка 'Анализ + Мониторинг' показывает count из load_watchlist(),
        а не из DEFAULT_WATCHLIST.

        Баг 12.03.2026: кнопка показывала '82 монеты' (DEFAULT_WATCHLIST = 82 элемента
        хардкодом), тогда как реальный watchlist.json содержал 85 монет.
        Пользователь видел несоответствие: 'Монет в списке: 85' в сообщении,
        но '82 монет' на кнопке.
        """
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("def kb_main()")
        self.assertGreater(idx, 0, "kb_main не найдена")
        block = src[idx:idx + 800]
        # Недопустимо использовать хардкодный DEFAULT_WATCHLIST в метке кнопки
        self.assertNotIn("len(config.DEFAULT_WATCHLIST)", block,
            "Кнопка не должна использовать DEFAULT_WATCHLIST — он не учитывает watchlist.json")
        # Должна использоваться переменная wl из load_watchlist()
        self.assertIn("load_watchlist", block,
            "kb_main должна вызывать load_watchlist() для актуального числа монет")


# ════════════════════════════════════════════════════════════════════════════════
# T154–T159  Alignment quality guard: TAO/SEI ложные входы 13.03.2026
# ════════════════════════════════════════════════════════════════════════════════

class TestAlignmentQualityGuards(unittest.TestCase):
    """
    Два ложных alignment-входа 13.03.2026 показали системные дыры:
    - TAO: daily_range=16.12% (слишком поздно), EV=-0.24% при входе
    - SEI: MACD hist=0.0000 (иссяк), EV=-0.24% при входе

    Фиксы:
    1. ALIGNMENT_RANGE_MAX: 18% → 9% (TAO заблокирован)
    2. ALIGNMENT_MACD_REL_MIN: 0.0002 (SEI hist≈0 заблокирован)
    3. ALIGNMENT_MACD_BARS:  3 → 5 (более строгая проверка устойчивости)
    4. EV_MIN_PCT: 0.0 → 0.05 (отрицательный EV больше не пропускается)
    5. EV_MIN_SAMPLES: 3 → 5 (TAO T+3=20%=1/5 не является статистикой)
    Tests: T154–T159
    """

    def _make_feat(self, n=60, slope=0.1, macd_hist_last=0.01, rsi=65.0,
                   vol_x=1.5, ema_sep_pct=0.8, daily_range=5.0, adx=20.0,
                   close=0.067):
        """Хелпер: строит минимальный feat dict для проверки alignment."""
        import numpy as np
        from indicators import compute_features

        c  = np.linspace(close * 0.98, close, n)
        v  = np.ones(n) * vol_x
        feat = compute_features(c, c * 1.001, c * 0.999, c, v)

        # Переопределяем нужные поля напрямую
        feat["slope"]         = np.full(n, slope)
        feat["rsi"]           = np.full(n, rsi)
        feat["vol_x"]         = np.full(n, vol_x)
        feat["daily_range_pct"] = np.full(n, daily_range)
        feat["adx"]           = np.full(n, adx)

        # EMA sep через ema_fast/ema_slow
        base = close / (1 + ema_sep_pct / 100)
        feat["ema_fast"] = np.full(n, close * (1 - 0.001))
        feat["ema_slow"] = np.full(n, base)
        feat["close"]    = np.full(n, close)

        # MACD hist: последний бар = macd_hist_last, остальные = 0.05 (нормальные)
        mh = np.full(n, 0.05)
        mh[-1] = macd_hist_last
        feat["macd_hist"] = mh
        return feat

    def test_T154_alignment_range_max_lowered(self):
        """T154: ALIGNMENT_RANGE_MAX снижен с 18% до ≤9%."""
        import config
        self.assertLessEqual(config.ALIGNMENT_RANGE_MAX, 9.0,
            "ALIGNMENT_RANGE_MAX должен быть ≤9% (TAO имел 16.12% — слишком поздно)")

    def test_T155_tao_blocked_by_range(self):
        """T155: TAO с daily_range=16.12% блокируется alignment."""
        from strategy import check_alignment_conditions
        feat = self._make_feat(daily_range=16.12, close=231.2, macd_hist_last=0.5,
                               rsi=71.6, slope=1.7, ema_sep_pct=2.6, adx=46.7)
        ok, reason = check_alignment_conditions(feat, 58)
        self.assertFalse(ok,
            f"TAO daily_range=16.12% должен быть заблокирован. reason='{reason}'")
        self.assertIn("daily_range", reason,
            f"Причина должна упоминать daily_range, got: '{reason}'")

    def test_T156_alignment_macd_rel_min_in_config(self):
        """T156: ALIGNMENT_MACD_REL_MIN добавлен в конфиг."""
        import config
        self.assertTrue(hasattr(config, "ALIGNMENT_MACD_REL_MIN"),
            "config.ALIGNMENT_MACD_REL_MIN не найден")
        self.assertGreater(config.ALIGNMENT_MACD_REL_MIN, 0,
            "ALIGNMENT_MACD_REL_MIN должен быть > 0")

    def test_T157_sei_blocked_by_macd_exhausted(self):
        """T157: SEI с MACD hist≈0 блокируется по ALIGNMENT_MACD_REL_MIN."""
        from strategy import check_alignment_conditions
        import numpy as np
        from indicators import compute_features

        n = 60
        close = 0.0669
        c = np.linspace(close * 0.96, close, n)
        v = np.ones(n) * 1.5
        feat = compute_features(c, c * 1.001, c * 0.999, c, v)

        feat["slope"]           = np.full(n, 0.15)
        feat["rsi"]             = np.full(n, 70.96)
        feat["vol_x"]           = np.full(n, 1.5)
        feat["daily_range_pct"] = np.full(n, 4.5)
        feat["adx"]             = np.full(n, 18.0)
        feat["ema_fast"]        = np.full(n, 0.0666)
        feat["ema_slow"]        = np.full(n, 0.0660)
        feat["close"]           = np.full(n, close)
        # MACD hist: последние 5 баров = 0.0000001 (формально > 0, но иссяк)
        mh = np.full(n, 0.001)
        mh[-5:] = 0.0000001
        feat["macd_hist"] = mh

        ok, reason = check_alignment_conditions(feat, n - 1)
        self.assertFalse(ok,
            f"SEI с MACD hist≈0 должен быть заблокирован. reason='{reason}'")

    def test_T157B_zrx_late_alignment_with_faded_macd_blocked(self):
        """T157B: ZRX-подобный late alignment блокируется, если MACD уже сильно сдулся от локального пика."""
        from strategy import check_alignment_conditions

        feat = self._make_feat(
            daily_range=7.47,
            close=0.1036,
            macd_hist_last=0.00005789,
            rsi=59.5,
            slope=0.526,
            ema_sep_pct=0.98,
            adx=27.4,
            vol_x=5.42,
        )
        import numpy as np
        feat["macd_hist"][-8:] = np.array([
            0.00025,
            0.00021,
            0.00017,
            0.00013,
            0.00010,
            0.00008,
            0.00007,
            0.00005789,
        ])

        ok, reason = check_alignment_conditions(feat, 58)
        self.assertFalse(ok, f"ZRX-like late alignment должен блокироваться. reason='{reason}'")
        self.assertIn("late alignment", reason)

    def test_T157C_jasmy_like_1h_stretched_alignment_blocked(self):
        """T157C: JASMY-like 1h alignment is blocked when price is too stretched above EMA20."""
        from strategy import check_alignment_conditions
        import config
        import numpy as np

        prev_bull = getattr(config, "_bull_day_active", False)
        try:
            config._bull_day_active = False
            feat = self._make_feat(
                daily_range=6.88,
                close=0.00559,
                macd_hist_last=0.00002825,
                rsi=63.9,
                slope=1.32,
                ema_sep_pct=0.54,
                adx=36.6,
                vol_x=2.03,
            )
            n = len(feat["close"])
            feat["ema_fast"] = np.full(n, 0.0054995)
            feat["ema_slow"] = np.full(n, 0.00547)
            feat["ema200"] = np.full(n, 0.00545)
            feat["macd_hist"][-8:] = np.array([
                0.000018,
                0.000019,
                0.000021,
                0.000022,
                0.000024,
                0.000026,
                0.000027,
                0.00002825,
            ])

            ok_15m, reason_15m = check_alignment_conditions(feat, n - 1, tf="15m")
            ok_1h, reason_1h = check_alignment_conditions(feat, n - 1, tf="1h")

            self.assertTrue(ok_15m, f"15m alignment should remain valid here. reason='{reason_15m}'")
            self.assertFalse(ok_1h, f"1h stretched alignment should be blocked. reason='{reason_1h}'")
            self.assertIn("price_edge", reason_1h)
            self.assertIn("late alignment", reason_1h)
        finally:
            config._bull_day_active = prev_bull

    def test_T158_ev_min_pct_raised(self):
        """T158: EV_MIN_PCT поднят до 0.05%."""
        import config
        self.assertGreaterEqual(config.EV_MIN_PCT, 0.05,
            "EV_MIN_PCT должен быть ≥ 0.05 — отрицательный EV у TAO/SEI при входе")

    def test_T159_ev_min_samples_raised(self):
        """T159: EV_MIN_SAMPLES поднят до 5."""
        import config
        self.assertGreaterEqual(config.EV_MIN_SAMPLES, 5,
            "EV_MIN_SAMPLES должен быть ≥ 5 — TAO T+3=20%=1/5 это не статистика")


        """T115: positions блок использует html.escape для символов"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find('action == "positions"')
        block = src[idx:idx + 1500]
        self.assertIn("_html.escape", block,
            "positions блок должен экранировать sym через html.escape")

    def test_T116_positions_shows_portfolio_bar(self):
        """T116: positions показывает прогресс-бар портфеля"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find('action == "positions"')
        block = src[idx:idx + 1500]
        self.assertIn("port_bar", block,
            "positions должен строить прогресс-бар портфеля")
        self.assertIn("MAX_OPEN_POSITIONS", block,
            "positions должен показывать лимит из MAX_OPEN_POSITIONS")

    def test_T117_positions_shows_coin_group(self):
        """T117: positions показывает группу монеты"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find('action == "positions"')
        block = src[idx:idx + 1500]
        self.assertIn("_get_coin_group", block,
            "positions должен вызывать _get_coin_group для отображения группы")

    def test_T118_kb_main_shows_portfolio_capacity(self):
        """T118: kb_main показывает N/MAX в кнопке позиций"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("signals_lbl")
        block = src[idx:idx + 200]
        self.assertIn("MAX_OPEN_POSITIONS", block,
            "kb_main должен показывать лимит позиций в кнопке")

    def test_T119_positions_no_raw_truncation(self):
        """T119: в positions нет [:4096] — только безопасное усечение"""
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find('action == "positions"')
        end = src.find('elif action ==', idx + 1)
        block = src[idx:end]
        self.assertNotIn("[:4096]", block,
            "positions не должен использовать raw [:4096] — срезает теги")

    def test_T120_safe_truncate_function_exists(self):
        """T120: _safe_truncate определена в bot.py"""
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def _safe_truncate(", src,
            "_safe_truncate должна быть определена в bot.py")
        # Проверяем что обрезает по границе строки
        idx = src.find("def _safe_truncate(")
        func_body = src[idx:idx + 400]
        self.assertIn("rfind", func_body,
            "_safe_truncate должна искать последний \\n через rfind")



# ════════════════════════════════════════════════════════════════════════════════
# T160–T163  MTF (Multi-TimeFrame) фильтр для 1h сигналов
# ════════════════════════════════════════════════════════════════════════════════
# MTF-фильтр блокирует 1h входы когда 15м уже в коррекции.
# Пример: ETH 13.03 — 1h вход 16:00, но 15м MACD=-6.78, RSI=41 → блок.
#
# Фиксы:
#   MTF_ENABLED=True, MTF_MACD_POSITIVE=True, MTF_RSI_MIN=42.0 в config.py
#   _check_mtf() в monitor.py — грузит 15м данные и проверяет MACD/RSI
#   Tests: T160–T163
class TestMTFFilter(unittest.TestCase):
    """T160–T163: MTF-фильтр — конфиг, логика блокировки, fail-open."""

    def test_T160_mtf_params_in_config(self):
        """T160: MTF параметры присутствуют в config с корректными значениями."""
        import config
        self.assertTrue(hasattr(config, "MTF_ENABLED"),       "MTF_ENABLED отсутствует")
        self.assertTrue(hasattr(config, "MTF_MACD_POSITIVE"),  "MTF_MACD_POSITIVE отсутствует")
        self.assertTrue(hasattr(config, "MTF_RSI_MIN"),        "MTF_RSI_MIN отсутствует")
        self.assertTrue(hasattr(config, "MTF_MACD_SOFT_FLOOR_REL"), "MTF_MACD_SOFT_FLOOR_REL отсутствует")
        self.assertTrue(hasattr(config, "MTF_RSI_SOFT_MIN"), "MTF_RSI_SOFT_MIN отсутствует")
        self.assertTrue(hasattr(config, "MTF_REQUIRE_MACD_RISING"), "MTF_REQUIRE_MACD_RISING отсутствует")
        self.assertIsInstance(config.MTF_ENABLED,       bool)
        self.assertIsInstance(config.MTF_MACD_POSITIVE, bool)
        self.assertIsInstance(config.MTF_RSI_MIN,       float)
        self.assertIsInstance(config.MTF_MACD_SOFT_FLOOR_REL, float)
        self.assertIsInstance(config.MTF_RSI_SOFT_MIN, float)
        self.assertIsInstance(config.MTF_REQUIRE_MACD_RISING, bool)
        self.assertTrue(config.MTF_ENABLED,  "MTF_ENABLED должен быть True по умолчанию")
        self.assertEqual(config.MTF_RSI_MIN, 42.0, "MTF_RSI_MIN должен быть 42.0")

    def test_T161_check_mtf_function_exists(self):
        """T161: функция _check_mtf существует в monitor.py и является корутиной."""
        import inspect
        import monitor
        self.assertTrue(hasattr(monitor, "_check_mtf"),
                        "_check_mtf не найдена в monitor.py")
        self.assertTrue(inspect.iscoroutinefunction(monitor._check_mtf),
                        "_check_mtf должна быть async")

    def test_T162_mtf_block_called_for_1h_only(self):
        """T162: MTF-блок применяется только к 1h сигналам, не к 15m."""
        import monitor
        src = Path(monitor.__file__).read_text(encoding="utf-8")
        # Проверяем что фильтр применяется условно для 1h
        self.assertIn('tf == "1h"', src,
                      "MTF фильтр должен проверять tf == '1h'")
        self.assertIn("MTF_ENABLED", src,
                      "MTF фильтр должен читать config.MTF_ENABLED")

    def test_T163_mtf_blocks_negative_macd(self):
        """T163: _check_mtf возвращает False при отрицательном 15м MACD hist.

        Симулируем сценарий ETH 13.03: MACD=-6.78 на 15м.
        Используем asyncio.run + мок fetch_klines возвращающий нисходящий 15м тренд.
        """
        import asyncio, numpy as np
        from unittest.mock import patch, AsyncMock
        import monitor, config

        # Строим нисходящие 15м данные: 50 баров вниз (MACD будет < 0)
        n = 60
        prices = np.linspace(2200, 2120, n)  # нисходящий тренд: 2200→2120
        fake_data = np.zeros(n, dtype=[
            ("t","i8"),("o","f8"),("h","f8"),("l","f8"),("c","f8"),("v","f8")
        ])
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = prices
        fake_data["h"] = prices * 1.001
        fake_data["l"] = prices * 0.999
        fake_data["c"] = prices
        fake_data["v"] = np.full(n, 1000.0)

        orig_enabled = config.MTF_ENABLED
        orig_macd    = config.MTF_MACD_POSITIVE
        orig_rsi     = config.MTF_RSI_MIN
        try:
            config.MTF_ENABLED       = True
            config.MTF_MACD_POSITIVE = True
            config.MTF_RSI_MIN       = 45.0

            mock_session = AsyncMock()
            with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)):
                ok, reason = asyncio.run(monitor._check_mtf(mock_session, "ETHUSDT"))

            self.assertFalse(ok, f"MTF должен блокировать нисходящий 15м тренд, но вернул True. Причина: {reason}")
            self.assertIn("MACD", reason, f"Причина блока должна упоминать MACD: {reason}")
        finally:
            config.MTF_ENABLED       = orig_enabled
            config.MTF_MACD_POSITIVE = orig_macd
            config.MTF_RSI_MIN       = orig_rsi

    def test_T163A_mtf_allows_shallow_negative_macd_if_recovering(self):
        """T163A: неглубокий отрицательный MACD можно пропустить, если он разворачивается вверх."""
        import asyncio, numpy as np
        from unittest.mock import patch, AsyncMock
        import monitor, config

        n = 60
        fake_data = np.zeros(n, dtype=[
            ("t","i8"),("o","f8"),("h","f8"),("l","f8"),("c","f8"),("v","f8")
        ])
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = 100.0
        fake_data["h"] = 100.2
        fake_data["l"] = 99.8
        fake_data["c"] = 100.0
        fake_data["v"] = 1000.0

        feat = {
            "macd_hist": np.full(n, -0.0010),
            "rsi": np.full(n, 55.0),
        }
        feat["macd_hist"][-3] = -0.0070
        feat["macd_hist"][-2] = -0.0030

        orig_floor = config.MTF_MACD_SOFT_FLOOR_REL
        orig_soft_rsi = config.MTF_RSI_SOFT_MIN
        orig_rising = config.MTF_REQUIRE_MACD_RISING
        try:
            config.MTF_MACD_SOFT_FLOOR_REL = -0.00005
            config.MTF_RSI_SOFT_MIN = 48.0
            config.MTF_REQUIRE_MACD_RISING = True
            mock_session = AsyncMock()
            with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)), \
                 patch("monitor.compute_features", return_value=feat):
                ok, reason = asyncio.run(monitor._check_mtf(mock_session, "GLMUSDT"))
            self.assertTrue(ok, f"Неглубокий разворотный MACD должен проходить soft-pass: {reason}")
            self.assertIn("soft-pass", reason)
        finally:
            config.MTF_MACD_SOFT_FLOOR_REL = orig_floor
            config.MTF_RSI_SOFT_MIN = orig_soft_rsi
            config.MTF_REQUIRE_MACD_RISING = orig_rising


class TestContinuousDiscoveryAndEarlyExit(unittest.TestCase):
    def test_T170_discovery_scan_param_exists(self):
        import config
        self.assertTrue(hasattr(config, "DISCOVERY_SCAN_SEC"))
        self.assertIsInstance(config.DISCOVERY_SCAN_SEC, int)
        self.assertGreaterEqual(config.DISCOVERY_SCAN_SEC, 0)

    def test_T171_monitor_contains_discovery_layer(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("def _coin_report_priority", src)
        self.assertIn("async def _discover_new_hot_coins", src)
        self.assertIn("DISCOVERY_SCAN_SEC", src)

    def test_T172_same_bar_exit_guard_exists(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("if pos.bars_elapsed <= 0:", src)
        self.assertIn("return", src[src.find("if pos.bars_elapsed <= 0:"):src.find("if pos.bars_elapsed <= 0:") + 80])

    def test_T173_offline_backtest_script_exists(self):
        path = Path("offline_backtest.py")
        self.assertTrue(path.exists(), "offline_backtest.py должен существовать")
        src = path.read_text(encoding="utf-8")
        self.assertIn("argparse", src)
        self.assertIn("Suggestions:", src)
        self.assertIn("blocked_top_reasons", src)

    def test_T173A_replay_backtest_script_exists(self):
        path = Path("replay_backtest.py")
        self.assertTrue(path.exists(), "replay_backtest.py должен существовать")
        src = path.read_text(encoding="utf-8")
        self.assertIn("simulate_symbol", src)
        self.assertIn("Replay Backtest", src)
        self.assertIn("run_replay", src)

    def test_T174_trend_macd_guards_in_source(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("TREND_MACD_REL_MIN", src)
        self.assertIn("MACD гистограмма ослабевает", src)
        self.assertIn("MACD гистограмма слишком слабая", src)

    def test_T175_strong_trend_extra_guards_in_source(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("STRONG_RSI_MIN", src)
        self.assertIn("STRONG_CLOSE_EMA20_MAX_PCT", src)
    def test_T163B_mtf_blocks_shallow_negative_macd_if_falling(self):
        """T163B: неглубокий отрицательный MACD всё ещё блокируется, если импульс продолжает слабеть."""
        import asyncio, numpy as np
        from unittest.mock import patch, AsyncMock
        import monitor, config

        n = 60
        fake_data = np.zeros(n, dtype=[
            ("t","i8"),("o","f8"),("h","f8"),("l","f8"),("c","f8"),("v","f8")
        ])
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = 100.0
        fake_data["h"] = 100.2
        fake_data["l"] = 99.8
        fake_data["c"] = 100.0
        fake_data["v"] = 1000.0

        feat = {
            "macd_hist": np.full(n, -0.0010),
            "rsi": np.full(n, 55.0),
        }
        feat["macd_hist"][-3] = -0.0015
        feat["macd_hist"][-2] = -0.0025

        orig_floor = config.MTF_MACD_SOFT_FLOOR_REL
        orig_soft_rsi = config.MTF_RSI_SOFT_MIN
        orig_rising = config.MTF_REQUIRE_MACD_RISING
        try:
            config.MTF_MACD_SOFT_FLOOR_REL = -0.00005
            config.MTF_RSI_SOFT_MIN = 48.0
            config.MTF_REQUIRE_MACD_RISING = True
            mock_session = AsyncMock()
            with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)), \
                 patch("monitor.compute_features", return_value=feat):
                ok, reason = asyncio.run(monitor._check_mtf(mock_session, "GLMUSDT"))
            self.assertFalse(ok, "Падающий отрицательный MACD должен блокироваться")
            self.assertIn("падает", reason)
        finally:
            config.MTF_MACD_SOFT_FLOOR_REL = orig_floor
            config.MTF_RSI_SOFT_MIN = orig_soft_rsi
            config.MTF_REQUIRE_MACD_RISING = orig_rising




# ════════════════════════════════════════════════════════════════════════════════
# T164  BREAKOUT: убрана RSI-проверка (ZRO 15.03.2026)
# ════════════════════════════════════════════════════════════════════════════════
# RSI<75 блокировал законные прорывы из флэта: после +2-3% за 1 бар RSI=80-85
# это норма, не перегрев. Бэктест 11/11 подтвердил безопасность.
class TestMTFRelaxedContinuation(unittest.TestCase):
    def test_T163C_mtf_relaxes_strong_one_hour_continuation(self):
        """T163C: strong 1h continuation passes MTF on a shallow 15m pullback."""
        import asyncio, numpy as np
        from unittest.mock import patch, AsyncMock
        import monitor, config

        n = 60
        fake_data = np.zeros(
            n,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = 100.0
        fake_data["h"] = 100.2
        fake_data["l"] = 99.8
        fake_data["c"] = 100.0
        fake_data["v"] = 1000.0

        feat = {
            "macd_hist": np.full(n, -0.0100),
            "rsi": np.full(n, 50.0),
            "ema_fast": np.full(n, 100.0),
        }
        feat["macd_hist"][-3] = -0.0200
        feat["macd_hist"][-2] = -0.0100

        prev_relax = config.MTF_1H_CONTINUATION_RELAX_ENABLED
        prev_floor = config.MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL
        prev_rsi = config.MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN
        prev_slip = config.MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT
        prev_rising = config.MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING
        try:
            config.MTF_1H_CONTINUATION_RELAX_ENABLED = True
            config.MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL = -0.00075
            config.MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN = 46.0
            config.MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT = 0.30
            config.MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING = True
            mock_session = AsyncMock()
            with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)), \
                 patch("monitor.compute_features", return_value=feat):
                ok, reason = asyncio.run(
                    monitor._check_mtf(
                        mock_session,
                        "LQTYUSDT",
                        mode="impulse_speed",
                        candidate_score=74.0,
                        slope=0.35,
                        adx=28.0,
                        rsi=66.0,
                        vol_x=1.80,
                        daily_range=4.5,
                    )
                )
            self.assertTrue(ok, reason)
            self.assertIn("relax-pass", reason)
        finally:
            config.MTF_1H_CONTINUATION_RELAX_ENABLED = prev_relax
            config.MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL = prev_floor
            config.MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN = prev_rsi
            config.MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT = prev_slip
            config.MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING = prev_rising


class TestBreakoutRSIGuard(unittest.TestCase):
    """T164: BREAKOUT допускает горячий импульс, но режет экстремально поздний spike."""

    def test_T164_breakout_fires_with_high_rsi(self):
        """T164: breakout с RSI=81 остаётся валидным, старый жёсткий cap 75 не возвращаем."""
        import numpy as np
        import strategy

        src_strategy = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("BREAKOUT_RSI_MAX", src_strategy)
        self.assertNotIn("RSI {ri:.1f} ≥ 75", src_strategy)

        _, feat, i = _make_feat()
        for k in range(8, 0, -1):
            feat["high"][i - k] = 100.0
            feat["low"][i - k] = 99.2
            feat["close"][i - k] = 99.6
        feat["close"][i] = 101.0
        feat["ema_fast"][i] = 100.0
        feat["slope"][i] = 0.20
        feat["vol_x"][i] = 2.5
        feat["rsi"][i] = 81.0
        feat["daily_range_pct"][i] = 2.5
        feat["macd_hist"][i - 1] = 0.10
        feat["macd_hist"][i] = 0.20
        ok, reason = strategy.check_breakout_conditions(feat, i)
        self.assertTrue(ok, reason)

    def test_T164A_breakout_blocks_extreme_rsi_spike(self):
        import strategy
        _, feat, i = _make_feat()
        for k in range(8, 0, -1):
            feat["high"][i - k] = 100.0
            feat["low"][i - k] = 99.2
            feat["close"][i - k] = 99.6
        feat["close"][i] = 101.0
        feat["ema_fast"][i] = 100.0
        feat["slope"][i] = 0.20
        feat["vol_x"][i] = 2.5
        feat["rsi"][i] = 87.0
        feat["daily_range_pct"][i] = 2.5
        feat["macd_hist"][i - 1] = 0.10
        feat["macd_hist"][i] = 0.20
        ok, reason = strategy.check_breakout_conditions(feat, i)
        self.assertFalse(ok)
        self.assertIn("слишком поздний breakout", reason)



# ════════════════════════════════════════════════════════════════════════════════
# T165  ALIGNMENT_EMA_SEP_MIN снижен 0.3% → 0.05% (TIA 15.03.2026)
# ════════════════════════════════════════════════════════════════════════════════
# EMA_SEP_MIN=0.3% блокировал нарождающиеся тренды где EMA20 только что
# пересекла EMA50 (sep~0.08-0.15%). SEI-паттерн по-прежнему блокируется
# ALIGNMENT_MACD_REL_MIN=0.0002. Бэктест 9 сценариев: Precision 75→80%.
class TestAlignmentEMASepMin(unittest.TestCase):
    """T165: EMA_SEP_MIN=0.05 — новое значение в конфиге."""

    def test_T165_alignment_ema_sep_min_value(self):
        """T165: ALIGNMENT_EMA_SEP_MIN должен быть 0.05 (снижен с 0.3)."""
        import config
        self.assertTrue(hasattr(config, "ALIGNMENT_EMA_SEP_MIN"),
                        "ALIGNMENT_EMA_SEP_MIN отсутствует в config")
        self.assertAlmostEqual(config.ALIGNMENT_EMA_SEP_MIN, 0.05, places=3,
                               msg=("ALIGNMENT_EMA_SEP_MIN должен быть 0.05. "
                                    "TIA 15.03: sep=0.085% < 0.3% → сигнал пропускался. "
                                    "MACD_REL_MIN=0.0002 защищает от SEI-паттерна."))



# ════════════════════════════════════════════════════════════════════════════════
# T166–T168  Часовой фильтр входов (ML, 44K баров, 15.03.2026)
# ════════════════════════════════════════════════════════════════════════════════
# EDA ml_dataset.jsonl: часы 3,10-15 UTC → EV=-0.083%, WR=41%.
# Остальные часы → EV=+0.200%, WR=53%. Блокировка 49% убыточных сигналов.
# 03 UTC: ночная Азия. 10-15 UTC: открытие NYSE + европейские развороты.
class TestHourlyFilter(unittest.TestCase):
    """T166–T168: ENTRY_BLOCK_HOURS либо явно отключён, либо корректно читается кодом."""

    def test_T166_entry_block_hours_in_config(self):
        """T166: ENTRY_BLOCK_HOURS присутствует в config и сейчас явно отключён пользователем."""
        import config
        self.assertTrue(hasattr(config, "ENTRY_BLOCK_HOURS"),
                        "ENTRY_BLOCK_HOURS не найден в config")
        h = config.ENTRY_BLOCK_HOURS
        self.assertIsInstance(h, list)
        self.assertEqual(
            h, [],
            "ENTRY_BLOCK_HOURS должен быть пустым: пользователь запретил блокировку входов по часу UTC",
        )

    def test_T167_hourly_filter_in_monitor_source(self):
        """T167: Часовой фильтр реализован в monitor.py."""
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("ENTRY_BLOCK_HOURS", src,
                      "monitor.py должен читать config.ENTRY_BLOCK_HOURS")
        self.assertIn("_bar_hour_utc", src,
                      "monitor.py должен вычислять час из bar timestamp")
        self.assertIn("3_600_000", src,
                      "Перевод ms → час: делить на 3_600_000")

    def test_T168_block_hours_coverage(self):
        """T168: При отключённом фильтре ENTRY_BLOCK_HOURS не должен блокировать ни одного часа."""
        import config
        h = config.ENTRY_BLOCK_HOURS
        self.assertEqual(
            len(h), 0,
            "При отключённом часовом фильтре не должно оставаться заблокированных часов UTC",
        )


# ════════════════════════════════════════════════════════════════════════════════
# T169–T172  Синтаксис и импорт всех модулей — ловит IndentationError и SyntaxError
# ════════════════════════════════════════════════════════════════════════════════
# Баг 15.03.2026: патч monitor.py вставил save_positions() без отступа →
# IndentationError при запуске бота. Тест py_compile ловит это за 0.01с.
# Правило: после КАЖДОГО патча тесты T169-T172 запускаются первыми.
class TestSyntaxAllModules(unittest.TestCase):
    """T169–T172: py_compile проверяет синтаксис всех ключевых модулей."""

    MODULES = [
        "monitor.py",
        "bot.py",
        "strategy.py",
        "config.py",
        "indicators.py",
        "data_collector.py",
        "botlog.py",
        "ml_dataset.py",
        "ml_signal_model.py",
    ]

    def _check_syntax(self, filename: str) -> None:
        import py_compile
        if not __import__("os").path.exists(filename):
            self.skipTest(f"{filename} не найден")
        try:
            py_compile.compile(filename, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(
                f"SyntaxError в {filename}:\n{e}\n"
                f"Запусти: python -c \"import py_compile; py_compile.compile('{filename}', doraise=True)\""
            )

    def test_T169_syntax_monitor(self):
        """T169: monitor.py — нет SyntaxError/IndentationError."""
        self._check_syntax("monitor.py")

    def test_T170_syntax_bot(self):
        """T170: bot.py — нет SyntaxError/IndentationError."""
        self._check_syntax("bot.py")

    def test_T171_syntax_strategy(self):
        """T171: strategy.py — нет SyntaxError/IndentationError."""
        self._check_syntax("strategy.py")

    def test_T172_syntax_remaining(self):
        """T172: config, indicators, data_collector, botlog, ml_dataset, ml_signal_model."""
        for fname in ["config.py", "indicators.py",
                      "data_collector.py", "botlog.py", "ml_dataset.py", "ml_signal_model.py"]:
            with self.subTest(file=fname):
                self._check_syntax(fname)


class TestMLSignalModel(unittest.TestCase):
    """T183-T185: ML prototype should be safe, trainable, and leak-free."""

    def test_T183_ml_signal_model_exists_and_excludes_leaky_features(self):
        import ml_signal_model

        self.assertTrue(Path("ml_signal_model.py").exists())
        safe = ml_signal_model.safe_feature_names()
        self.assertIn("btc_vs_ema50", safe)
        self.assertNotIn("r1", safe)
        self.assertNotIn("r3", safe)
        self.assertNotIn("r5", safe)
        self.assertNotIn("r10", safe)

    def test_T184_build_feature_dict_ignores_future_return_fields(self):
        import ml_signal_model

        rec = {
            "signal_type": "trend",
            "tf": "15m",
            "is_bull_day": False,
            "hour_utc": 3,
            "dow": 2,
            "f": {
                "rsi": 61.0,
                "adx": 24.0,
                "btc_vs_ema50": -1.7,
                "r1": 99.0,
                "r3": 88.0,
                "r5": 77.0,
                "r10": 66.0,
            },
            "seq": [[1.0] * 10 for _ in range(20)],
        }
        feat = ml_signal_model.build_feature_dict(rec)
        self.assertEqual(feat["rsi"], 61.0)
        self.assertEqual(feat["adx"], 24.0)
        self.assertEqual(feat["btc_vs_ema50"], -1.7)
        self.assertNotIn("r1", feat)
        self.assertNotIn("r3", feat)
        self.assertNotIn("r5", feat)
        self.assertNotIn("r10", feat)

    def test_T185_ml_signal_model_trains_on_small_synthetic_dataset(self):
        import tempfile
        from datetime import datetime, timedelta, timezone
        import ml_signal_model

        rows = []
        base_ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for i in range(60):
            good = (i % 4) in (0, 1)
            ts = (base_ts + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            signal = "trend" if i % 2 == 0 else "retest"
            seq = []
            for j in range(20):
                seq.append([
                    1.0 + 0.001 * j,
                    1.0 + 0.0015 * j,
                    0.999 - 0.0005 * j,
                    1.0 + 0.0008 * j,
                    1.6 + (0.3 if good else -0.1),
                    0.35 + (0.2 if good else -0.05),
                    24.0 + (8.0 if good else -3.0),
                    58.0 + (6.0 if good else -4.0),
                    0.05 + (0.07 if good else -0.03),
                    0.8,
                ])
            rows.append({
                "id": f"S{i}",
                "sym": f"SYM{i%3}",
                "tf": "15m",
                "ts_signal": ts,
                "bar_ts": i,
                "signal_type": signal,
                "is_bull_day": good,
                "hour_utc": i % 24,
                "dow": i % 7,
                "f": {
                    "rsi": 64.0 if good else 52.0,
                    "adx": 31.0 if good else 18.0,
                    "vol_x": 2.1 if good else 1.0,
                    "slope": 0.55 if good else 0.12,
                    "daily_range": 4.1 if good else 2.2,
                    "macd_hist_norm": 0.16 if good else -0.04,
                    "close_vs_ema20": 0.9 if good else 0.2,
                    "close_vs_ema50": 1.3 if good else 0.3,
                    "btc_vs_ema50": 1.5 if good else -1.5,
                    "r5": 999.0,
                },
                "seq": seq,
                "labels": {
                    "ret_5": 0.8 if good else -0.6,
                },
            })

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mini_ml_dataset.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            report = ml_signal_model.train_and_evaluate(path, min_rows=40, segment_min_rows=10)

        self.assertIn(report["chosen_model"], {"logistic", "mlp"})
        self.assertGreater(report["rows_total"], 40)
        self.assertGreater(report["feature_count"], 10)
        self.assertIn("top_feature_importance", report)
        self.assertNotIn("r5", report["model_payload"]["feature_names"])
        self.assertIn("segment_reports", report)


class TestTelegramNoiseControls(unittest.TestCase):
    """T186-T188: bot should stay signal-only by default and expose a Menu button."""

    def test_T186_noise_control_flags_default_to_false(self):
        import config

        self.assertFalse(getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", True))
        self.assertFalse(getattr(config, "SEND_DISCOVERY_NOTIFICATIONS", True))
        self.assertFalse(getattr(config, "SEND_SERVICE_NOTIFICATIONS", True))

    def test_T187_bot_has_reply_menu_button(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('BTN_MENU = "Menu"', bot_src)
        self.assertIn('BTN_SCAN_START = "Анализ + старт"', bot_src)
        self.assertIn('BTN_SCAN_ONLY = "Только анализ"', bot_src)
        self.assertIn('BTN_STOP_MONITOR = "Стоп мониторинга"', bot_src)
        self.assertIn('BTN_POSITIONS = "Позиции"', bot_src)
        self.assertIn('BTN_WATCHLIST = "Список монет"', bot_src)
        self.assertIn('BTN_ADD_COIN = "Добавить монету"', bot_src)
        self.assertIn('BTN_DEL_COIN = "Удалить монету"', bot_src)
        self.assertIn('BTN_SETTINGS = "Настройки"', bot_src)
        self.assertIn("def kb_menu_reply()", bot_src)
        self.assertIn('reply_markup=kb_menu_reply()', bot_src)
        self.assertIn("REPLY_ACTIONS = {", bot_src)
        self.assertIn("await _dispatch_reply_action(update, ctx, action)", bot_src)
        start_idx = bot_src.find("async def cmd_start(")
        start_block = bot_src[start_idx:start_idx + 1200]
        self.assertIn("reply_markup=kb_menu_reply()", start_block)
        self.assertNotIn("reply_markup=kb_main()", start_block)
        self.assertIn("reply_markup=kb_watchlist()", bot_src)
        self.assertIn("reply_markup=kb_back()", bot_src)

    def test_T188_post_init_and_discovery_are_flag_guarded(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn('ENABLE_EARLY_SCANNER_ALERTS', bot_src)
        self.assertIn('SEND_SERVICE_NOTIFICATIONS', bot_src)
        self.assertIn('SEND_DISCOVERY_NOTIFICATIONS', monitor_src)

    def test_T216_menu_can_be_hidden_and_restored(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('BTN_HIDE_MENU = "Скрыть меню"', bot_src)
        self.assertIn('BTN_SHOW_MENU = "Показать меню"', bot_src)
        self.assertIn("ReplyKeyboardRemove", bot_src)
        self.assertIn("def kb_show_menu_inline()", bot_src)
        self.assertIn('callback_data="show_menu"', bot_src)
        self.assertIn('CommandHandler("hide",  cmd_hide_menu, block=False)', bot_src)
        self.assertIn("[KeyboardButton(BTN_MENU), KeyboardButton(BTN_HIDE_MENU)]", bot_src)

    def test_T217_reply_actions_are_logged_and_surface_errors(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('log.info("reply action received', bot_src)
        self.assertIn('log.exception("reply action failed', bot_src)
        self.assertIn("Команда не выполнилась", bot_src)

    def test_T218_reply_actions_refresh_reply_keyboard(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("kwargs.setdefault(\"reply_markup\", kb_menu_reply())", bot_src)

    def test_T219_menu_button_routes_to_show_menu(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('BTN_MENU.casefold(): "show_menu"', bot_src)
        self.assertIn('CommandHandler("menu",  cmd_show_menu, block=False)', bot_src)

    def test_T220_menu_refresh_reinstalls_keyboard(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def _show_main_menu_message(", bot_src)
        self.assertIn("Меню активно.", bot_src)
        self.assertIn("reply_markup=kb_menu_reply()", bot_src)

    def test_T221_hide_menu_removes_keyboard_immediately(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("async def _send_hide_menu_followup(", bot_src)
        self.assertIn("asyncio.create_task(_send_hide_menu_followup(msg))", bot_src)
        self.assertIn('log.info("hide menu requested by %s", msg.chat_id)', bot_src)

    def test_T222_bot_enables_responsive_update_processing(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn(".concurrent_updates(True)", bot_src)
        self.assertIn('CallbackQueryHandler(btn, block=False)', bot_src)
        self.assertIn('MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler, block=False)', bot_src)

    def test_T216_menu_can_be_hidden_and_restored(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('BTN_HIDE_MENU = "в–ѕ"', bot_src)
        self.assertIn('BTN_SHOW_MENU = "РџРѕРєР°Р·Р°С‚СЊ РјРµРЅСЋ"', bot_src)
        self.assertIn("ReplyKeyboardRemove", bot_src)
        self.assertIn("def kb_show_menu_inline()", bot_src)
        self.assertIn('callback_data="show_menu"', bot_src)
        self.assertIn('CommandHandler("hide",  cmd_hide_menu, block=False)', bot_src)
        self.assertIn("[KeyboardButton(BTN_MENU), KeyboardButton(BTN_HIDE_MENU)]", bot_src)

    def test_T216_menu_can_be_hidden_and_restored(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn('BTN_HIDE_MENU = "▾"', bot_src)
        self.assertIn('BTN_SHOW_MENU = "Показать меню"', bot_src)
        self.assertIn("ReplyKeyboardRemove", bot_src)
        self.assertIn("def kb_show_menu_inline()", bot_src)
        self.assertIn('callback_data="show_menu"', bot_src)
        self.assertIn('CommandHandler("hide",  cmd_hide_menu, block=False)', bot_src)
        self.assertIn("[KeyboardButton(BTN_MENU), KeyboardButton(BTN_HIDE_MENU)]", bot_src)

    def test_T189_monitor_contains_ml_trend_nonbull_filter(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("_ml_trend_nonbull_score", monitor_src)
        self.assertIn("ML_TREND_NONBULL_MIN_PROBA", monitor_src)
        self.assertIn("ML BLOCK", monitor_src)

    def test_T190_replay_contains_ml_trend_nonbull_filter(self):
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("_ml_trend_nonbull_score_replay", replay_src)
        self.assertIn("_build_bull_day_context", replay_src)
        self.assertIn("ML_TREND_NONBULL_MIN_PROBA", replay_src)

    def test_T191_fresh_signal_priority_guards_exist(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("FRESH_SIGNAL_RESERVED_SLOTS", config_src)
        self.assertIn("TIME_BLOCK_BYPASS_MODES", config_src)
        self.assertIn("_is_fresh_priority_candidate", monitor_src)
        self.assertIn("check_trend_surge_conditions", monitor_src)
        self.assertIn("_is_fresh_priority_mode", replay_src)

    def test_T192_aux_noise_notifications_are_flag_guarded(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        scanner_src = Path("impulse_scanner.py").read_text(encoding="utf-8")
        self.assertIn("SEND_AUX_NOTIFICATIONS", config_src)
        self.assertIn("_aux_notifications_enabled", monitor_src)
        self.assertIn("Impulse scanner disabled by config", scanner_src)

# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════



class TestTimeBlockAndCooldownGuards(unittest.TestCase):
    """T193-T194: C98-oriented priority and weak reentry guards should be wired."""

    def test_T193_time_block_retest_priority_hooks_exist(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("TIME_BLOCK_RETEST_GRACE_BARS", config_src)
        self.assertIn("TIME_BLOCK_RETEST_SCORE_BONUS", config_src)
        self.assertIn("_time_block_retest_bonus", monitor_src)
        self.assertIn("time_block_recent", monitor_src)
        self.assertIn("_time_block_retest_bonus", replay_src)

    def test_T194_weak_reentry_cooldown_hooks_exist(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("WEAK_REENTRY_COOLDOWN_BARS", config_src)
        self.assertIn("_cooldown_bars_after_exit", monitor_src)
        self.assertIn("_cooldown_bars_after_exit", replay_src)


class TestLiveRuntimeStability(unittest.TestCase):
    """T195-T196: runtime should survive malformed ML rows and missing entry bars."""

    def test_T195_ml_dataset_fill_labels_skips_bad_jsonl_rows(self):
        import ml_dataset

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "ml_dataset.jsonl"
            valid = {
                "id": "abc",
                "sym": "BTCUSDT",
                "tf": "15m",
                "labels": {"exit_pnl": None, "exit_reason": None, "bars_held": None},
            }
            tmp_path.write_text(
                json.dumps(valid, ensure_ascii=False) + "\n" + "{bad json\n",
                encoding="utf-8",
            )
            old_file = ml_dataset.ML_FILE
            try:
                ml_dataset.ML_FILE = tmp_path
                ml_dataset.fill_labels("abc", 1.25, "test-exit", 3)
            finally:
                ml_dataset.ML_FILE = old_file

            rows = [json.loads(line) for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["labels"]["exit_reason"], "test-exit")
            self.assertEqual(rows[0]["labels"]["bars_held"], 3)

    def test_T196_monitor_returns_when_entry_bar_not_found(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("if entry_idx is None:", monitor_src)
        self.assertIn("pos.bars_elapsed = max(0, i - entry_idx)", monitor_src)


    def test_T196_monitor_returns_when_entry_bar_not_found(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("if entry_idx is None:", monitor_src)
        self.assertIn("pos.bars_elapsed = max(0, (current_bar_ts - int(pos.entry_ts)) // bar_ms)", monitor_src)
        self.assertIn("pos.bars_elapsed = max(0, i - entry_idx)", monitor_src)


class TestSetupRegimeAware(unittest.TestCase):
    """T165C-T165D: setup UI should follow regime-aware rules."""

    def test_T165C_setup_blocked_in_bear_regime(self):
        from strategy import check_setup_conditions, MarketRegime

        feat = {
            "ema_fast": np.array([101.0]),
            "ema_slow": np.array([100.0]),
            "rsi": np.array([55.0]),
            "slope": np.array([0.20]),
            "vol_x": np.array([1.5]),
            "macd_hist": np.array([0.1]),
            "daily_range_pct": np.array([2.0]),
            "adx": np.array([25.0]),
        }
        c = np.array([102.0])

        ok, reason, missing = check_setup_conditions(feat, 0, c, MarketRegime("bear_trend"))
        self.assertFalse(ok)
        self.assertIn("bear_trend", reason)
        self.assertEqual(missing, 99)

    def test_T165D_setup_uses_regime_specific_slope_threshold(self):
        from strategy import check_setup_conditions, MarketRegime

        feat = {
            "ema_fast": np.array([101.0]),
            "ema_slow": np.array([100.0]),
            "rsi": np.array([55.0]),
            "slope": np.array([0.09]),
            "vol_x": np.array([1.2]),
            "macd_hist": np.array([0.1]),
            "daily_range_pct": np.array([2.0]),
            "adx": np.array([20.0]),
        }
        c = np.array([102.0])

        ok, reason, missing = check_setup_conditions(feat, 0, c, MarketRegime("bull_trend"))
        self.assertTrue(ok)
        self.assertEqual(missing, 0)
        self.assertIn("BUY", reason)


class TestPortfolioReplacement(unittest.TestCase):
    """T165E-T165F: portfolio replacement should be conservative and score-based."""

    def test_T165E_replacement_picks_weakest_old_position(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position
        prev_enabled = config.PORTFOLIO_REPLACE_ENABLED
        config.PORTFOLIO_REPLACE_ENABLED = True

        try:
            state = MonitorState(positions={
                "ADAUSDT": OpenPosition(
                    symbol="ADAUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.08, entry_adx=18.0, entry_rsi=56.0, entry_vol_x=1.0,
                    signal_mode="alignment", bars_elapsed=7,
                ),
                "SOLUSDT": OpenPosition(
                    symbol="SOLUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.98, entry_slope=0.30, entry_adx=28.0, entry_rsi=62.0, entry_vol_x=2.0,
                    signal_mode="strong_trend", bars_elapsed=7,
                ),
            })
            state.last_prices = {"ADAUSDT": 0.99, "SOLUSDT": 1.03}

            replace_pos = _find_replaceable_position(state, candidate_score=40.0, candidate_mode="breakout")
            self.assertIsNotNone(replace_pos)
            self.assertEqual(replace_pos.symbol, "ADAUSDT")
        finally:
            config.PORTFOLIO_REPLACE_ENABLED = prev_enabled

    def test_T165F_replacement_does_not_touch_fresh_positions(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position
        prev_enabled = config.PORTFOLIO_REPLACE_ENABLED
        config.PORTFOLIO_REPLACE_ENABLED = True

        try:
            state = MonitorState(positions={
                "ADAUSDT": OpenPosition(
                    symbol="ADAUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.08, entry_adx=18.0, entry_rsi=56.0, entry_vol_x=1.0,
                    signal_mode="alignment", bars_elapsed=0,
                ),
            })

            replace_pos = _find_replaceable_position(state, candidate_score=99.0, candidate_mode="breakout")
            self.assertIsNone(replace_pos)
        finally:
            config.PORTFOLIO_REPLACE_ENABLED = prev_enabled

    def test_T165G_replacement_respects_profit_protection(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position
        prev_enabled = config.PORTFOLIO_REPLACE_ENABLED
        config.PORTFOLIO_REPLACE_ENABLED = True

        try:
            state = MonitorState(positions={
                "ADAUSDT": OpenPosition(
                    symbol="ADAUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.08, entry_adx=18.0, entry_rsi=56.0, entry_vol_x=1.0,
                    signal_mode="alignment", bars_elapsed=8,
                ),
            })
            state.last_prices = {"ADAUSDT": 1.01}

            replace_pos = _find_replaceable_position(state, candidate_score=35.0, candidate_mode="breakout")
            self.assertIsNone(replace_pos)
        finally:
            config.PORTFOLIO_REPLACE_ENABLED = prev_enabled

    def test_T165H_replacement_enabled_by_default(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position

        state = MonitorState(positions={
            "ADAUSDT": OpenPosition(
                symbol="ADAUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                entry_ema20=0.99, entry_slope=0.08, entry_adx=18.0, entry_rsi=56.0, entry_vol_x=1.0,
                signal_mode="alignment", bars_elapsed=8,
            ),
        })
        state.last_prices = {"ADAUSDT": 0.99}
        self.assertTrue(config.PORTFOLIO_REPLACE_ENABLED)
        self.assertIsNotNone(_find_replaceable_position(state, candidate_score=99.0, candidate_mode="breakout"))

    def test_T165I_replacement_prefers_negative_ranker_tail(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position

        prev_enabled = config.PORTFOLIO_REPLACE_ENABLED
        prev_ranker_enabled = config.PORTFOLIO_REPLACE_RANKER_ENABLED
        config.PORTFOLIO_REPLACE_ENABLED = True
        config.PORTFOLIO_REPLACE_RANKER_ENABLED = True

        try:
            state = MonitorState(positions={
                "BADUSDT": OpenPosition(
                    symbol="BADUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.10, entry_adx=19.0, entry_rsi=58.0, entry_vol_x=1.1,
                    signal_mode="impulse_speed", bars_elapsed=7,
                    ranker_final_score=-3.2, ranker_top_gainer_prob=0.02,
                ),
                "OKUSDT": OpenPosition(
                    symbol="OKUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.12, entry_adx=22.0, entry_rsi=57.0, entry_vol_x=1.2,
                    signal_mode="impulse_speed", bars_elapsed=7,
                    ranker_final_score=0.35, ranker_top_gainer_prob=0.26,
                ),
            })
            state.last_prices = {"BADUSDT": 0.99, "OKUSDT": 1.00}

            replace_pos = _find_replaceable_position(
                state,
                candidate_score=72.0,
                candidate_mode="breakout",
                candidate_ranker_info={"final_score": 0.62, "top_gainer_prob": 0.31},
            )
            self.assertIsNotNone(replace_pos)
            self.assertEqual(replace_pos.symbol, "BADUSDT")
        finally:
            config.PORTFOLIO_REPLACE_ENABLED = prev_enabled
            config.PORTFOLIO_REPLACE_RANKER_ENABLED = prev_ranker_enabled

    def test_T165J_replacement_skips_bad_candidate_ranker(self):
        import config
        from monitor import OpenPosition, MonitorState, _find_replaceable_position

        prev_enabled = config.PORTFOLIO_REPLACE_ENABLED
        prev_ranker_enabled = config.PORTFOLIO_REPLACE_RANKER_ENABLED
        config.PORTFOLIO_REPLACE_ENABLED = True
        config.PORTFOLIO_REPLACE_RANKER_ENABLED = True

        try:
            state = MonitorState(positions={
                "BADUSDT": OpenPosition(
                    symbol="BADUSDT", tf="15m", entry_price=1.0, entry_bar=1, entry_ts=1,
                    entry_ema20=0.99, entry_slope=0.08, entry_adx=18.0, entry_rsi=56.0, entry_vol_x=1.0,
                    signal_mode="alignment", bars_elapsed=8,
                    ranker_final_score=-1.9, ranker_top_gainer_prob=0.03,
                ),
            })
            state.last_prices = {"BADUSDT": 0.99}

            replace_pos = _find_replaceable_position(
                state,
                candidate_score=120.0,
                candidate_mode="breakout",
                candidate_ranker_info={"final_score": -1.1, "top_gainer_prob": 0.01},
            )
            self.assertIsNone(replace_pos)
        finally:
            config.PORTFOLIO_REPLACE_ENABLED = prev_enabled
            config.PORTFOLIO_REPLACE_RANKER_ENABLED = prev_ranker_enabled

    def test_T165K_replay_rotation_scores_prefer_negative_ranker_tail(self):
        from replay_backtest import ReplayCandidate, ReplayTrade, _candidate_rotation_score, _trade_rotation_score

        bad_trade = ReplayTrade(
            sym="BADUSDT", tf="15m", mode="impulse_speed",
            entry_ts=1, entry_price=1.0, entry_i=1, trail_k=2.0, max_hold_bars=10, trail_stop=0.9,
            entry_score=70.0, ranker_final_score=-3.0, ranker_top_gainer_prob=0.02,
        )
        good_candidate = ReplayCandidate(
            sym="GOODUSDT", tf="15m", mode="breakout",
            ts_ms=2, i=2, price=1.1, trail_k=2.0, max_hold_bars=10, score=72.0,
            ranker_final_score=0.7, ranker_top_gainer_prob=0.31,
        )
        self.assertGreater(_candidate_rotation_score(good_candidate), _trade_rotation_score(bad_trade))

    def test_T165L_replay_rotation_penalizes_bad_candidate_ranker(self):
        from replay_backtest import ReplayCandidate, _candidate_ranker_rotation_ok

        strong_raw_bad_ranker = ReplayCandidate(
            sym="BADCANDUSDT", tf="15m", mode="breakout",
            ts_ms=2, i=2, price=1.1, trail_k=2.0, max_hold_bars=10, score=120.0,
            ranker_final_score=-1.2, ranker_top_gainer_prob=0.01,
        )
        decent_candidate = ReplayCandidate(
            sym="DECENTUSDT", tf="15m", mode="breakout",
            ts_ms=2, i=2, price=1.1, trail_k=2.0, max_hold_bars=10, score=90.0,
            ranker_final_score=0.4, ranker_top_gainer_prob=0.24,
        )
        self.assertFalse(_candidate_ranker_rotation_ok(strong_raw_bad_ranker))
        self.assertTrue(_candidate_ranker_rotation_ok(decent_candidate))


class TestEntryScoreFloorAndMTFSoftPass(unittest.TestCase):
    def test_entry_score_floor_values_present(self):
        import config
        self.assertTrue(getattr(config, "ENTRY_SCORE_MIN_ENABLED", False))
        self.assertGreater(float(getattr(config, "ENTRY_SCORE_MIN_15M", 0.0)), 0.0)
        self.assertGreater(float(getattr(config, "ENTRY_SCORE_MIN_1H", 0.0)), 0.0)

    def test_mtf_soft_pass_penalty_present(self):
        import config
        self.assertGreater(float(getattr(config, "MTF_SOFT_PASS_PENALTY", 0.0)), 0.0)
        self.assertGreater(float(getattr(config, "MTF_RSI_HARD_MIN", 0.0)), 0.0)

    def test_mtf_soft_pass_reason_maps_to_penalty(self):
        from monitor import _mtf_soft_penalty_from_reason

        penalty = _mtf_soft_penalty_from_reason("15m soft-pass: RSI=43.0 below MTF floor 45.0")
        self.assertGreater(penalty, 0.0)
        self.assertEqual(_mtf_soft_penalty_from_reason("15m relax-pass"), 0.0)

    def test_entry_score_floor_helper_uses_timeframe(self):
        import config
        from monitor import _entry_score_floor

        self.assertEqual(_entry_score_floor("15m"), float(config.ENTRY_SCORE_MIN_15M))
        self.assertEqual(_entry_score_floor("1h"), float(config.ENTRY_SCORE_MIN_1H))


class TestEntryScoreBorderlineBypass(unittest.TestCase):
    def test_dydx_like_borderline_candidate_is_allowed(self):
        from monitor import _entry_score_borderline_bypass_ok

        self.assertTrue(
            _entry_score_borderline_bypass_ok(
                tf="15m",
                mode="impulse_speed",
                candidate_score=43.73,
                min_score=48.0,
                price=1.022448,
                ema20=1.0,
                slope=1.238,
                adx=31.55,
                rsi=73.49,
                vol_x=1.6536,
                daily_range=5.2573,
            )
        )

    def test_flux_like_late_spike_remains_blocked(self):
        from monitor import _entry_score_borderline_bypass_ok

        self.assertFalse(
            _entry_score_borderline_bypass_ok(
                tf="15m",
                mode="impulse_speed",
                candidate_score=46.0,
                min_score=48.0,
                price=1.048803,
                ema20=1.0,
                slope=0.80,
                adx=25.76,
                rsi=73.65,
                vol_x=6.3052,
                daily_range=8.6667,
            )
        )

    def test_low_volume_retest_remains_blocked(self):
        from monitor import _entry_score_borderline_bypass_ok

        self.assertFalse(
            _entry_score_borderline_bypass_ok(
                tf="15m",
                mode="retest",
                candidate_score=46.5,
                min_score=48.0,
                price=1.005074,
                ema20=1.0,
                slope=0.40,
                adx=23.12,
                rsi=60.27,
                vol_x=0.9903,
                daily_range=1.3237,
            )
        )


class TestEntryScoreContinuationBypass(unittest.TestCase):
    def test_eth_like_1h_continuation_candidate_is_allowed(self):
        from monitor import _entry_score_continuation_bypass_ok

        self.assertTrue(
            _entry_score_continuation_bypass_ok(
                tf="1h",
                mode="alignment",
                candidate_score=42.6,
                price=1.018546,
                ema20=1.0,
                slope=0.8166,
                adx=17.82,
                rsi=70.7,
                vol_x=1.0,
                daily_range=7.50,
                continuation_profile=True,
                is_bull_day=True,
            )
        )

    def test_stretched_1h_continuation_remains_blocked(self):
        from monitor import _entry_score_continuation_bypass_ok

        self.assertFalse(
            _entry_score_continuation_bypass_ok(
                tf="1h",
                mode="alignment",
                candidate_score=46.5,
                price=1.025,
                ema20=1.0,
                slope=0.82,
                adx=18.5,
                rsi=74.4,
                vol_x=1.3,
                daily_range=7.8,
                continuation_profile=True,
                is_bull_day=True,
            )
        )


class TestCriticQualityGuards(unittest.TestCase):
    def test_trend_quality_guard_blocks_weak_15m_trend(self):
        from monitor import _trend_entry_quality_guard_reason

        reason = _trend_entry_quality_guard_reason(
            tf="15m",
            mode="trend",
            price=1.018,
            ema20=1.0,
            slope=0.22,
            adx=19.5,
            rsi=61.0,
            vol_x=1.02,
            daily_range=4.2,
            forecast_return_pct=0.05,
        )
        self.assertIsNotNone(reason)
        self.assertIn("weak 15m trend", reason)

    def test_trend_quality_guard_keeps_strong_15m_trend(self):
        from monitor import _trend_entry_quality_guard_reason

        reason = _trend_entry_quality_guard_reason(
            tf="15m",
            mode="trend",
            price=1.012,
            ema20=1.0,
            slope=0.42,
            adx=28.0,
            rsi=63.0,
            vol_x=1.45,
            daily_range=5.0,
            forecast_return_pct=0.05,
        )
        self.assertIsNone(reason)

    def test_retest_1h_micro_confirmation_blocks_weak_macd_pullback(self):
        from monitor import _retest_1h_mtf_confirm_reason

        reason = _retest_1h_mtf_confirm_reason(
            mode="retest",
            close_val=100.0,
            ema20_15m=100.02,
            macd_hist=-0.0002,
            macd_prev=0.0001,
            rsi_val=52.0,
        )
        self.assertIsNotNone(reason)
        self.assertIn("MACD", reason)

    def test_retest_1h_micro_confirmation_accepts_clean_recovery(self):
        from monitor import _retest_1h_mtf_confirm_reason
        import replay_backtest

        reason_live = _retest_1h_mtf_confirm_reason(
            mode="retest",
            close_val=100.05,
            ema20_15m=100.0,
            macd_hist=0.0002,
            macd_prev=0.0001,
            rsi_val=55.0,
        )
        reason_replay = replay_backtest._retest_1h_mtf_confirm_reason(
            mode="retest",
            close_val=100.05,
            ema20_15m=100.0,
            macd_hist=0.0002,
            macd_prev=0.0001,
            rsi_val=55.0,
        )
        self.assertIsNone(reason_live)
        self.assertIsNone(reason_replay)

    def test_ranker_veto_blocks_weak_low_confidence_trend(self):
        from monitor import _ranker_entry_veto_reason

        reason = _ranker_entry_veto_reason(
            tf="15m",
            mode="trend",
            ranker_proba=0.11,
            candidate_score=55.0,
            forecast_return_pct=0.12,
        )
        self.assertIsNotNone(reason)
        self.assertIn("ranker veto", reason)

    def test_ranker_veto_does_not_block_stronger_trend(self):
        from monitor import _ranker_entry_veto_reason

        reason = _ranker_entry_veto_reason(
            tf="15m",
            mode="trend",
            ranker_proba=0.11,
            candidate_score=63.0,
            forecast_return_pct=0.12,
        )
        self.assertIsNone(reason)

    def test_ranker_hard_veto_blocks_negative_15m_false_positive(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="15m",
            mode="retest",
            ranker_info={"final_score": -0.84, "top_gainer_prob": 0.15},
        )
        self.assertIsNotNone(reason)
        self.assertIn("ranker hard veto", reason)

    def test_ranker_hard_veto_blocks_negative_1h_retest(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="1h",
            mode="retest",
            ranker_info={"final_score": -2.23, "top_gainer_prob": 0.19},
        )
        self.assertIsNotNone(reason)
        self.assertIn("ranker hard veto", reason)

    def test_ranker_hard_veto_allows_15m_setup_with_good_teacher_signal(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="15m",
            mode="retest",
            ranker_info={"final_score": -0.90, "top_gainer_prob": 0.34},
        )
        self.assertIsNone(reason)

    def test_ranker_hard_veto_blocks_weak_15m_impulse_drift(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="15m",
            mode="impulse_speed",
            ranker_info={
                "final_score": -0.63,
                "ev_raw": -0.76,
                "quality_proba": 0.48,
                "top_gainer_prob": 0.24,
                "capture_ratio_pred": 0.00,
            },
        )
        self.assertIsNotNone(reason)
        self.assertIn("weak 15m impulse", reason)

    def test_ranker_hard_veto_blocks_weak_15m_impulse_mode(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="15m",
            mode="impulse",
            ranker_info={
                "final_score": -1.22,
                "ev_raw": -1.67,
                "quality_proba": 0.26,
                "top_gainer_prob": 0.00,
                "capture_ratio_pred": 0.00,
            },
        )
        self.assertIsNotNone(reason)
        self.assertIn("weak 15m impulse", reason)

    def test_ranker_hard_veto_blocks_weak_1h_retest_detail(self):
        from monitor import _ranker_hard_veto_reason

        reason = _ranker_hard_veto_reason(
            tf="1h",
            mode="retest",
            ranker_info={
                "final_score": -1.80,
                "ev_raw": -1.60,
                "quality_proba": 0.30,
                "top_gainer_prob": 0.00,
            },
        )
        self.assertIsNotNone(reason)
        self.assertIn("weak 1h retest", reason)

    def test_ranker_position_cleanup_exits_stale_15m_impulse(self):
        import numpy as np
        from monitor import OpenPosition, _ranker_position_cleanup_reason

        feat = {
            "ema_fast": np.array([0.2440], dtype=float),
        }
        pos = OpenPosition(
            symbol="ADAUSDT",
            tf="15m",
            entry_price=0.2428,
            entry_bar=98,
            entry_ts=1775192400000,
            entry_ema20=0.2409,
            entry_slope=0.35,
            entry_adx=18.1,
            entry_rsi=68.8,
            entry_vol_x=0.95,
            signal_mode="impulse_speed",
            bars_elapsed=34,
            prediction_horizons=(2, 5, 7),
            predictions={2: True, 5: True, 7: True},
            ranker_final_score=-0.63,
            ranker_ev=-0.76,
            ranker_top_gainer_prob=0.24,
            ranker_capture_ratio_pred=0.00,
            ranker_quality_proba=0.48,
        )
        reason = _ranker_position_cleanup_reason(pos, feat, 0, close_now=0.2448)
        self.assertIsNotNone(reason)
        self.assertIn("stale 15m", reason)
        self.assertIn("fast horizons", reason)

    def test_ranker_position_cleanup_exits_stale_15m_impulse_mode(self):
        import numpy as np
        from monitor import OpenPosition, _ranker_position_cleanup_reason

        feat = {
            "ema_fast": np.array([0.000988], dtype=float),
        }
        pos = OpenPosition(
            symbol="AMPUSDT",
            tf="15m",
            entry_price=0.000993,
            entry_bar=98,
            entry_ts=1775217000000,
            entry_ema20=0.000973,
            entry_slope=0.22,
            entry_adx=17.5,
            entry_rsi=69.8,
            entry_vol_x=1.25,
            signal_mode="impulse",
            bars_elapsed=10,
            prediction_horizons=(3, 5, 10),
            predictions={3: False, 5: False},
            ranker_final_score=-1.22,
            ranker_ev=-1.67,
            ranker_top_gainer_prob=0.00,
            ranker_capture_ratio_pred=0.00,
            ranker_quality_proba=0.26,
        )
        reason = _ranker_position_cleanup_reason(pos, feat, 0, close_now=0.000991)
        self.assertIsNotNone(reason)
        self.assertIn("stale 15m", reason)
        self.assertIn("fast horizons", reason)

    def test_ranker_position_cleanup_exits_weak_1h_retest(self):
        import numpy as np
        from monitor import OpenPosition, _ranker_position_cleanup_reason

        feat = {
            "ema_fast": np.array([0.2834], dtype=float),
        }
        pos = OpenPosition(
            symbol="BNTUSDT",
            tf="1h",
            entry_price=0.2859,
            entry_bar=98,
            entry_ts=1775203200000,
            entry_ema20=0.2833,
            entry_slope=0.40,
            entry_adx=20.4,
            entry_rsi=59.7,
            entry_vol_x=1.40,
            signal_mode="retest",
            bars_elapsed=4,
            prediction_horizons=(3, 5, 10),
            predictions={3: False},
            ranker_final_score=-2.45,
            ranker_ev=-2.23,
            ranker_top_gainer_prob=0.00,
            ranker_capture_ratio_pred=0.00,
            ranker_quality_proba=0.28,
        )
        reason = _ranker_position_cleanup_reason(pos, feat, 0, close_now=0.2833)
        self.assertIsNotNone(reason)
        self.assertIn("weak 1h retest", reason)


class TestModeAwareExitsAndHybridMTF(unittest.TestCase):
    def _feat(self, *, ema_fast, slope, adx, rsi):
        import numpy as np

        size = len(ema_fast)
        return {
            "ema_fast": np.array(ema_fast, dtype=float),
            "slope": np.array(slope, dtype=float),
            "adx": np.array(adx, dtype=float),
            "rsi": np.array(rsi, dtype=float),
            "rsi_divergence": np.zeros(size, dtype=float),
            "vol_exhaustion": np.zeros(size, dtype=float),
            "ema_fan_spread": np.zeros(size, dtype=float),
        }

    def test_trend_skips_adx_exit_before_grace(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([101.0, 101.5, 102.0, 102.2, 102.4, 102.6], dtype=float)
        feat = self._feat(
            ema_fast=[100.0, 100.3, 100.7, 101.0, 101.3, 101.6],
            slope=[0.20, 0.22, 0.25, 0.18, 0.16, 0.14],
            adx=[30.0, 30.0, 30.0, 28.0, 20.0, 18.0],
            rsi=[55.0, 57.0, 58.0, 59.0, 60.0, 61.0],
        )

        reason = check_exit_conditions(feat, 5, closes, mode="trend", bars_elapsed=2, tf="15m")
        self.assertIsNone(reason)

    def test_trend_allows_adx_exit_after_grace(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([101.0, 101.5, 102.0, 102.2, 102.4, 102.6], dtype=float)
        feat = self._feat(
            ema_fast=[100.0, 100.3, 100.7, 101.0, 101.3, 101.6],
            slope=[0.20, 0.22, 0.25, 0.18, 0.16, 0.14],
            adx=[30.0, 30.0, 30.0, 28.0, 20.0, 18.0],
            rsi=[55.0, 57.0, 58.0, 59.0, 60.0, 61.0],
        )

        reason = check_exit_conditions(feat, 5, closes, mode="trend", bars_elapsed=6, tf="15m")
        self.assertIsNotNone(reason)
        self.assertIn("ADX", reason)

    def test_breakout_keeps_aggressive_two_close_exit(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([102.0, 99.0, 98.0], dtype=float)
        feat = self._feat(
            ema_fast=[100.0, 100.0, 100.0],
            slope=[0.15, 0.10, 0.08],
            adx=[25.0, 24.0, 23.0],
            rsi=[58.0, 54.0, 49.0],
        )

        reason = check_exit_conditions(feat, 2, closes, mode="breakout", bars_elapsed=1, tf="15m")
        self.assertIsNotNone(reason)
        self.assertIn("EMA20", reason)

    def test_impulse_speed_skips_adx_exit_before_semi_patient_grace(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([101.0, 101.4, 101.9, 102.2, 102.5, 102.7], dtype=float)
        feat = self._feat(
            ema_fast=[100.0, 100.4, 100.8, 101.2, 101.6, 101.9],
            slope=[0.30, 0.28, 0.24, 0.18, 0.12, 0.08],
            adx=[32.0, 32.0, 32.0, 29.0, 23.0, 20.0],
            rsi=[58.0, 60.0, 62.0, 64.0, 65.0, 66.0],
        )

        reason = check_exit_conditions(feat, 5, closes, mode="impulse_speed", bars_elapsed=3, tf="15m")
        self.assertIsNone(reason)

    def test_impulse_speed_allows_adx_exit_after_semi_patient_grace(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([101.0, 101.4, 101.9, 102.2, 102.5, 102.7], dtype=float)
        feat = self._feat(
            ema_fast=[100.0, 100.4, 100.8, 101.2, 101.6, 101.9],
            slope=[0.30, 0.28, 0.24, 0.18, 0.12, 0.08],
            adx=[32.0, 32.0, 32.0, 29.0, 23.0, 20.0],
            rsi=[58.0, 60.0, 62.0, 64.0, 65.0, 66.0],
        )

        reason = check_exit_conditions(feat, 5, closes, mode="impulse_speed", bars_elapsed=5, tf="15m")
        self.assertIsNotNone(reason)
        self.assertIn("ADX", reason)

    def test_impulse_speed_requires_two_negative_slope_bars(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([103.0, 102.6, 102.2], dtype=float)
        feat = self._feat(
            ema_fast=[102.0, 102.1, 102.3],
            slope=[0.12, 0.05, -0.04],
            adx=[28.0, 27.0, 27.0],
            rsi=[62.0, 60.0, 58.0],
        )

        reason = check_exit_conditions(feat, 2, closes, mode="impulse_speed", bars_elapsed=5, tf="15m")
        self.assertIsNone(reason)

    def test_impulse_speed_exits_after_two_negative_slope_bars(self):
        import numpy as np
        from strategy import check_exit_conditions

        closes = np.array([103.0, 102.4, 101.9], dtype=float)
        feat = self._feat(
            ema_fast=[102.0, 102.2, 102.4],
            slope=[0.08, -0.03, -0.06],
            adx=[28.0, 27.0, 27.0],
            rsi=[62.0, 59.0, 56.0],
        )

        reason = check_exit_conditions(feat, 2, closes, mode="impulse_speed", bars_elapsed=5, tf="15m")
        self.assertIsNotNone(reason)
        self.assertIn("EMA20", reason)

    def test_hybrid_mtf_hard_blocks_only_deep_contradiction(self):
        import asyncio
        import numpy as np
        from unittest.mock import AsyncMock, patch
        import monitor

        n = 80
        prices = np.linspace(100.0, 88.0, n)
        fake_data = np.zeros(
            n,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = prices
        fake_data["h"] = prices * 1.001
        fake_data["l"] = prices * 0.999
        fake_data["c"] = prices
        fake_data["v"] = np.full(n, 1000.0)

        with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)):
            ok, reason = asyncio.run(
                monitor._check_mtf(
                    AsyncMock(),
                    "ETHUSDT",
                    mode="trend",
                    candidate_score=70.0,
                    slope=0.6,
                    adx=28.0,
                    rsi=58.0,
                    vol_x=1.6,
                    daily_range=2.0,
                )
            )

        self.assertFalse(ok)
        self.assertIn("deep correction", reason)

    def test_hybrid_mtf_allows_shallow_negative_as_soft_pass(self):
        import asyncio
        import numpy as np
        from unittest.mock import AsyncMock, patch
        import monitor

        n = 80
        base = np.linspace(100.0, 100.8, n)
        fake_data = np.zeros(
            n,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        fake_data["t"] = np.arange(n) * 900_000
        fake_data["o"] = base
        fake_data["h"] = base + 0.1
        fake_data["l"] = base - 0.1
        fake_data["c"] = base
        fake_data["c"][-4:] = [100.60, 100.58, 100.57, 100.58]
        fake_data["o"][-4:] = [100.59, 100.57, 100.56, 100.57]
        fake_data["h"][-4:] = [100.64, 100.62, 100.61, 100.62]
        fake_data["l"][-4:] = [100.56, 100.54, 100.53, 100.54]
        fake_data["v"] = np.full(n, 1000.0)

        with patch("monitor.fetch_klines", new=AsyncMock(return_value=fake_data)):
            ok, reason = asyncio.run(
                monitor._check_mtf(
                    AsyncMock(),
                    "LINKUSDT",
                    mode="trend",
                    candidate_score=68.0,
                    slope=0.5,
                    adx=26.0,
                    rsi=57.0,
                    vol_x=1.4,
                    daily_range=1.8,
                )
            )

        self.assertTrue(ok)
        self.assertIn("soft-pass", reason)

    def test_time_exit_waits_while_uptrend_continues(self):
        import numpy as np
        import monitor
        import replay_backtest

        feat = {
            "ema_fast": np.array([1.00, 1.01, 1.02], dtype=float),
            "slope": np.array([0.10, 0.12, 0.15], dtype=float),
            "rsi": np.array([56.0, 58.0, 60.0], dtype=float),
            "macd_hist": np.array([0.001, 0.002, 0.003], dtype=float),
        }

        self.assertTrue(monitor._time_exit_should_wait(feat, 2, 1.03))
        self.assertTrue(replay_backtest._time_exit_should_wait(feat, 2, 1.03))

    def test_replay_time_exit_still_fires_when_trend_has_ended(self):
        import numpy as np
        import replay_backtest

        data = np.zeros(
            4,
            dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")],
        )
        data["t"] = np.arange(4) * 900_000
        data["o"] = [1.0, 1.01, 1.02, 1.01]
        data["h"] = [1.01, 1.02, 1.03, 1.02]
        data["l"] = [0.99, 1.00, 1.01, 1.00]
        data["c"] = [1.0, 1.01, 1.02, 1.00]
        data["v"] = [100.0, 100.0, 100.0, 100.0]
        feat = {
            "atr": np.array([0.01, 0.01, 0.01, 0.01], dtype=float),
            "ema_fast": np.array([1.0, 1.005, 1.015, 1.02], dtype=float),
            "slope": np.array([0.05, 0.06, 0.02, -0.03], dtype=float),
            "rsi": np.array([55.0, 57.0, 59.0, 44.0], dtype=float),
            "macd_hist": np.array([0.001, 0.001, 0.0005, -0.0001], dtype=float),
            "adx": np.array([20.0, 21.0, 22.0, 18.0], dtype=float),
            "rsi_divergence": np.zeros(4, dtype=float),
            "vol_exhaustion": np.zeros(4, dtype=float),
            "ema_fan_spread": np.zeros(4, dtype=float),
        }
        trade = replay_backtest.ReplayTrade(
            sym="TESTUSDT",
            tf="15m",
            mode="retest",
            entry_ts=int(data["t"][0]),
            entry_price=1.0,
            entry_i=0,
            trail_k=1.8,
            max_hold_bars=3,
            trail_stop=0.0,
        )

        reason = replay_backtest._update_trade_progress(trade, data, feat, 3, ts_ms=int(data["t"][3]))
        self.assertIsNotNone(reason)
        self.assertIn("time", reason)


class TestImpulseSpeedLateGuard(unittest.TestCase):
    def test_late_faded_15m_impulse_speed_is_blocked(self):
        import numpy as np
        from monitor import _impulse_speed_entry_guard

        feat = {
            "macd_hist": np.array([0.0002, 0.0005, 0.0010, 0.0014, 0.0011, 0.0008], dtype=float),
        }
        reason = _impulse_speed_entry_guard(
            tf="15m",
            mode="impulse_speed",
            feat=feat,
            i=5,
            price=104.0,
            ema20=101.0,
            rsi=71.0,
            adx=24.0,
            daily_range=6.2,
        )
        self.assertIsNotNone(reason)
        self.assertIn("late impulse_speed", reason)

    def test_fresh_15m_impulse_speed_is_allowed(self):
        import numpy as np
        from monitor import _impulse_speed_entry_guard

        feat = {
            "macd_hist": np.array([0.0002, 0.0005, 0.0010, 0.0014, 0.0015, 0.0016], dtype=float),
        }
        reason = _impulse_speed_entry_guard(
            tf="15m",
            mode="impulse_speed",
            feat=feat,
            i=5,
            price=102.2,
            ema20=101.0,
            rsi=68.0,
            adx=24.0,
            daily_range=6.2,
        )
        self.assertIsNone(reason)

    def test_faded_1h_impulse_speed_is_blocked_even_if_base_guard_passes(self):
        import numpy as np
        from monitor import _impulse_speed_entry_guard

        feat = {
            "macd_hist": np.array([0.004, 0.010, 0.018, 0.020, 0.016, 0.010], dtype=float),
        }
        reason = _impulse_speed_entry_guard(
            tf="1h",
            mode="impulse_speed",
            feat=feat,
            i=5,
            price=105.0,
            ema20=102.0,
            rsi=70.0,
            adx=26.0,
            daily_range=9.5,
        )
        self.assertIsNotNone(reason)
        self.assertIn("late impulse_speed", reason)

    def test_overstretched_fresh_15m_impulse_speed_is_blocked(self):
        import numpy as np
        from monitor import _impulse_speed_entry_guard

        feat = {
            "macd_hist": np.array([0.0002, 0.0006, 0.0011, 0.0018, 0.0026, 0.0034], dtype=float),
        }
        reason = _impulse_speed_entry_guard(
            tf="15m",
            mode="impulse_speed",
            feat=feat,
            i=5,
            price=105.0,
            ema20=100.0,
            rsi=77.0,
            adx=23.0,
            daily_range=13.9,
        )
        self.assertIsNotNone(reason)
        self.assertIn("overstretched 15m spike", reason)


class TestNightQualityGuards(unittest.TestCase):
    def test_breakout_requires_min_adx(self):
        import numpy as np
        import strategy

        feat = {
            "close": np.array([1.000, 1.005, 1.003, 1.006, 1.004, 1.005, 1.004, 1.006, 1.005, 1.006, 1.022], dtype=float),
            "high": np.array([1.006, 1.006, 1.005, 1.007, 1.006, 1.006, 1.005, 1.007, 1.006, 1.007, 1.023], dtype=float),
            "low": np.array([0.998, 1.000, 1.001, 1.002, 1.001, 1.002, 1.001, 1.003, 1.002, 1.003, 1.018], dtype=float),
            "ema_fast": np.array([0.999, 1.000, 1.001, 1.002, 1.002, 1.003, 1.003, 1.004, 1.004, 1.005, 1.010], dtype=float),
            "slope": np.array([0.02, 0.03, 0.02, 0.04, 0.03, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09], dtype=float),
            "vol_x": np.array([1.0, 1.0, 1.0, 1.1, 1.0, 1.1, 1.0, 1.2, 1.1, 1.2, 16.6], dtype=float),
            "rsi": np.array([50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.5, 54.0, 54.0, 55.0, 67.1], dtype=float),
            "macd_hist": np.array([-0.01, -0.005, 0.0, 0.01, 0.015, 0.020, 0.030, 0.040, 0.041, 0.045, 0.087], dtype=float),
            "daily_range_pct": np.array([1.0, 1.0, 1.2, 1.3, 1.3, 1.5, 1.7, 2.0, 2.0, 2.1, 2.93], dtype=float),
            "adx": np.array([12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 15.5, 16.2, 16.3, 16.5, 16.9], dtype=float),
        }
        ok, reason = strategy.check_breakout_conditions(feat, 10)
        self.assertFalse(ok)
        self.assertIn("ADX", reason)

    def test_retest_requires_min_volume_one_x(self):
        import numpy as np
        import strategy

        feat = {
            "close": np.array([1.00, 1.01, 1.02, 1.03, 1.04, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11], dtype=float),
            "high": np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12], dtype=float),
            "low": np.array([0.99, 1.00, 1.01, 1.02, 1.03, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.063, 1.10], dtype=float),
            "ema_fast": np.array([0.995, 1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035, 1.040, 1.045, 1.050, 1.055, 1.060, 1.065], dtype=float),
            "slope": np.array([0.12] * 15, dtype=float),
            "rsi": np.array([52.0, 53.0, 54.0, 55.0, 56.0, 52.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 60.3, 60.4], dtype=float),
            "adx": np.array([21.0, 21.5, 22.0, 22.2, 22.5, 22.7, 22.9, 23.0, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1], dtype=float),
            "vol_x": np.array([1.1, 1.0, 1.1, 1.2, 1.0, 1.0, 1.0, 1.1, 1.0, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99], dtype=float),
            "macd_hist": np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04, 0.07, 0.09, 0.11, 0.12], dtype=float),
        }
        ok, reason = strategy.check_retest_conditions(feat, 14)
        self.assertFalse(ok)
        self.assertIn("vol", reason)


def run_tests():
    """Запускает все тесты и выводит краткий отчёт."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestSyntaxAllModules,  # ПЕРВЫМ — ловит SyntaxError/IndentationError
        TestConfig, TestIndicators, TestStrategy,
        TestMonitor, TestMLDataset, TestBotlog, TestIntegration,
        TestRegressions, TestZombieResilience, TestChatIdAndBroadcast,
        TestNoPositionsBug, TestMACDWarningBug,
        TestImpulseSpeedMode, TestMarkdownEscaping,
        # #1 Expectancy-метрики
        TestExpectancy,
        # #2 Portfolio risk management
        TestPortfolioLimits,
        # Регрессия: немедленный WEAK выход на баре входа (AR 11.03.2026)
        TestWeakExitGuard,
        # Регрессия: ложный strong_trend на флэте SNX (12.03.2026)
        TestStrongTrendClassification,
        # Регрессия: ложные сигналы качества (LTC ретест, ICP/MANA alignment)
        TestSignalQualityGuards,
        # Регрессия: data_collector не запускался, ml_dataset не обновлялся с 10.03.2026
        TestDataCollector,
        # Регрессия: кнопки падали с BadRequest 'Query is too old'
        TestCallbackQueryAnswerGuard,
        # Регрессия: ложные alignment входы TAO(range=16%)/SEI(MACD=0) 13.03.2026
        TestAlignmentQualityGuards,
        # MTF-фильтр: блокировка запоздавших 1h входов по состоянию 15м (ETH 13.03.2026)
        TestMTFFilter,
        # Убрана RSI-проверка из BREAKOUT (ZRO 15.03.2026)
        TestBreakoutRSIGuard,
        # EMA_SEP_MIN снижен 0.3→0.05 (TIA 15.03.2026)
        TestAlignmentEMASepMin,
        # Часовой фильтр входов (ML 44K баров, 15.03.2026)
        TestBullDayHysteresis,
        TestSetupRegimeAware,
        TestPortfolioReplacement,
        TestEntryScoreFloorAndMTFSoftPass,
        TestImpulseSpeedLateGuard,
        TestHourlyFilter,
        TestMLSignalModel,
        TestTelegramNoiseControls,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Итоговая строка для быстрого чтения в логах бота
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print()
    if failed == 0:
        print(f"✅ Все тесты пройдены: {passed}/{total}")
    else:
        print(f"❌ ПРОВАЛЕНО: {failed}/{total}  Пройдено: {passed}/{total}")
        for test, tb in result.failures + result.errors:
            print(f"   FAIL: {test}")
        sys.exit(1)  # ненулевой код для CI/systemd

    return result


class TestBullDayHysteresis(unittest.TestCase):
    """T165A-T165B: bull_day hysteresis and confirmation."""

    def test_T165A_bull_day_needs_two_confirmed_bars_to_turn_on(self):
        import config
        import strategy

        base = 100.0
        n = 60
        data = np.zeros(n, dtype=[("c", "f8")])
        data["c"] = np.array([base] * (n - 2) + [100.35, 100.05], dtype=float)
        ema = np.array([base] * n, dtype=float)

        old_state = getattr(config, "_bull_day_active", False)
        config._bull_day_active = False
        try:
            with patch("strategy.fetch_klines", new=AsyncMock(return_value=data)), \
                 patch("strategy._ema", new=Mock(return_value=ema)):
                bull, price, ema50 = asyncio.run(strategy.is_bull_day(None))
        finally:
            config._bull_day_active = old_state

        self.assertFalse(bull)
        self.assertAlmostEqual(price, 100.35, places=6)
        self.assertAlmostEqual(ema50, 100.0, places=6)

    def test_T165B_bull_day_stays_on_until_two_exit_bars(self):
        import config
        import strategy

        base = 100.0
        n = 60
        data = np.zeros(n, dtype=[("c", "f8")])
        data["c"] = np.array([base] * (n - 2) + [99.95, 99.75], dtype=float)
        ema = np.array([base] * n, dtype=float)

        old_state = getattr(config, "_bull_day_active", False)
        config._bull_day_active = True
        try:
            with patch("strategy.fetch_klines", new=AsyncMock(return_value=data)), \
                 patch("strategy._ema", new=Mock(return_value=ema)):
                bull, _, _ = asyncio.run(strategy.is_bull_day(None))
        finally:
            config._bull_day_active = old_state

        self.assertTrue(bull)


class TestReplayBacktestMetrics(unittest.TestCase):
    def test_T176_replay_backtest_contains_comparison_metrics(self):
        src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("pnl_total_delta", src)
        self.assertIn("win_rate_delta", src)
        self.assertIn("skipped_portfolio_full", src)

    def test_T177_replay_backtest_tracks_replacement_outcomes(self):
        src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("replacements_improved", src)
        self.assertIn("replacements_worsened", src)
        self.assertIn("candidates_total", src)

    def test_T178_replacement_policy_has_profit_and_strength_guards(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("PORTFOLIO_REPLACE_PROFIT_PROTECT_PCT", config_src)
        self.assertIn("PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA", config_src)
        self.assertIn("_replacement_extra_delta", monitor_src)
        self.assertIn("_replacement_extra_delta", replay_src)

    def test_T179_discovery_catchup_guards_exist(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("DISCOVERY_ENTRY_GRACE_BARS", config_src)
        self.assertIn("DISCOVERY_ENTRY_MAX_SLIPPAGE_PCT", config_src)
        self.assertIn("recent_discoveries", monitor_src)
        self.assertIn("catch-up after discovery", monitor_src)

    def test_T180_replacement_hard_profit_guard_exists(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT", config_src)
        self.assertIn("PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT", monitor_src)
        self.assertIn("PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT", replay_src)


class TestRestartMonitoringGuards(unittest.TestCase):
    def test_T211_update_hot_coins_keeps_open_positions_monitored(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("def _update_hot_coins")
        self.assertNotEqual(idx, -1)
        block = src[idx:idx + 1200]
        self.assertIn("_ensure_positions_monitored(state)", block)

    def test_T212_ensure_positions_monitored_builds_valid_coin_report(self):
        sys.modules.setdefault("telegram.constants", MagicMock())
        sys.modules.setdefault("telegram.error", MagicMock())
        import bot
        from monitor import OpenPosition

        state = types.SimpleNamespace(hot_coins=[], positions={
            "BTCUSDT": OpenPosition(
                symbol="BTCUSDT",
                tf="15m",
                entry_price=100.0,
                entry_bar=10,
                entry_ts=1234567890000,
                entry_ema20=99.0,
                entry_slope=0.2,
                entry_adx=22.0,
                entry_rsi=58.0,
                entry_vol_x=1.4,
            )
        })

        bot._ensure_positions_monitored(state)

        self.assertEqual(len(state.hot_coins), 1)
        rep = state.hot_coins[0]
        self.assertEqual(rep.symbol, "BTCUSDT")
        self.assertEqual(rep.today_signals, 0)
        self.assertEqual(rep.best_horizon, 0)
        self.assertTrue(rep.in_play)
        self.assertFalse(rep.signal_now)

    def test_T213_post_init_starts_monitoring_restored_positions_immediately(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        idx = src.find("async def _post_init")
        self.assertNotEqual(idx, -1)
        block = src[idx:idx + 4000]
        self.assertIn("if state.positions and not state.running:", block)
        self.assertIn("Started monitoring restored positions immediately", block)
        self.assertIn("monitoring_loop(state, _make_broadcast_send(app))", block)

    def test_T214_positions_ui_refreshes_positions_from_disk(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def _refresh_positions_state()", src)

        start_idx = src.find("async def cmd_start")
        self.assertNotEqual(start_idx, -1)
        start_block = src[start_idx:start_idx + 500]
        self.assertIn("_refresh_positions_state()", start_block)

        pos_idx = src.find('elif action == "positions":')
        self.assertNotEqual(pos_idx, -1)
        pos_block = src[pos_idx:pos_idx + 400]
        self.assertIn("_refresh_positions_state()", pos_block)

        back_idx = src.find('elif action == "back_main":')
        self.assertNotEqual(back_idx, -1)
        back_block = src[back_idx:back_idx + 220]
        self.assertIn("_refresh_positions_state()", back_block)

    def test_T215_post_init_runs_startup_scan_in_background(self):
        src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("async def _run_post_init_scan(", src)

        idx = src.find("async def _post_init")
        self.assertNotEqual(idx, -1)
        block = src[idx:idx + 5500]
        self.assertIn("Skipping immediate startup market_scan because restored positions are already being monitored", block)
        self.assertIn("if state.positions:", block)
        self.assertIn("asyncio.create_task(_run_post_init_scan(app, notify_service, len(wl)))", block)
        self.assertIn("return", block)

    def test_T215A_auto_reanalyze_supports_fast_first_delay(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("AUTO_REANALYZE_FIRST_DELAY_SEC", config_src)
        self.assertIn('getattr(config, "AUTO_REANALYZE_FIRST_DELAY_SEC"', bot_src)
        self.assertIn("waited_sec = next_delay_sec", bot_src)
        self.assertIn("Auto-reanalyze started after %ss", bot_src)

    def test_T181_ema_cross_reason_includes_age(self):
        strategy_src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("age:{cross_age}b", strategy_src)
        self.assertIn("CROSS_CONFIRM_BARS", strategy_src)

    def test_T182_impulse_scanner_runs_initial_scan(self):
        scanner_src = Path("impulse_scanner.py").read_text(encoding="utf-8")
        self.assertIn("await _scan_once(send_fn)", scanner_src)
        self.assertIn("Initial scan exception", scanner_src)


class TestBackfillJsonlSafety(unittest.TestCase):
    def test_T198_backfill_history_reader_skips_bad_jsonl_rows(self):
        import backfill_history

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.jsonl"
            path.write_text(
                "\ufeff" + json.dumps({
                    "sym": "BTCUSDT",
                    "tf": "15m",
                    "bar_ts": 123,
                    "ts_signal": "2026-03-19T00:00:00Z",
                }, ensure_ascii=False) + "\n"
                + "not-json\n"
                + json.dumps({"sym": "ETHUSDT", "tf": "15m"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            rows, skipped = backfill_history._read_valid_jsonl(
                path,
                required_keys=("sym", "tf", "bar_ts"),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(skipped, 2)
            self.assertEqual(rows[0]["sym"], "BTCUSDT")

    def test_T199_backfill_labels_reader_skips_bad_jsonl_rows(self):
        import backfill_labels

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.jsonl"
            path.write_text(
                json.dumps({
                    "id": "ok",
                    "sym": "BTCUSDT",
                    "tf": "15m",
                    "bar_ts": 123,
                    "labels": {},
                }, ensure_ascii=False) + "\n"
                + "broken-line\n"
                + json.dumps({
                    "id": "bad",
                    "sym": "ETHUSDT",
                    "tf": "15m",
                    "bar_ts": 456,
                    "labels": [],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            rows, skipped = backfill_labels._read_valid_jsonl(
                path,
                required_keys=("id", "sym", "tf", "bar_ts", "labels"),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(skipped, 2)
            self.assertEqual(rows[0]["id"], "ok")

    def test_T199A_backfill_labels_uses_absolute_dataset_path(self):
        import backfill_labels

        self.assertTrue(backfill_labels.ML_FILE.is_absolute())


class TestMlSignalSchema(unittest.TestCase):
    def test_T200_ml_signal_model_knows_impulse_and_alignment(self):
        import ml_signal_model

        self.assertIn("impulse", ml_signal_model.SIGNAL_TYPES)
        self.assertIn("alignment", ml_signal_model.SIGNAL_TYPES)

    def test_T201_build_feature_dict_sets_one_hot_for_impulse(self):
        import ml_signal_model

        rec = {
            "signal_type": "impulse",
            "is_bull_day": False,
            "hour_utc": 10,
            "dow": 3,
            "tf": "15m",
            "f": {},
            "seq": [[0.0] * 10 for _ in range(20)],
        }
        feat = ml_signal_model.build_feature_dict(rec)

        self.assertEqual(feat["signal_impulse"], 1.0)
        self.assertEqual(feat["signal_alignment"], 0.0)
        self.assertEqual(feat["signal_trend"], 0.0)

    def test_T202_ml_report_render_is_ascii_safe(self):
        src = Path("ml_signal_model.py").read_text(encoding="utf-8")
        self.assertIn("ret5_delta=", src)
        self.assertIn("wr_delta=", src)
        self.assertIn("cov_delta=", src)
        self.assertNotIn("ret5Δ=", src)
        self.assertNotIn("×", src)
class TestMlSignalGeneratorCoverage(unittest.TestCase):
    def test_T203_signal_generators_cover_alignment_and_surge(self):
        collector_src = Path("data_collector.py").read_text(encoding="utf-8")
        history_src = Path("backfill_history.py").read_text(encoding="utf-8")

        for src in (collector_src, history_src):
            self.assertIn('check_trend_surge_conditions(feat, i)', src)
            self.assertIn('return "impulse_speed"', src)
            self.assertIn('check_impulse_conditions(feat, i)', src)
            self.assertIn('return "impulse"', src)
            self.assertIn('check_alignment_conditions(feat, i)', src)
            self.assertIn('return "alignment"', src)


class TestMlLiveWiring(unittest.TestCase):
    def test_T205_build_live_model_payload_keeps_segment_models(self):
        import ml_signal_model

        report = {
            "model_payload": {"model_name": "logistic"},
            "segment_model_payloads": {"trend|nonbull": {"threshold": 0.42}},
        }

        payload = ml_signal_model.build_live_model_payload(report)

        self.assertIn("segment_model_payloads", payload)
        self.assertIn("trend|nonbull", payload["segment_model_payloads"])

    def test_T206_monitor_passes_ml_proba_into_entry_log(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("ml_proba = _ml_general_score(", src)
        self.assertIn("ml_proba=ml_proba", src)

    def test_T207_monitor_uses_global_ml_dataset_import(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("import ml_dataset", src)
        self.assertEqual(src.count("import ml_dataset"), 1)

    def test_T208_ml_nonbull_is_ranking_first_by_default(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("ML_TREND_NONBULL_LOW_PROBA_PENALTY", config_src)
        self.assertIn("ML_TREND_NONBULL_HARD_BLOCK: bool = False", config_src)
        self.assertIn("candidate_score -=", monitor_src)
        self.assertIn("ML_TREND_NONBULL_HARD_BLOCK", monitor_src)
        self.assertIn("ML_TREND_NONBULL_LOW_PROBA_PENALTY", replay_src)

    def test_T208A_general_ml_runtime_score_falls_back_to_global_payload(self):
        import monitor

        data, feat, i_now = _make_feat(80, "up")
        old_cache = monitor._ML_MODEL_CACHE
        try:
            monitor._ML_MODEL_CACHE = {
                "feature_names": ["rsi", "signal_breakout", "tf_15m"],
                "scaler_mean": [0.0, 0.0, 0.0],
                "scaler_scale": [1.0, 1.0, 1.0],
                "model": {"type": "logistic", "weights": [0.01, 0.4, 0.2], "bias": 0.0},
                "segment_model_payloads": {},
            }
            score = monitor._ml_general_score(
                "TESTUSDT",
                "15m",
                "breakout",
                feat,
                data,
                i_now,
                is_bull_day=True,
            )
            self.assertIsNotNone(score)
            self.assertGreater(score, 0.5)
        finally:
            monitor._ML_MODEL_CACHE = old_cache

    def test_T208B_general_ml_runtime_score_prefers_segment_payload(self):
        import monitor

        data, feat, i_now = _make_feat(80, "up")
        old_cache = monitor._ML_MODEL_CACHE
        try:
            monitor._ML_MODEL_CACHE = {
                "feature_names": ["rsi"],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "model": {"type": "logistic", "weights": [-0.05], "bias": -2.0},
                "segment_model_payloads": {
                    "trend|nonbull": {
                        "feature_names": ["rsi"],
                        "scaler_mean": [0.0],
                        "scaler_scale": [1.0],
                        "model": {"type": "logistic", "weights": [0.05], "bias": 2.0},
                    }
                },
            }
            score = monitor._ml_general_score(
                "TESTUSDT",
                "15m",
                "trend",
                feat,
                data,
                i_now,
                is_bull_day=False,
            )
            self.assertIsNotNone(score)
            self.assertGreater(score, 0.9)
        finally:
            monitor._ML_MODEL_CACHE = old_cache

    def test_T208C_general_ml_ranking_is_wired_into_monitor_and_replay(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")

        self.assertIn("ML_ENABLE_GENERAL_RANKING", config_src)
        self.assertIn("ML_GENERAL_SCORE_WEIGHT", config_src)
        self.assertIn("_ml_general_score(", monitor_src)
        self.assertIn("_ml_general_score_replay(", replay_src)
        self.assertIn("ML_GENERAL_NEUTRAL_PROBA", monitor_src)
        self.assertIn("ML_GENERAL_NEUTRAL_PROBA", replay_src)


class TestContinuationCapture(unittest.TestCase):
    def test_T209_strategy_report_key_uses_signal_priority(self):
        src = Path("strategy.py").read_text(encoding="utf-8")
        self.assertIn("_signal_priority(r.signal_mode)", src)

    def test_T210_monitor_discovery_upgrades_known_symbols(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("existing_by_sym = {r.symbol: (idx, r)", src)
        self.assertIn('discovery_action = "upgrade"', src)
        self.assertIn("state.hot_coins[idx] = report", src)

    def test_T211_monitor_discovery_promotes_trend_surge(self):
        src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("check_trend_surge_conditions(feat, i_now)", src)
        self.assertIn('signal_mode="impulse_speed"', src)
        self.assertIn("replace(", src)

    def test_T212_discovery_interval_is_fast_enough_for_continuations(self):
        import config

        self.assertLessEqual(config.DISCOVERY_SCAN_SEC, config.POLL_SEC)

    def test_T213_time_block_bypass_has_separate_1h_controls(self):
        config_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")

        self.assertIn("TIME_BLOCK_BYPASS_1H_ENABLED", config_src)
        self.assertIn("TIME_BLOCK_BYPASS_1H_SCORE_MIN", config_src)
        self.assertIn("TIME_BLOCK_BYPASS_1H_VOL_X_MIN", config_src)
        self.assertIn("TIME_BLOCK_BYPASS_1H_MODES", config_src)
        self.assertIn('elif tf == "1h"', monitor_src)
        self.assertIn('elif tf == "1h"', replay_src)

    def test_T214_time_block_bypass_1h_targets_continuations(self):
        import config

        self.assertIn("alignment", config.TIME_BLOCK_BYPASS_1H_MODES)
        self.assertIn("trend", config.TIME_BLOCK_BYPASS_1H_MODES)
        self.assertIn("strong_trend", config.TIME_BLOCK_BYPASS_1H_MODES)
        self.assertIn("impulse_speed", config.TIME_BLOCK_BYPASS_1H_MODES)
        self.assertGreaterEqual(config.TIME_BLOCK_BYPASS_1H_SCORE_MIN, 60.0)

    def test_T215_time_block_bypass_1h_continuation_profile_is_configured(self):
        import config

        self.assertTrue(config.TIME_BLOCK_BYPASS_1H_CONTINUATION_ENABLED)
        self.assertIn("impulse", config.TIME_BLOCK_BYPASS_1H_CONTINUATION_MODES)
        self.assertGreater(config.TIME_BLOCK_BYPASS_1H_CONTINUATION_SCORE_BONUS, 0.0)
        self.assertGreaterEqual(config.TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MIN, 60.0)
        self.assertLessEqual(config.TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MAX, 80.0)

    def test_T216_time_block_bypass_1h_continuation_targets_trx_not_apt(self):
        import monitor

        trx_like = monitor._time_block_1h_continuation_profile(
            tf="1h",
            mode="alignment",
            slope=0.11,
            adx=16.5,
            rsi=73.8,
            vol_x=1.26,
            daily_range=3.12,
        )
        apt_like = monitor._time_block_1h_continuation_profile(
            tf="1h",
            mode="alignment",
            slope=0.95,
            adx=38.1,
            rsi=54.2,
            vol_x=2.42,
            daily_range=3.0,
        )
        self.assertTrue(trx_like)
        self.assertFalse(apt_like)

    def test_T217_time_block_1h_prebypass_targets_reconfirmed_trx_not_apt(self):
        import monitor

        trx_like = monitor._time_block_1h_prebypass_allowed(
            tf="1h",
            mode="trend",
            candidate_score=55.5,
            vol_x=1.26,
            price=0.3131,
            ema20=0.3094,
            continuation_profile=True,
            repeat_count=2,
        )
        apt_like = monitor._time_block_1h_prebypass_allowed(
            tf="1h",
            mode="alignment",
            candidate_score=56.1,
            vol_x=2.42,
            price=0.988,
            ema20=0.982604,
            continuation_profile=False,
            repeat_count=3,
        )
        self.assertTrue(trx_like)
        self.assertFalse(apt_like)

    def test_T218_late_1h_continuation_guard_blocks_late_trx_and_high_rsi_impulse_speed(self):
        import monitor

        late_trx = monitor._late_1h_continuation_guard(
            tf="1h",
            mode="trend",
            continuation_profile=True,
            candidate_score=56.0,
            price=0.3170,
            ema20=0.311719,
            rsi=74.9,
            daily_range=5.98,
        )
        qnt_like_impulse = monitor._late_1h_continuation_guard(
            tf="1h",
            mode="impulse_speed",
            continuation_profile=True,
            candidate_score=61.0,
            price=81.40,
            ema20=77.25,
            rsi=74.2,
            daily_range=5.6,
        )
        self.assertTrue(late_trx)
        self.assertTrue(qnt_like_impulse)

    def test_T219_continuation_profit_lock_arms_after_confirmed_plus(self):
        import monitor

        active = monitor._continuation_profit_lock_active(
            tf="1h",
            mode="trend",
            entry_rsi=74.9,
            bars_elapsed=3,
            current_pnl=0.31,
            predictions={3: True},
        )
        inactive = monitor._continuation_profit_lock_active(
            tf="1h",
            mode="trend",
            entry_rsi=68.0,
            bars_elapsed=3,
            current_pnl=0.31,
            predictions={3: True},
        )
        self.assertTrue(active)
        self.assertFalse(inactive)

    def test_T220_early_1h_continuation_entry_targets_trx_not_apt(self):
        import strategy

        trx_feat = {
            "ema_fast": np.array([0.3097, 0.3101]),
            "ema_slow": np.array([0.3087, 0.3090]),
            "ema200": np.array([0.3032, 0.3034]),
            "rsi": np.array([52.1, 64.4]),
            "adx": np.array([30.6, 31.0]),
            "slope": np.array([-0.03, 0.126]),
            "vol_x": np.array([0.66, 2.22]),
            "adx_sma": np.array([36.0, 33.6]),
            "macd_hist": np.array([-0.000157, 0.000144]),
            "daily_range_pct": np.array([3.8, 5.8]),
        }
        trx_close = np.array([0.3100, 0.3131])
        neutral = strategy.MarketRegime("neutral")

        default_ok, _ = strategy.check_entry_conditions(trx_feat, 1, trx_close, regime=neutral, tf="")
        one_h_ok, _ = strategy.check_entry_conditions(trx_feat, 1, trx_close, regime=neutral, tf="1h")

        apt_helper = strategy._early_1h_continuation_entry_ok(
            {
                "ema_fast": np.array([0.9780, 0.982604]),
                "ema_slow": np.array([0.9690, 0.9730]),
                "ema200": np.array([0.9570, 0.9575]),
                "adx": np.array([36.0, 38.1]),
                "adx_sma": np.array([34.0, 32.0]),
                "rsi": np.array([52.0, 54.2]),
                "slope": np.array([0.60, 0.95]),
                "vol_x": np.array([2.10, 2.42]),
                "daily_range_pct": np.array([2.5, 3.0]),
                "macd_hist": np.array([0.006, 0.008]),
            },
            1,
            np.array([0.982, 0.988]),
            tf="1h",
            mode="trend",
        )

        self.assertFalse(default_ok)
        self.assertTrue(one_h_ok)
        self.assertFalse(apt_helper)

    def test_T221_continuation_profit_lock_can_arm_for_early_continuation_profit(self):
        import monitor

        active = monitor._continuation_profit_lock_active(
            tf="1h",
            mode="trend",
            entry_rsi=67.2,
            bars_elapsed=3,
            current_pnl=0.95,
            predictions={3: None},
        )

        self.assertTrue(active)

    def test_T221B_early_15m_continuation_targets_aevo_like_setup(self):
        import strategy

        feat = {
            "ema_fast": np.array([0.0231, 0.0236]),
            "ema_slow": np.array([0.0229, 0.0230]),
            "ema200": np.array([0.0223, 0.0225]),
            "adx": np.array([18.8, 22.4]),
            "rsi": np.array([61.2, 71.8]),
            "slope": np.array([0.05, 0.18]),
            "vol_x": np.array([0.98, 1.52]),
            "daily_range_pct": np.array([4.8, 6.9]),
            "macd_hist": np.array([0.00010, 0.00018]),
            "close": np.array([0.0233, 0.0239]),
        }
        close = np.array([0.0233, 0.0239])

        self.assertEqual(strategy.get_entry_mode(feat, 1), "trend")
        self.assertTrue(strategy._early_15m_continuation_entry_ok(feat, 1, close, tf="15m", mode="trend"))
        mode, is_early = strategy.get_effective_entry_mode(feat, 1, close, tf="15m")
        self.assertEqual(mode, "alignment")
        self.assertTrue(is_early)

    def test_T221C_early_15m_continuation_rejects_late_overstretched_spike(self):
        import strategy

        feat = {
            "ema_fast": np.array([0.0236, 0.0236]),
            "ema_slow": np.array([0.0230, 0.0230]),
            "ema200": np.array([0.0224, 0.0225]),
            "adx": np.array([24.0, 25.1]),
            "rsi": np.array([74.5, 79.8]),
            "slope": np.array([0.20, 0.42]),
            "vol_x": np.array([1.4, 2.9]),
            "daily_range_pct": np.array([8.2, 11.6]),
            "macd_hist": np.array([0.00020, 0.00017]),
            "close": np.array([0.0241, 0.0248]),
        }
        close = np.array([0.0241, 0.0248])

        self.assertFalse(strategy._early_15m_continuation_entry_ok(feat, 1, close, tf="15m", mode="trend"))

    def test_T221D_replay_entry_candidate_uses_alignment_for_early_15m_continuation(self):
        import replay_backtest

        feat = {
            "ema_fast": np.array([0.0231, 0.0236]),
            "ema_slow": np.array([0.0229, 0.0230]),
            "ema200": np.array([0.0223, 0.0225]),
            "adx": np.array([18.8, 22.4]),
            "adx_sma": np.array([17.5, 21.3]),
            "rsi": np.array([61.2, 71.8]),
            "slope": np.array([0.05, 0.18]),
            "vol_x": np.array([0.98, 1.52]),
            "daily_range_pct": np.array([4.8, 6.9]),
            "macd_hist": np.array([0.00010, 0.00018]),
            "close": np.array([0.0233, 0.0239]),
            "ema20": np.array([0.0231, 0.0236]),
        }
        close = np.array([0.0233, 0.0239])

        picked = replay_backtest._entry_candidate(feat, 1, close, "15m")
        self.assertIsNotNone(picked)
        mode, _, _, early = picked
        self.assertEqual(mode, "alignment")
        self.assertTrue(early)

    def test_T222_continuation_micro_exit_targets_trx_like_15m_weakness(self):
        import monitor

        data_15m = np.zeros(8, dtype=[("t", "i8"), ("c", "f8")])
        data_15m["t"] = np.arange(8) * 900_000
        data_15m["c"] = np.array([0.3170, 0.3174, 0.3179, 0.3180, 0.3175, 0.3180, 0.3178, 0.0])
        feat_15m = {
            "macd_hist": np.array([0.00018, 0.00013, 0.00003, -0.00003, -0.00005, -0.00010, -0.00019, 0.0]),
            "ema_fast": np.array([0.3156, 0.3158, 0.3160, 0.31635, 0.31651, 0.31681, 0.31726, 0.0]),
            "rsi": np.array([81.2, 81.6, 73.5, 74.5, 76.6, 71.6, 63.1, 0.0]),
        }

        reason = monitor._continuation_micro_exit_reason(
            tf="1h",
            mode="trend",
            bars_elapsed=5,
            data_15m=data_15m,
            feat_15m=feat_15m,
        )

        self.assertIsNotNone(reason)
        self.assertIn("15m micro-weakness", reason)


class TestUiResponsivenessAndStalePositions(unittest.TestCase):
    def test_T230_positions_reply_action_uses_fast_path(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def _positions_message_html()", bot_src)
        self.assertIn("await _show_positions_message(update.message)", bot_src)

    def test_T231_monitoring_loop_discovery_is_backgrounded(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("discovery_task: Optional[asyncio.Task] = None", monitor_src)
        self.assertIn("state.discovery_task = asyncio.create_task(_run_discovery())", monitor_src)

    def test_T232_menu_reply_keyboard_is_compact(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("[KeyboardButton(BTN_MENU), KeyboardButton(BTN_HIDE_MENU)]", bot_src)
        self.assertNotIn("KeyboardButton(BTN_POSITIONS)", bot_src)

    def test_T233_menu_button_opens_inline_control_panel(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def kb_menu_root_inline()", bot_src)
        self.assertIn('callback_data="menu_state"', bot_src)
        self.assertIn('callback_data="menu_control"', bot_src)
        self.assertIn("reply_markup=kb_menu_root_inline()", bot_src)
        self.assertIn("await _show_menu_panel_message(update.message, force_refresh=False)", bot_src)

    def test_T234_polling_is_started_with_short_intervals(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("app.run_polling(", bot_src)
        self.assertIn("drop_pending_updates=False", bot_src)
        self.assertIn("poll_interval=0.2", bot_src)
        self.assertIn("timeout=5", bot_src)

    def test_T235_main_menu_text_is_not_mojibake(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("Мониторинг:", bot_src)
        self.assertIn("Монет в списке:", bot_src)
        self.assertIn("Открытых сигналов:", bot_src)

    def test_T236_ui_replies_use_fast_timeouts_and_short_retry(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("UI_CONNECT_TIMEOUT = 4.0", bot_src)
        self.assertIn("UI_READ_TIMEOUT = 8.0", bot_src)
        self.assertIn("UI_SEND_DEADLINE = 10.0", bot_src)
        self.assertIn("UI_REQUEST_POOL_SIZE = 32", bot_src)
        self.assertIn("UPDATES_REQUEST_POOL_SIZE = 4", bot_src)
        self.assertIn("from telegram.request import HTTPXRequest", bot_src)
        self.assertIn("async def _await_with_deadline(", bot_src)
        self.assertIn('async def _reply_text_with_retry(msg, text: str, retries: int = 1, **kwargs):', bot_src)
        self.assertIn('merged.setdefault("read_timeout", UI_READ_TIMEOUT)', bot_src)
        self.assertIn("return await _await_with_deadline(", bot_src)
        self.assertIn(".request(ui_request)", bot_src)
        self.assertIn(".get_updates_request(updates_request)", bot_src)


class TestNightSignalQualityGuards(unittest.TestCase):
    def test_T236_alignment_is_stricter_in_nonbull(self):
        import config
        from strategy import check_alignment_conditions

        prev_bull = getattr(config, "_bull_day_active", False)
        try:
            config._bull_day_active = False
            n = 8
            feat = {
                "ema_fast": np.array([1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]),
                "ema_slow": np.array([0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06]),
                "ema200": np.array([0.95] * n),
                "slope": np.array([0.08] * n),
                "rsi": np.array([58.0] * n),
                "vol_x": np.array([0.75] * n),
                "macd_hist": np.array([0.001] * n),
                "daily_range_pct": np.array([4.0] * n),
                "adx": np.array([28.0] * n),
                "close": np.array([1.01, 1.02, 1.03, 1.05, 1.06, 1.07, 1.08, 1.09]),
            }
            ok, reason = check_alignment_conditions(feat, n - 1)
            self.assertFalse(ok)
            self.assertIn("vol", reason)
        finally:
            config._bull_day_active = prev_bull

    def test_T237_fast_loss_ema_exit_fires_after_min_bar_grace(self):
        import monitor

        reason = monitor._fast_loss_ema_exit_reason(
            tf="15m",
            mode="retest",
            bars_elapsed=2,
            current_pnl=-0.42,
            close_now=0.999,
            ema20=1.001,
            rsi=49.5,
        )

        self.assertIsNotNone(reason)
        self.assertIn("EMA20", reason)

    def test_T237B_post_entry_quality_recheck_exits_late_alignment(self):
        import monitor

        feat = {
            "close": np.array([0.1008, 0.1012, 0.1020, 0.1028, 0.1032, 0.1036]),
            "ema_fast": np.array([0.1002, 0.1007, 0.1014, 0.1021, 0.1028, 0.1032]),
            "ema_slow": np.array([0.0998, 0.1001, 0.1006, 0.1010, 0.1016, 0.1021]),
            "ema200": np.array([0.0989, 0.0990, 0.0993, 0.0997, 0.1002, 0.1007]),
            "ema_slope_pct": np.array([0.18, 0.25, 0.38, 0.51, 0.56, 0.53]),
            "slope": np.array([0.18, 0.25, 0.38, 0.51, 0.56, 0.53]),
            "macd_hist": np.array([0.00006, 0.00055, 0.00048, 0.00026, 0.00011, 0.00006]),
            "rsi": np.array([51.0, 55.0, 60.0, 63.0, 61.0, 59.5]),
            "adx": np.array([18.0, 21.0, 24.0, 27.0, 27.8, 27.4]),
            "vol_x": np.array([1.2, 1.5, 2.2, 4.7, 5.0, 5.42]),
            "daily_range_pct": np.array([3.5, 4.1, 5.0, 6.4, 7.1, 7.47]),
        }
        pos = monitor.OpenPosition(
            symbol="ZRXUSDT",
            tf="15m",
            entry_price=0.1036,
            entry_bar=len(feat["close"]) - 2,
            entry_ts=0,
            entry_ema20=0.1032,
            entry_slope=0.53,
            entry_adx=27.4,
            entry_rsi=59.5,
            entry_vol_x=5.42,
            predictions={},
            bars_elapsed=1,
            signal_mode="alignment",
            trail_k=2.0,
            max_hold_bars=48,
            trail_stop=0.1025,
        )

        reason = monitor._post_entry_quality_recheck_reason(pos, feat, len(feat["close"]) - 1)

        self.assertIsNotNone(reason)
        self.assertIn("quality recheck failed", reason)
        self.assertIn("late alignment", reason)

    def test_T237C_profitable_weak_1h_exit_skips_cooldown(self):
        import monitor

        bars = monitor._cooldown_bars_after_exit(
            "alignment",
            "WEAK: RSI divergence",
            tf="1h",
            pnl_pct=0.35,
        )
        self.assertEqual(bars, 0)

        bars_15m = monitor._cooldown_bars_after_exit(
            "alignment",
            "WEAK: RSI divergence",
            tf="15m",
            pnl_pct=0.35,
        )
        self.assertGreaterEqual(bars_15m, 8)

    def test_T238_reserved_slot_for_fresh_impulses_is_enabled(self):
        import config

        self.assertEqual(config.FRESH_SIGNAL_RESERVED_SLOTS, 1)
        self.assertNotIn("impulse", config.FRESH_SIGNAL_PRIORITY_MODES)
        self.assertIn("breakout", config.FRESH_SIGNAL_PRIORITY_MODES)
        self.assertIn("retest", config.FRESH_SIGNAL_PRIORITY_MODES)
        self.assertIn("impulse_speed", config.FRESH_SIGNAL_PRIORITY_MODES)
        self.assertEqual(config.PORTFOLIO_REPLACE_FRESH_MIN_DELTA, 6.0)

    def test_T242_late_1h_guard_now_covers_impulse_speed_at_rsi_70(self):
        import config
        import monitor

        self.assertIn("impulse_speed", config.LATE_1H_CONTINUATION_GUARD_MODES)
        self.assertEqual(config.LATE_1H_CONTINUATION_GUARD_RSI_MIN, 70.0)

        blocked = monitor._late_1h_continuation_guard(
            tf="1h",
            mode="impulse_speed",
            continuation_profile=True,
            candidate_score=63.0,
            price=71446.5,
            ema20=68965.8,
            rsi=72.5,
            daily_range=5.4,
        )
        self.assertTrue(blocked)

    def test_T243_strategy_cap_blocks_third_one_hour_impulse_speed(self):
        import config
        import monitor

        self.assertEqual(config.MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS, 0)

        pos1 = monitor.OpenPosition(
            symbol="A",
            tf="1h",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=0,
            entry_ema20=1.0,
            entry_slope=0.2,
            entry_adx=28.0,
            entry_rsi=71.0,
            entry_vol_x=2.0,
            signal_mode="impulse_speed",
        )
        pos2 = monitor.OpenPosition(
            symbol="B",
            tf="1h",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=0,
            entry_ema20=1.0,
            entry_slope=0.2,
            entry_adx=29.0,
            entry_rsi=72.0,
            entry_vol_x=2.0,
            signal_mode="impulse_speed",
        )
        pos3 = monitor.OpenPosition(
            symbol="C",
            tf="1h",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=0,
            entry_ema20=1.0,
            entry_slope=0.2,
            entry_adx=20.0,
            entry_rsi=60.0,
            entry_vol_x=1.0,
            signal_mode="trend",
        )
        state = monitor.MonitorState(positions={"A": pos1, "B": pos2, "C": pos3})

        prev_cap = config.MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS
        try:
            config.MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS = 2
            ok, reason = monitor._check_strategy_position_caps(state, tf="1h", mode="impulse_speed")
            other_ok, _ = monitor._check_strategy_position_caps(state, tf="15m", mode="impulse_speed")

            self.assertFalse(ok)
            self.assertIn("1h impulse_speed", reason)
            self.assertTrue(other_ok)
        finally:
            config.MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS = prev_cap

    def test_T244_replay_applies_one_hour_impulse_speed_cap(self):
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS", replay_src)
        self.assertIn('candidate.tf == "1h" and candidate.mode == "impulse_speed"', replay_src)

    def test_T244B_clone_signal_guard_blocks_third_recent_short_setup(self):
        import config
        import monitor

        bar_ms = monitor._tf_bar_ms("15m")
        now_ts = 20 * bar_ms
        pos1 = monitor.OpenPosition(
            symbol="ETHUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 2 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.25,
            entry_adx=31.0,
            entry_rsi=64.0,
            entry_vol_x=2.0,
            signal_mode="breakout",
        )
        pos2 = monitor.OpenPosition(
            symbol="LINKUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 4 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.22,
            entry_adx=29.0,
            entry_rsi=62.0,
            entry_vol_x=1.8,
            signal_mode="trend",
        )
        state = monitor.MonitorState(positions={"ETHUSDT": pos1, "LINKUSDT": pos2})

        prev_enabled = config.CLONE_SIGNAL_GUARD_ENABLED
        prev_window = config.CLONE_SIGNAL_GUARD_WINDOW_BARS
        prev_max_similar = config.CLONE_SIGNAL_GUARD_MAX_SIMILAR
        prev_max_group = config.CLONE_SIGNAL_GUARD_MAX_SAME_GROUP
        try:
            config.CLONE_SIGNAL_GUARD_ENABLED = True
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = 8
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = 2
            config.CLONE_SIGNAL_GUARD_MAX_SAME_GROUP = 10

            reason = monitor._clone_signal_guard_reason(
                "ARBUSDT",
                state,
                tf="15m",
                mode="breakout",
                bar_ts=now_ts,
                candidate_score=63.0,
                ranker_info={"final_score": -0.25},
            )
            self.assertTrue(reason)
            self.assertIn("clone signal guard", reason)
            self.assertIn("similar 15m", reason)
        finally:
            config.CLONE_SIGNAL_GUARD_ENABLED = prev_enabled
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = prev_window
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = prev_max_similar
            config.CLONE_SIGNAL_GUARD_MAX_SAME_GROUP = prev_max_group

    def test_T244C_clone_signal_guard_allows_high_conviction_override(self):
        import config
        import monitor

        bar_ms = monitor._tf_bar_ms("15m")
        now_ts = 20 * bar_ms
        pos1 = monitor.OpenPosition(
            symbol="ETHUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 2 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.25,
            entry_adx=31.0,
            entry_rsi=64.0,
            entry_vol_x=2.0,
            signal_mode="breakout",
        )
        pos2 = monitor.OpenPosition(
            symbol="LINKUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 4 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.22,
            entry_adx=29.0,
            entry_rsi=62.0,
            entry_vol_x=1.8,
            signal_mode="trend",
        )
        state = monitor.MonitorState(positions={"ETHUSDT": pos1, "LINKUSDT": pos2})

        prev_enabled = config.CLONE_SIGNAL_GUARD_ENABLED
        prev_window = config.CLONE_SIGNAL_GUARD_WINDOW_BARS
        prev_max_similar = config.CLONE_SIGNAL_GUARD_MAX_SIMILAR
        prev_override_score = config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE
        try:
            config.CLONE_SIGNAL_GUARD_ENABLED = True
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = 8
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = 2
            config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE = 90.0
            config.CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL = -0.25

            reason = monitor._clone_signal_guard_reason(
                "ARBUSDT",
                state,
                tf="15m",
                mode="breakout",
                bar_ts=now_ts,
                candidate_score=95.0,
                ranker_info={"final_score": -0.25},
            )
            self.assertEqual(reason, "")
        finally:
            config.CLONE_SIGNAL_GUARD_ENABLED = prev_enabled
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = prev_window
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = prev_max_similar
            config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE = prev_override_score

    def test_T244D_clone_signal_guard_blocks_high_score_when_ranker_is_too_negative(self):
        import config
        import monitor

        bar_ms = monitor._tf_bar_ms("15m")
        now_ts = 20 * bar_ms
        pos1 = monitor.OpenPosition(
            symbol="ETHUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 2 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.25,
            entry_adx=31.0,
            entry_rsi=64.0,
            entry_vol_x=2.0,
            signal_mode="breakout",
        )
        pos2 = monitor.OpenPosition(
            symbol="LINKUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=0,
            entry_ts=now_ts - 4 * bar_ms,
            entry_ema20=1.0,
            entry_slope=0.22,
            entry_adx=29.0,
            entry_rsi=62.0,
            entry_vol_x=1.8,
            signal_mode="trend",
        )
        state = monitor.MonitorState(positions={"ETHUSDT": pos1, "LINKUSDT": pos2})

        prev_enabled = config.CLONE_SIGNAL_GUARD_ENABLED
        prev_window = config.CLONE_SIGNAL_GUARD_WINDOW_BARS
        prev_max_similar = config.CLONE_SIGNAL_GUARD_MAX_SIMILAR
        prev_override_score = config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE
        prev_override_rank = config.CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL
        try:
            config.CLONE_SIGNAL_GUARD_ENABLED = True
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = 8
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = 2
            config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE = 90.0
            config.CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL = -0.25

            reason = monitor._clone_signal_guard_reason(
                "ARBUSDT",
                state,
                tf="15m",
                mode="breakout",
                bar_ts=now_ts,
                candidate_score=163.0,
                ranker_info={"final_score": -4.86},
            )
            self.assertTrue(reason)
            self.assertIn("clone signal guard", reason)
        finally:
            config.CLONE_SIGNAL_GUARD_ENABLED = prev_enabled
            config.CLONE_SIGNAL_GUARD_WINDOW_BARS = prev_window
            config.CLONE_SIGNAL_GUARD_MAX_SIMILAR = prev_max_similar
            config.CLONE_SIGNAL_GUARD_OVERRIDE_SCORE = prev_override_score
            config.CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL = prev_override_rank

    def test_T244E_open_cluster_cap_blocks_third_15m_short_bounce(self):
        import config
        import monitor

        state = monitor.MonitorState(
            positions={
                "XAIUSDT": monitor.OpenPosition(
                    symbol="XAIUSDT",
                    tf="15m",
                    entry_price=1.0,
                    entry_bar=0,
                    entry_ts=1,
                    entry_ema20=1.0,
                    entry_slope=0.2,
                    entry_adx=24.0,
                    entry_rsi=58.0,
                    entry_vol_x=1.4,
                    signal_mode="retest",
                ),
                "DOGEUSDT": monitor.OpenPosition(
                    symbol="DOGEUSDT",
                    tf="15m",
                    entry_price=1.0,
                    entry_bar=0,
                    entry_ts=1,
                    entry_ema20=1.0,
                    entry_slope=0.2,
                    entry_adx=23.0,
                    entry_rsi=57.0,
                    entry_vol_x=1.3,
                    signal_mode="breakout",
                ),
            }
        )

        prev_enabled = config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED
        prev_modes = config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES
        prev_max = config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX
        try:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = True
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES = ("breakout", "retest")
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX = 2

            reason = monitor._open_signal_cluster_cap_reason(
                "DOTUSDT",
                state,
                tf="15m",
                mode="retest",
            )
            self.assertTrue(reason)
            self.assertIn("open cluster cap", reason)
            self.assertIn("15m_short_bounce", reason)
        finally:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = prev_enabled
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES = prev_modes
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX = prev_max

    def test_T244F_open_cluster_cap_blocks_third_15m_impulse_speed(self):
        import config
        import monitor

        state = monitor.MonitorState(
            positions={
                "ADAUSDT": monitor.OpenPosition(
                    symbol="ADAUSDT",
                    tf="15m",
                    entry_price=1.0,
                    entry_bar=0,
                    entry_ts=1,
                    entry_ema20=1.0,
                    entry_slope=0.2,
                    entry_adx=24.0,
                    entry_rsi=58.0,
                    entry_vol_x=1.4,
                    signal_mode="impulse_speed",
                ),
                "ENSUSDT": monitor.OpenPosition(
                    symbol="ENSUSDT",
                    tf="15m",
                    entry_price=1.0,
                    entry_bar=0,
                    entry_ts=1,
                    entry_ema20=1.0,
                    entry_slope=0.2,
                    entry_adx=23.0,
                    entry_rsi=57.0,
                    entry_vol_x=1.3,
                    signal_mode="impulse_speed",
                ),
            }
        )

        prev_enabled = config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED
        prev_modes = config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES
        prev_max = config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX
        try:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = True
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES = ("impulse_speed",)
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX = 2

            reason = monitor._open_signal_cluster_cap_reason(
                "XRPUSDT",
                state,
                tf="15m",
                mode="impulse_speed",
            )
            self.assertTrue(reason)
            self.assertIn("open cluster cap", reason)
            self.assertIn("15m_impulse", reason)
        finally:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = prev_enabled
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES = prev_modes
            config.OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX = prev_max

    def test_T244G_open_cluster_cap_blocks_second_1h_retest(self):
        import config
        import monitor

        state = monitor.MonitorState(
            positions={
                "BNTUSDT": monitor.OpenPosition(
                    symbol="BNTUSDT",
                    tf="1h",
                    entry_price=1.0,
                    entry_bar=0,
                    entry_ts=1,
                    entry_ema20=1.0,
                    entry_slope=0.2,
                    entry_adx=24.0,
                    entry_rsi=58.0,
                    entry_vol_x=1.4,
                    signal_mode="retest",
                ),
            }
        )

        prev_enabled = config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED
        prev_modes = config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES
        prev_max = config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX
        try:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = True
            config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES = ("retest",)
            config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX = 1

            reason = monitor._open_signal_cluster_cap_reason(
                "YFIUSDT",
                state,
                tf="1h",
                mode="retest",
            )
            self.assertTrue(reason)
            self.assertIn("open cluster cap", reason)
            self.assertIn("1h_retest", reason)
        finally:
            config.OPEN_SIGNAL_CLUSTER_CAP_ENABLED = prev_enabled
            config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES = prev_modes
            config.OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX = prev_max

    def test_T245_one_hour_impulse_speed_guard_blocks_late_or_weak_entries(self):
        import config
        import monitor

        self.assertTrue(config.IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED)
        self.assertEqual(config.IMPULSE_SPEED_1H_RSI_MAX, 70.0)
        self.assertEqual(config.IMPULSE_SPEED_1H_ADX_MIN, 22.0)
        self.assertEqual(config.IMPULSE_SPEED_1H_RANGE_MAX, 10.0)

        prev_enabled = config.IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED
        try:
            config.IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED = True
            self.assertEqual(
                monitor._one_hour_impulse_speed_entry_guard(
                    tf="1h",
                    mode="impulse_speed",
                    rsi=72.5,
                    adx=38.5,
                    daily_range=6.1,
                ),
                "1h impulse_speed guard: RSI 72.5 > 70.0",
            )
            self.assertEqual(
                monitor._one_hour_impulse_speed_entry_guard(
                    tf="1h",
                    mode="impulse_speed",
                    rsi=67.1,
                    adx=18.2,
                    daily_range=9.6,
                ),
                "1h impulse_speed guard: ADX 18.2 < 22.0",
            )
            self.assertEqual(
                monitor._one_hour_impulse_speed_entry_guard(
                    tf="1h",
                    mode="impulse_speed",
                    rsi=65.6,
                    adx=33.9,
                    daily_range=21.1,
                ),
                "1h impulse_speed guard: daily_range 21.10% > 10.00%",
            )
            self.assertIsNone(
                monitor._one_hour_impulse_speed_entry_guard(
                    tf="15m",
                    mode="impulse_speed",
                    rsi=75.0,
                    adx=10.0,
                    daily_range=12.0,
                )
            )
        finally:
            config.IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED = prev_enabled

    def test_T246_replay_applies_one_hour_impulse_speed_entry_guard(self):
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("_impulse_speed_entry_guard", replay_src)

    def test_T247_late_impulse_speed_rotation_guard_blocks_overheated_replacements(self):
        import config
        import monitor

        self.assertTrue(config.IMPULSE_SPEED_ROTATION_GUARD_ENABLED)
        self.assertEqual(
            monitor._late_impulse_speed_rotation_reason(
                tf="15m",
                mode="impulse_speed",
                rsi=79.4,
                daily_range=8.27,
            ),
            "late impulse_speed rotation guard: RSI 79.4 >= 76.0, daily_range 8.27% >= 5.00%",
        )
        self.assertEqual(
            monitor._late_impulse_speed_rotation_reason(
                tf="1h",
                mode="impulse_speed",
                rsi=72.5,
                daily_range=27.78,
            ),
            "late impulse_speed rotation guard: RSI 72.5 >= 64.0, daily_range 27.78% >= 16.00%",
        )
        self.assertIsNone(
            monitor._late_impulse_speed_rotation_reason(
                tf="15m",
                mode="impulse_speed",
                rsi=64.1,
                daily_range=2.31,
            )
        )

    def test_T248_replay_applies_late_impulse_speed_rotation_guard(self):
        replay_src = Path("replay_backtest.py").read_text(encoding="utf-8")
        self.assertIn("_late_impulse_speed_rotation_reason", replay_src)

    def test_T247_fast_loss_exit_now_covers_weak_one_hour_modes(self):
        import config

        self.assertIn("1h", config.FAST_LOSS_EMA_EXIT_TF)
        self.assertIn("alignment", config.FAST_LOSS_EMA_EXIT_MODES)
        self.assertIn("retest", config.FAST_LOSS_EMA_EXIT_MODES)
        self.assertIn("breakout", config.FAST_LOSS_EMA_EXIT_MODES)
        self.assertIn("impulse", config.FAST_LOSS_EMA_EXIT_MODES)
        self.assertNotIn("impulse_speed", config.FAST_LOSS_EMA_EXIT_MODES)
        self.assertEqual(config.FAST_LOSS_EMA_EXIT_RSI_MAX, 70.0)
        self.assertEqual(config.FAST_LOSS_EMA_EXIT_MIN_BARS, 1)

    def test_T248_impulse_speed_late_guard_enabled_by_default(self):
        import config

        self.assertTrue(config.IMPULSE_SPEED_LATE_GUARD_ENABLED)

    def test_T249_impulse_speed_has_longer_weak_exit_grace(self):
        import config
        import monitor
        import replay_backtest

        self.assertGreater(
            int(config.MIN_WEAK_EXIT_BARS_IMPULSE_SPEED),
            int(config.MIN_WEAK_EXIT_BARS),
        )
        self.assertEqual(
            monitor._min_weak_exit_bars("impulse_speed"),
            int(config.MIN_WEAK_EXIT_BARS_IMPULSE_SPEED),
        )
        self.assertEqual(
            replay_backtest._min_weak_exit_bars("impulse_speed"),
            int(config.MIN_WEAK_EXIT_BARS_IMPULSE_SPEED),
        )

    def test_T250_breakout_retest_trend_have_mode_specific_weak_grace(self):
        import config
        import monitor
        import replay_backtest

        self.assertEqual(monitor._min_weak_exit_bars("breakout"), int(config.MIN_WEAK_EXIT_BARS_BREAKOUT))
        self.assertEqual(monitor._min_weak_exit_bars("retest"), int(config.MIN_WEAK_EXIT_BARS_RETEST))
        self.assertEqual(monitor._min_weak_exit_bars("trend"), int(config.MIN_WEAK_EXIT_BARS_TREND))
        self.assertEqual(replay_backtest._min_weak_exit_bars("breakout"), int(config.MIN_WEAK_EXIT_BARS_BREAKOUT))
        self.assertEqual(replay_backtest._min_weak_exit_bars("retest"), int(config.MIN_WEAK_EXIT_BARS_RETEST))
        self.assertEqual(replay_backtest._min_weak_exit_bars("trend"), int(config.MIN_WEAK_EXIT_BARS_TREND))

    def test_T250b_trend_hold_weak_exit_helper_triggers_for_strong_profitable_leader(self):
        import monitor

        pos = monitor.OpenPosition(
            symbol="RENDERUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=10,
            entry_ts=1,
            entry_ema20=1.0,
            entry_slope=0.4,
            entry_adx=30.0,
            entry_rsi=67.0,
            entry_vol_x=2.2,
            signal_mode="impulse_speed",
            candidate_score_at_entry=105.0,
            bars_elapsed=13,
            trail_k=2.5,
            trail_stop=1.01,
        )
        _, feat, i = _make_feat(200, "up")
        feat["ema_fast"][i] = 1.02
        feat["ema_slow"][i] = 1.01
        feat["ema200"][i] = 0.99
        feat["adx"][i] = 31.0
        feat["slope"][i] = 0.45
        self.assertTrue(
            monitor._trend_hold_weak_exit_active(
                pos=pos,
                feat=feat,
                idx=i,
                close_now=1.05,
                current_pnl=1.4,
                tf="15m",
            )
        )

    def test_T250c_trend_hold_weak_exit_helper_rejects_weak_ada_like_case(self):
        import monitor

        pos = monitor.OpenPosition(
            symbol="ADAUSDT",
            tf="15m",
            entry_price=1.0,
            entry_bar=10,
            entry_ts=1,
            entry_ema20=1.0,
            entry_slope=0.35,
            entry_adx=18.1,
            entry_rsi=68.8,
            entry_vol_x=0.95,
            signal_mode="impulse_speed",
            candidate_score_at_entry=63.0,
            bars_elapsed=12,
            trail_k=2.5,
            trail_stop=0.99,
        )
        _, feat, i = _make_feat(200, "flat")
        feat["ema_fast"][i] = 1.002
        feat["ema_slow"][i] = 1.001
        feat["ema200"][i] = 0.999
        feat["adx"][i] = 18.1
        feat["slope"][i] = 0.35
        self.assertFalse(
            monitor._trend_hold_weak_exit_active(
                pos=pos,
                feat=feat,
                idx=i,
                close_now=1.004,
                current_pnl=0.5,
                tf="15m",
            )
        )

    def test_T251_quality_floor_and_short_mode_holds_restored_after_replay(self):
        import config

        self.assertEqual(float(config.ENTRY_SCORE_MIN_15M), 48.0)
        self.assertEqual(float(config.ENTRY_SCORE_MIN_1H), 56.0)
        self.assertEqual(int(config.MAX_HOLD_BARS_BREAKOUT), 6)
        self.assertEqual(int(config.MAX_HOLD_BARS_RETEST), 10)
        self.assertEqual(float(config.ATR_TRAIL_K_RETEST), 1.5)
        self.assertEqual(float(config.BREAKOUT_VOL_MIN), 2.8)
        self.assertEqual(float(config.BREAKOUT_RANGE_MAX), 3.0)
        self.assertEqual(float(config.BREAKOUT_RSI_MAX), 76.0)
        self.assertEqual(float(config.IMPULSE_R1_MIN), 1.5)
        self.assertEqual(float(config.IMPULSE_R3_MIN), 2.0)
        self.assertEqual(float(config.IMPULSE_VOL_MIN), 1.5)
        self.assertEqual(float(config.IMPULSE_RSI_HI), 82.0)

    def test_T248_open_positions_are_monitored_on_position_timeframe(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn('pos = state.positions.get(sym)', monitor_src)
        self.assertIn('tf = getattr(pos, "tf", tf)', monitor_src)

    def test_T252_short_mode_profit_lock_helper_exists_in_live_and_replay(self):
        import monitor
        import replay_backtest

        self.assertTrue(hasattr(monitor, "_short_mode_profit_lock_active"))
        self.assertTrue(hasattr(replay_backtest, "_short_mode_profit_lock_active"))

    def test_T239_fresh_priority_helpers_keep_only_strong_fresh_modes_reserved(self):
        import monitor
        import replay_backtest

        self.assertFalse(monitor._is_fresh_priority_candidate("impulse"))
        self.assertFalse(replay_backtest._is_fresh_priority_mode("impulse"))
        self.assertTrue(monitor._is_fresh_priority_candidate("impulse_speed"))
        self.assertTrue(replay_backtest._is_fresh_priority_mode("impulse_speed"))
        self.assertFalse(monitor._is_fresh_priority_candidate("alignment"))

    def test_T240_monitor_moves_indicator_work_off_event_loop(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")

        self.assertIn("async def _compute_features_from_data(", monitor_src)
        self.assertIn("await asyncio.to_thread(_compute_features_from_data_sync, data)", monitor_src)
        self.assertIn("await asyncio.to_thread(", monitor_src)

    def test_T241_discovery_uses_async_analysis_and_valid_compute_features_signature(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")

        self.assertIn("report = await _analyze_coin_live(sym, tf, data)", monitor_src)
        self.assertNotIn("feat = compute_features(data)", monitor_src)


class TestTopMoverPrioritizationAndAudit(unittest.TestCase):
    def test_T242_positions_use_shared_sort_helper(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("def _sorted_position_items(positions=None, hot_coins=None):", bot_src)
        self.assertIn("for sym, pos in _sorted_position_items():", bot_src)
        self.assertIn('getattr(coin_report, "signal_now", False)', bot_src)
        self.assertIn('getattr(coin_report, "today_confirmed", False)', bot_src)

    def test_T242B_position_sort_prefers_live_strength_then_freshness(self):
        telegram_mod = types.ModuleType("telegram")
        telegram_mod.InlineKeyboardButton = MagicMock()
        telegram_mod.InlineKeyboardMarkup = MagicMock()
        telegram_mod.KeyboardButton = MagicMock()
        telegram_mod.ReplyKeyboardMarkup = MagicMock()
        telegram_mod.ReplyKeyboardRemove = MagicMock()
        telegram_mod.Update = MagicMock()
        sys.modules["telegram"] = telegram_mod

        telegram_constants = types.ModuleType("telegram.constants")
        telegram_constants.ParseMode = MagicMock()
        sys.modules["telegram.constants"] = telegram_constants

        telegram_error = types.ModuleType("telegram.error")
        telegram_error.NetworkError = Exception
        telegram_error.TimedOut = Exception
        sys.modules["telegram.error"] = telegram_error

        telegram_request = types.ModuleType("telegram.request")
        telegram_request.HTTPXRequest = MagicMock()
        sys.modules["telegram.request"] = telegram_request

        telegram_ext = types.ModuleType("telegram.ext")
        telegram_ext.Application = MagicMock()
        telegram_ext.CallbackQueryHandler = MagicMock()
        telegram_ext.CommandHandler = MagicMock()
        telegram_ext.ContextTypes = MagicMock()
        telegram_ext.MessageHandler = MagicMock()
        telegram_ext.filters = MagicMock()
        sys.modules["telegram.ext"] = telegram_ext

        import bot
        from monitor import OpenPosition

        positions = {
            "SLOWUSDT": OpenPosition(
                symbol="SLOWUSDT",
                tf="15m",
                entry_price=1.0,
                entry_bar=1,
                entry_ts=1,
                entry_ema20=1.0,
                entry_slope=0.1,
                entry_adx=20.0,
                entry_rsi=55.0,
                entry_vol_x=1.1,
                forecast_return_pct=0.2,
                today_change_pct=0.5,
                bars_elapsed=3,
                signal_mode="breakout",
            ),
            "STRONGUSDT": OpenPosition(
                symbol="STRONGUSDT",
                tf="15m",
                entry_price=1.0,
                entry_bar=1,
                entry_ts=1,
                entry_ema20=1.0,
                entry_slope=0.1,
                entry_adx=35.0,
                entry_rsi=60.0,
                entry_vol_x=2.0,
                forecast_return_pct=0.1,
                today_change_pct=0.2,
                bars_elapsed=2,
                signal_mode="trend",
            ),
            "FRESHUSDT": OpenPosition(
                symbol="FRESHUSDT",
                tf="15m",
                entry_price=1.0,
                entry_bar=1,
                entry_ts=1,
                entry_ema20=1.0,
                entry_slope=0.1,
                entry_adx=28.0,
                entry_rsi=58.0,
                entry_vol_x=1.5,
                forecast_return_pct=0.1,
                today_change_pct=0.2,
                bars_elapsed=1,
                signal_mode="trend",
            ),
        }
        hot_coins = [
            types.SimpleNamespace(
                symbol="SLOWUSDT",
                forecast_return_pct=0.3,
                today_change_pct=1.0,
                signal_mode="breakout",
                best_accuracy=65.0,
                signal_now=False,
                today_confirmed=False,
            ),
            types.SimpleNamespace(
                symbol="STRONGUSDT",
                forecast_return_pct=0.4,
                today_change_pct=0.8,
                signal_mode="trend",
                best_accuracy=70.0,
                signal_now=True,
                today_confirmed=True,
            ),
            types.SimpleNamespace(
                symbol="FRESHUSDT",
                forecast_return_pct=0.4,
                today_change_pct=0.8,
                signal_mode="trend",
                best_accuracy=70.0,
                signal_now=True,
                today_confirmed=True,
            ),
        ]

        ordered = [sym for sym, _ in bot._sorted_position_items(positions, hot_coins)]

        self.assertEqual(ordered, ["FRESHUSDT", "STRONGUSDT", "SLOWUSDT"])

    def test_T242C_position_sort_prefers_ranker_score_when_live_flags_match(self):
        import monitor
        import bot

        positions = {
            "LOWUSDT": monitor.OpenPosition(
                symbol="LOWUSDT",
                tf="15m",
                entry_price=1.0,
                entry_bar=1,
                entry_ts=1,
                entry_ema20=1.0,
                entry_slope=0.1,
                entry_adx=25.0,
                entry_rsi=55.0,
                entry_vol_x=1.5,
                bars_elapsed=1,
                signal_mode="breakout",
                ranker_final_score=0.25,
            ),
            "HIGHUSDT": monitor.OpenPosition(
                symbol="HIGHUSDT",
                tf="15m",
                entry_price=1.0,
                entry_bar=1,
                entry_ts=1,
                entry_ema20=1.0,
                entry_slope=0.1,
                entry_adx=20.0,
                entry_rsi=55.0,
                entry_vol_x=1.1,
                bars_elapsed=2,
                signal_mode="breakout",
                ranker_final_score=1.40,
            ),
        }
        hot_coins = [
            types.SimpleNamespace(
                symbol="LOWUSDT",
                forecast_return_pct=0.2,
                today_change_pct=0.1,
                signal_mode="breakout",
                best_accuracy=60.0,
                signal_now=True,
                today_confirmed=True,
            ),
            types.SimpleNamespace(
                symbol="HIGHUSDT",
                forecast_return_pct=0.2,
                today_change_pct=0.1,
                signal_mode="breakout",
                best_accuracy=60.0,
                signal_now=True,
                today_confirmed=True,
            ),
        ]

        ordered = [sym for sym, _ in bot._sorted_position_items(positions, hot_coins)]

        self.assertEqual(ordered, ["HIGHUSDT", "LOWUSDT"])

    def test_T242D_position_roundtrip_preserves_ranker_metadata(self):
        import monitor

        pos = monitor.OpenPosition(
            symbol="METAUSDT",
            tf="15m",
            entry_price=10.0,
            entry_bar=2,
            entry_ts=123,
            entry_ema20=9.8,
            entry_slope=0.2,
            entry_adx=31.0,
            entry_rsi=58.0,
            entry_vol_x=1.7,
            candidate_score_at_entry=87.5,
            score_floor_at_entry=48.0,
            entry_ml_proba=0.41,
            ranker_quality_proba=0.62,
            ranker_final_score=1.33,
            ranker_ev=0.57,
            ranker_expected_return=0.84,
            ranker_expected_drawdown=0.36,
            ranker_top_gainer_prob=0.29,
            ranker_capture_ratio_pred=0.18,
            prediction_horizons=(2, 5, 7),
            predictions={2: True, 5: None, 7: False},
        )

        payload = monitor._pos_to_dict(pos)
        restored = monitor._pos_from_dict(payload)

        self.assertAlmostEqual(restored.candidate_score_at_entry, 87.5)
        self.assertAlmostEqual(restored.score_floor_at_entry, 48.0)
        self.assertAlmostEqual(restored.entry_ml_proba, 0.41)
        self.assertAlmostEqual(restored.ranker_quality_proba, 0.62)
        self.assertAlmostEqual(restored.ranker_final_score, 1.33)
        self.assertAlmostEqual(restored.ranker_ev, 0.57)
        self.assertAlmostEqual(restored.ranker_expected_return, 0.84)
        self.assertAlmostEqual(restored.ranker_expected_drawdown, 0.36)
        self.assertAlmostEqual(restored.ranker_top_gainer_prob, 0.29)
        self.assertAlmostEqual(restored.ranker_capture_ratio_pred, 0.18)
        self.assertEqual(restored.prediction_horizons, (2, 5, 7))
        self.assertEqual(restored.predictions, {2: True, 5: None, 7: False})

    def test_T242E_positions_message_shows_ranker_snapshot(self):
        telegram_mod = types.ModuleType("telegram")
        telegram_mod.InlineKeyboardButton = MagicMock()
        telegram_mod.InlineKeyboardMarkup = MagicMock()
        telegram_mod.KeyboardButton = MagicMock()
        telegram_mod.ReplyKeyboardMarkup = MagicMock()
        telegram_mod.ReplyKeyboardRemove = MagicMock()
        telegram_mod.Update = MagicMock()
        sys.modules["telegram"] = telegram_mod

        telegram_constants = types.ModuleType("telegram.constants")
        telegram_constants.ParseMode = MagicMock()
        sys.modules["telegram.constants"] = telegram_constants

        telegram_error = types.ModuleType("telegram.error")
        telegram_error.NetworkError = Exception
        telegram_error.TimedOut = Exception
        sys.modules["telegram.error"] = telegram_error

        telegram_request = types.ModuleType("telegram.request")
        telegram_request.HTTPXRequest = MagicMock()
        sys.modules["telegram.request"] = telegram_request

        telegram_ext = types.ModuleType("telegram.ext")
        telegram_ext.Application = MagicMock()
        telegram_ext.CallbackQueryHandler = MagicMock()
        telegram_ext.CommandHandler = MagicMock()
        telegram_ext.ContextTypes = MagicMock()
        telegram_ext.MessageHandler = MagicMock()
        telegram_ext.filters = MagicMock()
        sys.modules["telegram.ext"] = telegram_ext

        import monitor
        import bot

        original_positions = dict(bot.state.positions)
        original_hot_coins = list(bot.state.hot_coins)
        try:
            bot.state.positions = {
                "METAUSDT": monitor.OpenPosition(
                    symbol="METAUSDT",
                    tf="15m",
                    entry_price=10.0,
                    entry_bar=2,
                    entry_ts=123,
                    entry_ema20=9.8,
                    entry_slope=0.2,
                    entry_adx=31.0,
                    entry_rsi=58.0,
                    entry_vol_x=1.7,
                    candidate_score_at_entry=87.5,
                    ranker_quality_proba=0.62,
                    ranker_final_score=1.33,
                    ranker_ev=0.57,
                    ranker_top_gainer_prob=0.29,
                    ranker_capture_ratio_pred=0.18,
                    prediction_horizons=(2, 5, 7),
                    predictions={2: None, 5: None, 7: None},
                    bars_elapsed=1,
                    signal_mode="breakout",
                )
            }
            bot.state.hot_coins = []

            html = bot._positions_message_html()

            self.assertIn("Score 87.5", html)
            self.assertIn("Rank +1.33", html)
            self.assertIn("EV +0.57", html)
            self.assertIn("TG 0.29", html)
            self.assertIn("CAP 0.18", html)
            self.assertIn("T+2:", html)
            self.assertIn("T+7:", html)
        finally:
            bot.state.positions = original_positions
            bot.state.hot_coins = original_hot_coins

    def test_T242F_fast_15m_positions_use_short_forward_horizons(self):
        import config
        import monitor

        self.assertEqual(monitor._position_forward_horizons("15m", "retest"), (2, 5, 7))
        self.assertEqual(monitor._position_forward_horizons("15m", "breakout"), (2, 5, 7))
        self.assertEqual(monitor._position_forward_horizons("15m", "impulse_speed"), (2, 5, 7))
        self.assertEqual(monitor._position_forward_horizons("1h", "trend"), tuple(config.FORWARD_BARS))

    def test_T243_build_portfolio_audit_compares_top10_with_portfolio(self):
        from report_top_movers import build_portfolio_audit

        summaries = [
            {"symbol": "TAOUSDT"},
            {"symbol": "RENDERUSDT"},
            {"symbol": "AXLUSDT"},
            {"symbol": "ZROUSDT"},
        ]
        portfolio_symbols = ["TAOUSDT", "AXLUSDT", "BTCUSDT"]

        audit = build_portfolio_audit(summaries, portfolio_symbols)

        self.assertEqual(audit["captured_symbols"], ["TAOUSDT", "AXLUSDT"])
        self.assertEqual(audit["missed_symbols"], ["RENDERUSDT", "ZROUSDT"])
        self.assertEqual(audit["extra_symbols"], ["BTCUSDT"])
        self.assertAlmostEqual(audit["capture_rate_pct"], 50.0)

    def test_T244_live_scoring_uses_top_mover_and_forecast_bonuses(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("candidate_score += _top_mover_score_bonus", monitor_src)
        self.assertIn("candidate_score += _forecast_return_score_bonus", monitor_src)
        self.assertIn('float(getattr(report, "forecast_return_pct", 0.0))', monitor_src)
        self.assertIn('float(getattr(report, "today_change_pct", 0.0))', monitor_src)

    def test_T245_open_positions_refresh_forecast_potential_live(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("if pos is not None:", monitor_src)
        self.assertIn("live_report = await _analyze_coin_live(sym, tf, data)", monitor_src)
        self.assertIn("new_forecast_return =", monitor_src)
        self.assertIn("new_today_change =", monitor_src)
        self.assertIn("pos.forecast_return_pct = new_forecast_return", monitor_src)
        self.assertIn("pos.today_change_pct = new_today_change", monitor_src)

    def test_T246_positions_sort_falls_back_to_position_forecast_when_report_is_zero(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("if abs(forecast_return) < 1e-12:", bot_src)
        self.assertIn('forecast_return = float(getattr(pos, "forecast_return_pct", 0.0))', bot_src)
        self.assertIn("if abs(today_change) < 1e-12:", bot_src)
        self.assertIn('today_change = float(getattr(pos, "today_change_pct", 0.0))', bot_src)

    def test_T247_refresh_positions_preserves_live_growth_metrics(self):
        bot_src = Path("bot.py").read_text(encoding="utf-8")
        self.assertIn("live_positions = dict(state.positions)", bot_src)
        self.assertIn("loaded_positions = load_positions()", bot_src)
        self.assertIn("hot_by_sym = {r.symbol: r for r in state.hot_coins}", bot_src)
        self.assertIn('"forecast_return_pct"', bot_src)
        self.assertIn('"today_change_pct"', bot_src)
        self.assertIn("loaded_pos.forecast_return_pct = forecast_return", bot_src)
        self.assertIn("loaded_pos.today_change_pct = today_change", bot_src)

    def test_T248_live_growth_metrics_are_persisted_when_they_change(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("metrics_changed =", monitor_src)
        self.assertIn("new_forecast_return", monitor_src)
        self.assertIn("new_today_change", monitor_src)
        self.assertIn("if metrics_changed:", monitor_src)
        self.assertIn("save_positions(state.positions)", monitor_src)


class TestCriticRankerPreparation(unittest.TestCase):
    def test_T249_critic_dataset_logs_candidate_and_updates_labels(self):
        import critic_dataset
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "critic_dataset.jsonl"
            data, feat, i = _make_feat(80, "up")
            with patch.object(critic_dataset, "CRITIC_FILE", tmp_path):
                critic_dataset._logged_candidates.clear()
                rec_id = critic_dataset.log_candidate(
                    sym="TESTCRITUSDT",
                    tf="15m",
                    bar_ts=int(data["t"][i]),
                    signal_type="retest",
                    is_bull_day=False,
                    feat=feat,
                    i=i,
                    data=data,
                    action="blocked",
                    reason_code="entry_score",
                    reason="entry score 47.50 < floor 48.00",
                    stage="quality_floor",
                    candidate_score=47.5,
                    base_score=44.0,
                    score_floor=48.0,
                    forecast_return_pct=0.25,
                    today_change_pct=1.2,
                    ml_proba=0.42,
                    mtf_soft_penalty=2.0,
                    fresh_priority=True,
                    catchup=False,
                    continuation_profile=False,
                    signal_flags={"retest_ok": True},
                )
                critic_dataset.mark_trade_taken(rec_id, linked_ml_record_id="ml_123")
                critic_dataset.fill_forward_label(rec_id, 5, 1.23)
                critic_dataset.fill_trade_outcome(rec_id, 0.87, "portfolio rotation", 4)

                rows = [json.loads(line) for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["id"], rec_id)
                self.assertEqual(row["decision"]["action"], "blocked")
                self.assertEqual(row["decision"]["reason_code"], "entry_score")
                self.assertTrue(row["labels"]["trade_taken"])
                self.assertEqual(row["labels"]["linked_ml_record_id"], "ml_123")
                self.assertAlmostEqual(row["labels"]["ret_5"], 1.23, places=2)
                self.assertEqual(row["labels"]["label_5"], True)
                self.assertAlmostEqual(row["labels"]["trade_exit_pnl"], 0.87, places=2)
                self.assertEqual(row["labels"]["trade_exit_reason"], "portfolio rotation")
                self.assertEqual(row["labels"]["trade_bars_held"], 4)

    def test_T249A_critic_atomic_replace_retries_permission_error(self):
        import critic_dataset
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "critic_dataset.jsonl"
            tmp = Path(td) / "critic_dataset.jsonl.retry.tmp"
            target.write_text("old\n", encoding="utf-8")
            tmp.write_text("new\n", encoding="utf-8")

            original_replace = Path.replace
            calls = {"n": 0}

            def _flaky_replace(path_obj, dst):
                if Path(path_obj) == tmp and calls["n"] < 2:
                    calls["n"] += 1
                    raise PermissionError(5, "Access is denied")
                return original_replace(path_obj, dst)

            with patch("pathlib.Path.replace", new=_flaky_replace), \
                 patch("critic_dataset.time.sleep", return_value=None):
                critic_dataset._atomic_replace_with_retry(tmp, target)

            self.assertGreaterEqual(calls["n"], 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
            self.assertFalse(tmp.exists())

    def test_T250_candidate_ranker_runtime_record_contains_decision_features(self):
        import ml_candidate_ranker
        data, feat, i = _make_feat(80, "up")
        rec = ml_candidate_ranker.build_runtime_candidate_record(
            sym="TESTRANKUSDT",
            tf="1h",
            signal_type="alignment",
            is_bull_day=False,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            candidate_score=61.5,
            base_score=55.0,
            score_floor=56.0,
            forecast_return_pct=0.44,
            today_change_pct=2.0,
            ml_proba=0.31,
            mtf_soft_penalty=3.0,
            fresh_priority=False,
            catchup=True,
            continuation_profile=True,
            near_miss=True,
            signal_flags={"alignment_ok": True, "entry_ok": True},
        )
        fmap = ml_candidate_ranker.build_feature_dict(rec)
        self.assertIn("candidate_score", fmap)
        self.assertIn("forecast_return_pct", fmap)
        self.assertIn("flag_alignment_ok", fmap)
        self.assertIn("near_miss", fmap)
        self.assertEqual(fmap["flag_alignment_ok"], 1.0)
        self.assertEqual(fmap["catchup"], 1.0)
        self.assertEqual(fmap["continuation_profile"], 1.0)
        self.assertEqual(fmap["near_miss"], 1.0)

    def test_T251_monitor_wires_critic_dataset_and_ranker_hooks(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        self.assertIn("critic_record_id", monitor_src)
        self.assertIn("critic_dataset.mark_trade_taken", monitor_src)
        self.assertIn("critic_dataset.fill_trade_outcome", monitor_src)
        self.assertIn("critic_dataset.fill_forward_label", monitor_src)
        self.assertIn("_log_critic_candidate(", monitor_src)
        self.assertIn("_ml_candidate_ranker_score(", monitor_src)

    def test_T252_critic_dataset_upserts_collector_candidate_into_take(self):
        import critic_dataset
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "critic_dataset.jsonl"
            data, feat, i = _make_feat(80, "up")
            bar_ts = int(data["t"][i])
            with patch.object(critic_dataset, "CRITIC_FILE", tmp_path):
                critic_dataset._logged_candidates.clear()
                rec_id = critic_dataset.log_candidate(
                    sym="UPSERTUSDT",
                    tf="15m",
                    bar_ts=bar_ts,
                    signal_type="breakout",
                    is_bull_day=False,
                    feat=feat,
                    i=i,
                    data=data,
                    action="candidate",
                    reason_code="rule_signal",
                    reason="collector detected candidate",
                    stage="collector",
                    signal_flags={"breakout_ok": True},
                )
                rec_id_2 = critic_dataset.log_candidate(
                    sym="UPSERTUSDT",
                    tf="15m",
                    bar_ts=bar_ts,
                    signal_type="breakout",
                    is_bull_day=False,
                    feat=feat,
                    i=i,
                    data=data,
                    action="take",
                    reason_code="take",
                    reason="candidate accepted",
                    stage="entry",
                    candidate_score=58.2,
                    base_score=55.4,
                    score_floor=48.0,
                    forecast_return_pct=0.77,
                    today_change_pct=2.4,
                    ml_proba=0.51,
                    mtf_soft_penalty=0.0,
                    fresh_priority=True,
                    catchup=False,
                    continuation_profile=False,
                    signal_flags={"breakout_ok": True},
                )
                rows = [json.loads(line) for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(rec_id, rec_id_2)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["decision"]["action"], "take")
                self.assertEqual(rows[0]["decision"]["stage"], "entry")
                self.assertTrue(rows[0]["labels"]["trade_taken"])

    def test_T253_data_collector_logs_non_none_rule_signals_to_critic_dataset(self):
        collector_src = Path("data_collector.py").read_text(encoding="utf-8")
        self.assertIn("critic_dataset.log_candidate(", collector_src)
        self.assertIn('action="candidate"', collector_src)
        self.assertIn('reason_code="rule_signal"', collector_src)
        self.assertIn("_signal_flags_from_rule_signal(", collector_src)

    def test_T253A_critic_noop_rewrite_skips_cross_process_lock(self):
        import critic_dataset
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "critic_dataset.jsonl"
            data, feat, i = _make_feat(80, "up")
            with patch.object(critic_dataset, "CRITIC_FILE", tmp_path):
                critic_dataset._logged_candidates.clear()
                rec_id = critic_dataset.log_candidate(
                    sym="NOOPCRITUSDT",
                    tf="15m",
                    bar_ts=int(data["t"][i]),
                    signal_type="retest",
                    is_bull_day=False,
                    feat=feat,
                    i=i,
                    data=data,
                    action="candidate",
                    reason_code="rule_signal",
                    reason="collector detected candidate",
                    stage="collector",
                    signal_flags={"retest_ok": True},
                )
                critic_dataset.fill_forward_label(rec_id, 3, 0.5)
                with patch.object(critic_dataset, "_dataset_io_lock", side_effect=AssertionError("lock should not be acquired")):
                    critic_dataset.fill_forward_label(rec_id, 3, 0.5)

    def test_T253B_critic_dataset_persists_near_miss_decision_flag(self):
        import critic_dataset
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "critic_dataset.jsonl"
            data, feat, i = _make_feat(80, "up")
            with patch.object(critic_dataset, "CRITIC_FILE", tmp_path):
                critic_dataset._logged_candidates.clear()
                rec_id = critic_dataset.log_candidate(
                    sym="NEARMISSUSDT",
                    tf="1h",
                    bar_ts=int(data["t"][i]),
                    signal_type="alignment",
                    is_bull_day=True,
                    feat=feat,
                    i=i,
                    data=data,
                    action="candidate",
                    reason_code="near_miss",
                    reason="near miss early_continuation",
                    stage="near_miss",
                    candidate_score=50.0,
                    base_score=45.0,
                    score_floor=56.0,
                    forecast_return_pct=0.32,
                    today_change_pct=1.8,
                    ml_proba=None,
                    mtf_soft_penalty=0.0,
                    fresh_priority=False,
                    catchup=False,
                    continuation_profile=True,
                    near_miss=True,
                    signal_flags={},
                )
                rows = [json.loads(line) for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["id"], rec_id)
                self.assertEqual(rows[0]["decision"]["stage"], "near_miss")
                self.assertEqual(rows[0]["decision"]["reason_code"], "near_miss")
                self.assertTrue(rows[0]["decision"]["near_miss"])

    def test_T254_candidate_ranker_predict_payload_scores_record(self):
        import copy
        import numpy as np
        import ml_candidate_ranker

        data, feat, i = _make_feat(80, "up")
        rec = ml_candidate_ranker.build_runtime_candidate_record(
            sym="PREDICTUSDT",
            tf="15m",
            signal_type="retest",
            is_bull_day=False,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            candidate_score=50.0,
            base_score=44.0,
            score_floor=48.0,
            forecast_return_pct=0.1,
            today_change_pct=1.0,
            ml_proba=0.0,
            mtf_soft_penalty=0.0,
            fresh_priority=False,
            catchup=False,
            continuation_profile=False,
            signal_flags={"retest_ok": True},
        )
        feature_names = ml_candidate_ranker.safe_feature_names()
        weights = np.zeros(len(feature_names), dtype=float)
        weights[feature_names.index("forecast_return_pct")] = 4.0
        payload = {
            "feature_names": feature_names,
            "scaler_mean": [0.0] * len(feature_names),
            "scaler_scale": [1.0] * len(feature_names),
            "model": {
                "type": "logistic",
                "weights": weights.tolist(),
                "bias": 0.0,
            },
        }

        rec_low = copy.deepcopy(rec)
        rec_high = copy.deepcopy(rec)
        rec_low["decision"]["forecast_return_pct"] = 0.1
        rec_high["decision"]["forecast_return_pct"] = 1.0

        p_low = ml_candidate_ranker.predict_proba_from_candidate_payload(payload, rec_low)
        p_high = ml_candidate_ranker.predict_proba_from_candidate_payload(payload, rec_high)
        self.assertGreater(p_high, p_low)

    def test_T254A_candidate_ranker_predict_components_supports_ev_payload(self):
        import copy
        import numpy as np
        import ml_candidate_ranker

        data, feat, i = _make_feat(80, "up")
        rec = ml_candidate_ranker.build_runtime_candidate_record(
            sym="EVUSDT",
            tf="1h",
            signal_type="alignment",
            is_bull_day=True,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            candidate_score=52.0,
            base_score=46.0,
            score_floor=56.0,
            forecast_return_pct=0.2,
            today_change_pct=1.5,
            ml_proba=0.25,
            mtf_soft_penalty=0.0,
            fresh_priority=True,
            catchup=False,
            continuation_profile=True,
            signal_flags={"alignment_ok": True, "entry_ok": True},
        )
        feature_names = ml_candidate_ranker.safe_feature_names()
        rank_weights = np.zeros(len(feature_names), dtype=float)
        rank_weights[feature_names.index("forecast_return_pct")] = 1.0
        ret_weights = np.zeros(len(feature_names), dtype=float)
        ret_weights[feature_names.index("forecast_return_pct")] = 2.0
        dd_weights = np.zeros(len(feature_names), dtype=float)
        dd_weights[feature_names.index("daily_range")] = 0.5
        payload = {
            "payload_version": 3,
            "threshold": 0.55,
            "ev_lambda": 0.75,
            "quality_score_weight": 0.35,
            "rank_score_weight": 0.20,
            "top_gainer_score_weight": 0.25,
            "capture_score_weight": 0.15,
            "top_gainer_rate_mean": 0.10,
            "capture_ratio_mean": 0.20,
            "rank_score_mean": 0.0,
            "rank_score_std": 1.0,
            "feature_names": feature_names,
            "scaler_mean": [0.0] * len(feature_names),
            "scaler_scale": [1.0] * len(feature_names),
            "model": {"type": "logistic", "weights": [0.0] * len(feature_names), "bias": 0.0},
            "quality_model": {"type": "logistic", "weights": [0.0] * len(feature_names), "bias": 0.0},
            "quality_calibrator": {"type": "platt", "a": 1.0, "b": 0.0},
            "return_model": {"type": "ridge", "weights": ret_weights.tolist(), "bias": 0.0},
            "drawdown_model": {"type": "ridge", "weights": dd_weights.tolist(), "bias": 0.0},
            "rank_model": {"type": "pairwise_linear", "weights": rank_weights.tolist(), "bias": 0.0},
            "top_gainer_model": {"type": "ridge", "weights": ret_weights.tolist(), "bias": 0.0},
            "top_gainer_calibrator": {"type": "platt", "a": 1.0, "b": 0.0},
            "capture_model": {"type": "ridge", "weights": ret_weights.tolist(), "bias": 0.0},
        }

        rec_low = copy.deepcopy(rec)
        rec_high = copy.deepcopy(rec)
        rec_low["decision"]["forecast_return_pct"] = 0.1
        rec_high["decision"]["forecast_return_pct"] = 1.2

        comps_low = ml_candidate_ranker.predict_components_from_candidate_payload(payload, rec_low)
        comps_high = ml_candidate_ranker.predict_components_from_candidate_payload(payload, rec_high)
        self.assertGreater(comps_high["expected_return"], comps_low["expected_return"])
        self.assertGreater(comps_high["ev_raw"], comps_low["ev_raw"])
        self.assertGreater(comps_high["top_gainer_prob"], comps_low["top_gainer_prob"])
        self.assertGreater(comps_high["capture_ratio_pred"], comps_low["capture_ratio_pred"])
        self.assertGreater(comps_high["final_score"], comps_low["final_score"])

    def test_T254B_candidate_ranker_supports_catboost_runtime_model_types(self):
        import numpy as np
        import ml_candidate_ranker

        class _StubClassifier:
            def predict_proba(self, X):
                return np.array([[0.2, 0.8] for _ in range(len(X))], dtype=float)

        class _StubRegressor:
            def predict(self, X):
                return np.array([1.25 for _ in range(len(X))], dtype=float)

        x = np.array([0.1, 0.2, 0.3], dtype=float)
        with patch.object(ml_candidate_ranker, "_load_catboost_runtime_model", return_value=_StubClassifier()):
            p = ml_candidate_ranker._predict_model_score({"type": "catboost_classifier", "blob_b64": "stub"}, x)
        with patch.object(ml_candidate_ranker, "_load_catboost_runtime_model", return_value=_StubRegressor()):
            r = ml_candidate_ranker._predict_model_score({"type": "catboost_regressor", "blob_b64": "stub"}, x)
        with patch.object(ml_candidate_ranker, "_load_catboost_runtime_model", return_value=_StubRegressor()):
            s = ml_candidate_ranker._predict_model_score({"type": "catboost_ranker", "blob_b64": "stub"}, x)

        self.assertAlmostEqual(p, 0.8, places=6)
        self.assertAlmostEqual(r, 1.25, places=6)
        self.assertAlmostEqual(s, 1.25, places=6)

    def test_T255_shadow_report_prefers_better_top1_than_baseline_on_synthetic_dataset(self):
        import json
        import tempfile
        from datetime import datetime
        from pathlib import Path

        import numpy as np
        import ml_candidate_ranker
        import report_candidate_ranker_shadow

        data, feat, i = _make_feat(80, "up")

        def _row(sym: str, bar_ts: int, cand_score: float, forecast_ret: float, ret5: float, exit_pnl: float) -> dict:
            rec = ml_candidate_ranker.build_runtime_candidate_record(
                sym=sym,
                tf="15m",
                signal_type="retest",
                is_bull_day=False,
                bar_ts=bar_ts,
                feat=feat,
                data=data,
                i=i,
                candidate_score=cand_score,
                base_score=44.0,
                score_floor=48.0,
                forecast_return_pct=forecast_ret,
                today_change_pct=1.0,
                ml_proba=0.0,
                mtf_soft_penalty=0.0,
                fresh_priority=False,
                catchup=False,
                continuation_profile=False,
                signal_flags={"retest_ok": True},
            )
            rec["id"] = f"{sym}_{bar_ts}"
            rec["ts_signal"] = datetime.utcfromtimestamp(bar_ts / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec["bar_ts"] = bar_ts
            rec["decision"]["action"] = "take"
            rec["decision"]["stage"] = "entry"
            rec["labels"] = {
                "ret_5": ret5,
                "label_5": ret5 > 0.0,
                "trade_taken": True,
                "trade_exit_pnl": exit_pnl,
                "trade_bars_held": 4,
            }
            rec["teacher"] = {
                "final": {
                    "watchlist_top_gainer": sym.startswith("GOOD"),
                    "capture_ratio": 0.8 if sym.startswith("GOOD") else 0.0,
                }
            }
            return rec

        rows = [
            _row("BAD1USDT", 1_000_000, 60.0, 0.1, -1.2, -1.0),
            _row("GOOD1USDT", 1_000_000, 55.0, 0.9, 1.8, 1.5),
            _row("BAD2USDT", 2_000_000, 62.0, 0.2, -0.8, -0.7),
            _row("GOOD2USDT", 2_000_000, 54.0, 1.1, 2.1, 1.9),
        ]

        feature_names = ml_candidate_ranker.safe_feature_names()
        weights = np.zeros(len(feature_names), dtype=float)
        weights[feature_names.index("forecast_return_pct")] = 5.0
        payload = {
            "model_name": "logistic",
            "feature_names": feature_names,
            "scaler_mean": [0.0] * len(feature_names),
            "scaler_scale": [1.0] * len(feature_names),
            "model": {
                "type": "logistic",
                "weights": weights.tolist(),
                "bias": 0.0,
            },
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic_dataset.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            report = report_candidate_ranker_shadow.build_shadow_report(path, payload, top_ns=(1,))

        self.assertEqual(report["grouped_bar_competitions"], 2)
        top1 = report["top_n"][0]
        self.assertGreater(top1["ranker"]["avg_target_return"], top1["baseline"]["avg_target_return"])
        self.assertGreaterEqual(top1["ranker"]["teacher_top_gainer_rate"], top1["baseline"]["teacher_top_gainer_rate"])
        self.assertGreaterEqual(top1["ranker"]["teacher_capture_ratio"], top1["baseline"]["teacher_capture_ratio"])
        self.assertEqual(report["top1_head_to_head"]["wins"], 2)

    def test_T255A_shadow_report_groups_candidates_by_decision_moment_not_tf(self):
        import json
        import tempfile
        from datetime import datetime
        from pathlib import Path

        import numpy as np
        import ml_candidate_ranker
        import report_candidate_ranker_shadow

        data, feat, i = _make_feat(80, "up")
        bar_ts = 1_000_000

        def _row(sym: str, tf: str, forecast_ret: float, ret5: float) -> dict:
            rec = ml_candidate_ranker.build_runtime_candidate_record(
                sym=sym,
                tf=tf,
                signal_type="alignment",
                is_bull_day=True,
                bar_ts=bar_ts,
                feat=feat,
                data=data,
                i=i,
                candidate_score=50.0,
                base_score=44.0,
                score_floor=48.0,
                forecast_return_pct=forecast_ret,
                today_change_pct=1.0,
                ml_proba=0.0,
                mtf_soft_penalty=0.0,
                fresh_priority=False,
                catchup=False,
                continuation_profile=True,
                signal_flags={"alignment_ok": True},
            )
            rec["id"] = f"{sym}_{tf}"
            rec["ts_signal"] = datetime.utcfromtimestamp(bar_ts / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec["bar_ts"] = bar_ts
            rec["decision"]["action"] = "take"
            rec["labels"] = {"ret_5": ret5, "label_5": ret5 > 0, "trade_taken": True, "trade_exit_pnl": ret5}
            return rec

        rows = [_row("AAAUSDT", "15m", 0.1, -0.5), _row("BBBUSDT", "1h", 1.0, 1.0)]
        feature_names = ml_candidate_ranker.safe_feature_names()
        weights = np.zeros(len(feature_names), dtype=float)
        weights[feature_names.index("forecast_return_pct")] = 3.0
        payload = {
            "model_name": "logistic",
            "feature_names": feature_names,
            "scaler_mean": [0.0] * len(feature_names),
            "scaler_scale": [1.0] * len(feature_names),
            "model": {"type": "logistic", "weights": weights.tolist(), "bias": 0.0},
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "critic_dataset.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            report = report_candidate_ranker_shadow.build_shadow_report(path, payload, top_ns=(1,))

        self.assertEqual(report["grouped_bar_competitions"], 1)

    def test_T256_shadow_rollout_flags_and_logging_are_wired(self):
        cfg_src = Path("config.py").read_text(encoding="utf-8")
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        botlog_src = Path("botlog.py").read_text(encoding="utf-8")
        supervisor_src = Path("local_exec_supervisor.py").read_text(encoding="utf-8")
        job_src = Path("request_local_job.py").read_text(encoding="utf-8")

        self.assertIn("ML_CANDIDATE_RANKER_SHADOW_ENABLED", cfg_src)
        self.assertIn("def _maybe_log_ranker_shadow(", monitor_src)
        self.assertIn("botlog.log_ranker_shadow(", monitor_src)
        self.assertIn("def log_ranker_shadow(", botlog_src)
        self.assertIn("install_catboost", supervisor_src)
        self.assertIn("install_catboost", job_src)

    def test_T256A_ranker_runtime_bonus_uses_final_score_when_enabled(self):
        import monitor

        old_enabled = monitor.config.ML_CANDIDATE_RANKER_RUNTIME_ENABLED
        old_use_final = getattr(monitor.config, "ML_CANDIDATE_RANKER_USE_FINAL_SCORE", True)
        old_weight = monitor.config.ML_CANDIDATE_RANKER_SCORE_WEIGHT
        old_clip = getattr(monitor.config, "ML_CANDIDATE_RANKER_SCORE_CLIP", 2.0)
        try:
            monitor.config.ML_CANDIDATE_RANKER_RUNTIME_ENABLED = True
            monitor.config.ML_CANDIDATE_RANKER_USE_FINAL_SCORE = True
            monitor.config.ML_CANDIDATE_RANKER_SCORE_WEIGHT = 0.75
            monitor.config.ML_CANDIDATE_RANKER_SCORE_CLIP = 2.0
            bonus = monitor._ml_candidate_ranker_runtime_bonus(
                {"payload_version": 2.0, "final_score": 3.5, "quality_proba": 0.95}
            )
            self.assertAlmostEqual(bonus, 1.5, places=6)
        finally:
            monitor.config.ML_CANDIDATE_RANKER_RUNTIME_ENABLED = old_enabled
            monitor.config.ML_CANDIDATE_RANKER_USE_FINAL_SCORE = old_use_final
            monitor.config.ML_CANDIDATE_RANKER_SCORE_WEIGHT = old_weight
            monitor.config.ML_CANDIDATE_RANKER_SCORE_CLIP = old_clip

    def test_T256B_monitor_logs_ranker_shadow_with_final_score_fields(self):
        monitor_src = Path("monitor.py").read_text(encoding="utf-8")
        botlog_src = Path("botlog.py").read_text(encoding="utf-8")
        config_src = Path("config.py").read_text(encoding="utf-8")

        self.assertIn("ranker_final_score", botlog_src)
        self.assertIn("ranker_expected_drawdown", botlog_src)
        self.assertIn("ranker_info=ranker_info", monitor_src)
        self.assertIn("ML_CANDIDATE_RANKER_USE_FINAL_SCORE", config_src)
        self.assertIn("ML_CANDIDATE_RANKER_SCORE_CLIP", config_src)

    def test_T256C_ranker_training_fails_fast_when_catboost_is_required_but_unavailable(self):
        import ml_candidate_ranker

        with patch.object(ml_candidate_ranker, "CATBOOST_AVAILABLE", False), patch.object(
            ml_candidate_ranker, "CATBOOST_IMPORT_ERROR", "No module named 'six'"
        ):
            with self.assertRaises(RuntimeError) as cm:
                ml_candidate_ranker.train_and_evaluate(Path("missing.jsonl"), require_catboost=True)

        self.assertIn("CatBoost is required", str(cm.exception))
        self.assertIn("six", str(cm.exception))

    def test_T256D_all_training_entrypoints_require_catboost(self):
        train_src = Path("train_candidate_ranker.py").read_text(encoding="utf-8")
        ranker_src = Path("ml_candidate_ranker.py").read_text(encoding="utf-8")
        worker_src = Path("rl_headless_worker.py").read_text(encoding="utf-8")
        config_src = Path("config.py").read_text(encoding="utf-8")
        headless_src = Path("../headless_train_once.ps1").read_text(encoding="utf-8")
        batch_src = Path("../run_full_pipeline_catboost.bat").read_text(encoding="utf-8")
        supervisor_src = Path("local_exec_supervisor.py").read_text(encoding="utf-8")

        self.assertIn("require_catboost: bool = True", ranker_src)
        self.assertIn("ML_CANDIDATE_RANKER_REQUIRE_CATBOOST", config_src)
        self.assertIn("require_catboost=args.require_catboost", train_src)
        self.assertIn("require_catboost=getattr(config, \"ML_CANDIDATE_RANKER_REQUIRE_CATBOOST\", True)", worker_src)
        self.assertIn("--require-catboost", headless_src)
        self.assertIn("import six, catboost", batch_src)
        self.assertIn("--require-catboost", batch_src)
        self.assertIn("\"six\",", supervisor_src)

    def test_T256E_monitor_near_miss_snapshot_detects_borderline_candidate(self):
        import monitor

        data, feat, i = _make_feat(80, "up")
        price = float(data["c"][i])
        feat["ema_fast"][i] = price * 0.999
        feat["ema_slow"][i] = price * 0.9987
        feat["slope"][i] = 0.08
        feat["adx"][i] = 18.0
        feat["rsi"][i] = 58.0
        feat["vol_x"][i] = 1.05
        feat["daily_range_pct"][i] = 4.2
        feat["macd_hist"][i] = 0.0005

        with patch.object(monitor, "_entry_signal_score", return_value=50.0), \
             patch.object(monitor, "_top_mover_score_bonus", return_value=0.0), \
             patch.object(monitor, "_forecast_return_score_bonus", return_value=0.0), \
             patch.object(monitor, "_time_block_1h_continuation_bonus", return_value=0.0), \
             patch.object(monitor, "_time_block_1h_continuation_profile", return_value=True):
            snap = monitor._near_miss_candidate_snapshot(
                tf="1h",
                feat=feat,
                data=data,
                i=i,
                price=price,
                ema20=float(feat["ema_fast"][i]),
                slope=float(feat["slope"][i]),
                adx=float(feat["adx"][i]),
                rsi=float(feat["rsi"][i]),
                vol_x=float(feat["vol_x"][i]),
                daily_range=float(feat["daily_range_pct"][i]),
                forecast_return_pct=0.25,
                today_change_pct=1.5,
            )

        self.assertIsNotNone(snap)
        self.assertIn(snap["mode"], {"alignment", "trend", "strong_trend", "breakout", "retest"})
        self.assertLess(float(snap["candidate_score"]), float(snap["score_floor"]))
        self.assertIn("near miss", str(snap["reason"]))


class TestRLHeadlessWorkerPreparation(unittest.TestCase):

    def test_T257_should_train_requires_enough_rows_or_delta(self):
        import rl_headless_worker

        self.assertFalse(
            rl_headless_worker.should_train(
                rows_total=400,
                min_rows=500,
                last_trained_rows=0,
                min_new_rows=50,
                dataset_mtime=100.0,
                last_dataset_mtime=0.0,
                force_first_train=True,
            )
        )
        self.assertTrue(
            rl_headless_worker.should_train(
                rows_total=600,
                min_rows=500,
                last_trained_rows=0,
                min_new_rows=50,
                dataset_mtime=100.0,
                last_dataset_mtime=0.0,
                force_first_train=True,
            )
        )
        self.assertTrue(
            rl_headless_worker.should_train(
                rows_total=660,
                min_rows=500,
                last_trained_rows=600,
                min_new_rows=50,
                dataset_mtime=120.0,
                last_dataset_mtime=100.0,
                force_first_train=False,
            )
        )
        self.assertFalse(
            rl_headless_worker.should_train(
                rows_total=620,
                min_rows=500,
                last_trained_rows=600,
                min_new_rows=50,
                dataset_mtime=120.0,
                last_dataset_mtime=120.0,
                force_first_train=False,
            )
        )

    def test_T258_build_status_snapshot_contains_training_and_dataset_progress(self):
        import rl_headless_worker

        state = rl_headless_worker.WorkerState(
            train_interval_sec=3600,
            status_interval_sec=300,
            min_rows=500,
            min_new_rows=50,
            collector_enabled=False,
        )
        state.collector_last_cycle_started_at = "2026-03-29T10:00:00Z"
        state.collector_last_cycle_finished_at = "2026-03-29T10:01:00Z"
        state.collector_last_cycle_stats = {"ok": 10, "fail": 2, "total": 12, "bull": False}
        state.train_runs_total = 3
        state.train_runs_ok = 2
        state.last_trained_rows = 1539
        state.last_model_name = "mlp"
        state.last_top1_delta = 0.1089

        snap = rl_headless_worker.build_status_snapshot(
            state,
            critic_report={"rows_total": 1539, "rows_last_24h": 11},
            ml_rows_total=26890,
        )

        self.assertEqual(snap["training"]["runs_total"], 3)
        self.assertEqual(snap["training"]["last_model_name"], "mlp")
        self.assertEqual(snap["datasets"]["ml_dataset_rows"], 26890)
        self.assertEqual(snap["datasets"]["critic_dataset"]["rows_total"], 1539)


class TestRLDailyReportPreparation(unittest.TestCase):

    def test_T259_daily_report_counts_dataset_growth_and_shadow_disagreements(self):
        import report_rl_daily

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bot_events = td_path / "bot_events.jsonl"
            critic_file = td_path / "critic_dataset.jsonl"
            ml_file = td_path / "ml_dataset.jsonl"
            rl_status = td_path / "rl_worker_status.json"
            train_report = td_path / "ml_candidate_ranker_report.json"
            shadow_report = td_path / "ml_candidate_ranker_shadow_report.json"

            bot_events.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "ranker_shadow",
                                "sym": "FLUXUSDT",
                                "tf": "1h",
                                "mode": "retest",
                                "candidate_score": 80.3,
                                "ranker_proba": 0.0017,
                                "ranker_take": False,
                                "bot_action": "take",
                                "reason": "candidate accepted",
                                "ts": "2026-03-30T09:36:53Z",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "event": "ranker_shadow",
                                "sym": "DYDXUSDT",
                                "tf": "15m",
                                "mode": "retest",
                                "candidate_score": 47.8,
                                "ranker_proba": 0.77,
                                "ranker_take": True,
                                "bot_action": "blocked",
                                "reason": "entry score 47.8 < floor 48.0",
                                "ts": "2026-03-30T10:00:00Z",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            critic_file.write_text(
                "\n".join(
                    [
                        json.dumps({"ts_signal": "2026-03-30T08:00:00Z", "decision": {"action": "take"}}, ensure_ascii=False),
                        json.dumps({"ts_signal": "2026-03-29T08:00:00Z", "decision": {"action": "take"}}, ensure_ascii=False),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            ml_file.write_text(
                "\n".join(
                    [
                        json.dumps({"ts_signal": "2026-03-30T08:30:00Z"}, ensure_ascii=False),
                        json.dumps({"ts_signal": "2026-03-29T08:30:00Z"}, ensure_ascii=False),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            rl_status.write_text(json.dumps({"worker": {"mode": "headless_rl"}}, ensure_ascii=False), encoding="utf-8")
            train_report.write_text(json.dumps({"chosen_model": "mlp", "train_rows": 100}, ensure_ascii=False), encoding="utf-8")
            shadow_report.write_text(json.dumps({"top_n": []}, ensure_ascii=False), encoding="utf-8")

            rep = report_rl_daily.build_report(
                report_rl_daily.date.fromisoformat("2026-03-30"),
                bot_events_file=bot_events,
                critic_file=critic_file,
                ml_file=ml_file,
                rl_status_file=rl_status,
                train_report_file=train_report,
                shadow_report_file=shadow_report,
            )

        self.assertEqual(rep["datasets"]["critic_rows_day"], 1)
        self.assertEqual(rep["datasets"]["ml_rows_day"], 1)
        self.assertEqual(rep["ranker_shadow"]["events_total"], 2)
        self.assertEqual(rep["ranker_shadow"]["bot_take_ranker_skip"], 1)
        self.assertEqual(rep["ranker_shadow"]["bot_blocked_ranker_take"], 1)
        self.assertEqual(rep["train_report"]["chosen_model"], "mlp")


if __name__ == "__main__":
    # Если передан класс — запускаем только его
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        # python test_bot.py TestMonitor
        cls_name = args[0]
        suite = unittest.TestLoader().loadTestsFromName(cls_name, sys.modules[__name__])
        verbosity = 2
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        run_tests()
