from __future__ import annotations

"""
Structured event logger for the crypto trend bot.

Writes one JSON object per line to bot_events.jsonl
Each event has: ts (ISO UTC), event type, and relevant fields.

Event types:
  analysis_done  — результат market_scan
  bull_day       — определение характера дня
  entry          — открытие позиции
  exit           — закрытие позиции
  blocked        — сигнал был, но не прошёл условие (только топ-причины)
  forward        — результат T+N прогноза
  cooldown       — повторный вход заблокирован кулдауном
  reanalyze      — авто-реанализ завершён

Usage:
  from botlog import log_entry, log_exit, log_blocked, log_analysis, log_forward

Анализ логов: скопируй bot_events.jsonl и загрузи в чат.
"""

import json
import logging
import numpy as _np


class _SafeEncoder(json.JSONEncoder):
    """Конвертирует numpy типы в Python-примитивы перед сериализацией."""
    def default(self, obj):
        if isinstance(obj, _np.bool_):     return bool(obj)
        if isinstance(obj, _np.integer):   return int(obj)
        if isinstance(obj, _np.floating):  return float(obj)
        if isinstance(obj, _np.ndarray):   return obj.tolist()
        return super().default(obj)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_FILE = Path("bot_events.jsonl")
_pylog   = logging.getLogger("botlog")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(event: Dict[str, Any]) -> None:
    event.setdefault("ts", _now())
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, cls=_SafeEncoder) + "\n")
    except Exception as e:
        _pylog.warning("botlog write error: %s", e)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_bull_day(is_bull: bool, btc_price: float, btc_ema50: float,
                 eff_range_max: float, eff_rsi_hi: float) -> None:
    """П5: характер дня — вызывать после is_bull_day() в market_scan."""
    _write({
        "event":         "bull_day",
        "is_bull":       is_bull,
        "btc_price":     round(btc_price, 2),
        "btc_ema50":     round(btc_ema50, 2),
        "btc_vs_ema50":  round((btc_price / btc_ema50 - 1) * 100, 2),
        "eff_range_max": eff_range_max,
        "eff_rsi_hi":    eff_rsi_hi,
    })


def log_analysis(n_scanned: int, n_confirmed: int, n_signal_now: int,
                 n_setup: int, n_early: int, is_bull: bool,
                 confirmed_symbols: list[str]) -> None:
    """Результат market_scan."""
    _write({
        "event":             "analysis_done",
        "n_scanned":         n_scanned,
        "n_confirmed":       n_confirmed,
        "n_signal_now":      n_signal_now,
        "n_setup":           n_setup,
        "n_early":           n_early,
        "is_bull_day":       is_bull,
        "confirmed_symbols": confirmed_symbols,
    })


def log_entry(sym: str, tf: str, mode: str, price: float,
              ema20: float, slope: float, rsi: float, adx: float,
              vol_x: float, macd_hist: float, daily_range: float,
              trail_k: float, max_hold_bars: int,
              ml_proba: Optional[float] = None) -> None:
    """Открытие позиции."""
    rec: Dict[str, Any] = {
        "event":        "entry",
        "sym":          sym,
        "tf":           tf,
        "mode":         mode,         # trend/strong_trend/retest/breakout
        "price":        price,
        "ema20":        round(ema20, 8),
        "slope_pct":    round(slope, 3),
        "rsi":          round(rsi, 1),
        "adx":          round(adx, 1),
        "vol_x":        round(vol_x, 2),
        "macd_hist":    round(macd_hist, 8),
        "daily_range":  round(daily_range, 2),
        "trail_k":      trail_k,
        "max_hold_bars":max_hold_bars,
    }
    if ml_proba is not None:
        rec["ml_proba"] = round(float(ml_proba), 4)
    _write(rec)


def log_exit(sym: str, tf: str, mode: str, entry_price: float,
             exit_price: float, reason: str, bars_held: int,
             trail_k: float) -> None:
    """Закрытие позиции."""
    pnl = round((exit_price / entry_price - 1) * 100, 3) if entry_price > 0 else 0.0
    _write({
        "event":       "exit",
        "sym":         sym,
        "tf":          tf,
        "mode":        mode,
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "pnl_pct":     pnl,
        "reason":      reason,
        "bars_held":   bars_held,
        "trail_k":     trail_k,
    })


def log_forward(sym: str, tf: str, mode: str, horizon: int,
                entry_price: float, forward_price: float, correct: bool) -> None:
    """Результат форвард-прогноза T+N."""
    pnl = round((forward_price / entry_price - 1) * 100, 3) if entry_price > 0 else 0.0
    _write({
        "event":         "forward",
        "sym":           sym,
        "tf":            tf,
        "mode":          mode,
        "horizon":       horizon,
        "entry_price":   entry_price,
        "forward_price": forward_price,
        "pnl_pct":       pnl,
        "correct":       correct,
    })


def log_blocked(sym: str, tf: str, price: float, reason: str,
                rsi: Optional[float] = None, adx: Optional[float] = None,
                vol_x: Optional[float] = None, daily_range: Optional[float] = None,
                signal_type: str = "buy") -> None:
    """
    Сигнал был (все основные условия выполнены), но что-то заблокировало вход.
    Логируем только когда блокировка случилась в мониторинге (не в анализе) —
    иначе будет слишком много записей.
    signal_type: "buy"/"retest"/"breakout" — какой тип сигнала проверялся
    """
    rec: Dict[str, Any] = {
        "event":       "blocked",
        "sym":         sym,
        "tf":          tf,
        "signal_type": signal_type,
        "price":       price,
        "reason":      reason,
    }
    if rsi        is not None: rec["rsi"]        = round(rsi, 1)
    if adx        is not None: rec["adx"]        = round(adx, 1)
    if vol_x      is not None: rec["vol_x"]      = round(vol_x, 2)
    if daily_range is not None: rec["daily_range"] = round(daily_range, 2)
    _write(rec)


def log_ranker_shadow(
    sym: str,
    tf: str,
    mode: str,
    price: float,
    candidate_score: float,
    score_floor: float,
    ranker_proba: float,
    ranker_threshold: float,
    bot_action: str,
    reason: str = "",
    ranker_final_score: Optional[float] = None,
    ranker_ev: Optional[float] = None,
    ranker_expected_return: Optional[float] = None,
    ranker_expected_drawdown: Optional[float] = None,
) -> None:
    rec: Dict[str, Any] = {
        "event": "ranker_shadow",
        "sym": sym,
        "tf": tf,
        "mode": mode,
        "price": round(price, 8),
        "candidate_score": round(candidate_score, 4),
        "score_floor": round(score_floor, 4),
        "ranker_proba": round(ranker_proba, 6),
        "ranker_threshold": round(ranker_threshold, 6),
        "ranker_take": bool(ranker_proba >= ranker_threshold),
        "bot_action": bot_action,
        "reason": reason,
    }
    if ranker_final_score is not None:
        rec["ranker_final_score"] = round(float(ranker_final_score), 6)
    if ranker_ev is not None:
        rec["ranker_ev"] = round(float(ranker_ev), 6)
    if ranker_expected_return is not None:
        rec["ranker_expected_return"] = round(float(ranker_expected_return), 6)
    if ranker_expected_drawdown is not None:
        rec["ranker_expected_drawdown"] = round(float(ranker_expected_drawdown), 6)
    _write(rec)


def log_cooldown(sym: str, tf: str, bars_remaining: int, first: bool = False) -> None:
    """
    Вход заблокирован кулдауном. Логируется только при first=True
    (первое срабатывание) чтобы не спамить одинаковыми записями.
    """
    if not first:
        return
    _write({
        "event":          "cooldown_start",
        "sym":            sym,
        "tf":             tf,
        "bars_remaining": bars_remaining,
    })


def log_reanalyze(n_confirmed: int, n_dropped: int,
                  dropped_symbols: list[str]) -> None:
    """Авто-реанализ: сколько монет подтверждено, сколько выпало."""
    _write({
        "event":          "reanalyze",
        "n_confirmed":    n_confirmed,
        "n_dropped":      n_dropped,
        "dropped_symbols":dropped_symbols,
    })
