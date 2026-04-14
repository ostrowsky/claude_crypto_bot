"""
data_collector.py — Непрерывный сборщик данных для ML датасета.

Запускается при старте бота как фоновая asyncio задача.
Работает 24/7 независимо от анализа, мониторинга и действий пользователя.

Каждые 15 минут (синхронизовано с закрытием бара):
  - Для всех 81 монеты из watchlist
  - Загружает последние FETCH_LIMIT баров
  - Логирует последний ЗАКРЫТЫЙ бар в ml_dataset.jsonl
  - Заполняет forward labels (T+3/T+5/T+10) для ранее записанных баров

Дедупликация встроена в ml_dataset.log_bar_snapshot —
повторный вызов с тем же (sym, tf, bar_ts) игнорируется.

Запуск из bot.py:
    import data_collector
    asyncio.create_task(data_collector.run_forever(app))
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import numpy as np

import config
import critic_dataset
import ml_dataset
from indicators import compute_features
from strategy import fetch_klines, check_entry_conditions, check_retest_conditions, \
    check_breakout_conditions, check_impulse_conditions, check_alignment_conditions, \
    check_trend_surge_conditions, get_entry_mode

log = logging.getLogger("data_collector")

# ── Настройки ──────────────────────────────────────────────────────────────────
FETCH_LIMIT   = 120    # баров на монету (достаточно для SEQ_LEN=20 + прогрев)
BAR_SECONDS   = 900    # 15 минут = длина бара
POLL_INTERVAL = BAR_SECONDS  # опрашиваем раз в закрытие бара
BATCH_SIZE    = 10     # монет параллельно (не перегружать API)
BATCH_DELAY   = 0.3    # секунды между батчами


# ── Вычисление rule_signal ─────────────────────────────────────────────────────

def _detect_rule_signal(feat: dict, i: int, data: np.ndarray) -> str:
    """Что сказала бы стратегия на этом баре."""
    try:
        c = data["c"].astype(float)
        brk_ok, _ = check_breakout_conditions(feat, i)  # нет третьего аргумента
        if brk_ok:
            return "breakout"
        ret_ok, _ = check_retest_conditions(feat, i)   # нет третьего аргумента
        if ret_ok:
            return "retest"
        buy_ok, _ = check_entry_conditions(feat, i, c)
        if buy_ok:
            return get_entry_mode(feat, i)
        surge_ok, _ = check_trend_surge_conditions(feat, i)
        if surge_ok:
            return "impulse_speed"
        imp_ok, _ = check_impulse_conditions(feat, i)
        if imp_ok:
            return "impulse"
        aln_ok, _ = check_alignment_conditions(feat, i)
        if aln_ok:
            return "alignment"
    except Exception:
        pass
    return "none"


def _signal_flags_from_rule_signal(rule_signal: str) -> dict:
    return {
        "entry_ok": rule_signal in {"trend", "strong_trend"},
        "breakout_ok": rule_signal == "breakout",
        "retest_ok": rule_signal == "retest",
        "surge_ok": rule_signal == "impulse_speed",
        "impulse_ok": rule_signal == "impulse",
        "alignment_ok": rule_signal == "alignment",
    }


def _process_coin_sync(
    sym: str,
    tf: str,
    is_bull_day: bool,
    btc_vs_ema50: float,
    btc_momentum_4h: float,
    market_vol_24h: float,
    data,
) -> bool:
    """
    CPU/file-heavy part of the collector hot path.

    Runs in a worker thread so the main asyncio loop can continue servicing
    Telegram polling while we compute features and append/fill dataset rows.
    """
    c = data["c"].astype(float)
    i = len(c) - 2  # последний ЗАКРЫТЫЙ бар
    if i < 20:
        return False

    feat = compute_features(
        data["o"], data["h"], data["l"], c, data["v"]
    )

    rule_signal = _detect_rule_signal(feat, i, data)

    if rule_signal != "none":
        critic_dataset.log_candidate(
            sym=sym,
            tf=tf,
            bar_ts=int(data["t"][i]),
            signal_type=rule_signal,
            is_bull_day=is_bull_day,
            feat=feat,
            i=i,
            data=data,
            action="candidate",
            reason_code="rule_signal",
            reason="collector detected candidate",
            stage="collector",
            candidate_score=0.0,
            base_score=0.0,
            score_floor=0.0,
            forecast_return_pct=0.0,
            today_change_pct=0.0,
            ml_proba=None,
            mtf_soft_penalty=0.0,
            fresh_priority=False,
            catchup=False,
            continuation_profile=rule_signal in {"impulse_speed", "impulse", "alignment"},
            signal_flags=_signal_flags_from_rule_signal(rule_signal),
            btc_vs_ema50=btc_vs_ema50,
            btc_momentum_4h=btc_momentum_4h,
            market_vol_24h=market_vol_24h,
        )

    ml_dataset.log_bar_snapshot(
        sym=sym, tf=tf,
        bar_ts=int(data["t"][i]),
        rule_signal=rule_signal,
        is_bull_day=is_bull_day,
        feat=feat, i=i, data=data,
        btc_vs_ema50=btc_vs_ema50,
        btc_momentum_4h=btc_momentum_4h,
        market_vol_24h=market_vol_24h,
    )

    # Заполняем forward labels для ранее записанных баров
    # ВАЖНО: bar_ms должен соответствовать tf записи, а не всегда 15m!
    # Если tf="1h" и horizon=3, то T+3 = 3 часа (180 мин), а не 45 минут.
    _TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    bar_ms = _TF_SECONDS.get(tf, BAR_SECONDS) * 1000
    ml_dataset.fill_pending_from_data(
        sym=sym, tf=tf,
        t_arr=data["t"].astype(int),
        c_arr=c,
        bar_ms=bar_ms,
    )
    critic_dataset.fill_pending_from_data(
        sym=sym, tf=tf,
        t_arr=data["t"].astype(int),
        c_arr=c,
        bar_ms=bar_ms,
    )
    return True


# ── Обработка одной монеты ─────────────────────────────────────────────────────

async def _process_coin(
    session:          aiohttp.ClientSession,
    sym:              str,
    tf:               str,
    is_bull_day:      bool,
    btc_vs_ema50:     float,
    btc_momentum_4h:  float = 0.0,
    market_vol_24h:   float = 0.0,
) -> bool:
    """
    Загружает бары монеты, логирует последний закрытый бар,
    заполняет forward labels для ранее залогированных.
    """
    try:
        data = await fetch_klines(session, sym, tf, limit=FETCH_LIMIT)
        if data is None or len(data) < 30:
            return False

        return await asyncio.to_thread(
            _process_coin_sync,
            sym,
            tf,
            is_bull_day,
            btc_vs_ema50,
            btc_momentum_4h,
            market_vol_24h,
            data,
        )

    except Exception as e:
        log.debug("_process_coin %s error: %s", sym, e)
        return False


# ── Один цикл сбора данных ─────────────────────────────────────────────────────

async def _collect_once(btc_context: dict) -> dict:
    """
    Обходит все монеты вотчлиста батчами в одной сессии.
    Первый батч — BTC контекст (уже вычислен), затем монеты.
    """
    symbols    = config.load_watchlist()
    timeframes = list(config.TIMEFRAMES)
    pairs      = [(sym, tf) for sym in symbols for tf in timeframes]

    is_bull_day       = btc_context.get("is_bull", False)
    btc_vs_ema50      = btc_context.get("btc_vs_ema50", 0.0)
    btc_momentum_4h   = btc_context.get("btc_momentum_4h", 0.0)
    market_vol_24h    = btc_context.get("market_vol_24h", 0.0)

    ok = 0
    fail = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[batch_start: batch_start + BATCH_SIZE]
            tasks = [
                _process_coin(
                    session, sym, tf,
                    is_bull_day, btc_vs_ema50,
                    btc_momentum_4h, market_vol_24h,
                )
                for sym, tf in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if r is True:
                    ok += 1
                else:
                    fail += 1
            await asyncio.sleep(0)
            if batch_start + BATCH_SIZE < len(pairs):
                await asyncio.sleep(BATCH_DELAY)

    return {"ok": ok, "fail": fail, "total": len(pairs)}


# ── Рыночный контекст (вычисляется один раз за цикл) ──────────────────────────

async def _get_market_context(session: aiohttp.ClientSession) -> dict:
    """
    Собирает признаки режима рынка — одинаковые для всех монет в цикле.

    Возвращает:
      is_bull         — bool, BTC выше EMA50 и slope > 0
      btc_vs_ema50    — % отклонение BTC от EMA50 (1h)
      btc_momentum_4h — % изменение BTC за последние 4 часа (16 × 15m баров)
      market_vol_24h  — средняя волатильность вотчлиста за 24ч (прокси: ATR% BTC)
      green_ratio     — доля монет вотчлиста с close > open на последнем баре
                        (дорогостоящий — считаем только если данные уже загружены)
    """
    ctx = {
        "is_bull":        False,
        "btc_vs_ema50":   0.0,
        "btc_momentum_4h": 0.0,
        "market_vol_24h": 0.0,
    }
    try:
        from indicators import _ema
        # BTC на 15m для momentum и волатильности
        btc15 = await fetch_klines(session, "BTCUSDT", "15m", limit=100)
        if btc15 is not None and len(btc15) >= 50:
            c15 = btc15["c"].astype(float)
            # 4h momentum = изменение за последние 16 баров по 15m
            if len(c15) >= 17 and c15[-17] > 0:
                ctx["btc_momentum_4h"] = round((c15[-2] / c15[-17] - 1) * 100, 4)
            # Волатильность 24ч = std доходностей за 96 баров
            if len(c15) >= 97:
                rets = np.diff(c15[-97:]) / c15[-97:-1]
                ctx["market_vol_24h"] = round(float(np.std(rets)) * 100, 4)

        # BTC на 1h для EMA50 и is_bull
        btc1h = await fetch_klines(session, "BTCUSDT", "1h", limit=60)
        if btc1h is not None and len(btc1h) >= 55:
            c1h   = btc1h["c"].astype(float)
            ema50 = _ema(c1h, 50)
            btc_price = float(c1h[-2])  # закрытый бар
            btc_ema50 = float(ema50[-2])
            slope = (ema50[-2] - ema50[-8]) / ema50[-8] * 100 if ema50[-8] > 0 else 0.0
            is_bull = btc_price > btc_ema50 and slope > 0
            btc_vs_ema50 = (btc_price / btc_ema50 - 1) * 100 if btc_ema50 > 0 else 0.0
            ctx["is_bull"]      = is_bull
            ctx["btc_vs_ema50"] = round(btc_vs_ema50, 4)

        # Обновляем config для мониторинга
        config._bull_day_active = ctx["is_bull"]
        config._btc_vs_ema50    = ctx["btc_vs_ema50"]

    except Exception as e:
        log.debug("_get_market_context error: %s", e)
    return ctx


async def _get_btc_context() -> dict:
    """Обёртка для обратной совместимости."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    ) as session:
        return await _get_market_context(session)


# ── Синхронизация с закрытием бара ────────────────────────────────────────────

def _seconds_until_next_bar() -> float:
    """
    Возвращает секунды до закрытия следующего 15-минутного бара
    плюс небольшой буфер (~5 секунд) чтобы бар точно закрылся.
    """
    now = time.time()
    bar_boundary = (now // BAR_SECONDS + 1) * BAR_SECONDS
    return max(5.0, bar_boundary - now + 5.0)


# ── Основной loop ──────────────────────────────────────────────────────────────

async def run_forever(app=None) -> None:
    """
    Главный loop сборщика данных.
    Запускается один раз при старте бота и работает пока бот жив.

    app — telegram Application (опционально, для будущих уведомлений).
    """
    log.info("DataCollector started — logging all %d watchlist coins every 15m",
             len(config.load_watchlist()))

    # Первый запуск — сразу, без ожидания
    _first_run = True

    while True:
        if not _first_run:
            # Ждём следующего закрытия бара
            wait = _seconds_until_next_bar()
            log.debug("DataCollector: sleeping %.0fs until next bar", wait)
            await asyncio.sleep(wait)
        _first_run = False

        cycle_start = time.time()
        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

        try:
            # Получаем рыночный контекст (один раз на цикл, одна сессия)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)
            ) as _ctx_session:
                btc_ctx = await _get_market_context(_ctx_session)

            # Собираем данные по всем монетам
            stats = await _collect_once(btc_ctx)

            elapsed = time.time() - cycle_start
            log.info(
                "DataCollector %s: %d/%d coins OK, %.1fs, bull=%s",
                now_str, stats["ok"], stats["total"],
                elapsed, btc_ctx["is_bull"],
            )

            # Статистика размера датасета раз в час
            if datetime.now(timezone.utc).minute < 15:
                await asyncio.to_thread(_log_dataset_stats)

        except asyncio.CancelledError:
            log.info("DataCollector cancelled — stopping")
            break
        except Exception as e:
            log.error("DataCollector cycle error: %s", e)
            await asyncio.sleep(60)  # при ошибке — пауза и повтор


def _log_dataset_stats() -> None:
    """Логирует статистику датасетов раз в час."""
    import json
    try:
        if ml_dataset.ML_FILE.exists():
            lines = ml_dataset.ML_FILE.read_text(encoding="utf-8").splitlines()
            records = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
            total = len(records)
            labeled = sum(1 for r in records if r["labels"]["ret_3"] is not None)
            signals = sum(1 for r in records if r["signal_type"] != "none")
            size_kb = ml_dataset.ML_FILE.stat().st_size // 1024
            log.info(
                "ML Dataset: %d records (%d labeled, %d signals) — %d KB",
                total, labeled, signals, size_kb,
            )

        if critic_dataset.CRITIC_FILE.exists():
            lines = critic_dataset.CRITIC_FILE.read_text(encoding="utf-8").splitlines()
            records = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
            total = len(records)
            labeled = sum(1 for r in records if r.get("labels", {}).get("ret_3") is not None)
            taken = sum(1 for r in records if bool(r.get("labels", {}).get("trade_taken")))
            outcomes = sum(1 for r in records if r.get("labels", {}).get("trade_exit_pnl") is not None)
            size_kb = critic_dataset.CRITIC_FILE.stat().st_size // 1024
            log.info(
                "Critic Dataset: %d records (%d labeled, %d taken, %d outcomes) — %d KB",
                total, labeled, taken, outcomes, size_kb,
            )
        else:
            log.info("Critic Dataset: 0 records (file not created yet)")
    except Exception:
        pass
