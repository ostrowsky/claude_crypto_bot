"""
Background scanner for early movement alerts.

Runs independently from the main monitoring loop and scans the current
watchlist on each 15m bar close.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
import numpy as np

import config
from strategy import (
    MarketRegime,
    check_ema_cross_conditions,
    check_exit_conditions,
    check_impulse_conditions,
    check_trend_surge_conditions,
    compute_features,
    detect_market_regime,
    fetch_klines,
    is_bull_day,
)

log = logging.getLogger("impulse")

_last_signal_ts: dict[str, int] = {}
_last_surge_ts: dict[str, int] = {}
_last_weak_ts: dict[str, int] = {}
_last_squeeze_ts: dict[str, int] = {}
_last_cross_ts: dict[str, int] = {}

WEAK_COOLDOWN_BARS: int = 4


def _seconds_until_next_bar() -> float:
    now = datetime.now(timezone.utc)
    secs = (15 - now.minute % 15) * 60 - now.second + 8
    return max(float(secs), 5.0)


def _in_cooldown(
    key: str,
    bar_ts_ms: int,
    tf: str,
    cooldown_bars: int,
    store: dict[str, int],
) -> bool:
    bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
    cooldown_ms = cooldown_bars * bar_ms
    return bar_ts_ms - store.get(key, 0) < cooldown_ms


async def _scan_once(send_fn) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    symbols = config.load_watchlist()

    async with aiohttp.ClientSession() as session:
        bull, _, _ = await is_bull_day(session)
        regime = await detect_market_regime(session)

        config._bull_day_active = bull
        config._current_regime = regime.name
        config._effective_range_max = (
            getattr(config, "BULL_DAY_RANGE_MAX", 14.0)
            if bull
            else config.DAILY_RANGE_MAX
        )

        cross_found: list[tuple[str, str, str]] = []
        impulse_found: list[tuple[str, str, str]] = []
        surge_found: list[tuple[str, str, str]] = []
        squeeze_found: list[tuple[str, str, str]] = []
        weak_found: list[tuple[str, str, str]] = []

        for batch_start in range(0, len(symbols), 10):
            batch = symbols[batch_start : batch_start + 10]
            fetch_tasks = {
                (sym, tf): asyncio.create_task(fetch_klines(session, sym, tf))
                for sym in batch
                for tf in config.TIMEFRAMES
            }

            for (sym, tf), task in fetch_tasks.items():
                try:
                    data = await task
                except Exception:
                    continue

                if data is None or len(data) < 30:
                    continue

                feat = compute_features(
                    data["o"], data["h"], data["l"], data["c"], data["v"]
                )
                c_arr = data["c"].astype(float)
                i_now = len(data) - 2
                bar_ts = int(data["t"][i_now])
                key = f"{sym}_{tf}"

                ok, reason = check_impulse_conditions(feat, i_now)
                if ok and not _in_cooldown(
                    key, bar_ts, tf, config.IMPULSE_COOLDOWN_BARS, _last_signal_ts
                ):
                    _last_signal_ts[key] = bar_ts
                    impulse_found.append((sym, tf, reason))

                surge_ok, surge_reason = check_trend_surge_conditions(feat, i_now)
                surge_cd = getattr(config, "SURGE_COOLDOWN_BARS", 20)
                if surge_ok and not _in_cooldown(
                    key, bar_ts, tf, surge_cd, _last_surge_ts
                ):
                    _last_surge_ts[key] = bar_ts
                    surge_found.append((sym, tf, surge_reason))

                sq_arr = feat.get("squeeze_breakout")
                if sq_arr is not None and i_now < len(sq_arr) and sq_arr[i_now] == 1.0:
                    if not _in_cooldown(key, bar_ts, tf, surge_cd, _last_squeeze_ts):
                        _last_squeeze_ts[key] = bar_ts
                        price = float(c_arr[i_now])
                        atr_v = (
                            feat["atr"][i_now]
                            if np.isfinite(feat["atr"][i_now])
                            else 0.0
                        )
                        accel = (
                            feat["slope_accel"][i_now]
                            if np.isfinite(feat["slope_accel"][i_now])
                            else 0.0
                        )
                        squeeze_found.append(
                            (
                                sym,
                                tf,
                                f"P={price:.6g} ATRx{atr_v:.4g} accel={accel:+.3f}%",
                            )
                        )

                cross_ok, cross_reason = check_ema_cross_conditions(feat, i_now)
                cross_cd = getattr(config, "CROSS_COOLDOWN_BARS", 6)
                if cross_ok and not _in_cooldown(
                    key, bar_ts, tf, cross_cd, _last_cross_ts
                ):
                    _last_cross_ts[key] = bar_ts
                    cross_found.append((sym, tf, cross_reason))

                ef = feat["ema_fast"][i_now]
                price_now = float(c_arr[i_now])
                if np.isfinite(ef) and price_now > float(ef) * 0.97:
                    exit_r = check_exit_conditions(feat, i_now, c_arr)
                    if exit_r and exit_r.startswith("⚠️ WEAK:"):
                        if not _in_cooldown(
                            key, bar_ts, tf, WEAK_COOLDOWN_BARS, _last_weak_ts
                        ):
                            _last_weak_ts[key] = bar_ts
                            weak_found.append((sym, tf, exit_r[8:].strip()))

            await asyncio.sleep(0.3)

    if cross_found:
        await _send_cross(cross_found, bull, regime, send_fn)
    if impulse_found:
        await _send_impulse(impulse_found, bull, send_fn)
    if surge_found:
        await _send_surge(surge_found, bull, send_fn)
    if squeeze_found:
        await _send_squeeze(squeeze_found, bull, regime, send_fn)
    if weak_found:
        await _send_weak(weak_found, send_fn)

    total = (
        len(cross_found)
        + len(impulse_found)
        + len(surge_found)
        + len(squeeze_found)
        + len(weak_found)
    )
    if total == 0:
        log.debug("Scan: 0 signals (regime: %s)", regime.name)


async def _send_cross(
    found: list[tuple[str, str, str]],
    bull: bool,
    regime: MarketRegime,
    send_fn,
) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    day_icon = "🐂 бычий" if bull else "➡️ нейтральный"
    regime_str = str(regime).replace("_", "\\-")

    lines = [
        f"📈 *EMA CROSS — {now_str}*  \\({day_icon} / {regime_str}\\)\n",
        "Пробой EMA20 снизу вверх — сигнал до подтверждения ADX/slope\n",
    ]
    for sym, tf, reason in sorted(found, key=lambda x: x[0]):
        reason_esc = (
            reason.replace("_", "\\_")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace(".", "\\.")
            .replace("-", "\\-")
            .replace("+", "\\+")
        )
        lines.append(f"  `{sym}` \\[{tf}\\]  {reason_esc}")
    lines.append(
        "\n_Ранний вход — ADX ещё не подтвердил\\._ "
        "_Обычный сигнал придёт через 3\\-5 баров\\._ "
        "_Стоп под последним минимумом\\._"
    )

    text = "\n".join(lines)
    log.info("EMA_CROSS: %d signals: %s", len(found), [s for s, _, _ in found])
    try:
        await send_fn(text)
    except Exception as e:
        log.warning("EMA_CROSS send error: %s", e)


async def _send_impulse(
    found: list[tuple[str, str, str]],
    bull: bool,
    send_fn,
) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    day_icon = "🐂 бычий" if bull else "➡️ нейтральный"

    lines = [
        f"⚡ *IMPULSE — {now_str}*  \\({day_icon}\\)\n",
        "Ранний вход — за 1\\-2 бара до стандартного сигнала\n",
    ]
    for sym, tf, reason in sorted(found, key=lambda x: x[0]):
        lines.append(f"  `{sym}` \\[{tf}\\]  {reason}")
    lines.append("\n_Форвард\\-тест не проводился — повышенный риск_")

    text = "\n".join(lines)
    log.info("IMPULSE: %d signals: %s", len(found), [s for s, _, _ in found])
    try:
        await send_fn(text)
    except Exception as e:
        log.warning("IMPULSE send error: %s", e)


async def _send_surge(
    found: list[tuple[str, str, str]],
    bull: bool,
    send_fn,
) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    day_icon = "🐂 бычий" if bull else "➡️ нейтральный"

    lines = [
        f"🚀 *TREND SURGE — {now_str}*  \\({day_icon}\\)\n",
        "Начало устойчивого тренда — slope ускорился \\+ MACD растёт\n",
    ]
    for sym, tf, reason in sorted(found, key=lambda x: x[0]):
        lines.append(f"  `{sym}` \\[{tf}\\]  {reason}")
    lines.append(
        "\n_Не подтверждён форвард\\-тестом\\._ "
        "_Вход на собственный риск, кулдаун 5ч\\._"
    )

    text = "\n".join(lines)
    log.info("TREND_SURGE: %d signals: %s", len(found), [s for s, _, _ in found])
    try:
        await send_fn(text)
    except Exception as e:
        log.warning("TREND_SURGE send error: %s", e)


async def _send_squeeze(
    found: list[tuple[str, str, str]],
    bull: bool,
    regime: MarketRegime,
    send_fn,
) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    regime_str = str(regime).replace("_", "\\-")

    lines = [
        f"🗜 *SQUEEZE BREAKOUT — {now_str}*  \\({regime_str}\\)\n",
        "ATR вышел из сжатия — пружина распрямляется\n",
    ]
    for sym, tf, reason in sorted(found, key=lambda x: x[0]):
        lines.append(f"  `{sym}` \\[{tf}\\]  {reason}")
    lines.append(
        "\n_Ранний сигнал\\. ADX ещё не подтвердил — объём и структура важнее\\. "
        "Повышенный риск\\._"
    )

    text = "\n".join(lines)
    log.info("SQUEEZE: %d signals: %s", len(found), [s for s, _, _ in found])
    try:
        await send_fn(text)
    except Exception as e:
        log.warning("SQUEEZE send error: %s", e)


async def _send_weak(
    found: list[tuple[str, str, str]],
    send_fn,
) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        return
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

    lines = [
        f"⚠️ *TREND WEAKNESS — {now_str}*\n",
        "Сигналы ослабления — рассмотрите ужесточение стопа или выход\n",
    ]
    for sym, tf, reason in sorted(found, key=lambda x: x[0]):
        reason_esc = (
            reason.replace("_", "\\_")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace(".", "\\.")
            .replace("-", "\\-")
        )
        lines.append(f"  `{sym}` \\[{tf}\\]  {reason_esc}")
    lines.append(
        "\n_Не сигнал выхода — но тренд теряет силу\\._ "
        "_Проверь ATR\\-трейл и MAX\\_HOLD\\_BARS\\._"
    )

    text = "\n".join(lines)
    log.info("WEAK: %d warnings: %s", len(found), [s for s, _, _ in found])
    try:
        await send_fn(text)
    except Exception as e:
        log.warning("WEAK send error: %s", e)


async def run_forever(app, send_fn=None) -> None:
    if not getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        log.info("Impulse scanner disabled by config")
        return
    log.info("Impulse scanner started")

    if send_fn is None:
        from bot import _make_broadcast_send

        send_fn = _make_broadcast_send(app)

    try:
        await _scan_once(send_fn)
    except Exception as e:
        log.error("Initial scan exception: %s", e, exc_info=True)

    while True:
        wait = _seconds_until_next_bar()
        log.debug("Next impulse scan in %.0f sec", wait)
        await asyncio.sleep(wait)
        try:
            await _scan_once(send_fn)
        except Exception as e:
            log.error("Scan exception: %s", e, exc_info=True)
