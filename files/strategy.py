from __future__ import annotations

"""
Intraday analysis logic.

Алгоритм для каждой монеты:
  1. Берём данные с начала сегодняшнего дня (00:00 UTC) + историю для прогрева индикаторов
  2. На сегодняшних барах находим все сигналы стратегии
  3. Для каждого сигнала у которого уже прошло T+3/5/10 свечей — проверяем был ли он верным
  4. Если точность сегодня ≥ порога → стратегия подтверждена для этой монеты сегодня
  5. Проверяем активен ли сигнал прямо сейчас (последняя закрытая свеча)

Если сегодня ещё мало данных (утро, мало сигналов) — помечаем как "не подтверждено"
и не добавляем в мониторинг.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np

import config
from indicators import compute_features, _ema
import botlog


# ── Market Regime ──────────────────────────────────────────────────────────────

class MarketRegime:
    """
    Определяет режим рынка по BTC и применяет адаптивные пороги.

    Режимы:
      bull_trend    — BTC > EMA20 > EMA50, ADX сильный → мягче RSI, range
      bear_trend    — BTC < EMA50, ADX сильный → только ретесты
      consolidation — ADX слабый → ждём пробоя, строже vol
      recovery      — BTC только что пробил EMA50 снизу → агрессивный вход
      neutral       — базовые параметры config

    Параметры хранятся в config.REGIME_PARAMS.
    """

    def __init__(self, regime_name: str = "neutral"):
        self.name = regime_name
        params = getattr(config, "REGIME_PARAMS", {})
        self._p = params.get(regime_name, params.get("neutral", {}))

    def get(self, key: str, fallback):
        """Возвращает параметр режима или fallback из config."""
        val = self._p.get(key)
        return fallback if val is None else val

    @property
    def rsi_hi(self)        -> float: return self.get("rsi_hi",    config.RSI_BUY_HI)
    @property
    def vol_mult(self)      -> float: return self.get("vol_mult",  config.VOL_MULT)
    @property
    def range_max(self)     -> float: return self.get("range_max", config.DAILY_RANGE_MAX)
    @property
    def adx_min(self)       -> float: return self.get("adx_min",   config.ADX_MIN)
    @property
    def slope_min(self)     -> float: return self.get("slope_min", config.EMA_SLOPE_MIN)
    @property
    def allow_new_buy(self) -> bool:  return self.name != "bear_trend"

    def __str__(self) -> str:
        icons = {
            "bull_trend":    "🐂",
            "bear_trend":    "🐻",
            "consolidation": "↔️",
            "recovery":      "🌱",
            "neutral":       "➡️",
        }
        return f"{icons.get(self.name, '?')} {self.name}"


def _get_coin_regime(feat: dict, i: int) -> MarketRegime:
    """Режим конкретной монеты по её собственным индикаторам."""
    regime_arr = feat.get("regime")
    if regime_arr is not None and i < len(regime_arr):
        name = str(regime_arr[i])
        return MarketRegime(name)
    return MarketRegime("neutral")


def _get_effective_range_max(feat: dict, i: int, regime: "MarketRegime") -> float:
    """
    Итоговый порог daily_range_max с учётом:
    1. Динамического (по волатильности монеты) — если DYNAMIC_RANGE_ENABLED
    2. Режима рынка
    3. Бычьего дня (_effective_range_max от market_scan)
    Берём наибольший из применимых порогов.
    """
    base = getattr(config, "_effective_range_max", config.DAILY_RANGE_MAX)

    # Динамический порог
    if getattr(config, "DYNAMIC_RANGE_ENABLED", True):
        dyn_arr = feat.get("dyn_range_max")
        if dyn_arr is not None and i < len(dyn_arr) and np.isfinite(dyn_arr[i]):
            dyn = float(dyn_arr[i])
            base = max(base, dyn)

    # Режим рынка
    return max(base, regime.range_max)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class HorizonAccuracy:
    """
    Метрики качества сигналов для горизонта T+h.

    Win% (pct) — старая метрика, оставлена для совместимости.
    Expectancy-метрики — новые, решают проблему "60% точность, убыточная стратегия":

      expected_return  — средний доход по всем сделкам (%)
                         E[R] = avg(return_i)
                         Если E[R] < 0 — стратегия убыточна даже при win% > 50%.

      median_return    — медианный доход (устойчив к выбросам)

      downside_q10     — 10% квантиль доходности (worst-case типичный исход)
                         Если downside < -ATR*2 — стоп недостаточно защищает.

      ev_proxy         — E[R] / |downside_q10| — отношение ожидания к хвостовому риску.
                         > 0.5 = хорошо, > 1.0 = отлично, < 0 = убыточно.
    """
    horizon: int
    total:   int
    correct: int

    # Expectancy поля (None если total == 0)
    expected_return:  Optional[float] = None   # среднее по всем T+h возвратам (%)
    median_return:    Optional[float] = None   # медиана возвратов (%)
    downside_q10:     Optional[float] = None   # 10-й процентиль (самые плохие 10%)
    upside_q90:       Optional[float] = None   # 90-й процентиль (самые хорошие 10%)
    ev_proxy:         Optional[float] = None   # expected_return / abs(downside_q10)

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    @property
    def is_positive_ev(self) -> bool:
        """True если стратегия имеет положительное математическое ожидание."""
        if self.expected_return is not None:
            return self.expected_return > 0
        return self.pct > 50.0  # fallback к win% если нет данных

    def __str__(self) -> str:
        base = f"T+{self.horizon}: {self.pct:.0f}% ({self.correct}/{self.total})"
        if self.expected_return is not None:
            ev_str = f" EV={self.expected_return:+.2f}%"
            if self.downside_q10 is not None:
                ev_str += f" q10={self.downside_q10:+.2f}%"
            return base + ev_str
        return base

    def short_str(self) -> str:
        """Компактная строка для Telegram-сообщений."""
        if self.total == 0:
            return f"T+{self.horizon}: —"
        if self.expected_return is not None:
            ev_icon = "✅" if self.expected_return > 0 else "❌"
            return f"T+{self.horizon}: {self.pct:.0f}% {ev_icon} EV{self.expected_return:+.2f}%"
        return f"T+{self.horizon}: {self.pct:.0f}% ({self.correct}/{self.total})"


@dataclass
class CoinReport:
    symbol:        str
    tf:            str

    # Сегодняшние данные форвард-теста
    today_signals:  int                        # сигналов найдено сегодня
    today_accuracy: Dict[int, HorizonAccuracy] # точность на сегодняшних данных
    today_confirmed: bool                      # стратегия подтверждена сегодня

    best_horizon:  int
    best_accuracy: float
    in_play:       bool

    note:          str   = ""
    from_scan:     bool  = False

    # Текущий статус сигнала (последняя закрытая свеча)
    signal_now:    bool  = False
    current_price: float = 0.0
    current_slope: float = 0.0
    current_rsi:   float = 0.0
    current_adx:   float = 0.0
    current_vol_x: float = 0.0
    current_macd:  float = 0.0

    # Почему сигнала нет прямо сейчас (для диагностики)
    no_signal_reason: str = ""

    # SETUP: структура бычья, но не хватает 1 жёсткого фильтра
    setup_now:          bool = False
    setup_reason:       str  = ""  # что именно не дотягивает до BUY
    setup_missing_count: int = 99  # П8: кол-во недостающих условий (меньше = ближе к BUY)

    # П7: тип активного сигнала: "trend"/"strong_trend"/"retest"/"breakout"
    signal_mode:  str  = ""
    forecast_return_pct: float = 0.0
    today_change_pct: float = 0.0

    def summary(self) -> str:
        scan  = " 🔍" if self.from_scan else ""

        # Заголовок с метриками сегодня — теперь показывает EV если доступен
        acc_parts_list = []
        for h in config.FORWARD_BARS:
            if h not in self.today_accuracy:
                continue
            fa = self.today_accuracy[h]
            if fa.total == 0:
                continue
            # Показываем EV если есть, иначе только win%
            if fa.expected_return is not None:
                ev_icon = "▲" if fa.expected_return > 0 else "▼"
                acc_parts_list.append(
                    f"T\\+{h}: {fa.pct:.0f}% {ev_icon}{fa.expected_return:+.2f}%"
                )
            else:
                acc_parts_list.append(f"T\\+{h}: {fa.pct:.0f}%")
        acc_parts = "  ".join(acc_parts_list)

        if self.today_confirmed:
            conf_icon = "✅"
            conf_note = f"Подтверждено: {acc_parts}"
        else:
            conf_icon = "⚠️"
            conf_note = f"Мало данных ({self.today_signals} сигн.) — {acc_parts or 'нет оценки'}"

        # П7: тип активного сигнала
        _mode_label = {
            "trend":          "📈 BUY",
            "strong_trend":   "💪 BUY strong",
            "impulse_speed":  "⚡ Быстрое движение",
            "retest":         "🔄 RETEST",
            "breakout":       "⚡ BREAKOUT",
            "impulse_cross":  "🚀 IMPULSE cross",
            "alignment":      "🌊 ALIGNMENT",
        }.get(self.signal_mode, "📈 BUY")

        # Текущий статус
        if self.signal_now:
            now_line = (
                f"   🟢 *{_mode_label}*  "
                f"`{self.current_price:.6g}`  "
                f"slope:`{self.current_slope:+.2f}%`  "
                f"RSI:`{self.current_rsi:.1f}`  "
                f"ADX:`{self.current_adx:.1f}`  "
                f"vol×:`{self.current_vol_x:.2f}`"
            )
        else:
            _r = (self.no_signal_reason
                  .replace("_",  r"\_")
                  .replace("[",  r"\[")
                  .replace("]",  r"\]")
                  .replace("(",  r"\(")
                  .replace(")",  r"\)"))
            now_line = f"   ⏸ Сигнала нет сейчас: {_r}"

        return (
            f"{conf_icon} {self.symbol}{scan} [{self.tf}]\n"
            f"   {conf_note}\n"
            f"{now_line}"
        )


def _signal_priority(mode: str) -> int:
    return {
        "breakout": 5,
        "retest": 4,
        "impulse_speed": 4,
        "strong_trend": 3,
        "trend": 2,
        "impulse": 2,
        "alignment": 1,
    }.get(mode, 0)


# ── Binance fetch ──────────────────────────────────────────────────────────────

async def fetch_klines(
    session:  aiohttp.ClientSession,
    symbol:   str,
    interval: str,
    limit:    int = config.HISTORY_LIMIT,
) -> Optional[np.ndarray]:
    url = f"{config.BINANCE_REST}/api/v3/klines"
    try:
        async with session.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            r.raise_for_status()
            js = await r.json()
    except Exception:
        return None

    if not isinstance(js, list) or len(js) < 60:
        return None

    arr = np.zeros(len(js), dtype=[
        ("t", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("v", "f8"),
    ])
    arr["t"] = [int(x[0])   for x in js]
    arr["o"] = [float(x[1]) for x in js]
    arr["h"] = [float(x[2]) for x in js]
    arr["l"] = [float(x[3]) for x in js]
    arr["c"] = [float(x[4]) for x in js]
    arr["v"] = [float(x[5]) for x in js]
    return arr


async def fetch_top_symbols(session: aiohttp.ClientSession) -> List[str]:
    url = f"{config.BINANCE_REST}/api/v3/ticker/24hr"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            data = await r.json()
    except Exception:
        return []

    rows: List[Tuple[float, str]] = []
    for it in data:
        sym = it.get("symbol", "")
        if not sym.endswith(config.SCAN_QUOTE):
            continue
        if any(x in sym for x in config.SCAN_EXCLUDE):
            continue
        try:
            qv = float(it.get("quoteVolume", 0))
        except Exception:
            qv = 0.0
        rows.append((qv, sym))

    rows.sort(reverse=True)
    return [s for _, s in rows[:config.SCAN_TOP_N]]


# ── Signal detection ───────────────────────────────────────────────────────────

def _price_edge_pct(price: float, ema20: float) -> float:
    if ema20 <= 0:
        return 0.0
    return max(0.0, ((price / ema20) - 1.0) * 100.0)


def _early_1h_continuation_entry_ok(
    feat: Dict,
    i: int,
    c: np.ndarray,
    *,
    tf: str = "",
    mode: str = "trend",
) -> bool:
    if tf != "1h" or not getattr(config, "EARLY_1H_CONTINUATION_ENTRY_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(config, "EARLY_1H_CONTINUATION_ENTRY_MODES", ("trend", "strong_trend", "impulse_speed"))
    )
    if mode not in allowed_modes:
        return False

    price = float(c[i])
    ef = float(feat["ema_fast"][i])
    es = float(feat["ema_slow"][i])
    e200 = float(feat["ema200"][i])
    slp = float(feat["slope"][i])
    adx = float(feat["adx"][i])
    adx_sma = float(feat["adx_sma"][i]) if np.isfinite(feat["adx_sma"][i]) else np.nan
    rsi = float(feat["rsi"][i])
    vx = float(feat["vol_x"][i])
    dr_pct = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
    macd_h = float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else np.nan
    macd_prev = float(feat["macd_hist"][i - 1]) if i > 0 and np.isfinite(feat["macd_hist"][i - 1]) else macd_h

    if not all(np.isfinite([price, ef, es, e200, slp, adx, rsi, vx, macd_h])):
        return False
    if not (price > ef > es > e200):
        return False
    if slp < float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_SLOPE_MIN", 0.08)):
        return False
    if adx < float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_ADX_MIN", 24.0)):
        return False
    if np.isfinite(adx_sma):
        tol = float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_ADX_SMA_TOLERANCE", 3.0))
        if adx + tol < adx_sma:
            return False
    rsi_min = float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_RSI_MIN", 60.0))
    rsi_max = float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_RSI_MAX", 78.0))
    if not (rsi_min <= rsi <= rsi_max):
        return False
    if vx < float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_VOL_X_MIN", 1.20)):
        return False
    if dr_pct > float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_RANGE_MAX", 6.5)):
        return False
    if _price_edge_pct(price, ef) > float(getattr(config, "EARLY_1H_CONTINUATION_ENTRY_PRICE_EDGE_MAX_PCT", 1.60)):
        return False
    macd_min_abs = price * float(getattr(config, "TREND_MACD_REL_MIN", 0.00005))
    if macd_h < macd_min_abs:
        return False
    if np.isfinite(macd_prev) and macd_h < macd_prev:
        return False
    return True


def _early_15m_continuation_entry_ok(
    feat: Dict,
    i: int,
    c: np.ndarray,
    *,
    tf: str = "",
    mode: str = "trend",
) -> bool:
    if tf != "15m" or not getattr(config, "EARLY_15M_CONTINUATION_ENTRY_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(config, "EARLY_15M_CONTINUATION_ENTRY_MODES", ("trend",))
    )
    if mode not in allowed_modes:
        return False

    price = float(c[i])
    ef = float(feat["ema_fast"][i])
    es = float(feat["ema_slow"][i])
    e200 = float(feat["ema200"][i])
    slp = float(feat["slope"][i])
    adx = float(feat["adx"][i])
    rsi = float(feat["rsi"][i])
    vx = float(feat["vol_x"][i])
    dr_pct = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
    macd_h = float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else np.nan
    macd_prev = float(feat["macd_hist"][i - 1]) if i > 0 and np.isfinite(feat["macd_hist"][i - 1]) else macd_h

    if not all(np.isfinite([price, ef, es, e200, slp, adx, rsi, vx, macd_h])):
        return False
    if not (price > ef > es > e200):
        return False
    if slp < float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_SLOPE_MIN", 0.10)):
        return False
    if adx < float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_ADX_MIN", 20.0)):
        return False
    rsi_min = float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_RSI_MIN", 60.0))
    rsi_max = float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_RSI_MAX", 76.0))
    if not (rsi_min <= rsi <= rsi_max):
        return False
    if vx < float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_VOL_X_MIN", 1.05)):
        return False
    if dr_pct > float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_RANGE_MAX", 8.0)):
        return False
    if _price_edge_pct(price, ef) > float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_PRICE_EDGE_MAX_PCT", 1.60)):
        return False
    macd_min_abs = price * float(getattr(config, "TREND_MACD_REL_MIN", 0.00005))
    if macd_h < macd_min_abs:
        return False
    if np.isfinite(macd_prev) and macd_h < macd_prev:
        return False
    return True


def get_effective_entry_mode(
    feat: Dict,
    i: int,
    c: np.ndarray,
    *,
    tf: str = "",
) -> tuple[str, bool]:
    mode = get_entry_mode(feat, i)
    early_15m_continuation = _early_15m_continuation_entry_ok(feat, i, c, tf=tf, mode=mode)
    if early_15m_continuation and tf == "15m" and mode == "trend":
        return "alignment", True
    return mode, False


def check_entry_conditions(
    feat: Dict, i: int, c: np.ndarray,
    regime: "MarketRegime" = None,
    tf: str = "",
) -> Tuple[bool, str]:
    """
    Проверяет все условия входа.
    v2: адаптивные пороги через MarketRegime + slope acceleration + squeeze bypass.
    """
    if regime is None:
        regime = _get_coin_regime(feat, i)

    ef  = feat["ema_fast"][i]
    es  = feat["ema_slow"][i]
    rsi = feat["rsi"][i]
    adx = feat["adx"][i]
    slp = feat["slope"][i]
    vx  = feat["vol_x"][i]

    if not all(np.isfinite([ef, es, rsi, adx, slp, vx])):
        return False, "нет данных индикаторов"

    # Медвежий режим: запрет новых BUY — проверяем в первую очередь
    if not regime.allow_new_buy:
        return False, f"режим {regime.name} — новые BUY запрещены"

    if not (float(c[i]) > ef > es):
        return False, f"цена {float(c[i]):.6g} не выше EMA20 {ef:.6g} > EMA50 {es:.6g}"

    # Slope: режимный порог, снижается при наличии slope acceleration
    slope_min = regime.slope_min
    accel_arr = feat.get("slope_accel")
    has_accel = (
        accel_arr is not None and i < len(accel_arr) and
        np.isfinite(accel_arr[i]) and
        accel_arr[i] >= getattr(config, "SLOPE_ACCEL_MIN", 0.05)
    )
    if has_accel:
        slope_min = slope_min * 0.7  # slope acceleration снижает порог на 30%
    early_1h_override = _early_1h_continuation_entry_ok(feat, i, c, tf=tf, mode="trend")
    if slp < slope_min and not early_1h_override:
        return False, f"наклон EMA20 {slp:+.2f}% < {slope_min:.2f}%"

    # ADX: режимный порог, снижается при squeeze breakout
    adx_min = regime.adx_min
    sq_arr = feat.get("squeeze_breakout")
    squeeze_active = (
        sq_arr is not None and i < len(sq_arr) and sq_arr[i] == 1.0
    )
    if squeeze_active:
        adx_min = max(adx_min * 0.7, 15.0)
    if adx < adx_min:
        return False, f"ADX {adx:.1f} < {adx_min:.1f}"

    adx_sma = feat["adx_sma"][i]
    if np.isfinite(adx_sma) and adx <= adx_sma:
        bypass_threshold = getattr(config, "ADX_SMA_BYPASS", 35.0)
        if adx < bypass_threshold and not squeeze_active and not early_1h_override:
            return False, f"ADX {adx:.1f} ≤ SMA(ADX,10) {adx_sma:.1f} — тренд слабый"

    # Volume: режимный порог, снижается при squeeze
    vol_min = regime.vol_mult
    if squeeze_active:
        vol_min = vol_min * 0.85
    if vx < vol_min:
        return False, f"объём {vx:.2f}× < {vol_min:.2f}×"

    # RSI: адаптивная верхняя граница (режим + strong_trend + squeeze)
    strong_trend = (
        np.isfinite(adx) and adx >= config.STRONG_ADX_MIN and
        np.isfinite(vx)  and vx  >= config.STRONG_VOL_MIN
    )
    rsi_hi = config.RSI_BUY_HI_STRONG if strong_trend else regime.rsi_hi
    if squeeze_active:
        rsi_hi = min(rsi_hi + 3, 85.0)
    if not (config.RSI_BUY_LO <= rsi <= rsi_hi):
        mode = " [сильный тренд]" if strong_trend else ""
        sq_note = " [squeeze]" if squeeze_active else ""
        return False, f"RSI {rsi:.1f} вне зоны [{config.RSI_BUY_LO}-{rsi_hi:.0f}]{mode}{sq_note}"

    macd_h = feat["macd_hist"][i]
    macd_prev = feat["macd_hist"][i - 1] if i > 0 else np.nan
    if np.isfinite(macd_h) and macd_h < 0:
        return False, f"MACD гистограмма отрицательная ({macd_h:.6g})"
    macd_min_abs = float(c[i]) * getattr(config, "TREND_MACD_REL_MIN", 0.00005)
    if np.isfinite(macd_h) and macd_h < macd_min_abs:
        return False, f"MACD гистограмма слишком слабая ({macd_h:.6g} < {macd_min_abs:.6g})"
    if np.isfinite(macd_h) and np.isfinite(macd_prev) and macd_h < macd_prev:
        return False, f"MACD гистограмма ослабевает ({macd_h:.6g} < {macd_prev:.6g})"

    # Range max: динамический + режим
    _range_max = _get_effective_range_max(feat, i, regime)
    dr_pct = feat["daily_range_pct"][i]
    if np.isfinite(dr_pct) and dr_pct > _range_max:
        return False, f"монета уже +{dr_pct:.1f}% от дна дня (> {_range_max:.1f}%)"

    if not regime.allow_new_buy:
        return False, f"режим {regime.name} — новые BUY запрещены"

    return True, ""


def check_setup_conditions(
    feat: Dict, i: int, c: np.ndarray,
    regime: "MarketRegime" = None,
) -> Tuple[bool, str, int]:
    """
    Мягкая проверка "почти готового сигнала".
    Используется для UI и раннего мониторинга, когда до полноценного BUY
    не хватает 1-2 подтверждений. Возвращает (ok, reason, missing_count).
    """
    if regime is None:
        regime = _get_coin_regime(feat, i)

    ef = feat["ema_fast"][i]
    es = feat["ema_slow"][i]
    rsi = feat["rsi"][i]
    slp = feat["slope"][i]
    vx = feat["vol_x"][i]
    macd_h = feat["macd_hist"][i]
    dr_pct = feat["daily_range_pct"][i]
    adx = feat["adx"][i]

    if not all(np.isfinite([ef, rsi, slp])):
        return False, "недостаточно данных", 99

    price = float(c[i])
    if not regime.allow_new_buy:
        return False, f"режим {regime.name} — setup BUY запрещён", 99

    accel_arr = feat.get("slope_accel")
    has_accel = (
        accel_arr is not None and i < len(accel_arr) and
        np.isfinite(accel_arr[i]) and
        accel_arr[i] >= getattr(config, "SLOPE_ACCEL_MIN", 0.05)
    )
    slope_min = regime.slope_min * (0.7 if has_accel else 1.0)

    sq_arr = feat.get("squeeze_breakout")
    squeeze_active = (
        sq_arr is not None and i < len(sq_arr) and sq_arr[i] == 1.0
    )
    adx_min = regime.adx_min
    if squeeze_active:
        adx_min = max(adx_min * 0.7, 15.0)
    vol_min = regime.vol_mult * (0.85 if squeeze_active else 1.0)

    strong_trend = (
        np.isfinite(adx) and adx >= config.STRONG_ADX_MIN and
        np.isfinite(vx) and vx >= config.STRONG_VOL_MIN
    )
    rsi_hi = config.RSI_BUY_HI_STRONG if strong_trend else regime.rsi_hi
    if squeeze_active:
        rsi_hi = min(rsi_hi + 3, 85.0)
    range_max = _get_effective_range_max(feat, i, regime)

    if not (price > ef):
        return False, f"цена {price:.6g} ниже EMA20 {ef:.6g}", 99
    if slp <= 0:
        return False, f"EMA20 не растёт (slope {slp:+.2f}%)", 99
    if np.isfinite(rsi) and rsi < config.RSI_BUY_LO:
        return False, f"RSI {rsi:.1f} < {config.RSI_BUY_LO:.0f}", 99
    if np.isfinite(macd_h) and macd_h < 0:
        return False, f"MACD отрицательный ({macd_h:.6g})", 99
    if np.isfinite(dr_pct) and dr_pct > range_max:
        return False, f"рост от дна +{dr_pct:.1f}% уже поздний (> {range_max:.1f}%)", 99
    if np.isfinite(adx) and adx < 15:
        return False, f"ADX {adx:.1f} < 15 — рынок слишком слабый", 99
    if np.isfinite(vx) and vx < 0.8:
        return False, f"объём {vx:.2f}× — нет участия покупателей", 99

    missing = []
    if np.isfinite(es) and ef <= es:
        missing.append(f"EMA20 {ef:.4g} <= EMA50 {es:.4g}")
    if slp < slope_min:
        missing.append(f"slope {slp:+.2f}% < {slope_min:.2f}%")
    if np.isfinite(vx) and vx < vol_min:
        missing.append(f"vol× {vx:.2f} < {vol_min:.2f}")
    if np.isfinite(adx) and adx < adx_min:
        missing.append(f"ADX {adx:.1f} < {adx_min:.1f}")
    if np.isfinite(rsi) and not (config.RSI_BUY_LO <= rsi <= rsi_hi):
        missing.append(f"RSI {rsi:.1f} вне [{config.RSI_BUY_LO}-{rsi_hi:.0f}]")

    reason = "Почти готово: " + ", ".join(missing) if missing else "готовый BUY"
    return True, reason, len(missing)


def _negative_slope_confirmed(feat: Dict, i: int, bars: int) -> bool:
    if bars <= 1:
        return bool(np.isfinite(feat["slope"][i]) and feat["slope"][i] < 0)
    if i < bars - 1:
        return False
    for j in range(i - bars + 1, i + 1):
        if not np.isfinite(feat["slope"][j]) or feat["slope"][j] >= 0:
            return False
    return True


def check_exit_conditions(
    feat: Dict,
    i: int,
    c: np.ndarray,
    *,
    mode: str = "trend",
    bars_elapsed: int = 0,
    tf: str = "15m",
) -> Optional[str]:
    """
    Правила выхода из позиции.
    WEAK-причины не закрывают сделку напрямую на уровне ATR, а маркируют
    раннее ослабление импульса для ужесточения сопровождения.
    """
    close = float(c[i])
    ef = feat["ema_fast"][i]
    rsi = feat["rsi"][i]
    adx = feat["adx"][i]
    slp = feat["slope"][i]

    mode_aware = bool(getattr(config, "MODE_AWARE_EXITS_ENABLED", True))
    patient_modes = set(getattr(config, "EXIT_PATIENT_MODES", ("trend", "strong_trend", "alignment")))
    semi_patient_modes = set(getattr(config, "EXIT_SEMI_PATIENT_MODES", ("impulse_speed",)))
    aggressive_modes = set(getattr(config, "EXIT_AGGRESSIVE_MODES", ("breakout", "retest")))

    below_ema20 = bool(np.isfinite(ef) and close < ef)
    two_closes_below = False
    if i >= 1 and np.isfinite(ef):
        prev_ef = feat["ema_fast"][i - 1]
        prev_close = float(c[i - 1])
        two_closes_below = bool(np.isfinite(prev_ef) and prev_close < prev_ef and close < ef)

    j = i - config.ADX_GROW_BARS
    adx_weak = False
    if j >= 0 and np.isfinite(adx) and np.isfinite(feat["adx"][j]):
        if adx < feat["adx"][j] * config.ADX_DROP_RATIO:
            adx_weak = True

    adx_exit_allowed = True
    if mode_aware and mode in patient_modes:
        adx_exit_allowed = bars_elapsed >= int(getattr(config, "MIN_BARS_BEFORE_ADX_EXIT", 5))
    elif mode_aware and mode in semi_patient_modes:
        adx_exit_allowed = bars_elapsed >= int(
            getattr(config, "SEMI_PATIENT_MIN_BARS_BEFORE_ADX_EXIT", 4)
        )
    adx_weak_effective = adx_weak and adx_exit_allowed

    slope_confirm_bars = 1
    if mode_aware and mode in patient_modes:
        slope_confirm_bars = int(getattr(config, "PATIENT_SLOPE_CONFIRM_BARS", 2))
    elif mode_aware and mode in semi_patient_modes:
        slope_confirm_bars = int(getattr(config, "SEMI_PATIENT_SLOPE_CONFIRM_BARS", 2))
    negative_slope = _negative_slope_confirmed(feat, i, slope_confirm_bars)

    if two_closes_below:
        if not mode_aware or mode in aggressive_modes or negative_slope or adx_weak_effective:
            return f"2 закрытия подряд ниже EMA20 ({ef:.6g}) — ранний разворот"

    if below_ema20:
        if not mode_aware:
            if (np.isfinite(slp) and slp < 0) or adx_weak:
                return f"цена ниже EMA20 ({ef:.6g}) + подтверждённая слабость"
        elif mode in patient_modes or mode in semi_patient_modes:
            if negative_slope or adx_weak_effective:
                return f"цена ниже EMA20 ({ef:.6g}) + подтверждённая слабость"
        else:
            if negative_slope or adx_weak_effective:
                return f"цена ниже EMA20 ({ef:.6g}) + подтверждённая слабость"

    if np.isfinite(rsi) and rsi > config.RSI_OVERBOUGHT:
        return f"RSI перекуплен ({rsi:.1f})"

    if not mode_aware:
        if np.isfinite(slp) and slp < 0:
            return f"EMA20 разворачивается вниз (slope {slp:+.2f}%)"
        if adx_weak:
            return f"ADX ослабевает ({adx:.1f} vs {feat['adx'][j]:.1f})"
    elif mode in patient_modes or mode in semi_patient_modes:
        if negative_slope:
            return f"EMA20 разворачивается вниз (slope {slp:+.2f}%)"
        if adx_weak_effective:
            return f"ADX ослабевает ({adx:.1f} vs {feat['adx'][j]:.1f})"
    else:
        if negative_slope:
            return f"EMA20 разворачивается вниз (slope {slp:+.2f}%)"
        if adx_weak_effective:
            return f"ADX ослабевает ({adx:.1f} vs {feat['adx'][j]:.1f})"

    rsi_div_arr = feat.get("rsi_divergence")
    if rsi_div_arr is not None and i < len(rsi_div_arr) and rsi_div_arr[i] == 1.0:
        return "⚠️ WEAK: RSI дивергенция — momentum ослабевает (стоп ужесточён)"

    vol_ex_arr = feat.get("vol_exhaustion")
    if vol_ex_arr is not None and i < len(vol_ex_arr) and vol_ex_arr[i] == 1.0:
        return "⚠️ WEAK: объёмное истощение — покупатели заканчиваются"

    fan_arr = feat.get("ema_fan_spread")
    fan_threshold = getattr(config, "EMA_FAN_DECAY_THRESHOLD", 0.30)
    if fan_arr is not None and i < len(fan_arr) and np.isfinite(fan_arr[i]):
        if float(fan_arr[i]) >= fan_threshold:
            decay_pct = float(fan_arr[i]) * 100
            return f"⚠️ WEAK: EMA-веер сжался на {decay_pct:.0f}% — тренд теряет ширину"

    return None


# ── Entry mode helper ──────────────────────────────────────────────────────────

def get_entry_mode(feat: Dict, i: int) -> str:
    """
    П1: режим входа — выбирает ATR_TRAIL_K и MAX_HOLD_BARS.
      'strong_trend' → ATR_TRAIL_K_STRONG (шире, держим дольше)
      'trend'        → ATR_TRAIL_K (стандартный)

    Критерии strong_trend (все три обязательны):
      1. ADX ≥ STRONG_ADX_MIN (28) + Vol× ≥ STRONG_VOL_MIN (2.0)
      2. EMA20 > EMA50 — ценовая структура бычья, не флэт
         (SNX-баг: ADX=29.9 на флэте где EMA20 ≈ EMA50)
      3. EMA50 наклонена вверх — не горизонтальный канал

    Если только ADX+Vol без п.2-3 → режим 'trend' (стандартный стоп).
    """
    adx = feat["adx"][i]
    vx  = feat["vol_x"][i]
    ri  = float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0

    # Скорость за 3 бара из feat["close"]
    price_speed = 0.0
    c_arr = feat.get("close")
    if c_arr is not None and i >= 3:
        c0 = float(c_arr[i - 3])
        ci = float(c_arr[i])
        if c0 > 0:
            price_speed = (ci - c0) / c0 * 100.0

    # ADX + vol ≥ порогов → кандидат в сильный тренд
    if (np.isfinite(adx) and adx >= config.STRONG_ADX_MIN
            and np.isfinite(vx) and vx >= config.STRONG_VOL_MIN):

        ef = feat["ema_fast"][i]   # EMA20
        es = feat["ema_slow"][i]   # EMA50

        # П.2: EMA20 должна быть выше EMA50 — структура бычья
        # STRONG_EMA_SEP_MIN: минимальный разрыв EMA20/EMA50 в % от цены.
        # При флэте (SNX): EMA20=0.3156, EMA50=0.3130 → разрыв 0.08% — почти ноль.
        # При реальном тренде (AR): EMA20=1.70, EMA50=1.65 → разрыв 3%.
        ema_sep_min = getattr(config, "STRONG_EMA_SEP_MIN", 0.3)  # 0.3% от цены
        ema_sep_ok = False
        if np.isfinite(ef) and np.isfinite(es) and es > 0:
            ema_sep_pct = (ef - es) / es * 100.0
            ema_sep_ok = ema_sep_pct >= ema_sep_min

        close_edge_pct = 0.0
        if np.isfinite(ef) and ef > 0 and np.isfinite(ci):
            close_edge_pct = (ci / ef - 1.0) * 100.0

        # П.3: EMA50 наклонена вверх — не горизонтальный канал
        # Используем те же 3 бара что и slope для price_speed
        ema50_rising = False
        if i >= 3 and np.isfinite(feat["ema_slow"][i]) and np.isfinite(feat["ema_slow"][i - 3]):
            es_prev = float(feat["ema_slow"][i - 3])
            if es_prev > 0:
                ema50_slope = (float(feat["ema_slow"][i]) - es_prev) / es_prev * 100.0
                ema50_min_slope = getattr(config, "STRONG_EMA50_SLOPE_MIN", 0.05)
                ema50_rising = ema50_slope >= ema50_min_slope

        strong_rsi_min = getattr(config, "STRONG_RSI_MIN", 55.0)
        strong_close_max = getattr(config, "STRONG_CLOSE_EMA20_MAX_PCT", 1.8)
        if ema_sep_ok and ema50_rising and ri >= strong_rsi_min and close_edge_pct <= strong_close_max:
            return "strong_trend"
        # ADX высокий, но структура не подтверждает → обычный тренд
        return "trend"

    # Быстрое ценовое движение при слабом ADX (ADX лагует ~10 баров после импульса).
    # Стоп такой же широкий как у strong_trend, но метка честная — не «сильный тренд».
    if price_speed >= 1.5:
        return "impulse_speed"
    return "trend"






# ── П5: Bull Day detector ───────────────────────────────────────────────────────

async def is_bull_day(session: aiohttp.ClientSession) -> tuple:
    """
    П5: бычий день = BTC выше EMA50 на 1h и EMA50 наклонена вверх.
    Возвращает (bool, btc_price, btc_ema50) для логирования.
    """
    try:
        data = await fetch_klines(session, "BTCUSDT", "1h", limit=60)
        if data is None or len(data) < 55:
            return False, 0.0, 0.0

        c_btc = data["c"].astype(float)
        ema50 = _ema(c_btc, 50)
        confirm_bars = max(1, int(getattr(config, "BULL_DAY_CONFIRM_BARS", 2)))
        if len(c_btc) < 50 + confirm_bars + 5:
            return False, 0.0, 0.0

        i = len(c_btc) - 2
        slope_ref = i - 5
        if slope_ref >= 0 and ema50[slope_ref] > 0:
            slope = (ema50[i] - ema50[slope_ref]) / ema50[slope_ref] * 100
        else:
            slope = 0.0

        start = i - confirm_bars + 1
        closes = c_btc[start:i + 1]
        ema_slice = ema50[start:i + 1]
        if len(closes) < confirm_bars or not np.all(np.isfinite(ema_slice)):
            return False, 0.0, 0.0

        enter_band = 1.0 + getattr(config, "BULL_DAY_ENTER_PCT", 0.20) / 100.0
        exit_band = 1.0 - getattr(config, "BULL_DAY_EXIT_PCT", 0.20) / 100.0
        enter_confirmed = bool(np.all(closes > ema_slice * enter_band) and slope > 0)
        exit_confirmed = bool(np.all(closes < ema_slice * exit_band))

        prev_state = bool(getattr(config, "_bull_day_active", False))
        is_bull = prev_state
        if prev_state:
            if exit_confirmed:
                is_bull = False
        elif enter_confirmed:
            is_bull = True

        btc_price = float(c_btc[i])
        btc_ema50 = float(ema50[i])
        return is_bull, btc_price, btc_ema50
    except Exception:
        return False, 0.0, 0.0


async def detect_market_regime(session: aiohttp.ClientSession) -> MarketRegime:
    """
    Определяет глобальный режим рынка по BTC 1h.
    Использует данные за последние 60 часов для расчёта ADX и EMA.

    Возвращает MarketRegime с именем режима и адаптивными параметрами.
    """
    try:
        data = await fetch_klines(session, "BTCUSDT", "1h", limit=100)
        if data is None or len(data) < 60:
            return MarketRegime("neutral")

        from indicators import (
            _ema as ema_fn, _adx as adx_fn, _ema_slope as slope_fn
        )

        c = data["c"].astype(float)
        h = data["h"].astype(float)
        l = data["l"].astype(float)

        ema20  = ema_fn(c, 20)
        ema50  = ema_fn(c, 50)
        adx_v  = adx_fn(h, l, c, 14)
        slope  = slope_fn(ema20, 5)

        i = len(c) - 2  # последний закрытый бар
        cur_c   = float(c[i])
        cur_ef  = float(ema20[i])
        cur_es  = float(ema50[i])
        cur_adx = float(adx_v[i]) if np.isfinite(adx_v[i]) else 0.0
        cur_slp = float(slope[i]) if np.isfinite(slope[i]) else 0.0

        adx_trend = getattr(config, "REGIME_BTC_ADX_TREND", 22.0)
        adx_flat  = getattr(config, "REGIME_BTC_ADX_FLAT",  18.0)

        # Проверяем был ли BTC под EMA50 предыдущий бар (recovery детектор)
        prev_c  = float(c[i - 1]) if i >= 1 else cur_c
        prev_es = float(ema50[i - 1]) if i >= 1 else cur_es
        was_below = prev_c < prev_es

        if cur_adx >= adx_trend:
            if cur_c > cur_ef > cur_es:
                name = "recovery" if was_below else "bull_trend"
            elif cur_c < cur_es:
                name = "bear_trend"
            else:
                name = "neutral"
        elif cur_adx < adx_flat:
            name = "consolidation"
        else:
            if cur_c > cur_ef and cur_slp > 0:
                name = "bull_trend"
            elif cur_c < cur_es:
                name = "bear_trend"
            else:
                name = "consolidation"

        return MarketRegime(name)

    except Exception:
        return MarketRegime("neutral")


# ── RETEST: откат к EMA20 в существующем тренде ────────────────────────────────

def check_retest_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    RETEST — вход не в начало тренда а в его продолжение после отката к EMA20.
    Условия:
      1. Тренд существовал RETEST_LOOKBACK баров назад (close > EMA20)
      2. В последние RETEST_TOUCH_BARS баров: low касалось EMA20 (откат)
      3. Текущий бар: close > EMA20 и close > prev close (отскок подтверждён)
      4. EMA20 slope > 0 (тренд не сломан)
      5. RSI < RETEST_RSI_MAX (не перегрет)
      6. ADX > 20 (тренд существует)
      7. vol_x > RETEST_VOL_MIN (объём необязательно высокий)
    """
    lb      = getattr(config, "RETEST_LOOKBACK",   12)
    tb      = getattr(config, "RETEST_TOUCH_BARS",  5)
    rsi_mx  = getattr(config, "RETEST_RSI_MAX",    65.0)
    vol_mn  = getattr(config, "RETEST_VOL_MIN",     0.8)

    if i < lb + 2:
        return False, "недостаточно баров"

    c_arr    = feat.get("close")
    e200_arr = feat.get("ema200")
    e200_arr = feat.get("ema200")
    lo       = feat.get("low")
    ema_fast = feat["ema_fast"]
    slope    = feat["slope"]
    rsi      = feat["rsi"]
    adx      = feat["adx"]
    vol_x    = feat["vol_x"]

    if c_arr is None or lo is None:
        return False, "нет ценовых рядов в feat"

    # 1. Тренд существовал lb баров назад
    lb_idx = i - lb
    if not (np.isfinite(float(c_arr[lb_idx])) and np.isfinite(float(ema_fast[lb_idx]))):
        return False, "нет данных для проверки тренда"
    if float(c_arr[lb_idx]) <= float(ema_fast[lb_idx]):
        return False, f"тренда не было {lb} баров назад (close ≤ EMA20)"

    # 2. Откат: low касалось EMA20 в последние tb баров (исключая текущий)
    touched = False
    for k in range(1, tb + 1):
        ki = i - k
        if ki < 0:
            break
        if not (np.isfinite(float(lo[ki])) and np.isfinite(float(ema_fast[ki]))):
            continue
        # Касание: low <= EMA20 × 1.005 (допуск 0.5%)
        if float(lo[ki]) <= float(ema_fast[ki]) * 1.005:
            touched = True
            break
    if not touched:
        return False, f"нет касания EMA20 за {tb} баров"

    # 3. Отскок: текущий бар выше EMA20 и выше предыдущего закрытия
    if not (np.isfinite(float(c_arr[i])) and np.isfinite(float(ema_fast[i]))):
        return False, "нет данных текущего бара"
    if float(c_arr[i]) <= float(ema_fast[i]):
        return False, f"close {float(c_arr[i]):.6g} ≤ EMA20 {float(ema_fast[i]):.6g}"
    if float(c_arr[i]) <= float(c_arr[i - 1]):
        return False, "нет отскока (close не выше предыдущего бара)"

    # 3а. Минимальный отскок от EMA20 — "цена на линии" не считается ретестом.
    # LTC-баг (12.03.2026): close=54.97, EMA20=54.9694 → зазор 0.001%, сигнал не имел смысла.
    # При реальном отскоке цена уходит хотя бы на RETEST_MIN_BOUNCE_PCT выше EMA20.
    bounce_min = getattr(config, "RETEST_MIN_BOUNCE_PCT", 0.05)
    _bounce_pct = (float(c_arr[i]) / float(ema_fast[i]) - 1.0) * 100.0
    if _bounce_pct < bounce_min:
        return False, f"отскок {_bounce_pct:.3f}% < {bounce_min}% (цена слишком близко к EMA20)"

    # 4. Slope EMA20 > 0 — но фактически нужен минимальный рост.
    # LTC 12.03.2026: slope=+0.08% — EMA20 почти горизонтальна.
    # Ретест на горизонтальной EMA — не тренд, это флэт с отскоком.
    retest_slope_min = getattr(config, "RETEST_SLOPE_MIN", 0.1)
    if not np.isfinite(float(slope[i])) or float(slope[i]) < retest_slope_min:
        return False, f"slope EMA20 {float(slope[i]):.2f}% < {retest_slope_min}% (EMA≈горизонталь)"

    # 4b. MACD hist >= 0 — momentum не должен быть отрицательным при ретесте.
    # LTC 12.03.2026: MACD hist=-0.02 — покупатели ещё не вернулись после отката.
    mh_arr = feat.get("macd_hist")
    if mh_arr is not None and np.isfinite(float(mh_arr[i])):
        if float(mh_arr[i]) < 0:
            return False, f"MACD hist {float(mh_arr[i]):.6g} < 0 (импульс ещё не вернулся)"

    # 5. RSI не перегрет
    if not np.isfinite(float(rsi[i])) or float(rsi[i]) >= rsi_mx:
        return False, f"RSI {float(rsi[i]):.1f} ≥ {rsi_mx} (перегрет для ретеста)"

    # 6. ADX подтверждает существование тренда
    if not np.isfinite(float(adx[i])) or float(adx[i]) < 20:
        return False, f"ADX {float(adx[i]):.1f} < 20"

    # 7. Минимальный объём
    if not np.isfinite(float(vol_x[i])) or float(vol_x[i]) < vol_mn:
        return False, f"vol× {float(vol_x[i]):.2f} < {vol_mn}"

    return True, ""


# ── BREAKOUT: пробой флэта с объёмом ───────────────────────────────────────────

def check_breakout_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    BREAKOUT — пробой диапазона флэта с высоким объёмом.
    Условия:
      1. Последние BREAKOUT_FLAT_BARS баров: диапазон (max-min)/min < BREAKOUT_FLAT_MAX_PCT
      2. Текущий close > max(high флэта) — реальный пробой уровня
      3. vol_x ≥ BREAKOUT_VOL_MIN (сильный объём подтверждает пробой)
      4. MACD_hist > 0 и вырос vs предыдущий бар (импульс)
      5. daily_range < BREAKOUT_RANGE_MAX (движение только началось, не поздно)
      6. close > EMA20 (минимальная бычья структура)
    Примечание: RSI-проверка намеренно убрана. После прорыва из флэта
    +2-3% за 1 бар RSI=80-85 это НОРМА (не перегрев). RSI-условие
    блокировало именно те входы что нужно ловить (ZRO 15.03.2026).
    Остальные 6 условий (флэт, vol_x, daily_range, MACD, пробой, EMA20)
    достаточно защищают. Бэктест 11/11 подтвердил безопасность.
    """
    flat_bars = getattr(config, "BREAKOUT_FLAT_BARS",    8)
    flat_pct  = getattr(config, "BREAKOUT_FLAT_MAX_PCT", 2.0)
    vol_mn    = getattr(config, "BREAKOUT_VOL_MIN",      2.0)
    rng_mx    = getattr(config, "BREAKOUT_RANGE_MAX",    4.0)

    if i < flat_bars + 2:
        return False, "недостаточно баров"

    c_arr      = feat.get("close")
    hi         = feat.get("high")
    lo         = feat.get("low")
    ema_fast   = feat["ema_fast"]
    slope      = feat["slope"]
    vol_x      = feat["vol_x"]
    rsi        = feat["rsi"]
    macd_hist  = feat["macd_hist"]
    daily_rng  = feat["daily_range_pct"]
    adx        = feat["adx"]

    if c_arr is None or hi is None or lo is None:
        return False, "нет ценовых рядов в feat"

    # 1. Флэт: диапазон последних flat_bars баров (не включая текущий)
    flat_hi = float(np.max(hi[i - flat_bars: i]))
    flat_lo = float(np.min(lo[i - flat_bars: i]))
    if flat_lo <= 0:
        return False, "некорректные данные low"
    flat_rng = (flat_hi - flat_lo) / flat_lo * 100
    if flat_rng > flat_pct:
        return False, f"нет флэта: диапазон {flat_rng:.1f}% > {flat_pct}%"

    # 2. Пробой: текущий close выше максимума флэта
    ci = float(c_arr[i])
    if ci <= flat_hi:
        return False, f"нет пробоя: close {ci:.6g} ≤ max флэта {flat_hi:.6g}"

    # 3. Объём
    vx = float(vol_x[i])
    if not np.isfinite(vx) or vx < vol_mn:
        return False, f"vol× {vx:.2f} < {vol_mn} (нужен сильный объём на пробое)"

    # 4. MACD растёт
    mh = float(macd_hist[i])
    mh_prev = float(macd_hist[i - 1]) if i > 0 else float("nan")
    if not (np.isfinite(mh) and mh > 0 and np.isfinite(mh_prev) and mh > mh_prev):
        return False, "MACD_hist ≤ 0 или не растёт"

    # 5. Движение только началось
    dr = float(daily_rng[i])
    if np.isfinite(dr) and dr > rng_mx:
        return False, f"daily_range {dr:.1f}% > {rng_mx}% (поздно входить)"

    # 6. Структура (RSI не проверяем: после спайка RSI=80-85 — это норма)
    ef = float(ema_fast[i])
    if not np.isfinite(ef) or ci <= ef:
        return False, f"close {ci:.6g} ≤ EMA20 {ef:.6g}"

    slope_min = getattr(config, "BREAKOUT_SLOPE_MIN", 0.08)
    slp = float(slope[i])
    if not np.isfinite(slp) or slp < slope_min:
        return False, f"slope {slp:+.2f}% < {slope_min:.2f}%"

    rsi_max = getattr(config, "BREAKOUT_RSI_MAX", 84.0)
    ri = float(rsi[i])
    if np.isfinite(ri) and ri > rsi_max:
        return False, f"RSI {ri:.1f} > {rsi_max:.0f} (слишком поздний breakout)"

    adx_min = getattr(config, "BREAKOUT_ADX_MIN", 18.0)
    ax = float(adx[i])
    if not np.isfinite(ax) or ax < adx_min:
        return False, f"ADX {ax:.1f} < {adx_min:.1f} (пробой без трендовой опоры)"

    return True, ""


def check_impulse_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    IMPULSE — детектор самого начала тренда, на 1-2 бара раньше BUY/STRONG_TREND.

    ADX лагует 10+ баров и не успевает подтвердить быстрый импульс.
    IMPULSE обходит ADX, вместо этого смотрит на скорость ценового движения.

    Откалиброван по данным 04.03.2026:
      ETH 15:15 — r1=+2.37% r3=+3.50% RSI=78.8 vol×2.63 → поймал за 1 бар до BUY
      SOL 15:15 — r1=+2.11% r3=+3.27% RSI=76.4 vol×2.35
      XRP 15:15 — r1=+2.05% r3=+2.69% RSI=79.5 vol×3.77

    Условия:
      1. close > EMA20 (бычья структура)
      2. EMA20 slope > 0 (направление правильное)
      3. r1 >= IMPULSE_R1_MIN (текущий бар вырос %)
      4. r3 >= IMPULSE_R3_MIN (за 3 бара выросло %)
      5. body >= IMPULSE_BODY_MIN (реальное тело, не wick)
      6. vol_x >= IMPULSE_VOL_MIN (объём подтверждает)
      7. RSI в [IMPULSE_RSI_LO, IMPULSE_RSI_HI]
      8. daily_range <= effective_range_max (не слишком поздно)
    """
    if i < 5:
        return False, "недостаточно баров"

    c_arr     = feat.get("close")
    o_arr     = feat.get("open")
    ema_fast  = feat["ema_fast"]
    slope     = feat["slope"]
    rsi_arr   = feat["rsi"]
    vol_x_arr = feat["vol_x"]
    dr_arr    = feat["daily_range_pct"]

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    ef  = float(ema_fast[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x_arr[i])
    dr  = float(dr_arr[i])

    if not np.isfinite(ci) or not np.isfinite(ef):
        return False, "нет данных"

    # 1. Цена выше EMA20
    if ci <= ef:
        return False, f"close {ci:.4g} ≤ EMA20 {ef:.4g}"

    # 2. EMA20 растёт
    if not np.isfinite(slp) or slp <= 0:
        return False, f"slope EMA20 {slp:+.3f}% ≤ 0"

    # 3. r1 — рост текущего бара
    prev = float(c_arr[i - 1])
    if prev <= 0:
        return False, "нет предыдущего закрытия"
    r1 = (ci / prev - 1) * 100
    if r1 < config.IMPULSE_R1_MIN:
        return False, f"r1 {r1:+.2f}% < {config.IMPULSE_R1_MIN}%"

    # 4. r3 — рост за 3 бара
    if i < 3:
        return False, "мало баров для r3"
    base3 = float(c_arr[i - 3])
    if base3 <= 0:
        return False, "нет базы для r3"
    r3 = (ci / base3 - 1) * 100
    if r3 < config.IMPULSE_R3_MIN:
        return False, f"r3 {r3:+.2f}% < {config.IMPULSE_R3_MIN}% (импульс слабый)"

    # 5. Тело свечи (если есть данные open)
    if o_arr is not None:
        oi = float(o_arr[i])
        if oi > 0:
            body = (ci - oi) / oi * 100
            if body < config.IMPULSE_BODY_MIN:
                return False, f"тело свечи {body:+.2f}% < {config.IMPULSE_BODY_MIN}% (wick)"

    # 6. Объём
    if not np.isfinite(vx) or vx < config.IMPULSE_VOL_MIN:
        return False, f"vol× {vx:.2f} < {config.IMPULSE_VOL_MIN}"

    # 7. RSI
    if not np.isfinite(ri) or not (config.IMPULSE_RSI_LO <= ri <= config.IMPULSE_RSI_HI):
        return False, f"RSI {ri:.1f} вне [{config.IMPULSE_RSI_LO:.0f}–{config.IMPULSE_RSI_HI:.0f}]"

    # 8. daily_range — не слишком поздно
    _range_max = getattr(config, "_effective_range_max", config.DAILY_RANGE_MAX)
    if np.isfinite(dr) and dr > _range_max:
        return False, f"daily_range {dr:.1f}% > {_range_max}% (поздно)"

    return True, f"r1={r1:+.2f}% r3={r3:+.2f}% vol×{vx:.1f} RSI={ri:.0f}"


# ── ALIGNMENT: плавный бычий тренд, ADX не требуется ─────────────────────────

def check_alignment_conditions(feat: Dict, i: int, tf: str = "") -> Tuple[bool, str]:
    """
    ALIGNMENT — устойчивый бычий тренд без требования к ADX и скорости.

    Назначение: ловить медленные альт-тренды где ADX лагует 28+ баров (7ч на 15m)
    и не успевает подтвердить, но структура чётко и устойчиво бычья.

    Пример: CHZ 08.03.2026 09:00-18:00 UTC — цена +8% за 9 часов по +0.1-0.3% за свечу.
    IMPULSE требовал r1>=1.5% — ни один бар не дотянул.
    ALIGNMENT поймал бы с первых баров выше EMA20.

    Условия:
      1. close > EMA20 > EMA50 (полная бычья структура)
      2. EMA20 slope >= ALIGNMENT_SLOPE_MIN (тренд растёт, пусть и плавно)
      3. MACD hist > 0 последние ALIGNMENT_MACD_BARS баров подряд
         (устойчивый импульс, не случайный всплеск)
      4. RSI в [ALIGNMENT_RSI_LO, ALIGNMENT_RSI_HI] (не перегрет, не слаб)
      5. vol_x > ALIGNMENT_VOL_MIN (минимальная активность)
      6. daily_range < ALIGNMENT_RANGE_MAX (не слишком поздно)

    ADX не проверяется — это принципиальное отличие от BUY.
    """
    if i < 5:\
        return False, "недостаточно баров"

    ef       = feat["ema_fast"]
    es       = feat["ema_slow"]
    slope    = feat["slope"]
    rsi_arr  = feat["rsi"]
    vol_x    = feat["vol_x"]
    mh_arr   = feat["macd_hist"]
    dr_arr   = feat["daily_range_pct"]
    c_arr    = feat.get("close")
    e200_arr = feat.get("ema200")

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    efv = float(ef[i])
    esv = float(es[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x[i])
    dr  = float(dr_arr[i])

    if not all(np.isfinite([ci, efv, esv, slp, ri, vx])):
        return False, "нет данных индикаторов"

    # 1. Полная бычья структура: цена > EMA20 > EMA50
    if not (ci > efv > esv):
        return False, f"цена/EMA структура нарушена"

    # 1b. EMA20/EMA50 разрыв — защита от флэта где EMA20 ≈ EMA50.
    # MANA 12.03.2026: EMA20=0.0928, EMA50=0.0927 → разрыв 0.11% → сигнал на боковике.
    # Порог 0.3% — мягче чем для strong_trend (0.9%), т.к. alignment ловит медленные тренды.
    bull_active = bool(getattr(config, "_bull_day_active", False))
    if not bull_active and getattr(config, "ALIGNMENT_NONBULL_REQUIRE_ABOVE_EMA200", False):
        if e200_arr is not None and i < len(e200_arr) and np.isfinite(e200_arr[i]):
            e200v = float(e200_arr[i])
            if ci <= e200v:
                return False, f"close {ci:.6g} <= EMA200 {e200v:.6g} (non-bull alignment)"

    # Не слишком поздно входить
    range_max = getattr(config, "ALIGNMENT_RANGE_MAX", 9.0)
    if np.isfinite(dr) and dr > range_max:
        return False, f"daily_range {dr:.1f}% > {range_max}% (поздно)"

    if not bull_active:
        nonbull_vol_min = float(getattr(config, "ALIGNMENT_NONBULL_VOL_MIN", getattr(config, "ALIGNMENT_VOL_MIN", 0.8)))
        if vx < nonbull_vol_min:
            return False, f"vol× {vx:.2f} < {nonbull_vol_min:.2f} (non-bull alignment)"

        nonbull_rsi_lo = float(getattr(config, "ALIGNMENT_NONBULL_RSI_LO", getattr(config, "ALIGNMENT_RSI_LO", 45.0)))
        nonbull_rsi_hi = float(getattr(config, "ALIGNMENT_NONBULL_RSI_HI", getattr(config, "ALIGNMENT_RSI_HI", 82.0)))
        if not (nonbull_rsi_lo <= ri <= nonbull_rsi_hi):
            return False, f"RSI {ri:.1f} вне [{nonbull_rsi_lo:.0f}–{nonbull_rsi_hi:.0f}] (non-bull alignment)"

    price_edge_max = float(
        getattr(
            config,
            "ALIGNMENT_1H_PRICE_EDGE_MAX_PCT" if tf == "1h" else "ALIGNMENT_PRICE_EDGE_MAX_PCT",
            getattr(config, "ALIGNMENT_PRICE_EDGE_MAX_PCT", 2.0),
        )
    )
    price_edge_pct = _price_edge_pct(ci, efv)
    if np.isfinite(price_edge_pct) and price_edge_pct > price_edge_max:
        return False, (
            f"price_edge {price_edge_pct:.2f}% > {price_edge_max:.2f}% "
            f"(late alignment / stretched from EMA20)"
        )

    ema_sep_min = getattr(config, "ALIGNMENT_EMA_SEP_MIN", 0.3)
    if esv > 0:
        ema_sep_pct = (efv - esv) / esv * 100.0
        if ema_sep_pct < ema_sep_min:
            return False, f"EMA sep {ema_sep_pct:.2f}% < {ema_sep_min}% (EMA20≈EMA50, флэт)"

    # 2. EMA20 растёт (мягкий порог)
    slope_min = getattr(config, "ALIGNMENT_SLOPE_MIN", 0.05)
    if slp < slope_min:
        return False, f"slope {slp:+.2f}% < {slope_min}%"

    # 3. MACD hist > 0 последние N баров подряд
    macd_bars = getattr(config, "ALIGNMENT_MACD_BARS", 5)
    for k in range(macd_bars):
        ki = i - k
        if ki < 0:
            return False, "мало баров для MACD проверки"
        mh = float(mh_arr[ki])
        if not np.isfinite(mh) or mh <= 0:
            return False, f"MACD hist ≤ 0 на баре -{k} (нужно {macd_bars} подряд)"

    # 3b. MACD hist минимальный относительный порог — защита от «иссякшего» импульса.
    # SEI 13.03.2026: MACD hist=0.0000 — формально > 0, но momentum полностью иссяк.
    # Порог: hist должен быть ≥ 0.02% от цены (не абсолютный, масштабируется с ценой).
    # Для SEI цена~0.067: 0.0002 * 0.0 = нуль → провал. Для BTC цена~70000: 0.0002 * 70000 = 14.
    mh_now    = float(mh_arr[i])
    macd_rel_min = getattr(config, "ALIGNMENT_MACD_REL_MIN", 0.0002)  # 0.02% от цены
    macd_min_abs = ci * macd_rel_min
    if mh_now < macd_min_abs:
        return False, f"MACD hist {mh_now:.6f} < {macd_min_abs:.6f} (иссяк, < {macd_rel_min*100:.3f}% цены)"

    # 3c. Late-alignment guard: если ход за день уже приличный, а текущий MACD
    # заметно просел от недавнего локального пика, это чаще похоже на поздний
    # добор после уже состоявшегося импульса, а не на перспективный старт.
    late_range_min = float(getattr(config, "ALIGNMENT_LATE_RANGE_MIN", 6.5))
    peak_ratio_min = float(getattr(config, "ALIGNMENT_MACD_PEAK_RATIO_MIN", 0.0))
    peak_lookback = int(getattr(config, "ALIGNMENT_MACD_PEAK_LOOKBACK", 8))
    if peak_ratio_min > 0 and np.isfinite(dr) and dr >= late_range_min:
        start = max(0, i - max(1, peak_lookback) + 1)
        recent_vals = [float(x) for x in mh_arr[start : i + 1] if np.isfinite(float(x))]
        recent_peak = max(recent_vals) if recent_vals else 0.0
        if recent_peak > 0 and mh_now < recent_peak * peak_ratio_min:
            return False, (
                f"MACD hist {mh_now:.6f} < {peak_ratio_min:.2f}× recent peak "
                f"{recent_peak:.6f} (late alignment fades)"
            )

    # 4. RSI в рабочей зоне
    rsi_lo = getattr(config, "ALIGNMENT_RSI_LO", 45.0)
    rsi_hi = getattr(config, "ALIGNMENT_RSI_HI", 72.0)
    if not (rsi_lo <= ri <= rsi_hi):
        return False, f"RSI {ri:.1f} вне [{rsi_lo:.0f}–{rsi_hi:.0f}]"

    # 5. Минимальный объём (не требует спайка)
    vol_min = getattr(config, "ALIGNMENT_VOL_MIN", 0.8)
    if not np.isfinite(vx) or vx < vol_min:
        return False, f"vol× {vx:.2f} < {vol_min}"

    # 5а. Минимальный ADX — защита от чистого флэта.
    # ICP-баг (12.03.2026): ADX=13.2 — явный флэт без направленности.
    # Alignment намеренно не требует ADX ≥ 20 (ловим медленные тренды где ADX лагует),
    # но 13 — это шум. Нижний порог 15 отсекает явный флэт, не трогая слабые тренды.
    adx_min_aln = getattr(config, "ALIGNMENT_ADX_MIN", 15)
    if not bull_active:
        adx_min_aln = max(
            adx_min_aln,
            int(getattr(config, "ALIGNMENT_NONBULL_ADX_MIN", adx_min_aln)),
        )
    adx_v = feat["adx"][i]
    if not np.isfinite(float(adx_v)) or float(adx_v) < adx_min_aln:
        return False, f"ADX {float(adx_v):.1f} < {adx_min_aln} (флэт, нет направленности)"

    return True, f"EMA↑ slope={slp:+.2f}% MACD×{macd_bars}б RSI={ri:.0f} vol×{vx:.1f}"


# ── TREND_SURGE: детектор начала устойчивого тренда ──────────────────────────

def check_trend_surge_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    TREND_SURGE — фиксирует момент когда тренд «включается».

    Отличие от существующих сигналов:
      - IMPULSE: смотрит на r1/r3 (прыжок цены за 1-3 бара). Работает при быстрых
                 пробоях, но не ловит плавное начало многочасового тренда.
      - ALIGNMENT: требует 3 бара подряд MACD > 0. Пропускает первый бар разворота.
      - BUY: блокируется если ADX < 20 или ADX <= SMA(ADX,10). Лагует 10+ баров.

    TREND_SURGE фокусируется на СТРУКТУРНОМ ускорении:
      1. close > EMA20 > EMA50 — бычья структура
      2. EMA20 slope резко выросло по сравнению с 3 барами назад (ускорение)
      3. MACD hist > 0 И растёт (минимум 2 бара подряд)
      4. vol_x ≥ SURGE_VOL_MIN (объём подтверждает)
      5. RSI в [SURGE_RSI_LO, SURGE_RSI_HI] — импульс есть, не перегрет
      6. daily_range НЕ проверяется — специально для монет где BUY уже заблокирован

    Кулдаун: SURGE_COOLDOWN_BARS (20 × 15m = 5 часов) — один сигнал на тренд.

    Примеры: JASMY 09.03 03:00 UTC, BONK 09.03 12:00 UTC.
    """
    if i < 10:
        return False, "недостаточно баров"

    ef      = feat["ema_fast"]
    es      = feat["ema_slow"]
    slope   = feat["slope"]
    rsi_arr = feat["rsi"]
    vol_x   = feat["vol_x"]
    mh_arr  = feat["macd_hist"]
    c_arr   = feat.get("close")

    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    efv = float(ef[i])
    esv = float(es[i])
    slp = float(slope[i])
    ri  = float(rsi_arr[i])
    vx  = float(vol_x[i])

    if not all(np.isfinite([ci, efv, esv, slp, ri, vx])):
        return False, "нет данных индикаторов"

    # 1. Бычья структура: close > EMA20 > EMA50
    if not (ci > efv > esv):
        return False, f"структура нарушена"

    # 2. EMA20 slope ускорился: текущий slope > slope 3 бара назад + абс. порог
    slope_min = getattr(config, "SURGE_SLOPE_MIN", 0.15)
    if slp < slope_min:
        return False, f"slope {slp:+.2f}% < {slope_min}%"

    slp_prev = float(slope[i - 3]) if i >= 3 and np.isfinite(slope[i - 3]) else -999.0
    if slp <= slp_prev:
        return False, f"slope не ускоряется ({slp:+.2f}% ≤ {slp_prev:+.2f}% 3б назад)"

    # 3. MACD hist > 0 И растёт (2 бара подряд)
    mh_now  = float(mh_arr[i])
    mh_prev = float(mh_arr[i - 1]) if i >= 1 and np.isfinite(mh_arr[i - 1]) else 0.0
    if not np.isfinite(mh_now) or mh_now <= 0:
        return False, f"MACD hist ≤ 0"
    if mh_now <= mh_prev:
        return False, f"MACD hist не растёт"

    # 4. Объём
    vol_min = getattr(config, "SURGE_VOL_MIN", 1.5)
    if not np.isfinite(vx) or vx < vol_min:
        return False, f"vol× {vx:.2f} < {vol_min}"

    # 5. RSI в рабочей зоне
    rsi_lo = getattr(config, "SURGE_RSI_LO", 50.0)
    rsi_hi = getattr(config, "SURGE_RSI_HI", 80.0)
    if not np.isfinite(ri) or not (rsi_lo <= ri <= rsi_hi):
        return False, f"RSI {ri:.1f} вне [{rsi_lo:.0f}–{rsi_hi:.0f}]"

    pct_above = (ci / efv - 1) * 100
    return True, f"slope={slp:+.2f}%(↑{slp-slp_prev:+.2f}%) MACD↑ vol×{vx:.1f} RSI={ri:.0f} +{pct_above:.1f}%>EMA"



def check_ema_cross_conditions(feat: Dict, i: int) -> Tuple[bool, str]:
    """
    EMA_CROSS — самый ранний сигнал: пробой EMA20 снизу вверх с объёмом.

    Срабатывает на 3-5 баров (45-75 мин) РАНЬШЕ стандартного BUY, потому что:
      - BUY требует: slope > EMA_SLOPE_MIN, ADX > 20, ADX > SMA(ADX)
        ADX лагует 10+ баров, slope только разворачивается
      - EMA_CROSS требует только: пробой EMA20 + объём + RSI не перекуплен

    Паттерн (GLM/ORDI/AXS 11.03.2026 ~10:00 UTC):
      - 8:00-9:45: цена ниже EMA20 (боковик или слабый даун)
      - 10:00: объёмный бар закрывается выше EMA20
      - 10:15: бот даёт обычный IMPULSE/BUY
      - Цель EMA_CROSS: дать сигнал на баре 10:00

    Условия:
      1. ema_cross[i] == 1.0  (индикатор: пробой EMA20 + объём + RSI фильтр)
      2. regime != bear_trend
    """
    if i < 5:
        return False, "недостаточно баров"

    cross_arr = feat.get("ema_cross")
    if cross_arr is None:
        return False, "ema_cross не вычислен"
    if not np.isfinite(float(cross_arr[i])) or float(cross_arr[i]) < 1.0:
        return False, "нет пробоя EMA20 с объёмом"

    # Блокируем только если ГЛОБАЛЬНЫЙ рынок (BTC) в bear_trend.
    # Режим самой монеты намеренно игнорируется — EMA_CROSS ловит начало
    # восстановления именно из медвежьего/бокового периода монеты.
    global_regime = getattr(config, "_current_regime", "neutral")
    if global_regime == "bear_trend":
        return False, "BTC в режиме bear_trend — EMA_CROSS запрещён"

    ef_arr  = feat["ema_fast"]
    es_arr  = feat["ema_slow"]
    rsi_arr = feat["rsi"]
    vol_arr = feat["vol_x"]
    dr_arr  = feat["daily_range_pct"]
    c_arr   = feat.get("close")
    if c_arr is None:
        return False, "нет ценового ряда"

    ci  = float(c_arr[i])
    ef  = float(ef_arr[i])
    es  = float(es_arr[i])
    ri  = float(rsi_arr[i]) if np.isfinite(rsi_arr[i]) else 0.0
    vx  = float(vol_arr[i]) if np.isfinite(vol_arr[i]) else 0.0
    dr  = float(dr_arr[i])  if np.isfinite(dr_arr[i])  else 0.0

    pct_above = (ci / ef - 1) * 100 if ef > 0 else 0.0
    above50   = "✓" if ci > es else "–"
    cross_age = 0
    confirm_bars = int(getattr(config, "CROSS_CONFIRM_BARS", 2))
    for k in range(0, confirm_bars + 1):
        bar = i - k
        if bar < 1:
            break
        if not (np.isfinite(float(c_arr[bar])) and np.isfinite(float(ef_arr[bar]))):
            continue
        if float(c_arr[bar]) < float(ef_arr[bar]):
            continue
        below_before = True
        lookback = int(getattr(config, "CROSS_LOOKBACK", 3))
        for m in range(1, lookback + 1):
            prev = bar - m
            if prev < 0:
                below_before = False
                break
            if not (np.isfinite(float(c_arr[prev])) and np.isfinite(float(ef_arr[prev])) and float(c_arr[prev]) < float(ef_arr[prev])):
                below_before = False
                break
        if below_before:
            cross_age = k
            break

    age_note = f" age:{cross_age}b" if cross_age > 0 else ""

    return (
        True,
        f"пробой EMA20 +{pct_above:.2f}% EMA50:{above50} "
        f"vol×{vx:.1f} RSI={ri:.0f} DR={dr:.1f}%{age_note}"
    )


def _forward_accuracy(
    signals: List[int], c: np.ndarray
) -> Dict[int, HorizonAccuracy]:
    """
    Вычисляет метрики качества сигналов для каждого горизонта.

    Помимо win% считает expectancy-метрики:
      - expected_return: среднее изменение цены T+h (%)
      - median_return:   медиана
      - downside_q10:    10-й процентиль (worst case)
      - upside_q90:      90-й процентиль (best case)
      - ev_proxy:        expected_return / |downside_q10|

    Пример:
      Win% = 65%, expected_return = +0.08% → стратегия "точная, но бесполезная"
                                              (комиссия 0.1% уже убыточна)
      Win% = 58%, expected_return = +0.45% → стратегия торгуемая
    """
    result: Dict[int, HorizonAccuracy] = {}
    for h in config.FORWARD_BARS:
        correct = total = 0
        returns: List[float] = []

        for idx in signals:
            if idx + h >= len(c):
                continue  # форвард-бар ещё не закрылся
            total += 1
            entry_price = float(c[idx])
            exit_price  = float(c[idx + h])
            if entry_price <= 0:
                continue
            ret = (exit_price / entry_price - 1.0) * 100.0
            returns.append(ret)
            if ret > 0:
                correct += 1

        # Expectancy-метрики (только если есть достаточно данных)
        if len(returns) >= 2:
            arr = np.array(returns)
            expected_return = float(np.mean(arr))
            median_return   = float(np.median(arr))
            downside_q10    = float(np.percentile(arr, 10))
            upside_q90      = float(np.percentile(arr, 90))
            # EV proxy: ожидание относительно хвостового риска
            # Избегаем деление на ноль: если downside >= 0, риска нет, ev_proxy = inf → 99
            if downside_q10 < 0:
                ev_proxy = expected_return / abs(downside_q10)
            elif expected_return > 0:
                ev_proxy = 99.0   # нет downside, есть upside
            else:
                ev_proxy = 0.0
        else:
            expected_return = median_return = downside_q10 = upside_q90 = ev_proxy = None

        result[h] = HorizonAccuracy(
            horizon=h,
            total=total,
            correct=correct,
            expected_return=expected_return,
            median_return=median_return,
            downside_q10=downside_q10,
            upside_q90=upside_q90,
            ev_proxy=ev_proxy,
        )
    return result


# ── Today start index ──────────────────────────────────────────────────────────

def _today_start_ms() -> int:
    """Unix ms начало окна форвард-теста.

    Скользящее окно 24ч вместо UTC-midnight:
    - в 03:00 UTC не теряем ночные сигналы (Азия)
    - в 06:00 UTC всегда достаточно данных для подтверждения
    - confirmed работает круглосуточно без провала в 00:00-06:00 UTC
    """
    now = datetime.now(timezone.utc)
    window_hours = getattr(config, "FORWARD_TEST_WINDOW_HOURS", 24)
    window_start = now - timedelta(hours=window_hours)
    return int(window_start.timestamp() * 1000)


def _local_today_start_ms() -> int:
    tz_name = str(getattr(config, "LOCAL_TIMEZONE", "Europe/Budapest"))
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    start_local = datetime.combine(now_local.date(), dt_time.min, tzinfo=tz)
    return int(start_local.astimezone(timezone.utc).timestamp() * 1000)


def _find_today_start(timestamps: np.ndarray) -> int:
    """Индекс первого бара начиная с 00:00 UTC сегодня."""
    today_ms = _today_start_ms()
    for i, t in enumerate(timestamps):
        if int(t) >= today_ms:
            return i
    return len(timestamps) - 1  # fallback: последний бар


def _find_local_today_start(timestamps: np.ndarray) -> int:
    today_ms = _local_today_start_ms()
    for i, t in enumerate(timestamps):
        if int(t) >= today_ms:
            return i
    return len(timestamps) - 1


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze_coin(
    symbol:    str,
    tf:        str,
    data:      np.ndarray,
    from_scan: bool = False,
) -> CoinReport:
    c    = data["c"].astype(float)
    feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])

    # Минимальный прогрев индикаторов
    warmup = max(
        config.EMA_SLOW + config.SLOPE_LOOKBACK + 2,
        config.ADX_PERIOD * 2 + config.ADX_GROW_BARS + 2,
        config.VOL_LOOKBACK + 2,
        60,
    )

    # ── Найти начало сегодняшнего дня ─────────────────────────────────────────
    today_start = _find_today_start(data["t"])
    local_today_start = _find_local_today_start(data["t"])
    # Убедиться что прогрев индикаторов уже завершён к началу дня
    today_start = max(today_start, warmup)

    # ── Сигналы сегодня (только оцениваемые: есть все форвард-бары) ──────────
    # Последняя закрытая свеча
    i_now = len(c) - 2

    # Сигналы для которых уже прошло T+3 бара (45 мин) → можно оценить точность.
    # Используем min(FORWARD_BARS)=3 вместо max=10:
    # _forward_accuracy пропускает недоступные бары (if idx+h >= len(c): continue),
    # поэтому T+10 вычислится там где уже есть данные, остальные игнорируются.
    # Старая логика (i_now - 10) задерживала подтверждение на 150 мин — ранние
    # тренды (TON 08:30) не могли подтвердиться раньше 11:30.
    min_fwd  = min(config.FORWARD_BARS)
    max_fwd  = max(config.FORWARD_BARS)
    eval_end = i_now - min_fwd  # последний бар у которого T+3 уже прошёл

    today_eval_signals: List[int] = []
    if eval_end >= today_start:
        today_eval_signals = [
            i for i in range(today_start, eval_end + 1)
            if check_entry_conditions(feat, i, c, tf=tf)[0]
            or check_alignment_conditions(feat, i, tf=tf)[0]
        ]

    # Все сигналы сегодня (включая последние, ещё не оценимые)
    today_all_signals: List[int] = [
        i for i in range(today_start, i_now + 1)  # включаем последнюю закрытую свечу
        if check_entry_conditions(feat, i, c, tf=tf)[0]
        or check_alignment_conditions(feat, i, tf=tf)[0]
    ]

    # ── Форвард-тест на сегодняшних данных ───────────────────────────────────
    empty_acc = {h: HorizonAccuracy(h, 0, 0) for h in config.FORWARD_BARS}

    if len(today_eval_signals) >= config.TODAY_MIN_SIGNALS:
        today_acc = _forward_accuracy(today_eval_signals, c)
        acc_pct   = {h: fa.pct for h, fa in today_acc.items() if fa.total > 0}

        if acc_pct:
            best_h   = max(acc_pct, key=acc_pct.get)
            best_acc = acc_pct[best_h]

            t3  = today_acc.get(3)
            t10 = today_acc.get(10)

            # ── Win% фильтры (как раньше) ─────────────────────────────────────
            # T+10: если оценок < 2 — не блокируем (рано утром ещё нет данных)
            if t10 and t10.total >= 2:
                t10_ok = t10.pct >= config.TODAY_T10_MIN
            else:
                t10_ok = True

            # ── Expectancy фильтр (новый) ─────────────────────────────────────
            # Стратегия должна иметь положительное ожидание хотя бы на одном горизонте.
            # Это отсекает "точные но бесполезные" сигналы (EV < 0 несмотря на win%>50%).
            # EV_MIN_PCT: мин. ожидаемый доход в %. По умолчанию 0.0 (просто > 0).
            # Увеличить до 0.05% когда будет достаточно данных для калибровки.
            ev_min  = getattr(config, "EV_MIN_PCT", 0.0)
            ev_bars = getattr(config, "EV_MIN_SAMPLES", 3)   # мин. баров для EV-проверки

            ev_ok_any = False
            ev_detail: List[str] = []
            for h, fa in today_acc.items():
                if fa.total >= ev_bars and fa.expected_return is not None:
                    if fa.expected_return > ev_min:
                        ev_ok_any = True
                    ev_detail.append(f"T+{h} EV={fa.expected_return:+.2f}%")

            # Если данных для EV ещё мало — не блокируем (как t10 утром)
            if not ev_detail:
                ev_ok = True   # нет данных — не блокируем
            else:
                ev_ok = ev_ok_any

            confirmed = best_acc >= config.MIN_ACCURACY and t10_ok and ev_ok

            # Формируем диагностическое сообщение
            ev_note = f"  [{', '.join(ev_detail)}]" if ev_detail else ""
            note = (
                f"Сегодня {len(today_eval_signals)} сигн., подтверждено{ev_note}"
                if confirmed else
                f"Сегодня {len(today_eval_signals)} сигн., не подтверждено{ev_note}"
            )
        else:
            best_h = best_acc = 0
            confirmed = False
            note = f"Сегодня {len(today_eval_signals)} сигн., нет оценок"
    else:
        today_acc = empty_acc
        best_h    = 0
        best_acc  = 0.0
        confirmed = False
        note = (
            f"Сегодня {len(today_eval_signals)} оцен. сигн. "
            f"(нужно ≥ {config.TODAY_MIN_SIGNALS}) — не подтверждено"
        )

    # Монета в игре только если стратегия подтверждена сегодняшними данными
    in_play = confirmed

    # ── Текущий сигнал (последняя закрытая свеча) ─────────────────────────────
    signal_now     = False
    no_signal_reason = "недостаточно баров"
    signal_mode    = ""

    if i_now >= warmup:
        buy_ok, buy_reason = check_entry_conditions(feat, i_now, c, tf=tf)
        if buy_ok:
            signal_now  = True
            signal_mode, _ = get_effective_entry_mode(feat, i_now, c, tf=tf)
        else:
            # П7: проверяем RETEST
            retest_ok, _ = check_retest_conditions(feat, i_now)
            if retest_ok:
                signal_now       = True
                signal_mode      = "retest"
                no_signal_reason = ""
            else:
                # П7: проверяем BREAKOUT
                brk_ok, _ = check_breakout_conditions(feat, i_now)
                if brk_ok:
                    signal_now       = True
                    signal_mode      = "breakout"
                    no_signal_reason = ""
                else:
                    # IMPULSE: начало тренда, ADX ещё не вырос
                    imp_ok, _ = check_impulse_conditions(feat, i_now)
                    if imp_ok:
                        signal_now       = True
                        signal_mode      = "impulse"
                        no_signal_reason = ""
                    else:
                        # ALIGNMENT: плавный тренд, ADX не требуется совсем
                        aln_ok, aln_reason = check_alignment_conditions(feat, i_now, tf=tf)
                        if aln_ok:
                            signal_now       = True
                            signal_mode      = "alignment"
                            no_signal_reason = ""
                        else:
                            no_signal_reason = f"ALIGNMENT: {aln_reason}  |  BUY: {buy_reason}"

    # SETUP: только если нет ни одного сигнала — не влияет на форвард-тест
    setup_now, setup_reason, setup_missing = (False, "", 99)
    if not signal_now and i_now >= warmup:
        setup_now, setup_reason, setup_missing = check_setup_conditions(feat, i_now, c)

    forecast_candidates = [
        float(fa.expected_return)
        for fa in today_acc.values()
        if fa.total > 0 and fa.expected_return is not None
    ]
    forecast_return_pct = max(forecast_candidates) if forecast_candidates else 0.0

    local_open_idx = min(max(local_today_start, 0), i_now)
    local_open_price = float(c[local_open_idx]) if len(c) and local_open_idx < len(c) else 0.0
    current_close = float(c[i_now]) if len(c) and i_now < len(c) else 0.0
    today_change_pct = (current_close / local_open_price - 1.0) * 100.0 if local_open_price > 0 else 0.0

    def _safe(arr, idx):
        v = arr[idx] if idx < len(arr) else np.nan
        return float(v) if np.isfinite(v) else 0.0

    return CoinReport(
        symbol=symbol, tf=tf,
        today_signals=len(today_all_signals),
        today_accuracy=today_acc,
        today_confirmed=confirmed,
        best_horizon=best_h,
        best_accuracy=best_acc,
        in_play=in_play,
        note=note,
        from_scan=from_scan,
        signal_now=signal_now,
        current_price=_safe(c, i_now),
        current_slope=_safe(feat["slope"], i_now),
        current_rsi=_safe(feat["rsi"], i_now),
        current_adx=_safe(feat["adx"], i_now),
        current_vol_x=_safe(feat["vol_x"], i_now),
        current_macd=_safe(feat["macd_hist"], i_now),
        no_signal_reason=no_signal_reason,
        setup_now=setup_now,
        setup_reason=setup_reason,
        setup_missing_count=setup_missing,
        signal_mode=signal_mode,
        forecast_return_pct=forecast_return_pct,
        today_change_pct=today_change_pct,
    )


# ── Batch runner ───────────────────────────────────────────────────────────────

async def _run_analysis(
    symbols:   List[str],
    from_scan: bool = False,
) -> Tuple[List[CoinReport], List[CoinReport]]:
    all_reports: List[CoinReport] = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            (sym, tf, asyncio.create_task(fetch_klines(session, sym, tf)))
            for sym in symbols
            for tf in config.TIMEFRAMES
        ]
        for sym, tf, task in tasks:
            data = await task
            if data is None:
                continue
            all_reports.append(analyze_coin(sym, tf, data, from_scan=from_scan))

    # Лучший таймфрейм на монету.
    # Приоритет (по убыванию важности):
    #   1. today_confirmed — стратегия подтверждена сегодня
    #   2. signal_now      — активный сигнал прямо сейчас (важно для ⚡ секции)
    #   3. best_accuracy   — наибольшая точность среди оставшихся
    def _report_key(r: "CoinReport"):
        return (
            r.today_confirmed,
            r.signal_now,
            _signal_priority(r.signal_mode),
            r.today_signals,
            r.best_accuracy,
        )

    best: Dict[str, CoinReport] = {}
    for r in all_reports:
        prev = best.get(r.symbol)
        if prev is None or _report_key(r) > _report_key(prev):
            best[r.symbol] = r

    ranked   = sorted(best.values(), key=_report_key, reverse=True)
    in_play  = [r for r in ranked if r.in_play]
    skipped  = [r for r in ranked if not r.in_play]
    return in_play, skipped


async def morning_analysis(symbols: List[str]) -> Tuple[List[CoinReport], List[CoinReport]]:
    """Анализ вашего рабочего списка монет."""
    return await _run_analysis(symbols, from_scan=False)


async def market_scan() -> Tuple[List[CoinReport], List[CoinReport]]:
    """
    Полный скан всех монет из вотчлиста (load_watchlist).
    П5: перед анализом определяет характер дня (BTC 1h EMA50).
    v2: определяет Market Regime и применяет адаптивные пороги.
    """
    async with aiohttp.ClientSession() as _sess:
        bull, btc_price, btc_ema50 = await is_bull_day(_sess)
        regime = await detect_market_regime(_sess)

    config._bull_day_active       = bull
    config._effective_range_max   = getattr(config, "BULL_DAY_RANGE_MAX", 10.0) if bull else config.DAILY_RANGE_MAX
    config._effective_rsi_hi      = getattr(config, "BULL_DAY_RSI_HI",    75.0) if bull else config.RSI_BUY_HI
    config._btc_vs_ema50          = round((btc_price / btc_ema50 - 1) * 100, 4) if btc_ema50 > 0 else 0.0

    # v2: сохраняем глобальный режим рынка для использования в анализе
    config._current_regime        = regime.name
    config._regime_params_active  = getattr(config, "REGIME_PARAMS", {}).get(regime.name, {})

    # В бычьем тренде или recovery — расширяем диапазон дополнительно
    if regime.name in ("bull_trend", "recovery") and not bull:
        # BTC в хорошем состоянии, но не официальный бычий день → небольшое расширение
        config._effective_range_max = max(
            config._effective_range_max, regime.range_max
        )

    botlog.log_bull_day(
        is_bull=bull, btc_price=btc_price, btc_ema50=btc_ema50,
        eff_range_max=config._effective_range_max,
        eff_rsi_hi=config._effective_rsi_hi,
    )

    watchlist = config.load_watchlist()
    in_play, skipped = await _run_analysis(watchlist, from_scan=False)

    regime_str = str(regime)
    botlog.log_analysis(
        n_scanned=len(watchlist),
        n_confirmed=len(in_play),
        n_signal_now=sum(1 for r in in_play if r.signal_now),
        n_setup=sum(1 for r in skipped if r.setup_now),
        n_early=sum(1 for r in skipped if r.signal_now),
        is_bull=bull,
        confirmed_symbols=[r.symbol for r in in_play],
    )

    # Добавляем информацию о режиме в первый отчёт (используется в bot.py для заголовка)
    config._regime_display = regime_str

    return in_play, skipped
