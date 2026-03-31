from __future__ import annotations

from typing import Dict

import numpy as np


# ── Core indicator functions ───────────────────────────────────────────────────

def _ema(x: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full_like(close, np.nan)
    if len(close) < period + 1:
        return out
    delta = np.diff(close)
    up    = np.where(delta > 0, delta, 0.0)
    down  = np.where(delta < 0, -delta, 0.0)
    avg_up   = np.full_like(close, np.nan)
    avg_down = np.full_like(close, np.nan)
    avg_up[period]   = up[:period].mean()
    avg_down[period] = down[:period].mean()
    for i in range(period + 1, len(close)):
        avg_up[i]   = (avg_up[i - 1]   * (period - 1) + up[i - 1])   / period
        avg_down[i] = (avg_down[i - 1] * (period - 1) + down[i - 1]) / period
    rs = np.full_like(close, np.inf)
    np.divide(avg_up, avg_down, out=rs, where=avg_down > 0)
    out[:] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    h, l, c = (np.asarray(x, dtype=float) for x in (h, l, c))
    out = np.full_like(c, np.nan)
    if len(c) < period + 1:
        return out
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out[period] = tr[1:period + 1].mean()
    for i in range(period + 1, len(c)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    h, l, c = (np.asarray(x, dtype=float) for x in (h, l, c))
    out = np.full_like(c, np.nan)
    n = len(c)
    if n < period * 2 + 2:
        return out
    up    = h[1:] - h[:-1]
    down  = l[:-1] - l[1:]
    pdm   = np.where((up > down) & (up > 0), up, 0.0)
    mdm   = np.where((down > up) & (down > 0), down, 0.0)
    pc    = c[:-1]
    tr    = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    tr_s  = np.full(n, np.nan)
    p_s   = np.full(n, np.nan)
    m_s   = np.full(n, np.nan)
    tr_s[period] = tr[:period].sum()
    p_s[period]  = pdm[:period].sum()
    m_s[period]  = mdm[:period].sum()
    for i in range(period + 1, n):
        tr_s[i] = tr_s[i - 1] - tr_s[i - 1] / period + tr[i - 1]
        p_s[i]  = p_s[i - 1]  - p_s[i - 1]  / period + pdm[i - 1]
        m_s[i]  = m_s[i - 1]  - m_s[i - 1]  / period + mdm[i - 1]
    pdi = np.zeros(n, dtype=float)
    mdi = np.zeros(n, dtype=float)
    np.divide(p_s, tr_s, out=pdi, where=tr_s > 0)
    np.divide(m_s, tr_s, out=mdi, where=tr_s > 0)
    pdi *= 100.0
    mdi *= 100.0
    denom = pdi + mdi
    dx = np.zeros(n, dtype=float)
    np.divide(np.abs(pdi - mdi), denom, out=dx, where=denom > 0)
    dx *= 100.0
    out[period * 2] = dx[period:period * 2].mean()
    for i in range(period * 2 + 1, n):
        out[i] = (out[i - 1] * (period - 1) + dx[i - 1]) / period
    return out


def _ema_slope(ema_arr: np.ndarray, lookback: int) -> np.ndarray:
    """
    % change of EMA over `lookback` bars.
    Positive value = EMA is rising (uptrend momentum).
    This is the key indicator for catching trend STARTS (not trend middles).
    """
    out = np.full_like(ema_arr, np.nan)
    for i in range(lookback, len(ema_arr)):
        base = ema_arr[i - lookback]
        if base > 0:
            out[i] = (ema_arr[i] - base) / base * 100.0
    return out


# ── New v2 indicators ──────────────────────────────────────────────────────────

def _slope_acceleration(slope_arr: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Ускорение наклона EMA: slope[i] - slope[i-lookback].
    Положительное значение = наклон растёт → тренд разгоняется.
    Ловит начало движения на 1-3 бара раньше BUY.
    """
    out = np.full_like(slope_arr, np.nan)
    for i in range(lookback, len(slope_arr)):
        if np.isfinite(slope_arr[i]) and np.isfinite(slope_arr[i - lookback]):
            out[i] = slope_arr[i] - slope_arr[i - lookback]
    return out


def _squeeze_state(atr: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    ATR Squeeze: массив состояний для каждого бара.
    Значение > 0 означает активное сжатие (ATR < 50% от своей SMA).
    Используется вместе с _squeeze_breakout для определения момента пробоя.
    """
    out = np.zeros(len(atr), dtype=float)
    atr_sma = np.full_like(atr, np.nan)
    for i in range(lookback - 1, len(atr)):
        window = atr[i - lookback + 1:i + 1]
        if np.all(np.isfinite(window)):
            atr_sma[i] = window.mean()
    for i in range(len(atr)):
        if np.isfinite(atr[i]) and np.isfinite(atr_sma[i]) and atr_sma[i] > 0:
            ratio = atr[i] / atr_sma[i]
            out[i] = 1.0 if ratio < 0.5 else 0.0
    return out


def _squeeze_breakout(
    atr: np.ndarray,
    close: np.ndarray,
    lookback: int = 20,
    squeeze_ratio: float = 0.5,
    expansion_mult: float = 1.8,
    min_squeeze_bars: int = 5,
) -> np.ndarray:
    """
    Пробой ATR-сжатия: была ли пружина сжата и теперь резко расширяется?

    Алгоритм:
    1. Найти период сжатия: ATR < squeeze_ratio * SMA(ATR, lookback)
    2. Минимум сжатия (дно ATR за период сжатия)
    3. Текущий ATR > expansion_mult * min_atr_squeeze — пробой

    Возвращает массив: 1.0 = пробой сжатия на этом баре, 0.0 = нет.
    """
    out = np.zeros(len(atr), dtype=float)
    atr_sma = np.full_like(atr, np.nan)
    for i in range(lookback - 1, len(atr)):
        window = atr[i - lookback + 1:i + 1]
        if np.all(np.isfinite(window)):
            atr_sma[i] = window.mean()

    for i in range(lookback + min_squeeze_bars, len(atr)):
        if not (np.isfinite(atr[i]) and np.isfinite(atr_sma[i])):
            continue
        # Проверяем что перед текущим баром был достаточный период сжатия
        squeeze_run = 0
        atr_in_squeeze = []
        for k in range(1, lookback + 1):
            ki = i - k
            if ki < 0:
                break
            if np.isfinite(atr[ki]) and np.isfinite(atr_sma[ki]) and atr_sma[ki] > 0:
                if atr[ki] / atr_sma[ki] < squeeze_ratio:
                    squeeze_run += 1
                    atr_in_squeeze.append(atr[ki])
                else:
                    break  # сжатие прервалось
            else:
                break

        if squeeze_run < min_squeeze_bars:
            continue

        min_atr = min(atr_in_squeeze)
        if min_atr <= 0:
            continue

        # Пробой: текущий ATR резко вырос от дна сжатия
        if atr[i] >= expansion_mult * min_atr:
            # Дополнительно: цена растёт (бычий пробой)
            if i >= 1 and np.isfinite(close[i]) and np.isfinite(close[i - 1]):
                if close[i] > close[i - 1]:
                    out[i] = 1.0

    return out


def _rsi_divergence(
    close: np.ndarray,
    rsi: np.ndarray,
    lookback: int = 10,
    price_margin: float = 0.001,
) -> np.ndarray:
    """
    Медвежья RSI-дивергенция: цена делает новый максимум, RSI — нет.

    Признак ослабления тренда: покупатели толкают цену вверх,
    но с каждым баром всё слабее (RSI не подтверждает).

    Возвращает: 1.0 = дивергенция обнаружена, 0.0 = нет.
    """
    out = np.zeros(len(close), dtype=float)
    for i in range(lookback, len(close)):
        if not (np.isfinite(close[i]) and np.isfinite(rsi[i])):
            continue
        # Ищем предыдущий локальный максимум цены и RSI в окне
        window_close = close[i - lookback:i]
        window_rsi   = rsi[i - lookback:i]
        if not (np.all(np.isfinite(window_close)) and np.all(np.isfinite(window_rsi))):
            continue
        prev_close_max = float(np.max(window_close))
        prev_rsi_max   = float(np.max(window_rsi))
        cur_close = float(close[i])
        cur_rsi   = float(rsi[i])
        # Цена выше предыдущего максимума (на значимую величину)
        price_higher = cur_close > prev_close_max * (1 + price_margin)
        # RSI не выше предыдущего максимума → дивергенция
        rsi_not_higher = cur_rsi <= prev_rsi_max
        if price_higher and rsi_not_higher:
            out[i] = 1.0
    return out


def _volume_exhaustion(
    close: np.ndarray,
    volume: np.ndarray,
    bars: int = 5,
    price_min_pct: float = 0.5,
) -> np.ndarray:
    """
    Объёмное истощение: цена растёт N баров подряд, объём каждый бар убывает.

    Классический признак конца тренда: быки выдыхаются, новые покупатели
    не приходят, движение держится только на инерции.

    Возвращает: 1.0 = истощение обнаружено, 0.0 = нет.
    """
    out = np.zeros(len(close), dtype=float)
    for i in range(bars, len(close)):
        if not (np.isfinite(close[i]) and np.isfinite(volume[i])):
            continue
        window_c = close[i - bars:i + 1]
        window_v = volume[i - bars + 1:i + 1]  # N объёмов для N убываний
        if not (np.all(np.isfinite(window_c)) and np.all(np.isfinite(window_v))):
            continue
        # Цена выросла минимально за весь период
        total_move = (float(close[i]) - float(close[i - bars])) / max(float(close[i - bars]), 1e-12) * 100
        if total_move < price_min_pct:
            continue
        # Объём убывал каждый бар (каждый следующий бар < предыдущего)
        vol_declining = all(window_v[j] < window_v[j - 1] for j in range(1, len(window_v)))
        if vol_declining:
            out[i] = 1.0
    return out


def _ema_fan_spread(
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    lookback: int = 8,
    decay_threshold: float = 0.30,
) -> np.ndarray:
    """
    EMA Fan Collapse: spread между EMA20 и EMA50 упал на decay_threshold
    от своего максимума за последние `lookback` баров.

    В тренде spread растёт — EMA20 всё дальше от EMA50.
    При развороте spread начинает сужаться — тренд слабеет.

    Возвращает: положительное значение = нормализованное сужение (0..1),
    0 = нет сужения, NaN = нет данных.
    Значение > decay_threshold = предупреждение о конце тренда.
    """
    out = np.full(len(ema_fast), np.nan)
    for i in range(lookback, len(ema_fast)):
        ef = float(ema_fast[i])
        es = float(ema_slow[i])
        if not (np.isfinite(ef) and np.isfinite(es) and es > 0):
            continue
        cur_spread = (ef - es) / es  # может быть отрицательным при медвежьем рынке
        if cur_spread <= 0:
            out[i] = 0.0
            continue
        # Максимальный spread за lookback баров назад
        window_ef = ema_fast[i - lookback:i]
        window_es = ema_slow[i - lookback:i]
        if not (np.all(np.isfinite(window_ef)) and np.all(np.isfinite(window_es))):
            continue
        spreads = np.where(window_es > 0, (window_ef - window_es) / window_es, 0.0)
        max_spread = float(np.max(spreads))
        if max_spread <= 0:
            out[i] = 0.0
            continue
        # Насколько текущий spread упал от максимума
        decay = (max_spread - cur_spread) / max_spread
        out[i] = float(np.clip(decay, 0.0, 1.0))
    return out


def _market_regime(
    close: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    adx: np.ndarray,
    slope: np.ndarray,
    adx_trend_threshold: float = 22.0,
    adx_flat_threshold:  float = 18.0,
) -> np.ndarray:
    """
    Определяет режим рынка для каждого бара.

    Коды режимов (строки):
      'bull_trend'    — цена > EMA20 > EMA50, ADX сильный, slope > 0
      'bear_trend'    — цена < EMA50, ADX сильный
      'consolidation' — ADX слабый, флэт
      'recovery'      — цена только что пробила EMA50 снизу вверх (slope ускоряется)
      'neutral'       — всё остальное

    Возвращает массив строк.
    """
    n = len(close)
    out = np.empty(n, dtype=object)
    out[:] = "neutral"

    prev_below_slow = False
    for i in range(1, n):
        c   = float(close[i])
        ef  = float(ema_fast[i])
        es  = float(ema_slow[i])
        adx_v = float(adx[i])
        slp   = float(slope[i])
        if not all(np.isfinite([c, ef, es])):
            continue

        prev_c  = float(close[i - 1])
        prev_es = float(ema_slow[i - 1])
        was_below = np.isfinite(prev_c) and np.isfinite(prev_es) and prev_c < prev_es

        if np.isfinite(adx_v) and adx_v >= adx_trend_threshold:
            if c > ef > es:
                # Был под EMA50, теперь над → recovery
                if was_below:
                    out[i] = "recovery"
                else:
                    out[i] = "bull_trend"
            elif c < es:
                out[i] = "bear_trend"
            else:
                out[i] = "neutral"
        elif np.isfinite(adx_v) and adx_v < adx_flat_threshold:
            out[i] = "consolidation"
        else:
            # ADX промежуточный
            if c > ef > es and np.isfinite(slp) and slp > 0:
                out[i] = "bull_trend"
            elif c < es:
                out[i] = "bear_trend"
            else:
                out[i] = "consolidation"

    return out


def _avg_daily_range(daily_range_pct: np.ndarray, hist_bars: int = 96 * 14) -> np.ndarray:
    """
    Средний дневной диапазон монеты за последние hist_bars баров.
    Используется для динамического порога daily_range_max.

    Вычисляет скользящее среднее daily_range_pct за окно hist_bars.
    Возвращает массив — среднее значение на каждом баре.
    """
    out = np.full_like(daily_range_pct, np.nan)
    for i in range(hist_bars - 1, len(daily_range_pct)):
        window = daily_range_pct[i - hist_bars + 1:i + 1]
        finite = window[np.isfinite(window)]
        if len(finite) >= hist_bars // 2:  # нужно хотя бы половину данных
            out[i] = float(np.mean(finite))
    return out


def _dynamic_range_max(
    avg_dr: np.ndarray,
    base_max: float = 7.0,
    ref_pct:  float = 5.0,
    min_val:  float = 3.0,
    max_cap:  float = 25.0,
) -> np.ndarray:
    """
    Динамический порог daily_range_max, адаптированный к волатильности монеты.

    Формула: base_max × (avg_daily_range / ref_pct)
    Например:
      XAI avg=15% → 7 × (15/5) = 21%
      BTC avg=3%  → 7 × (3/5)  = 4.2%

    Возвращает np.nan где нет данных (стратегия fallback на config.DAILY_RANGE_MAX).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            np.isfinite(avg_dr) & (avg_dr > 0),
            base_max * (avg_dr / ref_pct),
            np.nan,
        )
    return np.clip(result, min_val, max_cap)


# ── Feature bundle ─────────────────────────────────────────────────────────────

def _ema_cross(
    close:     np.ndarray,
    ema_fast:  np.ndarray,
    ema_slow:  np.ndarray,
    vol_x:     np.ndarray,
    macd_hist: np.ndarray,
    *,
    vol_min:         float = 1.2,
    ema50_slope_min: float = -0.05,
    rsi_lo:          float = 38.0,
    rsi_hi:          float = 72.0,
    range_max:       float = 6.0,
    daily_range_pct: np.ndarray | None = None,
    rsi:             np.ndarray | None = None,
    lookback:        int   = 3,
    confirm_bars:    int   = 2,
    macd_filter:     bool  = True,
) -> np.ndarray:
    """
    EMA Cross detector — ранний сигнал пробоя EMA20 снизу вверх.

    Возвращает массив float:
      1.0 = текущий бар является пробоем EMA20 снизу вверх (или подтверждением ≤ confirm_bars назад)
      0.0 = нет сигнала

    Условия пробоя на баре i:
      - close[i] >= ema_fast[i]                 (цена закрылась выше EMA20)
      - close[i-k] < ema_fast[i-k] для k=1..lookback  (до этого была ниже)
      - vol_x[i] >= vol_min                     (объём подтверждает)
      - ema50 slope >= ema50_slope_min           (EMA50 не падает резко)
      - macd_hist[i] >= 0 если macd_filter=True (MACD не медвежий)
      - rsi в диапазоне [rsi_lo, rsi_hi]
      - daily_range_pct <= range_max            (монета не разогнана)
    """
    n   = len(close)
    out = np.zeros(n, dtype=float)

    # EMA50 slope — процентное изменение за 3 бара
    es_slope = np.full(n, np.nan)
    for i in range(3, n):
        if ema_slow[i - 3] > 0 and np.isfinite(ema_slow[i]) and np.isfinite(ema_slow[i - 3]):
            es_slope[i] = (ema_slow[i] / ema_slow[i - 3] - 1.0) * 100.0

    cross_bars = np.zeros(n, dtype=int)   # бар на котором был пробой (0 = не было)

    for i in range(lookback + 1, n):
        # --- Нужны валидные данные ---
        if not (np.isfinite(close[i]) and np.isfinite(ema_fast[i])
                and np.isfinite(ema_fast[i - 1])):
            continue

        # --- Ищем момент пробоя в последние confirm_bars ---
        cross_at = -1
        for k in range(0, confirm_bars + 1):
            bar = i - k
            if bar < lookback + 1:
                break
            if not (np.isfinite(close[bar]) and np.isfinite(ema_fast[bar])):
                continue
            # Пробой: close[bar] >= ema20[bar] И предыдущие lookback баров были ниже
            if close[bar] >= ema_fast[bar]:
                below_before = all(
                    np.isfinite(close[bar - m]) and np.isfinite(ema_fast[bar - m])
                    and close[bar - m] < ema_fast[bar - m]
                    for m in range(1, lookback + 1)
                )
                if below_before:
                    cross_at = bar
                    break

        if cross_at < 0:
            continue

        # --- Фильтры на баре пробоя ---
        ci = cross_at

        # Объём
        if not (np.isfinite(vol_x[ci]) and vol_x[ci] >= vol_min):
            continue

        # EMA50 slope
        if np.isfinite(es_slope[ci]) and es_slope[ci] < ema50_slope_min:
            continue

        # MACD hist >= 0
        if macd_filter and np.isfinite(macd_hist[ci]) and macd_hist[ci] < 0:
            continue

        # RSI диапазон
        if rsi is not None and np.isfinite(rsi[ci]):
            if rsi[ci] < rsi_lo or rsi[ci] > rsi_hi:
                continue

        # Монета уже не слишком разогнана
        if daily_range_pct is not None and np.isfinite(daily_range_pct[ci]):
            if daily_range_pct[ci] > range_max:
                continue

        out[i] = 1.0

    return out


def compute_features(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute all indicators needed for entry/exit decisions.
    Returns a dict of equal-length arrays aligned to input candles.
    """
    import config  # late import to avoid circular

    c = np.asarray(c, dtype=float)
    v = np.asarray(v, dtype=float)

    ef  = _ema(c, config.EMA_FAST)
    es  = _ema(c, config.EMA_SLOW)
    e200 = _ema(c, 200)               # EMA200 — долгосрочный тренд
    rsi = _rsi(c, config.RSI_PERIOD)
    adx = _adx(h, l, c, config.ADX_PERIOD)
    atr = _atr(h, l, c, config.ATR_PERIOD)

    # SMA of ADX — вход разрешён только если ADX выше своей SMA (тренд подтверждён)
    adx_sma = np.full_like(adx, np.nan)
    sp = config.ADX_SMA_PERIOD
    for i in range(sp - 1, len(adx)):
        window = adx[i - sp + 1:i + 1]
        if np.all(np.isfinite(window)):
            adx_sma[i] = window.mean()
    slp = _ema_slope(ef, config.SLOPE_LOOKBACK)

    # Volume relative to N-bar average
    vol_sma = np.full_like(v, np.nan)
    lb = config.VOL_LOOKBACK
    for i in range(lb - 1, len(v)):
        vol_sma[i] = v[i - lb + 1:i + 1].mean()
    vol_x = np.where(vol_sma > 0, v / vol_sma, np.nan)

    # MACD (12/26/9) — гистограмма > 0 бычий импульс, < 0 медвежий
    ema12     = _ema(c, 12)
    ema26     = _ema(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = _ema(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # Daily range % — насколько цена выросла от реального минимума (low) последних 96 баров
    # На 15m: 96 баров = 24 часа. Блокирует вход если монета уже > DAILY_RANGE_MAX от дна
    # FIX: используем min(low), а не min(close) — реальное дно свечи всегда ≤ close
    # Это устраняло завышение daily_range_pct и преждевременные блокировки
    dr = 96
    l_arr = np.asarray(l, dtype=float)
    daily_low = np.full_like(c, np.nan)
    for i in range(dr - 1, len(c)):
        daily_low[i] = np.min(l_arr[i - dr + 1:i + 1])
    daily_range_pct = np.where(
        daily_low > 0, (c - daily_low) / daily_low * 100.0, np.nan
    )

    # ── v2 indicators ──────────────────────────────────────────────────────────

    # Slope acceleration — ускорение наклона EMA20
    slp_accel = _slope_acceleration(slp, getattr(config, "SLOPE_ACCEL_BARS", 3))

    # ATR Squeeze Breakout — пробой сжатия
    sq_brkout = _squeeze_breakout(
        atr, c,
        lookback         = getattr(config, "SQUEEZE_LOOKBACK",    20),
        squeeze_ratio    = getattr(config, "ATR_SQUEEZE_RATIO",   0.5),
        expansion_mult   = getattr(config, "ATR_EXPANSION_MULT",  1.8),
        min_squeeze_bars = getattr(config, "SQUEEZE_MIN_BARS",     5),
    )

    # RSI Divergence — медвежья дивергенция (сигнал конца тренда)
    rsi_div = _rsi_divergence(
        c, rsi,
        lookback     = getattr(config, "RSI_DIV_LOOKBACK",       10),
        price_margin = getattr(config, "RSI_DIV_PRICE_MARGIN", 0.001),
    )

    # Volume Exhaustion — истощение объёма (сигнал конца тренда)
    vol_exhaust = _volume_exhaustion(
        c, v,
        bars          = getattr(config, "VOL_EXHAUST_BARS",       5),
        price_min_pct = getattr(config, "VOL_EXHAUST_PRICE_MIN", 0.5),
    )

    # EMA Fan Spread — схлопывание веера EMA (начало разворота)
    fan_spread = _ema_fan_spread(
        ef, es,
        lookback        = getattr(config, "EMA_FAN_LOOKBACK",        8),
        decay_threshold = getattr(config, "EMA_FAN_DECAY_THRESHOLD", 0.30),
    )

    # Market Regime — текущий режим рынка
    regime = _market_regime(
        c, ef, es, adx, slp,
        adx_trend_threshold = getattr(config, "REGIME_BTC_ADX_TREND", 22.0),
        adx_flat_threshold  = getattr(config, "REGIME_BTC_ADX_FLAT",  18.0),
    )

    # Dynamic Range Max — адаптивный порог по волатильности монеты
    hist_bars = getattr(config, "DYNAMIC_RANGE_HIST_BARS", 96 * 14)
    avg_dr    = _avg_daily_range(daily_range_pct, hist_bars)
    dyn_range_max = _dynamic_range_max(
        avg_dr,
        base_max = getattr(config, "DAILY_RANGE_MAX",       7.0),
        ref_pct  = getattr(config, "DYNAMIC_RANGE_REF_PCT", 5.0),
        min_val  = getattr(config, "DYNAMIC_RANGE_MIN",     3.0),
        max_cap  = getattr(config, "DYNAMIC_RANGE_MAX_CAP", 25.0),
    )

    # EMA Cross — ранний сигнал пробоя EMA20 снизу вверх с объёмом
    ema_cross_sig = _ema_cross(
        close           = c,
        ema_fast        = ef,
        ema_slow        = es,
        vol_x           = vol_x,
        macd_hist       = macd_hist,
        vol_min         = getattr(config, "CROSS_VOL_MIN",         1.2),
        ema50_slope_min = getattr(config, "CROSS_EMA50_SLOPE_MIN", -0.05),
        rsi_lo          = getattr(config, "CROSS_RSI_LO",          38.0),
        rsi_hi          = getattr(config, "CROSS_RSI_HI",          72.0),
        range_max       = getattr(config, "CROSS_RANGE_MAX",        6.0),
        daily_range_pct = daily_range_pct,
        rsi             = rsi,
        lookback        = getattr(config, "CROSS_LOOKBACK",           3),
        confirm_bars    = getattr(config, "CROSS_CONFIRM_BARS",        2),
        macd_filter     = getattr(config, "CROSS_MACD_FILTER",     True),
    )

    return {
        "ema_fast":        ef,
        "ema200":          e200,
        "ema_slow":        es,
        "rsi":             rsi,
        "adx":             adx,
        "atr":             atr,
        "slope":           slp,
        "vol_x":           vol_x,
        "macd_hist":       macd_hist,       # > 0 бычий импульс
        "daily_range_pct": daily_range_pct, # % от минимума дня
        "adx_sma":         adx_sma,         # SMA(ADX,10) — фильтр силы тренда на входе
        # Сырые ценовые ряды — нужны для RETEST/BREAKOUT/IMPULSE детекторов
        "close":           c,
        "high":            np.asarray(h, dtype=float),
        "low":             np.asarray(l, dtype=float),
        "open":            np.asarray(o, dtype=float),
        # v2: новые индикаторы
        "slope_accel":      slp_accel,     # ускорение наклона EMA
        "squeeze_breakout": sq_brkout,     # 1.0 = пробой сжатия ATR
        "rsi_divergence":   rsi_div,       # 1.0 = медвежья дивергенция
        "vol_exhaustion":   vol_exhaust,   # 1.0 = объёмное истощение
        "ema_fan_spread":   fan_spread,    # 0..1, >0.3 = схлопывание веера
        "regime":           regime,        # строка: bull_trend/bear_trend/etc
        "dyn_range_max":    dyn_range_max, # адаптивный порог daily_range
        # v3: EMA Cross — ранний сигнал
        "ema_cross":        ema_cross_sig, # 1.0 = пробой EMA20 снизу вверх с объёмом
    }
