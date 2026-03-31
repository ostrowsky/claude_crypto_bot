from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Telegram UX / noise control
ENABLE_EARLY_SCANNER_ALERTS: bool = False
SEND_DISCOVERY_NOTIFICATIONS: bool = False
SEND_SERVICE_NOTIFICATIONS: bool = False
SEND_AUX_NOTIFICATIONS: bool = False
ML_ENABLE_GENERAL_RANKING: bool = True
ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE: bool = True
ML_GENERAL_NEUTRAL_PROBA: float = 0.50
ML_GENERAL_SCORE_WEIGHT: float = 0.0
ML_ENABLE_TREND_NONBULL_FILTER: bool = True
ML_TREND_NONBULL_SEGMENT_KEY: str = "trend|nonbull"
ML_TREND_NONBULL_MIN_PROBA: float = 0.35
ML_TREND_NONBULL_LOW_PROBA_PENALTY: float = 6.0
ML_TREND_NONBULL_HARD_BLOCK: bool = False
CRITIC_DATASET_ENABLED: bool = True
ML_CANDIDATE_RANKER_RUNTIME_ENABLED: bool = False
ML_CANDIDATE_RANKER_MODEL_FILE: str = "ml_candidate_ranker.json"
ML_CANDIDATE_RANKER_NEUTRAL_PROBA: float = 0.50
ML_CANDIDATE_RANKER_SCORE_WEIGHT: float = 0.0
ML_CANDIDATE_RANKER_SHADOW_ENABLED: bool = True
ML_CANDIDATE_RANKER_SHADOW_LOG_ALL: bool = False

# PATCH: Acceptance gate parameters (written by rl_headless_worker when gate passes)
# These are auto-updated via rl_config_overrides.json - do not edit manually.
# ML_CANDIDATE_RANKER_RUNTIME_ENABLED becomes True when:
#   AUC >= 0.55, top1_delta >= +0.15%, live_rows >= 1000
ML_RANKER_ACCEPTANCE_AUC_MIN: float = 0.55
ML_RANKER_ACCEPTANCE_TOP1_DELTA_MIN: float = 0.15
ML_RANKER_ACCEPTANCE_LIVE_ROWS_MIN: int = 1000

# ── Binance ───────────────────────────────────────────────────────────────────
BINANCE_REST: str = "https://api.binance.com"

# ── Watchlist ─────────────────────────────────────────────────────────────────
WATCHLIST_FILE = Path("watchlist.json")

DEFAULT_WATCHLIST: list[str] = [
    # Топ ликвидность
    "BTCUSDT",   "ETHUSDT",   "BNBUSDT",   "XRPUSDT",   "SOLUSDT",
    "TRXUSDT",   "DOGEUSDT",  "ADAUSDT",   "AVAXUSDT",  "SHIBUSDT",
    "DOTUSDT",   "LINKUSDT",  "LTCUSDT",   "BCHUSDT",   "UNIUSDT",
    "TONUSDT",   "NEARUSDT",  "ICPUSDT",   "AAVEUSDT",  "HBARUSDT",
    # L2 и новые сети
    "ARBUSDT",   "OPUSDT",    "POLUSDT",   "SUIUSDT",   "APTUSDT",   "STRKUSDT",
    "ZROUSDT",   "UMAUSDT",   "ENAUSDT",   "SEIUSDT",   "METISUSDT",
    # DeFi
    "MKRUSDT",   "CRVUSDT",   "SUSHIUSDT", "COMPUSDT",  "YFIUSDT",
    "SNXUSDT",   "LDOUSDT",   "1INCHUSDT", "DYDXUSDT",
    # AI и инфраструктура
    "FETUSDT",   "RNDRUSDT",  "TAOUSDT",   "INJUSDT",
    # Layer 1
    "ATOMUSDT",  "ALGOUSDT",  "XLMUSDT",   "XTZUSDT",   "EOSUSDT",   "CELOUSDT",
    "ETCUSDT",   "FILUSDT",   "EGLDUSDT",  "PAXGUSDT",  "QNTUSDT",   "RUNEUSDT",
    # GameFi, NFT, мемы
    "CAKEUSDT",  "GRTUSDT",   "AXSUSDT",   "SANDUSDT",  "MANAUSDT",
    "CHZUSDT",   "APEUSDT",   "FLOKIUSDT", "WIFUSDT",   "BONKUSDT",
    "ILVUSDT",   "AUDIOUSDT", "JASMYUSDT",
    # Средний эшелон
    "ACHUSDT",   "CFXUSDT",   "ENSUSDT",   "GMTUSDT",   "ORDIUSDT",  "WLDUSDT",
    "BLURUSDT",  "LRCUSDT",   "ZRXUSDT",   "ZILUSDT",   "KSMUSDT",
    "BATUSDT",   "AMPUSDT",   "BNTUSDT",   "MDTUSDT",   "GLMUSDT",   "FLUXUSDT",
    "OXTUSDT",   "BAKEUSDT",  "PYRUSDT",   "TRUUSDT",   "ARUSDT",    "COTIUSDT",
    "CELRUSDT",  "QIUSDT",    "SNTUSDT",   "AXLUSDT",   "TIAUSDT",   "AEVOUSDT",
    "RENDERUSDT","XAIUSDT",   "C98USDT",   "ACAUSDT",   "LQTYUSDT",
]


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return list(DEFAULT_WATCHLIST)


def save_watchlist(symbols: list[str]) -> None:
    WATCHLIST_FILE.write_text(json.dumps(symbols, indent=2))


# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES: list[str] = ["15m", "1h"]

# ── Wide market scan ──────────────────────────────────────────────────────────
SCAN_TOP_N:   int       = 50
SCAN_QUOTE:   str       = "USDT"
SCAN_EXCLUDE: list[str] = [
    "UP", "DOWN", "BULL", "BEAR",
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
]

# ── Indicator parameters ──────────────────────────────────────────────────────
EMA_FAST       = 20
EMA_SLOW       = 50
RSI_PERIOD     = 14
ADX_PERIOD     = 14
ATR_PERIOD     = 14
VOL_LOOKBACK   = 20
SLOPE_LOOKBACK = 5

# ── Entry conditions ──────────────────────────────────────────────────────────
EMA_SLOPE_MIN  = 0.10
ADX_MIN        = 20.0
ADX_GROW_BARS  = 3   # используется только в выходе (ADX ослаб)
ADX_SMA_PERIOD = 10  # период SMA для фильтра ADX на входе
VOL_MULT       = 1.30
RSI_BUY_LO         = 45.0
RSI_BUY_HI         = 72.0   # стандартная верхняя граница RSI
RSI_BUY_HI_STRONG  = 80.0   # расширенная граница при сильном тренде
STRONG_ADX_MIN     = 28.0   # ADX ≥ этого → "сильный тренд"
STRONG_VOL_MIN     = 2.0    # vol× ≥ этого → "сильный объём"
STRONG_RSI_MIN     = 55.0   # strong_trend не берём на вялом RSI
STRONG_CLOSE_EMA20_MAX_PCT: float = 6.0  # strong_trend оставляем и на зрелом импульсе, если структура ещё здорова
# При ADX ≥ STRONG_ADX_MIN И vol× ≥ STRONG_VOL_MIN → RSI разрешён до RSI_BUY_HI_STRONG

# ── IMPULSE: детектор начала тренда ──────────────────────────────────────────
# Срабатывает в самом начале движения — до того как ADX успевает вырасти.
# Ключевое отличие: ADX не требуется, вместо него — объём и скорость цены.

IMPULSE_VOL_MIN:      float = 2.0   # vol_x минимум — нужен реальный объём
IMPULSE_PRICE_SPEED:  float = 1.0   # % рост цены за IMPULSE_SPEED_BARS баров
IMPULSE_SPEED_BARS:   int   = 4     # окно для оценки скорости (4 бара = 1ч на 15m)
IMPULSE_RANGE_MAX:    float = 5.0   # daily_range < 5% — ещё не поздно входить
IMPULSE_RSI_LO:       float = 50.0  # RSI выше нейтрали
IMPULSE_RSI_HI:       float = 72.0  # RSI ещё не перегрет
IMPULSE_CROSS_BARS:   int   = 6     # окно поиска пересечения EMA20>EMA50

# Сканирование IMPULSE: независимая фоновая задача
IMPULSE_SCAN_SEC:     int   = 900   # каждые 15 минут (= 1 бар на 15m)
IMPULSE_COOLDOWN_SEC: int   = 3600  # не повторять сигнал по одной монете чаще 1ч


# ── Exit conditions ───────────────────────────────────────────────────────────
RSI_OVERBOUGHT = 85.0
ADX_DROP_RATIO = 0.75
TREND_MACD_REL_MIN: float = 0.00005  # 0.005% от цены — защита от почти нулевого MACD
MODE_AWARE_EXITS_ENABLED: bool = True
EXIT_AGGRESSIVE_MODES: tuple = ("breakout", "retest")
EXIT_PATIENT_MODES: tuple = ("trend", "strong_trend", "alignment")
EXIT_SEMI_PATIENT_MODES: tuple = ("impulse_speed",)
MIN_BARS_BEFORE_ADX_EXIT: int = 5
SEMI_PATIENT_MIN_BARS_BEFORE_ADX_EXIT: int = 4
PATIENT_SLOPE_CONFIRM_BARS: int = 2
SEMI_PATIENT_SLOPE_CONFIRM_BARS: int = 2

# ── Forward accuracy ─────────────────────────────────────────────────────────
FORWARD_BARS = [3, 5, 10]
MIN_ACCURACY = 60.0
MIN_SIGNALS  = 5  # общий минимум (не используется в дневном)

# ── Live monitoring ───────────────────────────────────────────────────────────
POLL_SEC      = 60
HISTORY_LIMIT = 300
LIVE_LIMIT    = 100
DISCOVERY_ENTRY_GRACE_BARS: int = 2
DISCOVERY_ENTRY_MAX_SLIPPAGE_PCT: float = 0.45
DISCOVERY_CATCHUP_SCORE_BONUS: float = 4.0
DISCOVERY_SCAN_SEC: int = 60  # ищем новые live-сигналы на каждом polling-цикле

# ── Часовой фильтр входов (ML-анализ ml_dataset.jsonl, 44K баров, 15.03.2026) ─
# EDA выявил устойчивый паттерн: 7 часов UTC стабильно убыточны (EV < -0.15%).
# Физическая интерпретация:
#   03 UTC: ночная Азия — малый объём, ложные движения
#   10-12 UTC: активная Европа + конец азиатской сессии — зона разворотов
#   13-15 UTC: пред-открытие NYSE (14:30) — крипта реагирует на ожидания
# Фильтр отключён: пользователь явно потребовал не блокировать входы по часу UTC.
ENTRY_BLOCK_HOURS: list = []
TIME_BLOCK_BYPASS_ENABLED: bool = True
TIME_BLOCK_BYPASS_SCORE_MIN: float = 60.0
TIME_BLOCK_BYPASS_VOL_X_MIN: float = 1.80
TIME_BLOCK_BYPASS_MODES: tuple = ("breakout", "retest")
TIME_BLOCK_BYPASS_1H_ENABLED: bool = True
TIME_BLOCK_BYPASS_1H_SCORE_MIN: float = 60.0
TIME_BLOCK_BYPASS_1H_VOL_X_MIN: float = 1.00
TIME_BLOCK_BYPASS_1H_MODES: tuple = ("alignment", "trend", "strong_trend", "impulse_speed")
TIME_BLOCK_BYPASS_1H_CONTINUATION_ENABLED: bool = True
TIME_BLOCK_BYPASS_1H_CONTINUATION_MODES: tuple = ("alignment", "trend", "strong_trend", "impulse_speed", "impulse")
TIME_BLOCK_BYPASS_1H_CONTINUATION_SCORE_BONUS: float = 8.0
TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MIN: float = 62.0
TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MAX: float = 78.0
TIME_BLOCK_BYPASS_1H_CONTINUATION_ADX_MIN: float = 16.0
TIME_BLOCK_BYPASS_1H_CONTINUATION_SLOPE_MIN: float = 0.08
TIME_BLOCK_BYPASS_1H_CONTINUATION_VOL_X_MIN: float = 1.00
TIME_BLOCK_BYPASS_1H_CONTINUATION_RANGE_MAX: float = 6.5
TIME_BLOCK_BYPASS_1H_PREBYPASS_ENABLED: bool = True
TIME_BLOCK_BYPASS_1H_PREBYPASS_MODES: tuple = ("alignment", "trend", "strong_trend", "impulse_speed", "impulse")
TIME_BLOCK_BYPASS_1H_PREBYPASS_CONFIRMATIONS: int = 2
TIME_BLOCK_BYPASS_1H_PREBYPASS_SCORE_MIN: float = 54.0
TIME_BLOCK_BYPASS_1H_PREBYPASS_VOL_X_MIN: float = 1.00
TIME_BLOCK_BYPASS_1H_PREBYPASS_PRICE_EDGE_MAX_PCT: float = 2.20
LATE_1H_CONTINUATION_GUARD_ENABLED: bool = True
LATE_1H_CONTINUATION_GUARD_MODES: tuple = ("trend", "alignment", "impulse_speed")
LATE_1H_CONTINUATION_GUARD_RSI_MIN: float = 70.0
LATE_1H_CONTINUATION_GUARD_PRICE_EDGE_MIN_PCT: float = 1.50
LATE_1H_CONTINUATION_GUARD_RANGE_MIN: float = 5.0
LATE_1H_CONTINUATION_GUARD_SCORE_MAX: float = 68.0
IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED: bool = True
IMPULSE_SPEED_1H_RSI_MAX: float = 70.0
IMPULSE_SPEED_1H_ADX_MIN: float = 22.0
IMPULSE_SPEED_1H_RANGE_MAX: float = 10.0
IMPULSE_SPEED_LATE_GUARD_ENABLED: bool = True
IMPULSE_SPEED_LATE_GUARD_15M_RSI_MIN: float = 70.0
IMPULSE_SPEED_LATE_GUARD_15M_RANGE_MIN: float = 6.0
IMPULSE_SPEED_LATE_GUARD_15M_PRICE_EDGE_MIN_PCT: float = 2.20
IMPULSE_SPEED_LATE_GUARD_15M_MACD_FADE_RATIO_MAX: float = 0.60
IMPULSE_SPEED_LATE_GUARD_15M_MACD_PEAK_LOOKBACK: int = 8
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RSI_MIN: float = 75.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RANGE_MIN: float = 10.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_PRICE_EDGE_MIN_PCT: float = 4.0
IMPULSE_SPEED_LATE_GUARD_1H_RSI_MIN: float = 68.0
IMPULSE_SPEED_LATE_GUARD_1H_RANGE_MIN: float = 8.0
IMPULSE_SPEED_LATE_GUARD_1H_PRICE_EDGE_MIN_PCT: float = 2.40
IMPULSE_SPEED_LATE_GUARD_1H_MACD_FADE_RATIO_MAX: float = 0.68
IMPULSE_SPEED_LATE_GUARD_1H_MACD_PEAK_LOOKBACK: int = 6
IMPULSE_SPEED_ROTATION_GUARD_ENABLED: bool = True
IMPULSE_SPEED_ROTATION_GUARD_15M_RSI_MIN: float = 76.0
IMPULSE_SPEED_ROTATION_GUARD_15M_RANGE_MIN: float = 5.0
IMPULSE_SPEED_ROTATION_GUARD_1H_RSI_MIN: float = 64.0
IMPULSE_SPEED_ROTATION_GUARD_1H_RANGE_MIN: float = 16.0
EARLY_1H_CONTINUATION_ENTRY_ENABLED: bool = True
EARLY_1H_CONTINUATION_ENTRY_RSI_MIN: float = 60.0
EARLY_1H_CONTINUATION_ENTRY_RSI_MAX: float = 78.0
EARLY_1H_CONTINUATION_ENTRY_ADX_MIN: float = 24.0
EARLY_1H_CONTINUATION_ENTRY_SLOPE_MIN: float = 0.08
EARLY_1H_CONTINUATION_ENTRY_VOL_X_MIN: float = 1.20
EARLY_1H_CONTINUATION_ENTRY_RANGE_MAX: float = 6.5
EARLY_1H_CONTINUATION_ENTRY_PRICE_EDGE_MAX_PCT: float = 1.60
EARLY_1H_CONTINUATION_ENTRY_ADX_SMA_TOLERANCE: float = 3.0
EARLY_1H_CONTINUATION_ENTRY_MODES: tuple = ("trend", "strong_trend", "impulse_speed")
FRESH_SIGNAL_SCORE_BONUS: float = 3.0
TIME_BLOCK_RETEST_GRACE_BARS: int = 4
TIME_BLOCK_RETEST_SCORE_BONUS: float = 5.0

# ── Новые фильтры входа ───────────────────────────────────────────────────────
# Макс. рост от минимума последних 96 баров (24ч на 15m)
# Если монета уже выросла больше — тренд устал, вход запрещён
DAILY_RANGE_MAX: float = 7.0

# Максимум баров в позиции — если за это время нет выхода по условиям,
# выходим принудительно. На 15m: 16 баров = 4 часа
MAX_HOLD_BARS: int = 16       # fallback и для 1h
MAX_HOLD_BARS_15M: int = 48   # 15m: 12 часов — ловим дневные тренды

# Минимум оценённых сигналов сегодня для подтверждения стратегии
# Если меньше — монета не проходит в мониторинг
TODAY_MIN_SIGNALS: int = 2
FORWARD_TEST_WINDOW_HOURS: int = 24  # скользящее окно форвард-теста (вместо UTC-полночь)

# Минимальная точность T+3 для подтверждения (было 50%, стало строже)
TODAY_T3_MIN: float = 60.0

# Минимальная точность T+10 — если ниже этого, монета опасна на длинном горизонте
TODAY_T10_MIN: float = 40.0

# Интервал авто-реанализа в секундах (0 = выключен)
# 7200 = каждые 2 часа бот сам пересчитывает список монет
AUTO_REANALYZE_SEC: int = 7200
POSITIONS_FILE: str = "positions.json"

ATR_TRAIL_K: float = 2.0   # множитель ATR для трейлинг-стопа
MACDWARN_BARS: int = 3     # баров подряд MACD hist падает → предупреждение о развороте

# ── П1: ATR-трейл и лимит баров по режиму входа ──────────────────────────────
ATR_TRAIL_K_STRONG:    float = 2.5   # BUY strong_trend (ADX высокий + объём)
ATR_TRAIL_K_RETEST:    float = 1.5   # RETEST (откат к EMA20 — tighter trail режет плохие возвраты)
ATR_TRAIL_K_BREAKOUT:  float = 1.5   # BREAKOUT (пробой флэта — быстрый выход)
MAX_HOLD_BARS_RETEST:  int   = 10    # RETEST: 10 баров × 15m = 2.5 часа
MAX_HOLD_BARS_BREAKOUT:int   = 6     # BREAKOUT: 6 баров × 15m = 1.5 часа
CONTINUATION_PROFIT_LOCK_ENABLED: bool = True
CONTINUATION_PROFIT_LOCK_MODES: tuple = ("trend", "alignment", "strong_trend", "impulse_speed", "impulse")
CONTINUATION_PROFIT_LOCK_TF: tuple = ("1h",)
CONTINUATION_PROFIT_LOCK_ENTRY_RSI_MIN: float = 70.0
CONTINUATION_PROFIT_LOCK_MIN_BARS: int = 3
CONTINUATION_PROFIT_LOCK_ACTIVATE_PNL_PCT: float = 0.20
CONTINUATION_PROFIT_LOCK_CONTINUATION_PNL_PCT: float = 0.60
CONTINUATION_PROFIT_LOCK_TRAIL_K: float = 1.4
CONTINUATION_PROFIT_LOCK_FLOOR_PCT: float = 0.10
CONTINUATION_MICRO_EXIT_ENABLED: bool = True
CONTINUATION_MICRO_EXIT_TF: tuple = ("1h",)
CONTINUATION_MICRO_EXIT_MODES: tuple = ("trend", "alignment", "strong_trend", "impulse_speed", "impulse")
CONTINUATION_MICRO_EXIT_MIN_BARS: int = 3
CONTINUATION_MICRO_EXIT_MACD_NEG_BARS: int = 4
CONTINUATION_MICRO_EXIT_RSI_MAX: float = 72.0
CONTINUATION_MICRO_EXIT_PRICE_EDGE_MAX_PCT: float = 0.50
SHORT_MODE_PROFIT_LOCK_ENABLED: bool = False
SHORT_MODE_PROFIT_LOCK_TF: tuple = ("15m",)
SHORT_MODE_PROFIT_LOCK_MODES: tuple = ("breakout", "retest")
SHORT_MODE_PROFIT_LOCK_MIN_BARS: int = 2
SHORT_MODE_PROFIT_LOCK_ACTIVATE_PNL_PCT: float = 0.30
SHORT_MODE_PROFIT_LOCK_TRAIL_K: float = 1.2
SHORT_MODE_PROFIT_LOCK_FLOOR_PCT: float = 0.05

# ── П5: Trend Day (BTC 1h EMA50) ─────────────────────────────────────────────
# В бычий день расширяем допустимые пороги
BULL_DAY_RANGE_MAX: float = 14.0  # DAILY_RANGE_MAX при бычьем дне
BULL_DAY_RSI_HI:    float = 75.0  # RSI_BUY_HI при бычьем дне

# ── ADX SMA bypass (уже использовался через getattr) ─────────────────────────
ADX_SMA_BYPASS: float = 35.0  # ADX ≥ этого → плато сильного тренда, bypass

# ── П3: Cooldown (уже использовался через getattr) ───────────────────────────
COOLDOWN_BARS: int = 8  # баров тишины после выхода (8 × 15m = 2 часа)

# ── RETEST: откат к EMA20 в существующем тренде ──────────────────────────────
RETEST_LOOKBACK:    int   = 12    # баров назад — проверяем что тренд был
RETEST_TOUCH_BARS:  int   = 5     # окно поиска касания EMA20
RETEST_RSI_MAX:     float = 65.0  # RSI на ретесте должен быть ниже
RETEST_VOL_MIN:     float = 1.0   # ретест без хотя бы среднего объёма слишком часто срывается

# ── BREAKOUT: пробой флэта с объёмом ─────────────────────────────────────────
BREAKOUT_FLAT_BARS:    int   = 8    # баров флэта перед пробоем
BREAKOUT_FLAT_MAX_PCT: float = 2.0  # макс диапазон флэта (%)
BREAKOUT_VOL_MIN:      float = 2.8  # vol_x на пробое — нужен более убедительный спрос
BREAKOUT_RANGE_MAX:    float = 3.0  # daily_range — движение только началось
BREAKOUT_SLOPE_MIN:    float = 0.08 # EMA20 уже должна смотреть вверх
BREAKOUT_RSI_MAX:      float = 76.0 # допускаем горячий breakout, но не экстремально поздний
BREAKOUT_ADX_MIN:      float = 18.0 # чистый объёмный всплеск без трендовой опоры не берём

# ── IMPULSE: детектор начала импульса (до подтверждения ADX) ─────────────────
# Откалиброван по реальным данным 04.03.2026:
#   ETH 15:15 — r1=+2.37% r3=+3.50% RSI=78.8 vol×2.63  ← поймал за 1 бар до BUY
#   SOL 15:15 — r1=+2.11% r3=+3.27% RSI=76.4 vol×2.35
#   XRP 15:15 — r1=+2.05% r3=+2.69% RSI=79.5 vol×3.77
#   XLMUSDT  — r1=+1.54% r3=+2.13% RSI=72.6 vol×1.73
IMPULSE_R1_MIN:        float = 1.5   # мин рост текущего бара (%)
IMPULSE_R3_MIN:        float = 2.0   # мин рост за 3 бара (%)
IMPULSE_VOL_MIN:       float = 1.5   # мин объём кратный среднему
IMPULSE_BODY_MIN:      float = 0.5   # мин тело свечи (%) — реальное движение
IMPULSE_RSI_LO:        float = 45.0  # RSI нижняя граница
IMPULSE_RSI_HI:        float = 80.0  # RSI верхняя (80 — ловим импульс при разгоне)
IMPULSE_COOLDOWN_BARS: int   = 8     # баров между сигналами одной монеты

# ── TREND_SURGE: детектор начала устойчивого тренда ──────────────────────────
# Ловит момент когда тренд «включается» — slope ускоряется + MACD растёт.
# Не зависит от ADX и форвард-теста. Кулдаун 5 часов — один сигнал на тренд.
# Примеры: JASMY 09.03 03:00 UTC (+8% за 12ч), BONK 09.03 12:00 UTC.
SURGE_SLOPE_MIN:     float = 0.30   # slope EMA20 (%) — для боевого входа берём только сильное ускорение
SURGE_VOL_MIN:       float = 1.5    # объём выше среднего
SURGE_RSI_LO:        float = 50.0   # RSI в зоне импульса
SURGE_RSI_HI:        float = 80.0   # не перегрет
SURGE_COOLDOWN_BARS: int   = 20     # 20 × 15m = 5 часов между сигналами одной монеты

# ── ALIGNMENT: плавный бычий тренд без ADX ────────────────────────────────────
# Ловит медленные альт-тренды где ADX не успевает подтвердить за 28+ баров,
# но структура устойчиво бычья (пример: CHZ 08.03 09:00-18:00 = +8% за 9 часов).
ALIGNMENT_SLOPE_MIN:  float = 0.05  # мягче чем BUY (0.10) — тренд может быть плавным
ALIGNMENT_VOL_MIN:    float = 0.8   # не нужен спайк, достаточно любой активности
ALIGNMENT_RSI_LO:     float = 45.0  # RSI выше нейтрали
ALIGNMENT_RSI_HI:     float = 82.0  # медленный тренд — RSI разогревается постепенно
ALIGNMENT_RANGE_MAX:  float = 9.0   # ↓ с 18% — не входить в alignment когда уже +9%+ за день (был TAO 16.12%)
ALIGNMENT_PRICE_EDGE_MAX_PCT: float = 2.0  # общий guard: alignment не должен быть слишком далеко от EMA20
ALIGNMENT_1H_PRICE_EDGE_MAX_PCT: float = 1.5  # на 1h режем поздние догоняющие alignment-входы
ALIGNMENT_MACD_BARS:  int   = 5     # ↑ с 3 — иссякший импульс SEI (hist≈0.0000) не прошёл бы 5 баров
ALIGNMENT_MACD_REL_MIN: float = 0.0002  # мин. MACD hist как доля цены (0.02%) — SEI hist=0.0000 заблокирован
ALIGNMENT_LATE_RANGE_MIN: float = 6.5  # при уже большом ходе за день late-alignment должен быть строже
ALIGNMENT_MACD_PEAK_LOOKBACK: int = 8  # сколько последних баров сравниваем с локальным пиком MACD
ALIGNMENT_MACD_PEAK_RATIO_MIN: float = 0.45  # текущий MACD hist должен держать ≥45% локального пика
ALIGNMENT_NONBULL_ADX_MIN: int = 20
ALIGNMENT_NONBULL_REQUIRE_ABOVE_EMA200: bool = True
ALIGNMENT_NONBULL_VOL_MIN: float = 1.0
ALIGNMENT_NONBULL_RSI_LO: float = 50.0
ALIGNMENT_NONBULL_RSI_HI: float = 66.0

# ── Fast loss exit: ускоренный выход после первого закрытия ниже EMA20 ─────────────
# Ночная проблема 23.03: часть 15m сделок выходила только после 2-го close ниже EMA20,
# хотя уже на первом закрытии под EMA20 сделка была в минусе и импульс слабел.
FAST_LOSS_EMA_EXIT_ENABLED: bool = True
FAST_LOSS_EMA_EXIT_TF: tuple = ("15m", "1h")
FAST_LOSS_EMA_EXIT_MODES: tuple = ("retest", "breakout", "alignment", "impulse")
FAST_LOSS_EMA_EXIT_MIN_BARS: int = 1
FAST_LOSS_EMA_EXIT_PNL_MAX: float = 0.0
FAST_LOSS_EMA_EXIT_RSI_MAX: float = 70.0

# Time exit should not cut an active uptrend. If the trend structure is still healthy,
# keep the position beyond the nominal bar limit and let other exits manage it.
TIME_EXIT_TREND_CONTINUATION_ENABLED: bool = True
TIME_EXIT_CONTINUE_CLOSE_ABOVE_EMA20: bool = True
TIME_EXIT_CONTINUE_SLOPE_MIN: float = 0.0
TIME_EXIT_CONTINUE_RSI_MIN: float = 50.0
TIME_EXIT_CONTINUE_MACD_HIST_MIN: float = 0.0

# ── MTF (Multi-TimeFrame) фильтр для 1h сигналов ─────────────────────────────
# Когда бот входит по 1h сигналу, 15м индикаторы могут уже показывать коррекцию
# (пример: ETH 13.03 — 1h вход в 16:00, но пик 14:15, MACD=-6.78, RSI=41 на 15м).
# Перед входом по 1H проверяем последний закрытый 15м бар:
#   • MTF_MACD_POSITIVE: True  → 15м MACD hist должен быть > 0
#   • MTF_RSI_MIN: 42.0        → 15м RSI допускаем чуть глубже в pullback
#   • MTF_MACD_SOFT_FLOOR_REL  → допускаем только очень неглубокий минус MACD,
#                                если он уже разворачивается вверх
#   • MTF_RSI_SOFT_MIN         → для soft-pass RSI должен быть чуть сильнее обычного
# При tf="15m" фильтр не применяется (не нужен — уже на нужном ТФ).
MTF_ENABLED:      bool  = True   # глобальный выключатель
MTF_MACD_POSITIVE: bool = True   # 15м MACD hist > 0 для входа по 1h
MTF_RSI_MIN:      float = 42.0  # 15м RSI минимум для входа по 1h
MTF_MACD_SOFT_FLOOR_REL: float = -0.00060  # допускаем до -0.06% цены
MTF_MACD_HARD_FLOOR_REL: float = -0.00120  # hard-block только при глубокой 15м коррекции
MTF_RSI_SOFT_MIN:       float = 45.0       # soft-pass допускает чуть более глубокий pullback
MTF_REQUIRE_MACD_RISING: bool = False      # soft-pass не требует обязательного улучшения MACD vs prev bar
MTF_1H_CONTINUATION_RELAX_ENABLED: bool = True
MTF_RSI_HARD_MIN:       float = 38.0
MTF_SOFT_PASS_PENALTY:  float = 5.0
MTF_1H_CONTINUATION_RELAX_MODES: tuple = ("alignment", "trend", "strong_trend", "impulse_speed", "impulse")
MTF_1H_CONTINUATION_RELAX_SCORE_MIN: float = 60.0
MTF_1H_CONTINUATION_RELAX_ADX_MIN: float = 18.0
MTF_1H_CONTINUATION_RELAX_SLOPE_MIN: float = 0.12
MTF_1H_CONTINUATION_RELAX_RSI_MIN: float = 52.0
MTF_1H_CONTINUATION_RELAX_RSI_MAX: float = 80.0
MTF_1H_CONTINUATION_RELAX_VOL_X_MIN: float = 0.65
MTF_1H_CONTINUATION_RELAX_RANGE_MAX: float = 12.0
MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL: float = -0.00120
MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN: float = 42.0
MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT: float = 0.45
MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING: bool = False

# ── IMPULSE: поднятая верхняя граница RSI ─────────────────────────────────────
# ↑ с 80 до 82 — ловит разгоняющиеся альткоины типа XAI (+34%) раньше
IMPULSE_RSI_HI:       float = 82.0  # (переопределяет выше)

# ═══════════════════════════════════════════════════════════════════════════════
# НОВЫЕ ПАРАМЕТРЫ v2: Начало/окончание тренда — расширенная сигнализация
# ═══════════════════════════════════════════════════════════════════════════════

# ── A. Ускорение наклона EMA (slope acceleration) ─────────────────────────────
# Ловит момент когда EMA20 начинает разгоняться — опережает BUY на 1-3 бара.
# slope[i] - slope[i-3] > SLOPE_ACCEL_MIN → тренд набирает силу прямо сейчас.
SLOPE_ACCEL_MIN:   float = 0.05   # ускорение slope (%) за 3 бара
SLOPE_ACCEL_BARS:  int   = 3      # окно оценки ускорения

# ── B. Squeeze Breakout (пробой сжатия ATR) ───────────────────────────────────
# ATR < 50% своей N-баровой средней = сжатие (накопление перед движением).
# ATR вырос в 1.8× от минимума сжатия = пробой.
# Идея: боковик сжимает пружину — выход из боковика взрывной.
SQUEEZE_LOOKBACK:       int   = 20    # баров для расчёта средней ATR
ATR_SQUEEZE_RATIO:      float = 0.5   # ATR < 50% от SMA(ATR,20) = сжатие
ATR_EXPANSION_MULT:     float = 1.8   # ATR вырос в 1.8× от дна сжатия = пробой
SQUEEZE_MIN_BARS:       int   = 5     # минимум баров в сжатии перед пробоем

# ── C. RSI Дивергенция (сигнал окончания тренда) ──────────────────────────────
# Цена делает новый maximum → RSI не подтверждает → скрытое ослабление.
# При обнаружении: ужесточить стоп (ATR_K * RSI_DIV_TRAIL_MULT), не открывать BUY.
RSI_DIV_LOOKBACK:       int   = 10    # баров назад для поиска предыдущего максимума
RSI_DIV_PRICE_MARGIN:   float = 0.001 # цена должна быть выше на >0.1% (фильтр шума)
RSI_DIV_TRAIL_MULT:     float = 0.6   # множитель ATR при дивергенции (2.0 → 1.2)

# ── D. Volume Exhaustion (истощение объёма — конец тренда) ────────────────────
# Цена растёт N баров подряд, но объём каждый бар ниже предыдущего.
# Сильный сигнал разворота — покупатели заканчиваются.
VOL_EXHAUST_BARS:       int   = 5     # баров убывающего объёма при росте цены
VOL_EXHAUST_PRICE_MIN:  float = 0.5   # минимальный рост цены (%) за эти N баров

# ── E. EMA Fan Collapse (схлопывание веера — конец тренда) ────────────────────
# В тренде: EMA20 >> EMA50 >> EMA200, расстояния растут.
# Разворот: spread EMA20-EMA50 уменьшился на SPREAD_DECAY от максимума.
EMA_FAN_LOOKBACK:       int   = 8     # баров назад для поиска максимума spread
EMA_FAN_DECAY_THRESHOLD: float = 0.30 # spread упал на 30% от максимума → предупреждение

# ── F. Market Regime (режим рынка) ────────────────────────────────────────────
# Явные режимы рынка меняют пороги для всех сигналов.
# BULL_TREND:    BTC > EMA50 + ADX > 25 → мягче RSI, range; строже vol
# CONSOLIDATION: ADX < 20 → строже всё, ждём пробоя
# RECOVERY:      BTC пробивает EMA50 снизу → агрессивный вход
# BEAR_TREND:    BTC < EMA50 + ADX > 25 → только ретесты, запрет новых BUY

# Параметры по режимам: {режим: {параметр: значение}}
REGIME_PARAMS: dict = {
    "bull_trend": {
        "rsi_hi":       75.0,
        "vol_mult":     1.1,
        "range_max":    10.0,
        "adx_min":      18.0,
        "slope_min":    0.08,
    },
    "consolidation": {
        "rsi_hi":       65.0,
        "vol_mult":     1.5,
        "range_max":    5.0,
        "adx_min":      22.0,
        "slope_min":    0.12,
    },
    "recovery": {
        "rsi_hi":       70.0,
        "vol_mult":     1.2,
        "range_max":    8.0,
        "adx_min":      18.0,
        "slope_min":    0.08,
    },
    "bear_trend": {
        "rsi_hi":       60.0,
        "vol_mult":     2.0,
        "range_max":    4.0,
        "adx_min":      25.0,
        "slope_min":    0.15,
    },
    "neutral": {
        # Базовые значения — не перезаписываем config
        "rsi_hi":       None,
        "vol_mult":     None,
        "range_max":    None,
        "adx_min":      None,
        "slope_min":    None,
    },
}

# Пороги для определения режима по BTC 1h
REGIME_BTC_ADX_TREND:   float = 22.0   # ADX >= этого → тренд (bull или bear)
REGIME_BTC_ADX_FLAT:    float = 18.0   # ADX < этого → консолидация
REGIME_BTC_RECOVERY_SLOPE: float = 0.05  # slope EMA50 при пробое снизу вверх

# ── G. Dynamic Range Max (адаптивный порог по волатильности монеты) ───────────
# Вместо фиксированного 7% — порог пропорционален исторической волатильности монеты.
# Монеты типа XAI (дневной диапазон 15%+) получают более широкий порог.
# Монеты типа BTC (диапазон 3%) — более узкий.
DYNAMIC_RANGE_ENABLED:   bool  = True
DYNAMIC_RANGE_REF_PCT:   float = 5.0   # эталонный дневной диапазон (нормировка)
DYNAMIC_RANGE_HIST_BARS: int   = 96 * 14  # 14 дней на 15m для расчёта avg_daily_range
DYNAMIC_RANGE_MIN:       float = 3.0   # нижний предел (защита от слишком узкого порога)
DYNAMIC_RANGE_MAX_CAP:   float = 25.0  # верхний предел (защита от бесконечного порога)

# ── Runtime overrides (устанавливаются динамически в market_scan / strategy) ──
_current_regime:         str   = "neutral"
_regime_params_active:   dict  = {}     # активные параметры текущего режима

# ── H. EMA Cross — ранний сигнал пробоя EMA20 снизу вверх ────────────────────
# Ловит момент когда цена пробивает EMA20 с объёмом ДО подтверждения slope/ADX.
# Типичный выигрыш: 3-5 баров (45-75 минут) раньше стандартного сигнала.
#
# Паттерн: close[i-1] < ema20[i-1]  AND  close[i] >= ema20[i]  (пробой)
#          vol_x[i] >= CROSS_VOL_MIN                             (объём подтверждает)
#          ema50_slope >= CROSS_EMA50_SLOPE_MIN                  (EMA50 не падает)
#          RSI в диапазоне [CROSS_RSI_LO, CROSS_RSI_HI]         (не перекуплен/перепродан)
#          daily_range_pct <= CROSS_RANGE_MAX                    (нет уже разогнанного хода)
#          close > ema200 (выше долгосрочной поддержки)          (опционально)

CROSS_VOL_MIN:       float = 1.2    # мин объём на баре пробоя (ниже стандартного 1.3)
CROSS_EMA50_SLOPE_MIN: float = -0.40 # EMA50 не должна падать сильнее этого % (за 3 бара)
CROSS_RSI_LO:        float = 38.0   # нижняя RSI (шире стандартного 45)
CROSS_RSI_HI:        float = 72.0   # верхняя RSI (как стандарт, выше = уже разогнан)
CROSS_RANGE_MAX:     float = 6.0    # макс дневной диапазон (фильтр уже разогнанных)
CROSS_LOOKBACK:      int   = 3      # баров назад для проверки что было ниже EMA20
CROSS_MACD_FILTER:   bool  = True   # требовать MACD hist >= 0 на баре пробоя
CROSS_COOLDOWN_BARS: int   = 6      # баров между EMA_CROSS сигналами одной монеты
CROSS_CONFIRM_BARS:  int   = 2      # макс баров с момента пробоя (не слать старое)

# ── Runtime: EMA_CROSS ───────────────────────────────────────────────────────
_last_cross_ts:      dict  = {}     # sym → timestamp последнего CROSS-сигнала

# ── J. Exit Guards ────────────────────────────────────────────────────────────
# Защита от немедленного ложного выхода на баре входа.
#
# Проблема (AR 11.03.2026): RSI дивергенция существовала ДО входа → при 0 барах
# check_exit_conditions(WEAK) давал немедленный выход с +0.00%.
# Монета при этом росла +1.14% на 1h.
#
# Решение: WEAK сигналы (RSI div, vol exhaustion, EMA fan collapse) игнорируются
# первые MIN_WEAK_EXIT_BARS баров. Hard exits (ATR-трейл, время) — без изменений.
MIN_WEAK_EXIT_BARS: int = 2   # 2 бара = 30 мин на 15m, 2 часа на 1h
MIN_WEAK_EXIT_BARS_RETEST: int = 2
MIN_WEAK_EXIT_BARS_BREAKOUT: int = 2
MIN_WEAK_EXIT_BARS_TREND: int = 2
MIN_WEAK_EXIT_BARS_IMPULSE_SPEED: int = 4

# Лимиты открытых позиций — защита от "все альты двигаются пачкой".
# 11.03.2026: 12 монет вошли одновременно = 1 рыночное движение, не 12 независимых сигналов.

# Максимум одновременно открытых позиций (все монеты вместе)
LOCAL_TIMEZONE: str = "Europe/Budapest"

MAX_OPEN_POSITIONS: int = 10
# Резервируем один слот под самые свежие сильные импульсные входы,
# чтобы переполненный портфель не душил новые breakout/impulse-сетапы.
FRESH_SIGNAL_RESERVED_SLOTS: int = 1
FRESH_SIGNAL_PRIORITY_MODES: tuple = ("breakout", "retest", "impulse_speed")
TOP_MOVER_SCORE_ENABLED: bool = True
TOP_MOVER_MIN_DAY_CHANGE_PCT: float = 1.5
TOP_MOVER_DAY_CHANGE_CAP_PCT: float = 8.0
TOP_MOVER_SCORE_WEIGHT: float = 1.6
FORECAST_RETURN_SCORE_WEIGHT: float = 18.0
FORECAST_RETURN_NEGATIVE_WEIGHT: float = 10.0
MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS: int = 0
PORTFOLIO_REPLACE_ENABLED: bool = False
PORTFOLIO_REPLACE_MIN_DELTA: float = 8.0
PORTFOLIO_REPLACE_FRESH_MIN_DELTA: float = 6.0
PORTFOLIO_REPLACE_MIN_BARS: int = 2
PORTFOLIO_REPLACE_PROFIT_PROTECT_PCT: float = 0.35
PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT: float = 0.80
PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA: float = 12.0
PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA: float = 10.0
PORTFOLIO_REPLACE_ADX_PROTECT_MIN: float = 30.0
PORTFOLIO_REPLACE_TREND_GRACE_BARS: int = 5
WEAK_REENTRY_COOLDOWN_BARS: int = 8
ENTRY_SCORE_MIN_ENABLED: bool = True
ENTRY_SCORE_MIN_15M: float = 48.0
ENTRY_SCORE_MIN_1H: float = 56.0
ENTRY_SCORE_BORDERLINE_BYPASS_ENABLED: bool = True
ENTRY_SCORE_BORDERLINE_ALLOW_1H: bool = False
ENTRY_SCORE_BORDERLINE_MODES: tuple[str, ...] = ("breakout", "retest", "impulse_speed")
ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_15M: float = 6.0
ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_1H: float = 4.0
ENTRY_SCORE_BORDERLINE_ADX_MIN: float = 28.0
ENTRY_SCORE_BORDERLINE_SLOPE_MIN: float = 0.35
ENTRY_SCORE_BORDERLINE_VOL_MIN: float = 1.20
ENTRY_SCORE_BORDERLINE_DAILY_RANGE_MAX: float = 6.5
ENTRY_SCORE_BORDERLINE_PRICE_EDGE_MAX_PCT: float = 2.80
ENTRY_SCORE_BORDERLINE_RSI_MIN: float = 52.0
ENTRY_SCORE_BORDERLINE_RSI_MAX: float = 74.5

# Максимум в одной "группе" монет — корреляционный лимит
# Монеты одной группы двигаются синхронно (L1, AI, GameFi и т.д.)
MAX_POSITIONS_PER_GROUP: int = 2

# Группы монет по категориям (простая эвристика по суффиксу/имени)
# Ключ = название группы, значение = список монет
COIN_GROUPS: dict = {
    "L1_major":  ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
                  "SUIUSDT", "APTUSDT", "NEARUSDT", "ATOMUSDT"],
    "L2_eth":    ["ARBUSDT", "OPUSDT", "STRKUSDT", "MATICUSDT", "ZKUSDT"],
    "AI_data":   ["RENDERUSDT", "FETUSDT", "AGIUSDT", "WLDUSDT", "TAOUSDT",
                  "ARKMUSDT", "OCEANUSDT", "GRTUSDT", "GLMUSDT"],
    "GameFi":    ["AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "IMXUSDT",
                  "YGGUSDT"],
    "DeFi_amm":  ["UNIUSDT", "SUSHIUSDT", "CRVUSDT", "AAVEUSDT", "MKRUSDT",
                  "COMPUSDT", "LDOUSDT"],
    "Storage":   ["FILUSDT", "ARUSDT", "STXUSDT"],
    "Meme":      ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "WIFUSDT",
                  "FLOKIUSDT"],
    "Oracle":    ["LINKUSDT", "BANDUSDT", "PYTHUSDT"],
    "Interop":   ["DOTUSDT", "COSUSDT", "AXLUSDT"],
    "Infra":     ["TIAUSDT", "ORDIUSDT", "AEVOUSDT", "INJUSDT"],
}

# Expectancy фильтр в analyze_coin
EV_MIN_PCT:      float = 0.05  # ↑ с 0.0 — отрицательный EV у TAO и SEI при входе
EV_MIN_SAMPLES:  int   = 5     # ↑ с 3 — 3 сэмпла (TAO T+3=20%=1/5) не статистика
BLOCK_LOG_INTERVAL_BARS: int = 4   # логировать блокировку не чаще раза в 4 бара

# ── K. Strong Trend Classification Guards ────────────────────────────────────
# Защита от ложных "strong_trend" на флэте (SNX 12.03.2026: ADX=29.9 при EMA20≈EMA50).
#
# Проблема: ADX пересёк порог 28 на объёмном импульсе, но цена весь день во флэте.
# EMA20=0.3156, EMA50=0.3130 — разрыв 0.08%, MACD≈0.
# При реальном сильном тренде (AR): EMA20 > EMA50 на 3%+, EMA50 растёт.
#
# Решение: strong_trend требует ВСЕ три условия:
#   1. ADX ≥ STRONG_ADX_MIN + Vol× ≥ STRONG_VOL_MIN
#   2. EMA20 выше EMA50 на ≥ STRONG_EMA_SEP_MIN %
#   3. EMA50 наклонена вверх на ≥ STRONG_EMA50_SLOPE_MIN % за 3 бара
STRONG_EMA_SEP_MIN:    float = 0.9   # мин. разрыв EMA20/EMA50 в % от цены
STRONG_EMA50_SLOPE_MIN: float = 0.05  # мин. рост EMA50 за 3 бара (%)

# ── L. Signal Quality Guards ─────────────────────────────────────────────────
# Защита от слабых/ложных сигналов выявленных 12.03.2026.

# RETEST: минимальный отскок от EMA20.
# LTC-баг: close=54.97 EMA20=54.9694 → зазор 0.001% → ложный ретест.
# При реальном отскоке цена уходит хотя бы на 0.05% выше EMA20.
RETEST_MIN_BOUNCE_PCT: float = 0.05

# ALIGNMENT: нижний порог ADX.
# ICP-баг: ADX=13.2 — явный флэт. Alignment ловит медленные тренды (ADX лагует),
# но 13 это уже шум. Порог 15 отсекает флэт не трогая слабые бычьи тренды.
ALIGNMENT_ADX_MIN: int = 15

# ALIGNMENT: минимальный разрыв EMA20/EMA50 в %.
# MANA-баг: EMA20≈EMA50 (разрыв <0.1%) при MACD≈0 → сигнал на горизонтальном рынке.
ALIGNMENT_EMA_FAN_MIN: float = 0.1

# ── L. Alignment & Retest Quality Guards ─────────────────────────────────────
# Защита от слабых сигналов на флэте/горизонтали.
#
# MANA 12.03.2026: EMA20=0.0928 ≈ EMA50=0.0927 → alignment на боковике.
# LTC  12.03.2026: slope=+0.08%, MACD=-0.02 → retest на горизонтальной EMA.
#
# ALIGNMENT: добавлен минимальный разрыв EMA20/EMA50 (мягче чем для strong_trend).
ALIGNMENT_EMA_SEP_MIN: float = 0.05  # ↓ с 0.3% — разрешаем нарождающиеся тренды
# где EMA20 только что пересекла EMA50 (TIA 15.03.2026: sep=0.085%).
# 0.3% блокировал именно такие входы. SEI-паттерн (MACD=0) по-прежнему
# блокируется через ALIGNMENT_MACD_REL_MIN=0.0002. Бэктест: Precision 75→80%.
#
# RETEST: поднят минимальный slope (>0 недостаточно), добавлен MACD guard.
RETEST_SLOPE_MIN:      float = 0.10  # мин. наклон EMA20 % за бар
# RETEST_MACD guard встроен напрямую (MACD hist >= 0 обязательно)
ENTRY_QUALITY_RECHECK_ENABLED: bool = True
ENTRY_QUALITY_RECHECK_MODES: tuple = ("alignment",)
ENTRY_QUALITY_RECHECK_MAX_BARS: int = 8
ENTRY_QUALITY_RECHECK_REASON_MARKERS: tuple = ("late alignment",)
PROFITABLE_WEAK_EXIT_SKIP_COOLDOWN: bool = True
PROFITABLE_WEAK_EXIT_COOLDOWN_PNL_MIN: float = 0.0
PROFITABLE_WEAK_EXIT_TF: tuple = ("1h",)
