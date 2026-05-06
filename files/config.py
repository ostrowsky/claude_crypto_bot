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
# Hard block entries outside the profitable ml_proba zone (0.55-0.65 from backtest analysis)
ML_GENERAL_HARD_BLOCK_ENABLED: bool = True
# 19.04.2026: lowered 0.35 -> 0.28 after ml_signal_model retrain shifted
# probability distribution downward. Backtest 14d/481 trades: bucket [0.30,0.35)
# was the BEST (n=24, win=66.7%, avg=+0.67%, sum=+16%), and blocking it was
# leaving money on the table (impulse_speed win=73%, trend win=80%, impulse
# win=100% in the ml<0.35 bucket). Live effect: BTC blocked 34x/90min today
# with ml proba 0.26-0.32 while clearly trending up.
ML_GENERAL_HARD_BLOCK_MIN: float = 0.28   # block if ml_proba < this
ML_GENERAL_HARD_BLOCK_MAX: float = 1.01   # disabled: high-confidence signals must never be blocked (was 0.65)
ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN: float = 0.22  # looser floor on bull days (was 0.28, kept gap from non-bull)
ML_ENABLE_TREND_NONBULL_FILTER: bool = True
ML_TREND_NONBULL_SEGMENT_KEY: str = "trend|nonbull"
ML_TREND_NONBULL_MIN_PROBA: float = 0.35
ML_TREND_NONBULL_LOW_PROBA_PENALTY: float = 6.0
ML_TREND_NONBULL_HARD_BLOCK: bool = False
CRITIC_DATASET_ENABLED: bool = True
# Max records from critic_dataset.jsonl for entry bandit training.
# Was hardcoded 8000 — dataset grew to 10K+ on 2026-04-15 and counter stuck.
# 25_000 gives ~6 months headroom at current growth rate (~500/day).
BANDIT_CRITIC_MAX_RECORDS: int = 25_000
ML_CANDIDATE_RANKER_RUNTIME_ENABLED: bool = True
ML_CANDIDATE_RANKER_MODEL_FILE: str = "ml_candidate_ranker.json"
ML_CANDIDATE_RANKER_NEUTRAL_PROBA: float = 0.40  # was 0.50 — lowered: most signals score 0.38-0.48, neutral at 0.50 penalised them all
ML_CANDIDATE_RANKER_SCORE_WEIGHT: float = 0.75
ML_CANDIDATE_RANKER_USE_FINAL_SCORE: bool = True
ML_CANDIDATE_RANKER_SCORE_CLIP: float = 2.0
ML_CANDIDATE_RANKER_SHADOW_ENABLED: bool = True
ML_CANDIDATE_RANKER_SHADOW_LOG_ALL: bool = False
ML_CANDIDATE_RANKER_VETO_ENABLED: bool = True
ML_CANDIDATE_RANKER_REQUIRE_CATBOOST: bool = True
ML_CANDIDATE_RANKER_VETO_TF: tuple[str, ...] = ("15m",)
ML_CANDIDATE_RANKER_VETO_MODES: tuple[str, ...] = ("trend",)
ML_CANDIDATE_RANKER_VETO_PROBA_MAX: float = 0.20
ML_CANDIDATE_RANKER_VETO_SCORE_MAX: float = 60.0
ML_CANDIDATE_RANKER_VETO_FORECAST_MAX: float = 0.25
ML_CANDIDATE_RANKER_HARD_VETO_ENABLED: bool = True
ML_CANDIDATE_RANKER_HARD_VETO_15M_TF: tuple[str, ...] = ("15m",)
ML_CANDIDATE_RANKER_HARD_VETO_15M_MODES: tuple[str, ...] = ("breakout", "retest", "impulse_speed", "impulse", "alignment", "trend", "strong_trend")
ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX: float = -2.50   # was -0.75 — too aggressive (TRU -7.73, SUI -0.81 blocked)
ML_CANDIDATE_RANKER_HARD_VETO_15M_TOP_GAINER_MAX: float = 0.20
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_ENABLED: bool = True
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_MODES: tuple[str, ...] = ("impulse_speed", "impulse")
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_FINAL_MAX: float = -2.00  # was -0.50 — BONK, METIS blocked as "weak impulse"
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_EV_MAX: float = -0.60
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_QUALITY_MAX: float = 0.50
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_TOP_GAINER_MAX: float = 0.28
ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_CAPTURE_MAX: float = 0.05
ML_CANDIDATE_RANKER_HARD_VETO_1H_TF: tuple[str, ...] = ("1h",)
ML_CANDIDATE_RANKER_HARD_VETO_1H_MODES: tuple[str, ...] = ("retest", "alignment", "trend", "strong_trend", "impulse_speed", "impulse")
ML_CANDIDATE_RANKER_HARD_VETO_1H_FINAL_MAX: float = -1.50
ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX: float = 0.25  # veto only if TG prob also low (mirrors 15m logic)
ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_ENABLED: bool = True
ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_MODES: tuple[str, ...] = ("retest",)
ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_FINAL_MAX: float = -1.20
ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_QUALITY_MAX: float = 0.35
ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_EV_MAX: float = -1.25
# ── 1h impulse / impulse_speed composite veto (added 2026-04-19) ──
# Diagnostic: generic 1h gate (final<=-1.50 AND tg<=0.25) blocked 0/190 trades
# in critic_dataset backtest — threshold below p10 of actual distribution.
# V3 config backtested on 190 1h-impulse 'take' rows: blocks 33% with avg_ret5
# -0.27% (kept avg improves +0.04%, kept win% +4.6pp). Mirrors 15m impulse veto.
# BNT reference (2026-04-19): final=-0.33 EV=-0.38 Q=0.56 TG=0.23 CAP=0.06 -> veto ✓
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_ENABLED: bool = True
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_MODES: tuple[str, ...] = ("impulse_speed", "impulse")
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_FINAL_MAX: float = -0.20
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_EV_MAX: float = -0.30
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_QUALITY_MAX: float = 0.58
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_TOP_GAINER_MAX: float = 0.25
ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_CAPTURE_MAX: float = 0.08
TOP_GAINER_CRITIC_ENABLED: bool = True
TOP_GAINER_CRITIC_TIMEZONE: str = "Europe/Budapest"
TOP_GAINER_CRITIC_TOP_N: int = 15
TOP_GAINER_CRITIC_MIN_QUOTE_VOLUME_24H: float = 1_000_000.0
RL_TELEGRAM_REPORTS_ENABLED: bool = True
RL_TRAIN_TELEGRAM_REPORTS_ENABLED: bool = True
TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED: bool = True
RL_WORKER_ENABLE_COLLECTOR: bool = False
BOT_ENABLE_DATA_COLLECTOR: bool = False

# ── RL: Contextual Bandit + Exit Policy ──────────────────────────────────────
BANDIT_ENABLED: bool = True               # adaptive trail_k/max_hold via LinUCB
BANDIT_ALPHA: float = 1.5                 # UCB exploration coefficient
BANDIT_TRAIL_K_MIN: float = 2.0          # floor: bandit cannot set trail_k below this (prevents 1.05 stops)

# 19.04.2026: opt-in to apply CMA-ES-optimized params from rl_params.json on
# bot startup. Disabled by default — applying ~22 overrides at once to a
# profitable bot is risky. Enable only after canary validation. The optimizer
# (offline_rl.py + rl_optimizer.py) keeps writing rl_params.json regardless.
RL_PARAMS_APPLY_ON_STARTUP: bool = False

# ── Mode enable/disable flags ─────────────────────────────────────────────────
# Data: 77k events. breakout=33% win/-0.053% avg, retest=37%/-0.139%, strong_trend=26%/-0.851%
# All three have high fast-exit rates (11-29%) and generate BUY→SELL flip-flop chains.
# impulse/impulse_speed/alignment/trend remain active (positive or near-zero avg PnL).
BREAKOUT_MODE_ENABLED:    bool = False   # 15.04.2026: disabled — 33% win, -0.053% avg PnL
RETEST_MODE_ENABLED:      bool = False   # 15.04.2026: disabled — 37% win, -0.139% avg PnL
STRONG_TREND_MODE_ENABLED: bool = False  # 15.04.2026: disabled — 26% win, -0.851% avg PnL
EXIT_RL_ENABLED: bool = True              # RL-based exit recommendations
EXIT_RL_THRESHOLD: float = 0.3            # Q-advantage needed to trigger exit
EXIT_RL_TIGHTEN_THRESHOLD: float = 0.15   # Q-advantage for tightening stop
OFFLINE_RL_MIN_NEW_TRADES: int = 10       # minimum new trades before offline training

RANKER_POSITION_CLEANUP_ENABLED: bool = True
RANKER_POSITION_CLEANUP_15M_MODES: tuple[str, ...] = ("impulse_speed", "impulse")
RANKER_POSITION_CLEANUP_15M_MIN_BARS: int = 8
RANKER_POSITION_CLEANUP_15M_FINAL_MAX: float = -0.50
RANKER_POSITION_CLEANUP_15M_EV_MAX: float = -0.60
RANKER_POSITION_CLEANUP_15M_TOP_GAINER_MAX: float = 0.28
RANKER_POSITION_CLEANUP_15M_CAPTURE_MAX: float = 0.05
RANKER_POSITION_CLEANUP_15M_REQUIRE_BELOW_EMA20: bool = True
RANKER_POSITION_CLEANUP_15M_PROACTIVE_ENABLED: bool = True
RANKER_POSITION_CLEANUP_15M_PROACTIVE_FINAL_MAX: float = -0.60
RANKER_POSITION_CLEANUP_15M_PROACTIVE_EV_MAX: float = -0.70
RANKER_POSITION_CLEANUP_15M_PROACTIVE_QUALITY_MAX: float = 0.50
RANKER_POSITION_CLEANUP_15M_PROACTIVE_TOP_GAINER_MAX: float = 0.26
RANKER_POSITION_CLEANUP_15M_PROACTIVE_CAPTURE_MAX: float = 0.05
RANKER_POSITION_CLEANUP_15M_PROACTIVE_PNL_MAX: float = 1.25
RANKER_POSITION_CLEANUP_1H_RETEST_ENABLED: bool = True
RANKER_POSITION_CLEANUP_1H_RETEST_MIN_BARS: int = 3
RANKER_POSITION_CLEANUP_1H_RETEST_FINAL_MAX: float = -1.50
RANKER_POSITION_CLEANUP_1H_RETEST_QUALITY_MAX: float = 0.35
RANKER_POSITION_CLEANUP_1H_RETEST_PNL_MAX: float = 0.50

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
FORWARD_BARS_15M_FAST_MODES: tuple = ("breakout", "retest", "impulse_speed")
FORWARD_BARS_15M_FAST = [2, 5, 7]
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
IMPULSE_SPEED_1H_RSI_MAX: float = 76.0       # was 70 — raised: RSI=70-76 is normal for trending coins
# NOTE: IMPULSE_SPEED_1H_ADX_MIN is defined below (line ~333) as the hard ADX floor.
# The 17.04 value previously here was a dead duplicate (overwritten by the later definition).
# Removed to avoid confusion — see IMPULSE_SPEED_1H_ADX_MIN at the hard-floor block.
IMPULSE_SPEED_1H_RANGE_MAX: float = 10.0
# Bull-day relaxed thresholds for 1h impulse guard
IMPULSE_SPEED_1H_RSI_MAX_BULL: float = 82.0  # was 76 — ORDI RSI=79 was blocked on bull day (15.04.2026)
IMPULSE_SPEED_1H_RANGE_MAX_BULL: float = 15.0
# Stateless late-stage guard (added 2026-04-18): replace daily_range counter,
# which depends on UTC session start, with ATR-extension from 1h EMA20.
# When True and ext_atr is provided, uses (price - ema20_1h) / atr_1h instead of
# daily_range. Metric is identical regardless of time of day and self-resets on
# consolidation — blocks only when price is truly extended from the mean.
# Flip to False to roll back to daily_range behavior.
IMPULSE_SPEED_1H_USE_EXT_ATR: bool = True
IMPULSE_SPEED_1H_EXT_ATR_MAX: float = 6.0        # 6 ATR above EMA20 = extended
IMPULSE_SPEED_1H_EXT_ATR_MAX_BULL: float = 9.0   # bull day = wider runway
# Hard ADX floor for impulse_speed signals (added 2026-04-18).
# CRVUSDT (ADX 17.5, -2.17%) and OXTUSDT (ADX 11.3) slipped through after
# 15m late-guard was disabled — they had volume spikes but no trend strength.
# 15m floor backtest 2026-04-20: take rows ADX[15-17)=+0.33% win71%, ADX[17-20)=+0.22% win59%
# → lowered 20→15; ADX<15 = avg -0.36% (correctly blocked). 1h: take rows ADX[15-18)=-0.87%,
# ADX[18-20)=-1.29% → 1h floor kept at 18 (low ADX genuinely bad on 1h).
# Set to 0 to disable.
IMPULSE_SPEED_15M_ADX_MIN: float = 15.0   # was 20.0 (backtest 2026-04-20: -5pts, ADX15-20 positive)
IMPULSE_SPEED_1H_ADX_MIN: float = 14.0  # scout:22.04.2026 was 15.84
IMPULSE_SPEED_LATE_GUARD_ENABLED: bool = True
# A/B kill-switches per-tf (added 2026-04-18 after Pareto sweep showed 15m guard
# blocked +6.77% non-bull winners on 15m). Flip back to True for rollback.
IMPULSE_SPEED_LATE_GUARD_15M_ENABLED: bool = False
IMPULSE_SPEED_LATE_GUARD_1H_ENABLED: bool = True
IMPULSE_SPEED_LATE_GUARD_15M_RSI_MIN: float = 70.0
IMPULSE_SPEED_LATE_GUARD_15M_RANGE_MIN: float = 6.0
IMPULSE_SPEED_LATE_GUARD_15M_PRICE_EDGE_MIN_PCT: float = 2.20
IMPULSE_SPEED_LATE_GUARD_15M_MACD_FADE_RATIO_MAX: float = 0.60
IMPULSE_SPEED_LATE_GUARD_15M_MACD_PEAK_LOOKBACK: int = 8
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RSI_MIN: float = 75.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RANGE_MIN: float = 10.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_PRICE_EDGE_MIN_PCT: float = 4.0
IMPULSE_SPEED_LATE_GUARD_1H_RSI_MIN: float = 68.0
IMPULSE_SPEED_LATE_GUARD_1H_RANGE_MIN: float = 14.0  # was 8.0 — Pareto sweep: 1h dr<14 blocks were +0.0..+0.5% misses
IMPULSE_SPEED_LATE_GUARD_1H_PRICE_EDGE_MIN_PCT: float = 2.40
IMPULSE_SPEED_LATE_GUARD_1H_MACD_FADE_RATIO_MAX: float = 0.68
IMPULSE_SPEED_LATE_GUARD_1H_MACD_PEAK_LOOKBACK: int = 6
# Bull-day relaxed thresholds for late guard
IMPULSE_SPEED_LATE_GUARD_15M_RSI_MIN_BULL: float = 76.0
IMPULSE_SPEED_LATE_GUARD_15M_RANGE_MIN_BULL: float = 10.0
IMPULSE_SPEED_LATE_GUARD_15M_PRICE_EDGE_MIN_PCT_BULL: float = 3.20
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RSI_MIN_BULL: float = 80.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RANGE_MIN_BULL: float = 15.0
IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_PRICE_EDGE_MIN_PCT_BULL: float = 5.0
IMPULSE_SPEED_LATE_GUARD_1H_RSI_MIN_BULL: float = 74.0
IMPULSE_SPEED_LATE_GUARD_1H_RANGE_MIN_BULL: float = 16.0  # was 12 — bull-day 12-16% range was full of winners
IMPULSE_SPEED_LATE_GUARD_1H_PRICE_EDGE_MIN_PCT_BULL: float = 3.50
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
EARLY_15M_CONTINUATION_ENTRY_ENABLED: bool = True
EARLY_15M_CONTINUATION_ENTRY_RSI_MIN: float = 60.0
EARLY_15M_CONTINUATION_ENTRY_RSI_MAX: float = 76.0
EARLY_15M_CONTINUATION_ENTRY_ADX_MIN: float = 20.0
EARLY_15M_CONTINUATION_ENTRY_SLOPE_MIN: float = 0.10
EARLY_15M_CONTINUATION_ENTRY_VOL_X_MIN: float = 1.05
EARLY_15M_CONTINUATION_ENTRY_RANGE_MAX: float = 8.0
EARLY_15M_CONTINUATION_ENTRY_PRICE_EDGE_MAX_PCT: float = 1.60
EARLY_15M_CONTINUATION_ENTRY_SCORE_BONUS: float = 10.0
EARLY_15M_CONTINUATION_ENTRY_MODES: tuple = ("trend",)
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
# 2026-05-06: 7200 → 1800 (2h → 30min) per always-watch-pump-detector-spec.
# Trigger: STRKUSDT silent miss case (pump start within 2h gap window).
# Pump detector below catches sub-30min bursts; this catches medium-cycle
# transitions from quiet→active.
AUTO_REANALYZE_SEC: int = 1800   # was 7200
AUTO_REANALYZE_FIRST_DELAY_SEC: int = 90

# ── Always-watch pump detector (2026-05-06) ──────────────────────────────────
# Spec: docs/specs/features/always-watch-pump-detector-spec.md
# Polls Binance /ticker/24hr every PUMP_DETECTOR_INTERVAL_SEC, tracks
# priceChangePct delta per watchlist coin, injects to hot_coins on pump.
PUMP_DETECTOR_ENABLED: bool = True
PUMP_DETECTOR_INTERVAL_SEC: int = 300   # 5 min
PUMP_TRIGGER_PCT: float = 2.0           # min Δ in priceChangePct
PUMP_LOOKBACK_MIN: int = 15             # delta window
POSITIONS_FILE: str = "positions.json"

ATR_TRAIL_K: float = 2.0   # множитель ATR для трейлинг-стопа
MACDWARN_BARS: int = 3     # баров подряд MACD hist падает → предупреждение о развороте

# ── Trail-stop minimum buffer (anti-whipsaw, 2026-04-26) ──────────────────────
# Problem (ALGOUSDT 26.04 case): bandit picks "tight" arm (trail_k=2.12) which
# is statistically profitable on average (+0.21% over 85 trades on impulse_speed/15m).
# But on coins with compressed 15m ATR (0.4% of price) and wide daily-range (5%+),
# 2.12*ATR = 0.86% buffer → noise pierces stop → exit → price continues without us.
# 52 of 85 "tight"-arm trades on impulse_speed/strong_trend ended in trail-loss
# averaging -1.64%.
# Fix: enforce a MINIMUM trail buffer in % of entry price (volatility-adjusted),
# regardless of bandit/ATR-based choice. Buffer = max(trail_k*ATR, MIN_PCT*price).
# Mode-aware: high-vol modes (impulse_speed, strong_trend) need wider min-buffer.
TRAIL_MIN_BUFFER_PCT_ENABLED: bool = True
TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED: float = 0.015  # 1.5% min buffer (high-vol regime)
TRAIL_MIN_BUFFER_PCT_STRONG_TREND:  float = 0.015
TRAIL_MIN_BUFFER_PCT_IMPULSE:       float = 0.012  # 1.2% min buffer
TRAIL_MIN_BUFFER_PCT_TREND:         float = 0.0    # disabled — narrow stops work for trend
TRAIL_MIN_BUFFER_PCT_ALIGNMENT:     float = 0.0
TRAIL_MIN_BUFFER_PCT_RETEST:        float = 0.0    # retest by design uses tight stop
TRAIL_MIN_BUFFER_PCT_BREAKOUT:      float = 0.0    # breakout by design uses tight stop
TRAIL_MIN_BUFFER_PCT_DEFAULT:       float = 0.0    # other modes — no floor

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
COOLDOWN_BARS: int = 19  # баров тишины после выхода (Scout APPROVE: n=35, ret5=+0.79%, win=60%; было 24)

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
# G1+G4 quality gates — backtest 2026-04-20: 15m +0.19% delta, sharpe +2.50; 1h -0.06%→+0.42%
ALIGNMENT_QUALITY_GATES_ENABLED: bool = True
ALIGNMENT_MACD_SLOPE_LOOKBACK: int   = 2      # G1: hist[i] >= hist[i-N] (slope not fading)
ALIGNMENT_MIN_DIST_TO_HIGH_BARS: int = 20     # G4: rolling N-bar high for distance check
ALIGNMENT_MIN_DIST_TO_HIGH_ATR: float = 0.30  # G4: must be >= X ATRs below the N-bar high

# ── Fast loss exit: ускоренный выход после первого закрытия ниже EMA20 ─────────────
# Ночная проблема 23.03: часть 15m сделок выходила только после 2-го close ниже EMA20,
# хотя уже на первом закрытии под EMA20 сделка была в минусе и импульс слабел.
FAST_LOSS_EMA_EXIT_ENABLED: bool = True
FAST_LOSS_EMA_EXIT_TF: tuple = ("15m", "1h")
# 19.04: `impulse` удалён — первый бар после импульса часто волатилен и уходит
# ниже EMA20 даже при продолжении движения (TRUUSDT 15m churn: вход 0.0049 -> выход
# 0.0048 через 1 бар, -2.04%). Для impulse это не первый убыточный close, а шум.
FAST_LOSS_EMA_EXIT_MODES: tuple = ("retest", "breakout", "alignment")
# 19.04: было 1 — слишком агрессивно, ловили шумовой первый бар (TRUUSDT).
# Даём сделке 2 бара, чтобы убедиться, что это реальный разворот, а не фитиль.
FAST_LOSS_EMA_EXIT_MIN_BARS: int = 2
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
RETEST_1H_MTF_CONFIRM_ENABLED: bool = True
RETEST_1H_MTF_RSI_MIN: float = 48.0
RETEST_1H_MTF_RSI_MAX: float = 72.0
RETEST_1H_MTF_EMA20_SLIP_PCT: float = 0.15
RETEST_1H_MTF_MACD_MIN_REL: float = 0.0
RETEST_1H_MTF_REQUIRE_MACD_RISING: bool = True

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
EMA_CROSS_ENABLED:   bool  = False  # 19.04.2026: disabled — 7 трейдов 7д, win 14%, sum -9.2% PnL. Ранний кросс-вход слишком шумный, ловит фейковые пробои.
ATR_TRAIL_K_CROSS:   float = 2.5    # широкий стоп для ранних входов (цена ещё нестабильна)

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
MIN_WEAK_EXIT_BARS: int = 6   # 6 баров = 1.5ч на 15m (было 2=30мин — flip-flop fix)
MIN_WEAK_EXIT_BARS_RETEST: int = 6    # retest отключён, но на случай включения
MIN_WEAK_EXIT_BARS_BREAKOUT: int = 6  # breakout отключён, но на случай включения
MIN_WEAK_EXIT_BARS_TREND: int = 8     # trend: минимум 2ч до WEAK выхода (было 2)
MIN_WEAK_EXIT_BARS_IMPULSE_SPEED: int = 8  # impulse_speed: было 4
TREND_HOLD_WEAK_EXIT_ENABLED: bool = True
TREND_HOLD_WEAK_EXIT_TF: tuple = ("15m",)
TREND_HOLD_WEAK_EXIT_MODES: tuple = ("impulse_speed", "trend", "alignment")
TREND_HOLD_WEAK_EXIT_MIN_BARS: int = 5
TREND_HOLD_WEAK_EXIT_MIN_PNL_PCT: float = 0.75
TREND_HOLD_WEAK_EXIT_MIN_ENTRY_SCORE: float = 70.0
TREND_HOLD_WEAK_EXIT_MIN_ADX: float = 24.0
TREND_HOLD_WEAK_EXIT_MIN_SLOPE_PCT: float = 0.10
TREND_HOLD_WEAK_EXIT_TIGHTEN_ATR_K: float = 1.4

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
PORTFOLIO_REPLACE_ENABLED: bool = True
PORTFOLIO_REPLACE_MIN_DELTA: float = 8.0
PORTFOLIO_REPLACE_FRESH_MIN_DELTA: float = 6.0
PORTFOLIO_REPLACE_MIN_BARS: int = 2
PORTFOLIO_REPLACE_PROFIT_PROTECT_PCT: float = 0.35
PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT: float = 0.80
PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA: float = 12.0
PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA: float = 10.0
PORTFOLIO_REPLACE_ADX_PROTECT_MIN: float = 30.0
PORTFOLIO_REPLACE_TREND_GRACE_BARS: int = 5
PORTFOLIO_REPLACE_RANKER_ENABLED: bool = True
PORTFOLIO_REPLACE_RANKER_FINAL_WEIGHT: float = 6.0
PORTFOLIO_REPLACE_TOP_GAINER_WEIGHT: float = 10.0
PORTFOLIO_REPLACE_CANDIDATE_MIN_FINAL: float = -0.50
PORTFOLIO_REPLACE_CANDIDATE_MIN_TOP_GAINER: float = 0.10
PORTFOLIO_REPLACE_POSITION_FINAL_MAX: float = 0.00
PORTFOLIO_REPLACE_POSITION_TOP_GAINER_MAX: float = 0.20
WEAK_REENTRY_COOLDOWN_BARS: int = 24  # was 8 — увеличено чтобы пресечь flip-flop после WEAK
ENTRY_SCORE_MIN_ENABLED: bool = True
ENTRY_SCORE_MIN_15M: float = 40.0  # was 45.0 — Pareto sweep 2026-04-18: blocked/entry_score n=1374 had avg_r5=+0.123% vs take=-0.016%
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
ENTRY_SCORE_CONTINUATION_BYPASS_ENABLED: bool = True
ENTRY_SCORE_CONTINUATION_1H_ENABLED: bool = True
ENTRY_SCORE_CONTINUATION_MODES: tuple[str, ...] = ("alignment", "trend", "strong_trend")
ENTRY_SCORE_CONTINUATION_REQUIRE_BULL_DAY: bool = True
ENTRY_SCORE_CONTINUATION_SCORE_MIN_1H: float = 42.0
ENTRY_SCORE_CONTINUATION_ADX_MIN: float = 16.0
ENTRY_SCORE_CONTINUATION_SLOPE_MIN: float = 0.15   # was 0.30 — BNB slope=0.197 was blocked (15.04.2026)
ENTRY_SCORE_CONTINUATION_VOL_MIN: float = 0.90
ENTRY_SCORE_CONTINUATION_DAILY_RANGE_MAX: float = 8.5
ENTRY_SCORE_CONTINUATION_PRICE_EDGE_MAX_PCT: float = 2.0
ENTRY_SCORE_CONTINUATION_RSI_MIN: float = 54.0
ENTRY_SCORE_CONTINUATION_RSI_MAX: float = 72.0
TREND_15M_QUALITY_GUARD_ENABLED: bool = True
TREND_15M_QUALITY_FORECAST_MIN: float = 0.25
TREND_15M_QUALITY_ALT_VOL_MIN: float = 1.20
TREND_15M_QUALITY_ALT_ADX_MIN: float = 24.0
TREND_15M_QUALITY_ALT_SLOPE_MIN: float = 0.35
TREND_15M_QUALITY_RSI_MAX: float = 72.0              # was 68.0 — too tight on bull days (TAO blocked at 73.2)
TREND_15M_QUALITY_RSI_MAX_BULL_DAY: float = 76.0      # relaxed for bull days
TREND_15M_QUALITY_DAILY_RANGE_MAX: float = 10.0        # was 8.0
TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY: float = 14.0  # relaxed for bull days (TAO 12% blocked)
TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT: float = 3.20     # was 2.40 — TAO blocked at 2.43%
TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT: float = 4.00  # wider on bull days
# ── Trend/1h chop-filter (2026-05-01) ──────────────────────────────────────────
# Spec: docs/specs/features/trend-1h-chop-filter-spec.md
# Backtest 30 d (_validate_trend_chop_filter.py): trend/1h baseline precision
# 1.2 %, after filter (ADX>=25 & slope>=1.2 & vol>=1.3) precision 16.7 %,
# recall 100 %, avg_pnl +1.58 % vs −0.17 %.
# Live trigger case: STRKUSDT 2026-05-01 (ADX 20.2, slope +0.70 %, chop range).
TREND_1H_CHOP_FILTER_ENABLED: bool = True
TREND_1H_CHOP_ADX_MIN: float = 25.0
# 2026-05-06: 1.2 → 0.7 after TONUSDT case (slope 0.88-0.90% blocked 39 times,
# trend later +70%). ADX/vol still gate hard. See:
# docs/specs/features/trend-1h-chop-filter-spec.md §6 (regression risk noted).
TREND_1H_CHOP_SLOPE_MIN: float = 0.7  # was 1.2 (slow-build trends now allowed)
TREND_1H_CHOP_VOL_MIN: float = 1.3   # vol_x multiplier
# Bull-day relaxation (opt-in; not validated on bull-day subsample)
TREND_1H_CHOP_USE_BULL_DAY_RELAX: bool = False
TREND_1H_CHOP_ADX_MIN_BULL_DAY: float = 22.0
TREND_1H_CHOP_SLOPE_MIN_BULL_DAY: float = 1.0
TREND_1H_CHOP_VOL_MIN_BULL_DAY: float = 1.2
# ── H3 · Trend-surge precedence (2026-05-02) ───────────────────────────────────
# Spec: docs/specs/features/trend-surge-precedence-spec.md
# Когда True: surge_ok идёт ПЕРЕД entry_ok в pipeline (раньше ловим slope-
# ускорение). Default False для постепенного rollout. Acceptance: 7 d shadow
# с >=5 reclassifications и без regression в recall@top20.
TREND_SURGE_PRECEDENCE_ENABLED: bool = False
ATR_TRAIL_K_TREND_SURGE: float = 2.5  # = STRONG default; trail для нового режима
# ── H5 · Trailing-only after break-even (2026-05-02) ───────────────────────────
# Spec: docs/specs/features/h5-trailing-only-break-even-spec.md
# When position is profitable (pnl >= H5_BREAK_EVEN_PCT), suppress soft
# EMA-pattern exits ("2 closes below EMA20", "slope flip", "ADX weakening")
# and let ATR-trail handle reversals. Backtest 30d: 4/982 eligible exits,
# 1 top-20 winner left +471% on table (APEUSDT 04-30).
# Default: SHADOW on, ENABLED off (logging-only mode for 7d acceptance).
# 2026-05-05 ACTIVATED in production (was: ENABLED=False, SHADOW=True)
# Trigger: ICPUSDT 05-05 09:35→12:21 — bot exited at +2.5% on RSI divergence
# WEAK signal, trend continued to +6.8% (left +4.3% on table). Capture
# ratio 0.37, exact case for which H5 was written. Backtest baseline 4
# eligible exits in 30d, 1 of them top-20 with +471% potential left. Real
# expected impact at activation: 1-3 saved trades/week, +0.05 NS estimate.
# Rollback: flip ENABLED back to False, SHADOW remains as fallback log.
H5_TRAILING_ONLY_AFTER_BREAK_EVEN_ENABLED: bool = True   # was False (2026-05-05)
H5_TRAILING_ONLY_SHADOW: bool = False                    # was True  (2026-05-05)
H5_BREAK_EVEN_PCT: float = 0.5

# ── P1.3: H5 per-mode break-even thresholds (2026-05-07) ──────────────────────
# Spec: docs/specs/features/h5-trailing-only-break-even-spec.md (Phase 2)
# TON Trade #2 (2026-05-06): pnl=+0.24% at WEAK exit, H5 needed +0.5% so
# WEAK passed → coin continued to +16% (left ~14% on table). Fast modes
# (impulse_speed, impulse) need lower threshold for early protection;
# slow modes (strong_trend) stay patient (ATR-trail wider already).
# Resolution: H5_BREAK_EVEN_PCT_<MODE>_<TF> > H5_BREAK_EVEN_PCT_<MODE> > H5_BREAK_EVEN_PCT.
H5_BREAK_EVEN_PCT_IMPULSE_SPEED: float = 0.3   # fast: protect earlier
H5_BREAK_EVEN_PCT_IMPULSE: float = 0.3
H5_BREAK_EVEN_PCT_TREND_SURGE: float = 0.3     # if precedence flag flipped
H5_BREAK_EVEN_PCT_BREAKOUT: float = 0.3
H5_BREAK_EVEN_PCT_RETEST: float = 0.5
H5_BREAK_EVEN_PCT_ALIGNMENT: float = 0.5
H5_BREAK_EVEN_PCT_TREND: float = 0.5
H5_BREAK_EVEN_PCT_STRONG_TREND: float = 0.7    # slow: more patient

# ── P1.2: PEAK RISK shadow detector (2026-05-07) ──────────────────────────────
# Spec: docs/specs/features/peak-risk-shadow-spec.md
# Computes 0-100 risk score for open profitable positions; logs structured
# events when score crosses bucket thresholds (50/70/90). Shadow-only —
# no SELL triggered. Phase 2 (separate spec) will add TG alert + tighter
# trail once 7d shadow data validates the formula.
PEAK_RISK_SHADOW_ENABLED: bool = True
PEAK_RISK_SHADOW_THRESHOLD: float = 50.0
PEAK_RISK_RSI_FLOOR: float = 75.0     # RSI level where component starts ramping
PEAK_RISK_EDGE_FLOOR_PCT: float = 5.0  # price edge vs EMA20 where component starts

# ── UI watchdog (2026-05-06) ──────────────────────────────────────────────
# Detects stuck Telegram polling. If no update has been processed by handlers
# for WARN_THRESHOLD_SEC AND there are pending Telegram updates,
# warn N times then force-exit (wrapper relaunches bot).
# Spec: docs/specs/features/ui-watchdog-spec.md
UI_WATCHDOG_WARN_THRESHOLD_SEC: float = 90.0
UI_WATCHDOG_FORCE_EXIT_AFTER_WARNS: int = 3
# Mode daily-range / slope quality gate (backtest 2026-04-24, 60d, 2197 entries)
# Root cause: on quiet-market days (daily_range 3-4%) signals are almost all FP
# because coins don't make big moves regardless of technical setup.
# TP entries have significantly higher daily_range than FP across all modes.
MODE_RANGE_QUALITY_GUARD_ENABLED: bool = True
ALIGNMENT_15M_RANGE_MIN:  float = 4.0  # +5.9pp prec (23.0%->28.9%), blocks 49% entries
TREND_15M_RANGE_MIN:      float = 4.0  # +17.1pp prec (29.9%->47.0%), blocks 57% entries
ALIGNMENT_1H_RANGE_MIN:   float = 5.0  # +8.0pp prec (30.9%->38.9%), blocks 28% entries
TREND_1H_SLOPE_MIN:       float = 0.50 # +4.6pp prec (14.1%->18.8%), blocks 28% entries
IMPULSE_SPEED_1H_RANGE_MIN: float = 7.0  # +3.3pp prec (16.5%->19.8%), blocks 20% entries
NEAR_MISS_LOGGING_ENABLED: bool = True
NEAR_MISS_FORECAST_MIN: float = 0.05
NEAR_MISS_VOL_MIN: float = 0.85
NEAR_MISS_ADX_MIN: float = 12.0
NEAR_MISS_RSI_MIN: float = 46.0
NEAR_MISS_RSI_MAX: float = 74.0
NEAR_MISS_DAILY_RANGE_MAX_15M: float = 9.0
NEAR_MISS_DAILY_RANGE_MAX_1H: float = 11.0
NEAR_MISS_SCORE_DEFICIT_MAX_15M: float = 6.0
NEAR_MISS_SCORE_DEFICIT_MAX_1H: float = 8.0
NEAR_MISS_BREAKOUT_LOOKBACK: int = 6
NEAR_MISS_BREAKOUT_GAP_MAX_PCT: float = 0.45
NEAR_MISS_BREAKOUT_VOL_MIN: float = 0.95
NEAR_MISS_BREAKOUT_SLOPE_MIN: float = 0.05
NEAR_MISS_RETEST_UNDER_EMA20_MAX_PCT: float = 0.12
NEAR_MISS_RETEST_PRICE_EDGE_MAX_PCT: float = 0.35
NEAR_MISS_RETEST_SLOPE_MIN: float = 0.05
NEAR_MISS_RETEST_MACD_MIN_REL: float = -0.00005
NEAR_MISS_ALIGNMENT_UNDER_EMA20_MAX_PCT: float = 0.15
NEAR_MISS_ALIGNMENT_SLOPE_MIN: float = 0.04
NEAR_MISS_ALIGNMENT_EMA_SEP_MIN: float = -0.05
NEAR_MISS_TREND_PRICE_EDGE_MIN_PCT: float = 0.05
NEAR_MISS_TREND_PRICE_EDGE_MAX_PCT: float = 2.60
NEAR_MISS_TREND_SLOPE_MIN: float = 0.12
NEAR_MISS_TREND_ADX_MIN: float = 16.0
NEAR_MISS_TREND_VOL_MIN: float = 0.90

# Максимум в одной "группе" монет — корреляционный лимит
# Монеты одной группы двигаются синхронно (L1, AI, GameFi и т.д.)
MAX_POSITIONS_PER_GROUP: int = 2
CLONE_SIGNAL_GUARD_ENABLED: bool = True
CLONE_SIGNAL_GUARD_TF: tuple = ("15m",)
CLONE_SIGNAL_GUARD_MODES: tuple = ("impulse_speed", "breakout", "retest", "alignment", "trend")
CLONE_SIGNAL_GUARD_WINDOW_BARS: int = 8
CLONE_SIGNAL_GUARD_MAX_SIMILAR: int = 14  # was 4 — 116 blocks in recent events: FLUX/ORDI blocked by clone guard  # scout:22.04.2026 was 13
CLONE_SIGNAL_GUARD_MAX_SAME_GROUP: int = 1
CLONE_SIGNAL_GUARD_OVERRIDE_SCORE: float = 90.0
CLONE_SIGNAL_GUARD_OVERRIDE_RANKER_FINAL: float = 0.50
CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL: float = -0.25
OPEN_SIGNAL_CLUSTER_CAP_ENABLED: bool = True
OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES: tuple = ("breakout", "retest")
OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES: tuple = ("impulse_speed",)
OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX: int = 5   # was 2 — 149 blocks in recent events (ORDI, FLUX blocked)  # scout:16.04.2026 was 4
OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES: tuple = ("retest",)
OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX: int = 1

# ── Correlation Guard — защита от коррелированных позиций ────────────────────
# Проблема (16.04.2026): из 10 открытых позиций 5 двигались синхронно (rho 0.55-0.63):
# CRV/AAVE/STRK/MEME/DOT — по сути один «альтсезон» паттерн, занимающий 5 слотов.
# При развороте рынка все падают одновременно → пиковый drawdown умножается.
#
# Решение: ограничить количество сильно коррелированных монет в портфеле.
# Реализация: correlation_guard.py
CORR_GUARD_ENABLED: bool = True                  # включить/выключить guard
CORR_GUARD_TF: str = "1h"                        # таймфрейм для расчёта корреляций
CORR_GUARD_WINDOW_BARS: int = 48                 # окно в барах (~2 суток на 1h)
CORR_GUARD_THRESHOLD: float = 0.65              # порог кластеризации (базовый)
CORR_GUARD_THRESHOLD_BULL: float = 0.60         # порог в бычий день (строже)
CORR_GUARD_THRESHOLD_BEAR: float = 0.99         # порог в медвежий день (фактически off)
CORR_MAX_PER_CLUSTER: int = 2                    # макс. позиций в одном кластере
CORR_MARGINAL_WEIGHTING: bool = True             # штраф score при ранжировании дублей
CORR_PRUNE_ENABLED: bool = True                  # закрывать раздувшиеся кластеры
CORR_PRUNE_PROFIT_PROTECT_PCT: float = 2.0       # не закрывать позиции с PnL > этого %
CORR_PRUNE_MIN_BARS: int = 4                     # не прунить позиции моложе N баров (защита от 1-бар выходов)
CORR_CACHE_TTL_MIN: int = 15                     # TTL кэша матрицы в минутах

# ── ML-Gated Portfolio Rotation ──────────────────────────────────────────────
# Когда портфель полон и существующая score-based ротация не нашла замену,
# даём вторую попытку: если у кандидата высокий ml_proba — помечаем на выход
# самую слабую по EV позицию. Реализация: rotation.py
# Бэктест (files/backtest_portfolio_rotation_grid.py):
#   ml_proba>=0.62 → n=211 сделок, avg_r5=+0.24%, Sharpe=+2.75, sumPnL5=+51%
ROTATION_ENABLED: bool = True
ROTATION_ML_PROBA_MIN: float = 0.62              # грид-свип: лучший Sharpe
ROTATION_WEAK_EV_MAX: float = -0.40              # позиция «слабая» при EV < этого
ROTATION_WEAK_BARS_MIN: int = 3                  # минимум баров перед вытеснением
ROTATION_PROFIT_PROTECT_PCT: float = 0.5         # защищать позиции с PnL > этого %
ROTATION_MAX_PER_POLL: int = 1                   # максимум eviction за один poll

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

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: WebSocket, Order Flow, Derivatives, Regime, Top Gainer, Multi-Horizon
# ═══════════════════════════════════════════════════════════════════════════════

# ── WebSocket Manager ────────────────────────────────────────────────────────
WS_ENABLED: bool = True                    # Enable real-time WebSocket streams
WS_ENABLE_TRADES: bool = True              # aggTrade stream (for order flow)
WS_ENABLE_BOOK: bool = True                # bookTicker stream (bid/ask spread)
WS_ENABLE_KLINES_1M: bool = True           # 1-minute kline stream

# ── Order Flow Analyzer ─────────────────────────────────────────────────────
ORDER_FLOW_ENABLED: bool = True
ORDER_FLOW_WINDOWS: tuple = (1, 5, 15)     # Minutes: compute metrics for these windows
ORDER_FLOW_LARGE_TRADE_SIGMA: float = 2.0  # Trades > mean + K*σ = "large"

# Order flow entry bonus/penalty
ORDER_FLOW_SCORE_ENABLED: bool = True
ORDER_FLOW_BULL_CVD_MIN: float = 3.0       # CVD% above this → bullish confirmation
ORDER_FLOW_BULL_IMBALANCE_MIN: float = 0.55
ORDER_FLOW_BULL_BONUS: float = 8.0         # Score bonus for bullish order flow
ORDER_FLOW_BEAR_CVD_MAX: float = -3.0      # CVD% below this → bearish divergence
ORDER_FLOW_BEAR_PENALTY: float = 6.0       # Score penalty for bearish order flow
ORDER_FLOW_BREAKOUT_BONUS: float = 10.0    # Extra bonus for order flow breakout signal

# ── Derivatives Data Pipeline ────────────────────────────────────────────────
DERIVATIVES_ENABLED: bool = True
DERIVATIVES_FUNDING_INTERVAL: int = 60     # Fetch funding rate every N seconds
DERIVATIVES_OI_INTERVAL: int = 60          # Fetch open interest every N seconds
DERIVATIVES_LS_INTERVAL: int = 300         # Fetch L/S ratio every N seconds

# Derivatives entry adjustments
DERIV_OI_EXPANSION_BONUS: float = 5.0      # OI growing > 5% in 1h + uptrend
DERIV_FUNDING_FLIP_BONUS: float = 4.0      # Funding flipped neg → pos
DERIV_HIGH_FUNDING_PENALTY: float = 3.0    # Funding > 0.1% → crowded long
DERIV_SHORT_SQUEEZE_BONUS: float = 5.0     # Low L/S ratio + price up

# ── Regime Detector (HMM) ───────────────────────────────────────────────────
REGIME_DETECTOR_ENABLED: bool = True
REGIME_HMM_MODEL_FILE: str = "regime_model.json"
REGIME_UPDATE_INTERVAL_BARS: int = 1       # Update regime every N monitoring ticks
# Regime-based entry gating (applied on top of existing logic)
REGIME_ENTRY_GATING_ENABLED: bool = True
REGIME_RISK_OFF_BLOCK: bool = True         # Block entries in strong_bear / volatile_chop

# ── Top Gainer Classifier ───────────────────────────────────────────────────
TOP_GAINER_MODEL_ENABLED: bool = True
TOP_GAINER_MODEL_FILE: str = "top_gainer_model.json"
TOP_GAINER_DATASET_ENABLED: bool = True    # Log training samples
TOP_GAINER_LABEL_COLLECTION_HOUR_UTC: int = 0  # Collect EOD labels at 00:00 UTC

# Top gainer score bonuses
TOP_GAINER_SCORE_BONUS_TOP5: float = 15.0
TOP_GAINER_SCORE_BONUS_TOP10: float = 10.0
TOP_GAINER_SCORE_BONUS_TOP20: float = 6.0
TOP_GAINER_SCORE_BONUS_TOP50: float = 3.0

# ── Multi-Horizon Framework ───────────────────────────���─────────────────────
MULTI_HORIZON_ENABLED: bool = True
MULTI_HORIZON_SCALP_ENABLED: bool = True
MULTI_HORIZON_DAILY_ENABLED: bool = True
MULTI_HORIZON_SWING_ENABLED: bool = True
MULTI_HORIZON_CASCADE_BONUS_ENABLED: bool = True

# Cascade alignment bonuses (added to entry score)
MULTI_HORIZON_CASCADE_4_BONUS: float = 15.0  # 4+ horizons agree
MULTI_HORIZON_CASCADE_3_BONUS: float = 10.0  # 3 horizons agree
MULTI_HORIZON_CASCADE_2_BONUS: float = 5.0   # 2 horizons agree

# Portfolio allocation by horizon (% of total)
HORIZON_ALLOCATION_SCALP: float = 0.15
HORIZON_ALLOCATION_INTRADAY: float = 0.35
HORIZON_ALLOCATION_DAILY: float = 0.25
HORIZON_ALLOCATION_SWING: float = 0.15
HORIZON_ALLOCATION_POSITION: float = 0.10
