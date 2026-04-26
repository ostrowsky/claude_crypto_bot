from __future__ import annotations

"""
Live monitoring module.

Runs a background async loop that polls Binance every POLL_SEC seconds.
For each "hot" coin from morning analysis:
  - If no position: checks entry conditions on last closed bar
  - If in position: checks exit conditions + validates forward predictions
  
Sends Telegram messages via callback for all meaningful events.
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
import numpy as np

import config
from indicators import compute_features
from strategy import (
    CoinReport, fetch_klines, analyze_coin,
    check_entry_conditions, check_exit_conditions,
    check_retest_conditions, check_breakout_conditions,
    check_impulse_conditions, check_alignment_conditions,
    check_trend_surge_conditions, check_ema_cross_conditions,
    get_effective_entry_mode,
    get_entry_mode,
)
import botlog
import critic_dataset
import ml_dataset
import correlation_guard as corr_guard
import rotation

log = logging.getLogger(__name__)

# ── Enhanced signals (Phase 2: WS, order flow, derivatives, regime, top gainer) ──
_enhanced_signals = None

def _get_enhanced():
    """Lazy accessor for enhanced signals module."""
    global _enhanced_signals
    if _enhanced_signals is not None:
        return _enhanced_signals
    try:
        from enhanced_signals import get_enhanced_signals
        _enhanced_signals = get_enhanced_signals()
    except ImportError:
        _enhanced_signals = None
    return _enhanced_signals
_ML_MODEL_FILE = Path(__file__).resolve().parent / "ml_signal_model.json"
_ML_MODEL_CACHE: Optional[dict] = None
_RANKER_MODEL_FILE = Path(__file__).resolve().parent / str(
    getattr(config, "ML_CANDIDATE_RANKER_MODEL_FILE", "ml_candidate_ranker.json")
)
_RANKER_MODEL_CACHE: Optional[dict] = None
_BOT_EVENTS_FILE = Path(__file__).resolve().parent / "bot_events.jsonl"


def _aux_notifications_enabled() -> bool:
    return bool(getattr(config, "SEND_AUX_NOTIFICATIONS", False))


def _load_ml_model_payload() -> dict:
    global _ML_MODEL_CACHE
    if _ML_MODEL_CACHE is not None:
        return _ML_MODEL_CACHE
    try:
        payload = json.loads(_ML_MODEL_FILE.read_text(encoding="utf-8"))
        _ML_MODEL_CACHE = payload if isinstance(payload, dict) else {}
    except Exception:
        _ML_MODEL_CACHE = {}
    return _ML_MODEL_CACHE


def _load_ml_segment_payloads() -> dict:
    raw = _load_ml_model_payload()
    return raw.get("segment_model_payloads", {}) if isinstance(raw, dict) else {}


def _load_ml_general_payload() -> Optional[dict]:
    raw = _load_ml_model_payload()
    if not isinstance(raw, dict):
        return None
    if "feature_names" not in raw or "model" not in raw:
        return None
    return raw


def _select_ml_payload(signal_type: str, is_bull_day: bool) -> Optional[dict]:
    if getattr(config, "ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE", True):
        segment_key = f"{signal_type}|{'bull' if is_bull_day else 'nonbull'}"
        segment_payload = _load_ml_segment_payloads().get(segment_key)
        if isinstance(segment_payload, dict) and "feature_names" in segment_payload and "model" in segment_payload:
            return segment_payload
    return _load_ml_general_payload()


def _ml_general_score(
    sym: str,
    tf: str,
    signal_type: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    *,
    is_bull_day: bool,
) -> Optional[float]:
    if not getattr(config, "ML_ENABLE_GENERAL_RANKING", False):
        return None
    payload = _select_ml_payload(signal_type, is_bull_day)
    if not payload:
        return None
    try:
        from ml_signal_model import build_runtime_record, predict_proba_from_payload

        rec = build_runtime_record(
            sym=sym,
            tf=tf,
            signal_type=signal_type,
            is_bull_day=is_bull_day,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            btc_vs_ema50=float(getattr(config, "_btc_vs_ema50", 0.0)),
        )
        return float(predict_proba_from_payload(payload, rec))
    except Exception:
        return None


def _ml_trend_nonbull_score(sym: str, tf: str, feat: dict, data: np.ndarray, i: int) -> Optional[float]:
    if not getattr(config, "ML_ENABLE_TREND_NONBULL_FILTER", False):
        return None
    if getattr(config, "_bull_day_active", False):
        return None
    if tf != "15m":
        return None

    segment_key = str(getattr(config, "ML_TREND_NONBULL_SEGMENT_KEY", "trend|nonbull"))
    payload = _load_ml_segment_payloads().get(segment_key)
    if isinstance(payload, dict) and "feature_names" in payload and "model" in payload:
        try:
            from ml_signal_model import build_runtime_record, predict_proba_from_payload

            rec = build_runtime_record(
                sym=sym,
                tf=tf,
                signal_type="trend",
                is_bull_day=False,
                bar_ts=int(data["t"][i]),
                feat=feat,
                data=data,
                i=i,
                btc_vs_ema50=float(getattr(config, "_btc_vs_ema50", 0.0)),
            )
            return float(predict_proba_from_payload(payload, rec))
        except Exception:
            return None
    return _ml_general_score(sym, tf, "trend", feat, data, i, is_bull_day=False)


def _load_ranker_payload() -> Optional[dict]:
    global _RANKER_MODEL_CACHE
    if _RANKER_MODEL_CACHE is not None:
        return _RANKER_MODEL_CACHE
    try:
        payload = json.loads(_RANKER_MODEL_FILE.read_text(encoding="utf-8"))
        _RANKER_MODEL_CACHE = payload if isinstance(payload, dict) else {}
    except Exception:
        _RANKER_MODEL_CACHE = {}
    return _RANKER_MODEL_CACHE or None


def _ranker_shadow_enabled() -> bool:
    return bool(getattr(config, "ML_CANDIDATE_RANKER_SHADOW_ENABLED", False))


def _ranker_shadow_threshold() -> float:
    payload = _load_ranker_payload()
    if payload:
        try:
            return float(payload.get("threshold", getattr(config, "ML_CANDIDATE_RANKER_NEUTRAL_PROBA", 0.50)))
        except Exception:
            pass
    return float(getattr(config, "ML_CANDIDATE_RANKER_NEUTRAL_PROBA", 0.50))


def _maybe_log_ranker_shadow(
    *,
    sym: str,
    tf: str,
    mode: str,
    price: float,
    candidate_score: float,
    score_floor: float,
    ranker_proba: Optional[float],
    ranker_info: Optional[Dict[str, float]] = None,
    bot_action: str,
    reason: str = "",
) -> None:
    if ranker_proba is None or not _ranker_shadow_enabled():
        return
    threshold = _ranker_shadow_threshold()
    ranker_take = ranker_proba >= threshold
    disagreement = (bot_action == "take" and not ranker_take) or (bot_action != "take" and ranker_take)
    if not disagreement and not bool(getattr(config, "ML_CANDIDATE_RANKER_SHADOW_LOG_ALL", False)):
        return
    try:
        botlog.log_ranker_shadow(
            sym=sym,
            tf=tf,
            mode=mode,
            price=price,
            candidate_score=candidate_score,
            score_floor=score_floor,
            ranker_proba=ranker_proba,
            ranker_threshold=threshold,
            bot_action=bot_action,
            reason=reason,
            ranker_final_score=None if not ranker_info else ranker_info.get("final_score"),
            ranker_ev=None if not ranker_info else ranker_info.get("ev_raw"),
            ranker_expected_return=None if not ranker_info else ranker_info.get("expected_return"),
            ranker_expected_drawdown=None if not ranker_info else ranker_info.get("expected_drawdown"),
        )
    except Exception:
        return


def _ml_candidate_ranker_components(
    *,
    sym: str,
    tf: str,
    signal_type: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    is_bull_day: bool,
    candidate_score: float,
    base_score: float,
    score_floor: float,
    forecast_return_pct: float,
    today_change_pct: float,
    ml_proba: Optional[float],
    mtf_soft_penalty: float,
    fresh_priority: bool,
    catchup: bool,
    continuation_profile: bool,
    signal_flags: Optional[Dict[str, bool]] = None,
) -> Optional[Dict[str, float]]:
    if not getattr(config, "ML_CANDIDATE_RANKER_RUNTIME_ENABLED", False) and not _ranker_shadow_enabled():
        return None
    payload = _load_ranker_payload()
    if not payload:
        return None
    try:
        from ml_candidate_ranker import build_runtime_candidate_record, predict_components_from_candidate_payload

        rec = build_runtime_candidate_record(
            sym=sym,
            tf=tf,
            signal_type=signal_type,
            is_bull_day=is_bull_day,
            bar_ts=int(data["t"][i]),
            feat=feat,
            data=data,
            i=i,
            candidate_score=candidate_score,
            base_score=base_score,
            score_floor=score_floor,
            forecast_return_pct=forecast_return_pct,
            today_change_pct=today_change_pct,
            ml_proba=ml_proba,
            mtf_soft_penalty=mtf_soft_penalty,
            fresh_priority=fresh_priority,
            catchup=catchup,
            continuation_profile=continuation_profile,
            signal_flags=signal_flags,
            btc_vs_ema50=float(getattr(config, "_btc_vs_ema50", 0.0)),
        )
        comps = predict_components_from_candidate_payload(payload, rec)
        comps["payload_version"] = float(int(payload.get("payload_version", 1) or 1))
        return {str(k): float(v) for k, v in comps.items()}
    except Exception:
        return None


def _ml_candidate_ranker_score(
    **kwargs,
) -> Optional[float]:
    comps = _ml_candidate_ranker_components(**kwargs)
    if not comps:
        return None
    return float(comps.get("quality_proba", 0.0))


def _ml_candidate_ranker_runtime_bonus(ranker_info: Optional[Dict[str, float]]) -> float:
    if not ranker_info or not getattr(config, "ML_CANDIDATE_RANKER_RUNTIME_ENABLED", False):
        return 0.0
    weight = float(getattr(config, "ML_CANDIDATE_RANKER_SCORE_WEIGHT", 0.0))
    if weight == 0.0:
        return 0.0
    if bool(getattr(config, "ML_CANDIDATE_RANKER_USE_FINAL_SCORE", True)) and int(ranker_info.get("payload_version", 1.0)) >= 2:
        clip = max(0.1, float(getattr(config, "ML_CANDIDATE_RANKER_SCORE_CLIP", 2.0)))
        raw = float(ranker_info.get("final_score", 0.0))
        raw = max(-clip, min(clip, raw))
        return raw * weight
    neutral = float(getattr(config, "ML_CANDIDATE_RANKER_NEUTRAL_PROBA", 0.50))
    return (float(ranker_info.get("quality_proba", neutral)) - neutral) * weight


def _candidate_signal_flags(
    *,
    entry_ok: bool,
    brk_ok: bool,
    ret_ok: bool,
    surge_ok: bool,
    imp_ok: bool,
    aln_ok: bool,
    cross_ok: bool = False,
) -> Dict[str, bool]:
    return {
        "entry_ok": bool(entry_ok),
        "breakout_ok": bool(brk_ok),
        "retest_ok": bool(ret_ok),
        "surge_ok": bool(surge_ok),
        "impulse_ok": bool(imp_ok),
        "alignment_ok": bool(aln_ok),
        "cross_ok": bool(cross_ok),
    }


def _near_miss_score_deficit_max(tf: str) -> float:
    if tf == "1h":
        return float(getattr(config, "NEAR_MISS_SCORE_DEFICIT_MAX_1H", 8.0))
    return float(getattr(config, "NEAR_MISS_SCORE_DEFICIT_MAX_15M", 6.0))


def _near_miss_candidate_snapshot(
    *,
    tf: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    forecast_return_pct: float,
    today_change_pct: float,
) -> Optional[Dict[str, Any]]:
    if not getattr(config, "NEAR_MISS_LOGGING_ENABLED", False):
        return None
    if i <= 0 or ema20 <= 0 or price <= 0:
        return None
    if forecast_return_pct < float(getattr(config, "NEAR_MISS_FORECAST_MIN", 0.05)):
        return None
    if vol_x < float(getattr(config, "NEAR_MISS_VOL_MIN", 0.85)):
        return None
    if rsi < float(getattr(config, "NEAR_MISS_RSI_MIN", 46.0)):
        return None
    if rsi > float(getattr(config, "NEAR_MISS_RSI_MAX", 74.0)):
        return None
    if adx < float(getattr(config, "NEAR_MISS_ADX_MIN", 12.0)):
        return None
    daily_range_cap = float(
        getattr(
            config,
            "NEAR_MISS_DAILY_RANGE_MAX_1H" if tf == "1h" else "NEAR_MISS_DAILY_RANGE_MAX_15M",
            11.0 if tf == "1h" else 9.0,
        )
    )
    if daily_range > daily_range_cap:
        return None

    score_floor = _entry_score_floor(tf) if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False) else 0.0
    deficit_max = _near_miss_score_deficit_max(tf)
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    macd_hist_arr = feat.get("macd_hist")
    macd_hist = (
        float(macd_hist_arr[i])
        if macd_hist_arr is not None and i < len(macd_hist_arr) and np.isfinite(macd_hist_arr[i])
        else 0.0
    )
    ema_slow_arr = feat.get("ema_slow")
    ema_slow = (
        float(ema_slow_arr[i])
        if ema_slow_arr is not None and i < len(ema_slow_arr) and np.isfinite(ema_slow_arr[i])
        else 0.0
    )
    ema_sep_pct = ((ema20 - ema_slow) / ema_slow * 100.0) if ema_slow > 0 else 0.0
    recent_high = float(np.max(data["h"][max(0, i - int(getattr(config, "NEAR_MISS_BREAKOUT_LOOKBACK", 6))):i])) if i > 0 else price
    breakout_gap_pct = ((recent_high / price) - 1.0) * 100.0 if recent_high > 0 else 0.0

    candidates: List[Dict[str, Any]] = []

    def _append_candidate(mode: str, tag: str, extra_ok: bool) -> None:
        if not extra_ok:
            return
        base_score = _entry_signal_score(
            mode=mode,
            price=price,
            ema20=ema20,
            slope=slope,
            adx=adx,
            rsi=rsi,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        cand_score = base_score
        cand_score += _top_mover_score_bonus(today_change_pct)
        cand_score += _forecast_return_score_bonus(forecast_return_pct)
        continuation_profile = _time_block_1h_continuation_profile(
            tf=tf,
            mode=mode,
            slope=slope,
            adx=adx,
            rsi=rsi,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        cand_score += _time_block_1h_continuation_bonus(
            tf=tf,
            mode=mode,
            slope=slope,
            adx=adx,
            rsi=rsi,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        if score_floor > 0.0 and cand_score >= score_floor:
            return
        if score_floor > 0.0 and cand_score < (score_floor - deficit_max):
            return
        candidates.append(
            {
                "mode": mode,
                "tag": tag,
                "candidate_score": float(cand_score),
                "base_score": float(base_score),
                "score_floor": float(score_floor),
                "continuation_profile": bool(continuation_profile),
            }
        )

    _append_candidate(
        "breakout",
        "pre_breakout",
        breakout_gap_pct >= -0.05
        and breakout_gap_pct <= float(getattr(config, "NEAR_MISS_BREAKOUT_GAP_MAX_PCT", 0.45))
        and vol_x >= float(getattr(config, "NEAR_MISS_BREAKOUT_VOL_MIN", 0.95))
        and slope >= float(getattr(config, "NEAR_MISS_BREAKOUT_SLOPE_MIN", 0.05))
        and macd_hist >= 0.0,
    )
    _append_candidate(
        "retest",
        "soft_retest",
        price >= ema20 * (1.0 - float(getattr(config, "NEAR_MISS_RETEST_UNDER_EMA20_MAX_PCT", 0.12)) / 100.0)
        and price_edge <= float(getattr(config, "NEAR_MISS_RETEST_PRICE_EDGE_MAX_PCT", 0.35))
        and slope >= float(getattr(config, "NEAR_MISS_RETEST_SLOPE_MIN", 0.05))
        and macd_hist >= price * float(getattr(config, "NEAR_MISS_RETEST_MACD_MIN_REL", -0.00005)),
    )
    _append_candidate(
        "alignment",
        "early_continuation",
        price >= ema20 * (1.0 - float(getattr(config, "NEAR_MISS_ALIGNMENT_UNDER_EMA20_MAX_PCT", 0.15)) / 100.0)
        and slope >= float(getattr(config, "NEAR_MISS_ALIGNMENT_SLOPE_MIN", 0.04))
        and ema_sep_pct >= float(getattr(config, "NEAR_MISS_ALIGNMENT_EMA_SEP_MIN", -0.05)),
    )
    trend_mode = get_entry_mode(feat, i)
    if trend_mode not in {"trend", "strong_trend", "impulse_speed"}:
        trend_mode = "trend"
    _append_candidate(
        trend_mode,
        "proto_trend",
        price_edge >= float(getattr(config, "NEAR_MISS_TREND_PRICE_EDGE_MIN_PCT", 0.05))
        and price_edge <= float(getattr(config, "NEAR_MISS_TREND_PRICE_EDGE_MAX_PCT", 2.60))
        and slope >= float(getattr(config, "NEAR_MISS_TREND_SLOPE_MIN", 0.12))
        and adx >= float(getattr(config, "NEAR_MISS_TREND_ADX_MIN", 16.0))
        and vol_x >= float(getattr(config, "NEAR_MISS_TREND_VOL_MIN", 0.90)),
    )

    if not candidates:
        return None
    best = max(candidates, key=lambda row: (float(row["candidate_score"]), _signal_priority(str(row["mode"]))))
    deficit = max(0.0, float(best["score_floor"]) - float(best["candidate_score"]))
    best["reason"] = (
        f"near miss {best['tag']}: score {float(best['candidate_score']):.2f} "
        f"< floor {float(best['score_floor']):.2f} by {deficit:.2f}"
    )
    return best


def _log_critic_candidate(
    *,
    sym: str,
    tf: str,
    bar_ts: int,
    signal_type: str,
    feat: dict,
    data: np.ndarray,
    i: int,
    action: str,
    reason_code: str = "",
    reason: str = "",
    stage: str = "",
    candidate_score: float = 0.0,
    base_score: float = 0.0,
    score_floor: float = 0.0,
    forecast_return_pct: float = 0.0,
    today_change_pct: float = 0.0,
    ml_proba: Optional[float] = None,
    mtf_soft_penalty: float = 0.0,
    fresh_priority: bool = False,
    catchup: bool = False,
    continuation_profile: bool = False,
    signal_flags: Optional[Dict[str, bool]] = None,
    near_miss: bool = False,
) -> str:
    if not getattr(config, "CRITIC_DATASET_ENABLED", False):
        return ""
    try:
        return critic_dataset.log_candidate(
            sym=sym,
            tf=tf,
            bar_ts=bar_ts,
            signal_type=signal_type,
            is_bull_day=bool(getattr(config, "_bull_day_active", False)),
            feat=feat,
            i=i,
            data=data,
            action=action,
            reason_code=reason_code,
            reason=reason,
            stage=stage,
            candidate_score=candidate_score,
            base_score=base_score,
            score_floor=score_floor,
            forecast_return_pct=forecast_return_pct,
            today_change_pct=today_change_pct,
            ml_proba=ml_proba,
            mtf_soft_penalty=mtf_soft_penalty,
            fresh_priority=fresh_priority,
            catchup=catchup,
            continuation_profile=continuation_profile,
            signal_flags=signal_flags,
            near_miss=near_miss,
            btc_vs_ema50=float(getattr(config, "_btc_vs_ema50", 0.0)),
        )
    except Exception as exc:
        log.warning("critic_dataset.log_candidate failed for %s [%s]: %s", sym, tf, exc)
        return ""


def _compute_features_from_data_sync(data: np.ndarray) -> tuple[np.ndarray, dict]:
    c = data["c"].astype(float)
    feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])
    return c, feat


async def _compute_features_from_data(data: np.ndarray) -> tuple[np.ndarray, dict]:
    return await asyncio.to_thread(_compute_features_from_data_sync, data)


async def _analyze_coin_live(sym: str, tf: str, data: np.ndarray) -> CoinReport:
    return await asyncio.to_thread(analyze_coin, sym, tf, data, False)


# ── Portfolio risk helpers ─────────────────────────────────────────────────────

def _get_coin_group(sym: str) -> Optional[str]:
    """Возвращает название группы монеты или None."""
    groups = getattr(config, "COIN_GROUPS", {})
    for grp, members in groups.items():
        if sym in members:
            return grp
    return None



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


def _is_fresh_priority_candidate(mode: str, catchup_snapshot: Optional[dict] = None) -> bool:
    if catchup_snapshot is not None:
        return True
    return _is_fresh_priority_mode(mode)


def _is_fresh_priority_mode(mode: str) -> bool:
    priority_modes = tuple(
        getattr(config, "FRESH_SIGNAL_PRIORITY_MODES", ("breakout", "retest", "impulse_speed", "impulse"))
    )
    return mode in priority_modes


def _time_block_bypass_allowed(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    vol_x: float,
    catchup_snapshot: Optional[dict],
    continuation_profile: bool = False,
) -> bool:
    if not getattr(config, "TIME_BLOCK_BYPASS_ENABLED", True):
        return False
    if tf == "15m":
        bypass_modes = tuple(getattr(config, "TIME_BLOCK_BYPASS_MODES", ("breakout", "retest", "impulse_speed")))
        if mode not in bypass_modes and catchup_snapshot is None:
            return False
        min_score = float(getattr(config, "TIME_BLOCK_BYPASS_SCORE_MIN", 48.0))
        min_vol = float(getattr(config, "TIME_BLOCK_BYPASS_VOL_X_MIN", 1.1))
    elif tf == "1h":
        if not getattr(config, "TIME_BLOCK_BYPASS_1H_ENABLED", False):
            return False
        bypass_modes = tuple(getattr(config, "TIME_BLOCK_BYPASS_1H_MODES", ("alignment", "trend", "strong_trend", "impulse_speed")))
        if mode not in bypass_modes and not continuation_profile:
            return False
        min_score = float(getattr(config, "TIME_BLOCK_BYPASS_1H_SCORE_MIN", 60.0))
        min_vol = float(getattr(config, "TIME_BLOCK_BYPASS_1H_VOL_X_MIN", 1.0))
    else:
        return False
    return candidate_score >= min_score and vol_x >= min_vol


def _time_block_1h_continuation_profile(
    *,
    tf: str,
    mode: str,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> bool:
    if tf != "1h" or not getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "TIME_BLOCK_BYPASS_1H_CONTINUATION_MODES",
            ("alignment", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return False
    return (
        slope >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_SLOPE_MIN", 0.08))
        and adx >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_ADX_MIN", 16.0))
        and float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MIN", 62.0))
        <= rsi
        <= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RSI_MAX", 78.0))
        and vol_x >= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_VOL_X_MIN", 1.0))
        and daily_range <= float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_RANGE_MAX", 6.5))
    )


def _time_block_1h_continuation_bonus(
    *,
    tf: str,
    mode: str,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> float:
    if not _time_block_1h_continuation_profile(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    ):
        return 0.0
    return float(getattr(config, "TIME_BLOCK_BYPASS_1H_CONTINUATION_SCORE_BONUS", 0.0))


def _time_block_price_edge_pct(*, price: float, ema20: float) -> float:
    if ema20 <= 0:
        return 0.0
    return max(0.0, ((price / ema20) - 1.0) * 100.0)


def _time_block_1h_prebypass_allowed(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    vol_x: float,
    price: float,
    ema20: float,
    continuation_profile: bool,
    repeat_count: int,
) -> bool:
    if tf != "1h" or not getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_ENABLED", False):
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "TIME_BLOCK_BYPASS_1H_PREBYPASS_MODES",
            ("alignment", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes or not continuation_profile:
        return False
    if repeat_count < int(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_CONFIRMATIONS", 2)):
        return False
    if candidate_score < float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_SCORE_MIN", 54.0)):
        return False
    if vol_x < float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_VOL_X_MIN", 1.0)):
        return False
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if price_edge > float(getattr(config, "TIME_BLOCK_BYPASS_1H_PREBYPASS_PRICE_EDGE_MAX_PCT", 2.2)):
        return False
    return True


def _late_1h_continuation_guard(
    *,
    tf: str,
    mode: str,
    continuation_profile: bool,
    candidate_score: float,
    price: float,
    ema20: float,
    rsi: float,
    daily_range: float,
) -> bool:
    if tf != "1h" or not getattr(config, "LATE_1H_CONTINUATION_GUARD_ENABLED", False):
        return False
    if not continuation_profile:
        return False
    guard_modes = tuple(
        getattr(config, "LATE_1H_CONTINUATION_GUARD_MODES", ("trend", "alignment"))
    )
    if mode not in guard_modes:
        return False
    if rsi < float(getattr(config, "LATE_1H_CONTINUATION_GUARD_RSI_MIN", 72.0)):
        return False
    if daily_range < float(getattr(config, "LATE_1H_CONTINUATION_GUARD_RANGE_MIN", 5.0)):
        return False
    if _time_block_price_edge_pct(price=price, ema20=ema20) < float(
        getattr(config, "LATE_1H_CONTINUATION_GUARD_PRICE_EDGE_MIN_PCT", 1.5)
    ):
        return False
    if candidate_score > float(getattr(config, "LATE_1H_CONTINUATION_GUARD_SCORE_MAX", 68.0)):
        return False
    return True


def _one_hour_impulse_speed_entry_guard(
    *,
    tf: str,
    mode: str,
    rsi: float,
    adx: float,
    daily_range: float,
    is_bull_day: bool = False,
    ext_atr: Optional[float] = None,
) -> Optional[str]:
    if tf != "1h" or mode != "impulse_speed":
        return None
    if not getattr(config, "IMPULSE_SPEED_1H_ENTRY_GUARD_ENABLED", False):
        return None
    if is_bull_day:
        rsi_max = float(getattr(config, "IMPULSE_SPEED_1H_RSI_MAX_BULL", 76.0))
        range_max = float(getattr(config, "IMPULSE_SPEED_1H_RANGE_MAX_BULL", 15.0))
    else:
        rsi_max = float(getattr(config, "IMPULSE_SPEED_1H_RSI_MAX", 70.0))
        range_max = float(getattr(config, "IMPULSE_SPEED_1H_RANGE_MAX", 7.0))
    if rsi > rsi_max:
        return f"1h impulse_speed guard: RSI {rsi:.1f} > {rsi_max:.1f}"
    adx_min = float(getattr(config, "IMPULSE_SPEED_1H_ADX_MIN", 20.0))
    if adx < adx_min:
        return f"1h impulse_speed guard: ADX {adx:.1f} < {adx_min:.1f}"

    # Prefer stateless ATR-extension over daily_range (time-of-day artifact).
    # Falls back to legacy daily_range when ext_atr is not provided
    # (unit tests, or if ATR is unavailable).
    use_ext = bool(getattr(config, "IMPULSE_SPEED_1H_USE_EXT_ATR", True))
    if use_ext and ext_atr is not None:
        if is_bull_day:
            ext_max = float(getattr(config, "IMPULSE_SPEED_1H_EXT_ATR_MAX_BULL", 9.0))
        else:
            ext_max = float(getattr(config, "IMPULSE_SPEED_1H_EXT_ATR_MAX", 6.0))
        if ext_atr > ext_max:
            return (f"1h impulse_speed guard: ext {ext_atr:.1f} ATR above EMA20 "
                    f"> {ext_max:.1f}")
    else:
        if daily_range > range_max:
            return f"1h impulse_speed guard: daily_range {daily_range:.2f}% > {range_max:.2f}%"
    return None


def _recent_positive_macd_peak(feat: dict, i: int, lookback: int) -> Optional[float]:
    macd_hist_arr = feat.get("macd_hist")
    if macd_hist_arr is None or i < 0:
        return None
    start = max(0, i - max(1, lookback) + 1)
    peak = None
    for j in range(start, i + 1):
        val = float(macd_hist_arr[j]) if np.isfinite(macd_hist_arr[j]) else np.nan
        if not np.isfinite(val) or val <= 0:
            continue
        peak = val if peak is None else max(peak, val)
    return peak


def _impulse_speed_entry_guard(
    *,
    tf: str,
    mode: str,
    feat: dict,
    i: int,
    price: float,
    ema20: float,
    rsi: float,
    adx: float,
    daily_range: float,
    is_bull_day: bool = False,
) -> Optional[str]:
    if mode != "impulse_speed":
        return None

    # Hard ADX floor — reject "impulse" signals with no trend strength.
    # Added 2026-04-18 after observing CRVUSDT ADX=17.5 (-2.17%) and
    # OXTUSDT ADX=11.3 enter on volume spikes with no directional trend.
    if tf == "15m":
        adx_floor = float(getattr(config, "IMPULSE_SPEED_15M_ADX_MIN", 20.0))
        if adx < adx_floor:
            return f"weak 15m impulse: ADX {adx:.1f} < {adx_floor:.0f}"
    if tf == "1h":
        adx_floor_1h = float(getattr(config, "IMPULSE_SPEED_1H_ADX_MIN", 18.0))
        if adx < adx_floor_1h:
            return f"weak 1h impulse: ADX {adx:.1f} < {adx_floor_1h:.0f}"

    if tf == "1h":
        # Compute stateless extension metric: price distance from 1h EMA20 in ATR.
        # Replaces daily_range (time-of-day artifact). Falls back to daily_range
        # if ATR unavailable.
        ext_atr: Optional[float] = None
        try:
            atr_arr = feat.get("atr")
            if atr_arr is not None and ema20 and i is not None:
                atr_now = float(atr_arr[i])
                import math as _m
                if _m.isfinite(atr_now) and atr_now > 0:
                    ext_atr = (float(price) - float(ema20)) / atr_now
        except Exception:
            ext_atr = None

        base_reason = _one_hour_impulse_speed_entry_guard(
            tf=tf,
            mode=mode,
            rsi=rsi,
            adx=adx,
            daily_range=daily_range,
            is_bull_day=is_bull_day,
            ext_atr=ext_atr,
        )
        if base_reason:
            return base_reason

    if not getattr(config, "IMPULSE_SPEED_LATE_GUARD_ENABLED", False):
        return None

    # Per-tf A/B kill-switches (added 2026-04-18 after Pareto sweep)
    if tf == "15m" and not getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_ENABLED", True):
        return None
    if tf == "1h" and not getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_ENABLED", True):
        return None

    if tf == "15m":
        if is_bull_day:
            rsi_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_RSI_MIN_BULL", 76.0))
            range_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_RANGE_MIN_BULL", 10.0))
            edge_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_PRICE_EDGE_MIN_PCT_BULL", 3.20))
        else:
            rsi_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_RSI_MIN", 66.0))
            range_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_RANGE_MIN", 5.0))
            edge_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_PRICE_EDGE_MIN_PCT", 1.8))
        fade_ratio_max = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_MACD_FADE_RATIO_MAX", 0.60))
        lookback = int(getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_MACD_PEAK_LOOKBACK", 8))
    elif tf == "1h":
        if is_bull_day:
            rsi_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_RSI_MIN_BULL", 74.0))
            range_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_RANGE_MIN_BULL", 12.0))
            edge_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_PRICE_EDGE_MIN_PCT_BULL", 3.50))
        else:
            rsi_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_RSI_MIN", 68.0))
            range_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_RANGE_MIN", 8.0))
            edge_min = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_PRICE_EDGE_MIN_PCT", 2.4))
        fade_ratio_max = float(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_MACD_FADE_RATIO_MAX", 0.68))
        lookback = int(getattr(config, "IMPULSE_SPEED_LATE_GUARD_1H_MACD_PEAK_LOOKBACK", 6))
    else:
        return None

    if rsi < rsi_min or daily_range < range_min:
        return None

    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if price_edge < edge_min:
        return None

    if tf == "15m":
        if is_bull_day:
            spike_rsi_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RSI_MIN_BULL", 80.0)
            )
            spike_range_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RANGE_MIN_BULL", 15.0)
            )
            spike_edge_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_PRICE_EDGE_MIN_PCT_BULL", 5.0)
            )
        else:
            spike_rsi_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RSI_MIN", 75.0)
            )
            spike_range_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_RANGE_MIN", 10.0)
            )
            spike_edge_min = float(
                getattr(config, "IMPULSE_SPEED_LATE_GUARD_15M_FRESH_SPIKE_PRICE_EDGE_MIN_PCT", 4.0)
            )
        if rsi >= spike_rsi_min and daily_range >= spike_range_min and price_edge >= spike_edge_min:
            return (
                f"late impulse_speed: overstretched 15m spike, "
                f"RSI {rsi:.1f}, daily_range {daily_range:.2f}%, edge {price_edge:.2f}% > EMA20"
            )

    macd_hist_arr = feat.get("macd_hist")
    if macd_hist_arr is None or i < 0 or i >= len(macd_hist_arr):
        return None
    macd_hist = float(macd_hist_arr[i]) if np.isfinite(macd_hist_arr[i]) else np.nan
    if not np.isfinite(macd_hist):
        return None

    peak = _recent_positive_macd_peak(feat, i, lookback)
    if peak is None or peak <= 0:
        return None

    fade_ratio = macd_hist / peak
    if fade_ratio > fade_ratio_max:
        return None

    return (
        f"late impulse_speed: MACD {fade_ratio:.2f} of local peak, "
        f"edge {price_edge:.2f}% > EMA20"
    )


def _late_impulse_speed_rotation_reason(
    *,
    tf: str,
    mode: str,
    rsi: float,
    daily_range: float,
) -> Optional[str]:
    if mode != "impulse_speed":
        return None
    if not getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_ENABLED", False):
        return None
    if tf == "15m":
        rsi_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_15M_RSI_MIN", 76.0))
        range_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_15M_RANGE_MIN", 5.0))
    elif tf == "1h":
        rsi_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_1H_RSI_MIN", 64.0))
        range_min = float(getattr(config, "IMPULSE_SPEED_ROTATION_GUARD_1H_RANGE_MIN", 16.0))
    else:
        return None
    if rsi >= rsi_min and daily_range >= range_min:
        return (
            f"late impulse_speed rotation guard: RSI {rsi:.1f} >= {rsi_min:.1f}, "
            f"daily_range {daily_range:.2f}% >= {range_min:.2f}%"
        )
    return None


def _continuation_profit_lock_active(
    *,
    tf: str,
    mode: str,
    entry_rsi: float,
    bars_elapsed: int,
    current_pnl: float,
    predictions: Dict[int, Optional[bool]],
) -> bool:
    if not getattr(config, "CONTINUATION_PROFIT_LOCK_ENABLED", False):
        return False
    allowed_tf = tuple(getattr(config, "CONTINUATION_PROFIT_LOCK_TF", ("1h",)))
    if tf not in allowed_tf:
        return False
    allowed_modes = tuple(
        getattr(
            config,
            "CONTINUATION_PROFIT_LOCK_MODES",
            ("trend", "alignment", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return False
    if bars_elapsed < int(getattr(config, "CONTINUATION_PROFIT_LOCK_MIN_BARS", 3)):
        return False
    entry_rsi_min = float(getattr(config, "CONTINUATION_PROFIT_LOCK_ENTRY_RSI_MIN", 70.0))
    continuation_pnl_min = float(
        getattr(config, "CONTINUATION_PROFIT_LOCK_CONTINUATION_PNL_PCT", 0.60)
    )
    if entry_rsi < entry_rsi_min and current_pnl < continuation_pnl_min:
        return False
    if predictions.get(3) is True:
        return True
    return current_pnl >= float(getattr(config, "CONTINUATION_PROFIT_LOCK_ACTIVATE_PNL_PCT", 0.20))


def _continuation_micro_exit_reason(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    data_15m: np.ndarray,
    feat_15m: dict,
) -> Optional[str]:
    if not getattr(config, "CONTINUATION_MICRO_EXIT_ENABLED", False):
        return None
    allowed_tf = tuple(getattr(config, "CONTINUATION_MICRO_EXIT_TF", ("1h",)))
    if tf not in allowed_tf:
        return None
    allowed_modes = tuple(
        getattr(
            config,
            "CONTINUATION_MICRO_EXIT_MODES",
            ("trend", "alignment", "strong_trend", "impulse_speed", "impulse"),
        )
    )
    if mode not in allowed_modes:
        return None
    if bars_elapsed < int(getattr(config, "CONTINUATION_MICRO_EXIT_MIN_BARS", 3)):
        return None
    idx = len(data_15m["c"]) - 2
    neg_bars = int(getattr(config, "CONTINUATION_MICRO_EXIT_MACD_NEG_BARS", 4))
    if idx < neg_bars - 1:
        return None
    macd_hist = feat_15m["macd_hist"]
    for j in range(idx - neg_bars + 1, idx + 1):
        if not np.isfinite(macd_hist[j]) or macd_hist[j] >= 0:
            return None
    close_15m = float(data_15m["c"][idx])
    ema20_15m = float(feat_15m["ema_fast"][idx]) if np.isfinite(feat_15m["ema_fast"][idx]) else np.nan
    rsi_15m = float(feat_15m["rsi"][idx]) if np.isfinite(feat_15m["rsi"][idx]) else np.nan
    if not all(np.isfinite([close_15m, ema20_15m, rsi_15m])):
        return None
    if rsi_15m > float(getattr(config, "CONTINUATION_MICRO_EXIT_RSI_MAX", 72.0)):
        return None
    if _time_block_price_edge_pct(price=close_15m, ema20=ema20_15m) > float(
        getattr(config, "CONTINUATION_MICRO_EXIT_PRICE_EDGE_MAX_PCT", 0.50)
    ):
        return None
    return f"15m micro-weakness after profit-lock: MACD<0 {neg_bars} bars, RSI {rsi_15m:.1f}"


def _short_mode_profit_lock_active(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    current_pnl: float,
    predictions: Dict[int, Optional[bool]],
) -> bool:
    if not getattr(config, "SHORT_MODE_PROFIT_LOCK_ENABLED", False):
        return False
    allowed_tf = tuple(getattr(config, "SHORT_MODE_PROFIT_LOCK_TF", ("15m",)))
    if tf not in allowed_tf:
        return False
    allowed_modes = tuple(getattr(config, "SHORT_MODE_PROFIT_LOCK_MODES", ("breakout", "retest")))
    if mode not in allowed_modes:
        return False
    if bars_elapsed < int(getattr(config, "SHORT_MODE_PROFIT_LOCK_MIN_BARS", 2)):
        return False
    if predictions.get(3) is True:
        return True
    return current_pnl >= float(getattr(config, "SHORT_MODE_PROFIT_LOCK_ACTIVATE_PNL_PCT", 0.30))


def _bump_time_block_streak(
    state: "MonitorState",
    *,
    sym: str,
    tf: str,
    mode: str,
    bar_ts: int,
) -> int:
    key = f"{sym}|{tf}"
    prev = state.time_block_streaks.get(key)
    bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
    if (
        prev
        and prev.get("mode") == mode
        and int(bar_ts) <= int(prev.get("bar_ts", 0)) + bar_ms * 2
    ):
        count = int(prev.get("count", 0)) + 1
    else:
        count = 1
    state.time_block_streaks[key] = {"mode": mode, "bar_ts": int(bar_ts), "count": count}
    return count


def _clear_time_block_streak(state: "MonitorState", *, sym: str, tf: str) -> None:
    state.time_block_streaks.pop(f"{sym}|{tf}", None)


def _remember_time_block(
    state: "MonitorState",
    *,
    sym: str,
    tf: str,
    mode: str,
    bar_ts: int,
) -> None:
    state.time_block_recent[sym] = {"tf": tf, "mode": mode, "bar_ts": int(bar_ts)}


def _time_block_retest_bonus(
    state: "MonitorState",
    *,
    sym: str,
    tf: str,
    mode: str,
    current_bar_ts: int,
    is_bull_day: bool,
) -> float:
    if mode != "retest" or is_bull_day:
        return 0.0
    recent = state.time_block_recent.get(sym)
    if not recent or str(recent.get("tf")) != tf:
        return 0.0
    grace_bars = int(getattr(config, "TIME_BLOCK_RETEST_GRACE_BARS", 0))
    if grace_bars <= 0:
        return 0.0
    bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
    blocked_ts = int(recent.get("bar_ts", 0))
    if blocked_ts <= 0:
        state.time_block_recent.pop(sym, None)
        return 0.0
    if current_bar_ts <= blocked_ts:
        return 0.0
    if current_bar_ts - blocked_ts > grace_bars * bar_ms:
        state.time_block_recent.pop(sym, None)
        return 0.0
    return float(getattr(config, "TIME_BLOCK_RETEST_SCORE_BONUS", 0.0))


def _is_weak_exit_reason(reason: Optional[str]) -> bool:
    return bool(reason) and "WEAK:" in str(reason)


def _min_weak_exit_bars(mode: Optional[str]) -> int:
    mode_name = str(mode or "")
    if mode_name == "impulse_speed":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_IMPULSE_SPEED", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "retest":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_RETEST", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "breakout":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_BREAKOUT", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    if mode_name == "trend":
        return int(getattr(config, "MIN_WEAK_EXIT_BARS_TREND", getattr(config, "MIN_WEAK_EXIT_BARS", 2)))
    return int(getattr(config, "MIN_WEAK_EXIT_BARS", 2))


def _trend_hold_weak_exit_active(
    *,
    pos: "OpenPosition",
    feat: dict,
    idx: int,
    close_now: float,
    current_pnl: float,
    tf: str,
) -> bool:
    if not getattr(config, "TREND_HOLD_WEAK_EXIT_ENABLED", False):
        return False
    if tf not in tuple(getattr(config, "TREND_HOLD_WEAK_EXIT_TF", ("15m",))):
        return False
    if str(getattr(pos, "signal_mode", "")) not in tuple(
        getattr(config, "TREND_HOLD_WEAK_EXIT_MODES", ("impulse_speed", "trend", "alignment"))
    ):
        return False
    if int(getattr(pos, "bars_elapsed", 0)) < int(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_BARS", 5)):
        return False
    if current_pnl < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_PNL_PCT", 0.75)):
        return False
    if float(getattr(pos, "candidate_score_at_entry", 0.0)) < float(
        getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_ENTRY_SCORE", 70.0)
    ):
        return False

    ema20_arr = feat.get("ema_fast")
    ema50_arr = feat.get("ema_slow")
    ema200_arr = feat.get("ema200")
    adx_arr = feat.get("adx")
    slope_arr = feat.get("slope")
    if any(arr is None or idx >= len(arr) for arr in (ema20_arr, ema50_arr, ema200_arr, adx_arr, slope_arr)):
        return False

    ema20 = float(ema20_arr[idx])
    ema50 = float(ema50_arr[idx])
    ema200 = float(ema200_arr[idx])
    adx = float(adx_arr[idx])
    slope = float(slope_arr[idx])
    if not all(np.isfinite(v) for v in (ema20, ema50, ema200, adx, slope, close_now)):
        return False
    if close_now < ema20:
        return False
    if not (ema20 >= ema50 >= ema200):
        return False
    if adx < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_ADX", 24.0)):
        return False
    if slope < float(getattr(config, "TREND_HOLD_WEAK_EXIT_MIN_SLOPE_PCT", 0.10)):
        return False
    return True


def _apply_trend_hold_weak_exit_override(
    *,
    pos: "OpenPosition",
    feat: dict,
    idx: int,
    close_now: float,
    current_pnl: float,
    tf: str,
) -> bool:
    if not _trend_hold_weak_exit_active(
        pos=pos,
        feat=feat,
        idx=idx,
        close_now=close_now,
        current_pnl=current_pnl,
        tf=tf,
    ):
        return False
    atr_arr = feat.get("atr")
    if atr_arr is None or idx >= len(atr_arr) or not np.isfinite(atr_arr[idx]):
        return False
    atr_now = float(atr_arr[idx])
    if atr_now <= 0:
        return False
    tight_k = min(
        float(getattr(pos, "trail_k", 2.0)),
        float(getattr(config, "TREND_HOLD_WEAK_EXIT_TIGHTEN_ATR_K", 1.4)),
    )
    new_trail = close_now - tight_k * atr_now
    if new_trail > float(getattr(pos, "trail_stop", 0.0)):
        pos.trail_stop = new_trail
    return True


def _cooldown_bars_after_exit(
    mode: str,
    reason: Optional[str],
    *,
    tf: Optional[str] = None,
    pnl_pct: Optional[float] = None,
) -> int:
    base = int(getattr(config, "COOLDOWN_BARS", 8))
    if (
        getattr(config, "PROFITABLE_WEAK_EXIT_SKIP_COOLDOWN", False)
        and _is_weak_exit_reason(reason)
        and pnl_pct is not None
        and pnl_pct >= float(getattr(config, "PROFITABLE_WEAK_EXIT_COOLDOWN_PNL_MIN", 0.0))
        and (tf is None or tf in tuple(getattr(config, "PROFITABLE_WEAK_EXIT_TF", ("1h",))))
    ):
        return 0
    if _is_weak_exit_reason(reason) and mode in ("trend", "alignment"):
        return max(base, int(getattr(config, "WEAK_REENTRY_COOLDOWN_BARS", base)))
    return base


def _fast_loss_ema_exit_reason(
    *,
    tf: str,
    mode: str,
    bars_elapsed: int,
    current_pnl: float,
    close_now: float,
    ema20: float,
    rsi: float,
) -> Optional[str]:
    if not getattr(config, "FAST_LOSS_EMA_EXIT_ENABLED", False):
        return None
    if tf not in tuple(getattr(config, "FAST_LOSS_EMA_EXIT_TF", ("15m",))):
        return None
    if mode not in tuple(getattr(config, "FAST_LOSS_EMA_EXIT_MODES", ("retest", "alignment", "trend", "breakout"))):
        return None
    if bars_elapsed < int(getattr(config, "FAST_LOSS_EMA_EXIT_MIN_BARS", 1)):
        return None
    if not np.isfinite(ema20) or close_now >= ema20:
        return None
    if current_pnl > float(getattr(config, "FAST_LOSS_EMA_EXIT_PNL_MAX", 0.0)):
        return None
    if not np.isfinite(rsi) or rsi > float(getattr(config, "FAST_LOSS_EMA_EXIT_RSI_MAX", 55.0)):
        return None
    return f"первое закрытие ниже EMA20 ({ema20:.6g}) в убыточной сделке"


def _time_exit_should_wait(feat: dict, idx: int, close_now: float) -> bool:
    if not getattr(config, "TIME_EXIT_TREND_CONTINUATION_ENABLED", False):
        return False
    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    ema20 = float(ema20_arr[idx]) if ema20_arr is not None and np.isfinite(ema20_arr[idx]) else np.nan
    slope_arr = feat.get("ema20_slope", feat.get("slope"))
    slope = float(slope_arr[idx]) if slope_arr is not None and np.isfinite(slope_arr[idx]) else np.nan
    rsi = float(feat["rsi"][idx]) if np.isfinite(feat["rsi"][idx]) else np.nan
    macd_hist_arr = feat.get("macd_hist")
    macd_hist = float(macd_hist_arr[idx]) if macd_hist_arr is not None and np.isfinite(macd_hist_arr[idx]) else np.nan
    if bool(getattr(config, "TIME_EXIT_CONTINUE_CLOSE_ABOVE_EMA20", True)):
        if not np.isfinite(ema20) or close_now < ema20:
            return False
    if not np.isfinite(slope) or slope < float(getattr(config, "TIME_EXIT_CONTINUE_SLOPE_MIN", 0.0)):
        return False
    if not np.isfinite(rsi) or rsi < float(getattr(config, "TIME_EXIT_CONTINUE_RSI_MIN", 50.0)):
        return False
    if np.isfinite(macd_hist) and macd_hist < float(getattr(config, "TIME_EXIT_CONTINUE_MACD_HIST_MIN", 0.0)):
        return False
    return True


def _post_entry_quality_recheck_reason(
    pos: "OpenPosition",
    feat: dict,
    i: int,
) -> Optional[str]:
    if not getattr(config, "ENTRY_QUALITY_RECHECK_ENABLED", False):
        return None
    mode = str(getattr(pos, "signal_mode", "trend"))
    if mode not in tuple(getattr(config, "ENTRY_QUALITY_RECHECK_MODES", ())):
        return None
    if int(getattr(pos, "bars_elapsed", 0)) > int(getattr(config, "ENTRY_QUALITY_RECHECK_MAX_BARS", 0)):
        return None

    marker_set = tuple(str(x).lower() for x in getattr(config, "ENTRY_QUALITY_RECHECK_REASON_MARKERS", ()))
    if not marker_set:
        return None

    if mode == "alignment":
        ok, reason = check_alignment_conditions(feat, i, tf=str(getattr(pos, "tf", "")))
    else:
        return None

    if ok or not reason:
        return None

    reason_lc = str(reason).lower()
    if not any(marker in reason_lc for marker in marker_set):
        return None
    return f"WEAK: quality recheck failed - {reason}"


def _ranker_position_cleanup_reason(
    pos: "OpenPosition",
    feat: dict,
    i: int,
    *,
    close_now: float,
) -> Optional[str]:
    if not getattr(config, "RANKER_POSITION_CLEANUP_ENABLED", False):
        return None

    tf = str(getattr(pos, "tf", ""))
    mode = str(getattr(pos, "signal_mode", ""))
    bars_elapsed = int(getattr(pos, "bars_elapsed", 0))
    ranker_final = float(getattr(pos, "ranker_final_score", 0.0))
    ranker_ev = float(getattr(pos, "ranker_ev", 0.0))
    top_gainer_prob = float(getattr(pos, "ranker_top_gainer_prob", 0.0))
    capture_ratio_pred = float(getattr(pos, "ranker_capture_ratio_pred", 0.0))
    quality_proba = float(getattr(pos, "ranker_quality_proba", 0.0))
    current_pnl = pos.pnl_pct(float(close_now))
    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    ema20_now = (
        float(ema20_arr[i])
        if ema20_arr is not None and i < len(ema20_arr) and np.isfinite(ema20_arr[i])
        else np.nan
    )

    if (
        tf == "15m"
        and mode in tuple(str(x) for x in getattr(config, "RANKER_POSITION_CLEANUP_15M_MODES", ("impulse_speed",)))
    ):
        horizon_bars = max((int(h) for h in (getattr(pos, "prediction_horizons", ()) or (0,))), default=0)
        min_bars = max(
            int(getattr(config, "RANKER_POSITION_CLEANUP_15M_MIN_BARS", 8)),
            horizon_bars,
        )
        if bars_elapsed < min_bars:
            return None
        below_ema20 = bool(np.isfinite(ema20_now) and close_now < ema20_now)
        if (
            ranker_final <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_FINAL_MAX", -0.50))
            and ranker_ev <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_EV_MAX", -0.60))
            and top_gainer_prob <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_TOP_GAINER_MAX", 0.28))
            and capture_ratio_pred <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_CAPTURE_MAX", 0.05))
        ):
            require_below_ema20 = bool(getattr(config, "RANKER_POSITION_CLEANUP_15M_REQUIRE_BELOW_EMA20", True))
            if (not require_below_ema20) or below_ema20:
                return (
                    f"ranker cleanup: stale 15m {mode} "
                    f"(final {ranker_final:.2f}, EV {ranker_ev:.2f}, TG {top_gainer_prob:.2f}, "
                    f"CAP {capture_ratio_pred:.2f}, close<EMA20)"
                )
        if (
            getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_ENABLED", False)
            and bars_elapsed >= max(min_bars, horizon_bars)
            and ranker_final <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_FINAL_MAX", -0.60))
            and ranker_ev <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_EV_MAX", -0.70))
            and quality_proba <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_QUALITY_MAX", 0.50))
            and top_gainer_prob <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_TOP_GAINER_MAX", 0.26))
            and capture_ratio_pred <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_CAPTURE_MAX", 0.05))
            and current_pnl <= float(getattr(config, "RANKER_POSITION_CLEANUP_15M_PROACTIVE_PNL_MAX", 1.25))
        ):
            return (
                f"ranker cleanup: stale 15m {mode} after fast horizons "
                f"(final {ranker_final:.2f}, EV {ranker_ev:.2f}, Q {quality_proba:.2f}, "
                f"TG {top_gainer_prob:.2f}, CAP {capture_ratio_pred:.2f}, pnl {current_pnl:+.2f}%)"
            )

    if (
        tf == "1h"
        and getattr(config, "RANKER_POSITION_CLEANUP_1H_RETEST_ENABLED", False)
        and mode == "retest"
    ):
        if bars_elapsed < int(getattr(config, "RANKER_POSITION_CLEANUP_1H_RETEST_MIN_BARS", 3)):
            return None
        pred3 = getattr(pos, "predictions", {}).get(3)
        below_ema20 = bool(np.isfinite(ema20_now) and close_now < ema20_now)
        if not (pred3 is False or below_ema20):
            return None
        if (
            ranker_final <= float(getattr(config, "RANKER_POSITION_CLEANUP_1H_RETEST_FINAL_MAX", -1.50))
            and quality_proba <= float(getattr(config, "RANKER_POSITION_CLEANUP_1H_RETEST_QUALITY_MAX", 0.35))
            and current_pnl <= float(getattr(config, "RANKER_POSITION_CLEANUP_1H_RETEST_PNL_MAX", 0.50))
        ):
            return (
                f"ranker cleanup: weak 1h retest "
                f"(final {ranker_final:.2f}, Q {quality_proba:.2f}, pnl {current_pnl:+.2f}%)"
            )
    return None


def _entry_signal_score(
    *,
    mode: str,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> float:
    price_edge = max(0.0, ((price / ema20) - 1.0) * 100.0) if ema20 > 0 else 0.0
    range_room = max(0.0, 10.0 - daily_range)
    rsi_balance = max(0.0, 20.0 - abs(rsi - 60.0))
    return (
        _signal_priority(mode) * 10.0 +
        max(0.0, slope) * 25.0 +
        max(0.0, adx - 15.0) * 0.6 +
        max(0.0, vol_x) * 2.5 +
        price_edge * 2.0 +
        range_room * 0.2 +
        rsi_balance * 0.05
    )


def _top_mover_score_bonus(today_change_pct: float) -> float:
    if not getattr(config, "TOP_MOVER_SCORE_ENABLED", False):
        return 0.0
    min_move = float(getattr(config, "TOP_MOVER_MIN_DAY_CHANGE_PCT", 1.5))
    cap_move = float(getattr(config, "TOP_MOVER_DAY_CHANGE_CAP_PCT", 8.0))
    if today_change_pct <= min_move:
        return 0.0
    capped_move = min(today_change_pct, cap_move)
    return max(0.0, capped_move - min_move) * float(getattr(config, "TOP_MOVER_SCORE_WEIGHT", 1.6))


def _forecast_return_score_bonus(forecast_return_pct: float) -> float:
    if forecast_return_pct >= 0:
        return forecast_return_pct * float(getattr(config, "FORECAST_RETURN_SCORE_WEIGHT", 18.0))
    return forecast_return_pct * float(getattr(config, "FORECAST_RETURN_NEGATIVE_WEIGHT", 10.0))


def _entry_score_floor(tf: str) -> float:
    if tf == "1h":
        return float(getattr(config, "ENTRY_SCORE_MIN_1H", 0.0))
    return float(getattr(config, "ENTRY_SCORE_MIN_15M", 0.0))


def _entry_score_borderline_bypass_ok(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    min_score: float,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
) -> bool:
    if not getattr(config, "ENTRY_SCORE_BORDERLINE_BYPASS_ENABLED", False):
        return False
    if tf == "1h" and not getattr(config, "ENTRY_SCORE_BORDERLINE_ALLOW_1H", False):
        return False
    allowed_modes = tuple(str(x) for x in getattr(config, "ENTRY_SCORE_BORDERLINE_MODES", ()))
    if allowed_modes and mode not in allowed_modes:
        return False
    max_deficit = float(
        getattr(
            config,
            "ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_1H" if tf == "1h" else "ENTRY_SCORE_BORDERLINE_MAX_DEFICIT_15M",
            0.0,
        )
    )
    if candidate_score < (min_score - max_deficit):
        return False
    if ema20 <= 0:
        return False
    price_edge = ((price / ema20) - 1.0) * 100.0
    if price_edge > float(getattr(config, "ENTRY_SCORE_BORDERLINE_PRICE_EDGE_MAX_PCT", 2.8)):
        return False
    if slope < float(getattr(config, "ENTRY_SCORE_BORDERLINE_SLOPE_MIN", 0.35)):
        return False
    if adx < float(getattr(config, "ENTRY_SCORE_BORDERLINE_ADX_MIN", 28.0)):
        return False
    if vol_x < float(getattr(config, "ENTRY_SCORE_BORDERLINE_VOL_MIN", 1.2)):
        return False
    if daily_range > float(getattr(config, "ENTRY_SCORE_BORDERLINE_DAILY_RANGE_MAX", 6.5)):
        return False
    if rsi < float(getattr(config, "ENTRY_SCORE_BORDERLINE_RSI_MIN", 52.0)):
        return False
    if rsi > float(getattr(config, "ENTRY_SCORE_BORDERLINE_RSI_MAX", 74.5)):
        return False
    return True


def _entry_score_continuation_bypass_ok(
    *,
    tf: str,
    mode: str,
    candidate_score: float,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    continuation_profile: bool,
    is_bull_day: bool,
) -> bool:
    if not getattr(config, "ENTRY_SCORE_CONTINUATION_BYPASS_ENABLED", False):
        return False
    if tf != "1h" or not getattr(config, "ENTRY_SCORE_CONTINUATION_1H_ENABLED", False):
        return False
    if not continuation_profile:
        return False
    if getattr(config, "ENTRY_SCORE_CONTINUATION_REQUIRE_BULL_DAY", True) and not is_bull_day:
        return False
    allowed_modes = tuple(str(x) for x in getattr(config, "ENTRY_SCORE_CONTINUATION_MODES", ()))
    if allowed_modes and mode not in allowed_modes:
        return False
    if candidate_score < float(getattr(config, "ENTRY_SCORE_CONTINUATION_SCORE_MIN_1H", 42.0)):
        return False
    if ema20 <= 0:
        return False
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if price_edge > float(getattr(config, "ENTRY_SCORE_CONTINUATION_PRICE_EDGE_MAX_PCT", 2.0)):
        return False
    if slope < float(getattr(config, "ENTRY_SCORE_CONTINUATION_SLOPE_MIN", 0.30)):
        return False
    if adx < float(getattr(config, "ENTRY_SCORE_CONTINUATION_ADX_MIN", 16.0)):
        return False
    if vol_x < float(getattr(config, "ENTRY_SCORE_CONTINUATION_VOL_MIN", 0.90)):
        return False
    if daily_range > float(getattr(config, "ENTRY_SCORE_CONTINUATION_DAILY_RANGE_MAX", 8.5)):
        return False
    if rsi < float(getattr(config, "ENTRY_SCORE_CONTINUATION_RSI_MIN", 54.0)):
        return False
    if rsi > float(getattr(config, "ENTRY_SCORE_CONTINUATION_RSI_MAX", 72.0)):
        return False
    return True


def _retest_1h_mtf_confirm_reason(
    *,
    mode: Optional[str],
    close_val: float,
    ema20_15m: float,
    macd_hist: float,
    macd_prev: float,
    rsi_val: float,
) -> Optional[str]:
    if mode != "retest" or not getattr(config, "RETEST_1H_MTF_CONFIRM_ENABLED", False):
        return None
    if ema20_15m <= 0:
        return None
    ema_slip_pct = float(getattr(config, "RETEST_1H_MTF_EMA20_SLIP_PCT", 0.15))
    if close_val < ema20_15m * (1.0 - ema_slip_pct / 100.0):
        return (
            f"15m retest micro weak: close {close_val:.6g} below EMA20 {ema20_15m:.6g} "
            f"by > {ema_slip_pct:.2f}%"
        )
    macd_floor = close_val * float(getattr(config, "RETEST_1H_MTF_MACD_MIN_REL", 0.0))
    if macd_hist < macd_floor:
        return f"15m retest micro weak: MACD {macd_hist:.4g} < floor {macd_floor:.4g}"
    if bool(getattr(config, "RETEST_1H_MTF_REQUIRE_MACD_RISING", True)) and macd_hist < macd_prev:
        return f"15m retest micro weak: MACD {macd_hist:.4g} < prev {macd_prev:.4g}"
    rsi_min = float(getattr(config, "RETEST_1H_MTF_RSI_MIN", 48.0))
    rsi_max = float(getattr(config, "RETEST_1H_MTF_RSI_MAX", 72.0))
    if rsi_val < rsi_min:
        return f"15m retest micro weak: RSI {rsi_val:.1f} < {rsi_min:.1f}"
    if rsi_val > rsi_max:
        return f"15m retest micro weak: RSI {rsi_val:.1f} > {rsi_max:.1f}"
    return None


def _trend_entry_quality_guard_reason(
    *,
    tf: str,
    mode: str,
    price: float,
    ema20: float,
    slope: float,
    adx: float,
    rsi: float,
    vol_x: float,
    daily_range: float,
    forecast_return_pct: float,
    is_bull_day: bool = False,
) -> Optional[str]:
    if tf != "15m" or mode != "trend":
        return None
    if not getattr(config, "TREND_15M_QUALITY_GUARD_ENABLED", False):
        return None
    if ema20 <= 0:
        return None
    price_edge = _time_block_price_edge_pct(price=price, ema20=ema20)
    if is_bull_day:
        price_edge_max = float(getattr(config, "TREND_15M_QUALITY_PRICE_EDGE_MAX_BULL_DAY_PCT", 4.00))
    else:
        price_edge_max = float(getattr(config, "TREND_15M_QUALITY_PRICE_EDGE_MAX_PCT", 3.20))
    if price_edge > price_edge_max:
        return f"trend quality guard: price edge {price_edge:.2f}% > {price_edge_max:.2f}%"
    if is_bull_day:
        daily_range_max = float(getattr(config, "TREND_15M_QUALITY_DAILY_RANGE_MAX_BULL_DAY", 14.0))
    else:
        daily_range_max = float(getattr(config, "TREND_15M_QUALITY_DAILY_RANGE_MAX", 10.0))
    if daily_range > daily_range_max:
        return f"trend quality guard: daily_range {daily_range:.2f}% > {daily_range_max:.2f}%"
    if is_bull_day:
        rsi_max = float(getattr(config, "TREND_15M_QUALITY_RSI_MAX_BULL_DAY", 76.0))
    else:
        rsi_max = float(getattr(config, "TREND_15M_QUALITY_RSI_MAX", 72.0))
    if rsi > rsi_max:
        return f"trend quality guard: RSI {rsi:.1f} > {rsi_max:.1f}"
    forecast_min = float(getattr(config, "TREND_15M_QUALITY_FORECAST_MIN", 0.25))
    if forecast_return_pct >= forecast_min:
        return None
    alt_vol_min = float(getattr(config, "TREND_15M_QUALITY_ALT_VOL_MIN", 1.20))
    alt_adx_min = float(getattr(config, "TREND_15M_QUALITY_ALT_ADX_MIN", 24.0))
    alt_slope_min = float(getattr(config, "TREND_15M_QUALITY_ALT_SLOPE_MIN", 0.35))
    if vol_x >= alt_vol_min and adx >= alt_adx_min and slope >= alt_slope_min:
        return None
    return (
        "trend quality guard: weak 15m trend "
        f"(forecast {forecast_return_pct:.3f} < {forecast_min:.3f}, "
        f"vol {vol_x:.2f}, ADX {adx:.1f}, slope {slope:.3f})"
    )


def _trail_min_buffer_pct(mode: str) -> float:
    """
    Returns the minimum trail-stop buffer as a fraction of price for a given mode.
    Anti-whipsaw fix (2026-04-26 ALGOUSDT case): bandit picks "tight" arm
    (trail_k=2.12) which is statistically profitable on average, but on coins
    with compressed 15m ATR (~0.4% of price) and wide daily-range (5%+),
    2.12*ATR=0.86% buffer is too tight — noise pierces stop.
    Applied as floor: actual_buffer = max(trail_k*ATR, MIN_PCT*price).
    """
    if not getattr(config, "TRAIL_MIN_BUFFER_PCT_ENABLED", False):
        return 0.0
    if mode == "impulse_speed":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED", 0.015))
    if mode == "strong_trend":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_STRONG_TREND", 0.015))
    if mode == "impulse":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_IMPULSE", 0.012))
    if mode == "trend":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_TREND", 0.0))
    if mode == "alignment":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_ALIGNMENT", 0.0))
    if mode == "retest":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_RETEST", 0.0))
    if mode == "breakout":
        return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_BREAKOUT", 0.0))
    return float(getattr(config, "TRAIL_MIN_BUFFER_PCT_DEFAULT", 0.0))


def _compute_trail_buffer(*, price: float, trail_k: float, atr: float, mode: str) -> float:
    """
    Computes the trail-stop buffer (absolute price units), enforcing the
    mode-specific minimum buffer floor. Returns max(trail_k*ATR, MIN_PCT*price).
    """
    atr_buf = trail_k * atr if atr > 0 else 0.0
    min_pct = _trail_min_buffer_pct(mode)
    min_buf = min_pct * price if (price > 0 and min_pct > 0) else 0.0
    return max(atr_buf, min_buf)


def _mode_daily_range_guard_reason(
    *,
    mode: str,
    tf: str,
    daily_range: float,
    slope: float,
) -> Optional[str]:
    """
    Regime gate: block signals on quiet-market days (low daily_range / slope).
    Backtest 2026-04-24, 60d window:
      alignment/15m range>=4.0 -> +5.9pp precision (23%->28.9%)
      trend/15m     range>=4.0 -> +17.1pp precision (29.9%->47.0%)
      alignment/1h  range>=5.0 -> +8.0pp precision (30.9%->38.9%)
      trend/1h      slope>=0.50 -> +4.6pp precision (14.1%->18.8%)
      impulse_speed/1h range>=7.0 -> +3.3pp precision (16.5%->19.8%)
    """
    if not getattr(config, "MODE_RANGE_QUALITY_GUARD_ENABLED", True):
        return None

    if mode == "alignment" and tf == "15m":
        rmin = float(getattr(config, "ALIGNMENT_15M_RANGE_MIN", 4.0))
        if daily_range < rmin:
            return (
                f"mode_range_quality: alignment/15m daily_range "
                f"{daily_range:.2f}% < {rmin:.2f}%"
            )

    elif mode == "trend" and tf == "15m":
        rmin = float(getattr(config, "TREND_15M_RANGE_MIN", 4.0))
        if daily_range < rmin:
            return (
                f"mode_range_quality: trend/15m daily_range "
                f"{daily_range:.2f}% < {rmin:.2f}%"
            )

    elif mode == "alignment" and tf == "1h":
        rmin = float(getattr(config, "ALIGNMENT_1H_RANGE_MIN", 5.0))
        if daily_range < rmin:
            return (
                f"mode_range_quality: alignment/1h daily_range "
                f"{daily_range:.2f}% < {rmin:.2f}%"
            )

    elif mode == "trend" and tf == "1h":
        smin = float(getattr(config, "TREND_1H_SLOPE_MIN", 0.50))
        if slope < smin:
            return (
                f"mode_range_quality: trend/1h slope "
                f"{slope:.3f}% < {smin:.3f}%"
            )

    elif mode == "impulse_speed" and tf == "1h":
        rmin = float(getattr(config, "IMPULSE_SPEED_1H_RANGE_MIN", 7.0))
        if daily_range < rmin:
            return (
                f"mode_range_quality: impulse_speed/1h daily_range "
                f"{daily_range:.2f}% < {rmin:.2f}%"
            )

    return None


def _ranker_entry_veto_reason(
    *,
    tf: str,
    mode: str,
    ranker_proba: Optional[float],
    candidate_score: float,
    forecast_return_pct: float,
) -> Optional[str]:
    if ranker_proba is None or not getattr(config, "ML_CANDIDATE_RANKER_VETO_ENABLED", False):
        return None
    allowed_tf = tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_VETO_TF", ()))
    if allowed_tf and tf not in allowed_tf:
        return None
    allowed_modes = tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_VETO_MODES", ()))
    if allowed_modes and mode not in allowed_modes:
        return None
    proba_max = float(getattr(config, "ML_CANDIDATE_RANKER_VETO_PROBA_MAX", 0.20))
    if ranker_proba > proba_max:
        return None
    score_max = float(getattr(config, "ML_CANDIDATE_RANKER_VETO_SCORE_MAX", 60.0))
    if candidate_score > score_max:
        return None
    forecast_max = float(getattr(config, "ML_CANDIDATE_RANKER_VETO_FORECAST_MAX", 0.25))
    if forecast_return_pct > forecast_max:
        return None
    return (
        f"ranker veto: proba {ranker_proba:.2f} <= {proba_max:.2f} "
        f"for weak {tf} {mode}"
    )


def _ranker_hard_veto_reason(
    *,
    tf: str,
    mode: str,
    ranker_info: Optional[Dict[str, float]],
) -> Optional[str]:
    if not ranker_info or not getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_ENABLED", False):
        return None

    ranker_final = float(ranker_info.get("final_score", 0.0))
    top_gainer_prob = float(ranker_info.get("top_gainer_prob", 0.0))
    quality_proba = float(ranker_info.get("quality_proba", 0.0))
    ev_raw = float(ranker_info.get("ev_raw", 0.0))
    capture_ratio_pred = float(ranker_info.get("capture_ratio_pred", 0.0))

    if tf in tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_TF", ())):
        allowed_modes = tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_MODES", ()))
        if allowed_modes and mode not in allowed_modes:
            return None
        if (
            getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_ENABLED", False)
            and mode in tuple(
                str(x)
                for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_MODES", ("impulse_speed",))
            )
        ):
            final_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_FINAL_MAX", -0.50))
            ev_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_EV_MAX", -0.60))
            quality_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_QUALITY_MAX", 0.50))
            tg_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_TOP_GAINER_MAX", 0.28))
            cap_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_IMPULSE_CAPTURE_MAX", 0.05))
            if (
                ranker_final <= final_max
                and ev_raw <= ev_max
                and quality_proba <= quality_max
                and top_gainer_prob <= tg_max
                and capture_ratio_pred <= cap_max
            ):
                return (
                    f"ranker hard veto: weak 15m impulse "
                    f"(final {ranker_final:.2f}, EV {ev_raw:.2f}, Q {quality_proba:.2f}, "
                    f"TG {top_gainer_prob:.2f}, CAP {capture_ratio_pred:.2f})"
                )
        final_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_FINAL_MAX", -0.75))
        tg_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_15M_TOP_GAINER_MAX", 0.20))
        if ranker_final <= final_max and top_gainer_prob <= tg_max:
            return (
                f"ranker hard veto: final {ranker_final:.2f} <= {final_max:.2f} "
                f"and TG {top_gainer_prob:.2f} <= {tg_max:.2f}"
            )
        return None

    if tf in tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_TF", ())):
        allowed_modes = tuple(str(x) for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_MODES", ()))
        if allowed_modes and mode not in allowed_modes:
            return None
        if (
            getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_ENABLED", False)
            and mode in tuple(
                str(x)
                for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_MODES", ("retest",))
            )
        ):
            final_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_FINAL_MAX", -1.20))
            quality_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_QUALITY_MAX", 0.35))
            ev_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_RETEST_EV_MAX", -1.25))
            if ranker_final <= final_max and quality_proba <= quality_max and ev_raw <= ev_max:
                return (
                    f"ranker hard veto: weak 1h retest "
                    f"(final {ranker_final:.2f}, EV {ev_raw:.2f}, Q {quality_proba:.2f})"
                )
        if (
            getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_ENABLED", False)
            and mode in tuple(
                str(x)
                for x in getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_MODES", ("impulse_speed", "impulse"))
            )
        ):
            final_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_FINAL_MAX", -0.20))
            ev_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_EV_MAX", -0.30))
            quality_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_QUALITY_MAX", 0.58))
            tg_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_TOP_GAINER_MAX", 0.25))
            cap_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_IMPULSE_CAPTURE_MAX", 0.08))
            if (
                ranker_final <= final_max
                and ev_raw <= ev_max
                and quality_proba <= quality_max
                and top_gainer_prob <= tg_max
                and capture_ratio_pred <= cap_max
            ):
                return (
                    f"ranker hard veto: weak 1h impulse "
                    f"(final {ranker_final:.2f}, EV {ev_raw:.2f}, Q {quality_proba:.2f}, "
                    f"TG {top_gainer_prob:.2f}, CAP {capture_ratio_pred:.2f})"
                )
        final_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_FINAL_MAX", -1.50))
        tg_max = float(getattr(config, "ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX", 0.25))
        if ranker_final <= final_max and top_gainer_prob <= tg_max:
            return f"ranker hard veto: final {ranker_final:.2f} <= {final_max:.2f} for {tf} {mode}"
    return None


def _mtf_soft_penalty_from_reason(reason: str) -> float:
    if reason.startswith("15m soft-pass:") or reason.startswith("15Рј soft-pass:"):
        return float(getattr(config, "MTF_SOFT_PASS_PENALTY", 0.0))
    return 0.0


def _position_signal_score(pos: "OpenPosition") -> float:
    base_score = _entry_signal_score(
        mode=getattr(pos, "signal_mode", "trend"),
        price=float(getattr(pos, "entry_price", 0.0)),
        ema20=float(getattr(pos, "entry_ema20", 0.0)),
        slope=float(getattr(pos, "entry_slope", 0.0)),
        adx=float(getattr(pos, "entry_adx", 0.0)),
        rsi=float(getattr(pos, "entry_rsi", 50.0)),
        vol_x=float(getattr(pos, "entry_vol_x", 1.0)),
        daily_range=0.0,
    )
    return (
        base_score
        + _top_mover_score_bonus(float(getattr(pos, "today_change_pct", 0.0)))
        + _forecast_return_score_bonus(float(getattr(pos, "forecast_return_pct", 0.0)))
    )


def _ranker_rotation_bonus(*, final_score: float, top_gainer_prob: float) -> float:
    return (
        float(final_score) * float(getattr(config, "PORTFOLIO_REPLACE_RANKER_FINAL_WEIGHT", 6.0))
        + float(top_gainer_prob) * float(getattr(config, "PORTFOLIO_REPLACE_TOP_GAINER_WEIGHT", 10.0))
    )


def _position_rotation_score(pos: "OpenPosition") -> float:
    return _position_signal_score(pos) + _ranker_rotation_bonus(
        final_score=float(getattr(pos, "ranker_final_score", 0.0)),
        top_gainer_prob=float(getattr(pos, "ranker_top_gainer_prob", 0.0)),
    )


def _candidate_rotation_score(candidate_score: float, ranker_info: Optional[Dict[str, float]]) -> float:
    if not ranker_info or not getattr(config, "PORTFOLIO_REPLACE_RANKER_ENABLED", True):
        return float(candidate_score)
    return float(candidate_score) + _ranker_rotation_bonus(
        final_score=float(ranker_info.get("final_score", 0.0)),
        top_gainer_prob=float(ranker_info.get("top_gainer_prob", 0.0)),
    )


def _replacement_extra_delta(pos: "OpenPosition", current_price: Optional[float]) -> Optional[float]:
    bars_elapsed = int(getattr(pos, "bars_elapsed", 0))
    min_bars = int(getattr(config, "PORTFOLIO_REPLACE_MIN_BARS", 2))
    if bars_elapsed < min_bars:
        return None

    extra_delta = 0.0
    grace_bars = int(getattr(config, "PORTFOLIO_REPLACE_TREND_GRACE_BARS", 5))
    if bars_elapsed < grace_bars:
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA", 12.0))

    mode = getattr(pos, "signal_mode", "trend")
    if mode in ("strong_trend", "retest", "breakout", "impulse_speed"):
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA", 10.0))

    entry_adx = float(getattr(pos, "entry_adx", 0.0))
    if entry_adx >= float(getattr(config, "PORTFOLIO_REPLACE_ADX_PROTECT_MIN", 30.0)):
        extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_STRONG_EXTRA_DELTA", 10.0))

    if current_price is not None:
        pnl_pct = pos.pnl_pct(float(current_price))
        if pnl_pct >= float(getattr(config, "PORTFOLIO_REPLACE_HARD_PROFIT_PROTECT_PCT", 0.80)):
            return None
        if pnl_pct >= float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_PROTECT_PCT", 0.35)):
            extra_delta += float(getattr(config, "PORTFOLIO_REPLACE_PROFIT_EXTRA_DELTA", 12.0))

    return extra_delta


def _find_replaceable_position(
    state: "MonitorState",
    candidate_score: float,
    candidate_mode: str,
    *,
    candidate_ranker_info: Optional[Dict[str, float]] = None,
    restrict_tf: Optional[str] = None,
    restrict_mode: Optional[str] = None,
    min_delta_override: Optional[float] = None,
) -> Optional["OpenPosition"]:
    if not getattr(config, "PORTFOLIO_REPLACE_ENABLED", True):
        return None
    if not state.positions:
        return None

    min_delta = float(
        min_delta_override
        if min_delta_override is not None
        else getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0)
    )
    last_prices = state.__dict__.get("last_prices", {})
    use_ranker_rotation = bool(
        getattr(config, "PORTFOLIO_REPLACE_RANKER_ENABLED", True)
        and isinstance(candidate_ranker_info, dict)
    )
    candidate_rotation_score = float(candidate_score)
    if use_ranker_rotation:
        candidate_ranker_final = float(candidate_ranker_info.get("final_score", 0.0))
        candidate_top_gainer_prob = float(candidate_ranker_info.get("top_gainer_prob", 0.0))
        if (
            candidate_ranker_final <= float(getattr(config, "PORTFOLIO_REPLACE_CANDIDATE_MIN_FINAL", -0.50))
            and candidate_top_gainer_prob <= float(getattr(config, "PORTFOLIO_REPLACE_CANDIDATE_MIN_TOP_GAINER", 0.10))
        ):
            return None
        candidate_rotation_score = _candidate_rotation_score(candidate_score, candidate_ranker_info)

    replaceable: List[tuple[float, OpenPosition]] = []
    for pos in state.positions.values():
        if restrict_tf is not None and getattr(pos, "tf", "") != restrict_tf:
            continue
        if restrict_mode is not None and getattr(pos, "signal_mode", "") != restrict_mode:
            continue
        extra_delta = _replacement_extra_delta(pos, last_prices.get(pos.symbol))
        if extra_delta is None:
            continue
        if _signal_priority(candidate_mode) < _signal_priority(getattr(pos, "signal_mode", "trend")):
            continue
        weakest_score = _position_signal_score(pos)
        weakest_rotation_score = weakest_score
        if use_ranker_rotation:
            position_ranker_final = float(getattr(pos, "ranker_final_score", 0.0))
            position_top_gainer_prob = float(getattr(pos, "ranker_top_gainer_prob", 0.0))
            if (
                position_ranker_final > float(getattr(config, "PORTFOLIO_REPLACE_POSITION_FINAL_MAX", 0.0))
                and position_top_gainer_prob > float(getattr(config, "PORTFOLIO_REPLACE_POSITION_TOP_GAINER_MAX", 0.20))
            ):
                continue
            weakest_rotation_score = _position_rotation_score(pos)
        score_to_compare = candidate_rotation_score if use_ranker_rotation else float(candidate_score)
        baseline_score = weakest_rotation_score if use_ranker_rotation else weakest_score
        if score_to_compare < baseline_score + min_delta + extra_delta:
            continue
        replaceable.append((baseline_score, pos))

    if not replaceable:
        return None
    replaceable.sort(key=lambda item: item[0])
    return replaceable[0][1]


def _replacement_min_delta_for_candidate(mode: str) -> float:
    if _is_fresh_priority_mode(mode):
        return float(getattr(config, "PORTFOLIO_REPLACE_FRESH_MIN_DELTA", 4.0))
    return float(getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8.0))


def _count_open_positions(
    state: "MonitorState",
    *,
    tf: Optional[str] = None,
    mode: Optional[str] = None,
) -> int:
    count = 0
    for pos in state.positions.values():
        if tf is not None and getattr(pos, "tf", "") != tf:
            continue
        if mode is not None and getattr(pos, "signal_mode", "") != mode:
            continue
        count += 1
    return count


def _check_strategy_position_caps(
    state: "MonitorState",
    *,
    tf: str,
    mode: str,
) -> tuple[bool, str]:
    if tf == "1h" and mode == "impulse_speed":
        cap = int(getattr(config, "MAX_CONCURRENT_1H_IMPULSE_SPEED_POSITIONS", 0))
        if cap > 0:
            n_open = _count_open_positions(state, tf="1h", mode="impulse_speed")
            if n_open >= cap:
                return False, f"лимит 1h impulse_speed: {n_open}/{cap} позиций"
    return True, ""


def _mtf_relaxed_1h_candidate_ok(
    *,
    mode: Optional[str],
    candidate_score: Optional[float],
    slope: Optional[float],
    adx: Optional[float],
    rsi: Optional[float],
    vol_x: Optional[float],
    daily_range: Optional[float],
) -> bool:
    if not getattr(config, "MTF_1H_CONTINUATION_RELAX_ENABLED", False):
        return False
    if mode not in tuple(
        getattr(
            config,
            "MTF_1H_CONTINUATION_RELAX_MODES",
            ("alignment", "trend", "strong_trend", "impulse_speed", "impulse"),
        )
    ):
        return False
    if candidate_score is None or candidate_score < float(
        getattr(config, "MTF_1H_CONTINUATION_RELAX_SCORE_MIN", 68.0)
    ):
        return False
    if slope is None or slope < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_SLOPE_MIN", 0.18)):
        return False
    if adx is None or adx < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_ADX_MIN", 22.0)):
        return False
    if rsi is None:
        return False
    rsi_min = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_RSI_MIN", 56.0))
    rsi_max = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_RSI_MAX", 78.0))
    if rsi < rsi_min or rsi > rsi_max:
        return False
    if vol_x is None or vol_x < float(getattr(config, "MTF_1H_CONTINUATION_RELAX_VOL_X_MIN", 0.80)):
        return False
    if daily_range is None or daily_range > float(
        getattr(config, "MTF_1H_CONTINUATION_RELAX_RANGE_MAX", 8.0)
    ):
        return False
    return True


async def _check_mtf(
    session: aiohttp.ClientSession,
    sym: str,
    *,
    mode: Optional[str] = None,
    candidate_score: Optional[float] = None,
    slope: Optional[float] = None,
    adx: Optional[float] = None,
    rsi: Optional[float] = None,
    vol_x: Optional[float] = None,
    daily_range: Optional[float] = None,
) -> tuple[bool, str]:
    """
    MTF (Multi-TimeFrame) фильтр: перед входом по 1h проверяет 15м индикаторы.
    Защищает от запоздавших входов когда 1h ещё «красивый», но 15м уже в коррекции.

    Возвращает (True, "OK") если 15м подтверждает вход,
    или (False, причина) если 15м показывает коррекцию.

    Примеры ложных 1h входов без MTF:
      ETH 13.03 — пик 14:15, 1h вход 16:00, 15м MACD=-6.78, RSI=41.
    """
    if not getattr(config, "MTF_ENABLED", True):
        return True, "MTF disabled"

    # Грузим 15м данные (достаточно 100 баров для EMA50/MACD)
    data_15m = await fetch_klines(session, sym, "15m", limit=100)
    if data_15m is None or len(data_15m) < 40:
        # Нет данных → не блокируем (fail-open: лучше ложный вход, чем пропуск)
        return True, "no 15m data"

    c = data_15m["c"].astype(float)
    feat_15m = await asyncio.to_thread(
        compute_features,
        data_15m["o"],
        data_15m["h"],
        data_15m["l"],
        c,
        data_15m["v"],
    )
    i = len(c) - 2  # последний закрытый 15м бар

    macd_hist = float(feat_15m["macd_hist"][i]) if np.isfinite(feat_15m["macd_hist"][i]) else 0.0
    macd_prev = float(feat_15m["macd_hist"][i - 1]) if i >= 1 and np.isfinite(feat_15m["macd_hist"][i - 1]) else macd_hist
    rsi_val   = float(feat_15m["rsi"][i])       if np.isfinite(feat_15m["rsi"][i])       else 50.0
    close_val = float(c[i]) if np.isfinite(c[i]) else 0.0
    ema_fast_15m = feat_15m.get("ema_fast") if isinstance(feat_15m, dict) else None
    if ema_fast_15m is not None and len(ema_fast_15m) > i and np.isfinite(ema_fast_15m[i]):
        ema20_15m = float(ema_fast_15m[i])
    else:
        ema20_15m = close_val
    macd_floor = close_val * float(getattr(config, "MTF_MACD_SOFT_FLOOR_REL", -0.00005))
    macd_hard_floor = close_val * float(getattr(config, "MTF_MACD_HARD_FLOOR_REL", -0.00120))
    macd_rising = macd_hist >= macd_prev
    soft_rsi_min = float(getattr(config, "MTF_RSI_SOFT_MIN", max(getattr(config, "MTF_RSI_MIN", 45.0), 48.0)))
    relaxed_candidate = _mtf_relaxed_1h_candidate_ok(
        mode=mode,
        candidate_score=candidate_score,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    relaxed_floor = close_val * float(getattr(config, "MTF_1H_CONTINUATION_RELAX_MACD_FLOOR_REL", -0.00075))
    relaxed_rsi_min = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_15M_RSI_MIN", 46.0))
    relaxed_ema_slip_pct = float(getattr(config, "MTF_1H_CONTINUATION_RELAX_15M_EMA20_SLIP_PCT", 0.30))
    relaxed_require_rising = bool(getattr(config, "MTF_1H_CONTINUATION_RELAX_REQUIRE_MACD_RISING", True))
    hard_rsi_min = float(getattr(config, "MTF_RSI_HARD_MIN", 38.0))
    deep_negative = (
        macd_hist < 0
        and rsi_val < hard_rsi_min
        and close_val < ema20_15m
    )

    # Проверка 1: MACD hist > 0 на 15м
    retest_reason = _retest_1h_mtf_confirm_reason(
        mode=mode,
        close_val=close_val,
        ema20_15m=ema20_15m,
        macd_hist=macd_hist,
        macd_prev=macd_prev,
        rsi_val=rsi_val,
    )
    if retest_reason is not None:
        return False, retest_reason

    if getattr(config, "MTF_MACD_POSITIVE", True) and macd_hist <= 0:
        if (
            relaxed_candidate
            and macd_hist > relaxed_floor
            and (not relaxed_require_rising or macd_rising)
            and rsi_val >= relaxed_rsi_min
            and close_val >= ema20_15m * (1.0 - relaxed_ema_slip_pct / 100.0)
        ):
            return True, (
                f"15m relax-pass: MACD={macd_hist:.4g} vs prev {macd_prev:.4g}, "
                f"RSI={rsi_val:.1f}"
            )
        if deep_negative:
            return False, (
                f"15m deep correction: MACD={macd_hist:.4g} <= hard floor {macd_hard_floor:.4g}, "
                f"RSI={rsi_val:.1f} < {hard_rsi_min:.1f}, close<EMA20"
            )
        if macd_hist <= macd_floor:
            return True, (
                f"15m soft-pass: MACD={macd_hist:.4g} <= soft floor {macd_floor:.4g}, "
                f"RSI={rsi_val:.1f}"
            )
        if False and macd_hist <= macd_floor:
            return False, f"15м MACD hist={macd_hist:.4g} <= floor {macd_floor:.4g} (коррекция)"
        hard_rsi_min = float(getattr(config, "MTF_RSI_HARD_MIN", 38.0))
        if rsi_val < hard_rsi_min:
            return True, (
                f"15m soft-pass: MACD={macd_hist:.4g} and RSI={rsi_val:.1f} "
                f"below hard floor {hard_rsi_min:.1f}"
            )
        if False and rsi_val < hard_rsi_min:
            return False, f"15m MACD={macd_hist:.4g} and RSI={rsi_val:.1f} < hard floor {hard_rsi_min:.1f}"
        if False and getattr(config, "MTF_REQUIRE_MACD_RISING", True) and not macd_rising:
            return True, f"15m soft-pass: MACD={macd_hist:.4g} still below 0 and weaker than prev {macd_prev:.4g}"
        if rsi_val < soft_rsi_min:
            return True, f"15m soft-pass: RSI={rsi_val:.1f} below soft floor {soft_rsi_min:.1f}"
        if False and getattr(config, "MTF_REQUIRE_MACD_RISING", True) and not macd_rising:
            return False, f"15м MACD hist={macd_hist:.4g} < 0 и падает vs prev {macd_prev:.4g}"
        if False and rsi_val < soft_rsi_min:
            return False, f"15м MACD near 0, но RSI={rsi_val:.1f} < {soft_rsi_min:.1f}"
        return True, (
            f"15м soft-pass: MACD={macd_hist:.4g} vs prev {macd_prev:.4g}, "
            f"RSI={rsi_val:.1f}"
        )

    # Проверка 2: RSI >= MTF_RSI_MIN на 15м
    mtf_rsi_min = getattr(config, "MTF_RSI_MIN", 45.0)
    if rsi_val < mtf_rsi_min:
        hard_rsi_min = float(getattr(config, "MTF_RSI_HARD_MIN", 38.0))
        if deep_negative:
            return False, (
                f"15m deep correction: RSI={rsi_val:.1f} < {hard_rsi_min:.1f}, "
                f"MACD={macd_hist:.4g}, close<EMA20"
            )
        if True or rsi_val >= hard_rsi_min:
            return True, f"15m soft-pass: RSI={rsi_val:.1f} below MTF floor {mtf_rsi_min}"
        return False, f"15м RSI={rsi_val:.1f} < {mtf_rsi_min} (коррекция)"

    return True, f"15м OK: MACD={macd_hist:.4g} RSI={rsi_val:.1f}"


# ── Position persistence ───────────────────────────────────────────────────────
import os as _os_pers
import json as _json_pers


def _parse_event_ts_ms(raw: object) -> Optional[int]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return None


def _load_ranker_shadow_snapshot_for_position(pos: "OpenPosition") -> Optional[dict]:
    if not _BOT_EVENTS_FILE.exists():
        return None
    try:
        tail = deque(maxlen=8000)
        with _BOT_EVENTS_FILE.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    tail.append(line)
        best: Optional[dict] = None
        best_delta: Optional[int] = None
        expected_candidate_score = float(getattr(pos, "candidate_score_at_entry", 0.0))
        expected_price = float(getattr(pos, "entry_price", 0.0))
        max_delta_ms = max(5 * 60_000, 2 * _tf_bar_ms(str(getattr(pos, "tf", "15m"))))
        for line in reversed(tail):
            try:
                rec = _json_pers.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("event") != "ranker_shadow" or rec.get("bot_action") != "take":
                continue
            if str(rec.get("sym") or "") != str(getattr(pos, "symbol", "")):
                continue
            if str(rec.get("tf") or "") != str(getattr(pos, "tf", "")):
                continue
            if str(rec.get("mode") or "") != str(getattr(pos, "signal_mode", "")):
                continue
            event_ts_ms = _parse_event_ts_ms(rec.get("ts"))
            if event_ts_ms is None:
                continue
            delta = abs(event_ts_ms - int(getattr(pos, "entry_ts", 0)))
            if delta > max_delta_ms:
                continue
            try:
                event_price = float(rec.get("price"))
            except Exception:
                event_price = 0.0
            if expected_price > 0 and event_price > 0 and abs(event_price - expected_price) / expected_price > 0.05:
                continue
            if expected_candidate_score:
                try:
                    event_score = float(rec.get("candidate_score"))
                except Exception:
                    event_score = 0.0
                if abs(event_score - expected_candidate_score) > 1.0:
                    continue
            if best is None or best_delta is None or delta < best_delta:
                best = rec
                best_delta = delta
                if delta == 0:
                    break
        return best
    except Exception:
        return None


def _maybe_backfill_position_metadata(pos: "OpenPosition") -> bool:
    changed = False
    record_id = str(getattr(pos, "critic_record_id", "") or "")
    if record_id:
        rec = critic_dataset.get_record(record_id)
        if isinstance(rec, dict):
            decision = rec.get("decision") if isinstance(rec.get("decision"), dict) else {}
            if abs(float(getattr(pos, "candidate_score_at_entry", 0.0))) < 1e-12:
                value = decision.get("candidate_score")
                if value is not None:
                    pos.candidate_score_at_entry = float(value)
                    changed = True
            if abs(float(getattr(pos, "score_floor_at_entry", 0.0))) < 1e-12:
                value = decision.get("score_floor")
                if value is not None:
                    pos.score_floor_at_entry = float(value)
                    changed = True
            if abs(float(getattr(pos, "entry_ml_proba", 0.0))) < 1e-12:
                value = decision.get("ml_proba")
                if value is not None:
                    pos.entry_ml_proba = float(value)
                    changed = True
            if abs(float(getattr(pos, "forecast_return_pct", 0.0))) < 1e-12:
                value = decision.get("forecast_return_pct")
                if value is not None:
                    pos.forecast_return_pct = float(value)
                    changed = True
            if abs(float(getattr(pos, "today_change_pct", 0.0))) < 1e-12:
                value = decision.get("today_change_pct")
                if value is not None:
                    pos.today_change_pct = float(value)
                    changed = True

    if (
        abs(float(getattr(pos, "ranker_quality_proba", 0.0))) < 1e-12
        and abs(float(getattr(pos, "ranker_final_score", 0.0))) < 1e-12
        and abs(float(getattr(pos, "ranker_ev", 0.0))) < 1e-12
        and abs(float(getattr(pos, "ranker_expected_return", 0.0))) < 1e-12
        and abs(float(getattr(pos, "ranker_expected_drawdown", 0.0))) < 1e-12
    ):
        snap = _load_ranker_shadow_snapshot_for_position(pos)
        if isinstance(snap, dict):
            for attr, key in (
                ("ranker_quality_proba", "ranker_proba"),
                ("ranker_final_score", "ranker_final_score"),
                ("ranker_ev", "ranker_ev"),
                ("ranker_expected_return", "ranker_expected_return"),
                ("ranker_expected_drawdown", "ranker_expected_drawdown"),
            ):
                value = snap.get(key)
                if value is not None:
                    setattr(pos, attr, float(value))
                    changed = True
    return changed

def _pos_to_dict(pos: "OpenPosition") -> dict:
    return {
        "symbol": pos.symbol, "tf": pos.tf,
        "entry_price": pos.entry_price, "entry_bar": pos.entry_bar,
        "entry_ts": pos.entry_ts, "entry_ema20": pos.entry_ema20,
        "entry_slope": pos.entry_slope, "entry_adx": pos.entry_adx,
        "entry_rsi": pos.entry_rsi, "entry_vol_x": pos.entry_vol_x,
        "forecast_return_pct": getattr(pos, "forecast_return_pct", 0.0),
        "today_change_pct": getattr(pos, "today_change_pct", 0.0),
        "candidate_score_at_entry": getattr(pos, "candidate_score_at_entry", 0.0),
        "score_floor_at_entry": getattr(pos, "score_floor_at_entry", 0.0),
        "entry_ml_proba": getattr(pos, "entry_ml_proba", 0.0),
        "ranker_quality_proba": getattr(pos, "ranker_quality_proba", 0.0),
        "ranker_final_score": getattr(pos, "ranker_final_score", 0.0),
        "ranker_ev": getattr(pos, "ranker_ev", 0.0),
        "ranker_expected_return": getattr(pos, "ranker_expected_return", 0.0),
        "ranker_expected_drawdown": getattr(pos, "ranker_expected_drawdown", 0.0),
        "ranker_top_gainer_prob": getattr(pos, "ranker_top_gainer_prob", 0.0),
        "ranker_capture_ratio_pred": getattr(pos, "ranker_capture_ratio_pred", 0.0),
        "prediction_horizons": list(
            getattr(pos, "prediction_horizons", ()) or _position_forward_horizons(pos.tf, pos.signal_mode)
        ),
        "predictions": {str(k): v for k, v in pos.predictions.items()},
        "bars_elapsed": pos.bars_elapsed, "ml_record_id": pos.ml_record_id,
        "critic_record_id": pos.critic_record_id,
        "signal_mode": pos.signal_mode, "trail_k": pos.trail_k,
        "max_hold_bars": pos.max_hold_bars, "trail_stop": pos.trail_stop,
        "macd_warn_streak": pos.macd_warn_streak, "macd_warned": pos.macd_warned,
        "last_macd_bar_i": pos.last_macd_bar_i,
        "bandit_arm": getattr(pos, "bandit_arm", None),
        "buy_notified": getattr(pos, "buy_notified", True),
    }

def _pos_from_dict(d: dict) -> "OpenPosition":
    tf = d["tf"]
    signal_mode = d.get("signal_mode", "trend")
    prediction_horizons = _normalize_forward_horizons(
        d.get("prediction_horizons", _position_forward_horizons(tf, signal_mode))
    )
    return OpenPosition(
        symbol=d["symbol"], tf=tf, entry_price=d["entry_price"],
        entry_bar=d["entry_bar"], entry_ts=d["entry_ts"],
        entry_ema20=d.get("entry_ema20", 0.0), entry_slope=d.get("entry_slope", 0.0),
        entry_adx=d.get("entry_adx", 0.0), entry_rsi=d.get("entry_rsi", 50.0),
        entry_vol_x=d.get("entry_vol_x", 1.0),
        forecast_return_pct=d.get("forecast_return_pct", 0.0),
        today_change_pct=d.get("today_change_pct", 0.0),
        candidate_score_at_entry=d.get("candidate_score_at_entry", 0.0),
        score_floor_at_entry=d.get("score_floor_at_entry", 0.0),
        entry_ml_proba=d.get("entry_ml_proba", 0.0),
        ranker_quality_proba=d.get("ranker_quality_proba", 0.0),
        ranker_final_score=d.get("ranker_final_score", 0.0),
        ranker_ev=d.get("ranker_ev", 0.0),
        ranker_expected_return=d.get("ranker_expected_return", 0.0),
        ranker_expected_drawdown=d.get("ranker_expected_drawdown", 0.0),
        ranker_top_gainer_prob=d.get("ranker_top_gainer_prob", 0.0),
        ranker_capture_ratio_pred=d.get("ranker_capture_ratio_pred", 0.0),
        prediction_horizons=prediction_horizons,
        predictions={int(k): v for k, v in d.get("predictions", {}).items()},
        bars_elapsed=d.get("bars_elapsed", 0), ml_record_id=d.get("ml_record_id", ""),
        critic_record_id=d.get("critic_record_id", ""),
        signal_mode=signal_mode, trail_k=d.get("trail_k", 2.0),
        max_hold_bars=d.get("max_hold_bars", 16), trail_stop=d.get("trail_stop", 0.0),
        macd_warn_streak=d.get("macd_warn_streak", 0),
        macd_warned=d.get("macd_warned", False),
        last_macd_bar_i=d.get("last_macd_bar_i", -1),
        bandit_arm=d.get("bandit_arm"),
        buy_notified=d.get("buy_notified", True),  # True = старые позиции не получат повтор
    )

def save_positions(positions: dict) -> None:
    """Атомарно сохраняет позиции на диск."""
    path = getattr(config, "POSITIONS_FILE", "positions.json")
    try:
        data = {sym: _pos_to_dict(pos) for sym, pos in positions.items()}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json_pers.dump(data, f, ensure_ascii=False, indent=2)
        _os_pers.replace(tmp, path)
    except Exception as e:
        log.warning("save_positions failed: %s", e)

def load_positions() -> dict:
    """Восстанавливает позиции при старте бота."""
    path = getattr(config, "POSITIONS_FILE", "positions.json")
    if not _os_pers.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json_pers.load(f)
        positions = {sym: _pos_from_dict(d) for sym, d in data.items()}
        backfilled = False
        for pos in positions.values():
            if _maybe_backfill_position_metadata(pos):
                backfilled = True
        if backfilled:
            save_positions(positions)
        log.info("Restored %d position(s) from %s: %s",
                 len(positions), path, list(positions.keys()))
        return positions
    except Exception as e:
        log.warning("load_positions failed: %s — starting empty", e)
        return {}


def _fill_trade_outcome_labels(
    pos: "OpenPosition",
    *,
    exit_pnl: float,
    exit_reason: str,
    bars_held: int,
) -> None:
    if pos.ml_record_id:
        ml_dataset.fill_labels(
            pos.ml_record_id,
            exit_pnl=exit_pnl,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )
    if getattr(pos, "critic_record_id", ""):
        critic_dataset.fill_trade_outcome(
            pos.critic_record_id,
            exit_pnl=exit_pnl,
            exit_reason=exit_reason,
            bars_held=bars_held,
        )


def _fill_forward_labels(pos: "OpenPosition", horizon: int, ret_pct: float) -> None:
    if pos.ml_record_id:
        ml_dataset.fill_forward_label(pos.ml_record_id, horizon, ret_pct)
    if getattr(pos, "critic_record_id", ""):
        critic_dataset.fill_forward_label(pos.critic_record_id, horizon, ret_pct)


def _check_portfolio_limits(
    sym: str,
    state: "MonitorState",
    *,
    max_positions_override: Optional[int] = None,
    reserve_fresh_slots: int = 0,
) -> tuple[bool, str]:
    """
    Проверяет портфельные лимиты перед входом в позицию.

    Возвращает (ok: bool, reason: str).

    Лимиты:
      1. MAX_OPEN_POSITIONS — не более N одновременных позиций
      2. MAX_POSITIONS_PER_GROUP — не более M позиций в одной группе монет
         (защита от ситуации 11.03.2026 когда 12 L1/AI вошли одновременно)
    """
    max_pos_cfg = int(getattr(config, "MAX_OPEN_POSITIONS", 6))
    max_pos  = int(max_positions_override if max_positions_override is not None else max_pos_cfg)
    max_grp  = getattr(config, "MAX_POSITIONS_PER_GROUP", 2)

    # Лимит 1: общее число позиций
    n_open = len(state.positions)
    if n_open >= max_pos:
        return False, f"портфель полон: {n_open}/{max_pos} позиций"

    # Лимит 2: группа монеты
    my_group = _get_coin_group(sym)
    if my_group:
        group_count = sum(
            1 for s in state.positions
            if _get_coin_group(s) == my_group
        )
        if group_count >= max_grp:
            return False, (
                f"группа '{my_group}' уже {group_count}/{max_grp} позиций "
                f"({', '.join(s for s in state.positions if _get_coin_group(s) == my_group)})"
            )

    return True, ""


def _clone_signal_guard_reason(
    sym: str,
    state: "MonitorState",
    *,
    tf: str,
    mode: str,
    bar_ts: int,
    candidate_score: float = 0.0,
    ranker_info: Optional[Dict[str, float]] = None,
) -> str:
    if not bool(getattr(config, "CLONE_SIGNAL_GUARD_ENABLED", False)):
        return ""
    guard_tfs = tuple(getattr(config, "CLONE_SIGNAL_GUARD_TF", ("15m",)))
    guard_modes = tuple(
        getattr(
            config,
            "CLONE_SIGNAL_GUARD_MODES",
            ("impulse_speed", "breakout", "retest", "alignment", "trend"),
        )
    )
    if tf not in guard_tfs or mode not in guard_modes:
        return ""

    override_score = float(getattr(config, "CLONE_SIGNAL_GUARD_OVERRIDE_SCORE", 90.0))
    override_rank = float(getattr(config, "CLONE_SIGNAL_GUARD_OVERRIDE_RANKER_FINAL", 0.50))
    score_override_min_rank = float(
        getattr(config, "CLONE_SIGNAL_GUARD_SCORE_OVERRIDE_MIN_RANKER_FINAL", -0.25)
    )
    ranker_final = 0.0 if not ranker_info else float(ranker_info.get("final_score", 0.0))
    if ranker_final >= override_rank:
        return ""
    if candidate_score >= override_score and ranker_final >= score_override_min_rank:
        return ""

    window_bars = max(1, int(getattr(config, "CLONE_SIGNAL_GUARD_WINDOW_BARS", 8)))
    max_similar = max(1, int(getattr(config, "CLONE_SIGNAL_GUARD_MAX_SIMILAR", 2)))
    max_same_group = max(0, int(getattr(config, "CLONE_SIGNAL_GUARD_MAX_SAME_GROUP", 1)))
    window_ms = window_bars * _tf_bar_ms(tf)
    my_group = _get_coin_group(sym)

    recent_similar: list[str] = []
    same_group_recent: list[str] = []
    for open_sym, pos in state.positions.items():
        if open_sym == sym:
            continue
        if str(getattr(pos, "tf", "")) != tf:
            continue
        pos_mode = str(getattr(pos, "signal_mode", ""))
        if pos_mode not in guard_modes:
            continue
        pos_ts = int(getattr(pos, "entry_ts", 0) or 0)
        if bar_ts > 0 and pos_ts > 0:
            if max(0, bar_ts - pos_ts) > window_ms:
                continue
        elif int(getattr(pos, "bars_elapsed", window_bars + 1)) > window_bars:
            continue

        recent_similar.append(open_sym)
        if my_group and _get_coin_group(open_sym) == my_group:
            same_group_recent.append(open_sym)

    if my_group and max_same_group > 0 and len(same_group_recent) >= max_same_group:
        names = ", ".join(same_group_recent[:3])
        return (
            f"clone signal guard: recent {tf} setups in group '{my_group}' already "
            f"{len(same_group_recent)}/{max_same_group} ({names})"
        )
    if len(recent_similar) >= max_similar:
        names = ", ".join(recent_similar[:4])
        return (
            f"clone signal guard: recent similar {tf} setups already "
            f"{len(recent_similar)}/{max_similar} ({names})"
        )
    return ""


SendFn = Callable[[str], Awaitable[None]]


# ── Position tracking ──────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    symbol:      str
    tf:          str
    entry_price: float
    entry_bar:   int          # index within the candle array at entry time
    entry_ts:    int          # unix ms timestamp of entry bar

    # Snapshot of indicators at entry (for context in messages)
    entry_ema20: float
    entry_slope: float
    entry_adx:   float
    entry_rsi:   float
    entry_vol_x: float
    forecast_return_pct: float = 0.0
    today_change_pct: float = 0.0
    candidate_score_at_entry: float = 0.0
    score_floor_at_entry: float = 0.0
    entry_ml_proba: float = 0.0
    ranker_quality_proba: float = 0.0
    ranker_final_score: float = 0.0
    ranker_ev: float = 0.0
    ranker_expected_return: float = 0.0
    ranker_expected_drawdown: float = 0.0
    ranker_top_gainer_prob: float = 0.0
    ranker_capture_ratio_pred: float = 0.0

    # Forward prediction tracking: {horizon_bars: True/False/None}
    prediction_horizons: tuple[int, ...] = field(default_factory=tuple)
    predictions: Dict[int, Optional[bool]] = field(default_factory=dict)
    bars_elapsed: int = 0

    # ML: id записи в ml_dataset.jsonl для обновления меток
    ml_record_id:  str   = ""
    critic_record_id: str = ""

    # П1: режим сигнала определяет trail_k и max_hold_bars
    signal_mode:   str   = "trend"    # "trend"/"strong_trend"/"retest"/"breakout"
    trail_k:       float = 2.0        # множитель ATR (берётся из config по режиму)
    max_hold_bars: int   = 16         # лимит баров в позиции (по режиму)

    # ATR trailing stop — обновляется каждый бар вверх, никогда вниз
    # Выход срабатывает когда цена падает ниже trail_stop
    trail_stop: float = 0.0

    # Contextual bandit arm chosen at entry
    bandit_arm: Optional[int] = None
    bandit_info: Optional[dict] = None

    # MACD Exit Warning — счётчик баров подряд с падающей гистограммой
    macd_warn_streak: int = 0
    macd_warned:      bool = False  # не спамим одним предупреждением
    # ФИКС: streak обновляется только один раз на каждый новый закрытый бар,
    # а не при каждом поллинге (60с) — иначе через 3 минуты ложные алерты на 30 монетах.
    last_macd_bar_i: int = -1

    # ФИКС: флаг что BUY-уведомление было успешно отправлено.
    # Если False — значит await send() при входе бросил исключение и позиция
    # осталась «немой». При следующем поллинге перед SELL будет ретрансляция.
    buy_notified: bool = False

    def pnl_pct(self, current_price: float) -> float:
        return (current_price / self.entry_price - 1.0) * 100.0

    def prediction_summary(self) -> str:
        parts = []
        horizons = self.prediction_horizons or _position_forward_horizons(self.tf, self.signal_mode)
        for h in horizons:
            result = self.predictions.get(h)
            if result is None:
                parts.append(f"T+{h}: ⏳")
            elif result:
                parts.append(f"T+{h}: ✅")
            else:
                parts.append(f"T+{h}: ❌")
        return "  ".join(parts)


# ── Shared monitor state ───────────────────────────────────────────────────────

@dataclass
class MonitorState:
    hot_coins:  List[CoinReport] = field(default_factory=list)
    positions:  Dict[str, OpenPosition] = field(default_factory=dict)
    running:    bool = False
    task:       Optional[asyncio.Task] = None
    # П3: cooldown после выхода — {symbol: unix_ms_until_which_skip}
    # Хранит unix-timestamp в мс до которого повторный вход заблокирован.
    # Всегда устанавливается как: data["t"][i] + COOLDOWN_BARS * bar_ms
    cooldowns:     Dict[str, int]  = field(default_factory=dict)
    # Флаг «cooldown уже залогирован» — отдельный dict чтобы не мешать типам
    cd_logged:     Dict[str, bool] = field(default_factory=dict)
    # Деdup для portfolio BLOCK логов: {symbol: последний_ts_ms когда логировали}.
    # Портфельный лимит проверяется каждые 60с → без dedup = 100+ строк на монету.
    # Логируем не чаще 1 раза в BLOCK_LOG_INTERVAL_BARS баров (по умолчанию 4 = 1ч на 15m).
    block_logged:  Dict[str, int]  = field(default_factory=dict)
    time_block_recent: Dict[str, dict] = field(default_factory=dict)
    time_block_streaks: Dict[str, dict] = field(default_factory=dict)
    last_discovery_ts: int = 0
    recent_discoveries: Dict[str, dict] = field(default_factory=dict)
    discovery_task: Optional[asyncio.Task] = None


def _tf_bar_ms(tf: str) -> int:
    return 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000


def _normalize_forward_horizons(values) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for raw in values or ():
        try:
            horizon = int(raw)
        except Exception:
            continue
        if horizon <= 0 or horizon in seen:
            continue
        seen.add(horizon)
        result.append(horizon)
    if result:
        return tuple(result)
    return tuple(int(h) for h in getattr(config, "FORWARD_BARS", [3, 5, 10]))


def _position_forward_horizons(tf: str, mode: str) -> tuple[int, ...]:
    if tf == "15m" and mode in tuple(
        getattr(config, "FORWARD_BARS_15M_FAST_MODES", ("breakout", "retest", "impulse_speed"))
    ):
        return _normalize_forward_horizons(getattr(config, "FORWARD_BARS_15M_FAST", [2, 5, 7]))
    return _normalize_forward_horizons(getattr(config, "FORWARD_BARS", [3, 5, 10]))


def _signal_cluster_bucket(tf: str, mode: str) -> str:
    if tf == "15m" and mode in tuple(
        getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES", ("breakout", "retest"))
    ):
        return "15m_short_bounce"
    if tf == "15m" and mode in tuple(
        getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES", ("impulse_speed",))
    ):
        return "15m_impulse"
    if tf == "1h" and mode in tuple(
        getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES", ("retest",))
    ):
        return "1h_retest"
    return ""


def _open_signal_cluster_cap_reason(
    sym: str,
    state: "MonitorState",
    *,
    tf: str,
    mode: str,
) -> str:
    if not bool(getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_ENABLED", False)):
        return ""
    bucket = _signal_cluster_bucket(tf, mode)
    if not bucket:
        return ""

    if bucket == "15m_short_bounce":
        max_open = max(1, int(getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX", 2)))
    elif bucket == "15m_impulse":
        max_open = max(1, int(getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX", 2)))
    elif bucket == "1h_retest":
        max_open = max(1, int(getattr(config, "OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX", 1)))
    else:
        return ""

    open_cluster: list[str] = []
    for open_sym, pos in state.positions.items():
        if open_sym == sym:
            continue
        if _signal_cluster_bucket(str(getattr(pos, "tf", "")), str(getattr(pos, "signal_mode", ""))) != bucket:
            continue
        open_cluster.append(open_sym)

    if len(open_cluster) >= max_open:
        names = ", ".join(open_cluster[:4])
        return f"open cluster cap: {bucket} already {len(open_cluster)}/{max_open} ({names})"
    return ""


def _signal_snapshot(feat: dict, c: np.ndarray, idx: int, tf: str = "") -> Optional[dict]:
    if idx < 0 or idx >= len(c):
        return None
    entry_ok, _ = check_entry_conditions(feat, idx, c, tf=tf)
    brk_ok, _ = check_breakout_conditions(feat, idx)
    ret_ok, _ = check_retest_conditions(feat, idx)
    surge_ok, _ = check_trend_surge_conditions(feat, idx)
    imp_ok, _ = check_impulse_conditions(feat, idx)
    aln_ok, _ = check_alignment_conditions(feat, idx, tf=tf)
    if not any((entry_ok, brk_ok, ret_ok, surge_ok, imp_ok, aln_ok)):
        return None
    if brk_ok:
        mode = "breakout"
    elif ret_ok:
        mode = "retest"
    elif entry_ok:
        mode, _ = get_effective_entry_mode(feat, idx, c, tf=tf)
    elif surge_ok:
        mode = "impulse_speed"
    elif imp_ok:
        mode = "impulse"
    else:
        mode = "alignment"
    return {
        "idx": idx,
        "mode": mode,
        "price": float(c[idx]),
    }


def _coin_report_priority(report: CoinReport) -> tuple:
    return (
        bool(report.signal_now),
        _signal_priority(getattr(report, "signal_mode", "")),
        float(getattr(report, "forecast_return_pct", 0.0)),
        float(getattr(report, "today_change_pct", 0.0)),
        bool(report.today_confirmed),
        int(report.today_signals),
        float(report.best_accuracy),
    )


async def _discover_new_hot_coins(
    session: aiohttp.ClientSession,
    state: "MonitorState",
    send: SendFn,
) -> int:
    """
    Постоянный discovery-слой поверх основного мониторинга.
    Нужен чтобы новые сигналы не ждали редкого auto_reanalyze и не терялись между пересчётами.
    """
    watchlist = config.load_watchlist()
    if not watchlist:
        return 0
    existing_by_sym = {r.symbol: (idx, r) for idx, r in enumerate(state.hot_coins)}

    tasks = [
        (sym, tf, asyncio.create_task(fetch_klines(session, sym, tf, limit=config.LIVE_LIMIT)))
        for sym in watchlist
        for tf in config.TIMEFRAMES
    ]

    best: Dict[str, tuple[CoinReport, int]] = {}
    for idx, (sym, tf, task) in enumerate(tasks, start=1):
        data = await task
        if data is None:
            continue
        report = await _analyze_coin_live(sym, tf, data)
        if not report.signal_now and len(data) >= 2:
            try:
                _, feat = await _compute_features_from_data(data)
                i_now = len(data) - 2
                surge_ok, _ = check_trend_surge_conditions(feat, i_now)
                if surge_ok:
                    report = replace(
                        report,
                        signal_now=True,
                        signal_mode="impulse_speed",
                        no_signal_reason="",
                    )
            except Exception:
                pass
        if not report.signal_now:
            continue
        report_bar_ts = int(data["t"][-2]) if len(data["t"]) >= 2 else 0
        prev = best.get(sym)
        if prev is None or _coin_report_priority(report) > _coin_report_priority(prev[0]):
            best[sym] = (report, report_bar_ts)
        if idx % 12 == 0:
            await asyncio.sleep(0)

    added = 0
    for idx, (report, report_bar_ts) in enumerate(
        sorted(best.values(), key=lambda item: _coin_report_priority(item[0]), reverse=True),
        start=1,
    ):
        discovery_action = "add"
        existing = existing_by_sym.get(report.symbol)
        if existing is not None:
            idx, current = existing
            if _coin_report_priority(report) <= _coin_report_priority(current):
                return
            state.hot_coins[idx] = report
            discovery_action = "upgrade"
        else:
            if report.symbol in state.positions:
                return
            state.hot_coins.append(report)
        state.recent_discoveries[report.symbol] = {
            "tf": report.tf,
            "mode": report.signal_mode,
            "price": float(report.current_price),
            "bar_ts": report_bar_ts,
        }
        added += 1
        log.info(
            "DISCOVERY %s %s [%s] mode=%s price=%.6g",
            discovery_action.upper(),
            report.symbol,
            report.tf,
            report.signal_mode,
            report.current_price,
        )
        if getattr(config, "SEND_DISCOVERY_NOTIFICATIONS", False):
            try:
                await send(
                    f"🛰 *Новый live-сигнал добавлен в мониторинг*\n\n"
                    f"*{report.symbol}*  `[{report.tf}]`\n"
                    f"режим: `{report.signal_mode or 'buy'}`\n"
                    f"цена: `{report.current_price:.6g}`\n"
                    f"RSI: `{report.current_rsi:.1f}`  ADX: `{report.current_adx:.1f}`  vol×: `{report.current_vol_x:.2f}`"
                )
            except Exception as e:
                log.warning("discovery send failed for %s: %s", report.symbol, e)
        if idx % 12 == 0:
            await asyncio.sleep(0)

    return added


# ── Per-coin polling ───────────────────────────────────────────────────────────

async def _poll_coin(
    session: aiohttp.ClientSession,
    report:  CoinReport,
    state:   MonitorState,
    send:    SendFn,
) -> None:
    sym = report.symbol
    tf  = report.tf
    pos = state.positions.get(sym)
    if pos is not None:
        # Open positions must be monitored on their original timeframe.
        # Otherwise a 1h position can later be processed as 15m after discovery changes.
        tf = getattr(pos, "tf", tf)

    data = await fetch_klines(session, sym, tf, limit=config.LIVE_LIMIT)
    if data is None or len(data) < 60:
        return

    c, feat = await _compute_features_from_data(data)
    i    = len(c) - 2  # last *closed* bar (index -1 is the forming bar)

    if i < 10:
        return

    state.__dict__.setdefault("last_prices", {})[sym] = float(c[i])
    if pos is not None:
        try:
            live_report = await _analyze_coin_live(sym, tf, data)
        except Exception:
            live_report = None
        if live_report is not None:
            new_forecast_return = float(
                getattr(live_report, "forecast_return_pct", getattr(pos, "forecast_return_pct", 0.0))
            )
            new_today_change = float(
                getattr(live_report, "today_change_pct", getattr(pos, "today_change_pct", 0.0))
            )
            metrics_changed = (
                abs(new_forecast_return - float(getattr(pos, "forecast_return_pct", 0.0))) > 1e-9
                or abs(new_today_change - float(getattr(pos, "today_change_pct", 0.0))) > 1e-9
            )
            pos.forecast_return_pct = new_forecast_return
            pos.today_change_pct = new_today_change
            if metrics_changed:
                save_positions(state.positions)



    # ── No open position: look for entry ────────────────────────────────────
    if pos is None:
        # П3: cooldown — не входим повторно N баров после выхода
        # cooldown хранит unix-ms время до которого вход заблокирован
        cooldown_until_ms = state.cooldowns.get(sym, 0)
        current_ts_ms = int(data["t"][i])
        if current_ts_ms < cooldown_until_ms:
            bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
            bars_left = max(0, (cooldown_until_ms - current_ts_ms) // bar_ms)
            bars_total = int(getattr(config, "COOLDOWN_BARS", 8))
            # Логируем только начало cooldown (когда bars_left максимальное)
            if not state.cd_logged.get(sym):
                botlog.log_cooldown(sym=sym, tf=tf, bars_remaining=int(bars_left), first=True)
                state.cd_logged[sym] = True
                # ФИКС: логируем в critic_dataset чтобы Trend Scout видел cooldown-блоки.
                # Без этого в critic нет записей о монетах в cooldown → скаут не диагностирует.
                _log_critic_candidate(
                    sym=sym, tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type="cooldown",
                    feat=feat, data=data, i=i,
                    action="blocked",
                    reason_code="cooldown",
                    reason=f"cooldown: {bars_left}/{bars_total} bars remaining",
                    stage="cooldown",
                )
            return  # ещё в cooldown, пропускаем
        else:
            # Сбрасываем флаг «cooldown залогирован» когда cooldown истёк
            state.cd_logged.pop(sym, None)

        # ── Часовой фильтр (ML: 44K баров, 15.03.2026) ──────────────────
        # Блокируем входы в статистически убыточные часы UTC.
        # EV в часы [3,10-15] = -0.083%, в остальные = +0.200%.
        _block_hours = getattr(config, "ENTRY_BLOCK_HOURS", [])
        _bar_hour_utc = int(data["t"][i] // 3_600_000) % 24
        hour_blocked = bool(_block_hours and _bar_hour_utc in _block_hours)

        # Приоритет при выборе лучшего live-сигнала:
        # BREAKOUT > RETEST > strong_trend > trend > IMPULSE > ALIGNMENT
        entry_ok, _entry_reason = check_entry_conditions(feat, i, c, tf=tf)
        brk_ok,   _             = check_breakout_conditions(feat, i)
        ret_ok,   _             = check_retest_conditions(feat, i)
        surge_ok, _             = check_trend_surge_conditions(feat, i)
        imp_ok,   _             = check_impulse_conditions(feat, i)
        aln_ok,   _             = check_alignment_conditions(feat, i, tf=tf)

        # EMA_CROSS — самый ранний сигнал: пробой EMA20 снизу вверх.
        # Срабатывает на 3-5 баров раньше BUY (до роста ADX, до slope-порога).
        # Имеет собственный cooldown чтобы не спамить повторами одного пробоя.
        cross_ok = False
        if getattr(config, "EMA_CROSS_ENABLED", True):
            _cross_cds = state.__dict__.setdefault("cross_cooldowns", {})
            _bar_ms_cross = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
            _cross_until = _cross_cds.get(sym, 0)
            if int(data["t"][i]) >= _cross_until:
                cross_ok, _ = check_ema_cross_conditions(feat, i)
                if cross_ok:
                    _cd_cross = int(getattr(config, "CROSS_COOLDOWN_BARS", 6))
                    _cross_cds[sym] = int(data["t"][i]) + _cd_cross * _bar_ms_cross

        # Yield to event loop so Telegram UI requests aren't starved by concurrent
        # _poll_coin tasks whose synchronous sections would otherwise block the loop.
        await asyncio.sleep(0)

        any_signal = entry_ok or brk_ok or ret_ok or surge_ok or imp_ok or aln_ok or cross_ok
        catchup_snapshot = None
        recent_disc = state.recent_discoveries.get(sym)
        if not any_signal and recent_disc and recent_disc.get("tf") == tf:
            grace_bars = int(getattr(config, "DISCOVERY_ENTRY_GRACE_BARS", 2))
            max_slip = float(getattr(config, "DISCOVERY_ENTRY_MAX_SLIPPAGE_PCT", 0.45))
            bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
            recent_bar_ts = int(recent_disc.get("bar_ts", 0))
            current_bar_ts = int(data["t"][i]) if len(data["t"]) > i else 0
            if recent_bar_ts and current_bar_ts - recent_bar_ts > grace_bars * bar_ms:
                state.recent_discoveries.pop(sym, None)
                recent_disc = None
        if not any_signal and recent_disc and recent_disc.get("tf") == tf:
            grace_bars = int(getattr(config, "DISCOVERY_ENTRY_GRACE_BARS", 2))
            max_slip = float(getattr(config, "DISCOVERY_ENTRY_MAX_SLIPPAGE_PCT", 0.45))
            for lookback in range(1, grace_bars + 1):
                j = i - lookback
                snap = _signal_snapshot(feat, c, j, tf=tf)
                if snap is None:
                    continue
                signal_price = float(snap["price"])
                current_price = float(c[i])
                if signal_price <= 0:
                    continue
                slippage_pct = (current_price / signal_price - 1.0) * 100.0
                if slippage_pct > max_slip:
                    break
                catchup_snapshot = snap | {"slippage_pct": slippage_pct, "lookback": lookback}
                any_signal = True
                break
        if not any_signal:
            preview_price = float(c[i])
            preview_ema20 = float(feat["ema_fast"][i])
            preview_slope = float(feat["slope"][i])
            preview_adx = float(feat["adx"][i])
            preview_rsi = float(feat["rsi"][i])
            preview_vol = float(feat["vol_x"][i])
            preview_range = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
            near_miss = _near_miss_candidate_snapshot(
                tf=tf,
                feat=feat,
                data=data,
                i=i,
                price=preview_price,
                ema20=preview_ema20,
                slope=preview_slope,
                adx=preview_adx,
                rsi=preview_rsi,
                vol_x=preview_vol,
                daily_range=preview_range,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
            )
            if near_miss is not None:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=str(near_miss["mode"]),
                    feat=feat,
                    data=data,
                    i=i,
                    action="candidate",
                    reason_code="near_miss",
                    reason=str(near_miss["reason"]),
                    stage="near_miss",
                    candidate_score=float(near_miss["candidate_score"]),
                    base_score=float(near_miss["base_score"]),
                    score_floor=float(near_miss["score_floor"]),
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=None,
                    mtf_soft_penalty=0.0,
                    fresh_priority=False,
                    catchup=False,
                    continuation_profile=bool(near_miss.get("continuation_profile", False)),
                    signal_flags={},
                    near_miss=True,
                )
                log.info("NEAR MISS %s [%s]: %s", sym, tf, str(near_miss["reason"]))
            return
        if any_signal:
            early_15m_continuation = False
            if catchup_snapshot is not None:
                preview_mode = str(catchup_snapshot["mode"])
            elif brk_ok:
                preview_mode = "breakout"
            elif ret_ok:
                preview_mode = "retest"
            elif entry_ok:
                preview_mode, early_15m_continuation = get_effective_entry_mode(feat, i, c, tf=tf)
            elif surge_ok:
                preview_mode = "impulse_speed"
            elif imp_ok:
                preview_mode = "impulse"
            else:
                preview_mode = "alignment"

            preview_price = float(c[i])
            preview_ema20 = float(feat["ema_fast"][i])
            preview_slope = float(feat["slope"][i])
            preview_adx = float(feat["adx"][i])
            preview_rsi = float(feat["rsi"][i])
            preview_vol = float(feat["vol_x"][i])
            preview_range = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
            signal_flags = _candidate_signal_flags(
                entry_ok=entry_ok,
                brk_ok=brk_ok,
                ret_ok=ret_ok,
                surge_ok=surge_ok,
                imp_ok=imp_ok,
                aln_ok=aln_ok,
                cross_ok=cross_ok,
            )
            base_score = _entry_signal_score(
                mode=preview_mode,
                price=preview_price,
                ema20=preview_ema20,
                slope=preview_slope,
                adx=preview_adx,
                rsi=preview_rsi,
                vol_x=preview_vol,
                daily_range=preview_range,
            )
            candidate_score = base_score
            mtf_soft_penalty = 0.0
            score_floor = _entry_score_floor(tf) if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False) else 0.0
            candidate_score += _top_mover_score_bonus(float(getattr(report, "today_change_pct", 0.0)))
            candidate_score += _forecast_return_score_bonus(float(getattr(report, "forecast_return_pct", 0.0)))
            continuation_profile = _time_block_1h_continuation_profile(
                tf=tf,
                mode=preview_mode,
                slope=preview_slope,
                adx=preview_adx,
                rsi=preview_rsi,
                vol_x=preview_vol,
                daily_range=preview_range,
            )
            if early_15m_continuation:
                continuation_profile = True
                candidate_score += float(getattr(config, "EARLY_15M_CONTINUATION_ENTRY_SCORE_BONUS", 10.0))
            candidate_score += _time_block_1h_continuation_bonus(
                tf=tf,
                mode=preview_mode,
                slope=preview_slope,
                adx=preview_adx,
                rsi=preview_rsi,
                vol_x=preview_vol,
                daily_range=preview_range,
            )
            if _is_fresh_priority_candidate(preview_mode, catchup_snapshot):
                candidate_score += float(getattr(config, "FRESH_SIGNAL_SCORE_BONUS", 7.0))
            if catchup_snapshot is not None:
                candidate_score += float(getattr(config, "DISCOVERY_CATCHUP_SCORE_BONUS", 6.0))

            # ── Correlation Guard: мягкий downgrade для коррелированных кандидатов ──
            # Применяется до проверки score_floor, чтобы дубли не прошли по высокому баллу.
            if bool(getattr(config, "CORR_GUARD_ENABLED", False)) and bool(getattr(config, "CORR_MARGINAL_WEIGHTING", True)):
                _corr_cache_mg = corr_guard.get_or_create_cache(state.__dict__)
                _marginal = corr_guard.marginal_score(sym, state.positions, candidate_score, _corr_cache_mg)
                if _marginal < candidate_score:
                    log.debug("corr_guard marginal: %s score %.1f -> %.1f", sym, candidate_score, _marginal)
                    candidate_score = _marginal

            if _late_1h_continuation_guard(
                tf=tf,
                mode=preview_mode,
                continuation_profile=continuation_profile,
                candidate_score=candidate_score,
                price=preview_price,
                ema20=preview_ema20,
                rsi=preview_rsi,
                daily_range=preview_range,
            ):
                reason = (
                    f"late 1h continuation: RSI {preview_rsi:.1f}, "
                    f"price {_time_block_price_edge_pct(price=preview_price, ema20=preview_ema20):.2f}% > EMA20"
                )
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="late_continuation",
                    reason=reason,
                    stage="pre_entry_filter",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=None,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("LATE CONTINUATION BLOCK %s [%s]: %s", sym, tf, reason)
                botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="late_continuation")
                return
            is_bull_day_now = bool(getattr(config, "_bull_day_active", False))
            impulse_speed_reason = _impulse_speed_entry_guard(
                tf=tf,
                mode=preview_mode,
                feat=feat,
                i=i,
                price=preview_price,
                ema20=preview_ema20,
                rsi=preview_rsi,
                adx=preview_adx,
                daily_range=preview_range,
                is_bull_day=is_bull_day_now,
            )
            if impulse_speed_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="impulse_guard",
                    reason=impulse_speed_reason,
                    stage="pre_entry_filter",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=None,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("IMPULSE GUARD BLOCK %s [%s]: %s", sym, tf, impulse_speed_reason)
                botlog.log_blocked(sym, tf, float(c[i]), impulse_speed_reason, signal_type="impulse_guard")
                return

            if hour_blocked:
                repeat_count = _bump_time_block_streak(
                    state,
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    bar_ts=int(data["t"][i]) if len(data["t"]) > i else 0,
                )
                if _time_block_bypass_allowed(
                    tf=tf,
                        mode=preview_mode,
                        candidate_score=candidate_score,
                        vol_x=preview_vol,
                        catchup_snapshot=catchup_snapshot,
                        continuation_profile=continuation_profile,
                    ):
                    log.info(
                        "TIME BLOCK BYPASS %s [%s]: mode=%s score=%.2f vol_x=%.2f",
                        sym, tf, preview_mode, candidate_score, preview_vol,
                    )
                    _clear_time_block_streak(state, sym=sym, tf=tf)
                elif _time_block_1h_prebypass_allowed(
                    tf=tf,
                    mode=preview_mode,
                    candidate_score=candidate_score,
                    vol_x=preview_vol,
                    price=preview_price,
                    ema20=preview_ema20,
                    continuation_profile=continuation_profile,
                    repeat_count=repeat_count,
                ):
                    log.info(
                        "TIME BLOCK PRE-BYPASS %s [%s]: mode=%s score=%.2f repeats=%s price_edge=%.2f%%",
                        sym,
                        tf,
                        preview_mode,
                        candidate_score,
                        repeat_count,
                        _time_block_price_edge_pct(price=preview_price, ema20=preview_ema20),
                    )
                    _clear_time_block_streak(state, sym=sym, tf=tf)
                else:
                    _remember_time_block(
                        state,
                        sym=sym,
                        tf=tf,
                        mode=preview_mode,
                        bar_ts=int(data["t"][i]) if len(data["t"]) > i else 0,
                    )
                    time_block_logged = state.__dict__.setdefault("time_block_logged", {})
                    _block_interval = getattr(config, "BLOCK_LOG_INTERVAL_BARS", 4)
                    bar_ms_tb = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
                    _until_tb = time_block_logged.get(sym, 0)
                    _now_tb = int(data["t"][i]) if len(data["t"]) > i else 0
                    if _now_tb >= _until_tb:
                        reason = f"time block: UTC hour {_bar_hour_utc} filtered"
                        _log_critic_candidate(
                            sym=sym,
                            tf=tf,
                            bar_ts=int(data["t"][i]),
                            signal_type=preview_mode,
                            feat=feat,
                            data=data,
                            i=i,
                            action="blocked",
                            reason_code="time_block",
                            reason=reason,
                            stage="time_filter",
                            candidate_score=candidate_score,
                            base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=None,
                            mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("TIME BLOCK %s [%s]: %s", sym, tf, reason)
                        botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="time_block")
                        time_block_logged[sym] = _now_tb + _block_interval * bar_ms_tb
                    return
            else:
                _clear_time_block_streak(state, sym=sym, tf=tf)
            candidate_score += _time_block_retest_bonus(
                state,
                sym=sym,
                tf=tf,
                mode=preview_mode,
                current_bar_ts=int(data["t"][i]) if len(data["t"]) > i else 0,
                is_bull_day=bool(getattr(config, "_bull_day_active", False)),
            )
            # ── MTF-фильтр: для 1h сигналов проверяем 15м индикаторы ─────────
            # Защита от запоздавших 1h входов (пик на 15м уже давно прошёл).
            if tf == "1h" and getattr(config, "MTF_ENABLED", True):
                mtf_ok, mtf_reason = await _check_mtf(
                    session,
                    sym,
                    mode=preview_mode,
                    candidate_score=candidate_score,
                    slope=preview_slope,
                    adx=preview_adx,
                    rsi=preview_rsi,
                    vol_x=preview_vol,
                    daily_range=preview_range,
                )
                if not mtf_ok:
                    _log_critic_candidate(
                        sym=sym,
                        tf=tf,
                        bar_ts=int(data["t"][i]),
                        signal_type=preview_mode,
                        feat=feat,
                        data=data,
                        i=i,
                        action="blocked",
                        reason_code="mtf",
                        reason=mtf_reason,
                        stage="mtf_filter",
                        candidate_score=candidate_score,
                        base_score=base_score,
                        score_floor=score_floor,
                        forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                        today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                        ml_proba=None,
                        mtf_soft_penalty=mtf_soft_penalty,
                        fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                        catchup=catchup_snapshot is not None,
                        continuation_profile=continuation_profile,
                        signal_flags=signal_flags,
                    )
                    log.info("MTF BLOCK %s [1h->15m]: %s", sym, mtf_reason)
                    botlog.log_blocked(sym, tf, 0.0, f"MTF: {mtf_reason}")
                    return
                mtf_soft_penalty = _mtf_soft_penalty_from_reason(mtf_reason)
                candidate_score -= mtf_soft_penalty

            # ── Портфельный лимит ─────────────────────────────────────────────
            is_bull_day_now = bool(getattr(config, "_bull_day_active", False))
            # Offload to thread: CatBoost / numpy inference must not block event loop.
            ml_proba = await asyncio.to_thread(
                _ml_general_score,
                sym,
                tf,
                preview_mode,
                feat,
                data,
                i,
                is_bull_day=is_bull_day_now,
            )
            if ml_proba is not None:
                candidate_score += (
                    ml_proba - float(getattr(config, "ML_GENERAL_NEUTRAL_PROBA", 0.50))
                ) * float(getattr(config, "ML_GENERAL_SCORE_WEIGHT", 10.0))
                if getattr(config, "ML_GENERAL_HARD_BLOCK_ENABLED", False):
                    if is_bull_day_now:
                        _min = float(getattr(config, "ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN", 0.28))
                    else:
                        _min = float(getattr(config, "ML_GENERAL_HARD_BLOCK_MIN", 0.35))
                    _max = float(getattr(config, "ML_GENERAL_HARD_BLOCK_MAX", 1.01))
                    if ml_proba < _min or ml_proba > _max:
                        reason = f"ML proba {ml_proba:.3f} outside profitable zone [{_min:.2f},{_max:.2f}]"
                        _log_critic_candidate(
                            sym=sym, tf=tf, bar_ts=int(data["t"][i]),
                            signal_type=preview_mode, feat=feat, data=data, i=i,
                            action="blocked", reason_code="ml_proba_zone",
                            reason=reason, stage="ml_proba_zone",
                            candidate_score=candidate_score, base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=ml_proba, mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("ML ZONE BLOCK %s [%s]: %s", sym, tf, reason)
                        botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="ml_proba_zone")
                        return
            # CatBoost inference (possibly JIT-compiling on first call — can block for
            # minutes). MUST run in a thread to keep the event loop responsive.
            ranker_info = await asyncio.to_thread(
                _ml_candidate_ranker_components,
                sym=sym,
                tf=tf,
                signal_type=preview_mode,
                feat=feat,
                data=data,
                i=i,
                is_bull_day=is_bull_day_now,
                candidate_score=candidate_score,
                base_score=base_score,
                score_floor=score_floor,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                ml_proba=ml_proba,
                mtf_soft_penalty=mtf_soft_penalty,
                fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                catchup=catchup_snapshot is not None,
                continuation_profile=continuation_profile,
                signal_flags=signal_flags,
            )
            ranker_proba = None if ranker_info is None else float(ranker_info.get("quality_proba", 0.0))
            candidate_score += _ml_candidate_ranker_runtime_bonus(ranker_info)
            if preview_mode == "trend" and not is_bull_day_now:
                if ml_proba is None:
                    ml_proba = await asyncio.to_thread(
                        _ml_trend_nonbull_score, sym, tf, feat, data, i
                    )
                if ml_proba is not None:
                    min_proba = float(getattr(config, "ML_TREND_NONBULL_MIN_PROBA", 0.35))
                    if ml_proba < min_proba:
                        candidate_score -= float(
                            getattr(config, "ML_TREND_NONBULL_LOW_PROBA_PENALTY", 6.0)
                        )
                        if getattr(config, "ML_TREND_NONBULL_HARD_BLOCK", False):
                            reason = f"ML trend|nonbull quality {ml_proba:.2f} < {min_proba:.2f}"
                            _log_critic_candidate(
                                sym=sym,
                                tf=tf,
                                bar_ts=int(data["t"][i]),
                                signal_type=preview_mode,
                                feat=feat,
                                data=data,
                                i=i,
                                action="blocked",
                                reason_code="ml_filter",
                                reason=reason,
                                stage="quality_floor",
                                candidate_score=candidate_score,
                                base_score=base_score,
                                score_floor=score_floor,
                                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                                today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                                ml_proba=ml_proba,
                                mtf_soft_penalty=mtf_soft_penalty,
                                fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                                catchup=catchup_snapshot is not None,
                                continuation_profile=continuation_profile,
                                signal_flags=signal_flags,
                            )
                            log.info("ML BLOCK %s [%s]: %s", sym, tf, reason)
                            _maybe_log_ranker_shadow(
                                sym=sym,
                                tf=tf,
                                mode=preview_mode,
                                price=float(c[i]),
                                candidate_score=candidate_score,
                                score_floor=score_floor,
                                ranker_proba=ranker_proba,
                                ranker_info=ranker_info,
                                bot_action="blocked",
                                reason=reason,
                            )
                            botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="ml_filter")
                            return
            # ── Phase 2: Enhanced signals bonus/penalty ────────────────
            _enh = _get_enhanced()
            _enh_adj = None
            if _enh is not None:
                try:
                    _enh_adj = _enh.score_adjustment(
                        sym=sym,
                        tf=tf,
                        mode=preview_mode,
                        current_score=candidate_score,
                        feat=feat,
                        data=data,
                        i=i,
                        is_bull_day=is_bull_day_now,
                        report=report,
                    )
                    candidate_score += _enh_adj.total_bonus
                    if _enh_adj.is_blocked:
                        reason = f"enhanced: {_enh_adj.block_reason}"
                        _log_critic_candidate(
                            sym=sym, tf=tf, bar_ts=int(data["t"][i]),
                            signal_type=preview_mode, feat=feat, data=data, i=i,
                            action="blocked", reason_code="enhanced_block",
                            reason=reason, stage="enhanced_filter",
                            candidate_score=candidate_score, base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=ml_proba, mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("ENHANCED BLOCK %s [%s]: %s (adj=%s)", sym, tf, reason, _enh_adj.to_log_dict())
                        botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="enhanced_block")
                        return
                    if abs(_enh_adj.total_bonus) > 0.01:
                        log.info(
                            "ENHANCED ADJ %s [%s]: %+.1f (of=%+.1f deriv=%+.1f regime=%+.1f tg=%+.1f mh=%+.1f)",
                            sym, tf, _enh_adj.total_bonus,
                            _enh_adj.order_flow_bonus, _enh_adj.derivatives_bonus,
                            _enh_adj.regime_bonus, _enh_adj.top_gainer_bonus,
                            _enh_adj.multi_horizon_bonus,
                        )
                except Exception:
                    log.debug("Enhanced signals scoring error for %s", sym, exc_info=True)

            if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False):
                min_score = score_floor
                if candidate_score < min_score:
                    if _entry_score_borderline_bypass_ok(
                        tf=tf,
                        mode=preview_mode,
                        candidate_score=candidate_score,
                        min_score=min_score,
                        price=preview_price,
                        ema20=preview_ema20,
                        slope=preview_slope,
                        adx=preview_adx,
                        rsi=preview_rsi,
                        vol_x=preview_vol,
                        daily_range=preview_range,
                    ):
                        log.info(
                            "ENTRY SCORE BYPASS %s [%s]: borderline quality candidate %.2f < %.2f",
                            sym,
                            tf,
                            candidate_score,
                            min_score,
                        )
                    elif _entry_score_continuation_bypass_ok(
                        tf=tf,
                        mode=preview_mode,
                        candidate_score=candidate_score,
                        price=preview_price,
                        ema20=preview_ema20,
                        slope=preview_slope,
                        adx=preview_adx,
                        rsi=preview_rsi,
                        vol_x=preview_vol,
                        daily_range=preview_range,
                        continuation_profile=continuation_profile,
                        is_bull_day=is_bull_day_now,
                    ):
                        log.info(
                            "ENTRY SCORE CONTINUATION BYPASS %s [%s]: continuation candidate %.2f < %.2f",
                            sym,
                            tf,
                            candidate_score,
                            min_score,
                        )
                    else:
                        reason = f"entry score {candidate_score:.2f} < floor {min_score:.2f}"
                        _log_critic_candidate(
                            sym=sym,
                            tf=tf,
                            bar_ts=int(data["t"][i]),
                            signal_type=preview_mode,
                            feat=feat,
                            data=data,
                            i=i,
                            action="blocked",
                            reason_code="entry_score",
                            reason=reason,
                            stage="quality_floor",
                            candidate_score=candidate_score,
                            base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=ml_proba,
                            mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("ENTRY SCORE BLOCK %s [%s]: %s", sym, tf, reason)
                        _maybe_log_ranker_shadow(
                            sym=sym,
                            tf=tf,
                            mode=preview_mode,
                            price=float(c[i]),
                            candidate_score=candidate_score,
                            score_floor=score_floor,
                            ranker_proba=ranker_proba,
                            ranker_info=ranker_info,
                            bot_action="blocked",
                            reason=reason,
                        )
                        botlog.log_blocked(sym, tf, float(c[i]), reason, signal_type="entry_score")
                        return

            trend_guard_reason = _trend_entry_quality_guard_reason(
                tf=tf,
                mode=preview_mode,
                price=preview_price,
                ema20=preview_ema20,
                slope=preview_slope,
                adx=preview_adx,
                rsi=preview_rsi,
                vol_x=preview_vol,
                daily_range=preview_range,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                is_bull_day=is_bull_day_now,
            )
            if trend_guard_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="trend_quality",
                    reason=trend_guard_reason,
                    stage="quality_floor",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("TREND QUALITY BLOCK %s [%s]: %s", sym, tf, trend_guard_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=trend_guard_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), trend_guard_reason, signal_type="trend_quality")
                return

            mode_range_guard_reason = _mode_daily_range_guard_reason(
                mode=preview_mode,
                tf=tf,
                daily_range=preview_range,
                slope=preview_slope,
            )
            if mode_range_guard_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="mode_range_quality",
                    reason=mode_range_guard_reason,
                    stage="quality_floor",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("MODE RANGE QUALITY BLOCK %s [%s]: %s", sym, tf, mode_range_guard_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=mode_range_guard_reason,
                )
                botlog.log_blocked(
                    sym, tf, float(c[i]), mode_range_guard_reason,
                    signal_type="mode_range_quality",
                )
                return

            ranker_veto_reason = _ranker_entry_veto_reason(
                tf=tf,
                mode=preview_mode,
                ranker_proba=ranker_proba,
                candidate_score=candidate_score,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
            )
            if ranker_veto_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="ranker_veto",
                    reason=ranker_veto_reason,
                    stage="quality_floor",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("RANKER VETO BLOCK %s [%s]: %s", sym, tf, ranker_veto_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=ranker_veto_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), ranker_veto_reason, signal_type="ranker_veto")
                return
            ranker_hard_veto_reason = _ranker_hard_veto_reason(
                tf=tf,
                mode=preview_mode,
                ranker_info=ranker_info,
            )
            if ranker_hard_veto_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="ranker_hard_veto",
                    reason=ranker_hard_veto_reason,
                    stage="quality_floor",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=_is_fresh_priority_candidate(preview_mode, catchup_snapshot),
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("RANKER HARD VETO BLOCK %s [%s]: %s", sym, tf, ranker_hard_veto_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=ranker_hard_veto_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), ranker_hard_veto_reason, signal_type="ranker_hard_veto")
                return

            fresh_priority = _is_fresh_priority_candidate(preview_mode, catchup_snapshot)
            max_pos_cfg = int(getattr(config, "MAX_OPEN_POSITIONS", 6))
            reserve_slots = int(getattr(config, "FRESH_SIGNAL_RESERVED_SLOTS", 0))
            effective_max_pos = max_pos_cfg
            if reserve_slots > 0 and not fresh_priority:
                effective_max_pos = max(1, max_pos_cfg - reserve_slots)
            cap_ok, cap_reason = _check_strategy_position_caps(
                state,
                tf=tf,
                mode=preview_mode,
            )
            if not cap_ok:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="strategy_cap",
                    reason=cap_reason,
                    stage="portfolio",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=fresh_priority,
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("STRATEGY CAP BLOCK %s [%s]: %s", sym, tf, cap_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=cap_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), cap_reason, signal_type="strategy_cap")
                return
            clone_guard_reason = _clone_signal_guard_reason(
                sym,
                state,
                tf=tf,
                mode=preview_mode,
                bar_ts=int(data["t"][i]),
                candidate_score=candidate_score,
                ranker_info=ranker_info,
            )
            if clone_guard_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="clone_signal_guard",
                    reason=clone_guard_reason,
                    stage="portfolio",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=fresh_priority,
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("CLONE SIGNAL BLOCK %s [%s]: %s", sym, tf, clone_guard_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=clone_guard_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), clone_guard_reason, signal_type="clone_guard")
                return
            cluster_cap_reason = _open_signal_cluster_cap_reason(
                sym,
                state,
                tf=tf,
                mode=preview_mode,
            )
            if cluster_cap_reason:
                _log_critic_candidate(
                    sym=sym,
                    tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=preview_mode,
                    feat=feat,
                    data=data,
                    i=i,
                    action="blocked",
                    reason_code="open_cluster_cap",
                    reason=cluster_cap_reason,
                    stage="portfolio",
                    candidate_score=candidate_score,
                    base_score=base_score,
                    score_floor=score_floor,
                    forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                    today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                    ml_proba=ml_proba,
                    mtf_soft_penalty=mtf_soft_penalty,
                    fresh_priority=fresh_priority,
                    catchup=catchup_snapshot is not None,
                    continuation_profile=continuation_profile,
                    signal_flags=signal_flags,
                )
                log.info("OPEN CLUSTER CAP BLOCK %s [%s]: %s", sym, tf, cluster_cap_reason)
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=preview_mode,
                    price=float(c[i]),
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="blocked",
                    reason=cluster_cap_reason,
                )
                botlog.log_blocked(sym, tf, float(c[i]), cluster_cap_reason, signal_type="open_cluster_cap")
                return

            # ── Correlation Guard — блокировка клонов по пирсоновской rho ──────
            if bool(getattr(config, "CORR_GUARD_ENABLED", False)):
                _corr_cache = corr_guard.get_or_create_cache(state.__dict__)
                _is_bull, _is_bear = corr_guard.get_bull_bear_flags(state.__dict__)
                _cg_result = corr_guard.check_entry(
                    sym,
                    state.positions,
                    _corr_cache,
                    is_bull_day=_is_bull,
                    is_bear_day=_is_bear,
                )
                if not _cg_result.allowed:
                    _log_critic_candidate(
                        sym=sym,
                        tf=tf,
                        bar_ts=int(data["t"][i]),
                        signal_type=preview_mode,
                        feat=feat,
                        data=data,
                        i=i,
                        action="blocked",
                        reason_code="correlation_guard",
                        reason=_cg_result.reason,
                        stage="portfolio",
                        candidate_score=candidate_score,
                        base_score=base_score,
                        score_floor=score_floor,
                        forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                        today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                        ml_proba=ml_proba,
                        mtf_soft_penalty=mtf_soft_penalty,
                        fresh_priority=fresh_priority,
                        catchup=catchup_snapshot is not None,
                        continuation_profile=continuation_profile,
                        signal_flags=signal_flags,
                    )
                    log.info("CORR GUARD BLOCK %s [%s]: %s", sym, tf, _cg_result.reason)
                    _maybe_log_ranker_shadow(
                        sym=sym,
                        tf=tf,
                        mode=preview_mode,
                        price=float(c[i]),
                        candidate_score=candidate_score,
                        score_floor=score_floor,
                        ranker_proba=ranker_proba,
                        ranker_info=ranker_info,
                        bot_action="blocked",
                        reason=_cg_result.reason,
                    )
                    botlog.log_blocked(sym, tf, float(c[i]), _cg_result.reason, signal_type="correlation_guard")
                    return

            port_ok, port_reason = _check_portfolio_limits(
                sym,
                state,
                max_positions_override=effective_max_pos,
                reserve_fresh_slots=reserve_slots if not fresh_priority else 0,
            )
            if not port_ok:
                replace_pos = None
                max_pos = max_pos_cfg
                if len(state.positions) >= max_pos:
                    late_rotation_reason = _late_impulse_speed_rotation_reason(
                        tf=tf,
                        mode=preview_mode,
                        rsi=preview_rsi,
                        daily_range=preview_range,
                    )
                    if late_rotation_reason:
                        _log_critic_candidate(
                            sym=sym,
                            tf=tf,
                            bar_ts=int(data["t"][i]),
                            signal_type=preview_mode,
                            feat=feat,
                            data=data,
                            i=i,
                            action="blocked",
                            reason_code="late_impulse_rotation",
                            reason=late_rotation_reason,
                            stage="portfolio",
                            candidate_score=candidate_score,
                            base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=ml_proba,
                            mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=fresh_priority,
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("LATE IMPULSE ROTATION BLOCK %s [%s]: %s", sym, tf, late_rotation_reason)
                        _maybe_log_ranker_shadow(
                            sym=sym,
                            tf=tf,
                            mode=preview_mode,
                            price=float(c[i]),
                            candidate_score=candidate_score,
                            score_floor=score_floor,
                            ranker_proba=ranker_proba,
                            ranker_info=ranker_info,
                            bot_action="blocked",
                            reason=late_rotation_reason,
                        )
                        botlog.log_blocked(sym, tf, float(c[i]), late_rotation_reason, signal_type="late_impulse_rotation")
                        return
                    replace_pos = _find_replaceable_position(
                        state,
                        candidate_score,
                        preview_mode,
                        candidate_ranker_info=ranker_info,
                        min_delta_override=_replacement_min_delta_for_candidate(preview_mode),
                    )
                if replace_pos is not None:
                    last_prices = state.__dict__.setdefault("last_prices", {})
                    replace_price = last_prices.get(replace_pos.symbol)
                    if replace_price is not None:
                        candidate_ranker_final = float((ranker_info or {}).get("final_score", 0.0))
                        replaced_ranker_final = float(getattr(replace_pos, "ranker_final_score", 0.0))
                        replace_reason = (
                            f"portfolio rotation -> {sym} [{preview_mode}] "
                            f"(ranker {replaced_ranker_final:.2f}->{candidate_ranker_final:.2f})"
                        )
                        replace_pnl = replace_pos.pnl_pct(float(replace_price))
                        replace_icon = "??" if replace_pnl >= 0 else "??"
                        if _aux_notifications_enabled():
                            await send(
                                f"PORTFOLIO ROTATION\n\n"
                                f"replace {replace_pos.symbol} [{replace_pos.tf}] -> {sym} [{tf}]\n"
                                f"reason: stronger {preview_mode} candidate by rotation score\n"
                                f"exit {replace_pos.symbol}: `{float(replace_price):.6g}` ({replace_icon} `{replace_pnl:+.2f}%`)"
                            )
                        botlog.log_exit(
                            sym=replace_pos.symbol,
                            tf=replace_pos.tf,
                            mode=getattr(replace_pos, "signal_mode", "trend"),
                            entry_price=replace_pos.entry_price,
                            exit_price=float(replace_price),
                            reason=replace_reason,
                            bars_held=replace_pos.bars_elapsed,
                            trail_k=getattr(replace_pos, "trail_k", 2.0),
                        )
                        _fill_trade_outcome_labels(
                            replace_pos,
                            exit_pnl=replace_pos.pnl_pct(float(replace_price)),
                            exit_reason=replace_reason,
                            bars_held=replace_pos.bars_elapsed,
                        )
                        old_tf = replace_pos.tf
                        del state.positions[replace_pos.symbol]
                        save_positions(state.positions)
                        old_bar_ms = 15 * 60 * 1000 if old_tf == "15m" else 60 * 60 * 1000
                        state.cooldowns[replace_pos.symbol] = int(data["t"][i]) + getattr(config, "COOLDOWN_BARS", 8) * old_bar_ms
                        state.cd_logged.pop(replace_pos.symbol, None)
                        port_ok = True
                        port_reason = ""

                # ── ML-Gated Rotation: вторая попытка освободить слот ──────────
                # Если существующая score-based ротация не сработала, но у
                # кандидата высокий ml_proba — помечаем слабейшую по EV ногу
                # на выход через trail_stop. Слот освободится на следующем
                # поллинге (через стандартный ATR-exit), и аналогичный
                # кандидат сможет войти.
                if (
                    not port_ok
                    and replace_pos is None
                    and bool(getattr(config, "ROTATION_ENABLED", False))
                ):
                    _rot_last_prices = state.__dict__.setdefault("last_prices", {})
                    _rot_decision = rotation.should_rotate(
                        ml_proba,
                        state.positions,
                        config,
                        _rot_last_prices,
                    )
                    if _rot_decision.allowed:
                        _weak_pos = state.positions.get(_rot_decision.weak_sym)
                        _weak_price = _rot_last_prices.get(_rot_decision.weak_sym)
                        if _weak_pos is not None and _weak_price is not None:
                            _evicted = rotation.evict_position(_weak_pos, float(_weak_price))
                            if _evicted:
                                log.info(
                                    "ML-ROTATION eviction %s for %s [%s]: %s",
                                    _rot_decision.weak_sym, sym, tf, _rot_decision.reason,
                                )
                                if _aux_notifications_enabled():
                                    try:
                                        await send(
                                            rotation.format_rotation_message(
                                                sym=sym,
                                                mode=preview_mode,
                                                tf=tf,
                                                candidate_ml_proba=float(ml_proba or 0.0),
                                                decision=_rot_decision,
                                            )
                                        )
                                    except Exception:
                                        log.exception("ML-ROTATION send failed")
                                # Слот освободится на след. поллинге — candidate
                                # заблокируется этим разом, но войдёт при ретраях.
                                # port_ok остаётся False — это штатно.

                if not port_ok:
                    _block_interval = getattr(config, "BLOCK_LOG_INTERVAL_BARS", 4)
                    bar_ms_b = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
                    _block_until = state.block_logged.get(sym, 0)
                    _now_ms = int(data["t"][i]) if len(data["t"]) > i else 0
                    if _now_ms >= _block_until:
                        _log_critic_candidate(
                            sym=sym,
                            tf=tf,
                            bar_ts=int(data["t"][i]),
                            signal_type=preview_mode,
                            feat=feat,
                            data=data,
                            i=i,
                            action="blocked",
                            reason_code="portfolio",
                            reason=port_reason,
                            stage="portfolio",
                            candidate_score=candidate_score,
                            base_score=base_score,
                            score_floor=score_floor,
                            forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                            today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                            ml_proba=ml_proba,
                            mtf_soft_penalty=mtf_soft_penalty,
                            fresh_priority=fresh_priority,
                            catchup=catchup_snapshot is not None,
                            continuation_profile=continuation_profile,
                            signal_flags=signal_flags,
                        )
                        log.info("PORTFOLIO BLOCK %s [%s]: %s", sym, tf, port_reason)
                        _maybe_log_ranker_shadow(
                            sym=sym,
                            tf=tf,
                            mode=preview_mode,
                            price=float(c[i]) if len(c) > i else 0.0,
                            candidate_score=candidate_score,
                            score_floor=score_floor,
                            ranker_proba=ranker_proba,
                            ranker_info=ranker_info,
                            bot_action="blocked",
                            reason=port_reason,
                        )
                        botlog.log_blocked(sym, tf, 0.0, f"блок: {port_reason}")
                        state.block_logged[sym] = _now_ms + _block_interval * bar_ms_b
                    return
            if catchup_snapshot is not None:
                sig_mode = str(catchup_snapshot["mode"])
                if sig_mode == "breakout":
                    trail_k = getattr(config, "ATR_TRAIL_K_BREAKOUT", 1.5)
                    max_hold = getattr(config, "MAX_HOLD_BARS_BREAKOUT", 6)
                elif sig_mode == "retest":
                    trail_k = getattr(config, "ATR_TRAIL_K_RETEST", 1.8)
                    max_hold = getattr(config, "MAX_HOLD_BARS_RETEST", 10)
                elif sig_mode in ("strong_trend", "impulse_speed"):
                    trail_k = getattr(config, "ATR_TRAIL_K_STRONG", 2.5)
                    max_hold = (getattr(config, 'MAX_HOLD_BARS_15M', 48) if tf == '15m' else config.MAX_HOLD_BARS)
                else:
                    trail_k = config.ATR_TRAIL_K
                    max_hold = (getattr(config, 'MAX_HOLD_BARS_15M', 48) if tf == '15m' else config.MAX_HOLD_BARS)
            elif brk_ok:
                sig_mode  = "breakout"
                trail_k   = getattr(config, "ATR_TRAIL_K_BREAKOUT",  1.5)
                max_hold  = getattr(config, "MAX_HOLD_BARS_BREAKOUT", 6)
            elif ret_ok:
                sig_mode  = "retest"
                trail_k   = getattr(config, "ATR_TRAIL_K_RETEST",    1.8)
                max_hold  = getattr(config, "MAX_HOLD_BARS_RETEST",   10)
            elif entry_ok:
                mode, _  = get_effective_entry_mode(feat, i, c, tf=tf)
                sig_mode = mode  # "trend" / "strong_trend" / "impulse_speed"
                # impulse_speed: ADX ещё не вырос, но цена уже летит → широкий стоп как у strong_trend
                if mode in ("strong_trend", "impulse_speed"):
                    trail_k = getattr(config, "ATR_TRAIL_K_STRONG", 2.5)
                else:
                    trail_k = config.ATR_TRAIL_K
                max_hold = (getattr(config, 'MAX_HOLD_BARS_15M', 48)
                            if tf == '15m' else config.MAX_HOLD_BARS)
            elif surge_ok:
                sig_mode = "impulse_speed"
                trail_k  = getattr(config, "ATR_TRAIL_K_STRONG", 2.5)
                max_hold = (getattr(config, "MAX_HOLD_BARS_15M", 48)
                            if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16))
            elif imp_ok:
                sig_mode = "impulse"
                trail_k  = getattr(config, "ATR_TRAIL_K", 2.0)
                max_hold = (getattr(config, "MAX_HOLD_BARS_15M", 48)
                            if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16))
            elif aln_ok:
                sig_mode = "alignment"
                trail_k  = getattr(config, "ATR_TRAIL_K", 2.0)
                max_hold = (getattr(config, "MAX_HOLD_BARS_15M", 48)
                            if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16))
            else:  # cross_ok — самый ранний: EMA20 пробой снизу, стоп шире
                sig_mode = "impulse_cross"
                trail_k  = getattr(config, "ATR_TRAIL_K_CROSS", 2.5)
                max_hold = (getattr(config, "MAX_HOLD_BARS_15M", 48)
                            if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16))

            price = float(c[i])
            ef    = float(feat["ema_fast"][i])
            slp   = float(feat["slope"][i])
            adx   = float(feat["adx"][i])
            rsi   = float(feat["rsi"][i])
            vx    = float(feat["vol_x"][i])

            # Contextual bandit: enter/skip gate + adaptive trail_k
            _bandit_arm: Optional[int] = None
            _bandit_info: Optional[dict] = None
            _entry_state = {
                "slope_pct": slp, "adx": adx, "rsi": rsi,
                "vol_x": vx, "ml_proba": float(ml_proba or 0.5),
                "daily_range": float(getattr(report, "daily_range", 3.0)),
                "macd_hist": float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0,
            }
            _mkt_regime = str(getattr(config, "_market_regime", "neutral"))
            _btc_ema50 = float(getattr(config, "_btc_vs_ema50", 0.0))

            # 1) Enter/skip bandit gate
            if getattr(config, "BANDIT_ENABLED", True):
                try:
                    from contextual_bandit import should_enter as _bandit_should_enter
                    _enter_ok, _enter_info = _bandit_should_enter(
                        _entry_state,
                        sym=sym,
                        mode=sig_mode,
                        tf=tf,
                        is_bull_day=is_bull_day_now,
                        market_regime=_mkt_regime,
                        btc_vs_ema50=_btc_ema50,
                    )
                    if not _enter_ok:
                        log.info("BANDIT SKIP %s [%s]: ucbs=%s",
                                 sym, sig_mode, _enter_info.get("ucbs"))
                        botlog.log_blocked(sym, tf, float(c[i]),
                                           f"bandit skip: ucbs={_enter_info.get('ucbs')}",
                                           signal_type="bandit_skip")
                        return
                    log.info("BANDIT ENTER %s [%s]: ucbs=%s",
                             sym, sig_mode, _enter_info.get("ucbs"))
                except Exception as e:
                    log.warning("Bandit enter/skip error for %s: %s", sym, e)

            # 2) Trail_k / max_hold selection (trail bandit)
            if getattr(config, "BANDIT_ENABLED", True):
                try:
                    from contextual_bandit import select_entry_profile
                    trail_k, max_hold, _bandit_info = select_entry_profile(
                        _entry_state,
                        base_trail_k=trail_k,
                        base_max_hold=max_hold,
                        mode=sig_mode,
                        tf=tf,
                        is_bull_day=is_bull_day_now,
                        market_regime=_mkt_regime,
                        btc_vs_ema50=_btc_ema50,
                    )
                    _bandit_arm = _bandit_info.get("arm")
                    log.info("BANDIT %s [%s]: arm=%s trail_k=%.2f hold=%d",
                             sym, sig_mode, _bandit_info.get("arm_name"), trail_k, max_hold)
                except Exception as e:
                    log.warning("Bandit trail error for %s: %s", sym, e)

            atr_val    = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
            # Anti-whipsaw: enforce minimum trail buffer in % of price for high-vol modes.
            # See _trail_min_buffer_pct docstring for ALGOUSDT 26.04 case study.
            _init_buf  = _compute_trail_buffer(price=price, trail_k=trail_k, atr=atr_val, mode=sig_mode)
            init_trail = price - _init_buf if _init_buf > 0 else 0.0
            if atr_val > 0:
                _atr_buf_pct = (trail_k * atr_val) / price * 100 if price > 0 else 0.0
                _eff_buf_pct = _init_buf / price * 100 if price > 0 else 0.0
                if _eff_buf_pct > _atr_buf_pct + 0.01:
                    log.info("TRAIL FLOOR %s [%s]: ATR-buf=%.2f%% -> floor=%.2f%% (trail_k=%.2f)",
                             sym, sig_mode, _atr_buf_pct, _eff_buf_pct, trail_k)
            prediction_horizons = _position_forward_horizons(tf, sig_mode)

            pos = OpenPosition(
                symbol=sym, tf=tf,
                entry_price=price,
                entry_bar=i,
                entry_ts=int(data["t"][i]),
                entry_ema20=ef,
                entry_slope=slp,
                entry_adx=adx,
                entry_rsi=rsi,
                entry_vol_x=vx,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                candidate_score_at_entry=float(candidate_score),
                score_floor_at_entry=float(score_floor),
                entry_ml_proba=float(ml_proba or 0.0),
                ranker_quality_proba=float((ranker_info or {}).get("quality_proba", 0.0)),
                ranker_final_score=float((ranker_info or {}).get("final_score", 0.0)),
                ranker_ev=float((ranker_info or {}).get("ev_raw", 0.0)),
                ranker_expected_return=float((ranker_info or {}).get("expected_return", 0.0)),
                ranker_expected_drawdown=float((ranker_info or {}).get("expected_drawdown", 0.0)),
                ranker_top_gainer_prob=float((ranker_info or {}).get("top_gainer_prob", 0.0)),
                ranker_capture_ratio_pred=float((ranker_info or {}).get("capture_ratio_pred", 0.0)),
                prediction_horizons=prediction_horizons,
                predictions={h: None for h in prediction_horizons},
                trail_stop=init_trail,
                signal_mode=sig_mode,
                trail_k=trail_k,
                max_hold_bars=max_hold,
                bandit_arm=_bandit_arm,
                bandit_info=_bandit_info,
            )
            # ── КРИТИЧНО: позиция сохраняется ПЕРВОЙ, до любых внешних вызовов.
            # Если ml_dataset или botlog выбросят исключение — позиция уже в state
            # и не потеряется. Именно это было причиной "нет позиций при сигналах".
            pos.critic_record_id = _log_critic_candidate(
                sym=sym,
                tf=tf,
                bar_ts=int(data["t"][i]),
                signal_type=sig_mode,
                feat=feat,
                data=data,
                i=i,
                action="take",
                reason_code="take",
                reason="candidate accepted",
                stage="entry",
                candidate_score=candidate_score,
                base_score=base_score,
                score_floor=score_floor,
                forecast_return_pct=float(getattr(report, "forecast_return_pct", 0.0)),
                today_change_pct=float(getattr(report, "today_change_pct", 0.0)),
                ml_proba=ml_proba,
                mtf_soft_penalty=mtf_soft_penalty,
                fresh_priority=fresh_priority,
                catchup=catchup_snapshot is not None,
                continuation_profile=continuation_profile,
                signal_flags=signal_flags,
            )
            state.positions[sym] = pos
            save_positions(state.positions)  # персистируем при входе
            state.block_logged.pop(sym, None)  # сбросить dedup при входе
            state.time_block_recent.pop(sym, None)
            state.recent_discoveries.pop(sym, None)
            log.info(
                "ENTRY %s [%s] %s price=%.6g rsi=%.1f adx=%.1f vol_x=%.2f",
                sym, tf, sig_mode, price, rsi, adx, vx,
            )

            # ML Dataset (некритично — ошибка не должна блокировать вход)
            _btc_vs_ema50 = getattr(config, "_btc_vs_ema50", 0.0)
            try:
                ml_id = ml_dataset.log_signal_candidate(
                    sym=sym, tf=tf,
                    bar_ts=int(data["t"][i]),
                    signal_type=sig_mode,
                    is_bull_day=getattr(config, "_bull_day_active", False),
                    feat=feat, i=i, data=data,
                    btc_vs_ema50=_btc_vs_ema50,
                )
                pos.ml_record_id = ml_id
                if pos.critic_record_id:
                    critic_dataset.mark_trade_taken(pos.critic_record_id, linked_ml_record_id=ml_id)
            except Exception as _ml_err:
                log.warning("ml_dataset.log_signal_candidate failed for %s: %s", sym, _ml_err)

            # Операционный лог (некритично)
            try:
                _dr = float(feat["daily_range_pct"][i])
                _mh = float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0
                _maybe_log_ranker_shadow(
                    sym=sym,
                    tf=tf,
                    mode=sig_mode,
                    price=price,
                    candidate_score=candidate_score,
                    score_floor=score_floor,
                    ranker_proba=ranker_proba,
                    ranker_info=ranker_info,
                    bot_action="take",
                    reason="candidate accepted",
                )
                botlog.log_entry(
                    sym=sym, tf=tf, mode=sig_mode,
                    price=price, ema20=ef, slope=slp,
                    rsi=rsi, adx=adx, vol_x=vx,
                    macd_hist=_mh,
                    daily_range=_dr if np.isfinite(_dr) else 0.0,
                    trail_k=trail_k, max_hold_bars=max_hold,
                    ml_proba=ml_proba,
                )
            except Exception as _log_err:
                log.warning("botlog.log_entry failed for %s: %s", sym, _log_err)

            # П7: метка типа сигнала
            _mode_labels = {
                "trend":        "📈 Тренд",
                "strong_trend": "💪 Сильный тренд",
                "impulse_speed":"⚡ Быстрое движение",
                "retest":       "🔄 Ретест EMA20",
                "breakout":     "⚡ Пробой флэта",
                "impulse":      "🚀 Импульс",
                "impulse_cross":"🌱 Рождение тренда (EMA-пробой)",
                "alignment":    "🌊 Выравнивание тренда",
            }
            mode_label = _mode_labels.get(sig_mode, "📈 Тренд")

            catchup_note = ""
            if catchup_snapshot is not None:
                catchup_note = (
                    f"\n⚠️ catch-up after discovery: signal was {int(catchup_snapshot['lookback'])} bar(s) ago, "
                    f"price drift `{float(catchup_snapshot['slippage_pct']):+.2f}%`"
                )
            # Phase 2: enhanced signal summary for notification
            _enh_note = ""
            if _enh_adj is not None and abs(_enh_adj.total_bonus) > 0.5:
                _enh_parts = []
                if abs(_enh_adj.order_flow_bonus) > 0.5:
                    _enh_parts.append(f"OF {_enh_adj.order_flow_bonus:+.0f}")
                if abs(_enh_adj.derivatives_bonus) > 0.5:
                    _enh_parts.append(f"Deriv {_enh_adj.derivatives_bonus:+.0f}")
                if abs(_enh_adj.regime_bonus) > 0.5:
                    _regime_name = ""
                    if _enh_adj.details.get("regime"):
                        _regime_name = f" ({_enh_adj.details['regime'].get('name', '')})"
                    _enh_parts.append(f"Regime{_regime_name} {_enh_adj.regime_bonus:+.0f}")
                if abs(_enh_adj.top_gainer_bonus) > 0.5:
                    _enh_parts.append(f"TopGainer {_enh_adj.top_gainer_bonus:+.0f}")
                if abs(_enh_adj.multi_horizon_bonus) > 0.5:
                    _enh_parts.append(f"MH {_enh_adj.multi_horizon_bonus:+.0f}")
                if _enh_parts:
                    _enh_note = f"\n🔬 Enhanced: {' | '.join(_enh_parts)} = `{_enh_adj.total_bonus:+.1f}`"
            _buy_msg = (
                f"🟢 *СИГНАЛ ПОКУПКИ* — {mode_label}\n\n"
                f"*{sym}*  `[{tf}]`\n"
                f"💰 Цена: `{price:.6g}`\n"
                f"📐 EMA20: `{ef:.6g}`\n"
                f"📈 Наклон EMA20: `{slp:+.2f}%`\n"
                f"💪 ADX: `{adx:.1f}`\n"
                f"📊 RSI: `{rsi:.1f}`\n"
                f"🔊 Объём ×: `{vx:.2f}`\n"
                f"⚙️ Стоп: ATR×`{trail_k}`  |  Лимит: `{max_hold}` баров"
                f"{catchup_note}{_enh_note}\n\n"
                f"🎯 Буду проверять прогноз: {pos.prediction_summary()}"
            )
            # ФИКС: если send() бросит исключение (Telegram недоступен, rate-limit и т.п.),
            # позиция уже сохранена. Флаг buy_notified=False гарантирует, что
            # уведомление будет переотправлено в следующем цикле поллинга перед SELL.
            try:
                await send(_buy_msg)
                pos.buy_notified = True
                save_positions(state.positions)
            except Exception as _send_err:
                log.warning("BUY notification failed for %s, will retry: %s", sym, _send_err)
                # pos.buy_notified остаётся False — ретрансляция при следующем поллинге
        return

    # ── Open position: track predictions and check exit ──────────────────────

    # ФИКС: ретрансляция BUY-уведомления если send() упал при входе в позицию.
    # Это гарантирует что пользователь всегда получит BUY перед SELL.
    if not getattr(pos, "buy_notified", True):
        _mode_labels_retry = {
            "trend":        "📈 Тренд",
            "strong_trend": "💪 Сильный тренд",
            "impulse_speed":"⚡ Быстрое движение",
            "retest":       "🔄 Ретест EMA20",
            "breakout":     "⚡ Пробой флэта",
            "impulse":      "🚀 Импульс",
            "impulse_cross":"🌱 Рождение тренда (EMA-пробой)",
            "alignment":    "🌊 Выравнивание тренда",
        }
        _retry_label = _mode_labels_retry.get(getattr(pos, "signal_mode", "trend"), "📈 Тренд")
        try:
            await send(
                f"🟢 *СИГНАЛ ПОКУПКИ* — {_retry_label} _(повтор — первая отправка не удалась)_\n\n"
                f"*{sym}*  `[{tf}]`\n"
                f"💰 Цена входа: `{pos.entry_price:.6g}`\n"
                f"📐 EMA20: `{pos.entry_ema20:.6g}`\n"
                f"📈 Наклон EMA20: `{pos.entry_slope:+.2f}%`\n"
                f"💪 ADX: `{pos.entry_adx:.1f}`\n"
                f"📊 RSI: `{pos.entry_rsi:.1f}`\n"
                f"🔊 Объём ×: `{pos.entry_vol_x:.2f}`\n"
                f"⚙️ Стоп: ATR×`{pos.trail_k}`  |  Лимит: `{pos.max_hold_bars}` баров"
            )
            pos.buy_notified = True
            save_positions(state.positions)
            log.info("BUY notification retransmitted for %s [%s]", sym, tf)
        except Exception as _retry_err:
            log.warning("BUY notification retry also failed for %s: %s", sym, _retry_err)

    # ИСПРАВЛЕНИЕ: entry_idx определяется ЗДЕСЬ, до любого использования.
    # Ищем бар входа по timestamp в текущем окне свечей.
    entry_idx: Optional[int] = None
    for k in range(len(data["t"])):
        if int(data["t"][k]) == pos.entry_ts:
            entry_idx = k
            break
    current_bar_ts = int(data["t"][i])
    bar_ms = _tf_bar_ms(tf)
    if entry_idx is None:
        pos.bars_elapsed = max(0, (current_bar_ts - int(pos.entry_ts)) // bar_ms)

    # bars_elapsed: реальное число баров с момента входа
    if entry_idx is not None:
        pos.bars_elapsed = max(0, i - entry_idx)

    # Не проверяем выходы и форварды на том же самом закрытом баре, где вошли.
    # Иначе после перезапуска/следующего poll можно мгновенно получить fake exit при bars_elapsed=0.
    if pos.bars_elapsed <= 0:
        return

    # ── ATR Trailing Stop: обновляем уровень каждый бар вверх ────────────────
    close_now = float(c[i])
    atr_now   = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
    current_pnl = pos.pnl_pct(close_now)
    effective_trail_k = pos.trail_k
    continuation_profit_lock_active = _continuation_profit_lock_active(
        tf=tf,
        mode=getattr(pos, "signal_mode", "trend"),
        entry_rsi=float(getattr(pos, "entry_rsi", 50.0)),
        bars_elapsed=int(getattr(pos, "bars_elapsed", 0)),
        current_pnl=current_pnl,
        predictions=getattr(pos, "predictions", {}),
    )
    short_mode_profit_lock_active = _short_mode_profit_lock_active(
        tf=tf,
        mode=getattr(pos, "signal_mode", "trend"),
        bars_elapsed=int(getattr(pos, "bars_elapsed", 0)),
        current_pnl=current_pnl,
        predictions=getattr(pos, "predictions", {}),
    )
    if continuation_profit_lock_active or short_mode_profit_lock_active:
        trail_candidates = [pos.trail_k]
        floor_pcts: List[float] = []
        if continuation_profit_lock_active:
            trail_candidates.append(float(getattr(config, "CONTINUATION_PROFIT_LOCK_TRAIL_K", 1.4)))
            floor_pcts.append(float(getattr(config, "CONTINUATION_PROFIT_LOCK_FLOOR_PCT", 0.10)))
        if short_mode_profit_lock_active:
            trail_candidates.append(float(getattr(config, "SHORT_MODE_PROFIT_LOCK_TRAIL_K", 1.2)))
            floor_pcts.append(float(getattr(config, "SHORT_MODE_PROFIT_LOCK_FLOOR_PCT", 0.05)))
        effective_trail_k = min(trail_candidates)
        protect_floor_pct = max(floor_pcts) if floor_pcts else 0.0
        protect_floor = pos.entry_price * (1.0 + protect_floor_pct / 100.0)
        if protect_floor > pos.trail_stop:
            pos.trail_stop = protect_floor
        if continuation_profit_lock_active and tf == "1h":
            micro_data = await fetch_klines(session, sym, "15m", limit=config.LIVE_LIMIT)
            if micro_data is not None and len(micro_data) >= 60:
                micro_feat = await asyncio.to_thread(
                    compute_features,
                    micro_data["o"],
                    micro_data["h"],
                    micro_data["l"],
                    micro_data["c"].astype(float),
                    micro_data["v"],
                )
                micro_reason = _continuation_micro_exit_reason(
                    tf=tf,
                    mode=getattr(pos, "signal_mode", "trend"),
                    bars_elapsed=int(getattr(pos, "bars_elapsed", 0)),
                    data_15m=micro_data,
                    feat_15m=micro_feat,
                )
                if micro_reason:
                    micro_idx = len(micro_data["c"]) - 2
                    micro_close = float(micro_data["c"][micro_idx])
                    pnl = pos.pnl_pct(micro_close)
                    pnl_icon = "🟢" if pnl >= 0 else "🔴"
                    await send(
                        f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
                        f"*{sym}*  `[{tf}]`\n"
                        f"💰 Выход: `{micro_close:.6g}`\n"
                        f"📉 Причина: {micro_reason}\n"
                        f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
                        f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
                        f"⏱️ Баров в позиции: {pos.bars_elapsed}"
                    )
                    botlog.log_exit(
                        sym=sym,
                        tf=tf,
                        mode=getattr(pos, "signal_mode", "trend"),
                        entry_price=pos.entry_price,
                        exit_price=micro_close,
                        reason=micro_reason,
                        bars_held=pos.bars_elapsed,
                        trail_k=getattr(pos, "trail_k", 2.0),
                    )
                    _fill_trade_outcome_labels(
                        pos,
                        exit_pnl=pos.pnl_pct(micro_close),
                        exit_reason=micro_reason,
                        bars_held=pos.bars_elapsed,
                    )
                    del state.positions[sym]
                    save_positions(state.positions)
                    bar_ms_cd = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
                    cooldown_bars = _cooldown_bars_after_exit(
                        getattr(pos, "signal_mode", "trend"),
                        micro_reason,
                        tf=tf,
                        pnl_pct=pnl,
                    )
                    state.cooldowns[sym] = int(micro_data["t"][micro_idx]) + cooldown_bars * bar_ms_cd
                    state.cd_logged.pop(sym, None)
                    return
    if atr_now > 0:
        # Anti-whipsaw: enforce minimum trail buffer (% of price) per mode.
        # Floor protects against compressed-ATR + wide-daily-range coins where
        # bandit-chosen trail_k*ATR yields buffer < 1% on a 5%-daily-range coin.
        _mode_for_buf = getattr(pos, "signal_mode", "trend")
        _buf_pos = _compute_trail_buffer(
            price=close_now, trail_k=pos.trail_k, atr=atr_now, mode=_mode_for_buf
        )
        new_trail = close_now - _buf_pos  # П1: trail_k зависит от режима
        if effective_trail_k < pos.trail_k:
            # Profit-lock: tighter trail mid-trade. Apply floor here too so we don't
            # under-shoot the safety net.
            _buf_eff = _compute_trail_buffer(
                price=close_now, trail_k=effective_trail_k, atr=atr_now, mode=_mode_for_buf
            )
            new_trail = max(new_trail, close_now - _buf_eff)
        if new_trail > pos.trail_stop:
            pos.trail_stop = new_trail

    # Выход по ATR-трейлу
    if pos.trail_stop > 0 and close_now < pos.trail_stop:
        pnl      = pos.pnl_pct(close_now)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        await send(
            f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{close_now:.6g}`\n"
            f"📉 Причина: ATR-трейлинг пробит (стоп `{pos.trail_stop:.6g}`)\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        _exit_reason_atr = f"ATR-трейл пробит (стоп {pos.trail_stop:.6g})"
        botlog.log_exit(sym=sym, tf=tf, mode=getattr(pos,"signal_mode","trend"),
                        entry_price=pos.entry_price, exit_price=close_now,
                        reason=_exit_reason_atr,
                        bars_held=pos.bars_elapsed, trail_k=getattr(pos,"trail_k",2.0))
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(close_now),
            exit_reason=_exit_reason_atr,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)  # персистируем при выходе
        bar_ms_cd = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            _exit_reason_atr,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms_cd
        state.cd_logged.pop(sym, None)
        return

    # ── MACD Exit Warning: гистограмма падает N баров подряд ─────────────────
    # ФИКС Bug 8: streak обновляется только на НОВОМ закрытом баре (i != last_macd_bar_i).
    # Без этого при поллинге каждые 60с streak рос по тем же двум барам.
    macd_now  = feat["macd_hist"][i]
    macd_prev = feat["macd_hist"][i - 1] if i > 0 else np.nan
    if np.isfinite(macd_now) and np.isfinite(macd_prev) and i != pos.last_macd_bar_i:
        pos.last_macd_bar_i = i
        if macd_now < macd_prev:
            pos.macd_warn_streak += 1
        else:
            pos.macd_warn_streak = 0
            pos.macd_warned      = False  # сброс если MACD восстановился

    # Предупреждение только если: порог достигнут + ещё не предупреждали + прошло ≥ MACDWARN_BARS баров
    if (pos.macd_warn_streak >= config.MACDWARN_BARS
            and not pos.macd_warned
            and pos.bars_elapsed >= config.MACDWARN_BARS):
        pos.macd_warned = True
        cur_pnl = pos.pnl_pct(close_now)
        if _aux_notifications_enabled():
            await send(
                f"⚠️ *Предупреждение о развороте*\n\n"
                f"*{sym}*  `[{tf}]`\n"
                f"MACD гистограмма падает {pos.macd_warn_streak} баров подряд\n"
                f"Текущий PnL: `{cur_pnl:+.2f}%`  Цена: `{close_now:.6g}`\n"
                f"_Сигнала продажи ещё нет — наблюдаем_"
            )

    # Check forward prediction horizons
    horizons = getattr(pos, "prediction_horizons", ()) or _position_forward_horizons(pos.tf, pos.signal_mode)
    for h in horizons:
        if pos.predictions.get(h) is not None:
            continue  # already evaluated

        if entry_idx is None:
            continue  # entry bar no longer in window, skip

        target_bar = entry_idx + h
        # Target bar must exist AND be a closed bar (not the forming one)
        if target_bar > i:
            continue  # not yet reached

        future_price = float(c[target_bar])
        correct = future_price > pos.entry_price
        pos.predictions[h] = correct
        _ret_pct = (future_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0.0
        botlog.log_forward(sym=sym, tf=tf, mode=getattr(pos,"signal_mode","trend"),
                           horizon=h, entry_price=pos.entry_price,
                           forward_price=future_price, correct=correct)
        # ML: заполняем метку горизонта
        _fill_forward_labels(pos, h, _ret_pct)
        pnl = pos.pnl_pct(future_price)
        icon = "✅" if correct else "⚠️"
        if _aux_notifications_enabled():
            await send(
                f"{icon} *{sym}* — прогноз T+{h}\n"
                f"Вход: `{pos.entry_price:.6g}` → T+{h}: `{future_price:.6g}` "
                f"({pnl:+.2f}%)\n"
                f"{'✅ Предсказание верно' if correct else '❌ Предсказание НЕ подтвердилось'}\n"
                f"Статус: {pos.prediction_summary()}"
            )

    # П1: Принудительный выход — лимит баров зависит от режима сигнала
    if pos.bars_elapsed >= pos.max_hold_bars and not _time_exit_should_wait(feat, i, close_now):
        price    = float(c[i])
        pnl      = pos.pnl_pct(price)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        mins     = pos.max_hold_bars * (15 if tf == "15m" else 60)
        await send(
            f"⏱ *ВЫХОД ПО ВРЕМЕНИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{price:.6g}`\n"
            f"📉 Причина: лимит {pos.max_hold_bars} баров ({mins} мин) истёк\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        _exit_reason_time = f"время ({pos.max_hold_bars} баров)"
        botlog.log_exit(sym=sym, tf=tf, mode=getattr(pos,"signal_mode","trend"),
                        entry_price=pos.entry_price, exit_price=price,
                        reason=_exit_reason_time,
                        bars_held=pos.bars_elapsed, trail_k=getattr(pos,"trail_k",2.0))
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(price),
            exit_reason=_exit_reason_time,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)  # персистируем при выходе
        # Cooldown: COOLDOWN_BARS * длина бара в мс
        bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            _exit_reason_time,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms
        state.cd_logged.pop(sym, None)
        return

    fast_loss_reason = _fast_loss_ema_exit_reason(
        tf=tf,
        mode=getattr(pos, "signal_mode", "trend"),
        bars_elapsed=int(getattr(pos, "bars_elapsed", 0)),
        current_pnl=current_pnl,
        close_now=close_now,
        ema20=float(feat["ema_fast"][i]) if np.isfinite(feat["ema_fast"][i]) else np.nan,
        rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else np.nan,
    )
    if fast_loss_reason:
        pnl = pos.pnl_pct(close_now)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        await send(
            f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{close_now:.6g}`\n"
            f"📉 Причина: {fast_loss_reason}\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        botlog.log_exit(
            sym=sym,
            tf=tf,
            mode=getattr(pos, "signal_mode", "trend"),
            entry_price=pos.entry_price,
            exit_price=close_now,
            reason=fast_loss_reason,
            bars_held=pos.bars_elapsed,
            trail_k=getattr(pos, "trail_k", 2.0),
        )
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(close_now),
            exit_reason=fast_loss_reason,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)
        bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            fast_loss_reason,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms
        state.cd_logged.pop(sym, None)
        return

    cleanup_reason = _ranker_position_cleanup_reason(pos, feat, i, close_now=close_now)
    if cleanup_reason:
        pnl = pos.pnl_pct(close_now)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        await send(
            f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{close_now:.6g}`\n"
            f"📉 Причина: ⚠️ {cleanup_reason}\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        botlog.log_exit(
            sym=sym,
            tf=tf,
            mode=getattr(pos, "signal_mode", "trend"),
            entry_price=pos.entry_price,
            exit_price=close_now,
            reason=cleanup_reason,
            bars_held=pos.bars_elapsed,
            trail_k=getattr(pos, "trail_k", 2.0),
        )
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(close_now),
            exit_reason=cleanup_reason,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)
        bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            cleanup_reason,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms
        state.cd_logged.pop(sym, None)
        return

    quality_recheck_reason = _post_entry_quality_recheck_reason(pos, feat, i)
    if quality_recheck_reason:
        pnl = pos.pnl_pct(close_now)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        await send(
            f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{close_now:.6g}`\n"
            f"📉 Причина: ⚠️ {quality_recheck_reason}\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        botlog.log_exit(
            sym=sym,
            tf=tf,
            mode=getattr(pos, "signal_mode", "trend"),
            entry_price=pos.entry_price,
            exit_price=close_now,
            reason=quality_recheck_reason,
            bars_held=pos.bars_elapsed,
            trail_k=getattr(pos, "trail_k", 2.0),
        )
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(close_now),
            exit_reason=quality_recheck_reason,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)
        bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            quality_recheck_reason,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms
        state.cd_logged.pop(sym, None)
        return

    # ── Exit RL policy: adaptive exit/tighten recommendation ────────────────
    if getattr(config, "EXIT_RL_ENABLED", True) and pos.bars_elapsed >= 2:
        try:
            from exit_rl import recommend_exit_action, ACTION_EXIT, ACTION_TIGHTEN
            _prev_macd = float(feat["macd_hist"][i - 1]) if i > 0 and np.isfinite(feat["macd_hist"][i - 1]) else 0.0
            _exit_action, _exit_info = recommend_exit_action(
                current_pnl=current_pnl,
                bars_held=pos.bars_elapsed,
                max_hold_bars=pos.max_hold_bars,
                rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else 50.0,
                slope=float(feat["slope"][i]) if np.isfinite(feat["slope"][i]) else 0.0,
                adx=float(feat["adx"][i]) if np.isfinite(feat["adx"][i]) else 20.0,
                entry_adx=float(getattr(pos, "entry_adx", 20.0)),
                vol_x=float(feat["vol_x"][i]) if np.isfinite(feat["vol_x"][i]) else 1.0,
                macd_hist=float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0,
                prev_macd_hist=_prev_macd,
                close=close_now,
                trail_stop=pos.trail_stop,
                is_bull_day=is_bull_day_now,
                market_regime=str(getattr(config, "_market_regime", "neutral")),
                mode=getattr(pos, "signal_mode", "trend"),
            )
            if _exit_action == ACTION_TIGHTEN:
                _tighter_k = pos.trail_k * 0.75
                _tighter_trail = close_now - _tighter_k * atr_now
                if _tighter_trail > pos.trail_stop:
                    pos.trail_stop = _tighter_trail
                    log.info("EXIT_RL TIGHTEN %s: trail→%.6g (k×0.75) Q: %s",
                             sym, pos.trail_stop, _exit_info)
            elif _exit_action == ACTION_EXIT:
                log.info("EXIT_RL recommends EXIT %s: pnl=%.2f%% Q: %s",
                         sym, current_pnl, _exit_info)
                # RL exit triggers only if position is in profit or deep loss
                if current_pnl > 0.3 or current_pnl < -1.5:
                    price = close_now
                    pnl = current_pnl
                    pnl_icon = "🟢" if pnl >= 0 else "🔴"
                    _rl_reason = f"RL exit (Q_exit={_exit_info['q_exit']:.2f} vs Q_hold={_exit_info['q_hold']:.2f})"
                    await send(
                        f"🤖 *RL СИГНАЛ ПРОДАЖИ*\n\n"
                        f"*{sym}*  `[{tf}]`\n"
                        f"💰 Выход: `{price:.6g}`\n"
                        f"📉 Причина: {_rl_reason}\n"
                        f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
                        f"⏱ Баров в позиции: {pos.bars_elapsed}"
                    )
                    botlog.log_exit(sym=sym, tf=tf, mode=getattr(pos, "signal_mode", "trend"),
                                    entry_price=pos.entry_price, exit_price=price,
                                    reason=_rl_reason, bars_held=pos.bars_elapsed,
                                    trail_k=getattr(pos, "trail_k", 2.0))
                    _fill_trade_outcome_labels(pos, exit_pnl=pnl, exit_reason=_rl_reason,
                                               bars_held=pos.bars_elapsed)
                    del state.positions[sym]
                    save_positions(state.positions)
                    bar_ms_rl = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
                    cooldown_bars_rl = _cooldown_bars_after_exit(
                        getattr(pos, "signal_mode", "trend"), _rl_reason, tf=tf, pnl_pct=pnl)
                    state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars_rl * bar_ms_rl
                    state.cd_logged.pop(sym, None)
                    return
        except Exception as e:
            log.debug("Exit RL error for %s: %s", sym, e)

    # Check exit conditions
    # ФИКС: WEAK сигналы игнорируются первые MIN_WEAK_EXIT_BARS баров.
    # RSI дивергенция / vol exhaustion / EMA fan collapse могут существовать
    # ДО входа и немедленно выбивать позицию (AR 11.03.2026: выход на 0 баре).
    # Hard exits (ATR-трейл, время) защищены выше и не затронуты этим фильтром.
    min_weak_bars = _min_weak_exit_bars(getattr(pos, "signal_mode", "trend"))
    reason = check_exit_conditions(
        feat,
        i,
        c,
        mode=getattr(pos, "signal_mode", "trend"),
        bars_elapsed=int(getattr(pos, "bars_elapsed", 0)),
        tf=tf,
    )
    if reason:
        is_weak = reason.startswith("⚠️ WEAK:")
        if is_weak and pos.bars_elapsed < min_weak_bars:
            log.debug(
                "EXIT SUPPRESSED %s [%s] bars=%d < %d: %s",
                sym, tf, pos.bars_elapsed, min_weak_bars, reason
            )
            reason = None  # игнорируем WEAK на первых барах
        elif is_weak and _apply_trend_hold_weak_exit_override(
            pos=pos,
            feat=feat,
            idx=i,
            close_now=close_now,
            current_pnl=current_pnl,
            tf=tf,
        ):
            log.info(
                "TREND HOLD SUPPRESSED %s [%s] bars=%d pnl=%.2f score=%.1f trail=%.6g: %s",
                sym,
                tf,
                pos.bars_elapsed,
                current_pnl,
                float(getattr(pos, "candidate_score_at_entry", 0.0)),
                float(getattr(pos, "trail_stop", 0.0)),
                reason,
            )
            reason = None

    if reason:
        price = float(c[i])
        pnl   = pos.pnl_pct(price)
        pnl_icon = "🟢" if pnl >= 0 else "🔴"

        await send(
            f"🔴 *СИГНАЛ ПРОДАЖИ*\n\n"
            f"*{sym}*  `[{tf}]`\n"
            f"💰 Выход: `{price:.6g}`\n"
            f"📉 Причина: {reason}\n"
            f"{pnl_icon} Изменение от входа: `{pnl:+.2f}%`\n"
            f"🎯 Точность прогнозов: {pos.prediction_summary()}\n"
            f"⏱ Баров в позиции: {pos.bars_elapsed}"
        )
        botlog.log_exit(sym=sym, tf=tf, mode=getattr(pos,"signal_mode","trend"),
                        entry_price=pos.entry_price, exit_price=price,
                        reason=reason, bars_held=pos.bars_elapsed,
                        trail_k=getattr(pos,"trail_k",2.0))
        _fill_trade_outcome_labels(
            pos,
            exit_pnl=pos.pnl_pct(price),
            exit_reason=reason,
            bars_held=pos.bars_elapsed,
        )
        del state.positions[sym]
        save_positions(state.positions)  # персистируем при выходе
        bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
        cooldown_bars = _cooldown_bars_after_exit(
            getattr(pos, "signal_mode", "trend"),
            reason,
            tf=tf,
            pnl_pct=pnl,
        )
        state.cooldowns[sym] = int(data["t"][i]) + cooldown_bars * bar_ms
        state.cd_logged.pop(sym, None)


# ── Main monitoring loop ───────────────────────────────────────────────────────

async def monitoring_loop(state: MonitorState, send: SendFn) -> None:
    symbols_str = ", ".join(r.symbol for r in state.hot_coins)
    if _aux_notifications_enabled():
        try:
            await send(
                f"▶️ *Мониторинг запущен*\n"
                f"Слежу за *{len(state.hot_coins)}* монетами:\n"
                f"`{symbols_str}`\n"
                f"Интервал опроса: {config.POLL_SEC}с"
            )
        except Exception as e:
            log.warning("monitoring_loop: startup send failed: %s", e)

    # ── Phase 2: Start enhanced data sources ────────────────────
    _enhanced_instance = None
    if getattr(config, "WS_ENABLED", False) or getattr(config, "DERIVATIVES_ENABLED", False):
        try:
            from enhanced_signals import start_enhanced_signals
            watchlist_syms = [r.symbol for r in state.hot_coins]
            _enhanced_instance = await start_enhanced_signals(watchlist_syms)
            log.info("Enhanced signals started for %d symbols", len(watchlist_syms))
        except Exception:
            log.warning("Enhanced signals startup failed, continuing without", exc_info=True)

    _heartbeat_counter = 0
    async with aiohttp.ClientSession() as session:
        while state.running:
            try:
                tasks = [
                    _poll_coin(session, r, state, send)
                    for r in state.hot_coins
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [exc for exc in results if isinstance(exc, Exception)]
                if errors:
                    for exc in errors:
                        log.warning("Poll error: %s", exc)
                await asyncio.sleep(0)

                now_ms = int(asyncio.get_running_loop().time() * 1000)
                discovery_sec = int(getattr(config, "DISCOVERY_SCAN_SEC", 0))
                if discovery_sec > 0:
                    discovery_ms = discovery_sec * 1000
                    discovery_busy = state.discovery_task is not None and not state.discovery_task.done()
                    if not discovery_busy and now_ms >= state.last_discovery_ts + discovery_ms:
                        async def _run_discovery() -> None:
                            try:
                                added = await _discover_new_hot_coins(session, state, send)
                                if added:
                                    log.info("discovery scan added %d coin(s)", added)
                            except Exception as exc:
                                log.warning("discovery scan failed: %s", exc)
                        state.last_discovery_ts = now_ms
                        state.discovery_task = asyncio.create_task(_run_discovery())

                # ── Correlation Guard: обновление матрицы + prune каждые ~15 минут ──
                _corr_guard_interval = max(1, int(getattr(config, "CORR_CACHE_TTL_MIN", 15)) * 60 // config.POLL_SEC)
                if bool(getattr(config, "CORR_GUARD_ENABLED", False)) and _heartbeat_counter % _corr_guard_interval == 0:
                    try:
                        _cg_cache = corr_guard.get_or_create_cache(state.__dict__)
                        _open_syms = list(state.positions.keys())
                        if len(_open_syms) >= 2:
                            _cg_cache = await corr_guard.build_matrix_async(session, _open_syms, _cg_cache)
                            _is_bull_cg, _is_bear_cg = corr_guard.get_bull_bear_flags(state.__dict__)
                            _prune_list = corr_guard.prune_candidates(
                                state.positions,
                                state.__dict__.get("last_prices", {}),
                                _cg_cache,
                                is_bull_day=_is_bull_cg,
                                is_bear_day=_is_bear_cg,
                            )
                            for _pc in _prune_list:
                                _prune_pos = state.positions.get(_pc.symbol)
                                if _prune_pos is None:
                                    continue
                                _prune_price = state.__dict__.get("last_prices", {}).get(_pc.symbol)
                                if _prune_price is None:
                                    log.warning("corr_guard prune: no last_price for %s, skip", _pc.symbol)
                                    continue
                                log.info("CORR GUARD PRUNE %s: %s", _pc.symbol, _pc.reason)
                                # Форсируем выход: устанавливаем trail_stop выше текущей цены.
                                # Следующий poll-цикл (через POLL_SEC) закроет позицию
                                # стандартным ATR-trail путём — отправит SELL + запишет в critic/botlog.
                                _prune_pos.trail_stop = _prune_price * 1.001  # +0.1% — гарантированный триггер
                    except Exception as _cg_err:
                        log.warning("corr_guard periodic update failed: %s", _cg_err)

                # Heartbeat каждые ~10 минут (600с / POLL_SEC итераций)
                _heartbeat_counter += 1
                if _heartbeat_counter % max(1, 600 // config.POLL_SEC) == 0:
                    log.info(
                        "monitoring_loop alive: %d coins, %d positions, errors=%d",
                        len(state.hot_coins), len(state.positions), len(errors),
                    )

            except Exception as e:
                log.error("Monitor loop error: %s", e)

            await asyncio.sleep(config.POLL_SEC)

    if state.discovery_task is not None and not state.discovery_task.done():
        state.discovery_task.cancel()
    state.discovery_task = None

    # ── Phase 2: Stop enhanced data sources ───────────────────
    if _enhanced_instance is not None:
        try:
            from enhanced_signals import stop_enhanced_signals
            await stop_enhanced_signals()
            log.info("Enhanced signals stopped")
        except Exception:
            log.warning("Enhanced signals shutdown failed", exc_info=True)

    if _aux_notifications_enabled():
        try:
            await send("⏹ *Мониторинг остановлен*")
        except Exception as e:
            log.warning("monitoring_loop: shutdown send failed: %s", e)
