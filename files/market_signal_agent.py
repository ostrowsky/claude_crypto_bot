from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import numpy as np

import agentlog
import config
from monitor import (
    _analyze_coin_live,
    _check_mtf,
    _continuation_micro_exit_reason,
    _continuation_profit_lock_active,
    _cooldown_bars_after_exit,
    _entry_score_borderline_bypass_ok,
    _entry_score_floor,
    _entry_signal_score,
    _fast_loss_ema_exit_reason,
    _forecast_return_score_bonus,
    _impulse_speed_entry_guard,
    _late_1h_continuation_guard,
    _min_weak_exit_bars,
    _mtf_soft_penalty_from_reason,
    _post_entry_quality_recheck_reason,
    _short_mode_profit_lock_active,
    _time_block_1h_continuation_bonus,
    _time_block_1h_continuation_profile,
    _time_exit_should_wait,
    _top_mover_score_bonus,
)
from strategy import (
    check_alignment_conditions,
    check_breakout_conditions,
    check_entry_conditions,
    check_exit_conditions,
    check_impulse_conditions,
    check_retest_conditions,
    check_trend_surge_conditions,
    compute_features,
    detect_market_regime,
    fetch_klines,
    get_entry_mode,
    is_bull_day,
)


log = logging.getLogger("market_agent")
POSITIONS_FILE = Path("agent_positions.json")
CHAT_IDS_FILE = Path(".chat_ids")
STATE_FILE = Path(".runtime") / "market_agent_state.json"
STATUS_FILE = Path(".runtime") / "market_agent_status.json"


@dataclass
class AgentPosition:
    symbol: str
    tf: str
    entry_price: float
    entry_bar: int
    entry_ts: int
    entry_ema20: float
    entry_slope: float
    entry_adx: float
    entry_rsi: float
    entry_vol_x: float
    forecast_return_pct: float = 0.0
    today_change_pct: float = 0.0
    predictions: Dict[int, Optional[bool]] = field(default_factory=dict)
    bars_elapsed: int = 0
    signal_mode: str = "trend"
    trail_k: float = 2.0
    max_hold_bars: int = 16
    trail_stop: float = 0.0
    last_bar_ts: int = 0

    def pnl_pct(self, current_price: float) -> float:
        return (current_price / self.entry_price - 1.0) * 100.0


def _position_key(symbol: str, tf: str) -> str:
    return f"{symbol}|{tf}"


def _tf_bar_ms(tf: str) -> int:
    return 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000


def _load_chat_ids() -> list[int]:
    for candidate in (CHAT_IDS_FILE, Path(__file__).resolve().parent / ".chat_ids"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [int(x) for x in payload]
        except Exception:
            continue
    return []


def _save_positions(positions: Dict[str, AgentPosition]) -> None:
    payload = {}
    for key, pos in positions.items():
        payload[key] = {
            "symbol": pos.symbol,
            "tf": pos.tf,
            "entry_price": pos.entry_price,
            "entry_bar": pos.entry_bar,
            "entry_ts": pos.entry_ts,
            "entry_ema20": pos.entry_ema20,
            "entry_slope": pos.entry_slope,
            "entry_adx": pos.entry_adx,
            "entry_rsi": pos.entry_rsi,
            "entry_vol_x": pos.entry_vol_x,
            "forecast_return_pct": pos.forecast_return_pct,
            "today_change_pct": pos.today_change_pct,
            "predictions": pos.predictions,
            "bars_elapsed": pos.bars_elapsed,
            "signal_mode": pos.signal_mode,
            "trail_k": pos.trail_k,
            "max_hold_bars": pos.max_hold_bars,
            "trail_stop": pos.trail_stop,
            "last_bar_ts": pos.last_bar_ts,
        }
    POSITIONS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_positions() -> Dict[str, AgentPosition]:
    if not POSITIONS_FILE.exists():
        return {}
    try:
        raw = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    positions: Dict[str, AgentPosition] = {}
    if not isinstance(raw, dict):
        return positions
    for key, data in raw.items():
        if not isinstance(data, dict):
            continue
        positions[key] = AgentPosition(
            symbol=str(data.get("symbol", "")),
            tf=str(data.get("tf", "15m")),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_bar=int(data.get("entry_bar", 0)),
            entry_ts=int(data.get("entry_ts", 0)),
            entry_ema20=float(data.get("entry_ema20", 0.0)),
            entry_slope=float(data.get("entry_slope", 0.0)),
            entry_adx=float(data.get("entry_adx", 0.0)),
            entry_rsi=float(data.get("entry_rsi", 0.0)),
            entry_vol_x=float(data.get("entry_vol_x", 0.0)),
            forecast_return_pct=float(data.get("forecast_return_pct", 0.0)),
            today_change_pct=float(data.get("today_change_pct", 0.0)),
            predictions={int(k): v for k, v in dict(data.get("predictions", {})).items()},
            bars_elapsed=int(data.get("bars_elapsed", 0)),
            signal_mode=str(data.get("signal_mode", "trend")),
            trail_k=float(data.get("trail_k", 2.0)),
            max_hold_bars=int(data.get("max_hold_bars", 16)),
            trail_stop=float(data.get("trail_stop", 0.0)),
            last_bar_ts=int(data.get("last_bar_ts", 0)),
        )
    return positions


def _save_state(last_exit_bar: Dict[str, int]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"last_exit_bar": last_exit_bar}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_state() -> Dict[str, int]:
    if not STATE_FILE.exists():
        return {}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = payload.get("last_exit_bar", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


def _status_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(
    *,
    started_at: str,
    poll_sec: int,
    cycle_running: bool,
    n_open_positions: int,
    last_cycle_started_at: Optional[str] = None,
    last_cycle_finished_at: Optional[str] = None,
    last_cycle_stats: Optional[dict] = None,
    last_error: str = "",
) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "worker": {
            "started_at": started_at,
            "last_heartbeat": _status_now(),
            "mode": "market_agent_headless",
            "poll_sec": int(poll_sec),
        },
        "collector": {
            "running": bool(cycle_running),
            "last_cycle_started_at": last_cycle_started_at,
            "last_cycle_finished_at": last_cycle_finished_at,
            "last_cycle_stats": last_cycle_stats or {},
            "last_error": str(last_error or ""),
            "open_positions": int(n_open_positions),
        },
    }
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _send_telegram(session: aiohttp.ClientSession, text: str) -> None:
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=12)
    for chat_id in chat_ids:
        try:
            async with session.post(
                url,
                json={"chat_id": chat_id, "text": text},
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                await resp.text()
        except Exception as exc:
            log.warning("telegram send failed for %s: %s", chat_id, exc)


def _entry_params(mode: str, tf: str) -> tuple[float, int]:
    if mode == "breakout":
        return float(getattr(config, "ATR_TRAIL_K_BREAKOUT", 1.5)), int(getattr(config, "MAX_HOLD_BARS_BREAKOUT", 6))
    if mode == "retest":
        return float(getattr(config, "ATR_TRAIL_K_RETEST", 1.8)), int(getattr(config, "MAX_HOLD_BARS_RETEST", 10))
    if mode in ("strong_trend", "impulse_speed"):
        return float(getattr(config, "ATR_TRAIL_K_STRONG", 2.5)), int(
            getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
        )
    if mode == "impulse":
        return float(getattr(config, "ATR_TRAIL_K", 2.0)), int(
            getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
        )
    return float(getattr(config, "ATR_TRAIL_K", 2.0)), int(
        getattr(config, "MAX_HOLD_BARS_15M", 48) if tf == "15m" else getattr(config, "MAX_HOLD_BARS", 16)
    )


def _prediction_summary(pos: AgentPosition) -> str:
    parts = []
    for h in getattr(config, "FORWARD_BARS", [3, 5, 10]):
        value = pos.predictions.get(h)
        if value is None:
            parts.append(f"T+{h}: pending")
        elif value:
            parts.append(f"T+{h}: ok")
        else:
            parts.append(f"T+{h}: fail")
    return " | ".join(parts)


async def _send_entry_alert(session: aiohttp.ClientSession, pos: AgentPosition) -> None:
    text = (
        "AGENT BUY\n\n"
        f"{pos.symbol} [{pos.tf}] {pos.signal_mode}\n"
        f"price: {pos.entry_price:.6g}\n"
        f"ema20: {pos.entry_ema20:.6g}\n"
        f"slope: {pos.entry_slope:+.2f}%\n"
        f"adx: {pos.entry_adx:.1f}\n"
        f"rsi: {pos.entry_rsi:.1f}\n"
        f"vol_x: {pos.entry_vol_x:.2f}\n"
        f"trail_k: {pos.trail_k:.2f}  hold: {pos.max_hold_bars} bars\n"
        f"forecast: {pos.forecast_return_pct:+.2f}%  today: {pos.today_change_pct:+.2f}%"
    )
    await _send_telegram(session, text)


async def _send_exit_alert(session: aiohttp.ClientSession, pos: AgentPosition, exit_price: float, reason: str) -> None:
    text = (
        "AGENT SELL\n\n"
        f"{pos.symbol} [{pos.tf}] {pos.signal_mode}\n"
        f"exit: {exit_price:.6g}\n"
        f"reason: {reason}\n"
        f"pnl: {pos.pnl_pct(exit_price):+.2f}%\n"
        f"bars: {pos.bars_elapsed}\n"
        f"{_prediction_summary(pos)}"
    )
    await _send_telegram(session, text)


async def _compute_features(data) -> tuple[np.ndarray, dict]:
    return await asyncio.to_thread(
        lambda: (
            data["c"].astype(float),
            compute_features(data["o"], data["h"], data["l"], data["c"].astype(float), data["v"]),
        )
    )


def _determine_signal_mode(
    *,
    entry_ok: bool,
    brk_ok: bool,
    ret_ok: bool,
    surge_ok: bool,
    imp_ok: bool,
    tf: str,
    feat: dict,
    c: np.ndarray,
    i: int,
) -> Optional[str]:
    if brk_ok:
        return "breakout"
    if ret_ok:
        return "retest"
    if entry_ok:
        return get_entry_mode(feat, i)
    if surge_ok:
        return "impulse_speed"
    if imp_ok:
        return "impulse"
    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
    if aln_ok:
        return "alignment"
    return None


async def _entry_candidate(
    session: aiohttp.ClientSession,
    symbol: str,
    tf: str,
    data,
    c: np.ndarray,
    feat: dict,
) -> Optional[dict]:
    i = len(c) - 2
    entry_ok, _ = check_entry_conditions(feat, i, c, tf=tf)
    brk_ok, _ = check_breakout_conditions(feat, i)
    ret_ok, _ = check_retest_conditions(feat, i)
    surge_ok, _ = check_trend_surge_conditions(feat, i)
    imp_ok, _ = check_impulse_conditions(feat, i)
    aln_ok, _ = check_alignment_conditions(feat, i, tf=tf)
    if not any((entry_ok, brk_ok, ret_ok, surge_ok, imp_ok, aln_ok)):
        return None

    mode = _determine_signal_mode(
        entry_ok=entry_ok,
        brk_ok=brk_ok,
        ret_ok=ret_ok,
        surge_ok=surge_ok,
        imp_ok=imp_ok,
        tf=tf,
        feat=feat,
        c=c,
        i=i,
    )
    if mode is None:
        return None

    report = await _analyze_coin_live(symbol, tf, data)
    price = float(c[i])
    ema20 = float(feat["ema_fast"][i])
    slope = float(feat["slope"][i])
    adx = float(feat["adx"][i])
    rsi = float(feat["rsi"][i])
    vol_x = float(feat["vol_x"][i])
    daily_range = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0

    candidate_score = _entry_signal_score(
        mode=mode,
        price=price,
        ema20=ema20,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    candidate_score += _top_mover_score_bonus(float(getattr(report, "today_change_pct", 0.0)))
    candidate_score += _forecast_return_score_bonus(float(getattr(report, "forecast_return_pct", 0.0)))

    continuation_profile = _time_block_1h_continuation_profile(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )
    candidate_score += _time_block_1h_continuation_bonus(
        tf=tf,
        mode=mode,
        slope=slope,
        adx=adx,
        rsi=rsi,
        vol_x=vol_x,
        daily_range=daily_range,
    )

    if _late_1h_continuation_guard(
        tf=tf,
        mode=mode,
        continuation_profile=continuation_profile,
        candidate_score=candidate_score,
        price=price,
        ema20=ema20,
        rsi=rsi,
        daily_range=daily_range,
    ):
        agentlog.log_blocked(symbol, tf, price, "late 1h continuation", signal_type="late_continuation", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
        return None

    impulse_guard_reason = _impulse_speed_entry_guard(
        tf=tf,
        mode=mode,
        feat=feat,
        i=i,
        price=price,
        ema20=ema20,
        rsi=rsi,
        adx=adx,
        daily_range=daily_range,
    )
    if impulse_guard_reason:
        agentlog.log_blocked(symbol, tf, price, impulse_guard_reason, signal_type="impulse_guard", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
        return None

    if tf == "1h" and getattr(config, "MTF_ENABLED", True):
        mtf_ok, mtf_reason = await _check_mtf(
            session,
            symbol,
            mode=mode,
            candidate_score=candidate_score,
            slope=slope,
            adx=adx,
            rsi=rsi,
            vol_x=vol_x,
            daily_range=daily_range,
        )
        if not mtf_ok:
            agentlog.log_blocked(symbol, tf, price, f"MTF: {mtf_reason}", signal_type="mtf", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
            return None
        candidate_score -= _mtf_soft_penalty_from_reason(mtf_reason)

    if getattr(config, "ENTRY_SCORE_MIN_ENABLED", False):
        min_score = _entry_score_floor(tf)
        if candidate_score < min_score:
            if not _entry_score_borderline_bypass_ok(
                tf=tf,
                mode=mode,
                candidate_score=candidate_score,
                min_score=min_score,
                price=price,
                ema20=ema20,
                slope=slope,
                adx=adx,
                rsi=rsi,
                vol_x=vol_x,
                daily_range=daily_range,
            ):
                agentlog.log_blocked(symbol, tf, price, f"entry score {candidate_score:.2f} < floor {min_score:.2f}", signal_type="entry_score", rsi=rsi, adx=adx, vol_x=vol_x, daily_range=daily_range)
                return None

    trail_k, max_hold_bars = _entry_params(mode, tf)
    atr_val = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
    trail_stop = price - trail_k * atr_val if atr_val > 0 else 0.0
    return {
        "mode": mode,
        "price": price,
        "ema20": ema20,
        "slope": slope,
        "adx": adx,
        "rsi": rsi,
        "vol_x": vol_x,
        "daily_range": daily_range,
        "macd_hist": float(feat["macd_hist"][i]) if np.isfinite(feat["macd_hist"][i]) else 0.0,
        "trail_k": trail_k,
        "max_hold_bars": max_hold_bars,
        "trail_stop": trail_stop,
        "forecast_return_pct": float(getattr(report, "forecast_return_pct", 0.0)),
        "today_change_pct": float(getattr(report, "today_change_pct", 0.0)),
        "bar_ts": int(data["t"][i]),
        "bar_idx": i,
    }


async def _evaluate_open_position(
    session: aiohttp.ClientSession,
    pos: AgentPosition,
    data,
    c: np.ndarray,
    feat: dict,
) -> Optional[dict]:
    i = len(c) - 2
    live_report = await _analyze_coin_live(pos.symbol, pos.tf, data)
    pos.forecast_return_pct = float(getattr(live_report, "forecast_return_pct", pos.forecast_return_pct))
    pos.today_change_pct = float(getattr(live_report, "today_change_pct", pos.today_change_pct))

    entry_idx: Optional[int] = None
    for idx, ts in enumerate(data["t"]):
        if int(ts) == int(pos.entry_ts):
            entry_idx = idx
            break
    if entry_idx is None:
        pos.bars_elapsed = max(0, (int(data["t"][i]) - int(pos.entry_ts)) // _tf_bar_ms(pos.tf))
    else:
        pos.bars_elapsed = max(0, i - entry_idx)
    if pos.bars_elapsed <= 0:
        pos.last_bar_ts = int(data["t"][i])
        return None

    close_now = float(c[i])
    current_pnl = pos.pnl_pct(close_now)
    atr_now = float(feat["atr"][i]) if np.isfinite(feat["atr"][i]) else 0.0
    effective_trail_k = pos.trail_k

    if _continuation_profit_lock_active(
        tf=pos.tf,
        mode=pos.signal_mode,
        entry_rsi=pos.entry_rsi,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        predictions=pos.predictions,
    ):
        effective_trail_k = min(
            pos.trail_k,
            float(getattr(config, "CONTINUATION_PROFIT_LOCK_TRAIL_K", 1.4)),
        )
        protect_floor_pct = float(getattr(config, "CONTINUATION_PROFIT_LOCK_FLOOR_PCT", 0.10))
        protect_floor = pos.entry_price * (1.0 + protect_floor_pct / 100.0)
        if protect_floor > pos.trail_stop:
            pos.trail_stop = protect_floor
        if pos.tf == "1h":
            micro_data = await fetch_klines(session, pos.symbol, "15m", limit=config.LIVE_LIMIT)
            if micro_data is not None and len(micro_data) >= 60:
                micro_c, micro_feat = await _compute_features(micro_data)
                micro_reason = _continuation_micro_exit_reason(
                    tf=pos.tf,
                    mode=pos.signal_mode,
                    bars_elapsed=pos.bars_elapsed,
                    data_15m=micro_data,
                    feat_15m=micro_feat,
                )
                if micro_reason:
                    exit_idx = len(micro_c) - 2
                    exit_price = float(micro_c[exit_idx])
                    return {"price": exit_price, "reason": micro_reason, "bar_ts": int(micro_data["t"][exit_idx])}

    if _short_mode_profit_lock_active(
        tf=pos.tf,
        mode=pos.signal_mode,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        predictions=pos.predictions,
    ):
        effective_trail_k = min(
            effective_trail_k,
            float(getattr(config, "SHORT_MODE_PROFIT_LOCK_TRAIL_K", 1.2)),
        )
        protect_floor_pct = float(getattr(config, "SHORT_MODE_PROFIT_LOCK_FLOOR_PCT", 0.05))
        protect_floor = pos.entry_price * (1.0 + protect_floor_pct / 100.0)
        if protect_floor > pos.trail_stop:
            pos.trail_stop = protect_floor

    if atr_now > 0:
        new_trail = close_now - pos.trail_k * atr_now
        if effective_trail_k < pos.trail_k:
            new_trail = max(new_trail, close_now - effective_trail_k * atr_now)
        if new_trail > pos.trail_stop:
            pos.trail_stop = new_trail

    if pos.trail_stop > 0 and close_now < pos.trail_stop:
        return {"price": close_now, "reason": f"ATR trail broken ({pos.trail_stop:.6g})", "bar_ts": int(data["t"][i])}

    if pos.bars_elapsed >= pos.max_hold_bars and not _time_exit_should_wait(feat, i, close_now):
        return {"price": close_now, "reason": f"time ({pos.max_hold_bars} bars)", "bar_ts": int(data["t"][i])}

    fast_loss_reason = _fast_loss_ema_exit_reason(
        tf=pos.tf,
        mode=pos.signal_mode,
        bars_elapsed=pos.bars_elapsed,
        current_pnl=current_pnl,
        close_now=close_now,
        ema20=float(feat["ema_fast"][i]) if np.isfinite(feat["ema_fast"][i]) else np.nan,
        rsi=float(feat["rsi"][i]) if np.isfinite(feat["rsi"][i]) else np.nan,
    )
    if fast_loss_reason:
        return {"price": close_now, "reason": fast_loss_reason, "bar_ts": int(data["t"][i])}

    quality_recheck_reason = _post_entry_quality_recheck_reason(pos, feat, i)
    if quality_recheck_reason:
        return {"price": close_now, "reason": quality_recheck_reason, "bar_ts": int(data["t"][i])}

    weak_bars = _min_weak_exit_bars(pos.signal_mode)
    reason = check_exit_conditions(feat, i, c, mode=pos.signal_mode, bars_elapsed=pos.bars_elapsed, tf=pos.tf)
    if reason:
        if reason.startswith("⚠️ WEAK:") and pos.bars_elapsed < weak_bars:
            reason = None
    if reason:
        return {"price": close_now, "reason": reason, "bar_ts": int(data["t"][i])}

    pos.last_bar_ts = int(data["t"][i])
    return None


async def _poll_symbol_tf(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
    symbol: str,
    tf: str,
) -> tuple[bool, Optional[str]]:
    key = _position_key(symbol, tf)
    pos = positions.get(key)
    data = await fetch_klines(session, symbol, tf, limit=config.LIVE_LIMIT)
    if data is None or len(data) < 60:
        return False, None
    c, feat = await _compute_features(data)
    i = len(c) - 2
    current_bar_ts = int(data["t"][i])
    if pos is None and last_exit_bar.get(key) == current_bar_ts:
        return False, None

    if pos is None:
        candidate = await _entry_candidate(session, symbol, tf, data, c, feat)
        if candidate is None:
            return False, None
        pos = AgentPosition(
            symbol=symbol,
            tf=tf,
            entry_price=float(candidate["price"]),
            entry_bar=int(candidate["bar_idx"]),
            entry_ts=int(candidate["bar_ts"]),
            entry_ema20=float(candidate["ema20"]),
            entry_slope=float(candidate["slope"]),
            entry_adx=float(candidate["adx"]),
            entry_rsi=float(candidate["rsi"]),
            entry_vol_x=float(candidate["vol_x"]),
            forecast_return_pct=float(candidate["forecast_return_pct"]),
            today_change_pct=float(candidate["today_change_pct"]),
            predictions={h: None for h in getattr(config, "FORWARD_BARS", [3, 5, 10])},
            signal_mode=str(candidate["mode"]),
            trail_k=float(candidate["trail_k"]),
            max_hold_bars=int(candidate["max_hold_bars"]),
            trail_stop=float(candidate["trail_stop"]),
            last_bar_ts=int(candidate["bar_ts"]),
        )
        positions[key] = pos
        _save_positions(positions)
        agentlog.log_entry(
            sym=pos.symbol,
            tf=pos.tf,
            mode=pos.signal_mode,
            price=pos.entry_price,
            ema20=pos.entry_ema20,
            slope=pos.entry_slope,
            rsi=pos.entry_rsi,
            adx=pos.entry_adx,
            vol_x=pos.entry_vol_x,
            macd_hist=float(candidate["macd_hist"]),
            daily_range=float(candidate["daily_range"]),
            trail_k=pos.trail_k,
            max_hold_bars=pos.max_hold_bars,
            forecast_return_pct=pos.forecast_return_pct,
            today_change_pct=pos.today_change_pct,
        )
        await _send_entry_alert(session, pos)
        return True, f"{symbol} [{tf}] {pos.signal_mode}"

    exit_event = await _evaluate_open_position(session, pos, data, c, feat)
    if exit_event is None:
        _save_positions(positions)
        return False, None

    exit_price = float(exit_event["price"])
    exit_reason = str(exit_event["reason"])
    agentlog.log_exit(
        sym=pos.symbol,
        tf=pos.tf,
        mode=pos.signal_mode,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        reason=exit_reason,
        bars_held=pos.bars_elapsed,
        trail_k=pos.trail_k,
    )
    await _send_exit_alert(session, pos, exit_price, exit_reason)
    last_exit_bar[key] = int(exit_event["bar_ts"])
    del positions[key]
    _save_positions(positions)
    _save_state(last_exit_bar)
    return False, None


async def _run_cycle(
    session: aiohttp.ClientSession,
    positions: Dict[str, AgentPosition],
    last_exit_bar: Dict[str, int],
) -> tuple[list[str], int]:
    symbols = list(config.load_watchlist())
    bull, btc_price, btc_ema50 = await is_bull_day(session)
    regime = await detect_market_regime(session)
    config._bull_day_active = bull
    config._current_regime = regime.name
    config._btc_vs_ema50 = ((btc_price / btc_ema50) - 1.0) * 100.0 if btc_ema50 > 0 else 0.0

    sem = asyncio.Semaphore(12)
    entries: list[str] = []

    async def _wrapped(sym: str, tf: str) -> None:
        async with sem:
            try:
                opened, desc = await _poll_symbol_tf(session, positions, last_exit_bar, sym, tf)
                if opened and desc:
                    entries.append(desc)
            except Exception as exc:
                log.warning("agent poll error %s [%s]: %s", sym, tf, exc)

    tasks = [_wrapped(sym, tf) for sym in symbols for tf in config.TIMEFRAMES]
    await asyncio.gather(*tasks)
    agentlog.log_analysis(
        n_scanned=len(symbols) * len(config.TIMEFRAMES),
        n_entries=len(entries),
        n_open_positions=len(positions),
    )
    _save_state(last_exit_bar)
    return entries, len(symbols) * len(config.TIMEFRAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone market signal agent for watchlist.")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit")
    parser.add_argument("--poll-sec", type=int, default=None, help="Override polling interval")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    positions = _load_positions()
    last_exit_bar = _load_state()
    if args.poll_sec:
        config.POLL_SEC = int(args.poll_sec)

    log.info("market agent started with %d restored positions", len(positions))
    headers = {"User-Agent": "market-signal-agent/1.0"}
    started_at = _status_now()
    poll_sec = max(5, int(getattr(config, "POLL_SEC", 60)))
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            cycle_started_at = _status_now()
            _write_status(
                started_at=started_at,
                poll_sec=poll_sec,
                cycle_running=True,
                n_open_positions=len(positions),
                last_cycle_started_at=cycle_started_at,
                last_cycle_finished_at=None,
                last_cycle_stats={},
                last_error="",
            )
            try:
                entries, n_scanned = await _run_cycle(session, positions, last_exit_bar)
                cycle_finished_at = _status_now()
                _write_status(
                    started_at=started_at,
                    poll_sec=poll_sec,
                    cycle_running=False,
                    n_open_positions=len(positions),
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_finished_at=cycle_finished_at,
                    last_cycle_stats={
                        "n_scanned": int(n_scanned),
                        "n_entries": int(len(entries)),
                        "n_open_positions": int(len(positions)),
                        "bull": bool(getattr(config, "_bull_day_active", False)),
                    },
                    last_error="",
                )
            except Exception as exc:
                cycle_finished_at = _status_now()
                log.exception("market agent cycle error: %s", exc)
                _write_status(
                    started_at=started_at,
                    poll_sec=poll_sec,
                    cycle_running=False,
                    n_open_positions=len(positions),
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_finished_at=cycle_finished_at,
                    last_cycle_stats={},
                    last_error=str(exc),
                )
                raise
            if cycle_started_at == started_at:
                log.info("first cycle done: entries=%d open_positions=%d", len(entries), len(positions))
            if args.once:
                return
            await asyncio.sleep(poll_sec)


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
