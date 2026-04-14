from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

from ml_signal_model import build_runtime_record


ROOT = Path(__file__).resolve().parent
CRITIC_FILE = ROOT / "critic_dataset.jsonl"
SEQ_FEATURE_NAMES = [
    "close_norm",
    "high_norm",
    "low_norm",
    "open_norm",
    "vol_x",
    "slope",
    "adx",
    "rsi",
    "macd_hist_norm",
    "atr_pct",
]
_FILE_LOCK = threading.RLock()
_pylog = logging.getLogger("critic_dataset")
_logged_candidates: OrderedDict[str, bool] = OrderedDict()
_MAX_LOGGED = 100_000
_CROSS_PROCESS_LOCK_TIMEOUT_SEC = 10.0
_CROSS_PROCESS_LOCK_POLL_SEC = 0.05
_REPLACE_RETRIES = 12
_REPLACE_RETRY_SEC = 0.10


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe(v: object) -> float:
    try:
        f = float(v)
        return 0.0 if f != f else f
    except Exception:
        return 0.0


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _safe_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _candidate_id(sym: str, tf: str, bar_ts: int) -> str:
    ts_str = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{sym}_{tf}_{ts_str.replace(':', '').replace('-', '')}"


def _parse_utc_iso(raw: object) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _mark_logged(record_id: str) -> None:
    if record_id in _logged_candidates:
        return
    if len(_logged_candidates) >= _MAX_LOGGED:
        evict = max(1, _MAX_LOGGED // 10)
        for _ in range(evict):
            _logged_candidates.popitem(last=False)
    _logged_candidates[record_id] = True


def _is_logged(record_id: str) -> bool:
    return record_id in _logged_candidates


def _lock_file_path() -> Path:
    return CRITIC_FILE.with_name(CRITIC_FILE.name + ".lock")


@contextmanager
def _dataset_io_lock():
    with _FILE_LOCK:
        if msvcrt is None:
            yield
            return

        lock_path = _lock_file_path()
        lock_handle = None
        start = time.monotonic()
        while True:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_handle = lock_path.open("a+b")
                lock_handle.seek(0, os.SEEK_END)
                if lock_handle.tell() == 0:
                    lock_handle.write(b"0")
                    lock_handle.flush()
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if lock_handle is not None:
                    try:
                        lock_handle.close()
                    except Exception:
                        pass
                    lock_handle = None
                if time.monotonic() - start >= _CROSS_PROCESS_LOCK_TIMEOUT_SEC:
                    raise TimeoutError(f"timeout acquiring critic_dataset lock: {lock_path}")
                time.sleep(_CROSS_PROCESS_LOCK_POLL_SEC)

        try:
            yield
        finally:
            try:
                if lock_handle is not None:
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                if lock_handle is not None:
                    lock_handle.close()


def _atomic_replace_with_retry(tmp: Path, target: Path) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(1, _REPLACE_RETRIES + 1):
        try:
            tmp.replace(target)
            return
        except PermissionError as e:
            last_error = e
            if attempt >= _REPLACE_RETRIES:
                raise
            time.sleep(_REPLACE_RETRY_SEC * attempt)
    if last_error is not None:
        raise last_error


def _collect_mutated_lines(mutator):
    updated: List[str] = []
    changed = False
    had_bad_rows = False
    for line in CRITIC_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            had_bad_rows = True
            continue
        if not isinstance(rec, dict):
            had_bad_rows = True
            continue
        rec_changed = bool(mutator(rec))
        changed = changed or rec_changed
        updated.append(json.dumps(rec, ensure_ascii=False, cls=_Enc))
    return updated, changed, had_bad_rows


def _append(record: Dict[str, Any]) -> None:
    with _dataset_io_lock():
        CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CRITIC_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, cls=_Enc) + "\n")


def get_record(record_id: str) -> Optional[Dict[str, Any]]:
    if not record_id or not CRITIC_FILE.exists():
        return None
    try:
        with _dataset_io_lock():
            for line in CRITIC_FILE.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("id") == record_id:
                    return rec
    except Exception as e:
        _pylog.warning("critic_dataset get_record error for %s: %s", record_id, e)
    return None


def _rewrite_records(mutator) -> None:
    if not CRITIC_FILE.exists():
        return
    try:
        _, maybe_changed, maybe_bad_rows = _collect_mutated_lines(mutator)
        if not (maybe_changed or maybe_bad_rows):
            return

        with _dataset_io_lock():
            updated, changed, had_bad_rows = _collect_mutated_lines(mutator)
            if not (changed or had_bad_rows):
                return

            CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CRITIC_FILE.with_name(
                f"{CRITIC_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp.write_text("\n".join(updated) + "\n", encoding="utf-8")
            _atomic_replace_with_retry(tmp, CRITIC_FILE)
    except Exception as e:
        _pylog.warning("critic_dataset rewrite error: %s", e)


def _decision_priority(action: str, stage: str) -> int:
    a = str(action or "").lower()
    s = str(stage or "").lower()
    if a == "take":
        return 30
    if a == "blocked":
        return 20
    if a in {"candidate", "snapshot"} or s == "collector":
        return 10
    return 0


def _update_existing_candidate(
    *,
    record_id: str,
    action: str,
    reason_code: str,
    reason: str,
    stage: str,
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
    signal_flags: Optional[Dict[str, bool]],
    near_miss: bool,
) -> bool:
    changed = False

    def _mutate(rec: Dict[str, Any]) -> bool:
        nonlocal changed
        if rec.get("id") != record_id:
            return False
        rec.setdefault("decision", {})
        old_action = str(rec["decision"].get("action", ""))
        old_stage = str(rec["decision"].get("stage", ""))
        if _decision_priority(action, stage) < _decision_priority(old_action, old_stage):
            return False

        rec["decision"] = {
            "action": action,
            "reason_code": reason_code,
            "reason": reason,
            "stage": stage,
            "candidate_score": round(_safe(candidate_score), 4),
            "base_score": round(_safe(base_score), 4),
            "score_floor": round(_safe(score_floor), 4),
            "forecast_return_pct": round(_safe(forecast_return_pct), 4),
            "today_change_pct": round(_safe(today_change_pct), 4),
            "ml_proba": None if ml_proba is None else round(_safe(ml_proba), 6),
            "mtf_soft_penalty": round(_safe(mtf_soft_penalty), 4),
            "fresh_priority": bool(fresh_priority),
            "catchup": bool(catchup),
            "continuation_profile": bool(continuation_profile),
            "near_miss": bool(near_miss),
            "signal_flags": signal_flags or {},
        }
        rec.setdefault("labels", {})
        if action == "take":
            rec["labels"]["trade_taken"] = True
        changed = True
        return True

    _rewrite_records(_mutate)
    return changed


def log_candidate(
    *,
    sym: str,
    tf: str,
    bar_ts: int,
    signal_type: str,
    is_bull_day: bool,
    feat: Dict[str, Any],
    i: int,
    data: Any,
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
    btc_vs_ema50: float = 0.0,
    btc_momentum_4h: float = 0.0,
    market_vol_24h: float = 0.0,
) -> str:
    record_id = _candidate_id(sym, tf, bar_ts)
    if _is_logged(record_id):
        _update_existing_candidate(
            record_id=record_id,
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
        )
        return record_id

    rec = build_runtime_record(
        sym=sym,
        tf=tf,
        signal_type=signal_type,
        is_bull_day=is_bull_day,
        bar_ts=bar_ts,
        feat=feat,
        data=data,
        i=i,
        btc_vs_ema50=btc_vs_ema50,
        btc_momentum_4h=btc_momentum_4h,
        market_vol_24h=market_vol_24h,
    )
    rec_id = record_id
    rec["id"] = rec_id
    rec["ts_signal"] = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["bar_ts"] = bar_ts
    rec["seq_feature_names"] = SEQ_FEATURE_NAMES
    rec["decision"] = {
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "stage": stage,
        "candidate_score": round(_safe(candidate_score), 4),
        "base_score": round(_safe(base_score), 4),
        "score_floor": round(_safe(score_floor), 4),
        "forecast_return_pct": round(_safe(forecast_return_pct), 4),
        "today_change_pct": round(_safe(today_change_pct), 4),
        "ml_proba": None if ml_proba is None else round(_safe(ml_proba), 6),
        "mtf_soft_penalty": round(_safe(mtf_soft_penalty), 4),
        "fresh_priority": bool(fresh_priority),
        "catchup": bool(catchup),
        "continuation_profile": bool(continuation_profile),
        "near_miss": bool(near_miss),
        "signal_flags": signal_flags or {},
    }
    rec["labels"] = {
        "ret_3": None,
        "ret_5": None,
        "ret_10": None,
        "label_3": None,
        "label_5": None,
        "label_10": None,
        "trade_taken": action == "take",
        "trade_exit_pnl": None,
        "trade_exit_reason": None,
        "trade_bars_held": None,
        "linked_ml_record_id": "",
    }
    try:
        _append(rec)
        _mark_logged(record_id)
    except Exception as e:
        _pylog.warning("critic_dataset write error: %s", e)
    return rec_id


def mark_trade_taken(record_id: str, linked_ml_record_id: str = "") -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        changed = False
        if rec["labels"].get("trade_taken") is not True:
            rec["labels"]["trade_taken"] = True
            changed = True
        if linked_ml_record_id and rec["labels"].get("linked_ml_record_id") != linked_ml_record_id:
            rec["labels"]["linked_ml_record_id"] = linked_ml_record_id
            changed = True
        if not linked_ml_record_id and not changed:
            return False
        return changed

    _rewrite_records(_mutate)


def fill_trade_outcome(record_id: str, exit_pnl: float, exit_reason: str, bars_held: int) -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        new_exit_pnl = round(_safe(exit_pnl), 4)
        new_exit_reason = exit_reason
        new_bars_held = int(bars_held)
        if (
            rec["labels"].get("trade_exit_pnl") == new_exit_pnl
            and rec["labels"].get("trade_exit_reason") == new_exit_reason
            and rec["labels"].get("trade_bars_held") == new_bars_held
        ):
            return False
        rec["labels"]["trade_exit_pnl"] = new_exit_pnl
        rec["labels"]["trade_exit_reason"] = new_exit_reason
        rec["labels"]["trade_bars_held"] = new_bars_held
        return True

    _rewrite_records(_mutate)


def fill_forward_label(record_id: str, horizon: int, ret_pct: float) -> None:
    if not CRITIC_FILE.exists():
        return
    key_ret = f"ret_{horizon}"
    key_label = f"label_{horizon}"

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("id") != record_id:
            return False
        rec.setdefault("labels", {})
        new_ret = round(_safe(ret_pct), 4)
        new_label = _safe(ret_pct) > 0
        if rec["labels"].get(key_ret) == new_ret and rec["labels"].get(key_label) == new_label:
            return False
        rec["labels"][key_ret] = new_ret
        rec["labels"][key_label] = new_label
        return True

    _rewrite_records(_mutate)


def fill_pending_from_data(sym: str, tf: str, t_arr: Any, c_arr: Any, bar_ms: int) -> None:
    if not CRITIC_FILE.exists():
        return

    def _mutate(rec: Dict[str, Any]) -> bool:
        if rec.get("sym") != sym or rec.get("tf") != tf:
            return False
        lab = rec.get("labels", {})
        rec_bar_ts = rec.get("bar_ts", 0)
        idx_arr = np.where(t_arr == rec_bar_ts)[0]
        if len(idx_arr) == 0:
            return False
        entry_close = float(c_arr[idx_arr[0]])
        if entry_close <= 0:
            return False
        changed = False
        for h in (3, 5, 10):
            key_ret = f"ret_{h}"
            key_label = f"label_{h}"
            if lab.get(key_ret) is not None:
                continue
            future_ts = rec_bar_ts + h * bar_ms
            fut_idx = np.where(t_arr >= future_ts)[0]
            if len(fut_idx) == 0:
                continue
            future_close = float(c_arr[fut_idx[0]])
            ret_pct = (future_close / entry_close - 1) * 100
            rec["labels"][key_ret] = round(ret_pct, 4)
            rec["labels"][key_label] = ret_pct > 0
            changed = True
        return changed

    _rewrite_records(_mutate)


def _teacher_local_window(target_day: date, phase: str, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, dt_time.min, tzinfo=tz)
    if phase == "midday":
        end_local = datetime.combine(target_day, dt_time(hour=12), tzinfo=tz)
    else:
        end_local = datetime.combine(target_day, dt_time.max, tzinfo=tz)
    return start_local, end_local


def _record_local_dt(rec: Dict[str, Any], tz: ZoneInfo) -> Optional[datetime]:
    ts = _parse_utc_iso(rec.get("ts_signal"))
    if ts is None:
        bar_ts = _safe_int(rec.get("bar_ts"))
        if bar_ts is None:
            return None
        ts = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
    return ts.astimezone(tz)


def _phase_teacher_payload(
    *,
    sym: str,
    phase: str,
    target_day_local: str,
    timezone_name: str,
    early_capture_ratio_min: float,
    exchange_summary: Optional[Dict[str, Any]],
    exchange_rank: Optional[int],
    watchlist_summary: Optional[Dict[str, Any]],
    watchlist_rank: Optional[int],
    false_positive_buy: bool,
) -> Dict[str, Any]:
    chosen = watchlist_summary or exchange_summary or {}
    capture_ratio = chosen.get("capture_ratio")
    capture_ratio_f = None if capture_ratio is None else round(_safe(capture_ratio), 4)
    status = None
    reason = None
    if watchlist_summary:
        status = watchlist_summary.get("status")
        reason = watchlist_summary.get("reason")
    elif exchange_summary:
        status = exchange_summary.get("status")
        reason = exchange_summary.get("reason")

    return {
        "phase": phase,
        "target_day_local": target_day_local,
        "timezone": timezone_name,
        "exchange_top_gainer": exchange_summary is not None,
        "exchange_top_rank": exchange_rank,
        "watchlist_top_gainer": watchlist_summary is not None,
        "watchlist_top_rank": watchlist_rank,
        "status": status,
        "reason": reason,
        "capture_ratio": capture_ratio_f,
        "early_capture": bool(
            watchlist_summary is not None
            and capture_ratio_f is not None
            and capture_ratio_f >= float(early_capture_ratio_min)
            and str(status or "") == "bought"
        ),
        "bot_false_positive_buy": bool(false_positive_buy),
        "day_change_pct": None if chosen.get("day_change_pct") is None else round(_safe(chosen.get("day_change_pct")), 4),
        "quote_volume_24h": None if chosen.get("quote_volume_24h") is None else round(_safe(chosen.get("quote_volume_24h")), 2),
        "entries_count": _safe_int(chosen.get("entries_count")),
        "blocked_count": _safe_int(chosen.get("blocked_count")),
        "first_entry_time": chosen.get("first_entry_time"),
        "first_entry_mode": chosen.get("first_entry_mode"),
        "first_entry_price": None if chosen.get("first_entry_price") is None else round(_safe(chosen.get("first_entry_price")), 8),
        "opportunity_from_entry_pct": None if chosen.get("opportunity_from_entry_pct") is None else round(_safe(chosen.get("opportunity_from_entry_pct")), 4),
        "latest_exit_time": chosen.get("latest_exit_time"),
        "latest_exit_pnl_pct": None if chosen.get("latest_exit_pnl_pct") is None else round(_safe(chosen.get("latest_exit_pnl_pct")), 4),
    }


def annotate_top_gainer_teacher(report: Dict[str, Any]) -> Dict[str, Any]:
    if not CRITIC_FILE.exists():
        return {"rows_scanned": 0, "rows_annotated": 0, "symbols_tagged": 0, "phase": str(report.get("phase", ""))}

    phase = str(report.get("phase", "") or "").strip().lower()
    if phase not in {"midday", "final"}:
        raise ValueError(f"Unsupported top-gainer phase: {phase!r}")

    target_day_text = str(report.get("target_day_local", "") or "").strip()
    if not target_day_text:
        raise ValueError("Top-gainer report missing target_day_local")
    target_day = date.fromisoformat(target_day_text)

    settings = report.get("settings") or {}
    timezone_name = str(settings.get("timezone", "Europe/Budapest") or "Europe/Budapest")
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _teacher_local_window(target_day, phase, tz)
    early_capture_ratio_min = float(settings.get("early_capture_ratio_min", 0.35) or 0.35)

    exchange_map: Dict[str, Dict[str, Any]] = {}
    exchange_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("exchange_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        exchange_map[sym] = item
        exchange_rank_map[sym] = idx

    watchlist_map: Dict[str, Dict[str, Any]] = {}
    watchlist_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("watchlist_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        watchlist_map[sym] = item
        watchlist_rank_map[sym] = idx

    false_positive_symbols = {
        str(sym).strip()
        for sym in (report.get("bot_false_positive_symbols") or [])
        if str(sym).strip()
    }
    tagged_symbols = set(exchange_map) | set(watchlist_map) | false_positive_symbols
    rows_scanned = 0

    def _annotate(rec: Dict[str, Any], *, count_scan: bool) -> bool:
        nonlocal rows_scanned
        if count_scan:
            rows_scanned += 1
        local_dt = _record_local_dt(rec, tz)
        if local_dt is None or not (start_local <= local_dt <= end_local):
            return False

        sym = str(rec.get("sym", "")).strip()
        phase_payload = _phase_teacher_payload(
            sym=sym,
            phase=phase,
            target_day_local=target_day_text,
            timezone_name=timezone_name,
            early_capture_ratio_min=early_capture_ratio_min,
            exchange_summary=exchange_map.get(sym),
            exchange_rank=exchange_rank_map.get(sym),
            watchlist_summary=watchlist_map.get(sym),
            watchlist_rank=watchlist_rank_map.get(sym),
            false_positive_buy=sym in false_positive_symbols,
        )
        teacher = rec.setdefault("teacher", {})
        if teacher.get(phase) == phase_payload:
            return False
        teacher[phase] = phase_payload
        return True

    def _mutate_preview(rec: Dict[str, Any]) -> bool:
        nonlocal rows_scanned
        rows_scanned += 1
        return _annotate(rec, count_scan=False)

    _, maybe_changed, maybe_bad_rows = _collect_mutated_lines(_mutate_preview)
    rows_annotated = 0
    if maybe_changed or maybe_bad_rows:
        def _mutate_commit(rec: Dict[str, Any]) -> bool:
            nonlocal rows_annotated
            changed = _annotate(rec, count_scan=False)
            if changed:
                rows_annotated += 1
            return changed

        try:
            with _dataset_io_lock():
                updated, changed, had_bad_rows = _collect_mutated_lines(_mutate_commit)
                if changed or had_bad_rows:
                    CRITIC_FILE.parent.mkdir(parents=True, exist_ok=True)
                    tmp = CRITIC_FILE.with_name(
                        f"{CRITIC_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                    )
                    tmp.write_text("\n".join(updated) + "\n", encoding="utf-8")
                    _atomic_replace_with_retry(tmp, CRITIC_FILE)
        except Exception as e:
            _pylog.warning("critic_dataset rewrite error: %s", e)
            rows_annotated = 0
    return {
        "phase": phase,
        "target_day_local": target_day_text,
        "rows_scanned": rows_scanned,
        "rows_annotated": rows_annotated,
        "symbols_tagged": len(tagged_symbols),
        "timezone": timezone_name,
    }
