from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import aiohttp

import critic_dataset
import config
import data_collector
import ml_candidate_ranker
import ml_dataset
import report_candidate_ranker_shadow
import report_critic_dataset
import top_gainer_critic
from ml_signal_model import save_json


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
STATUS_FILE = RUNTIME_DIR / "rl_worker_status.json"
LOG_FILE = RUNTIME_DIR / "rl_worker_runtime.log"
REPORT_DIR = RUNTIME_DIR / "reports"
CHAT_IDS_FILE = ROOT / ".chat_ids"
TG_DEDUP_FILE = RUNTIME_DIR / "tg_send_dedup.json"

MODEL_FILE = ROOT / "ml_candidate_ranker.json"
TRAIN_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"
SHADOW_REPORT_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"

DEFAULT_TRAIN_INTERVAL_SEC = 60 * 60
DEFAULT_STATUS_INTERVAL_SEC = 5 * 60
DEFAULT_MIN_ROWS = 500
DEFAULT_MIN_NEW_ROWS = 50
DEFAULT_TOP_GAINER_CHECK_SEC = 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_claim_report_send(category: str, cooldown_minutes: int) -> bool:
    """Atomically claim the right to send a report of this category.

    Writes the timestamp BEFORE the caller sends, so concurrent instances
    racing on the same file all see the record and only the first one wins.
    Returns True if this instance should send (claim succeeded).
    """
    log = logging.getLogger("rl_headless_worker")
    lock_path = RUNTIME_DIR / "tg_send_dedup.lock"
    dedup_path = TG_DEDUP_FILE
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        # Acquire a simple exclusive lock via O_CREAT|O_EXCL (atomic on NTFS).
        # Retry briefly in case another process holds the lock right now.
        deadline = time.monotonic() + 3.0
        fd = None
        while fd is None:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                # Another process holds the lock; wait a bit and retry.
                if time.monotonic() > deadline:
                    # Give up on locking but still do the cooldown check.
                    break
                time.sleep(0.05)
        try:
            # Read current dedup state.
            data: dict = {}
            if dedup_path.exists():
                try:
                    data = json.loads(dedup_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            last_sent_iso = data.get(category)
            if last_sent_iso:
                last_sent = datetime.fromisoformat(last_sent_iso.replace("Z", "+00:00"))
                elapsed = (_utc_now() - last_sent).total_seconds() / 60
                if elapsed < cooldown_minutes:
                    log.info(
                        "TG dedup: skipping %s (sent %.1f min ago, cooldown=%d min)",
                        category, elapsed, cooldown_minutes,
                    )
                    return False
            # Claim: write timestamp NOW, before sending.
            data[category] = _utc_now_iso()
            dedup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        return True  # On unexpected error allow sending.


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows += 1
    return rows


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def should_train(
    *,
    rows_total: int,
    min_rows: int,
    last_trained_rows: int,
    min_new_rows: int,
    dataset_mtime: float,
    last_dataset_mtime: float,
    force_first_train: bool = False,
) -> bool:
    if rows_total < min_rows:
        return False
    if force_first_train and last_trained_rows <= 0:
        return True
    if rows_total >= last_trained_rows + min_new_rows:
        return True
    if dataset_mtime > last_dataset_mtime and last_trained_rows <= 0:
        return True
    return False


def build_status_snapshot(
    state: "WorkerState",
    *,
    critic_report: Dict[str, Any],
    ml_rows_total: int,
) -> Dict[str, Any]:
    return {
        "worker": {
            "started_at": state.started_at,
            "last_heartbeat": _utc_now_iso(),
            "mode": "headless_rl",
        },
        "collector": {
            "enabled": state.collector_enabled,
            "running": state.collector_enabled,
            "last_cycle_started_at": state.collector_last_cycle_started_at,
            "last_cycle_finished_at": state.collector_last_cycle_finished_at,
            "last_cycle_stats": state.collector_last_cycle_stats,
            "last_error": state.collector_last_error,
        },
        "training": {
            "interval_sec": state.train_interval_sec,
            "min_rows": state.min_rows,
            "min_new_rows": state.min_new_rows,
            "runs_total": state.train_runs_total,
            "runs_ok": state.train_runs_ok,
            "runs_failed": state.train_runs_failed,
            "last_started_at": state.train_last_started_at,
            "last_finished_at": state.train_last_finished_at,
            "last_error": state.train_last_error,
            "last_rows_total": state.last_trained_rows,
            "last_dataset_mtime": state.last_trained_dataset_mtime,
            "last_model_name": state.last_model_name,
            "last_top1_delta": state.last_top1_delta,
            "model_file": str(MODEL_FILE),
            "report_file": str(TRAIN_REPORT_FILE),
            "shadow_file": str(SHADOW_REPORT_FILE),
        },
        "datasets": {
            "ml_dataset_rows": ml_rows_total,
            "critic_dataset": critic_report,
        },
        "top_gainer_critic": {
            "enabled": state.top_gainer_enabled,
            "runs_total": state.top_gainer_runs_total,
            "runs_ok": state.top_gainer_runs_ok,
            "runs_failed": state.top_gainer_runs_failed,
            "last_slot_key": state.top_gainer_last_slot_key,
            "last_started_at": state.top_gainer_last_started_at,
            "last_finished_at": state.top_gainer_last_finished_at,
            "last_error": state.top_gainer_last_error,
            "last_phase": state.top_gainer_last_phase,
            "last_target_day_local": state.top_gainer_last_target_day_local,
            "last_report_json": state.top_gainer_last_report_json,
            "last_report_txt": state.top_gainer_last_report_txt,
            "last_watchlist_top_capture_rate_pct": state.top_gainer_last_capture_rate_pct,
            "last_watchlist_top_early_capture_rate_pct": state.top_gainer_last_early_capture_rate_pct,
        },
        "latest_training_report": {
            "json": state.latest_training_report_json,
            "txt": state.latest_training_report_txt,
            "latest_json": state.latest_training_latest_json,
            "latest_txt": state.latest_training_latest_txt,
        },
    }


def _write_status(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_FILE)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_known_chat_ids() -> list[int]:
    if not CHAT_IDS_FILE.exists():
        return []
    try:
        payload = json.loads(CHAT_IDS_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    out: list[int] = []
    for item in payload:
        try:
            out.append(int(item))
        except Exception:
            continue
    return sorted(set(out))


async def _send_telegram_text(text: str) -> None:
    token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    if not token:
        return
    if not bool(getattr(config, "RL_TELEGRAM_REPORTS_ENABLED", True)):
        return
    chat_ids = _load_known_chat_ids()
    if not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for chat_id in chat_ids:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    await resp.read()
            except Exception as exc:
                logging.getLogger("rl_headless_worker.notify").warning(
                    "Telegram report send failed for %s: %s",
                    chat_id,
                    exc,
                )


def _train_ranker_once(min_rows: int) -> Dict[str, Any]:
    report = ml_candidate_ranker.train_and_evaluate(
        critic_dataset.CRITIC_FILE,
        min_rows=min_rows,
        require_catboost=getattr(config, "ML_CANDIDATE_RANKER_REQUIRE_CATBOOST", True),
    )
    model_payload = ml_candidate_ranker.build_live_model_payload(report)
    report_without_payload = {k: v for k, v in report.items() if k != "model_payload"}
    save_json(MODEL_FILE, model_payload)
    save_json(TRAIN_REPORT_FILE, report_without_payload)

    shadow_report = report_candidate_ranker_shadow.build_shadow_report(
        critic_dataset.CRITIC_FILE,
        model_payload,
        top_ns=(1, 3, 5),
    )
    save_json(SHADOW_REPORT_FILE, shadow_report)

    top1_delta = None
    if shadow_report.get("top_n"):
        top1_delta = (
            shadow_report["top_n"][0]
            .get("delta", {})
            .get("avg_target_return")
        )

    return {
        "model_name": report_without_payload.get("chosen_model", ""),
        "rows_total": int(report_without_payload.get("rows_total", 0)),
        "dataset_mtime": _file_mtime(critic_dataset.CRITIC_FILE),
        "top1_delta": top1_delta,
        "train_report": report_without_payload,
        "shadow_report": shadow_report,
    }


def _build_train_session_report(
    *,
    state: "WorkerState",
    result: Dict[str, Any],
    critic_rows_total: int,
    ml_rows_total: int,
) -> Dict[str, Any]:
    return {
        "generated_at_utc": _utc_now_iso(),
        "worker_started_at": state.started_at,
        "training_run_index": state.train_runs_ok,
        "model_name": result.get("model_name", ""),
        "rows_total": int(result.get("rows_total", 0)),
        "top1_delta": result.get("top1_delta"),
        "critic_rows_total": critic_rows_total,
        "ml_rows_total": ml_rows_total,
        "collector_last_cycle_stats": state.collector_last_cycle_stats,
        "top_gainer_critic": {
            "last_phase": state.top_gainer_last_phase,
            "last_target_day_local": state.top_gainer_last_target_day_local,
            "last_capture_rate_pct": state.top_gainer_last_capture_rate_pct,
            "last_early_capture_rate_pct": state.top_gainer_last_early_capture_rate_pct,
            "last_report_json": state.top_gainer_last_report_json,
            "last_report_txt": state.top_gainer_last_report_txt,
        },
        "train_report": result.get("train_report", {}),
        "shadow_report": result.get("shadow_report", {}),
    }


def _render_train_session_text(report: Dict[str, Any]) -> str:
    lines = [
        f"RL training session {report.get('training_run_index')} at {report.get('generated_at_utc')}",
        f"model: {report.get('model_name')}",
        f"rows_total: {report.get('rows_total')}",
        f"top1_delta: {report.get('top1_delta')}",
        f"critic_rows_total: {report.get('critic_rows_total')}",
        f"ml_rows_total: {report.get('ml_rows_total')}",
    ]
    collector = report.get("collector_last_cycle_stats") or {}
    if collector:
        lines.append(
            "collector: "
            + f"ok={collector.get('ok')} total={collector.get('total')} bull={collector.get('bull')}"
        )
    tg = report.get("top_gainer_critic") or {}
    if tg.get("last_phase"):
        lines.append(
            "top_gainer_critic: "
            + f"{tg.get('last_phase')} {tg.get('last_target_day_local')} "
            + f"capture={tg.get('last_capture_rate_pct')}% early={tg.get('last_early_capture_rate_pct')}%"
        )
    tr = report.get("train_report") or {}
    improvement = tr.get("improvement_delta", {}) if isinstance(tr, dict) else {}
    if improvement:
        lines.append(
            "improvement: "
            + f"ret5_delta={improvement.get('ret5_avg_delta')} "
            + f"target_return_delta={improvement.get('target_return_delta')} "
            + f"win_rate_delta={improvement.get('win_rate_delta')}"
        )
    shadow = report.get("shadow_report") or {}
    top_n = shadow.get("top_n") if isinstance(shadow, dict) else None
    if top_n:
        best = top_n[0]
        delta = best.get("delta", {})
        lines.append(
            "shadow_top1: "
            + f"avg_target_return_delta={delta.get('avg_target_return')} "
            + f"ret5_avg_delta={delta.get('ret5_avg')}"
        )
    return "\n".join(lines)


def _render_train_session_telegram(report: Dict[str, Any]) -> str:
    tg = report.get("top_gainer_critic") or {}
    improvement = (report.get("train_report") or {}).get("improvement_delta", {})
    lines = [
        "RL train complete",
        f"time: {report.get('generated_at_utc')}",
        f"model: {report.get('model_name')}",
        f"rows: {report.get('rows_total')}",
        f"top1_delta: {report.get('top1_delta')}",
    ]
    if improvement:
        lines.append(
            "improvement: "
            f"target={improvement.get('target_return_delta')} "
            f"ret5={improvement.get('ret5_avg_delta')} "
            f"win={improvement.get('win_rate_delta')}"
        )
    if tg.get("last_phase"):
        lines.append(
            "critic: "
            f"{tg.get('last_phase')} {tg.get('last_target_day_local')} "
            f"capture={tg.get('last_capture_rate_pct')}% "
            f"early={tg.get('last_early_capture_rate_pct')}%"
        )
    return "\n".join(lines)


def _render_top_gainer_telegram(result: Dict[str, Any]) -> str:
    report = result or {}
    summary = report.get("summary") or {}
    phase = report.get("phase") or ""
    day = report.get("target_day_local") or ""
    top = report.get("watchlist_top_gainers") or []
    bought = [item for item in top if item.get("status") == "bought"][:3]
    missed = [item for item in top if item.get("status") in {"no_signal", "blocked_rule", "blocked_portfolio"}][:3]
    false_pos = (report.get("bot_false_positive_symbols") or [])[:5]

    lines = [
        f"Top gainer critic {phase}",
        f"day: {day}",
        f"watchlist top bought: {summary.get('watchlist_top_bought')}/{summary.get('watchlist_top_count')} ({summary.get('watchlist_top_capture_rate_pct')}%)",
        f"early captures: {summary.get('watchlist_top_early_captured')}/{summary.get('watchlist_top_count')} ({summary.get('watchlist_top_early_capture_rate_pct')}%)",
        f"false-positive buys: {summary.get('bot_false_positive_buys')}/{summary.get('bot_unique_buys')}",
    ]
    if bought:
        bought_line = ", ".join(
            f"{item.get('symbol')} {item.get('first_entry_mode') or ''}".strip()
            for item in bought
        )
        lines.append(f"bought: {bought_line}")
    if missed:
        missed_line = ", ".join(
            f"{item.get('symbol')} {item.get('status')}"
            for item in missed
        )
        lines.append(f"missed: {missed_line}")
    if false_pos:
        lines.append("false positives: " + ", ".join(str(x) for x in false_pos))
    return "\n".join(lines)


def _save_train_session_report(report: Dict[str, Any]) -> Dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"rl_train_{stamp}"
    json_path = REPORT_DIR / f"{base}.json"
    txt_path = REPORT_DIR / f"{base}.txt"
    latest_json = REPORT_DIR / "rl_train_latest.json"
    latest_txt = REPORT_DIR / "rl_train_latest.txt"
    json_payload = json.dumps(report, ensure_ascii=False, indent=2)
    text_payload = _render_train_session_text(report)
    _write_text(json_path, json_payload)
    _write_text(txt_path, text_payload)
    _write_text(latest_json, json_payload)
    _write_text(latest_txt, text_payload)
    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "latest_json": str(latest_json),
        "latest_txt": str(latest_txt),
    }


@dataclass
class WorkerState:
    train_interval_sec: int
    status_interval_sec: int
    min_rows: int
    min_new_rows: int
    collector_enabled: bool
    started_at: str = field(default_factory=_utc_now_iso)
    collector_last_cycle_started_at: Optional[str] = None
    collector_last_cycle_finished_at: Optional[str] = None
    collector_last_cycle_stats: Dict[str, Any] = field(default_factory=dict)
    collector_last_error: str = ""
    train_runs_total: int = 0
    train_runs_ok: int = 0
    train_runs_failed: int = 0
    train_last_started_at: Optional[str] = None
    train_last_finished_at: Optional[str] = None
    train_last_error: str = ""
    last_trained_rows: int = 0
    last_trained_dataset_mtime: float = 0.0
    last_model_name: str = ""
    last_top1_delta: Optional[float] = None
    top_gainer_enabled: bool = bool(getattr(config, "TOP_GAINER_CRITIC_ENABLED", True))
    top_gainer_runs_total: int = 0
    top_gainer_runs_ok: int = 0
    top_gainer_runs_failed: int = 0
    top_gainer_last_slot_key: str = ""
    top_gainer_last_started_at: Optional[str] = None
    top_gainer_last_finished_at: Optional[str] = None
    top_gainer_last_error: str = ""
    top_gainer_last_phase: str = ""
    top_gainer_last_target_day_local: str = ""
    top_gainer_last_report_json: str = ""
    top_gainer_last_report_txt: str = ""
    top_gainer_last_capture_rate_pct: Optional[float] = None
    top_gainer_last_early_capture_rate_pct: Optional[float] = None
    latest_training_report_json: str = ""
    latest_training_report_txt: str = ""
    latest_training_latest_json: str = ""
    latest_training_latest_txt: str = ""


async def _collector_supervisor(state: WorkerState) -> None:
    log = logging.getLogger("rl_headless_worker.collector")
    while True:
        state.collector_last_cycle_started_at = _utc_now_iso()
        try:
            btc_ctx = await data_collector._get_btc_context()
            stats = await data_collector._collect_once(btc_ctx)
            stats["bull"] = bool(btc_ctx.get("is_bull", False))
            state.collector_last_cycle_stats = stats
            state.collector_last_cycle_finished_at = _utc_now_iso()
            state.collector_last_error = ""
            log.info(
                "Collector cycle: %s/%s ok, bull=%s",
                stats.get("ok"),
                stats.get("total"),
                stats.get("bull"),
            )
            await _write_status_now(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.collector_last_error = str(exc)
            log.exception("Collector cycle failed: %s", exc)
            await _write_status_now(state)
        wait = data_collector._seconds_until_next_bar()
        await asyncio.sleep(wait)


async def _training_loop(state: WorkerState) -> None:
    log = logging.getLogger("rl_headless_worker.training")
    while True:
        await asyncio.sleep(5.0)
        rows_total = _count_jsonl_rows(critic_dataset.CRITIC_FILE)
        dataset_mtime = _file_mtime(critic_dataset.CRITIC_FILE)
        if not should_train(
            rows_total=rows_total,
            min_rows=state.min_rows,
            last_trained_rows=state.last_trained_rows,
            min_new_rows=state.min_new_rows,
            dataset_mtime=dataset_mtime,
            last_dataset_mtime=state.last_trained_dataset_mtime,
            force_first_train=True,
        ):
            await asyncio.sleep(state.train_interval_sec)
            continue

        state.train_runs_total += 1
        state.train_last_started_at = _utc_now_iso()
        state.train_last_error = ""
        try:
            result = await asyncio.to_thread(_train_ranker_once, state.min_rows)
            state.train_runs_ok += 1
            state.train_last_finished_at = _utc_now_iso()
            state.last_trained_rows = int(result["rows_total"])
            state.last_trained_dataset_mtime = float(result["dataset_mtime"])
            state.last_model_name = str(result["model_name"])
            state.last_top1_delta = result["top1_delta"]
            critic_rows_total = await asyncio.to_thread(_count_jsonl_rows, critic_dataset.CRITIC_FILE)
            ml_rows_total = await asyncio.to_thread(_count_jsonl_rows, ml_dataset.ML_FILE)
            session_report = _build_train_session_report(
                state=state,
                result=result,
                critic_rows_total=critic_rows_total,
                ml_rows_total=ml_rows_total,
            )
            report_paths = await asyncio.to_thread(_save_train_session_report, session_report)
            state.latest_training_report_json = str(report_paths.get("json", ""))
            state.latest_training_report_txt = str(report_paths.get("txt", ""))
            state.latest_training_latest_json = str(report_paths.get("latest_json", ""))
            state.latest_training_latest_txt = str(report_paths.get("latest_txt", ""))
            log.info(
                "Ranker trained: model=%s rows=%s top1_delta=%s",
                state.last_model_name,
                state.last_trained_rows,
                state.last_top1_delta,
            )
            if bool(getattr(config, "RL_TRAIN_TELEGRAM_REPORTS_ENABLED", True)):
                if _try_claim_report_send("train_session", cooldown_minutes=30):
                    await _send_telegram_text(_render_train_session_telegram(session_report))
            await _write_status_now(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.train_runs_failed += 1
            state.train_last_finished_at = _utc_now_iso()
            state.train_last_error = str(exc)
            log.exception("Ranker training failed: %s", exc)
            await _write_status_now(state)

        await asyncio.sleep(state.train_interval_sec)


async def _status_loop(state: WorkerState) -> None:
    while True:
        await _write_status_now(state)
        await asyncio.sleep(state.status_interval_sec)


def _scheduled_top_gainer_slot(now_local: datetime) -> tuple[str, date, str] | None:
    if now_local.hour == 0 and now_local.minute < 15:
        target_day = now_local.date() - timedelta(days=1)
        slot_key = f"{target_day.isoformat()}::final"
        return "final", target_day, slot_key
    if now_local.hour == 12 and now_local.minute < 15:
        target_day = now_local.date()
        slot_key = f"{target_day.isoformat()}::midday"
        return "midday", target_day, slot_key
    return None


async def _top_gainer_critic_loop(state: WorkerState) -> None:
    log = logging.getLogger("rl_headless_worker.top_gainer_critic")
    tz_name = str(getattr(config, "TOP_GAINER_CRITIC_TIMEZONE", "Europe/Budapest"))
    tz = ZoneInfo(tz_name)
    top_n = int(getattr(config, "TOP_GAINER_CRITIC_TOP_N", top_gainer_critic.DEFAULT_TOP_N))
    min_quote_volume = float(
        getattr(
            config,
            "TOP_GAINER_CRITIC_MIN_QUOTE_VOLUME_24H",
            top_gainer_critic.DEFAULT_MIN_QUOTE_VOLUME,
        )
    )
    while True:
        try:
            if not state.top_gainer_enabled:
                await asyncio.sleep(DEFAULT_TOP_GAINER_CHECK_SEC)
                continue

            slot = _scheduled_top_gainer_slot(datetime.now(tz))
            if slot is None:
                await asyncio.sleep(DEFAULT_TOP_GAINER_CHECK_SEC)
                continue

            phase, target_day, slot_key = slot
            if slot_key == state.top_gainer_last_slot_key:
                await asyncio.sleep(DEFAULT_TOP_GAINER_CHECK_SEC)
                continue

            state.top_gainer_runs_total += 1
            state.top_gainer_last_started_at = _utc_now_iso()
            state.top_gainer_last_error = ""
            result = await asyncio.to_thread(
                top_gainer_critic.run_report,
                target_day=target_day,
                phase=phase,
                timezone_name=tz_name,
                top_n=top_n,
                min_quote_volume=min_quote_volume,
            )
            summary = result.get("summary", {})
            files = result.get("files", {})
            state.top_gainer_runs_ok += 1
            state.top_gainer_last_slot_key = slot_key
            state.top_gainer_last_finished_at = _utc_now_iso()
            state.top_gainer_last_phase = phase
            state.top_gainer_last_target_day_local = target_day.isoformat()
            state.top_gainer_last_report_json = str(files.get("json", ""))
            state.top_gainer_last_report_txt = str(files.get("txt", ""))
            state.top_gainer_last_capture_rate_pct = summary.get("watchlist_top_capture_rate_pct")
            state.top_gainer_last_early_capture_rate_pct = summary.get("watchlist_top_early_capture_rate_pct")
            log.info(
                "Top-gainer critic done: phase=%s day=%s capture=%s%% early=%s%%",
                phase,
                target_day.isoformat(),
                state.top_gainer_last_capture_rate_pct,
                state.top_gainer_last_early_capture_rate_pct,
            )
            if bool(getattr(config, "TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED", True)):
                dedup_key = f"top_gainer_{phase}_{target_day.isoformat()}"
                if _try_claim_report_send(dedup_key, cooldown_minutes=60):
                    await _send_telegram_text(_render_top_gainer_telegram(result))
            await _write_status_now(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.top_gainer_runs_failed += 1
            state.top_gainer_last_finished_at = _utc_now_iso()
            state.top_gainer_last_error = str(exc)
            log.exception("Top-gainer critic failed: %s", exc)
            await _write_status_now(state)
        await asyncio.sleep(DEFAULT_TOP_GAINER_CHECK_SEC)


async def _write_status_now(state: WorkerState) -> None:
    log = logging.getLogger("rl_headless_worker.status")
    try:
        critic_report = await asyncio.to_thread(report_critic_dataset.build_report)
        ml_rows_total = await asyncio.to_thread(_count_jsonl_rows, ml_dataset.ML_FILE)
        snapshot = build_status_snapshot(
            state,
            critic_report=critic_report,
            ml_rows_total=ml_rows_total,
        )
        await asyncio.to_thread(_write_status, snapshot)
        log.info(
            "Status updated: critic_rows=%s ml_rows=%s train_ok=%s/%s",
            critic_report.get("rows_total", 0),
            ml_rows_total,
            state.train_runs_ok,
            state.train_runs_total,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("Status update failed: %s", exc)


async def _amain(args: argparse.Namespace) -> int:
    state = WorkerState(
        train_interval_sec=args.train_every_minutes * 60,
        status_interval_sec=args.status_every_seconds,
        min_rows=args.min_rows,
        min_new_rows=args.min_new_rows,
        collector_enabled=args.enable_collector,
    )

    logging.getLogger("rl_headless_worker").info(
        "Headless RL worker started: train_every=%sm min_rows=%s min_new_rows=%s collector=%s",
        args.train_every_minutes,
        args.min_rows,
        args.min_new_rows,
        args.enable_collector,
    )

    tasks = [
        asyncio.create_task(_training_loop(state), name="training"),
        asyncio.create_task(_status_loop(state), name="status"),
        asyncio.create_task(_top_gainer_critic_loop(state), name="top_gainer_critic"),
    ]
    if args.enable_collector:
        tasks.insert(0, asyncio.create_task(_collector_supervisor(state), name="collector"))
    else:
        state.collector_last_error = "disabled (bot-owned datasets)"
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RL/ML headless worker without Telegram: collect market data and retrain the candidate ranker.",
    )
    parser.add_argument(
        "--train-every-minutes",
        type=int,
        default=int(os.getenv("RL_WORKER_TRAIN_EVERY_MINUTES", "60")),
    )
    parser.add_argument(
        "--status-every-seconds",
        type=int,
        default=int(os.getenv("RL_WORKER_STATUS_EVERY_SECONDS", "300")),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=int(os.getenv("RL_WORKER_MIN_ROWS", str(DEFAULT_MIN_ROWS))),
    )
    parser.add_argument(
        "--min-new-rows",
        type=int,
        default=int(os.getenv("RL_WORKER_MIN_NEW_ROWS", str(DEFAULT_MIN_NEW_ROWS))),
    )
    parser.add_argument(
        "--enable-collector",
        action="store_true",
        default=str(os.getenv("RL_WORKER_ENABLE_COLLECTOR", str(getattr(config, "RL_WORKER_ENABLE_COLLECTOR", False)))).strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--disable-collector",
        action="store_true",
        help="Disable market collection in the headless RL worker and use datasets produced by the trading bot.",
    )
    args = parser.parse_args()
    if args.disable_collector:
        args.enable_collector = False
    return args


def main() -> int:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    args = _parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
