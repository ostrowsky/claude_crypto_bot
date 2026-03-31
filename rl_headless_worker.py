from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import critic_dataset
import data_collector
import ml_candidate_ranker
import ml_dataset
import report_candidate_ranker_shadow
import report_critic_dataset
from ml_signal_model import save_json


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
STATUS_FILE = RUNTIME_DIR / "rl_worker_status.json"
LOG_FILE = WORKSPACE_ROOT / "rl_worker_stderr.log"

MODEL_FILE = ROOT / "ml_candidate_ranker.json"
TRAIN_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"
SHADOW_REPORT_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"

DEFAULT_TRAIN_INTERVAL_SEC = 60 * 60
DEFAULT_STATUS_INTERVAL_SEC = 5 * 60
DEFAULT_MIN_ROWS = 500
DEFAULT_MIN_NEW_ROWS = 50

# PATCH: Acceptance gate thresholds for enabling ML_CANDIDATE_RANKER_RUNTIME_ENABLED
ACCEPTANCE_AUC_MIN: float = 0.55          # val AUC must exceed this
ACCEPTANCE_TOP1_DELTA_MIN: float = 0.15   # shadow top-1 delta > +0.15%
ACCEPTANCE_LIVE_ROWS_MIN: int = 1000      # live (non-bootstrap) rows
ACCEPTANCE_SCORE_WEIGHT: float = 5.0      # ML_CANDIDATE_RANKER_SCORE_WEIGHT when enabled
RL_CONFIG_OVERRIDE_FILE = WORKSPACE_ROOT / "rl_config_overrides.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows += 1
    return rows


def _count_live_rows(path: Path) -> int:
    """PATCH: Count non-bootstrap rows in critic_dataset.
    Live rows have real candidate_score != 0 (logged by _log_critic_candidate in monitor).
    Bootstrap rows have reason_code='bootstrap_ml_dataset' with all decision features zeroed.
    """
    if not path.exists():
        return 0
    live = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            rc = str((rec.get("decision") or {}).get("reason_code", ""))
            if rc != "bootstrap_ml_dataset":
                live += 1
        except Exception:
            pass
    return live


def _check_acceptance_gate(
    train_result: dict,
    live_rows: int,
) -> tuple[bool, str]:
    """
    PATCH: Formal acceptance gate for enabling ML ranker in production.
    Returns (should_enable, reason_string).

    Criteria (ALL must pass):
      1. val AUC (MLP) >= ACCEPTANCE_AUC_MIN (0.55)
      2. shadow top-1 delta > ACCEPTANCE_TOP1_DELTA_MIN (+0.15%)
      3. live_rows >= ACCEPTANCE_LIVE_ROWS_MIN (1000)
    """
    train_report = train_result.get("train_report", {})
    shadow_report = train_result.get("shadow_report", {})

    # Gate 1: AUC
    val = train_report.get("validation", {})
    mlp_auc = float((val.get("mlp") or {}).get("auc", 0.0))
    if mlp_auc < ACCEPTANCE_AUC_MIN:
        return False, f"AUC={mlp_auc:.4f} < {ACCEPTANCE_AUC_MIN} (need more diverse data)"

    # Gate 2: Shadow top-1 delta
    top1_delta = train_result.get("top1_delta")
    if top1_delta is None:
        return False, "No shadow top-1 delta available"
    if float(top1_delta) < ACCEPTANCE_TOP1_DELTA_MIN:
        return False, f"top1_delta={top1_delta:.4f}% < {ACCEPTANCE_TOP1_DELTA_MIN}%"

    # Gate 3: Live rows
    if live_rows < ACCEPTANCE_LIVE_ROWS_MIN:
        return False, f"live_rows={live_rows} < {ACCEPTANCE_LIVE_ROWS_MIN} (bootstrap-only data)"

    return True, f"AUC={mlp_auc:.4f} top1_delta={top1_delta:+.4f}% live_rows={live_rows}"


def _apply_config_override(key: str, value, reason: str) -> None:
    """
    PATCH: Write config overrides to rl_config_overrides.json.
    monitor.py reads this file on startup to apply ML parameter changes.
    This allows the RL worker to tune config without restarting the bot.
    """
    log = logging.getLogger("rl_headless_worker.acceptance")
    try:
        existing = {}
        if RL_CONFIG_OVERRIDE_FILE.exists():
            existing = json.loads(RL_CONFIG_OVERRIDE_FILE.read_text())
        existing[key] = value
        existing["_reason"] = reason
        existing["_ts"] = _utc_now_iso()
        RL_CONFIG_OVERRIDE_FILE.write_text(json.dumps(existing, indent=2))
        log.info("Config override written: %s=%s (%s)", key, value, reason)
    except Exception as exc:
        log.warning("Failed to write config override: %s", exc)


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
            "running": True,
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
        # PATCH: acceptance gate status
        "acceptance_gate": {
            "passed": state.acceptance_passed,
            "reason": state.acceptance_reason,
            "ts": state.acceptance_ts,
            "live_rows": state.last_live_rows,
            "thresholds": {
                "auc_min": ACCEPTANCE_AUC_MIN,
                "top1_delta_min": ACCEPTANCE_TOP1_DELTA_MIN,
                "live_rows_min": ACCEPTANCE_LIVE_ROWS_MIN,
            },
        },
    }


def _write_status(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_FILE)


def _train_ranker_once(min_rows: int) -> Dict[str, Any]:
    report = ml_candidate_ranker.train_and_evaluate(
        critic_dataset.CRITIC_FILE,
        min_rows=min_rows,
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

    # PATCH: extract val AUC for acceptance gate
    val = report_without_payload.get("validation", {})
    val_auc = float((val.get("mlp") or {}).get("auc", 0.0))
    days_covered = float(report_without_payload.get("dataset_days_covered", 0.0))
    excluded = report_without_payload.get("temporal_features_excluded", [])
    return {
        "model_name": report_without_payload.get("chosen_model", ""),
        "rows_total": int(report_without_payload.get("rows_total", 0)),
        "dataset_mtime": _file_mtime(critic_dataset.CRITIC_FILE),
        "top1_delta": top1_delta,
        "val_auc": val_auc,
        "days_covered": days_covered,
        "temporal_features_excluded": excluded,
        "train_report": report_without_payload,
        "shadow_report": shadow_report,
    }


@dataclass
class WorkerState:
    train_interval_sec: int
    status_interval_sec: int
    min_rows: int
    min_new_rows: int
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
    # PATCH: acceptance gate tracking
    last_live_rows: int = 0
    acceptance_passed: bool = False
    acceptance_reason: str = ""
    acceptance_ts: Optional[str] = None


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
            log.info(
                "Ranker trained: model=%s rows=%s top1_delta=%s val_auc=%.4f days=%.0f excluded=%s",
                state.last_model_name,
                state.last_trained_rows,
                state.last_top1_delta,
                float(result.get("val_auc", 0)),
                float(result.get("days_covered", 0)),
                result.get("temporal_features_excluded", []),
            )
            # PATCH: run acceptance gate to decide if ranker is ready for production
            live_rows = await asyncio.to_thread(_count_live_rows, critic_dataset.CRITIC_FILE)
            state.last_live_rows = live_rows
            gate_pass, gate_reason = _check_acceptance_gate(result, live_rows)
            if gate_pass and not state.acceptance_passed:
                state.acceptance_passed = True
                state.acceptance_reason = gate_reason
                state.acceptance_ts = _utc_now_iso()
                log.info("ACCEPTANCE GATE PASSED: %s", gate_reason)
                log.info("Enabling ML_CANDIDATE_RANKER_SCORE_WEIGHT = %.1f", ACCEPTANCE_SCORE_WEIGHT)
                _apply_config_override(
                    "ML_CANDIDATE_RANKER_SCORE_WEIGHT",
                    ACCEPTANCE_SCORE_WEIGHT,
                    f"Acceptance gate passed: {gate_reason}",
                )
                _apply_config_override(
                    "ML_CANDIDATE_RANKER_RUNTIME_ENABLED",
                    True,
                    f"Acceptance gate passed at {_utc_now_iso()}",
                )
            elif not gate_pass:
                log.info("Acceptance gate NOT passed: %s (live_rows=%d)", gate_reason, live_rows)
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
    )

    logging.getLogger("rl_headless_worker").info(
        "Headless RL worker started: train_every=%sm min_rows=%s min_new_rows=%s",
        args.train_every_minutes,
        args.min_rows,
        args.min_new_rows,
    )

    tasks = [
        asyncio.create_task(_collector_supervisor(state), name="collector"),
        asyncio.create_task(_training_loop(state), name="training"),
        asyncio.create_task(_status_loop(state), name="status"),
    ]
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
    return parser.parse_args()


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
