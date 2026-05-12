"""Shared utilities for the bot-improvement pipeline (L1..L6).

Single source of truth for file paths, status thresholds and tiny helpers.
Anything that pipeline scripts need in common goes here so individual scripts
stay short and readable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = REPO_ROOT / "files"
PYEMBED   = REPO_ROOT / "pyembed" / "python.exe"
RUNTIME   = REPO_ROOT / ".runtime"
REPORTS   = RUNTIME / "reports"
PIPELINE  = RUNTIME / "pipeline"

# CRYPTOBOT_PIPELINE_NAMESPACE allows stress tests / replays to redirect ALL
# writable state (decisions, hypotheses, shadow_runs, ...) into an isolated
# subdirectory so synthetic runs never pollute production logs. Read-only
# inputs (do_not_touch, learning_progress, critic history) are unaffected.
_NS = os.environ.get("CRYPTOBOT_PIPELINE_NAMESPACE", "").strip()
_NS_ROOT = (PIPELINE / "namespaces" / _NS) if _NS else PIPELINE

HEALTH      = _NS_ROOT / "health"
BASELINE    = _NS_ROOT / "baseline"
HYPOTHESES  = _NS_ROOT / "hypotheses"
SHADOW_RUNS = _NS_ROOT / "shadow_runs"
DECISIONS_DIR = _NS_ROOT / "decisions"
DECISIONS_LOG = DECISIONS_DIR / "decisions.jsonl"
ALREADY_TRIED = DECISIONS_DIR / "already_tried.jsonl"
DO_NOT_TOUCH = PIPELINE / "do_not_touch.json"   # always the prod copy

LEARNING_PROGRESS = RUNTIME / "learning_progress.jsonl"
METRICS_DAILY     = RUNTIME / "metrics_daily.jsonl"
CRITIC_HISTORY    = REPORTS / "top_gainer_critic_history.jsonl"
EVAL_OUTPUT_DIR   = REPO_ROOT / "evaluation_output"
PER_MODE_DIR      = EVAL_OUTPUT_DIR / "per_mode"

# Status thresholds — derived from references/metrics.md of signal-efficiency-evaluator
# and from CLAUDE.md success metric (top-20 daily gainer recall + early entry).
THRESHOLDS = {
    "watchlist_top_bought_pct":       {"green": 0.66, "yellow": 0.50, "red": 0.33},
    "watchlist_top_early_capture_pct":{"green": 0.40, "yellow": 0.25, "red": 0.15},
    "false_positive_rate":            {"green": 0.30, "yellow": 0.50, "red": 0.70, "lower_is_better": True},
    "median_capture_ratio":           {"green": 0.60, "yellow": 0.40, "red": 0.20},
    "median_buy_lateness_pct":        {"green": 0.10, "yellow": 0.25, "red": 0.40, "lower_is_better": True},
    "recall_at_20":                   {"green": 0.80, "yellow": 0.60, "red": 0.40},
    "ucb_separation":                 {"green": 0.10, "yellow": 0.05, "red": 0.02},
    "model_auc_top20":                {"green": 0.90, "yellow": 0.80, "red": 0.70},
}


def classify(value: float | None, key: str) -> str:
    """Return one of: green | yellow | red | critical | unknown."""
    if value is None:
        return "unknown"
    t = THRESHOLDS.get(key)
    if not t:
        return "unknown"
    lower_better = t.get("lower_is_better", False)
    g, y, r = t["green"], t["yellow"], t["red"]
    if lower_better:
        if value <= g: return "green"
        if value <= y: return "yellow"
        if value <= r: return "red"
        return "critical"
    else:
        if value >= g: return "green"
        if value >= y: return "yellow"
        if value >= r: return "red"
        return "critical"


def status_emoji(status: str) -> str:
    return {
        "green":    "🟢",
        "yellow":   "🟡",
        "red":      "🔴",
        "critical": "🚨",
        "unknown":  "❓",
    }.get(status, "❓")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_do_not_touch() -> dict:
    """Return parsed do_not_touch.json or an empty stub if missing."""
    d = read_json(DO_NOT_TOUCH)
    if not d:
        return {"gates": [], "config_keys_locked": []}
    return d


def load_already_tried() -> list[dict]:
    return list(iter_jsonl(ALREADY_TRIED))
