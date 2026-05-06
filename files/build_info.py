"""
build_info.py — версия и дата применения текущей версии бота.

ПРАВИЛО (из AGENTS.md):
  После любого изменения в коде:
    1) Инкрементировать BUILD_VERSION (semver: patch/minor/major).
    2) Обновить BUILD_APPLIED_AT_UTC на текущий UTC ISO timestamp.
    3) Добавить запись в .runtime/version_history.jsonl
       (это делается автоматически при импорте модуля).

BUILD_APPLIED_AT_UTC — это «когда новая версия начала действовать»,
не «когда последний раз тыкали редактором» и не «когда стартовал бот».
Конкретное время фиксируется человеком/агентом одновременно с
изменением кода.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Source of truth ────────────────────────────────────────────────────────
BUILD_VERSION: str = "2.15.0"
BUILD_APPLIED_AT_UTC: str = "2026-05-06T11:55:00Z"
BUILD_NOTES: str = "always-watch pump detector + auto-reanalyze 2h→30min (STRK-class miss fix)"

# ── Helpers ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent


def get_applied_at_utc() -> datetime:
    """Возвращает время применения текущей версии как UTC datetime."""
    try:
        s = BUILD_APPLIED_AT_UTC.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now(timezone.utc)


def get_build_info_str(*, local_tz_offset_hours: int = 3) -> str:
    """
    Возвращает строку вида: v2.5.1 · 2026-04-30 12:06 (UTC+3)

    Использует BUILD_APPLIED_AT_UTC, не mtime файлов и не время старта бота.
    """
    bd = get_applied_at_utc()
    local_bd = bd + timedelta(hours=local_tz_offset_hours)
    tz_label = (
        f"UTC{local_tz_offset_hours:+d}"
        if local_tz_offset_hours != 0
        else "UTC"
    )
    return (
        f"v{BUILD_VERSION} · "
        f"{local_bd.strftime('%Y-%m-%d %H:%M')} ({tz_label})"
    )


# ── Backward compatibility (старые callers могут вызывать get_build_date_utc) ──
def get_build_date_utc() -> datetime:
    """Алиас get_applied_at_utc — оставлен для обратной совместимости."""
    return get_applied_at_utc()


# ── Append to .runtime/version_history.jsonl on import (idempotent) ────────
def _append_version_history() -> None:
    """Append (version, applied_at) once per (version, applied_at) pair."""
    path = _ROOT / ".runtime" / "version_history.jsonl"
    try:
        path.parent.mkdir(exist_ok=True)
        # Idempotency: skip if last record matches
        last = None
        if path.exists():
            try:
                with io.open(path, encoding="utf-8") as f:
                    for ln in f:
                        if ln.strip():
                            try:
                                last = json.loads(ln)
                            except Exception:
                                pass
            except OSError:
                last = None
        rec = {
            "version": BUILD_VERSION,
            "applied_at_utc": BUILD_APPLIED_AT_UTC,
            "notes": BUILD_NOTES,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        if last and last.get("version") == BUILD_VERSION and \
           last.get("applied_at_utc") == BUILD_APPLIED_AT_UTC:
            return
        with io.open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # Никогда не валим бота из-за лога
        pass


_append_version_history()
