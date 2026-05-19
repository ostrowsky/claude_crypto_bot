"""
build_info.py — версия и дата текущего кода бота.

NO HARDCODE. Версия и дата ВСЕГДА вычисляются вживую из git HEAD
(committer-date коммита, который реально выполняется). Раньше это были
ручные константы (BUILD_VERSION/BUILD_APPLIED_AT_UTC) — их забывали
бампать, и баннер врал (показывал v2.20.0 · 07.05 на коде от 19.05).
Теперь ничего бампать не нужно: данные не могут устареть.

Источник истины (по приоритету):
  1. git: версия = "r<commit_count>-<short_sha>", дата = committer-date HEAD.
  2. fallback (git недоступен): самый свежий mtime среди files/*.py —
     тоже АКТУАЛЬНОЕ значение, а не замороженная строка.

Значения кэшируются на короткий TTL, поэтому даже уже запущенный
процесс подхватит новый деплой в течение ~5 минут без рестарта.
"""

from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FILES = Path(__file__).resolve().parent

# ── Live cache (TTL) ────────────────────────────────────────────────────────
_CACHE_TTL_SEC = 300
_cache: dict | None = None
_cache_ts: float = 0.0


def _git(*args: str) -> str | None:
    """Run `git -C <repo> ...`; return stripped stdout or None. Never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT), *args],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            s = (out.stdout or "").strip()
            return s or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _newest_src_mtime() -> datetime:
    """Fallback when git is unavailable: newest *.py mtime under files/.
    Still ACTUAL (reflects code on disk), never a frozen constant."""
    newest = 0.0
    try:
        for p in _FILES.glob("*.py"):
            try:
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
            except OSError:
                continue
    except OSError:
        pass
    if newest <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(newest, tz=timezone.utc)


def _compute() -> dict:
    """Resolve version + applied_at from git (fallback: src mtime)."""
    count = _git("rev-list", "--count", "HEAD")
    sha = _git("rev-parse", "--short", "HEAD")
    cdate = _git("log", "-1", "--format=%cI")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    if sha and cdate:
        version = f"r{count}-{sha}" if count else sha
        if branch and branch not in ("HEAD", "main", "master"):
            version = f"{version} ({branch})"
        try:
            applied = datetime.fromisoformat(cdate).astimezone(timezone.utc)
        except (ValueError, TypeError):
            applied = _newest_src_mtime()
        source = "git"
    else:
        applied = _newest_src_mtime()
        version = "src-" + applied.strftime("%Y%m%d-%H%M")
        source = "src-mtime"

    return {
        "version": version,
        "applied_at": applied,
        "applied_at_utc": applied.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": _git("log", "-1", "--format=%s") or "",
        "source": source,
    }


def _info() -> dict:
    """Cached live build info (TTL). Recomputes after a deploy without
    needing a process restart."""
    global _cache, _cache_ts
    import time

    now = time.time()
    if _cache is None or (now - _cache_ts) > _CACHE_TTL_SEC:
        _cache = _compute()
        _cache_ts = now
    return _cache


# ── Public API (live; backward-compatible names) ────────────────────────────

def get_applied_at_utc() -> datetime:
    """Время текущего кода (committer-date HEAD) как UTC datetime — вживую."""
    return _info()["applied_at"]


def get_build_date_utc() -> datetime:
    """Алиас get_applied_at_utc (обратная совместимость)."""
    return get_applied_at_utc()


def get_build_info_str(*, local_tz_offset_hours: int = 3) -> str:
    """Строка вида: v r78-f88bb81 · 2026-05-19 20:27 (UTC+3). Вживую из git."""
    info = _info()
    bd = info["applied_at"] + timedelta(hours=local_tz_offset_hours)
    tz_label = (
        f"UTC{local_tz_offset_hours:+d}" if local_tz_offset_hours != 0 else "UTC"
    )
    return f"v{info['version']} · {bd.strftime('%Y-%m-%d %H:%M')} ({tz_label})"


# Backward-compat module attributes (computed at import; the display path
# above is always live via the TTL cache regardless of these).
_boot = _info()
BUILD_VERSION: str = _boot["version"]
BUILD_APPLIED_AT_UTC: str = _boot["applied_at_utc"]
BUILD_NOTES: str = _boot["notes"]


# ── Append to .runtime/version_history.jsonl on import (idempotent) ─────────
def _append_version_history() -> None:
    """Append once per (version, applied_at) — i.e. once per deployed commit."""
    path = _ROOT / ".runtime" / "version_history.jsonl"
    try:
        path.parent.mkdir(exist_ok=True)
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
