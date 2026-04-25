"""
build_info.py — версия и дата билда бота.

BUILD_VERSION задаётся вручную при каждом значимом релизе.
BUILD_DATE автоматически определяется как дата последнего изменения
ключевых файлов (monitor.py, strategy.py, config.py, bot.py).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

BUILD_VERSION: str = "2.5.0"

# Ключевые файлы — берём самый «свежий» mtime
_KEY_FILES = ["monitor.py", "strategy.py", "config.py", "bot.py",
              "contextual_bandit.py", "trend_scout.py"]

_ROOT = Path(__file__).resolve().parent


def _latest_mtime() -> float:
    """Возвращает самый поздний mtime среди ключевых файлов."""
    latest = 0.0
    for name in _KEY_FILES:
        p = _ROOT / name
        try:
            mt = p.stat().st_mtime
            if mt > latest:
                latest = mt
        except OSError:
            pass
    return latest


def get_build_date_utc() -> datetime:
    """Возвращает дату билда как UTC datetime."""
    mt = _latest_mtime()
    if mt == 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(mt, tz=timezone.utc)


def get_build_info_str(*, local_tz_offset_hours: int = 3) -> str:
    """
    Возвращает строку вида: v2.5.0 · 2026-04-15 15:57 (UTC+3)
    """
    bd = get_build_date_utc()
    from datetime import timedelta
    local_bd = bd + timedelta(hours=local_tz_offset_hours)
    tz_label = f"UTC{local_tz_offset_hours:+d}" if local_tz_offset_hours != 0 else "UTC"
    return (
        f"v{BUILD_VERSION} · "
        f"{local_bd.strftime('%Y-%m-%d %H:%M')} ({tz_label})"
    )
