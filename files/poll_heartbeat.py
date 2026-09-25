"""Poll heartbeat: one line per coin per closed bar, saying what the live poll saw.

Why (audit 2026-09-25, P0 hypothesis P-2): on 17.5% of immutable top-20
winner-days an entry rule fired on the stored klines inside the birth window,
yet the bot logged no candidate at all. Nothing in the logs can say whether the
coin was not polled on that bar, was polled but the live bar differed, sat in
cooldown, or already had an open position -- `_poll_coin` returns silently in
all four cases. This file records the outcome of every poll, deduplicated to the
first poll of each closed bar, so the blind spot becomes measurable.

Logging only: no decision reads it. Every call is wrapped so a failure here can
never reach the poll loop. Files rotate daily under .runtime/poll_heartbeat/ and
are pruned after POLL_HEARTBEAT_KEEP_DAYS.

Spec: docs/specs/features/poll-heartbeat-spec.md
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import config

log = logging.getLogger(__name__)

DIR = Path(__file__).resolve().parent.parent / ".runtime" / "poll_heartbeat"
REASON_MAX = 80
STAGES = ("no_data", "short_history", "position_open", "cooldown", "evaluated")

_last: Dict[tuple, int] = {}
_pruned_day: Optional[str] = None


def enabled() -> bool:
    return bool(getattr(config, "POLL_HEARTBEAT_ENABLED", False))


def _bar_key(tf: str, bar_ts: Optional[int]) -> int:
    """The closed bar a record belongs to; polls without one use the wall clock."""
    if bar_ts is not None:
        return int(bar_ts)
    ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000
    return int(time.time() * 1000) // ms * ms


def _prune(today: str) -> None:
    global _pruned_day
    if _pruned_day == today:
        return
    _pruned_day = today
    keep = int(getattr(config, "POLL_HEARTBEAT_KEEP_DAYS", 45))
    cut = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=keep)).strftime("%Y-%m-%d")
    for f in DIR.glob("*.jsonl"):
        if f.stem < cut:
            try:
                f.unlink()
            except OSError:
                pass


def record(sym: str, tf: str, stage: str, bar_ts: Optional[int] = None,
           rules: Optional[Dict[str, Optional[bool]]] = None,
           reasons: Optional[Dict[str, str]] = None, **extra) -> bool:
    """Append one heartbeat line; returns True if written, False if deduplicated or off."""
    try:
        if not enabled():
            return False
        key = (sym, tf)
        bk = _bar_key(tf, bar_ts)
        if _last.get(key) == bk:
            return False
        _last[key] = bk
        now = datetime.now(timezone.utc)
        row = {"ts": now.isoformat(timespec="seconds"), "sym": sym, "tf": tf, "stage": stage,
               "bar_ts": bk, "bar_utc": datetime.fromtimestamp(bk / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M")}
        if rules is not None:
            row["rules"] = rules
            row["fired"] = sorted(k for k, v in rules.items() if v)
        if reasons:
            row["reasons"] = {k: str(v)[:REASON_MAX] for k, v in reasons.items() if v}
        row.update({k: v for k, v in extra.items() if v is not None})
        day = now.strftime("%Y-%m-%d")
        DIR.mkdir(parents=True, exist_ok=True)
        with open(DIR / (day + ".jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        _prune(day)
        return True
    except Exception as e:  # never let instrumentation touch the poll loop
        log.debug("poll heartbeat skipped for %s: %s", sym, e)
        return False
