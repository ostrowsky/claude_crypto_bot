"""Tiny in-memory snapshot used by Telegram menu/positions render paths.

Why this exists
---------------
The old hot path on every Menu/Positions button call did:
  1. config.load_watchlist()      — disk I/O
  2. _refresh_positions_state()   — disk I/O + dict merge
  3. kb_main()                    — recompute everything again

If the event loop is held by any heavy task (monitoring_loop, data_collector,
RL trainer), the menu handler can't even START until the loop frees. The
operator sees 6+ unanswered button presses.

The fix: render UI from a pre-computed snapshot that a background keeper
updates every TTL seconds. Hot-path reads become a single locked dict
lookup — sub-millisecond regardless of disk state or loop contention.

Design notes:
  - Threading lock, not asyncio lock — keeper may run via asyncio.to_thread,
    hot path may be sync (kb_main is non-async). A simple threading.Lock
    serialises both safely.
  - Immutable dataclass — once published, the snapshot value never mutates.
    Concurrent readers see a consistent view without copying.
  - No telegram / config deps in this module — keeps it testable without
    spinning up the whole bot stack.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

DEFAULT_TTL_SEC = 5.0


@dataclass(frozen=True)
class UISnapshot:
    captured_at: float
    wl_count:    int
    hot_count:   int
    pos_count:   int
    running:     bool


_snapshot: UISnapshot | None = None
_lock = threading.Lock()


def get() -> UISnapshot | None:
    """Hot-path read. Returns the latest cached snapshot or None if the
    keeper hasn't published one yet."""
    with _lock:
        return _snapshot


def get_or_default(default: UISnapshot) -> UISnapshot:
    """Return cached snapshot if available, otherwise the supplied fallback.
    Use this in render paths so a cold-start never blocks on the cache."""
    snap = get()
    return snap if snap is not None else default


def publish(snap: UISnapshot) -> None:
    """Replace the cached snapshot atomically."""
    global _snapshot
    with _lock:
        _snapshot = snap


def refresh(*, wl_count: int, hot_count: int, pos_count: int,
            running: bool, now: float | None = None) -> UISnapshot:
    """Convenience: build + publish a snapshot in one call. Returns the new value."""
    snap = UISnapshot(
        captured_at=now if now is not None else time.time(),
        wl_count=wl_count,
        hot_count=hot_count,
        pos_count=pos_count,
        running=running,
    )
    publish(snap)
    return snap


def age_seconds(now: float | None = None) -> float:
    """How long ago was the cache published. inf if uninitialised."""
    snap = get()
    if snap is None:
        return float("inf")
    return (now if now is not None else time.time()) - snap.captured_at


def is_stale(ttl_sec: float = DEFAULT_TTL_SEC, *, now: float | None = None) -> bool:
    return age_seconds(now) > ttl_sec


def _reset_for_tests() -> None:
    """Test helper — never call from production code."""
    global _snapshot
    with _lock:
        _snapshot = None
