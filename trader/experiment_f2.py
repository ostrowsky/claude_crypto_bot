"""F2 runner -- learned exit vs X4 in money (trader-learning-loop-f-1020-spec.md §4).

This module holds the one-time opening of the test half. The train/test runner on the B-history paths and
the bot's real entries is the next step of the spec (its own tests first); until then nothing here trades
or reads the test half.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import trader_config as TC

MARKER = TC.RUNTIME / "experiments" / "F2_TEST_OPENED.json"


def open_test(marker: Path = MARKER, info: dict | None = None) -> None:
    """The test half is opened once: a second call raises."""
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("F2: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**(info or {}), "opened_utc": datetime.now(timezone.utc).isoformat()},
                                 ensure_ascii=False), encoding="utf-8")
