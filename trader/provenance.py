"""Where a result came from (protocol-fixes-1105-spec.md P1): code state and fingerprints of the inputs.

  code_state     git HEAD and whether trader/ has uncommitted changes, plus bot_state
  bot_state      digests of the signal bot's files the replay imports or reads (config, overrides, exit_validator,
                 grid builders, watchlist) and whether files/ has uncommitted changes -- the trader reads them although
                 its brief says events, candles and .env only (a known deviation since E1)
  file_digest    sha256 of a file (first 16 hex)
  events_digest  sha256 + count of bot_events lines with ts < until: the file keeps growing, the past does not
  grid_digest    sha256 of the C/H/L/O arrays of a 15m grid on bars [i0, i1)
  block          the dict a result stores as "provenance"
  require_clean  refuses to start an experiment on uncommitted trader code
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOT_FILES = ("files/config.py", "files/exit_validator.py", "files/_backtest_trend_start_detector.py",
             "files/_compute_early_capture.py", ".runtime/release/runtime_overrides.json", "files/watchlist.json")


def _git(*a, cwd=HERE):
    try:
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return None


def bot_state() -> dict:
    return {"files": {f: file_digest(ROOT / f) for f in BOT_FILES},
            "dirty": bool(_git("status", "--porcelain", "--", "files", cwd=ROOT))}


def code_state() -> dict:
    return {"head": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", ".")),
            "bot": bot_state()}


def require_clean(status=None) -> None:
    """No experiment result on uncommitted trader code (fourth-review-1105-spec.md §4). files/ of the bot is only
    recorded: its models change all the time and they are not this code."""
    st = _git("status", "--porcelain", "--", ".") if status is None else status
    if st:
        raise RuntimeError("uncommitted code in trader/, commit before running:\n" + st)


def file_digest(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def events_digest(path, until: str) -> dict:
    h, n = hashlib.sha256(), 0
    with Path(path).open("rb") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                ts = str(json.loads(raw.decode("utf-8", "replace")).get("ts", ""))
            except Exception:                               # noqa: BLE001
                continue
            if ts and ts < until:
                h.update(raw.rstrip(b"\r\n"))
                h.update(b"\n")
                n += 1
    return {"sha": h.hexdigest()[:16], "n": n, "until": until}


def grid_digest(grid, i0: int, i1: int) -> str:
    h = hashlib.sha256()
    for a in ("C", "H", "L", "O"):
        arr = getattr(grid, a, None)
        if arr is not None:
            h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(arr[i0:i1], dtype=float), nan=-1.0)).tobytes())
    return h.hexdigest()[:16]


def block(**parts) -> dict:
    return {"code": code_state(), "utc": datetime.now(timezone.utc).isoformat(), **parts}
