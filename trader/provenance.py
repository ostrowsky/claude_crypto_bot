"""Where a result came from (protocol-fixes-1105-spec.md P1): code state and fingerprints of the inputs.

  code_state     git HEAD and whether trader/ has uncommitted changes
  file_digest    sha256 of a file (first 16 hex)
  events_digest  sha256 + count of bot_events lines with ts < until: the file keeps growing, the past does not
  grid_digest    sha256 of the C/H/L/O arrays of a 15m grid on bars [i0, i1)
  block          the dict a result stores as "provenance"
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def code_state() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return None
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}


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
