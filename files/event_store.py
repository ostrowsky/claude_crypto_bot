"""Append-only JSONL stays the journal; SQLite becomes the thing you query.

Every analysis in this repo re-parses `bot_events.jsonl` from byte zero — 98 MB
and growing — and every label write used to rewrite a whole dataset in place.
That write model is not merely slow, it fails: on 2026-08-13 the backfill
finished fetching every label and then died with

    PermissionError: [WinError 5]
      critic_dataset.jsonl.backfill.tmp -> critic_dataset.jsonl

because the live bot holds the file open. Nineteen orphaned `.tmp` files were
sitting in `files/` at the time, 1.09 GB, one per earlier failure, the oldest 40
days old.

This module borrows the shape that works in the sibling repo: the JSONL is never
rewritten, and a SQLite mirror is brought forward from the last byte offset it
consumed. Re-running a sync costs only the bytes appended since.

    pyembed\\python.exe files\\event_store.py sync
    pyembed\\python.exe files\\event_store.py stats

Design notes worth keeping:

* the primary key is `(source_file, byte_offset)`, so a replayed or duplicated
  sync cannot double-count a row;
* a source that shrank was truncated or rotated, and its offset is reset with
  its rows dropped, because continuing from a stale offset would silently splice
  two different files together;
* `SCHEMA_VERSION` bumps rebuild from scratch — a migration that half-applies is
  worse than a rebuild that takes two minutes;
* one writer at a time via a lock file carrying `{pid, ts}`, stale-aware for the
  same reason the backfill lock had to become stale-aware.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_PATH = ROOT / ".runtime" / "event_store.sqlite3"
LOCK_PATH = DB_PATH.with_suffix(".sqlite3.lock")
LOCK_TTL_SEC = 30 * 60
SCHEMA_VERSION = 1
DEFAULT_SOURCES = ("bot_events.jsonl",)
COMMIT_EVERY = 20_000


# ── lock ────────────────────────────────────────────────────────────────────

def _owner_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=10).stdout.lower()
        return "python" in out
    except Exception:
        return True          # cannot tell -> never evict a working sync


@contextmanager
def _single_writer() -> Iterator[None]:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "ts": time.time()})
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, payload.encode())
        os.close(fd)
    except FileExistsError:
        try:
            info = json.loads(LOCK_PATH.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            info = {}
        age = time.time() - LOCK_PATH.stat().st_mtime if LOCK_PATH.exists() else 0
        if age <= LOCK_TTL_SEC and _owner_alive(int(info.get("pid") or 0)):
            raise RuntimeError(f"another sync holds the lock (pid={info.get('pid')})")
        LOCK_PATH.write_text(payload, encoding="utf-8")
    try:
        yield
    finally:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass


# ── schema ──────────────────────────────────────────────────────────────────

def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA user_version").fetchone()
    version = int(row[0]) if row else 0
    if version and version != SCHEMA_VERSION:
        # A half-applied migration is worse than a rebuild: the store is a
        # derived artifact and the journal is still the source of truth.
        conn.executescript("DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS source_state;")
        version = 0
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS source_state (
            source_file TEXT PRIMARY KEY,
            byte_offset INTEGER NOT NULL,
            source_size INTEGER NOT NULL,
            rows_ingested INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            source_file TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            ts TEXT,
            day TEXT,
            event TEXT,
            sym TEXT,
            tf TEXT,
            mode TEXT,
            reason_code TEXT,
            pnl_pct REAL,
            payload TEXT NOT NULL,
            PRIMARY KEY (source_file, byte_offset)
        );
        CREATE INDEX IF NOT EXISTS idx_events_day_event ON events(day, event);
        CREATE INDEX IF NOT EXISTS idx_events_sym_day ON events(sym, day);
        CREATE INDEX IF NOT EXISTS idx_events_reason ON events(event, reason_code);
    """)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


# ── ingest ──────────────────────────────────────────────────────────────────

def _row_from(rec: dict, source: str, offset: int) -> tuple:
    ts = rec.get("ts") or ""
    day = str(ts)[:10] if ts else None
    reason = rec.get("reason_code")
    if not reason:
        dec = rec.get("decision")
        if isinstance(dec, dict):
            reason = dec.get("reason_code")
    if not reason:
        reason = rec.get("reason")
    pnl = rec.get("pnl_pct")
    return (
        source, offset, ts or None, day,
        rec.get("event"), rec.get("sym"), rec.get("tf"), rec.get("mode"),
        str(reason)[:400] if reason else None,
        float(pnl) if isinstance(pnl, (int, float)) else None,
        json.dumps(rec, ensure_ascii=False),
    )


def sync_source(conn: sqlite3.Connection, path: Path) -> dict:
    source = path.name
    if not path.exists():
        return {"source": source, "status": "missing"}
    size = path.stat().st_size

    state = conn.execute(
        "SELECT byte_offset, source_size FROM source_state WHERE source_file=?",
        (source,)).fetchone()
    offset = int(state[0]) if state else 0
    reset = bool(state and offset > size)
    if reset:
        # Shrunk since last time: rotated or truncated. Resuming from the old
        # offset would splice two different files into one table.
        conn.execute("DELETE FROM events WHERE source_file=?", (source,))
        offset = 0

    if state and offset == size:
        return {"source": source, "status": "up_to_date", "byte_offset": offset,
                "new_rows": 0, "reset": reset}

    new_rows = bad = 0
    batch: list[tuple] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        while True:
            line_offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                # Partial trailing line: a writer is mid-append. Stop before it
                # and pick it up next sync rather than storing half a record.
                break
            offset = fh.tell()
            text = raw.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad += 1
                continue
            if not isinstance(rec, dict):
                bad += 1
                continue
            batch.append(_row_from(rec, source, line_offset))
            new_rows += 1
            if len(batch) >= COMMIT_EVERY:
                _flush(conn, source, batch, offset, size, new_rows)
                batch.clear()
    _flush(conn, source, batch, offset, size, new_rows)
    return {"source": source, "status": "synced", "byte_offset": offset,
            "new_rows": new_rows, "malformed": bad, "reset": reset}


def _flush(conn: sqlite3.Connection, source: str, batch: list[tuple],
           offset: int, size: int, total: int) -> None:
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO events (source_file, byte_offset, ts, day, event,"
            " sym, tf, mode, reason_code, pnl_pct, payload)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
    conn.execute(
        "INSERT INTO source_state (source_file, byte_offset, source_size, rows_ingested,"
        " updated_at) VALUES (?,?,?,?,?)"
        " ON CONFLICT(source_file) DO UPDATE SET byte_offset=excluded.byte_offset,"
        " source_size=excluded.source_size, rows_ingested=excluded.rows_ingested,"
        " updated_at=excluded.updated_at",
        (source, offset, size, total,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def sync(sources: tuple[str, ...] = DEFAULT_SOURCES, db_path: Path = DB_PATH,
         files_dir: Path = HERE) -> dict:
    started = time.time()
    with _single_writer():
        conn = _connect(db_path)
        try:
            _ensure_schema(conn)
            results = [sync_source(conn, files_dir / name) for name in sources]
        finally:
            conn.close()
    return {"sources": results, "elapsed_sec": round(time.time() - started, 2)}


# ── queries ─────────────────────────────────────────────────────────────────

def stats(db_path: Path = DB_PATH) -> dict:
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_event = dict(conn.execute(
            "SELECT event, COUNT(*) FROM events GROUP BY event ORDER BY 2 DESC").fetchall())
        span = conn.execute("SELECT MIN(day), MAX(day) FROM events WHERE day IS NOT NULL").fetchone()
        state = [dict(zip(("source", "byte_offset", "size", "rows", "updated_at"), r))
                 for r in conn.execute(
                     "SELECT source_file, byte_offset, source_size, rows_ingested,"
                     " updated_at FROM source_state").fetchall()]
        return {"total_events": total, "by_event": by_event,
                "day_span": {"first": span[0], "last": span[1]}, "sources": state,
                "db_mb": round(db_path.stat().st_size / 1e6, 1) if db_path.exists() else 0.0}
    finally:
        conn.close()


def blocked_reason_counts(days: int = 14, db_path: Path = DB_PATH) -> list[tuple[str, int]]:
    """Raw block reasons over the window — normalise with `block_reasons`."""
    conn = _connect(db_path)
    try:
        cut = conn.execute(
            "SELECT date(MAX(day), ?) FROM events", (f"-{days} day",)).fetchone()[0]
        return conn.execute(
            "SELECT reason_code, COUNT(*) FROM events"
            " WHERE event='blocked' AND day >= ? GROUP BY reason_code ORDER BY 2 DESC",
            (cut,)).fetchall()
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="incremental JSONL -> SQLite event store")
    ap.add_argument("command", choices=("sync", "stats"), nargs="?", default="sync")
    args = ap.parse_args(argv)
    if args.command == "sync":
        res = sync()
        for r in res["sources"]:
            print(f"  {r['source']:<24}{r['status']:<12}"
                  f"+{r.get('new_rows', 0)} rows  offset={r.get('byte_offset', 0)}"
                  + ("  [RESET: source shrank]" if r.get("reset") else ""))
        print(f"elapsed {res['elapsed_sec']}s")
    else:
        print(json.dumps(stats(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
