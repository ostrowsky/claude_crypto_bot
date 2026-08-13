"""Roadmap gate for G1: the SQLite mirror must answer exactly like the JSONL.

A store that is fast and subtly different is worse than no store, so this
compares real aggregates computed both ways over the same window and times both
paths. Any mismatch means the sync lost or mangled rows.

Read-only.  pyembed\\python.exe files\\_verify_event_store_parity.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import event_store as ES  # noqa: E402
from block_reasons import normalize_block_reason  # noqa: E402

DAYS = 14
CUT = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime("%Y-%m-%d")


def from_jsonl() -> tuple[Counter, Counter, float]:
    t0 = time.time()
    per_event: Counter[str] = Counter()
    per_reason: Counter[str] = Counter()
    with io.open(HERE / "bot_events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            day = str(rec.get("ts") or "")[:10]
            if not day or day < CUT:
                continue
            ev = str(rec.get("event") or "")
            per_event[ev] += 1
            if ev == "blocked":
                # Mirror the store's field precedence exactly. Comparing the
                # store's `reason_code` against this side's `reason` would test
                # two different field choices rather than the sync itself —
                # which is how the first run reported 7 517 phantom mismatches.
                reason = rec.get("reason_code")
                if not reason:
                    dec = rec.get("decision")
                    if isinstance(dec, dict):
                        reason = dec.get("reason_code")
                if not reason:
                    reason = rec.get("reason")
                per_reason[normalize_block_reason(reason or "",
                                                  rec.get("signal_type", ""))] += 1
    return per_event, per_reason, time.time() - t0


def from_sqlite() -> tuple[Counter, Counter, float]:
    t0 = time.time()
    conn = ES._connect()
    try:
        per_event = Counter(dict(conn.execute(
            "SELECT event, COUNT(*) FROM events WHERE day >= ? GROUP BY event",
            (CUT,)).fetchall()))
        per_reason: Counter[str] = Counter()
        for reason, n in conn.execute(
                "SELECT reason_code, COUNT(*) FROM events"
                " WHERE event='blocked' AND day >= ? GROUP BY reason_code", (CUT,)):
            per_reason[normalize_block_reason(reason or "")] += n
    finally:
        conn.close()
    return per_event, per_reason, time.time() - t0


def main() -> int:
    ES.sync()
    j_ev, j_rs, j_t = from_jsonl()
    s_ev, s_rs, s_t = from_sqlite()

    print("=" * 74)
    print(f"Паритет хранилища событий · окно {DAYS} дней с {CUT}")
    print("=" * 74)
    print(f"  JSONL  : {sum(j_ev.values()):>8} событий за {j_t:.2f}с")
    print(f"  SQLite : {sum(s_ev.values()):>8} событий за {s_t:.2f}с"
          f"   ({j_t / max(s_t, 1e-6):.0f}× быстрее)")

    ok = True
    for name, a, b in (("по типам событий", j_ev, s_ev),
                       ("по причинам блокировок", j_rs, s_rs)):
        diff = {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)
                if a.get(k, 0) != b.get(k, 0)}
        if diff:
            ok = False
            print(f"\nРАСХОЖДЕНИЕ {name}: {len(diff)}")
            for k, (x, y) in sorted(diff.items())[:10]:
                print(f"    {k:<28} jsonl={x} sqlite={y}")
        else:
            print(f"  совпадает {name}: {len(a)} ключей")

    print()
    if ok:
        print("ВЕРДИКТ: агрегаты идентичны, хранилище можно использовать для анализа.")
    else:
        print("ВЕРДИКТ: расхождение — синхронизация теряет или искажает строки.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
