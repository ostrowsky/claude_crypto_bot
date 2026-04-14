"""
backfill_critic_labels.py — Заполняет ret_3/ret_5/ret_10 для заблокированных
и необработанных строк в critic_dataset.jsonl.

Запускать вручную:
    python backfill_critic_labels.py

Зачем:
    Метки ret_3/5/10 в critic_dataset пишутся только для ВЗЯТЫХ позиций
    (через fill_forward_label при закрытии). Заблокированные строки (~4 300)
    остаются без меток — модель обучается на 43% датасета вместо 100%.
    Этот скрипт загружает исторические OHLCV с Binance и дозаполняет метки,
    увеличивая обучающую выборку вдвое.

Безопасно прерывать — повторный запуск пропустит уже заполненные строки.
Запись атомарная: файл переписывается целиком после обработки всех батчей.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("backfill_critic")

ROOT         = Path(__file__).resolve().parent
CRITIC_FILE  = ROOT / "critic_dataset.jsonl"
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
BATCH_SIZE   = 10    # параллельных запросов к Binance
SLEEP_BATCH  = 0.35  # секунды между батчами (rate-limit)
HORIZONS     = (3, 5, 10)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_records(path: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue
            records.append(rec)
    return records, skipped


def _write_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".backfill.tmp")
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


# ── Binance fetch ──────────────────────────────────────────────────────────────

_TF_MAP = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
_BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000,
           "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}


async def _fetch_klines(
    session: aiohttp.ClientSession,
    sym: str,
    tf: str,
    start_ms: int,
    limit: int = 20,
) -> dict | None:
    interval = _TF_MAP.get(tf, "15m")
    params = {
        "symbol":    sym,
        "interval":  interval,
        "startTime": start_ms,
        "limit":     limit,
    }
    try:
        async with session.get(
            BINANCE_BASE, params=params, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                return None
            raw = await resp.json()
            if not raw:
                return None
            t = np.array([int(k[0]) for k in raw], dtype=np.int64)
            c = np.array([float(k[4]) for k in raw], dtype=np.float64)
            return {"t": t, "c": c}
    except Exception as exc:
        log.debug("fetch_klines %s %s: %s", sym, tf, exc)
        return None


# ── Per-row fill ───────────────────────────────────────────────────────────────

async def _fill_one(
    session: aiohttp.ClientSession,
    rec: dict,
) -> dict:
    """Fill ret_3/5/10 for one record. Returns updated record."""
    lab = rec.setdefault("labels", {})

    # Skip if all horizons already filled
    if all(lab.get(f"ret_{h}") is not None for h in HORIZONS):
        return rec

    sym    = rec.get("sym", "")
    tf     = rec.get("tf", "15m")
    bar_ts = rec.get("bar_ts")  # unix ms — close of signal bar = hypothetical entry

    if not sym or bar_ts is None:
        return rec

    bar_ms = _BAR_MS.get(tf, 15 * 60 * 1000)
    # Fetch enough bars: bar_ts + max(HORIZONS) + 2 extra
    max_h = max(HORIZONS)
    data = await _fetch_klines(session, sym, tf, int(bar_ts), limit=max_h + 5)
    if data is None:
        return rec

    t_arr = data["t"]
    c_arr = data["c"]

    # Entry price = close of bar at bar_ts
    idx0 = np.where(t_arr == int(bar_ts))[0]
    if len(idx0) == 0:
        idx0 = np.where(t_arr >= int(bar_ts))[0]
    if len(idx0) == 0:
        return rec

    entry_close = float(c_arr[idx0[0]])
    if entry_close <= 0:
        return rec

    for h in HORIZONS:
        if lab.get(f"ret_{h}") is not None:
            continue
        future_ts = int(bar_ts) + h * bar_ms
        fut_idx = np.where(t_arr >= future_ts)[0]
        if len(fut_idx) == 0:
            continue
        future_close = float(c_arr[fut_idx[0]])
        ret_pct = (future_close / entry_close - 1) * 100
        lab[f"ret_{h}"]   = round(ret_pct, 4)
        lab[f"label_{h}"] = ret_pct > 0

    rec["labels"] = lab
    return rec


# ── Batch runner ───────────────────────────────────────────────────────────────

async def run(dry_run: bool = False) -> None:
    if not CRITIC_FILE.exists():
        log.error("critic_dataset.jsonl not found: %s", CRITIC_FILE)
        return

    records, skipped = _read_records(CRITIC_FILE)
    if skipped:
        log.warning("Skipped %d malformed lines", skipped)
    log.info("Loaded %d records", len(records))

    # Collect records that need labels
    need_fill = [
        (i, r) for i, r in enumerate(records)
        if any(r.get("labels", {}).get(f"ret_{h}") is None for h in HORIZONS)
    ]
    log.info("Need label fill: %d / %d (%.1f%%)",
             len(need_fill), len(records),
             len(need_fill) / max(len(records), 1) * 100)

    # Stats on what we're filling
    by_action = defaultdict(int)
    for _, r in need_fill:
        by_action[r.get("decision", {}).get("action", "unknown")] += 1
    log.info("By action: %s", dict(by_action))

    if not need_fill:
        log.info("Nothing to fill.")
        return

    if dry_run:
        log.info("Dry-run mode — not writing.")
        return

    filled = 0
    failed = 0
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(need_fill), BATCH_SIZE):
            batch = need_fill[batch_start: batch_start + BATCH_SIZE]
            tasks = [_fill_one(session, records[i]) for i, _ in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (orig_idx, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                else:
                    lab = result.get("labels", {})
                    if any(lab.get(f"ret_{h}") is not None for h in HORIZONS):
                        filled += 1
                        records[orig_idx] = result
                    else:
                        failed += 1

            done = batch_start + len(batch)
            pct = done / len(need_fill) * 100
            elapsed = time.time() - t0
            log.info("[%5.1f%%] filled=%d  failed=%d  %.1fs",
                     pct, filled, failed, elapsed)

            if batch_start + BATCH_SIZE < len(need_fill):
                await asyncio.sleep(SLEEP_BATCH)

    # Final stats
    total = len(records)
    labeled = {h: sum(1 for r in records if r.get("labels", {}).get(f"ret_{h}") is not None)
               for h in HORIZONS}
    log.info("=== DONE ===")
    log.info("filled this run: %d  failed: %d", filled, failed)
    for h in HORIZONS:
        n = labeled[h]
        log.info("ret_%d labeled: %d/%d (%.1f%%)", h, n, total, n / max(total, 1) * 100)

    _write_records(records, CRITIC_FILE)
    log.info("Written: %s (%d KB)", CRITIC_FILE, CRITIC_FILE.stat().st_size // 1024)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(run(dry_run=dry))
