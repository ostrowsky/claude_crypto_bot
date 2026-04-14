"""
backfill_labels.py — Заполняет пропущенные метки в ml_dataset.jsonl.

Запускать вручную:
    python backfill_labels.py

Что делает:
  1. Читает ml_dataset.jsonl
  2. Удаляет "мусорные" записи из 2024/2025 (созданные старым кодом analyze_coin
     который брал бары из большого LIVE_LIMIT окна — не последний бар)
  3. Для каждой записи с пустыми ret_3/ret_5/ret_10:
     - Скачивает исторические данные вокруг bar_ts
     - Находит цены на T+3, T+5, T+10 барах
     - Записывает ret_pct и label (bool)
  4. Перезаписывает файл

Прогресс выводится в консоль. Безопасно прерывать — повторный запуск
пропустит уже заполненные записи.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import aiohttp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("backfill")

ROOT         = Path(__file__).resolve().parent
ML_FILE      = ROOT / "ml_dataset.jsonl"
# Минимальная дата: данные собранные до запуска data_collector — мусор
# Оставляем только записи начиная с даты когда data_collector был добавлен
MIN_DATE_STR = "2026-03-04"
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
BATCH_SIZE   = 8    # параллельных запросов
SLEEP_BATCH  = 0.4  # секунды между батчами


def _read_valid_jsonl(path: Path, *, required_keys: tuple[str, ...] = ()) -> tuple[list[dict], int]:
    """Потоково читает JSONL и пропускает битые или неполные строки."""
    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
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
            if required_keys and any(key not in rec for key in required_keys):
                skipped += 1
                continue
            if "labels" in required_keys and not isinstance(rec.get("labels"), dict):
                skipped += 1
                continue
            records.append(rec)
    return records, skipped


# ── Загрузка исторических баров ────────────────────────────────────────────────

async def fetch_bars_around(
    session:  aiohttp.ClientSession,
    sym:      str,
    tf:       str,
    start_ts: int,   # unix ms — начало окна (= bar_ts)
    n_bars:   int,   # сколько баров загрузить после start_ts
) -> dict | None:
    """
    Загружает n_bars баров начиная с start_ts.
    Возвращает {"t": array, "c": array} или None при ошибке.
    """
    tf_map = {"15m": "15m", "1h": "1h", "4h": "4h"}
    interval = tf_map.get(tf, "15m")
    params = {
        "symbol":    sym,
        "interval":  interval,
        "startTime": start_ts,
        "limit":     n_bars + 5,  # с запасом
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
    except Exception as e:
        log.debug("fetch_bars_around %s %s: %s", sym, tf, e)
        return None


# ── Заполнение меток одной записи ─────────────────────────────────────────────

async def fill_one(
    session: aiohttp.ClientSession,
    rec:     dict,
) -> dict:
    """
    Заполняет ret_3/ret_5/ret_10 для одной записи.
    Возвращает обновлённый record.
    """
    sym    = rec["sym"]
    tf     = rec["tf"]
    bar_ts = rec["bar_ts"]
    lab    = rec["labels"]

    # Если все три уже заполнены — пропускаем
    if all(lab.get(f"ret_{h}") is not None for h in (3, 5, 10)):
        return rec

    bar_ms = 15 * 60 * 1000 if tf == "15m" else 60 * 60 * 1000

    # Скачиваем 15 баров начиная с bar_ts (достаточно для T+10 + запас)
    data = await fetch_bars_around(session, sym, tf, bar_ts, n_bars=15)
    if data is None:
        return rec

    t_arr = data["t"]
    c_arr = data["c"]

    # Находим индекс bar_ts в t_arr
    idx0 = np.where(t_arr == bar_ts)[0]
    if len(idx0) == 0:
        # Пробуем первый бар >= bar_ts
        idx0 = np.where(t_arr >= bar_ts)[0]
    if len(idx0) == 0:
        return rec
    i0 = int(idx0[0])

    entry_close = float(c_arr[i0])
    if entry_close <= 0:
        return rec

    for h in (3, 5, 10):
        if lab.get(f"ret_{h}") is not None:
            continue  # уже заполнено
        target_ts  = bar_ts + h * bar_ms
        fut_idx    = np.where(t_arr >= target_ts)[0]
        if len(fut_idx) == 0:
            continue  # данных нет (бар ещё не закрылся)
        future_close = float(c_arr[fut_idx[0]])
        ret_pct = (future_close / entry_close - 1) * 100
        lab[f"ret_{h}"]   = round(ret_pct, 4)
        lab[f"label_{h}"] = ret_pct > 0

    rec["labels"] = lab
    return rec


# ── Чистка мусорных записей ────────────────────────────────────────────────────

def clean_records(records: list) -> tuple[list, int]:
    """
    Удаляет записи созданные старым кодом (до запуска data_collector):
      - bar_ts из 2024 или 2025 — явный артефакт старого кода
      - дубликаты (sym+tf+bar_ts)

    Возвращает (очищенные записи, количество удалённых).
    """
    min_ts = int(
        datetime.strptime(MIN_DATE_STR, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    seen    = set()
    cleaned = []
    removed = 0
    for rec in records:
        bar_ts = rec.get("bar_ts", 0)
        key    = (rec["sym"], rec["tf"], bar_ts)
        if bar_ts < min_ts:
            log.debug("  Remove old record: %s %s %s", rec["sym"], rec["tf"], rec["ts_signal"])
            removed += 1
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        cleaned.append(rec)
    return cleaned, removed


# ── Основная логика ────────────────────────────────────────────────────────────

async def run():
    if not ML_FILE.exists():
        log.error("Файл %s не найден", ML_FILE)
        return

    # Читаем
    records, skipped_rows = _read_valid_jsonl(
        ML_FILE,
        required_keys=("id", "sym", "tf", "bar_ts", "labels"),
    )
    if skipped_rows:
        log.warning("Пропущено %d битых/неполных строк из ml_dataset.jsonl", skipped_rows)
    log.info("Загружено записей: %d", len(records))

    # Чистим
    records, removed = clean_records(records)
    log.info("Удалено мусорных/дублей: %d → осталось: %d", removed, len(records))

    # Статистика меток
    need_fill = [r for r in records
                 if any(r["labels"].get(f"ret_{h}") is None for h in (3, 5, 10))]
    log.info("Нужно заполнить метки: %d / %d", len(need_fill), len(records))

    if not need_fill:
        log.info("Все метки уже заполнены!")
        _write(records)
        return

    # Заполняем батчами
    filled = 0
    skipped = 0
    t0 = time.time()

    # O(1) lookup: предварительно строим индекс id → позиция в records
    records_index: dict = {r["id"]: idx for idx, r in enumerate(records)}

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(need_fill), BATCH_SIZE):
            batch = need_fill[batch_start: batch_start + BATCH_SIZE]
            tasks = [fill_one(session, rec) for rec in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for orig, result in zip(batch, results):
                if isinstance(result, Exception):
                    skipped += 1
                else:
                    lab = result["labels"]
                    if lab.get("ret_3") is not None:
                        filled += 1
                        # O(1) обновление через индекс
                        idx = records_index.get(result["id"])
                        if idx is not None:
                            records[idx] = result
                    else:
                        skipped += 1

            elapsed = time.time() - t0
            pct = (batch_start + len(batch)) / len(need_fill) * 100
            log.info(
                "  [%5.1f%%] заполнено %d, пропущено %d, %.1fs",
                pct, filled, skipped, elapsed,
            )

            if batch_start + BATCH_SIZE < len(need_fill):
                await asyncio.sleep(SLEEP_BATCH)

    # Финальная статистика
    labeled_3  = sum(1 for r in records if r["labels"].get("ret_3")  is not None)
    labeled_10 = sum(1 for r in records if r["labels"].get("ret_10") is not None)
    positive_3  = sum(1 for r in records if r["labels"].get("label_3")  is True)
    positive_10 = sum(1 for r in records if r["labels"].get("label_10") is True)

    log.info("=== ИТОГ ===")
    log.info("Записей после чистки: %d", len(records))
    log.info("ret_3  заполнено: %d/%d (%.1f%%)", labeled_3, len(records), labeled_3/len(records)*100)
    log.info("ret_10 заполнено: %d/%d (%.1f%%)", labeled_10, len(records), labeled_10/len(records)*100)
    if labeled_3 > 0:
        log.info("Баланс label_3:  %.1f%% положительных", positive_3/labeled_3*100)
    if labeled_10 > 0:
        log.info("Баланс label_10: %.1f%% положительных", positive_10/labeled_10*100)

    signal_dist = Counter(r["signal_type"] for r in records)
    log.info("Типы сигналов: %s", dict(signal_dist))

    _write(records)
    log.info("Файл перезаписан: %s (%d KB)",
             ML_FILE, ML_FILE.stat().st_size // 1024)


def _write(records: list) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    ML_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ML_FILE.with_suffix(ML_FILE.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(ML_FILE)


if __name__ == "__main__":
    asyncio.run(run())
