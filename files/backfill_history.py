"""
backfill_history.py — заполняет пробелы в ml_dataset.jsonl.

Использование:
    python backfill_history.py

Что делает:
  1. Читает ml_dataset.jsonl, находит все пробелы в хронологии для каждой sym+tf
  2. Для каждого пропущенного бара: скачивает историю с Binance, вычисляет
     фичи, логирует бар (с дедупликацией — дубли не пишет)
  3. После записи всех баров: заполняет метки T+3/T+5/T+10 для тех записей
     где метки ещё null

Безопасно запускать повторно — дубликаты пропускаются по bar_ts.
Можно прерывать Ctrl+C — при следующем запуске продолжит с того места.

Настройки:
  BATCH_SIZE   — монет параллельно (не менять выше 10 — Binance rate limit)
  DELAY        — пауза между батчами (сек)
  LOOKBACK_LIMIT — максимум баров с Binance за один запрос (до 1000)
"""

import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
import ml_dataset
from indicators import compute_features
from strategy import (
    check_entry_conditions,
    check_retest_conditions,
    check_breakout_conditions,
    check_impulse_conditions,
    check_alignment_conditions,
    check_trend_surge_conditions,
    get_entry_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill")

# ── Параметры ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 8      # параллельных запросов к Binance
DELAY          = 0.4    # пауза между батчами (сек)
LOOKBACK_LIMIT = 500    # баров за один klines запрос (макс 1000)
BINANCE_URL    = "https://api.binance.com/api/v3/klines"
BAR_MS         = {
    "15m": 15 * 60 * 1000,
    "1h":  60 * 60 * 1000,
}


def _read_valid_jsonl(path: Path, *, required_keys: tuple[str, ...] = ()) -> tuple[list[dict], int]:
    """
    Stream JSONL rows and skip malformed/incomplete records.
    A single bad line in a large dataset must not abort the whole backfill.
    """
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
            records.append(rec)
    return records, skipped


# ── Шаг 1: анализ пробелов ─────────────────────────────────────────────────────

def find_gaps(
    records: List[dict],
    sym: str,
    tf: str,
) -> List[Tuple[datetime, datetime]]:
    """
    Возвращает список (gap_start, gap_end) — диапазоны пропущенных баров.
    Пробел = отсутствие записи там где должен быть бар по сетке tf.
    """
    bar_min = {"15m": 15, "1h": 60}[tf]

    # Все имеющиеся timestamps для этой монеты
    existing = set()
    for r in records:
        if r["sym"] == sym and r["tf"] == tf:
            existing.add(r["bar_ts"])

    if not existing:
        return []

    # Диапазон: от первого имеющегося бара до "сейчас"
    first_ts_ms = min(existing)
    now_utc     = datetime.now(timezone.utc)
    # Последний закрытый бар
    last_closed = now_utc.replace(second=0, microsecond=0)
    last_closed -= timedelta(minutes=last_closed.minute % bar_min)
    last_ts_ms  = int(last_closed.timestamp() * 1000)

    # Строим ожидаемую сетку
    step_ms = bar_min * 60 * 1000
    gaps: List[Tuple[datetime, datetime]] = []
    gap_start_ms = None

    t = first_ts_ms
    while t <= last_ts_ms:
        if t not in existing:
            if gap_start_ms is None:
                gap_start_ms = t
        else:
            if gap_start_ms is not None:
                # Закрываем пробел
                gaps.append((
                    datetime.fromtimestamp(gap_start_ms / 1000, tz=timezone.utc),
                    datetime.fromtimestamp((t - step_ms) / 1000, tz=timezone.utc),
                ))
                gap_start_ms = None
        t += step_ms

    # Хвост до конца
    if gap_start_ms is not None:
        gaps.append((
            datetime.fromtimestamp(gap_start_ms / 1000, tz=timezone.utc),
            last_closed,
        ))

    return gaps


# ── Шаг 2: загрузка и логирование пропущенных баров ───────────────────────────

def _detect_signal(feat: dict, i: int, data: np.ndarray) -> str:
    try:
        c = data["c"].astype(float)
        brk_ok, _ = check_breakout_conditions(feat, i)
        if brk_ok:
            return "breakout"
        ret_ok, _ = check_retest_conditions(feat, i)
        if ret_ok:
            return "retest"
        buy_ok, _ = check_entry_conditions(feat, i, c)
        if buy_ok:
            return get_entry_mode(feat, i)
        surge_ok, _ = check_trend_surge_conditions(feat, i)
        if surge_ok:
            return "impulse_speed"
        imp_ok, _ = check_impulse_conditions(feat, i)
        if imp_ok:
            return "impulse"
        aln_ok, _ = check_alignment_conditions(feat, i)
        if aln_ok:
            return "alignment"
    except Exception:
        pass
    return "none"


async def fetch_klines_range(
    session: aiohttp.ClientSession,
    sym: str,
    tf: str,
    start_ms: int,
) -> np.ndarray | None:
    """Скачивает LOOKBACK_LIMIT баров начиная с start_ms."""
    try:
        params = {
            "symbol":    sym,
            "interval":  tf,
            "startTime": start_ms,
            "limit":     LOOKBACK_LIMIT,
        }
        async with session.get(
            BINANCE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            r.raise_for_status()
            js = await r.json()
    except Exception as e:
        log.debug("fetch %s %s: %s", sym, tf, e)
        return None

    if not isinstance(js, list) or len(js) < 5:
        return None

    arr = np.zeros(len(js), dtype=[
        ("t", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("v", "f8"),
    ])
    arr["t"] = [int(x[0])   for x in js]
    arr["o"] = [float(x[1]) for x in js]
    arr["h"] = [float(x[2]) for x in js]
    arr["l"] = [float(x[3]) for x in js]
    arr["c"] = [float(x[4]) for x in js]
    arr["v"] = [float(x[5]) for x in js]
    return arr


async def backfill_coin(
    session: aiohttp.ClientSession,
    sym: str,
    tf: str,
    gaps: List[Tuple[datetime, datetime]],
    is_bull_day_cache: Dict[str, bool],
) -> int:
    """
    Для монеты sym+tf: скачивает историю охватывающую все пробелы,
    логирует пропущенные бары. Возвращает количество новых записей.
    """
    if not gaps:
        return 0

    # Нам нужно загрузить данные с запасом для прогрева индикаторов
    earliest_gap = min(g[0] for g in gaps)
    # Берём старт на 300 баров раньше для прогрева EMA/ADX
    warmup_bars = 300
    bar_ms   = BAR_MS[tf]
    start_ms = int(earliest_gap.timestamp() * 1000) - warmup_bars * bar_ms

    data = await fetch_klines_range(session, sym, tf, start_ms)
    if data is None or len(data) < 30:
        return 0

    # Строим set уже существующих timestamps для быстрой проверки
    existing_ts = {
        r["bar_ts"]
        for r in _cached_records
        if r["sym"] == sym and r["tf"] == tf
    }

    feat    = compute_features(data["o"], data["h"], data["l"], data["c"], data["v"])
    c_arr   = data["c"].astype(float)
    t_arr   = data["t"].astype(int)
    written = 0

    # Множество всех пропущенных timestamp
    missing_ts: set[int] = set()
    for gap_start, gap_end in gaps:
        t = int(gap_start.timestamp() * 1000)
        end_t = int(gap_end.timestamp() * 1000)
        while t <= end_t:
            missing_ts.add(t)
            t += bar_ms

    for idx in range(20, len(data) - 1):  # -1: последний бар ещё открыт
        bar_ts = int(t_arr[idx])
        if bar_ts not in missing_ts:
            continue
        if bar_ts in existing_ts:
            continue  # дедупликация

        # Определяем bull_day по дате (из кеша или False)
        date_key = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        is_bull  = is_bull_day_cache.get(date_key, False)

        rule_signal = _detect_signal(feat, idx, data)

        ml_dataset.log_bar_snapshot(
            sym=sym, tf=tf,
            bar_ts=bar_ts,
            rule_signal=rule_signal,
            is_bull_day=is_bull,
            feat=feat, i=idx, data=data,
            btc_vs_ema50=0.0,       # BTC контекст неизвестен для прошлого
            btc_momentum_4h=0.0,
            market_vol_24h=0.0,
        )
        existing_ts.add(bar_ts)
        written += 1

    return written


# ── Шаг 3: заполнение меток ───────────────────────────────────────────────────

async def fill_labels_for_coin(
    session: aiohttp.ClientSession,
    sym: str,
    tf: str,
) -> int:
    """
    Скачивает свежие данные и заполняет незаполненные метки T+3/T+5/T+10.
    Возвращает количество обновлённых записей.
    """
    # Проверяем есть ли записи с пустыми метками
    pending = [
        r for r in _cached_records
        if r["sym"] == sym and r["tf"] == tf
        and r.get("labels", {}).get("ret_3") is None
    ]
    if not pending:
        return 0

    # Достаточно взять последние 1000 баров — перекрывает T+10 для любой записи
    try:
        params = {"symbol": sym, "interval": tf, "limit": 1000}
        async with session.get(
            BINANCE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            r.raise_for_status()
            js = await r.json()
    except Exception as e:
        log.debug("fill_labels fetch %s %s: %s", sym, tf, e)
        return 0

    if not isinstance(js, list) or len(js) < 10:
        return 0

    t_arr = np.array([int(x[0]) for x in js], dtype=np.int64)
    c_arr = np.array([float(x[4]) for x in js], dtype=np.float64)
    bar_ms = BAR_MS[tf]

    ml_dataset.fill_pending_from_data(
        sym=sym, tf=tf,
        t_arr=t_arr, c_arr=c_arr,
        bar_ms=bar_ms,
    )
    return len(pending)


# ── Кеш существующих записей ───────────────────────────────────────────────────
_cached_records: List[dict] = []


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("=" * 60)
    log.info("Backfill History — заполнение пробелов в ml_dataset.jsonl")
    log.info("=" * 60)

    ml_file = ml_dataset.ML_FILE
    if not ml_file.exists():
        log.error("ml_dataset.jsonl не найден. Запустите бота хотя бы раз.")
        return

    # Загружаем существующие записи
    global _cached_records
    _cached_records, skipped_rows = _read_valid_jsonl(
        ml_file,
        required_keys=("sym", "tf", "bar_ts"),
    )
    if skipped_rows:
        log.warning("Пропущено %d битых/неполных строк из ml_dataset.jsonl", skipped_rows)
    # Отфильтровываем мусор из старого кода
    _cached_records = [r for r in _cached_records if r.get("ts_signal", "").startswith("2026")]
    log.info("Загружено %d актуальных записей из датасета", len(_cached_records))

    # Инициализируем ml_dataset дедупликацию из существующих записей
    ml_dataset._logged_bars.clear()
    for r in _cached_records:
        ml_dataset._logged_bars[f"{r['sym']}_{r['tf']}_{r['bar_ts']}"] = True
    log.info("Дедупликация: %d уникальных bar_ts зарегистрировано", len(ml_dataset._logged_bars))

    # Поддельный кеш bull_day (данные за прошлое — ставим False, не критично для ML)
    is_bull_day_cache: Dict[str, bool] = {}
    # Можно частично восстановить из bot_events.jsonl
    events_file = Path("bot_events.jsonl")
    if events_file.exists():
        for line in events_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if e.get("event") == "bull_day":
                    date = e["ts"][:10]
                    is_bull_day_cache[date] = e.get("is_bull", False)
            except Exception:
                pass
        log.info("Bull day кеш: %d дат из bot_events.jsonl", len(is_bull_day_cache))

    # Находим все символы и таймфреймы
    sym_tf_pairs = set(
        (r["sym"], r["tf"]) for r in _cached_records
    )
    # Добавляем весь вотчлист для полноты
    for sym in config.load_watchlist():
        for tf in config.TIMEFRAMES:
            sym_tf_pairs.add((sym, tf))

    log.info("Всего пар sym+tf для проверки: %d", len(sym_tf_pairs))

    # ── Фаза 1: анализ пробелов ───────────────────────────────────────────────
    log.info("")
    log.info("Фаза 1: поиск пробелов...")

    gaps_map: Dict[Tuple[str, str], List] = {}
    total_missing_bars = 0

    for sym, tf in sorted(sym_tf_pairs):
        gaps = find_gaps(_cached_records, sym, tf)
        if gaps:
            n_bars = sum(
                int((end - start).total_seconds() / (15 * 60 if tf == "15m" else 3600)) + 1
                for start, end in gaps
            )
            gaps_map[(sym, tf)] = gaps
            total_missing_bars += n_bars

    log.info("Монет с пробелами: %d / %d", len(gaps_map), len(sym_tf_pairs))
    log.info("Всего пропущенных баров: ~%d", total_missing_bars)
    if total_missing_bars == 0:
        log.info("Пробелов нет — датасет полный!")
    else:
        # Показываем топ пробелов
        for (sym, tf), gaps in list(gaps_map.items())[:3]:
            for gs, ge in gaps[:2]:
                log.info("  %s %s: %s → %s", sym, tf,
                         gs.strftime("%m-%d %H:%M"), ge.strftime("%m-%d %H:%M"))

    # ── Фаза 2: загрузка пропущенных баров ────────────────────────────────────
    if gaps_map:
        log.info("")
        log.info("Фаза 2: загрузка пропущенных баров (~%d баров)...", total_missing_bars)
        log.info("Ожидаемое время: ~%d сек", max(10, total_missing_bars // 50))

        pairs_list = sorted(gaps_map.items())
        total_written = 0

        async with aiohttp.ClientSession() as session:
            # Батчами
            for batch_start in range(0, len(pairs_list), BATCH_SIZE):
                batch = pairs_list[batch_start: batch_start + BATCH_SIZE]
                tasks = [
                    backfill_coin(session, sym, tf, gaps, is_bull_day_cache)
                    for (sym, tf), gaps in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for (sym, tf), res in zip([p for p, _ in batch], results):
                    if isinstance(res, Exception):
                        log.warning("%s %s: ошибка %s", sym, tf, res)
                    else:
                        if res > 0:
                            log.info("  %s %s: +%d баров", sym, tf, res)
                        total_written += res or 0

                progress = batch_start + len(batch)
                log.info("  Прогресс: %d / %d пар", progress, len(pairs_list))
                await asyncio.sleep(DELAY)

        log.info("Фаза 2 завершена: записано %d новых баров", total_written)

    # ── Фаза 3: заполнение меток ───────────────────────────────────────────────
    log.info("")
    log.info("Фаза 3: заполнение меток T+3/T+5/T+10...")

    # Перечитываем файл (могли добавиться новые записи в фазе 2)
    _cached_records, skipped_rows = _read_valid_jsonl(
        ml_dataset.ML_FILE,
        required_keys=("sym", "tf", "bar_ts"),
    )
    if skipped_rows:
        log.warning("Пропущено %d битых/неполных строк при повторной загрузке датасета", skipped_rows)
    _cached_records = [r for r in _cached_records if r.get("ts_signal", "").startswith("2026")]

    pending_count = sum(
        1 for r in _cached_records
        if r.get("labels", {}).get("ret_3") is None
    )
    log.info("Записей без меток T+3: %d / %d", pending_count, len(_cached_records))

    if pending_count == 0:
        log.info("Все метки заполнены!")
    else:
        # Группируем монеты с незаполненными метками
        pending_pairs = set(
            (r["sym"], r["tf"])
            for r in _cached_records
            if r.get("labels", {}).get("ret_3") is None
        )
        log.info("Монет с незаполненными метками: %d", len(pending_pairs))

        total_filled = 0
        pairs_list = sorted(pending_pairs)

        async with aiohttp.ClientSession() as session:
            for batch_start in range(0, len(pairs_list), BATCH_SIZE):
                batch = pairs_list[batch_start: batch_start + BATCH_SIZE]
                tasks = [
                    fill_labels_for_coin(session, sym, tf)
                    for sym, tf in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for (sym, tf), res in zip(batch, results):
                    if isinstance(res, int) and res > 0:
                        log.debug("  %s %s: метки для %d записей", sym, tf, res)
                        total_filled += res
                await asyncio.sleep(DELAY)

        log.info("Фаза 3 завершена: обновлено %d записей", total_filled)

    # ── Итог ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    final_records, skipped_rows = _read_valid_jsonl(
        ml_dataset.ML_FILE,
        required_keys=("sym", "tf", "bar_ts"),
    )
    if skipped_rows:
        log.warning("Пропущено %d битых/неполных строк в финальной статистике", skipped_rows)
    final_2026 = [r for r in final_records if r.get("ts_signal", "").startswith("2026")]
    filled_t3  = sum(1 for r in final_2026 if r.get("labels", {}).get("ret_3") is not None)
    filled_t10 = sum(1 for r in final_2026 if r.get("labels", {}).get("ret_10") is not None)

    log.info("ИТОГ:")
    log.info("  Всего записей (2026): %d", len(final_2026))
    log.info("  T+3  заполнено: %d / %d (%.1f%%)", filled_t3, len(final_2026),
             100 * filled_t3 / max(1, len(final_2026)))
    log.info("  T+10 заполнено: %d / %d (%.1f%%)", filled_t10, len(final_2026),
             100 * filled_t10 / max(1, len(final_2026)))
    log.info("Файл: %s (%.1f MB)", ml_dataset.ML_FILE,
             ml_dataset.ML_FILE.stat().st_size / 1e6)
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
