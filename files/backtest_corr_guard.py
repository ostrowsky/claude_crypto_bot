"""
backtest_corr_guard.py — Бэктест Correlation Guard на исторических данных.

Симулирует работу guard на реальных сигналах из critic_dataset.jsonl:
  - Берёт события action="take" (реальные входы) за последние N дней.
  - Группирует по bar_ts: все монеты открытые одновременно = «портфель».
  - На каждом шаге применяет correlation check: блокирует лишних, оставляет top-2 по кластеру.
  - Сравнивает PnL: BASELINE (все сигналы) vs. GUARD (отфильтрованные).

Метрики:
  - Общее количество сделок (baseline vs guard)
  - Средний ret_5 (5-барный форвард-ретёрн)
  - Win rate (ret_5 > 0)
  - Среднее количество одновременных позиций
  - Снижение max concurrent correlated cluster size
  - Сделки заблокированные guard с положительным ret_5 (false positives)
  - Сделки пропущенные guard с отрицательным ret_5 (correct blocks)

Запуск: python backtest_corr_guard.py [--days 14] [--threshold 0.65] [--max-cluster 2]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

# ── Пути ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_CRITIC = _ROOT / "critic_dataset.jsonl"
_BINANCE = "https://fapi.binance.com"
_BINANCE_SPOT = "https://api.binance.com"

# ── Ядро корреляций (дублируем из correlation_guard без зависимостей) ─────────

def _log_returns(closes: List[float]) -> List[float]:
    result = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            result.append(math.log(closes[i] / closes[i - 1]))
        else:
            result.append(0.0)
    return result


def _pearson(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    ax, bx = a[-n:], b[-n:]
    ma, mb = sum(ax) / n, sum(bx) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ax) / n)
    sb = math.sqrt(sum((x - mb) ** 2 for x in bx) / n)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    cov = sum((ax[i] - ma) * (bx[i] - mb) for i in range(n)) / n
    return max(-1.0, min(1.0, cov / (sa * sb)))


def fetch_closes(sym: str, interval: str = "1h", limit: int = 48) -> Optional[List[float]]:
    for base, path in [(_BINANCE, "/fapi/v1/klines"), (_BINANCE_SPOT, "/api/v3/klines")]:
        url = f"{base}{path}?symbol={sym}&interval={interval}&limit={limit}"
        try:
            r = urllib.request.urlopen(url, timeout=8)
            data = json.loads(r.read())
            if data and isinstance(data, list):
                return [float(k[4]) for k in data]
        except Exception:
            pass
    return None


def build_rho_matrix(syms: List[str], interval: str = "1h", limit: int = 48) -> Dict[Tuple[str,str], float]:
    closes = {}
    for s in syms:
        c = fetch_closes(s, interval, limit)
        if c and len(c) >= 10:
            closes[s] = _log_returns(c)
        time.sleep(0.05)

    matrix: Dict[Tuple[str,str], float] = {}
    loaded = list(closes.keys())
    for i, s1 in enumerate(loaded):
        for s2 in loaded[i+1:]:
            rho = _pearson(closes[s1], closes[s2])
            matrix[(min(s1,s2), max(s1,s2))] = rho
    return matrix


def get_rho(matrix: Dict, a: str, b: str) -> float:
    if a == b:
        return 1.0
    return matrix.get((min(a,b), max(a,b)), 0.0)


def cluster_size_for(sym: str, portfolio: List[str], matrix: Dict, threshold: float) -> int:
    """Размер кластера sym среди portfolio (сколько уже открытых коррелированы с sym)."""
    return sum(1 for s in portfolio if s != sym and get_rho(matrix, sym, s) >= threshold)


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load_critic_entries(days: int = 14) -> List[dict]:
    """Загружает action='take' записи за последние N дней."""
    if not _CRITIC.exists():
        print(f"ERROR: {_CRITIC} not found", file=sys.stderr)
        sys.exit(1)

    cutoff_ms = (int(time.time()) - days * 86400) * 1000
    rows = []
    with _CRITIC.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            # action может быть на верхнем уровне или внутри decision
            decision = r.get("decision") or {}
            action = r.get("action") or decision.get("action") or ""
            if action != "take":
                continue
            bar_ts = r.get("bar_ts") or r.get("ts") or 0
            if isinstance(bar_ts, str):
                try:
                    from datetime import datetime, timezone
                    bar_ts = int(datetime.fromisoformat(bar_ts.replace("Z", "+00:00")).timestamp() * 1000)
                except Exception:
                    bar_ts = 0
            if bar_ts < cutoff_ms:
                continue
            # labels содержит ret_5, trade_exit_pnl
            labels = r.get("labels") or {}
            ret_5 = labels.get("ret_5") or r.get("ret_5") or 0.0
            pnl   = labels.get("trade_exit_pnl") or r.get("trade_exit_pnl")
            # ranker_final из decision.features или верхнего уровня
            ranker = (decision.get("ranker_final_score")
                      or r.get("ranker_final_score")
                      or r.get("final_score")
                      or 0.0)
            rows.append({
                "sym": r.get("symbol") or r.get("sym", ""),
                "tf": r.get("tf", ""),
                "bar_ts": int(bar_ts),
                "ret_5": float(ret_5),
                "ranker_final": float(ranker),
                "signal_type": r.get("signal_type") or decision.get("signal_type") or "",
                "trade_exit_pnl": pnl,
            })
    return rows


# ── Симуляция ─────────────────────────────────────────────────────────────────

def simulate(
    entries: List[dict],
    matrix: Dict,
    threshold: float = 0.65,
    max_cluster: int = 2,
    max_positions: int = 10,
) -> dict:
    """
    Симулирует работу correlation guard.

    Алгоритм: сортируем входы по bar_ts; при каждом новом входе
    проверяем guard против текущего «портфеля» (список принятых).
    Позиция держится пока bars_elapsed < max_hold (упрощение: 8 баров 1h = 8ч).
    """
    MAX_HOLD_BARS = {"1h": 8, "15m": 24}
    BAR_MS = {"1h": 3600_000, "15m": 900_000}

    # Отсортированы по времени
    entries_sorted = sorted(entries, key=lambda x: x["bar_ts"])

    # BASELINE: все входы
    baseline_taken = list(entries_sorted)

    # GUARD: фильтруем через correlation check
    guard_taken: List[dict] = []
    guard_blocked: List[dict] = []

    open_portfolio: List[dict] = []  # текущие открытые позиции

    for entry in entries_sorted:
        sym = entry["sym"]
        ts  = entry["bar_ts"]
        tf  = entry.get("tf", "1h")
        bar_ms = BAR_MS.get(tf, 3600_000)
        max_hold = MAX_HOLD_BARS.get(tf, 8)

        # Убираем истёкшие позиции
        open_portfolio = [
            p for p in open_portfolio
            if ts - p["bar_ts"] < max_hold * bar_ms
        ]

        # Убираем позиции по той же монете (уже была открыта)
        open_portfolio = [p for p in open_portfolio if p["sym"] != sym]

        # Correlation check
        open_syms = [p["sym"] for p in open_portfolio]
        corr_count = cluster_size_for(sym, open_syms, matrix, threshold)

        if corr_count >= max_cluster:
            entry["_guard_block_reason"] = f"cluster {corr_count}/{max_cluster}"
            guard_blocked.append(entry)
        elif len(open_portfolio) >= max_positions:
            # Portfolio full (same as baseline limitation)
            guard_blocked.append(entry)
        else:
            guard_taken.append(entry)
            open_portfolio.append(entry)

    return {
        "baseline": baseline_taken,
        "guard_taken": guard_taken,
        "guard_blocked": guard_blocked,
    }


# ── Статистика ────────────────────────────────────────────────────────────────

def stats(entries: List[dict], label: str) -> dict:
    if not entries:
        return {"label": label, "n": 0}
    rets = [e["ret_5"] for e in entries if e["ret_5"] != 0.0]
    pnls = [float(e["trade_exit_pnl"]) for e in entries if e.get("trade_exit_pnl") is not None]
    n = len(entries)
    win = sum(1 for r in rets if r > 0)
    return {
        "label": label,
        "n": n,
        "avg_ret5": round(mean(rets), 4) if rets else None,
        "std_ret5": round(stdev(rets), 4) if len(rets) > 1 else None,
        "win_rate": round(win / len(rets), 3) if rets else None,
        "avg_pnl":  round(mean(pnls), 4) if pnls else None,
        "n_with_ret5": len(rets),
    }


def print_report(result: dict, threshold: float, max_cluster: int) -> None:
    baseline = result["baseline"]
    taken    = result["guard_taken"]
    blocked  = result["guard_blocked"]

    b_stat = stats(baseline, "BASELINE")
    g_stat = stats(taken,    "GUARD")

    print("\n" + "="*60)
    print(f"CORRELATION GUARD BACKTEST  (threshold={threshold}, max_cluster={max_cluster})")
    print("="*60)

    for s in (b_stat, g_stat):
        print(f"\n  [{s['label']}]")
        print(f"    Сделок:        {s['n']}")
        print(f"    с ret_5:       {s.get('n_with_ret5', 0)}")
        print(f"    avg ret_5:     {s.get('avg_ret5', 'n/a')}")
        print(f"    win_rate:      {s.get('win_rate', 'n/a')}")
        print(f"    avg_pnl:       {s.get('avg_pnl', 'n/a')}")

    # Guard эффект
    n_blocked = len(blocked)
    pct_blocked = 100 * n_blocked / len(baseline) if baseline else 0
    print(f"\n  Заблокировано guard: {n_blocked} ({pct_blocked:.1f}%)")

    # False positives (заблокировали выгодную сделку)
    fp = [e for e in blocked if e.get("ret_5", 0) > 0]
    fp_rate = 100 * len(fp) / n_blocked if n_blocked else 0
    print(f"  False positives (блок но ret_5>0): {len(fp)} ({fp_rate:.1f}%)")

    # True blocks (заблокировали убыточную сделку)
    tb = [e for e in blocked if e.get("ret_5", 0) <= 0]
    tb_rate = 100 * len(tb) / n_blocked if n_blocked else 0
    print(f"  Correct blocks  (блок и ret_5<=0): {len(tb)} ({tb_rate:.1f}%)")

    # Avg ret_5 of blocked
    b_rets = [e["ret_5"] for e in blocked if e["ret_5"] != 0.0]
    if b_rets:
        print(f"  avg ret_5 заблокированных: {mean(b_rets):.4f}")

    # Сравнение avg_ret5
    if b_stat.get("avg_ret5") and g_stat.get("avg_ret5"):
        delta = g_stat["avg_ret5"] - b_stat["avg_ret5"]
        print(f"\n  Delta avg_ret5 (guard - baseline): {delta:+.4f}")
        if delta > 0:
            print("  [OK] Guard uluchshil sredniy ret_5")
        else:
            print("  [!!] Guard snizil sredniy ret_5 — porog mozhet byt' slishkom agressivnym")

    # Топ-10 заблокированных монет
    blocked_syms = defaultdict(int)
    for e in blocked:
        blocked_syms[e["sym"]] += 1
    print("\n  Топ-10 заблокированных монет:")
    for sym, cnt in sorted(blocked_syms.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sym:12}: {cnt} блокировок")

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation Guard Backtest")
    parser.add_argument("--days",        type=int,   default=14,   help="Дней истории")
    parser.add_argument("--threshold",   type=float, default=0.65, help="rho порог кластеризации")
    parser.add_argument("--max-cluster", type=int,   default=2,    help="Макс. позиций в кластере")
    parser.add_argument("--max-pos",     type=int,   default=10,   help="Макс. позиций в портфеле")
    parser.add_argument("--interval",    type=str,   default="1h", help="Таймфрейм корреляций")
    parser.add_argument("--window",      type=int,   default=48,   help="Баров для матрицы")
    args = parser.parse_args()

    print(f"Загружаю critic_dataset (последние {args.days} дней)...")
    entries = load_critic_entries(args.days)
    print(f"Найдено {len(entries)} take-записей")

    if not entries:
        print("Нет данных для бэктеста.", file=sys.stderr)
        sys.exit(1)

    # Уникальные символы
    all_syms = list({e["sym"] for e in entries if e["sym"]})
    print(f"Уникальных символов: {len(all_syms)}")
    print(f"Загружаю корреляционную матрицу ({args.interval}, {args.window} баров)...")

    matrix = build_rho_matrix(all_syms, interval=args.interval, limit=args.window)
    print(f"Матрица: {len(matrix)} пар рассчитано")

    # Базовый прогон с заданным threshold
    result = simulate(entries, matrix,
                      threshold=args.threshold,
                      max_cluster=args.max_cluster,
                      max_positions=args.max_pos)
    print_report(result, args.threshold, args.max_cluster)

    # Sweep по разным порогам
    print("\n" + "="*60)
    print("SWEEP по порогам threshold (max_cluster=2):")
    print(f"{'Threshold':>12} {'N_taken':>8} {'N_blocked':>10} {'avg_ret5':>10} {'win_rate':>10}")
    print("-"*55)
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        r = simulate(entries, matrix, threshold=thr, max_cluster=2, max_positions=args.max_pos)
        s = stats(r["guard_taken"], "")
        print(f"  {thr:>10.2f} {s['n']:>8} {len(r['guard_blocked']):>10} "
              f"{str(s.get('avg_ret5','n/a')):>10} {str(s.get('win_rate','n/a')):>10}")

    # Sweep по max_cluster
    print("\nSWEEP по max_cluster (threshold=0.65):")
    print(f"{'max_cluster':>12} {'N_taken':>8} {'N_blocked':>10} {'avg_ret5':>10} {'win_rate':>10}")
    print("-"*55)
    for mc in [1, 2, 3, 4]:
        r = simulate(entries, matrix, threshold=0.65, max_cluster=mc, max_positions=args.max_pos)
        s = stats(r["guard_taken"], "")
        print(f"  {mc:>10} {s['n']:>8} {len(r['guard_blocked']):>10} "
              f"{str(s.get('avg_ret5','n/a')):>10} {str(s.get('win_rate','n/a')):>10}")

    print()


if __name__ == "__main__":
    main()
