"""
correlation_guard.py — Correlation Guard для защиты портфеля от коррелированных позиций.

Проблема: в бычий день бот открывает 10 позиций, которые двигаются синхронно
(CRV/AAVE/STRK/MEME/DOT показывают rho=0.55-0.63). При развороте рынка
все они падают одновременно, максимальный drawdown умножается.

Решение:
  1. CorrelationCache — кэшируемая матрица pairwise Pearson rho по 1h-барам.
  2. cluster() — Union-Find кластеризация по порогу rho >= threshold.
  3. check_entry() — блокирует вход если кластер уже заполнен.
  4. marginal_score() — мягкий downgrade дублей через (1 - max_rho).
  5. prune_candidates() — список позиций которые нужно закрыть при разбухании кластеров.

Интеграция в monitor.py: вызвать сразу после clone_signal_guard, до portfolio_limits.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import config

if TYPE_CHECKING:
    import aiohttp

log = logging.getLogger(__name__)

# ── Константы ──────────────────────────────────────────────────────────────────
_BINANCE_FUTURES = "https://fapi.binance.com"
_BINANCE_SPOT    = "https://api.binance.com"


# ── Dataclass результата check_entry ──────────────────────────────────────────

@dataclass
class CorrGuardResult:
    allowed:   bool
    reason:    str = ""
    max_rho:   float = 0.0        # максимальная rho с открытой позицией
    peer_sym:  str = ""           # монета с которой max_rho
    cluster_size: int = 0         # текущий размер кластера


# ── Correlation Cache ─────────────────────────────────────────────────────────

@dataclass
class CorrelationCache:
    """
    Хранит матрицу попарных корреляций log-returns (Pearson rho).
    Сбрасывается по TTL или при изменении набора символов.
    """
    matrix:    Dict[Tuple[str, str], float] = field(default_factory=dict)
    built_at:  float = 0.0   # time.monotonic() момент последнего построения
    syms_key:  str   = ""    # sorted join открытых символов при последнем построении

    def get(self, a: str, b: str) -> Optional[float]:
        if a == b:
            return 1.0
        key = (min(a, b), max(a, b))
        return self.matrix.get(key)

    def set(self, a: str, b: str, rho: float) -> None:
        key = (min(a, b), max(a, b))
        self.matrix[key] = rho

    def is_fresh(self, syms: List[str]) -> bool:
        ttl = int(getattr(config, "CORR_CACHE_TTL_MIN", 15)) * 60
        if time.monotonic() - self.built_at > ttl:
            return False
        key = ",".join(sorted(syms))
        return self.syms_key == key

    def mark_built(self, syms: List[str]) -> None:
        self.built_at = time.monotonic()
        self.syms_key = ",".join(sorted(syms))


# ── Получение log-returns из Binance ─────────────────────────────────────────

async def _fetch_closes_async(
    session: "aiohttp.ClientSession",
    sym: str,
    interval: str = "1h",
    limit: int = 48,
) -> Optional[List[float]]:
    """Возвращает список close-цен или None при ошибке."""
    # Сначала пробуем фьючерсы, потом спот
    for base in (_BINANCE_FUTURES, _BINANCE_SPOT):
        url = f"{base}/fapi/v1/klines" if "fapi" in base else f"{base}/api/v3/klines"
        params = f"?symbol={sym}&interval={interval}&limit={limit}"
        try:
            async with session.get(url + params, timeout=8) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and isinstance(data, list):
                        return [float(k[4]) for k in data]
        except asyncio.TimeoutError:
            log.debug("corr_guard: timeout fetching %s from %s", sym, base)
        except Exception as e:
            log.debug("corr_guard: error fetching %s: %s", sym, e)
    return None


def _log_returns(closes: List[float]) -> List[float]:
    """Вычисляет list log-returns из списка цен закрытия."""
    result = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            result.append(math.log(closes[i] / closes[i - 1]))
        else:
            result.append(0.0)
    return result


def _pearson(a: List[float], b: List[float]) -> float:
    """Pearson correlation coefficient. Возвращает 0.0 при нулевом std."""
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    ax = a[-n:]
    bx = b[-n:]
    ma = sum(ax) / n
    mb = sum(bx) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ax) / n)
    sb = math.sqrt(sum((x - mb) ** 2 for x in bx) / n)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    cov = sum((ax[i] - ma) * (bx[i] - mb) for i in range(n)) / n
    return max(-1.0, min(1.0, cov / (sa * sb)))


# ── Построение матрицы ────────────────────────────────────────────────────────

async def build_matrix_async(
    session: "aiohttp.ClientSession",
    syms: List[str],
    cache: CorrelationCache,
    *,
    interval: str = "1h",
    limit: int = 48,
) -> CorrelationCache:
    """
    Загружает закрытые цены для всех символов и заполняет матрицу корреляций.
    Если кэш ещё свеж — возвращает его без запросов.
    """
    if not bool(getattr(config, "CORR_GUARD_ENABLED", False)):
        return cache
    if cache.is_fresh(syms):
        return cache

    cfg_interval = str(getattr(config, "CORR_GUARD_TF", "1h"))
    cfg_limit    = int(getattr(config, "CORR_GUARD_WINDOW_BARS", 48))
    interval = cfg_interval
    limit    = cfg_limit

    # Загружаем все исторические данные параллельно
    tasks = {sym: asyncio.create_task(_fetch_closes_async(session, sym, interval, limit))
             for sym in syms}
    closes_map: Dict[str, List[float]] = {}
    for sym, task in tasks.items():
        try:
            closes = await task
            if closes and len(closes) >= 10:
                closes_map[sym] = closes
        except Exception as e:
            log.debug("corr_guard: skip %s: %s", sym, e)

    # Вычисляем попарные корреляции
    rets_map = {sym: _log_returns(cls) for sym, cls in closes_map.items()}
    loaded_syms = list(rets_map.keys())

    cache.matrix.clear()
    for i, s1 in enumerate(loaded_syms):
        for s2 in loaded_syms[i + 1:]:
            rho = _pearson(rets_map[s1], rets_map[s2])
            cache.set(s1, s2, rho)

    cache.mark_built(syms)
    log.debug("corr_guard: matrix built for %d syms, %d pairs", len(loaded_syms),
              len(cache.matrix))
    return cache


# ── Union-Find кластеризация ──────────────────────────────────────────────────

def _union_find_clusters(
    syms: List[str],
    cache: CorrelationCache,
    threshold: float,
) -> Dict[str, int]:
    """
    Строит словарь {sym: cluster_id} через single-linkage Union-Find.
    Два символа в одном кластере если rho >= threshold.
    """
    parent = {s: s for s in syms}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for i, s1 in enumerate(syms):
        for s2 in syms[i + 1:]:
            rho = cache.get(s1, s2)
            if rho is not None and rho >= threshold:
                union(s1, s2)

    # Присваиваем числовые id
    roots: Dict[str, int] = {}
    result: Dict[str, int] = {}
    cid = 0
    for s in syms:
        r = find(s)
        if r not in roots:
            roots[r] = cid
            cid += 1
        result[s] = roots[r]
    return result


# ── Получение порога с учётом режима рынка ────────────────────────────────────

def _effective_threshold(is_bull_day: bool = False, is_bear_day: bool = False) -> float:
    base = float(getattr(config, "CORR_GUARD_THRESHOLD", 0.65))
    if is_bear_day:
        return float(getattr(config, "CORR_GUARD_THRESHOLD_BEAR", 0.99))
    if is_bull_day:
        return float(getattr(config, "CORR_GUARD_THRESHOLD_BULL", 0.60))
    return base


# ── Основная проверка при входе ───────────────────────────────────────────────

def check_entry(
    sym: str,
    open_positions: Dict[str, object],  # {sym: OpenPosition}
    cache: CorrelationCache,
    *,
    is_bull_day: bool = False,
    is_bear_day: bool = False,
) -> CorrGuardResult:
    """
    Проверяет можно ли открыть позицию по sym.

    Алгоритм:
      1. Вычислить max_rho между sym и всеми открытыми позициями.
      2. Если max_rho >= threshold — определить кластер (монеты из open_positions
         с rho >= threshold к sym).
      3. Если кластер уже содержит >= CORR_MAX_PER_CLUSTER позиций → блок.

    Возвращает CorrGuardResult.
    """
    if not bool(getattr(config, "CORR_GUARD_ENABLED", False)):
        return CorrGuardResult(allowed=True)

    open_syms = [s for s in open_positions if s != sym]
    if not open_syms:
        return CorrGuardResult(allowed=True)

    threshold    = _effective_threshold(is_bull_day, is_bear_day)
    max_cluster  = int(getattr(config, "CORR_MAX_PER_CLUSTER", 2))

    # Найти max_rho и однокластерные монеты
    max_rho   = 0.0
    peer_sym  = ""
    cluster_members: List[str] = []

    for s in open_syms:
        rho = cache.get(sym, s)
        if rho is None:
            continue
        if rho > max_rho:
            max_rho  = rho
            peer_sym = s
        if rho >= threshold:
            cluster_members.append(s)

    cluster_size = len(cluster_members)

    if cluster_size >= max_cluster:
        reason = (
            f"corr_guard: rho={max_rho:.3f} to {peer_sym}, "
            f"cluster size {cluster_size}/{max_cluster} "
            f"({', '.join(cluster_members[:3])}{'...' if len(cluster_members) > 3 else ''})"
        )
        return CorrGuardResult(
            allowed=False,
            reason=reason,
            max_rho=max_rho,
            peer_sym=peer_sym,
            cluster_size=cluster_size,
        )

    return CorrGuardResult(
        allowed=True,
        reason="",
        max_rho=max_rho,
        peer_sym=peer_sym,
        cluster_size=cluster_size,
    )


# ── Marginal score weighting ──────────────────────────────────────────────────

def marginal_score(
    sym: str,
    open_positions: Dict[str, object],
    base_score: float,
    cache: CorrelationCache,
) -> float:
    """
    Применяет мягкое снижение оценки для коррелированных кандидатов.

    marginal = base_score * (1 - max_rho_with_open)

    Монета с max_rho=0.70 получит 30%-й штраф к score при ранжировании.
    Монеты без данных корреляции (cache miss) — не штрафуются.
    """
    if not bool(getattr(config, "CORR_MARGINAL_WEIGHTING", True)):
        return base_score
    if not bool(getattr(config, "CORR_GUARD_ENABLED", False)):
        return base_score

    open_syms = [s for s in open_positions if s != sym]
    if not open_syms:
        return base_score

    max_rho = 0.0
    for s in open_syms:
        rho = cache.get(sym, s)
        if rho is not None and rho > max_rho:
            max_rho = rho

    if max_rho < 0.30:   # ниже 0.30 — не штрафуем, это нормальный рыночный фон
        return base_score

    weight = 1.0 - max_rho
    return base_score * weight


# ── Prune — список позиций к закрытию при разбухании кластеров ───────────────

@dataclass
class PruneCandidate:
    symbol:    str
    cluster_id: int
    ranker_final_score: float
    unrealized_pnl_pct: float
    reason:    str


def prune_candidates(
    open_positions: Dict[str, object],  # {sym: OpenPosition}
    last_prices:    Dict[str, float],
    cache: CorrelationCache,
    *,
    is_bull_day: bool = False,
    is_bear_day: bool = False,
) -> List[PruneCandidate]:
    """
    Возвращает список позиций которые следует закрыть (худшие в кластере).

    Логика:
      1. Построить кластеры по порогу.
      2. Для кластеров с размером > CORR_MAX_PER_CLUSTER:
         - Отсортировать по ranker_final_score (хуже = к закрытию).
         - Защита: НЕ закрывать позиции с PnL > CORR_PRUNE_PROFIT_PROTECT_PCT%.
         - Отметить лишние к закрытию.
    """
    if not bool(getattr(config, "CORR_GUARD_ENABLED", False)):
        return []
    if not bool(getattr(config, "CORR_PRUNE_ENABLED", True)):
        return []

    syms = list(open_positions.keys())
    if len(syms) < 2:
        return []

    threshold   = _effective_threshold(is_bull_day, is_bear_day)
    max_cluster = int(getattr(config, "CORR_MAX_PER_CLUSTER", 2))
    profit_protect = float(getattr(config, "CORR_PRUNE_PROFIT_PROTECT_PCT", 2.0))

    clusters = _union_find_clusters(syms, cache, threshold)

    # Группируем по кластеру
    from collections import defaultdict
    cluster_groups: Dict[int, List[str]] = defaultdict(list)
    for s, cid in clusters.items():
        cluster_groups[cid].append(s)

    result: List[PruneCandidate] = []

    for cid, members in cluster_groups.items():
        if len(members) <= max_cluster:
            continue
        # Строим список с метриками
        scored = []
        for s in members:
            pos = open_positions[s]
            ranker_fs = float(getattr(pos, "ranker_final_score", 0.0))
            price = last_prices.get(s)
            pnl_pct = 0.0
            if price is not None:
                entry = float(getattr(pos, "entry_price", price))
                if entry > 0:
                    pnl_pct = (price / entry - 1.0) * 100.0
            scored.append((ranker_fs, pnl_pct, s))

        # Сортируем: лучшие (по ranker_final_score DESC) выживают
        scored.sort(key=lambda x: x[0], reverse=True)

        # Оставляем top-max_cluster, остальные — кандидаты на закрытие
        for ranker_fs, pnl_pct, s in scored[max_cluster:]:
            if pnl_pct > profit_protect:
                log.debug("corr_guard prune: skip %s (pnl=%.2f%% > protect=%.2f%%)",
                          s, pnl_pct, profit_protect)
                continue
            reason = (
                f"corr_cluster_prune: cluster {cid} size {len(members)}->{max_cluster}, "
                f"ranker_final={ranker_fs:.3f}, pnl={pnl_pct:.2f}%"
            )
            result.append(PruneCandidate(
                symbol=s,
                cluster_id=cid,
                ranker_final_score=ranker_fs,
                unrealized_pnl_pct=pnl_pct,
                reason=reason,
            ))
            log.info("corr_guard: prune candidate %s (cluster %d, ranker=%.3f, pnl=%.2f%%)",
                     s, cid, ranker_fs, pnl_pct)

    return result


# ── Утилиты для monitor.py ────────────────────────────────────────────────────

def get_or_create_cache(state_dict: dict) -> CorrelationCache:
    """Возвращает CorrelationCache из state.__dict__, создавая при необходимости."""
    if "corr_cache" not in state_dict:
        state_dict["corr_cache"] = CorrelationCache()
    return state_dict["corr_cache"]


def get_bull_bear_flags(state_dict: dict) -> Tuple[bool, bool]:
    """Извлекает флаги бычьего/медвежьего дня из state.__dict__."""
    is_bull = bool(state_dict.get("bull_day", False))
    is_bear = bool(state_dict.get("bear_day", False))
    return is_bull, is_bear


def format_cluster_report(
    open_positions: Dict[str, object],
    cache: CorrelationCache,
    *,
    is_bull_day: bool = False,
    is_bear_day: bool = False,
) -> str:
    """Возвращает текстовый отчёт о кластерах для Telegram /portfolio."""
    if not open_positions:
        return "Нет открытых позиций."

    syms = list(open_positions.keys())
    threshold = _effective_threshold(is_bull_day, is_bear_day)
    clusters = _union_find_clusters(syms, cache, threshold)

    from collections import defaultdict
    groups: Dict[int, List[str]] = defaultdict(list)
    for s, cid in clusters.items():
        groups[cid].append(s)

    lines = [f"*Correlation Guard* (порог rho={threshold:.2f}):"]
    for cid, members in sorted(groups.items()):
        if len(members) == 1:
            lines.append(f"  `{members[0]}` — независимая")
        else:
            lines.append(f"  Кластер {cid + 1}: {', '.join(f'`{m}`' for m in members)}")
            # Показать max pair rho
            max_rho = 0.0
            for i, s1 in enumerate(members):
                for s2 in members[i + 1:]:
                    rho = cache.get(s1, s2) or 0.0
                    if rho > max_rho:
                        max_rho = rho
            lines.append(f"    max rho = {max_rho:.3f}")
    return "\n".join(lines)
