# Correlation guard (Pearson cluster cap)

- **Slug:** `correlation-guard`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/correlation_guard.py`, `CLAUDE.md` §2

---

## 1. Problem

Бот может за один час зайти в 4–5 коинов из одного кластера (например,
DeFi, или alt-stack, который двигает BTC), и в шок-день они все идут вниз
синхронно. Нужен guard, который ограничивает количество позиций в одном
correlation-кластере.

## 2. Success metric

- Доля одновременно открытых позиций из одного кластера ≤ 2.
- Drawdown в шок-день ≤ X (метрика TBD после baseline).

## 3. Scope

### In scope
- Pearson-корреляция log-returns по N-барному окну.
- Union-Find кластеризация по threshold.
- Гард `correlation_guard` в `trend_scout_rules.py`.

### Out of scope
- Beta-hedging.
- Динамическое sizing.

## 4. Behaviour / design

### Алгоритм

1. На каждом poll-цикле берём `lookback` баров close для каждого open + new
   candidate.
2. Считаем Pearson по log-returns каждой пары.
3. Union-Find: соединяем символы, если `corr ≥ CORR_GUARD_THRESHOLD`.
4. Если в кластере нового кандидата уже `≥ CORR_GUARD_MAX_PER_CLUSTER`
   open positions → block с reason_code=`correlation_guard`.

### Open cluster cap

Дополнительно есть `open_cluster_cap` — глобальный лимит на сумму открытых
позиций в **любом** кластере (см. Pareto sweep: avg_r5=−0.36 %, working
correctly).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `CORR_GUARD_ENABLED` | True | Master switch |
| `CORR_GUARD_THRESHOLD` | (см. config.py) | Pearson порог для clustering |
| `CORR_GUARD_LOOKBACK_BARS` | (см. config.py) | Окно log-returns |
| `CORR_GUARD_MAX_PER_CLUSTER` | (см. config.py) | Лимит open-в-кластере |

Откат: `CORR_GUARD_ENABLED=False`.

## 6. Risks

- **False clustering** на коротком lookback → блокировка реально
  независимых сетапов.
- **Дрейф кластеров**: пересчёт каждый poll, не кэшируется.

## 7. Verification

- Pareto sweep: `open_cluster_cap` n=136, avg_r5=−0.36 %, win=35 % —
  работает корректно (вырезает реально плохое).

## 8. Follow-ups

- Кэшировать кластеризацию между бар-poll’ами (производительность).
- Добавить hierarchical clustering (single-linkage → complete-linkage).
