# Top-gainer model (daily CatBoost)

- **Slug:** `top-gainer-model`
- **Status:** shipped (retroactive)
- **Created:** 2026-04-26
- **Owner:** core
- **Related:** `files/train_top_gainer.py`, `files/top_gainer_critic.py`,
  `files/top_gainer_dataset.jsonl`, `files/top_gainer_model.json`

---

## 1. Problem

Главная метрика — попасть в top-N daily gainers. Нужна модель, которая
оценивает вероятность для произвольного watchlist-coin’а попасть в top5 /
top10 / top20 / top50 на сегодня, по фичам, видимым **до** основного движа.

## 2. Success metric

- `recall@top20 ≥ 95 %` (текущее: 100 % с 2026-04-10).
- AUC top20 ≥ 0.85 (текущее: 0.93).

## 3. Scope

### In scope
- 4 tier-классификатора (top5 / top10 / top20 / top50).
- Daily retrain через `daily_learning.py`.
- Top-gainer critic (оценка качества предсказаний бота).

### Out of scope
- Использование TG-prob внутри ranker (см. `ml-candidate-ranker-spec`).

## 4. Behaviour / design

### Dataset

`files/top_gainer_dataset.jsonl` (~29 MB). Snapshot фичей всех ~105 watchlist
coin × N (intraday: 08:30 / 14:30 / 20:30 local) + EOD-resolve (попал в
top-N или нет).

Snapshots создаёт scheduled task `CryptoBot_IntradaySnapshot`. EOD-resolve —
`CryptoBot_DailyLearning_EOD` (02:30 local).

### Training

`train_top_gainer.py` обучает 4 CatBoost-классификатора
(один-vs-остальные по tier). Артефакт: `top_gainer_model.json`.

### Inference

`monitor.py` зовёт модель на каждом цикле для каждого watchlist coin →
`ranker_top_gainer_prob`. Используется как фича в ranker и как условие
в `ML_CANDIDATE_RANKER_HARD_VETO_1H_TOP_GAINER_MAX`.

### Critic

`top_gainer_critic.py` каждые ~12 ч сравнивает предсказания бота
(сигналы из `critic_dataset.jsonl`) с фактическим top-N → midday + final
отчёты в Telegram (с дедупом, см. `CLAUDE.md` §12).

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `TOP_GAINER_CRITIC_ENABLED` | True | выкл. critic-отчёты |
| `TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED` | True | toggle TG-postings |

Откат: выключить флаги, модель остаётся в проде, но critic не пишет в TG.

## 6. Risks

- **Watchlist drift:** список 105 coin’ов immutable (`watchlist.json`).
  Если top-gainer’а нет в list — это **ожидаемо**, не bug.
- **Снапшоты пропущены** при сбое scheduler → recall просядет на 1 день.

## 7. Verification

- `files/report_rl_daily.py` репортит daily AUC + recall.
- Critic Telegram-отчёты дают human-readable breakdown.

## 8. Follow-ups

- Отдельная спека на `anti-fast-reversal` (label `label_fast_reversal`
  пишется в этот же dataset).
