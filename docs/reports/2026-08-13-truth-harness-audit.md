# Аудит достоверности метрик и соответствия MD — 2026-08-13

## Вердикт

Текущее состояние бота **не соответствует Truth Harness**: полный профиль
находит 7 блокирующих нарушений и 1 предупреждение. Торговую policy этот пакет
не меняет. Он прекращает выдавать недоказанные proxy/in-sample показатели за
достижение цели и делает оставшийся measurement debt видимым и блокирующим.

Применимые инварианты: TH-01, TH-02, TH-03, TH-04, TH-05, TH-06, TH-07,
TH-08, TH-09, TH-10, TH-11, TH-12.

## Где бот занимался самообманом

| Severity | Утверждение/механизм | Реальное доказательство | Коррекция |
|---|---|---|---|
| blocker | `recall@20=100%` трактовался как способность видеть победителей | Bandit сначала обучается на накопленном dataset, затем оценивается на тех же записях; при этом не публиковались action rate, base rate и lift | Метрика названа `in_sample_post_fit`, добавлены action/base/precision/lift; legacy ratio без контекста подавляется; training↔live gap не вычисляется |
| blocker | AUC≈0.99 называлась прогнозом будущего top-20 | `label_top20` и `tg_return_since_open` создаются из одного snapshot; признак содержит ответ | В report/model metadata добавлены scope, label timing и answer-encoding features; AUC называется diagnostic-only |
| blocker | North Star называлась итоговым EOD ранним захватом | `_compute_early_capture.py` использует тот же `label_top20` из rolling-24h snapshot; это не immutable later-EOD outcome | North Star помечена `provisional`; тренд отключён; P0 — создать later-EOD labels и пересчитать baseline |
| blocker | Holdout модели считался временным | После сортировки используется row-index split; строки одного UTC-дня могут попасть по обе стороны | Текущий scope явно назван `time_sorted_row_holdout_same_snapshot_label`; P0 — split по полным дням |
| blocker | «Большой training→live gap значит, что виноваты downstream filters» | Training side был in-sample, поэтому вычитание из live-rate не имеет смысла | Gap fail-closed и имеет значение `unknown` до OOS temporal holdout |
| blocker | Per-mode/per-trade P&L подменял ответ «бот зарабатывает?» | Свежего aggregate portfolio alpha vs buy-and-hold нет; aggregate evaluator от 2026-05-04 stale | Canonical scorecard показывает `portfolio_alpha=unknown`; восстановление weekly aggregate — P0 |
| blocker | EX1 из daily aggregator считался canonical exit monetization | Aggregator запускает deprecated proxy-mode без `--use-zigzag` | Canonical EX1 показывается `unknown`; proxy value не выдаётся за realized potential; ZigZag provenance — P0 |
| blocker | `do_not_touch` считался актуально доказанным | `last_verified=2026-05-28`, возраст 77 дней при бюджете 30 | Отчёт создаёт critical stale-evidence flag; gates остаются fail-closed до maximum-period replay |
| serious | Отсутствующий denominator превращался в 1 через `or 1` | Нулевой/unknown `watchlist_top_count` или `bot_unique_buys` давал численно определённый ratio | Деление теперь даёт `unknown`, а missing top-mover denominator — critical flag |
| serious | Midday critic подписывался строкой `final critic` | На 2026-08-13 фактически использован `phase=midday` | Telegram показывает `midday (partial)` |
| serious | Strategic learning analyzer строил тренды North Star/recall/AUC независимо от provenance | История включала provisional same-snapshot labels и in-sample training rows | `analyze_learning_progress.py` теперь допускает тренд только для verified NS и OOS temporal evidence; иначе `UNKNOWN` |
| warning | История отрицательных результатов неполна | 47 legacy `_backtest_*.py` не имеют durable verdict | Harness предупреждает; при следующем использовании каждого replay обязателен period/N/metrics/verdict |

## Реальный scorecard на доступных данных

| Целевой вопрос | Значение | Цель | Статус |
|---|---:|---:|---|
| EarlyCapture@top20 | 0.070; n=26; 10/14 полных дней | 0.40 (floor 0.25) | **provisional**, тренд не доказан |
| Portfolio alpha vs B&H | unknown | >0 | отсутствует fresh aggregate evidence |
| Signal precision | 7.76%; 18/232 | 35% | provisional top-20 labels |
| Telegram messages/day | 16.57 | ≤10 | измерено |
| Median time-to-signal | 3.33 h; n=18 | ≤0.5 h | provisional winner labels |
| Canonical EX1 ZigZag | unknown | 0.50 | daily source — deprecated proxy-mode |
| Fast reversal | 12.77%; n=423 | ≤8% | ниже цели |
| Whipsaw | 9.22%; n=423 | ≤5% | ниже цели |

Отдельно: средний net P&L на закрытую сделку в diagnostic window равен
−0.4533%, но это **не** portfolio alpha и не заменяет его.

## Приоритет исполнения

1. P0 — создать immutable later-EOD top-20 ground truth, version metric и
   пересчитать историю/baselines.
2. P0 — построить day-grouped temporal holdout и оценивать frozen pre-fit
   model/bandit на последующих днях; только после этого разрешить training→live
   gap.
3. P0 — восстановить fresh aggregate portfolio alpha vs buy-and-hold и
   canonical EX1 `--use-zigzag` с явной provenance.
4. P0 — восстановить полные runtime days и обновить stale `do_not_touch`
   targeted replay на максимальном доступном периоде текущей policy.
5. P1 — после восстановления truth sources проверять alert-ranker при
   фиксированном message budget и exit monetization hypotheses; production
   BUY/SELL не менять без положительного multi-objective backtest.

## Что теперь обеспечивает Harness

- `files/truth_harness.py full` проверяет MD/config, enforcement, label/feature
  timing, split, post-fit evaluation, fresh health report, canonical coverage,
  freshness и negative-memory.
- `files/truth_harness.py change --staged` блокирует source change без spec и
  focused tests, loop change без living-spec update и behaviour change без
  rollback/maximum-period/shadow-or-canary contract.
- Tracked `.githooks/pre-commit` запускает staged profile и
  `git diff --cached --check`.
- Project skill `crypto-bot-truth-harness` добавляет judgment-проверку целевой
  популяции, causality, leakage, rollout и соответствия всех MD.

## Ограничение аудита

`full=FAIL` после внедрения — ожидаемый и честный результат: Harness установлен,
но найденный measurement debt ещё не устранён. Наличие Harness не превращает
бота в compliant; оно не даёт следующей доработке скрыть эти нарушения.
