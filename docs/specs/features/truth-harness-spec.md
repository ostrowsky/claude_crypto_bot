# Truth Harness — защита от завышенных и недоказанных метрик

- **Slug:** `truth-harness`
- **Status:** shipped; current bot compliance is FAIL until the open audit blockers are repaired
- **Created:** 2026-08-13
- **Owner:** core / measurement
- **Related:** `metrics-canonical-spec.md`, `auto-improvement-loop-spec.md`,
  `health-report-integrity-spec.md`

## 1. Проблема

В проекте неоднократно принимались proxy- и in-sample-метрики за реальный
прогресс: насыщенный `recall@20`, AUC модели с признаками текущего дневного
движения, сравнение окон разного размера и отсутствие данных, записанное как
промах. Это создаёт положительный отчёт без доказанного улучшения цели бота.

Одних текстовых правил недостаточно: до этой спеки документация уже заявляла,
что truth-check подключён к `pre-commit`, хотя tracked hook отсутствовал.

## 2. Цель

Перед каждой доработкой и перед фиксацией результата получать воспроизводимый
ответ на три вопроса:

1. соответствует ли код обязательным MD и спецификациям;
2. подтверждают ли источники опубликованные метрики и выводы;
3. имеет ли изменение честную проверку на целевой популяции, защитные метрики и
   путь отката.

Harness не оптимизирует торговую политику и не разрешает production-relaxation.
Он блокирует недоказанные утверждения и указывает, какое evidence отсутствует.

## 3. Инварианты достоверности

| ID | Инвариант | Машинная проверка |
|----|-----------|-------------------|
| TH-01 | Доля публикуется только с числителем, знаменателем, базовой ставкой и lift либо явно помечается `diagnostic-only`. | report/schema audit |
| TH-02 | Proxy/training metric не называется прогрессом бизнес-метрики. Для каждого вопроса используется canonical metric из `metrics-canonical-spec.md`. | canonical coverage audit |
| TH-03 | Для модели указаны `feature_time`, `label_time` и признаки, способные прямо кодировать label. | model provenance audit |
| TH-04 | Achievement-метрики только out-of-sample; split по времени и не разрывает один UTC-день между train/holdout. | training metadata/static audit |
| TH-05 | Сравниваются окна с одинаковым определением и сопоставимым denominator; downtime/partial days — `unknown`, не `miss`. | report integrity audit |
| TH-06 | Gate проверяется на фактической популяции кандидатов/входов бота и на максимальном доступном периоде, а не только на market-wide эпизодах. | change evidence audit |
| TH-07 | Изменение поведения имеет feature flag, rollback, shadow/canary (когда применимо) и multi-objective guardrails. | staged-diff/spec audit |
| TH-08 | Отрицательный результат хранится с периодом, N, метриками и вердиктом; отклонённая гипотеза не предлагается повторно без новых данных. | evidence-memory audit |
| TH-09 | Утверждения о текущем состоянии MD совпадают с config, кодом и runtime; исторические утверждения явно помечены. | MD/config/runtime audit |
| TH-10 | Freshness, coverage и неизвестность являются частью метрики. При неполном evidence вывод — `unknown/рано судить`. | freshness/report audit |
| TH-11 | Для торгового результата показывается canonical portfolio alpha; PnL отдельного режима или сделки не заменяет его. | canonical coverage audit |
| TH-12 | Любая доработка имеет spec, focused tests, verification и ссылку на применимые TH-инварианты. Loop-компоненты обновляют living spec. | staged-diff audit |

## 4. Интерфейс

`files/truth_harness.py` предоставляет профили:

- `change --staged` — быстрый блокирующий прогон для `pre-commit`;
- `full` — аудит MD, config, runtime, свежего health report, metric provenance и
  текущих нарушений;
- `--json PATH` — машиночитаемый отчёт с `check_id`, severity, evidence и
  remediation.

Exit codes: `0` — блокирующих нарушений нет; `1` — compliance failure;
`2` — сам Harness не смог завершить проверку.

## 5. Enforcement

1. Tracked `.githooks/pre-commit` запускает `truth_harness.py change --staged`.
2. `AGENTS.md` требует запускать project skill `crypto-bot-truth-harness` в
   начале аудита и перед handoff любой доработки поведения/метрик/отчётов.
3. Skill выполняет механический профиль, затем обязательный judgment-check для
   инвариантов, которые нельзя доказать статически.
4. `git config core.hooksPath .githooks` включается в рабочей копии; наличие
   tracked hook и эта настройка проверяются полным профилем.
5. Нельзя обходить failed check формулировкой вывода. Допустим только явный
   waiver в spec: ID, причина, риск, срок и владелец.

## 6. Acceptance criteria

- [x] Current-state audit перечисляет каждое найденное завышение с source,
  denominator/split/freshness и severity.
- [x] In-sample bandit recall не отображается как способность/прогресс модели.
- [x] Полный профиль обнаруживает отсутствие canonical metric, label leakage,
  несопоставимые окна и MD/config/runtime drift.
- [x] Change-профиль обнаруживает code change без spec/test/evidence и
  loop-change без обновления living spec.
- [x] Tracked hook реально запускается; локальный `core.hooksPath` указывает на
  `.githooks`.
- [x] Skill создан через `skill-creator`, проходит `quick_validate.py` и
  установлен в Codex skills.
- [x] Focused unit tests и Harness self-test проходят; `git diff --check`
  выполняется после staging в финальном engineering cycle.

## 7. Rollback

Удалить `.githooks/pre-commit`, требование из `AGENTS.md` и skill; сам
`truth_harness.py` безопасен и read-only. Изменения формата training/health
метрик откатываются отдельно и не меняют торговые решения.

## 8. Verification

Проверено 2026-08-13:

- `test_truth_harness.py`: 7/7 PASS (честный fixture, leakage, row/day split,
  in-sample ratio, stale evidence, missing spec/test);
- `test_bot_health_report_integrity.py`: 17/17 PASS (partial/final provenance,
  unknown denominator, provisional trend, canonical scorecard, fail-closed gap);
- `test_learning_metric_integrity.py`: 2/2 PASS (ratio context/lift);
- `test_learning_progress_truth.py`: 1/1 PASS (provisional/in-sample trend
  остаётся `UNKNOWN`);
- `quick_validate.py`: PASS для repository skill и установленной копии;
- `truth_harness.py full`: ожидаемый **FAIL**, 7 blockers + 1 warning. Это
  current-state verdict, не ошибка Harness. Полный перечень —
  `docs/reports/2026-08-13-truth-harness-audit.md`.
