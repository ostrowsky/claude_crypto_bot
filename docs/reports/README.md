# Reports

Date-stamped operational reports (отличаются от спек: спеки — *что* и *почему*,
репорты — *как идёт*).

| File | Coverage |
|------|----------|
| [`2026-04-28-learning-and-scout.md`](./2026-04-28-learning-and-scout.md) | Прогресс обучения (recall, UCB sep, AUC) + Pareto-обзор скаута + trail-arm summary + action items. |
| [`2026-04-28-improvements.md`](./2026-04-28-improvements.md) | 9 верифицированных бэктестами предложений с ранжированием по impact на главную цель (early BUY на top-20). |
| [`2026-04-28-improvements-validated.md`](./2026-04-28-improvements-validated.md) | Каждое предложение прогнано через бэктест. P5 поднята до HIGH; P1 опущена (3 из 5 «silent» — это holding); P3 ушла в needs-redesign (75 FP/day). |
| [`2026-04-29-metrics-baseline.md`](./2026-04-29-metrics-baseline.md) | Baseline всех 13 метрик framework-спеки. North-star `EarlyCapture@top20 = 0.074` (vs target 0.40). Capture = 0.052 — главный отстающий, не coverage. |
| [`2026-04-29-north-star-roadmap.md`](./2026-04-29-north-star-roadmap.md) | 4 пути роста north-star (earliness, hold-longer, coverage, precision). 10-step rollout 0.077 → 0.42. Главный insight: earliness двигает 2 lever’а сразу (capture + lead). |
| [`2026-05-01-mode-audit.md`](./2026-05-01-mode-audit.md) | Комплексный аудит 7 режимов: 4 архитектурных конфликта (trend≡impulse_speed, brk-vs-impulse, ret-vs-trend, surge dead-code). Метрики не покрывают exit-side. 8 гипотез (H1-H8) с sequenced rollout 12 недель. |
| [`2026-05-02-ex1-baseline.md`](./2026-05-02-ex1-baseline.md) | EX1 baseline median +0.001 на top-20. Утечка через `ema20_weakness` exits подтверждена цифрой (worst cases: −10 % pnl при +172 % potential). H3 deployed flagged-off. |
| [`2026-04-30-roadmap-validation.md`](./2026-04-30-roadmap-validation.md) | 4 backtest-валидации: 2A dynamic max_hold подтверждено (+0.039 NS), 1C мельче (+0.005), 1A/4A defer до logger-fix. Re-rank: 2A — #1 win. |

## Конвенция

- Имя: `YYYY-MM-DD-<topic>.md`.
- Источники указываются в шапке (jsonl-файлы, скрипты).
- Action items — в таблице в конце; каждый пункт ссылается на спеку или
  заводит follow-up.
