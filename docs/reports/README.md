# Reports

Date-stamped operational reports (отличаются от спек: спеки — *что* и *почему*,
репорты — *как идёт*).

| File | Coverage |
|------|----------|
| [`2026-04-28-learning-and-scout.md`](./2026-04-28-learning-and-scout.md) | Прогресс обучения (recall, UCB sep, AUC) + Pareto-обзор скаута + trail-arm summary + action items. |
| [`2026-04-28-improvements.md`](./2026-04-28-improvements.md) | 9 верифицированных бэктестами предложений с ранжированием по impact на главную цель (early BUY на top-20). |
| [`2026-04-28-improvements-validated.md`](./2026-04-28-improvements-validated.md) | Каждое предложение прогнано через бэктест. P5 поднята до HIGH; P1 опущена (3 из 5 «silent» — это holding); P3 ушла в needs-redesign (75 FP/day). |

## Конвенция

- Имя: `YYYY-MM-DD-<topic>.md`.
- Источники указываются в шапке (jsonl-файлы, скрипты).
- Action items — в таблице в конце; каждый пункт ссылается на спеку или
  заводит follow-up.
