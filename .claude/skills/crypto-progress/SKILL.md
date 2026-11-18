---
name: crypto-progress
description: >
  Strategic learning-progress + ML/RL/scout analysis for the crypto trading
  bot. USE THIS SKILL whenever the user asks: is the bot improving toward its
  target metric? progress or degradation? are all components using data? is
  there enough data? what roadmap stage are we at / does it need correcting?
  next steps? Also: learning progress, ML/bandit metrics, recall@20, UCB
  separation, AUC trends, scout gate analysis, blocked gates, over-blocking,
  shadow disagreements, North Star, анализ прогресса обучения, скаут анализ,
  отчёт по обучению, прогресс/деградация, roadmap статус. Trigger on:
  "анализ прогресса", "прогресс обучения", "идёт ли прогресс", "деградация",
  "хватает ли данных", "стадия roadmap", "дальнейшие шаги", "learning
  progress", "scout analysis", "запусти анализ", "покажи прогресс" — even
  if worded differently.
---

# crypto-progress skill

## Purpose

Answer six recurring strategic questions about the bot, **token-efficiently**
(one aggregation script does all the heavy file reading; you only interpret a
compact structured report):

1. Is the bot improving toward its target (North Star) metric?
2. Progress or degradation? (longest available history)
3. Are all analytical components (ML / bandit / scout / pipeline) using data
   to approach the target?
4. Is there enough data to keep learning?
5. What roadmap stage are we at — and does it need correcting?
6. What are the concrete next steps?

## Step-by-step execution

### 1. Run the strategic analyzer (ALWAYS — this is the primary source)

```
cwd : D:\Projects\claude_crypto_bot
cmd : D:\Projects\claude_crypto_bot\pyembed\python.exe files\analyze_learning_progress.py
```

This single script aggregates `learning_progress.jsonl`, `metrics_daily.jsonl`,
the latest pipeline health JSON, `decisions.jsonl`, attribution, and the roadmap
specs (`continuous-improvement-agent-spec.md` = signal-bot target,
`autonomous-trading-roadmap-1010-spec.md` = trader; the legacy
`auto-improvement-loop-spec.md` is only flagged as stale). Its stdout is
organized into `[Q1]`..`[Q6]` blocks. Read it in full (~90 lines) — do **not**
read the underlying jsonl files yourself.

Trends compare only **comparable, non-overlapping windows** (same metric
definition, label store covering the window, same comparability epoch) and
carry n per window (spec `learning-progress-comparable-1118-spec.md`).
Verdicts:

| verdict | meaning |
|---|---|
| `IMPROVING` / `DEGRADING` | 95% CI of late − early half excludes 0 |
| `INCONCLUSIVE` | CI spans 0 — write «рано судить», never a direction |
| `INSUFFICIENT` | < 2 disjoint windows, n not logged, or pooled n < 20 per half |
| `UNKNOWN` | ground truth not verified / no out-of-sample rows |
| `DIAGNOSTIC` | UCB separation, AUC: no n or variance logged → first/last only, no verdict |

There is no `FLAT`: an interval spanning zero does not show flatness.

### 2. (OPTIONAL) Tactical deep-dive — only if the user wants gate/ML detail

Run these two **in parallel in the same message** only when the user asks for
scout/gate specifics, shadow disagreements, or per-day ML tables. Skip them for
routine "is it progressing?" checks to save tokens.

```
cwd : D:\Projects\claude_crypto_bot\files
cmd : D:\Projects\claude_crypto_bot\pyembed\python.exe report_rl_daily.py
```
```
cwd : D:\Projects\claude_crypto_bot
cmd : D:\Projects\claude_crypto_bot\pyembed\python.exe files\analyze_blocked_gates.py
```

### 3. Render the report

Use this exact template. Fill every `{...}` from the analyzer stdout. Keep it
tight — the value is in the verdicts and the next-steps, not in dumping raw
numbers.

---

## Отчёт прогресса обучения — {DATE}

### Q1. Совершенствуется ли бот к целевой метрике?

**Вердикт: {IMPROVING / DEGRADING / РАНО СУДИТЬ (INCONCLUSIVE) / НЕДОСТАТОЧНО ДАННЫХ (INSUFFICIENT) / UNKNOWN}**

- North Star `NS_EarlyCapture_top20_v2`: latest `{value}` on `{date}`, **n={n} winner-days**, full days `{full}/{window}`, days_down_or_partial `{down}`; target 0.40, floor 0.25
- Разложение: coverage `{cov}`, capture `{cap}`, lead `{lead}` (capture и lead — средние по вошедшим, их произведение ≠ NS)
- v3 move-lead (`NS_EarlyCapture_top20_v3`): `{v3}`, n `{v3_n}`, без дедлайна `{no_deadline}`
- Непересекающиеся окна: {each `date = value (n)`}
- Тренд: `{verdict}` — early `{days}` `{value}` (n={n}) → late `{days}` `{value}` (n={n}), Δ `{delta}`, 95% CI `[{lo}, {hi}]`
- Исключено строк: `{total}` (главные причины: {top 2 reasons}) — ряд до 2026-09-22 не сравнивается с текущим
- {If INSUFFICIENT/INCONCLUSIVE: say plainly the NS cannot answer the question at this n; never state a direction}

### Q2. Диагностика обучения (не прогресс цели, TH-02)

Эпохи сравнимости: bandit с `{bandit_epoch}`, model с `{model_epoch}` (причины — в выводе).

| Метрика | Вердикт | Значения | n / контекст |
|---------|---------|----------|--------------|
| recall@20 (OOS) | {v} | latest `{recall}` | n `{n or ?}`, ENTER `{enter}`, base `{base}`, lift `{lift}`, precision `{prec}` |
| UCB separation | DIAGNOSTIC | `{first}` → `{last}` (last−first `{d}`) | нет n — без вердикта |
| AUC top20 | DIAGNOSTIC | `{first}` → `{last}` (last−first `{d}`) | нет n — без вердикта |

Счётчики: n_signal `{a}` → `{b}` (с потолком), updates `{a}` → `{b}` — объём, не прогресс.

### Q3. Используют ли все компоненты данные для приближения к цели?

- **ML/bandit:** {fed YES/UNKNOWN}, recall `{r}`, AUC `{auc}`, updates `{u}`
- **Deployment health:** {available? else "критик ещё не отработал сегодня"}
- **Scout:** take_baseline `{r5}%`, over-blocking gates: {list top 3 with miss_pct}
- **Pipeline:** {n} decisions {by_stage}; attribution hit_rate `{hit_rate}`
- {If pending_manual_validation > 0: flag it as the structural blocker}

### Q4. Хватает ли данных?

- top-20 winners `{n}`, coverage `{cov}%`, silent-miss `{sm}%`
- entries/day `{epd}`, precision `{p}%`
- metrics_daily age: **`{age}`d** — {fresh / **STALE → canonical NS не обновляется**}
- Red flags ({n}): {list id + severity}

### Q5. Стадия roadmap — нужна ли коррекция?

- **Сигнальный бот** (`continuous-improvement-agent-spec.md`, целевая): статус `{status}`, ревизия `{revised}`, последний коммит `{date}` ({age} д.)
  - Фазы rollout: {list}
  - Достигнутая фаза: **не вычисляется** — в спеке нет статуса фаз; не называй стадию без проверки exit gate
- **Трейдер** (`autonomous-trading-roadmap-1010-spec.md` §7): последний коммит `{date}` ({age} д.), коммитов в `trader/` после него: `{n}`
  - {rows: spec [stage]: verdict — only the ones relevant to the question}
- **Legacy-матрица** (`auto-improvement-loop-spec.md`): STALE с `{date}`, не план; RM-4/RM-5/RM-18 сняты
- **Коррекция:** {your judgement — e.g. if the trader table lags many commits,
  say which verdicts it is missing; if metrics_daily is stale, that outranks
  any roadmap item}

### Q6. Дальнейшие шаги

{Number the analyzer's next-steps list, then add your own prioritized
recommendation. Order by: data-honesty blockers first (stale metrics, stuck
pipeline), then North-Star-moving roadmap items, then nice-to-haves.}

---

## Interpretation rules (apply when writing the report)

- An `INSUFFICIENT`/`INCONCLUSIVE` North Star, or a `STALE` metrics_daily,
  means **you cannot honestly answer Q1 with a direction** — say «рано судить»
  plainly; do not paper over it with bandit metrics. Per project §0, the
  train→live gap is the most important diagnostic; never collapse it into one
  rosy number.
- Never build a trend yourself from values the analyzer did not put in one
  chain (e.g. an April number vs today's): they are different definitions
  (TH-05).
- Every recall/coverage/precision you quote carries its n, base rate and lift
  (TH-01). Recall at a different ENTER rate is a different operating point.
- UCB separation and AUC are `DIAGNOSTIC`: report first → last, never as
  progress or degradation.
- A gate in over-blocking (miss_pct > take baseline) is a **candidate to
  relax**, but only flag it; the pipeline must validate via Pareto sweep.
- `pending_manual_validation` decisions have no validator verdict — flag them
  under Q3/Q6; do not infer from them which roadmap items shipped.
- If the analyzer prints `NOT FOUND` for a roadmap spec, that part of Q5 is
  unanswerable — say so instead of guessing.

## Output format

Render the markdown report directly in the conversation. Do not save to a
file. Bold the six verdicts. Keep it scannable in under one screen unless the
user explicitly asked for the tactical deep-dive.
