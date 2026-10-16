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

This single script aggregates the longest available history from
`learning_progress.jsonl`, `metrics_daily.jsonl`, the latest pipeline health
JSON, `decisions.jsonl`, attribution, and the living roadmap spec. Its stdout
is already organized into `[Q1]`..`[Q6]` blocks with computed trend verdicts
(IMPROVING / FLAT / DEGRADING / INSUFFICIENT). Read it in full; it is small
(~60 lines) by design — do **not** read the underlying jsonl files yourself.

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

**Вердикт: {IMPROVING/FLAT/DEGRADING/НЕТ ДАННЫХ}**

- North Star (`watchlist_top_early_capture_pct`): latest `{value}`, baseline_7d `{baseline_7d}`, status `{status}`
- Тренд early_capture: `{trend}` (first `{first}` → last `{last}`, Δ `{delta}`)
- {If INSUFFICIENT/STALE: explicitly say the NS measurement itself is the weak link and why}

### Q2. Прогресс или деградация? (вся история)

История: `{span}`

| Метрика | Вердикт | first → last | Δ |
|---------|---------|--------------|---|
| recall@20 | {v} | {first}→{last} | {delta} |
| UCB separation | {v} | {first}→{last} | {delta} |
| AUC top20 | {v} | {first}→{last} | {delta} |
| bandit n_signal | {v} | {first}→{last} | {delta} |

bandit updates: `{first}` → `{last}`

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

- Спека обновлена: `{last_updated}`
- Component matrix: OK `{done}` / partial `{partial}` / missing `{missing}` ({pct}% done)
- Завершено: `{roadmap_done}`
- Осталось ({n}): {first 5 RM-N + short desc}
- **Коррекция:** {your judgement — is the priority order still right given the
  red flags and the data-staleness findings? e.g. if metrics_daily is stale,
  fixing the EOD job outranks any new RM item}

### Q6. Дальнейшие шаги

{Number the analyzer's next-steps list, then add your own prioritized
recommendation. Order by: data-honesty blockers first (stale metrics, stuck
pipeline), then North-Star-moving roadmap items, then nice-to-haves.}

---

## Interpretation rules (apply when writing the report)

- A `FLAT`/`INSUFFICIENT` North Star with a `STALE` metrics_daily means **you
  cannot honestly answer Q1** — say so plainly; do not paper over it with
  bandit metrics. Per project §0, the train→live gap is the most important
  diagnostic; never collapse it into one rosy number.
- `recall@20 = 1.0` for many days is a **plateau ceiling**, not a triumph —
  pivot attention to UCB separation slope and the deployment-side NS.
- A gate in over-blocking (miss_pct > take baseline) is a **candidate to
  relax**, but only flag it; the pipeline must validate via Pareto sweep.
- `pending_manual_validation` decisions are the canonical sign that RM-1/RM-2
  (blocked-event logging) have not reached the production bot yet — call this
  out under Q3/Q6 whenever the count is > 0.
- If the analyzer prints `spec NOT FOUND`, Q5 is unanswerable — tell the user
  the roadmap spec path could not be resolved instead of guessing.

## Output format

Render the markdown report directly in the conversation. Do not save to a
file. Bold the six verdicts. Keep it scannable in under one screen unless the
user explicitly asked for the tactical deep-dive.
