---
name: md-compliance
description: Audit whether the crypto bot actually complies with the requirements written in CLAUDE.md and PROJECT_CONTEXT.md — config flags, monitored set, metric honesty rules (§0a truth harness), learning-loop obligations (§0), and whether the running process matches what the docs claim. USE THIS SKILL whenever the user asks to check compliance with the MD / спеки / требования, asks "соответствует ли бот MD", "проверь соответствие требованиям", "аудит бота", "проверь спеки", asks whether a planned or completed change respects the documented rules, or before shipping a change that touches behaviour, gates, metrics or reports. Also use it proactively when about to modify config flags, metric scripts or the morning report, since §0a demands those changes answer its checklist.
---

# MD compliance audit

CLAUDE.md is injected into every session, so when the bot drifts from it, every
future session starts from a false picture — and the project's own §0a exists
because good-looking numbers repeatedly meant nothing here. This skill answers
one question with evidence: **does the running bot match what the docs require?**

## How to run it

### 1. Mechanical checks (always start here)

```
pyembed\python.exe files\_harness_check.py
```

It covers what a script can verify and exits 1 on drift:

| Check | §0a rule | What it catches |
|---|---|---|
| CLAUDE.md flags & dataset sizes vs reality | 9 | doc claims a value config no longer has |
| running bot vs advertised behaviour | 9 | full-watchlist claimed, 9 coins watched; bot silent |
| ratio metrics print a base rate / lift | 1 | "recall 100%" that hides a 73% fire rate |
| metric windows are uptime-aware | 5 | an outage counted as missed rockets |
| refuted backtests keep their verdict | 8 | negative results that will be re-tested |

`files\_audit_md_vs_config.py` is the narrower flag/size audit if only that is
needed.

### 2. Judgment checks (the script cannot do these)

Read the actual change or current state and answer §0a explicitly. Do not tick
these off from memory — open the numbers:

1. **Base rate published?** Any recall / coverage / hit-rate must appear next to
   its base rate and lift. A ratio alone is not evidence.
2. **Leaky features named?** If a model scores well, check whether an input can
   encode the label (this repo's `tg_return_since_open` vs a "top-20 by daily
   return" label). Say it in the same breath as the score.
3. **Holdout, split by time?** In-sample numbers are never an achievement.
4. **Comparable windows?** Endpoints must match in sample size, else the honest
   answer is "рано судить".
6. **Gate validated on the bot's OWN entries?** Market-wide episode studies
   describe a population the gated bot never samples.
7. **Flag + rollback + shadow first?** Behaviour changes only.
8. **Negative result committed** with the numbers that killed it.
10. **Report claims nothing the data does not support.**

### 3. Learning-loop obligations (§0)

- Is the change blocked by missing logging? Then it is a logging task first.
- Does it touch a loop component? Then
  `docs/specs/features/auto-improvement-loop-spec.md` must be updated.
- Are decisions recorded in `decisions.jsonl` / `already_tried.jsonl`?
- Is the training↔live gap still reported rather than collapsed into one number?

### 4. Spec sync

CLAUDE.md and PROJECT_CONTEXT.md duplicate on purpose. Any architecture, filter,
schedule or known-issue change must land in **both**, in the same commit.

## Reporting the result

Lead with the verdict — compliant or not — then list only violations, each with
the evidence and the rule number. State plainly what was checked mechanically
versus by reading. If a rule was knowingly violated, say so and give the reason;
silence is the failure mode §0a exists to prevent.

Do not claim compliance for anything you did not actually check.
