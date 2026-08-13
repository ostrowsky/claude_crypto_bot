---
name: md-compliance
description: Audit whether the crypto bot actually complies with the requirements written in CLAUDE.md and PROJECT_CONTEXT.md — config flags, monitored set, metric honesty rules (§0a truth harness), learning-loop obligations (§0), and whether the running process matches what the docs claim. USE THIS SKILL whenever the user asks to check compliance with the MD / спеки / требования, asks "соответствует ли бот MD", "проверь соответствие требованиям", "аудит бота", "проверь спеки", asks whether a planned or completed change respects the documented rules, or before shipping a change that touches behaviour, gates, metrics or reports. Also use it proactively when about to modify config flags, metric scripts or the morning report, since §0a demands those changes answer its checklist.
---

# MD compliance audit

CLAUDE.md is injected into every session, so when the bot drifts from it, every
future session starts from a false picture — and the project's own §0a exists
because good-looking numbers repeatedly meant nothing here. This skill answers
one question with evidence: **does the running bot match what the docs require?**

## 1. Mechanical gates — always start here

```bash
bash .claude/skills/md-compliance/scripts/check.sh
```

Add `--staged` to audit only a staged diff (the pre-commit profile). Exit 0 is
clean, 1 is drift, 2 means the harness itself broke — a broken harness is not a
pass. Findings also land in `.runtime/harness_last.json`.

It covers doc-vs-config flags and dataset sizes, the running bot vs its
advertised behaviour, ratio metrics printed without a base rate, metric windows
that ignore downtime, and refuted backtests that lost their verdict.

## 2. Judgment checks — the script cannot do these

Read `reference.md` and answer its ten rules against the actual change and the
actual numbers. Do not tick them off from memory; open the numbers. In short:
base rate beside every ratio, name the leaky feature before quoting the score,
holdout split by time, comparable windows or "рано судить", downtime as no-data,
gates validated on the bot's own entries, flag + shadow before enforce, negative
results committed with the numbers that killed them, docs matching reality, and
a report that claims nothing more than the data supports.

`reference.md` carries the failure behind each rule — those specific memories are
what make the rules stick, so read it rather than working from the summary above.

Then check the §0 learning-loop obligations and CLAUDE.md ↔ PROJECT_CONTEXT.md
spec sync, both detailed at the end of `reference.md`.

## 3. Reporting the result

Lead with the verdict — compliant or not — then list only violations, each with
its evidence and rule number. State plainly what was checked mechanically versus
by reading. If a rule was knowingly violated, say so and give the reason; silence
is exactly the failure mode §0a exists to prevent.

Do not claim compliance for anything you did not actually check.
