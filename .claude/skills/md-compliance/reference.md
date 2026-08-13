# MD compliance — the judgment checks in full

`scripts/check.sh` covers what a script can verify. Everything below needs
reading the actual change and the actual numbers. Each rule carries the failure
that put it in CLAUDE.md §0a, because the abstract version of the rule never
stopped anyone here — the concrete memory does.

## Contents

- [1. Base rate and lift](#1-base-rate-and-lift)
- [2. Leaky features](#2-leaky-features)
- [3. Holdout split by time](#3-holdout-split-by-time)
- [4. Comparable windows](#4-comparable-windows)
- [5. Uptime-aware metric windows](#5-uptime-aware-metric-windows)
- [6. Validated on the bot's own entries](#6-validated-on-the-bots-own-entries)
- [7. Flag, rollback, shadow first](#7-flag-rollback-shadow-first)
- [8. Negative results keep their verdict](#8-negative-results-keep-their-verdict)
- [9. Docs match reality](#9-docs-match-reality)
- [10. The report claims only what the data supports](#10-the-report-claims-only-what-the-data-supports)
- [Learning-loop obligations (§0)](#learning-loop-obligations-0)
- [Spec sync](#spec-sync)

---

## 1. Base rate and lift

Any recall, coverage, hit rate or precision appears next to the base rate and
the lift, in the same sentence. A ratio on its own is not evidence.

**What it caught here:** the bandit was reported at "recall@20 = 100%" — it
fired ENTER on 100% of top-20 coins, and on 73.3% of everything else. Lift
1.36×, not a solved problem. Same shape in a gate study: "keeps 50% of big
movers" from a gate that keeps half of all entries by construction.

Check: does the number have a denominator a reader can question? If the gate
admits X% of everything and catches X% of winners, it has done nothing.

## 2. Leaky features

If a model scores unusually well, find the input that can encode the label
before reporting the score — and say it in the same breath.

**What it caught here:** `top_gainer_model` at AUC 0.99 against a label of
"the day's return landed in the top-20", while `tg_return_since_open` sat in the
feature set. At the 00 UTC snapshot a labelled rocket already showed +13.98%
against an `eod_return_pct` of +13.94% — the same number on both sides. The
model was confirming a finished move, which is the opposite of the product goal.

Related and worth re-checking whenever `label_top20` is used: the field does not
mean the same thing at every snapshot hour. ~12.9 positives/day at a median
+2.55% at 06 UTC, against ~1.5/day at +13.7% at 12/18 UTC. 84% of all positives
come from the loose 06:00 labelling.

Check: for each feature, could it be computed only *after* the outcome existed?
Sanity-print the feature's distribution split by label.

## 3. Holdout split by time

In-sample numbers are never an achievement, and a random split leaks across a
non-stationary market. Split by day, train on the earlier days.

**What it caught here:** impulse_speed's multivariate entry model — train AUC
0.60, out-of-sample 0.50. Without the temporal split it would have shipped.

## 4. Comparable windows

Two endpoints must match in sample size before their difference means anything.
When they do not, the honest verdict is "рано судить" and the report says so.

**What it caught here:** the morning report announced "СТАЛО ХУЖЕ" by comparing
a full window against one built from two working days.

## 5. Uptime-aware metric windows

A day the bot was down is "no data", never a missed rocket.

**What it caught here:** an 8-day outage (07-23..07-31) counted every top-20 of
those days as a miss, and the report read as a collapse. Restricted to days the
bot was actually alive: coverage 67%, silent-miss 8% — the best on record.

## 6. Validated on the bot's own entries

A market-wide episode study describes a population the gated bot never samples.
Before a gate ships, replay it against the bot's actual entries.

**What it caught here:** the weak-extension entry filter looked strong on
market episodes and did not transfer to the bot's entries at all.

## 7. Flag, rollback, shadow first

Behaviour changes only. New behaviour lands behind a boolean whose default
matches current live behaviour, ships in shadow, and is promoted on evidence.
Rollback is flipping the flag.

**What it caught here:** hard-block curtailment went live without a shadow
period and became the #1 block (266–813/day), starving entries to 1–10/day
during an altseason.

## 8. Negative results keep their verdict

A refuted hypothesis is committed with the numbers that killed it, in the script
that killed it. Otherwise it gets re-tested every few weeks.

Repo examples worth not repeating: discovery cadence, the 1d trend alert, the
stack-conditioned hold modifier, the weak-extension filter, RM-22 Step B.

## 9. Docs match reality

CLAUDE.md is injected into every session, so drift there makes every future
session start from a false picture. `scripts/check.sh` verifies flag values,
dataset sizes and the running bot's behaviour against what the docs claim.

## 10. The report claims only what the data supports

The morning report is the product surface for all of the above. A verdict it
cannot support must appear as "рано судить" or "нет данных", not as a trend.

---

## Learning-loop obligations (§0)

- Blocked by missing logging? Then it is a logging task first — do not
  speculate from incomplete state.
- Touches a loop component? `docs/specs/features/auto-improvement-loop-spec.md`
  must be updated in the same change, including its status row and the North
  Star progress table.
- Decisions recorded in `decisions.jsonl` / `already_tried.jsonl`? Note that
  logging a decision auto-applies its `diff.to` as a runtime override, so a
  rollback needs a superseding `rolled_back` record — editing `config.py` alone
  does not undo it.
- Is the training↔live gap still reported as two numbers rather than collapsed
  into one?

## Spec sync

CLAUDE.md and PROJECT_CONTEXT.md duplicate on purpose: the first is the working
brief, the second the human dossier. Architecture, filter, schedule and
known-issue changes land in **both**, in the same commit.
