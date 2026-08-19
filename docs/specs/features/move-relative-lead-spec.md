# Move-relative lead in the North Star (goal 2)

- **Slug:** `move-relative-lead`
- **Status:** shipped 2026-08-19, published beside v2
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-02 (the metric must measure the question),
  TH-04 (versioned, not silently redefined), TH-05 (a metric must know what it
  does not know)
- **Flag:** `NS_MOVE_RELATIVE_LEAD_ENABLED` — **True** since the side-by-side
  reading below was published; it adds a line and changes no behaviour
- **Rollback:** flip the flag; both lead definitions stay in the code

## Which goal this serves

**Goal 2 — signal entry as early as possible.** Not directly: it changes no
behaviour and catches no coin sooner. It makes goal 2 *measurable*, which is the
precondition for working on it. Nothing else in this spec claims progress.

## Problem

The North Star multiplies `coverage × capture × lead`, and

```python
time_lead = 1.0 - (edt.hour / 24.0)
```

That is the **clock hour**, not earliness relative to the move. The consequences
are not subtle:

- a coin that starts moving at 20:00 UTC and is alerted at 20:05 — as early as
  the bot could possibly be — scores **0.17**;
- a coin bought at 02:00 that does nothing for eleven hours scores **0.92**.

So the factor rewards being early in the calendar and penalises being early in
the move, which is the opposite of the objective. Progress on goal 2 cannot be
seen through it, and a change that genuinely improved entry timing could lower
the North Star.

## Change

`lead` becomes earliness against the move the coin actually made:

```
anchor         first crossing of +5% from the UTC open   (label_store.anchor_ts)
early deadline first crossing of +2.5%                   (label_store.early_deadline_ts)

lead = 1.0                          entry at or before the day's open
     = 1 - (t_entry - t_open) / (t_deadline - t_open)     entry before the deadline
     = 0.0                          entry at or after the deadline
```

Both timestamps already exist in the immutable store and already drive
`weekly_steering`. Nothing new is fetched or inferred.

**Only `resolution == "1h"` records carry crossing times.** A winner whose label
is daily-resolution has no deadline, so its lead is **not computable** — the
day is excluded from the move-relative figure and counted, not silently scored
as 0.0 (TH-05). A zero would read as "alerted late", which is a claim about the
bot rather than about the data.

## Versioned, published beside the old one

The metric becomes `NS_EarlyCapture_top20_v3` when the flag is on, with `v2`
still emitted. The `_v2` rename already blanked a report once by breaking four
lookups keyed on the old name; `_north_star_metric()` now resolves the newest
version, and this change is the first test of that.

Expect the value to **fall**, possibly a lot. Under the clock definition the bot
scores 0.61 on lead; measured against the move, `weekly_steering` already says
only 2.9% of qualifying moves are alerted before the deadline. A large drop is
the metric starting to answer the right question — it is not the bot getting
worse, and the report must say so or the next reader will revert it.

## What must be reported, not hidden

**The two leads are not comparable and the metric is renamed for that reason.**
`v2` answers "how early in the day", `v3` answers "how early in the move". A
series that switches definitions without renaming would show a collapse that
looks like a regression.

**Coverage and capture are untouched**, so any change in the North Star between
v2 and v3 is entirely the lead factor. That makes the size of the definitional
error directly readable, which is the point of publishing both.

## Verification

`test_move_relative_lead.py`:

1. an entry at the open scores 1.0; at the deadline, 0.0; halfway, 0.5;
2. an entry after the deadline scores 0.0, never negative;
3. a winner with no `early_deadline_ts` (daily-resolution label) is **excluded
   and counted**, not scored 0.0;
4. a zero-length window (deadline == open) does not divide by zero;
5. with the flag off, the emitted lead is byte-identical to today's;
6. the emitted metric carries `lead_definition` so no reader has to guess.

**Maximum-period evidence** (`_backtest_move_relative_lead.py`): recompute the
North Star both ways over every day both definitions cover, publishing value,
the three factors, n, and how many winner-days had no computable deadline.

**Shadow/canary: не применимо** — this changes a measurement, not behaviour.

## First reading — the clock lead overstated earliness ~30x

30-day window, immutable labels, same winners and same entries on both lines:

```
EarlyCapture@top20_immutable   0.141   cov 0.80   cap 0.29   lead 0.61   n=25
EarlyCapture@top20_move_lead   0.008   cov 0.67   cap 0.27   lead 0.02   n=15
```

Coverage and capture barely move, so **the entire gap is the lead factor:
0.61 → 0.02.** Measured against each coin's own move, the bot's alerts arrive
essentially at or after the +2.5% crossing. On goal 2 the bot scores **0.02 out
of 1.0**, and the clock-hour definition was reporting 0.61 for the same
behaviour.

`n` falls from 25 to 15 because ten winner-days carry a daily-resolution label
with no crossing time. They are counted in
`move_lead_winners_without_deadline`, not scored zero.

**Three independent measurements now agree**, which is why this is not a
plumbing artifact:

| measurement | reading |
|---|---|
| move-relative lead | 0.02 |
| `Coverage@move` (alerted before the +2.5% deadline) | 0.029 |
| `Precision@alert` on early alerts | lift **0.72x** — worse than chance |

The bot identifies movers *after* they have moved ~2.5% and is no better than
random before. That is a statement about the seven entry modes, every one of
which requires ADX / slope / volume / breakout confirmation — by construction
none of them can fire ahead of the move. It is not a tuning problem.
