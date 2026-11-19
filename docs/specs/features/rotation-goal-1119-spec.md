# Rotation when the portfolio is full, judged by the goal (2026-11-19)

- **Slug:** `rotation-goal-1119`
- **Status:** research, pre-registered. No behaviour change.
- **Origin:** the [gpt_crypto_bot design audit](../../reports/2026-11-19-gpt-bot-experiment-design-audit.md),
  hypothesis 1.
- **Truth harness:** TH-01, TH-03, TH-05, TH-06, TH-08, TH-10, TH-11, TH-12
- **Script:** `files/_backtest_rotation_goal.py`. **Tests:** `files/test_rotation_goal_1119.py`

## Why

ML-gated rotation (`rotation.should_rotate`) evicts the weakest leg for a
candidate with `ml_proba ≥ ROTATION_ML_PROBA_MIN = 0.62`. The threshold was set
on the old ML scale. After the 2026-09-07 peak-label switch, the share of
portfolio-blocked candidates at or above it fell from 32% (2026-04) to 0–2%
(2026-08..11). Only 5 ML evictions appear in the logs kept since 09-29, and
score-based replacements fell from about 100 a month to about 5. Rotation has
switched itself off without any decision to do so. This test asks whether it
should come back.

## Fixed before the run

**Population.** Every candidate in `critic_dataset.jsonl` that is blocked at
the portfolio limit (`decision.stage = portfolio`, `reason_code = portfolio`).
Such a candidate has passed every other gate, correlation_guard included.
- Rows are deduplicated per (sym, tf, bar).
- The decision is at the bar close.
- Maximum period: the first such row (2026-03) through the last resolved
  label day.

**Open positions** come from `bot_events.jsonl` entry/exit pairs
(`readouts.pair_trades`): a position is open on `[entry ts, exit ts)`. A
position is eligible for eviction at time t when all of these hold (the live
`find_weakest_leg` rules):
- the bars held in its own timeframe are ≥ `ROTATION_WEAK_BARS_MIN` (3);
- its unrealized PnL at the close of the bar before t is < `ROTATION_PROFIT_PROTECT_PCT` (0.5%);
- it was not switched to leader mode (`leader_exit_switch`) before t;
- its entry `ranker_ev` is < `ROTATION_WEAK_EV_MAX` (−0.40). A missing ranker_ev
  makes the position ineligible.

The weakest leg is the eligible position with the lowest ranker_ev.

**Arms.** In both, a candidate evicts the weakest eligible leg if one exists.

| arm | candidate qualifies when |
|---|---|
| `ROT_ALL` | always (it has already passed every gate) |
| `ROT_TOP25` | its `ml_proba` is in the top 25% of the month's portfolio-blocked rows. 25% restores the selectivity of 0.62 on the old scale (275 of 1,081 rows in 2026-03..07), i.e. matched admission |

**Sequencing.** Rows are processed in time order.
- **Eviction.** The evicted position is closed at the close of the decision bar.
  Its P&L becomes its P&L at that moment.
- **The candidate's trade.** It enters at the same close and is simulated with
  the calibrated live trail (`goal_validator.live_trail`): `trail_k` is the
  median of the logged entries, the floor is its mode's, and `max_bars` is 96 on
  15m and 48 on 1h.
- **After the trade.** The candidate blocks further rows of its own symbol until
  its exit plus `COOLDOWN_BARS`.
- **Limits on eviction.** A position is evicted at most once. Added positions
  are never evicted.
- **Not modelled:** a candidate holding its slot longer than the evicted leg
  would have, which could block later real entries. The count of such cases is
  reported.

**Goal.** Immutable global-top-20 winner-days on full-uptime days with a known
crossing (`goal_validator.winner_days`).
- **Gained day:** a winner-day with no real entry before the crossing, but with
  an added entry in `[day open, crossing)`. The downstream pass rate is 1: the
  portfolio limit is the last gate before the entry.
- **Δ** = gained / winner-days, with a 97.5% bootstrap over winner-days.
- **Capture lost** (reported): evicted positions that were early entries on a
  winner-day, and their P&L given up.

**Per trade.** Every trade is measured on the same live-trail engine. The
current arm is all real entries replayed with their own `trail_k`. Combined
minus current means:
- evicted trades take their P&L at eviction;
- added trades are appended.

The interval is a 97.5% day-cluster bootstrap.

**Context, never the verdict:**
- the tail: max drawdown of the cumulative daily sum and the worst day;
- months with Δ ≥ 0;
- the 2026-09-07 .. end subperiod (the new ML scale).

**Decision per arm (2 arms, Bonferroni: 97.5%).**

| verdict | condition |
|---|---|
| **needs_data** | < 30 winner-days or < 20 added trades |
| **supported** | Δ ≥ +0.5 pp AND its lower bound > 0 AND per-trade lower bound ≥ −0.10 pp |
| **refuted** | Δ ≤ 0 OR per-trade lower bound < −0.10 pp |
| **inconclusive** | otherwise |

## Results (run 2026-11-19)

**Data.**
- 1 493 portfolio-blocked candidates, 2026-03-30 .. 11-13.
- 6 014 real closed trades: 4 509 with a known ranker_ev, 2 498 of them below −0.40.
- 6 001 trades replayed on the live trail, `trail_k` 2.00.
- 599 winner-days.

| arm | evictions / added trades | added mean | goal Δ [97.5%] | per trade [97.5%] | evicted early winner-day legs | verdict |
|---|---|---|---|---|---|---|
| `ROT_ALL` | 202 / 202 | −0.282% | **+0.00 pp** (0 of 599) | −0.014 [−0.052, +0.018] | 2, P&L given up −6.06 pp | **refuted** |
| `ROT_TOP25` | 73 / 73 | +0.408% | **+0.00 pp** (0 of 599) | −0.006 [−0.038, +0.019] | 1, −3.97 pp | **refuted** |

Most rows found no weak leg: 1 244 of 1 493 in `ROT_ALL`, 293 of 374 in
`ROT_TOP25`. The tail is context only: max drawdown −1 387 → −1 526 / −1 441;
the worst day does not change.

**Why rotation cannot move the goal — a structural ceiling** (diagnostic run
after the verdict, not part of the gate). Only 3 of the 599 winner-days had a
portfolio-blocked candidate before their crossing. The bot had no real early
entry on just 2 of them (2026-09-18 ARUSDT and INJUSDT). Of the 34 winner-days
with a portfolio-blocked candidate at any time, 31 saw it only after the
crossing. Perfect rotation could therefore gain at most 2/599 = 0.33 pp, below
the +0.5 pp bar.

**The portfolio limit is not a cause of late entry in this bot.** The
neighbour's capacity regret (79 objective pairs in 29 days) does not transfer:
its bot ran ten slots at about 59 trades a day; ours rarely fills before
leaders cross. Evicting also costs: the few evicted early winner-day legs gave
up 4–6 pp each.

**Side finding, no action taken.** The 0.62 ML rotation threshold is stale: it
was set on the pre-2026-09-07 ML scale, which leaves ML-gated rotation
practically inert (0–2% of blocked candidates qualify). The evidence above says
re-activating it would not advance the goal and is about neutral per trade. The
config stays unchanged; this note keeps the inert state from being mistaken for
a working gate.
