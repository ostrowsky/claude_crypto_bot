# daily-range floor: 4% stays, 2% rejected

- **Slug:** `range-min-threshold`
- **Status:** MEASURED 2026-09-03 — **negative, not applied**
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-06
  (measured on the bot's own candidates), TH-08 (negative results committed with
  the numbers that killed them), TH-10, TH-13
- **Flags:** unchanged — `TREND_15M_RANGE_MIN = 4.0`,
  `ALIGNMENT_15M_RANGE_MIN = 4.0`, `ALIGNMENT_1H_RANGE_MIN = 5.0`
- **Rollback:** not applicable, nothing was changed.

## The question

TRXUSDT, 2026-09-03. The chart read as a clean staircase and the guard had
rejected it 18 times. The measurement said otherwise: **1.38% over 24h**, +0.20%
from 06:00, and 91st of 102 in the watchlist that day. The chart's y-axis spanned
about 3.8%, so a one-percent move filled a third of the screen.

The bot was right and the picture was misleading — the first time that way round
in this session. But the reasonable follow-up stood: should the floor be lower?

## Measurement

`_backtest_range_min_threshold.py`, 88 days, 16 490 decisions deduplicated by
symbol-hour, forward peak 8h out. Rejected candidates carry `daily_range` in the
event log, so the bands can be read directly.

| band | n | /day | median | p75 | >3% | >5% |
|---|---|---|---|---|---|---|
| **POOL** (all candidates) | 16 290 | 185.1 | 1.22% | 3.13% | 26% | 14% |
| **ENTRIES TAKEN** | 1 843 | 20.9 | 1.52% | 3.48% | 29% | 16% |
| 0–1% | 41 | 0.5 | 0.52% | 1.46% | 12% | 10% |
| 1–2% | 479 | 5.4 | 0.65% | 1.50% | 12% | 6% |
| 2–3% | 995 | 11.3 | 0.72% | 1.66% | 11% | 5% |
| 3–4% | 1 076 | 12.2 | 0.95% | 2.09% | 16% | 7% |
| **2–4% combined** | **2 071** | **23.5** | **0.83%** | **1.91%** | **14%** | **6%** |

## Verdict: no

The band a 2% floor would admit produces big moves at **half the rate of what the
bot already buys** — 14% against 29% above +3%, 6% against 16% above +5% — while
adding **23.5 candidates a day** against the current 20.9 entries. The flow would
roughly double, at half the quality, competing for the same `MAX_OPEN` slots.

Two baselines and not zero, deliberately: against POOL the band shows whether it
beats a coin-blind policy in the same hours; against ENTRIES TAKEN, whether it is
as good as what already gets bought. It fails both.

## What the bands also show

The ordering is **monotone** in forward move: 0–1% → 0.52%, 1–2% → 0.65%,
2–3% → 0.72%, 3–4% → 0.95%. `daily_range` is genuinely related to what happens
next, so the guard's premise holds rather than being a number fitted once and
inherited. The guard's own docstring credits a precision study (+17.1pp, 29.9% →
47.0%); this measures the same threshold against the operator's stated goal
instead — the **size** of the move rather than the hit rate — and lands in the
same place.

## Why this matters beyond the one threshold

Three gates were measured the same way this week with the same script family:

| gate | rejects vs pool | verdict |
|---|---|---|
| `ml_zone` (segment-routed) | blocked coins that then ran +19% | **fixed** |
| bandit | rejects beat the bot's own entries (1.52 vs 1.41) | **disabled** |
| `mode_range_quality` | rejects clearly worse than pool | **left alone** |

The test that convicted the first two acquits this one, which is the point of
having a test rather than an opinion.

## Verification

`test_range_min_threshold.py` — the measurement's own failure modes: dedup by
symbol-hour so a guard cannot weight itself by poll frequency, both baseline rows
present, band edges half-open so no candidate is counted twice, and thin bands
suppressed rather than reported as findings.
