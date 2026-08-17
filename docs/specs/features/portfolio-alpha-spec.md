# Portfolio alpha vs buy-and-hold (TH-11)

- **Slug:** `portfolio-alpha`
- **Status:** shipped 2026-08-17
- **Truth-harness invariants:** TH-11 (proxy is not the business outcome),
  TH-01 (base rate beside the ratio), TH-04 (comparable windows)
- **Rollback:** read-only reporting; nothing to roll back

## Problem

The canonical scorecard has read `portfolio_alpha: unknown` because its only
source was a signal-evaluator report generated **2026-05-04** — 105 days stale.
TH-11 exists because no proxy answers "is this worth acting on": training AUC,
recall and per-mode P&L can all look fine while the alert stream loses money.

## What this is and is not

**A diagnostic, not an optimisation target.** The bot is an alert system with no
position sizing, and the standing objective is early detection and coverage, not
per-trade income. Computing this number does not make it the thing to maximise —
it answers one question the proxies cannot, and the answer feeds judgement about
the entry path, not a threshold sweep.

## Method

A mean of trade P&Ls is not a portfolio return: twenty overlapping +1% trades on
the same capital are neither +20% nor +1%. Capital is modelled as `MAX_OPEN = 10`
equal slots, each closed trade moving one slot's worth, compounded by UTC day.
This is the smallest model that makes the two sides comparable.

The benchmark is the equal-weight watchlist held across the same window, priced
from the **immutable label store** — so the bot's outcome and its benchmark are
not measured against two different price sources.

Stated limits: intraday slot availability is not reconstructed; positions still
open at the window edge are excluded rather than marked to market; and the
watchlist is fixed, so the benchmark carries no survivorship correction.

## Result

```
window   trades   win rate   mean/trade      bot   buy-and-hold     ALPHA
 30d        415     32.77%     -0.343%   -13.45%       -7.21%     -6.24%
 60d       1088     36.31%     -0.305%   -28.66%      -14.09%    -14.56%
 90d       1676     36.87%     -0.275%   -37.53%      -25.69%    -11.84%
180d       4221     39.75%     -0.106%   -38.05%      -22.08%    -15.97%
```

**Alpha is negative on every window, and the market direction does not explain
it.** Buy-and-hold was also negative, so the honest statement is not "the bot
lost money in a falling market" — it is that acting on the alerts lost
**6 to 16 points more** than holding the same coins and doing nothing. The result
is stable across four windows, so it is a property of the strategy in this
period, not a regime artifact.

This corroborates the gate-lock replay from the same day, which found the bot's
own entries averaging **−0.883%** forward return over 64 events in the current
policy epoch against −0.073% across the max period.

**What it does not say.** It does not say the gates are wrong, that a particular
mode is at fault, or that earliness is the wrong objective. It says the entry
population as a whole is currently worse than the universe it selects from, which
makes the entry path — not the exit tuning and not the metric plumbing — the
place to look next.

## Verification

`test_portfolio_alpha.py`:

1. an ISO-string timestamp and an epoch-millisecond timestamp both resolve to
   the same UTC day;
2. a mean of trade P&Ls is not reported as the portfolio return — overlapping
   trades on one day compound through the slot model;
3. the benchmark reports how many watchlist symbols it could actually price;
4. a window with no closed trades returns `available: False` rather than 0%;
5. every emitted ratio carries its denominator.

## Findings from the review

**The timestamp shape silently emptied the input.** `bot_events` exit records
carry `ts` as an ISO string; assuming a number dropped all 4 221 of them and the
report read "no closed trades" for a bot that had closed one minutes earlier.
This was the **second** occurrence of the same class of bug in one session — the
gate replay had just been fixed for reading `ts`/`ts_ms` instead of `ts_signal` —
so the parsing now handles both shapes in one place with the reason recorded.
