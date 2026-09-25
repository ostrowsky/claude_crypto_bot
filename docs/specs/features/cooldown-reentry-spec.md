# Post-exit cooldown re-entry — measured, refuted

- **Slug:** `cooldown-reentry`
- **Status:** REFUTED 2026-09-25. `COOLDOWN_BARS = 19` stays.
- **Truth-harness invariants:** TH-01, TH-04 (candidates removed from the
  control), TH-06, TH-08, TH-11 (trailed outcome beside the 4h peak), TH-13
- **Backtest:** `files/_backtest_cooldown_reentry.py`
- **Tests:** `files/test_cooldown_reentry.py`

## Why

ATOM 2026-09-06..09 was entered at 1.559, 1.621, 1.687 and 1.965 — each exit
followed by the 19-bar cooldown and a re-entry higher up. The exits themselves
were measured and kept (WEAK, RSI). The cooldown was the last open piece of the
"exit and re-entry" leak named in `lateness-caps-and-rsi-partial-exit-spec.md`.

## Method

Every 15m exit (4 066, 2026-03-03 … 09-25). Candidate: the first bar inside the
19-bar cooldown where a live entry rule fires under today's caps (FIRE rows of
`_backtest_lateness_caps.py`, on the repaired kline store). Outcome: ATR trail
from that bar's close. Control: 24 993 rule firings outside any cooldown, by the
same engine. Split by exit class. Rule layer only — live gates would still apply.

## Result

A rule fires again inside the cooldown after 1 395 exits (34%), typically +1.42%
above the exit price (94% above it).

| group | n | trailed mean | minus control, 95% CI |
|---|---|---|---|
| all cooldown re-entries | 1 394 | +0.04% | [−0.07, +0.16] |
| after an exit in profit | 607 | +0.16% | [−0.04, +0.37] |
| after an exit at a loss | 787 | −0.05% | [−0.19, +0.08] |
| after a WEAK exit | 400 | +0.13% | [−0.08, +0.37] |
| after an ATR-trail exit | 585 | +0.02% | [−0.16, +0.21] |

Control: trailed mean −0.00%, median −0.34%. No group excludes zero; medians sit
below the control's; months 3 positive / 4 negative. A re-entry inside the
cooldown is worth what any rule firing is worth. Lifting the cooldown would add
entries without edge.

## Shadow / canary and rollback

Not applicable (не применимо): nothing changes. Maximum-period evidence above.
