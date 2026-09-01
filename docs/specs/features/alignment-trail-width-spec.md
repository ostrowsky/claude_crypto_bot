# alignment trail floor: 0.0 → 8%

- **Slug:** `alignment-trail-width`
- **Status:** DEPLOYED 2026-09-01
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-06
  (measured on the bot's OWN entries, and that limit stated), TH-07 (behaviour
  change behind a flag with a stated rollback), TH-10 (numbers committed),
  TH-11 (capture is not the outcome — realized is printed beside it), TH-13
  (nothing asserted here that was not measured)
- **Flag:** `TRAIL_MIN_BUFFER_PCT_ALIGNMENT = 0.08` (was `0.0`)
- **Rollback:** set it back to `0.0` and restart. One number, no state, no retrain.

## Trigger

FILUSDT, 2026-09-01. The bot entered at **08:32 at 0.6939** — eight hours *before*
the run — and was stopped out at **12:48 at 0.7034 for +1.37%** on a trail whose
effective width was about 1.5%. The coin then reached **0.7848, +13.10% from the
entry**. The bot captured a tenth of a move it had identified early and correctly.

The entry was not the problem, and neither were the gates. The exit was.

## Sweep — over the maximum period available

The window is the maximum the event log supports: every real entry the bot has
made since 2026-03-01, 4 972 of them, 4 926 with enough forward klines to replay.
No sub-window was selected and none was discarded; the per-mode splits below are
the whole population cut by mode, not a chosen slice.


`_backtest_trail_width_sweep.py` replays the forward path of every real entry
since 2026-03-01 under a trailing stop set `w` percent below the running peak.
The stop is tested against each bar's low *before* that bar's high updates the
peak, so no single bar can both set a new peak and be forgiven the drawdown it
printed getting there.

**alignment, n = 1323** (mean realized % per trade):

| trail | 1.0% | 2.0% | 3.0% | 4.0% | 5.0% | 6.0% | 8.0% | HOLD |
|---|---|---|---|---|---|---|---|---|
| realized | +0.295 | +0.191 | +0.232 | +0.343 | +0.429 | +0.565 | **+0.631** | +0.551 |

Monotone above 2%, and 8% beats holding with no stop at all. alignment is the one
mode that wants a wide stop.

Every other mode was swept at the same time and **none is being changed**:

| mode | n | best width | realized there | live setting |
|---|---|---|---|---|
| impulse_speed | 1692 | 1.0% | +0.343 | 1.5% — already near optimal |
| trend | 809 | 1.0% | −0.020 | 0.0 — negative at every width |
| breakout | 185 | 8.0% | +0.030 | 0.0 — thin, ~zero |
| strong_trend | 110 | 1.0% | −0.025 | 0.015 — thin, negative |

`trend`, `breakout` and `strong_trend` lose money at *every* width. Their exit
rule is not what is wrong with them, so widening theirs would be treating the
wrong thing.

## Why this sweep is worth acting on

It reproduces a known live outcome it was never told about. In June 2026 the
`impulse_speed` buffer was widened 1.5% → 8% on the strength of a 35-day backtest
and **rolled back after a live regression** (avg/trade +0.02 → −0.62 over 5 days,
n=89). This sweep, run on a different window with a different method, says
`impulse_speed` at 8% yields **−0.206** against **+0.212** at 1.5%. It would have
predicted that rollback.

That is the closest thing to out-of-sample validation available for a method like
this, and it is the reason the alignment result is being acted on rather than
filed.

## Limits, stated because the precedent is a live failure

- The sweep replays a **pure percentage trail**. The live rule is
  `max(trail_k × ATR, floor)`. For alignment, `trail_k × ATR` has been producing
  roughly 1.5%, so an 8% floor becomes the effective width — but the two are not
  identical, and on high-ATR names the ATR term can still dominate.
- The sweep uses a fixed **48-bar horizon**; live `max_hold` is chosen per trade
  by the trail bandit.
- It replays **only entries the gates admitted** (TH-06). It says nothing about
  trades never taken, and the gate configuration changed materially on 2026-08-20
  and 2026-08-21, so the admitted population is currently moving.
- A backtest-positive trail widening has failed live in this repo before. This is
  a flag, and the rollback is one number.

## Shadow / canary decision

**No shadow.** A shadow run of an exit rule would have to hold a parallel set of
imaginary positions for hours to produce one observation, and the replay above
already answers the question it would ask, on 1323 trades instead of a handful.
What replaces it is the watch below, with a pre-committed rollback trigger.

## Acceptance and rollback trigger, set before the fact

Read after **20 closed alignment trades**:

- **Keep** if mean realized per alignment trade is ≥ 0 and above the pre-change
  baseline of +0.213% (the sweep's value at 1.5%).
- **Roll back** if mean realized is below −0.30% per trade, or if the median
  falls below the pre-change median of −0.488%.

Stated now so the decision cannot be re-argued from whichever number looks better
later — the same discipline the June rollback established.

## Verification

`test_trail_width.py` pins the floor, that only alignment moved, that the sweep's
stop ordering cannot let a bar forgive its own drawdown, and that the rollback
value is recorded in the config comment.
