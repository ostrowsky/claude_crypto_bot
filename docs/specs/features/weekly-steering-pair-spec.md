# Weekly steering pair — `Coverage@move` and `Precision@alert`

- **Slug:** `weekly-steering-pair`
- **Status:** spec → implementation
- **Created:** 2026-08-14
- **Parent:** [`continuous-improvement-agent`](continuous-improvement-agent-spec.md) §11.2
- **Consumes:** [`immutable-label-store`](immutable-label-store-spec.md)
- **Truth-harness invariants:** TH-01 (ratio context), TH-04 (comparable
  windows), TH-05 (downtime is not a miss), TH-10 (no unsupported conclusion)
- **Rollback:** delete the module; nothing else depends on it

## Problem

Asked whether trading results have improved, the honest answer today is that no
instrument can tell. The North Star needs a change of **2.1× its own value** to
be visible at a weekly comparison — it can only see the metric tripling or
collapsing. Everything smaller is invisible, which is the same as having no
measurement at all.

The label store now provides ~16 qualifying events per day, about **114 a
week**, against ~20 top-20 winners. That is the sample size the mission metric
never had.

## Definition — one event, two rates

Both metrics read the same `MoveEvent` record and the same boundaries, so
improving one at the other's expense is visible rather than hidden.

```
event         (symbol, UTC day) with a label record
qualifies     max_move_pct >= +5%          from the day's high
deadline      early_deadline_ts            first +2.5% crossing, fixed
alert         earliest bot entry event for that symbol that day
eligible      alert exists AND (no deadline OR alert_ts < deadline_ts)
```

| metric | numerator | denominator |
|---|---|---|
| `Coverage@move` | qualifying events with an **eligible** alert | qualifying events |
| `Precision@alert` | eligible alerts whose event qualifies | eligible alerts |

**Both are reported, always.** Coverage alone rewards alerting on everything;
precision alone rewards alerting on nothing. They are the recall/precision pair
of one definition, which is what makes the trade visible.

### The gaming vector, closed by design

Precision counts only *eligible* (early) alerts. On its own that would reward
alerting **late**: a late alert is excluded from the denominator, so it cannot
hurt precision even when it is wrong. So `Precision@alert_all` is reported
beside it, over every alert regardless of timing. A divergence between the two
means the bot is buying precision with lateness, which is the opposite of the
mission.

### Resolution limit, stated

Hourly bars place the deadline at a bar open, not a minute. An alert inside that
same hour cannot be ordered against the crossing and is counted **late** —
conservative against the bot. Sharpening needs 15m bars.

### Downtime is not a miss

A day is scored only when the bot was observably alive for it: at least
`MIN_ACTIVE_HOURS = 18` distinct hours carrying any event. Otherwise the day is
`no_data` and leaves both numerator and denominator (TH-05). An 8-day outage
once read as a performance collapse; that must not recur.

## Statistics

- Every figure ships with `n`, base rate, and a **day-clustered bootstrap CI**.
  Resampling rows would treat 98 symbols on one day as 98 independent facts;
  they share market beta, so days are the resampling unit.
- Comparing two windows uses the **observed CI against a pre-registered
  practical-significance threshold**, never MDE — MDE is a design parameter, not
  a decision rule on observed data.
- When the interval spans the threshold the verdict is `INSUFFICIENT_EVIDENCE`,
  printed as "рано судить" with the interval. No arrow is drawn.
- Windows must be uptime-matched: comparison requires a similar count of scored
  days, or it is refused.

## Verification

`test_weekly_steering.py`:

1. an alert strictly before the deadline is eligible; one in the same hour is
   not (conservative);
2. a qualifying event with no alert lowers coverage but does not touch
   precision;
3. an eligible alert on a non-qualifying event lowers precision but does not
   touch coverage;
4. alerting late raises `Precision@alert` while `Precision@alert_all` falls —
   the gaming vector is observable;
5. a day below `MIN_ACTIVE_HOURS` is excluded from both, not scored as a miss;
6. every result carries `n`, base rate and a CI;
7. the bootstrap resamples days, not rows — a single-day input yields a
   degenerate interval rather than a falsely tight one;
8. comparing two windows returns `INSUFFICIENT_EVIDENCE` when the CI spans the
   threshold;
9. comparing windows of very different scored-day counts is refused.

## First reading — and what it says

`as_of 2026-08-14`, last 14 days, 10 scored days (3 excluded as downtime),
950 events of which 99 qualifying:

```
Coverage@move   (alerted before the deadline)  0.051  [0.013, 0.091]  n=99
Precision@alert (early alerts)                 0.109  [0.026, 0.180]  n=46   lift 1.04x
Precision@all   (every alert)                  0.413  [0.341, 0.469]  n=172  lift 3.96x
```

**The gap between the two precisions is the finding.** Of 172 alerts only 46
were before the deadline, and those early ones carry **lift 1.04× — the base
rate**. The late ones look accurate (3.96×) because by the time they fire the
move is already visible. The metric built to catch a gaming vector instead
diagnosed the bot: it confirms moves rather than predicting them, and now that
is measured on immutable exchange labels with day-clustered intervals, on the
alert population, rather than inferred.

Caveats that belong beside those numbers: 10 scored days is thin, the
same-hour rule is deliberately conservative, and +2.5% is a hard early bar —
roughly half the qualifying move. This is a level, not a trend; a trend needs a
second comparable window.

## Findings from the architecture review

**Coverage was reporting an invented lift.** Its base rate was set to 1.0 so it
could share the metric struct, and the renderer duly printed "lift 0.05x" — the
value divided by one, wearing a label it had not earned. Coverage's denominator
*is* the qualifying events; it has no base to lift against. It now reports
`None` and prints "(базы нет)".

**A day with no events at all was counted nowhere.** Days below the activity
floor were excluded and reported, but a day entirely absent from the event log
fell out of both sets, quietly shrinking the window instead of reporting an
outage. Now any day the labels know about is either scored or explicitly counted
as downtime.

## Explicitly not claimed

This measures alert timing against a `+5%` intraday move. It is a **steering
proxy**, not the mission: a +5% mover is not necessarily a day's top-20 rocket.
The relationship between this pair and canonical top-20 outcomes must be
published before any positive progress verdict rests on it. No trading behaviour
changes.
