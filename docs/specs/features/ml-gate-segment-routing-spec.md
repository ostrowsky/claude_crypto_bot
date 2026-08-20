# ml_zone gate: segment routing off, floors to 0.10

- **Slug:** `ml-gate-segment-routing`
- **Status:** DEPLOYED 2026-08-20 (live restart 21:41 UTC)
- **Truth-harness invariants:** TH-01 (base rate beside every ratio), TH-03
  (holdout split by time), TH-04 (comparable populations), TH-06 (validate on the
  bot's own candidates), TH-10 (results committed with their numbers), TH-11
  (proxy is not the outcome), TH-12 (evidence travels with the change)
- **Flags:** `ML_GENERAL_USE_SEGMENT_WHEN_AVAILABLE = False`,
  `ML_SIGNAL_SEGMENT_ROUTING_ENABLED = False`,
  `ML_GENERAL_HARD_BLOCK_MIN = 0.10`, `ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN = 0.10`
- **Rollback:** set both flags to `True` and restore the floors to `0.28 / 0.22`,
  then restart. One-line revert, no state migration, no retrain.

## The failure

On 2026-08-20 the gate admitted **zero of 4486** candidates while the market rose
all day. Coins it blocked and what they then did:

| coin | blocked | price then | peak after |
|---|---|---|---|
| XRP | 08:17 | 1.1285 | **+19.10%** |
| ORDI | 07:32 | 3.742 | **+18.84%** |
| ENA | 08:00 | 0.0939 | **+17.69%** |
| FIL | 08:30 | 0.6816 | +13.22% |
| WIF | 07:16 | 0.1489 | +11.48% |

Across all 83 blocked coins: median peak after the block **+3.21%**, 51% rose
more than 3%, 25% more than 5%. Seven watchlist coins reached Binance spot's
rolling-24h top-50 that day (ENA 8th, XRP 16th, WIF 19th, ORDI 20th, FET 34th,
BONK 40th, ZRX 45th) and the bot signalled none of them. This is the CLAUDE.md §7
condition verbatim: a filter blocking the eventual winners is broken.

## What it was not

- **Not the nightly retrain.** The model is healthy: median **0.4053** on its own
  training population, 75% of rows clearing the 0.22 floor, and 0.35–0.58 at mean
  features across the global and segment models.
- **Not missing features.** All 60 feature names are populated; a real training
  row scores 0.3301.
- **Not a degenerate scaler.** No zero-variance divisions; the two
  constant-in-training features carry weight exactly 0.

## What it was

**Per-segment routing.** `monitor.py::_select_ml_payload` sends each candidate to
a model trained on its `signal_type|regime` subset. Those subsets can be as small
as 160 rows, and a segment is promoted to production whenever it beats the
baseline on **its own validation** — a bar a small sample clears by luck. The
deployed `trend|bull` draw was such a segment. On the day's rows:

```
deployed, with routing     median 0.0713    6% above the floor
deployed, global only      median 0.2509   69% above the floor
```

The day's candidates were `trend|bull` (415) and `alignment|bull` (132), so one
unlucky segment model decided the entire day.

Why the level collapses in exactly this regime: the model's largest weights are
**negative on momentum sequences** (`seq_trend_slope` −0.198,
`seq_trend_macd_hist_norm` −0.094, `seq_trend_rsi` −0.076). Trained on "does
price rise over the next 5 bars", it has learned mean reversion. In a broad rally
that description fits every coin at once.

Measured against its own target that day it was simply wrong: it assigned ~6% to
these candidates rising and **35 of 37 rose** (median +1.79%).

## Maximum-period backtest

`_backtest_ml_gate_policy.py`, walk-forward over **37 339 labelled candidate
rows** (2026-03-24 … 2026-08-20), four folds, model retrained on everything
before each cut and applied to what follows, scored **with routing** so live
behaviour is reproduced.

| policy | admitted | rate | mean r5 | recall ≥3% | recall ≥5% |
|---|---|---|---|---|---|
| NO GATE | 18 670 | 100.0% | −0.061% | 100% | 100% |
| CURRENT .22/.28 | 18 552 | 99.4% | −0.055% | 99% | 99% |
| FIXED 0.18 | 18 658 | 99.9% | −0.059% | 100% | 100% |
| **FIXED 0.10** | 18 667 | 100.0% | −0.060% | 100% | 100% |
| PCTILE top40% | 7 568 | 40.5% | −0.049% | 32% | 28% |
| PCTILE top20% | 3 926 | 21.0% | −0.057% | 17% | 12% |
| PCTILE top10% | 2 067 | 11.1% | −0.070% | 10% | 7% |

Population: 18 670 candidates, of which 517 gained ≥3% and 154 gained ≥5%.

Two conclusions, and one of them killed the first proposal:

1. **Fixed floors of 0.10–0.18 are a no-op historically** — they admit ~100% with
   full recall. Lowering the floor therefore carries almost no historical risk,
   and it removes a threshold that can black out an entire day when the model's
   output level shifts under it.
2. **The percentile floor is actively harmful and was REJECTED.** Its recall of
   big movers sits at or below its admission rate — on bull days top-20% admits
   21.0% and catches 14% of ≥3% movers and 11% of ≥5%, where random selection at
   the same rate would catch ~21%. The model ranks the biggest movers lowest,
   which is the mean-reversion training target showing through. This refutes the
   percentile policy proposed earlier in the same session on the strength of a
   one-day correlation of +0.270 (n=37), which was called weak at the time and
   turned out to be worse than weak.

Routing's own contribution over the same folds — routed vs global median proba:
`0.4406/0.4441`, `0.4394/0.4402`, `0.4230/0.4309`, `0.4350/0.4328`. It earns
nothing on ordinary days.

## Same-day replay: the acceptance test

Rescoring 2026-08-20's own candidate rows with routing off and a 0.10 floor:

| coin | med routed | med global | first pass | move still ahead |
|---|---|---|---|---|
| ENA | 0.0539 | 0.1905 | 07:45 | **+21.6%** |
| XRP | 0.1794 | 0.1873 | 08:00 | +16.5% |
| WIF | 0.1117 | 0.2615 | 07:00 | +10.2% |
| ORDI | 0.0319 | 0.2826 | 07:15 | **+17.7%** |
| FET | 0.0687 | 0.1766 | 08:00 | +8.6% |
| BONK | 0.1128 | 0.1877 | 06:00 | +8.1% |
| ZRX | 0.0344 | 0.2464 | 08:00 | +6.1% |

**7 of 7**, median 10.2% of the day's move still ahead at the moment the gate
would have passed. ORDI is the decisive one: no threshold policy could reach it
(best routed proba all day 0.0530 against a market median of 0.0562), and turning
routing off lifts it to 0.2826.

## Shadow / canary decision

**No shadow period. Deployed directly, deliberately.**

A shadow run answers "what would this have admitted" — and that question is
already answered exactly, by replaying the day's own logged candidates above.
Shadow adds nothing a replay has not given, while every hour of it is another
hour of a bot that emits no signals during a rally. The change is also a strict
loosening of one gate with a one-line rollback, so the downside of being wrong is
noise, not silence.

What replaces the shadow period is a watch: the first entries after restart, and
tomorrow's block mix, read against the numbers in this file.

## Verification

`test_ml_gate_segment_routing.py` — 14 tests. Beyond the flags themselves they
pin the two mistakes made while building this, because both looked correct:

- **The wrong flag was set first.** `ML_SIGNAL_SEGMENT_ROUTING_ENABLED` guards
  `predict_proba_from_payload`, but `_select_ml_payload` chooses the segment model
  *before* that call, so the guard never fires for the live path. Live median
  proba after that edit: 0.0445, against 0.0569 before — no movement. A test now
  asserts the live switch is the one that is off, and that both agree.
- **The backtest's first version scored with the global model only** and reported
  the CURRENT gate admitting 99.9% against 0% observed live. A comparison that
  cannot reproduce the failure cannot evaluate its fix.

## Honest limit

The walk-forward does **not** reproduce the blackout: `trend|bull` survived
segment selection in only one of four folds, so the pathological draw that ran in
production never appeared in the replayed folds. This change is therefore
validated as "routing adds nothing on ordinary days" plus an exact same-day
replay — not as a proof against the failure mode itself. The residual risk is
that some other mechanism can still drive the level down; a percentile floor
would have been the structural answer to that, and the backtest says it costs
more winners than it saves.

## Follow-ups

- `CLAUDE.md` and `PROJECT_CONTEXT.md` describe `ml_signal_model.py` as a
  **CatBoost** classifier. It is not: the trainer fits a logistic model and an MLP
  and keeps whichever wins on validation. Currently logistic. Both documents need
  correcting (§16).
- Segment promotion should require more than beating a baseline on its own
  validation — a minimum sample and an out-of-fold check. Until then routing
  stays off.
- The bandit is now the binding gate (UCB SKIP ≈ 1.8 against ENTER ≈ 1.3). If
  entries stay at zero, that is the next thing to measure, not this gate.
