# Early ranking, shadow path (goals 1 and 2)

- **Slug:** `early-ranking-shadow`
- **Status:** shipped 2026-08-19, shadow only
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-01 (control beside every ratio), TH-05
  (absence of data is not failure), TH-07 (shadow before behaviour)
- **Flag:** `EARLY_RANKING_SHADOW_ENABLED`, default **True** — it writes a log
  and touches no decision
- **Rollback:** flip the flag; nothing else changes

## Which goals this serves, and by what mechanism

**Goal 1 — spot the day's winners as early as possible.** Directly: it names a
short list at 00 UTC, before the day has produced any move to react to.

**Goal 2 — signal entry as early as possible.** Not yet. This path emits no
alert; it records what it *would* have named so the claim can be checked against
outcomes before anything fires. Goal 2 is served only if this evidence holds up.

**Goal 3 — exit before the trend ends.** Not at all. Stated so the scope is not
overclaimed.

## Why a separate path is necessary rather than a better gate

All seven entry modes (`trend`, `strong_trend`, `retest`, `alignment`,
`impulse`, `impulse_speed`, `breakout`) require ADX, slope, volume or a breakout
— confirmation that the move has begun. **By construction none can fire ahead of
the move**, and the measurements agree: move-relative lead **0.02**,
`Coverage@move` **0.029**, early-alert lift **0.72x** — worse than chance —
against all-alert lift 4.12x.

No threshold on a confirmation gate produces an early signal. The signal has to
come from somewhere that does not wait.

## Evidence that the signal exists before the move

`_backtest_early_ranking.py`, 00 UTC snapshots (nothing of the day elapsed),
CatBoost on the existing feature set, split by time: train 85 days / 16 580
rows, test 37 days / 6 815 coin-days / 73 winners, base rate 1.07%.

```
list size   winners caught   coverage   precision   lift   random coverage
top-3          4/73             5.5%       3.6%    3.36x   2.8% [0.0,  6.8]
top-5          9/73            12.3%       4.9%    4.54x   4.6% [0.0, 11.0]
top-10        15/73            20.5%       4.1%    3.78x   8.9% [2.7, 15.1]
top-20        20/73            27.4%       2.7%    2.52x  17.5% [9.6, 24.7]
```

**top-3 sits inside the random band and is not evidence of anything.** top-10 and
top-20 are above it. The signal is real and modest: naming ten coins before the
day starts catches a fifth of its winners, against a ninth by chance.

### What that is worth against the North Star

```
live path today   cov 0.67 x cap 0.27 x lead 0.02  =  0.008
early top-10      cov 0.21 x cap 0.27 x lead 1.00  =  0.055
early top-20      cov 0.27 x cap 0.27 x lead 1.00  =  0.073
```

An order of magnitude, and not because coverage improves — it gets **worse**.
It is entirely because today's lead is ~0. That is what makes a low-coverage
early path worth measuring at all.

**The capture figure is assumed, not measured.** 0.27 is carried over from the
live path. Entering before a move may hold better (in ahead of the run) or worse
(more false starts). Nothing here establishes it, and the shadow log is what will.

## Change

`early_ranking_shadow.py` ranks the watchlist from the latest 00 UTC snapshot and
appends the top-K with their probabilities to
`.runtime/early_ranking_shadow.jsonl`. It reads a model, writes a log, and
returns. No gate, no alert, no config value is consulted by any decision path.

Scoring is a separate read-only step (`--score`) that joins past shadow lists to
immutable labels and reports coverage, precision and lift against a random
control over whatever days both cover.

## What must be reported, not hidden

**A shadow list is not an alert.** Coverage here is "named in the morning list",
which is a weaker claim than "alerted in time to act". They must never be
compared to `Coverage@move` as if they measured the same thing.

**37 test days and 73 winners is thin.** One good day moves top-5 coverage by
several points. The scoring step reports n beside every ratio and refuses a
verdict below 20 scored days.

**The model is retrained nightly on immutable labels.** A shadow list scored
against a model that later saw those days would flatter itself; the scorer
records the model's `evaluation_scope` and `label_timing` with each list so the
provenance travels with the number.

## Verification

`test_early_ranking_shadow.py`:

1. the shadow writer touches no decision path — an AST check that it imports
   neither `monitor` nor `strategy` and defines no gate;
2. a day with no 00 UTC snapshot writes nothing rather than an empty list;
3. the list is capped at K and ordered by probability descending;
4. scoring a day the label store does not know is skipped and counted, not
   scored zero;
5. every emitted ratio carries n and the random-control band;
6. fewer than 20 scored days yields "too early to judge", not a number.

**Shadow/canary:** this IS the shadow. Nothing is promoted until the scorer has
20+ days and the coverage sits above its control band.

## Two defects the first live run had, both silent

The first list it wrote looked entirely plausible and was wrong twice.

**Every coin appeared twice.** `universe` read 210 for a 105-coin watchlist: the
dataset carries two snapshots in hour 00 — the EOD job and the intraday one —
and the loader appended instead of keying by symbol. A "top-10" was really a
top-5, and the duplicate probabilities made the list look confident.

**The model was never loaded.** `TopGainerModel()` with no `model_path` never
calls `load()`, so `_model_payload` stayed `None` and every prediction came from
the **heuristic fallback** — not from the model whose early-hour AUC of 0.80 is
the entire reason this path exists. The tell was in the numbers: the fallback
list topped out at 0.300 with ties, the loaded model spans 0.765 → 0.419. The
writer now refuses to emit a list when `tier_models` is absent, rather than
writing one the fallback produced.

Both are pinned by tests. The provenance fields exist precisely so a list cannot
be graded later without knowing what produced it, and on the first run they were
`None` — which is what surfaced the second defect.

## First live list (2026-08-19, universe 105)

```
scope   day_grouped_holdout_immutable_later_eod_label
labels  immutable_later_eod_close
picks   EOSUSDT 0.765, XAIUSDT 0.557, RNDRUSDT 0.552, POLUSDT 0.538,
        COTIUSDT 0.530, CAKEUSDT 0.516, AEVOUSDT 0.487, ACAUSDT 0.442,
        PYRUSDT 0.421, COMPUSDT 0.419
```

No verdict for at least 20 scored days — the scorer refuses one below that, and
says so rather than printing a number.
