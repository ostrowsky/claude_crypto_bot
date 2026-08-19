# Entry reranking at a fixed alert budget (TH-01 / TH-06)

- **Slug:** `entry-reranking`
- **Status:** evidence only — no behaviour change proposed yet
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-01 (base rate and control beside every
  ratio), TH-03 (a feature that confirms is not a feature that predicts),
  TH-06 (validate on the bot's OWN entries), TH-08 (negative results committed)

## Why the entry path

Three independent measurements point at the same place:

- portfolio alpha is negative on every window (30d −6.24% … 180d −15.97%);
- the current policy epoch's take baseline is **−0.77%** forward return (n=88);
- half the EX1 misses are trades sitting within two hours of a detected uptrend.

## The question asked first, because it is the cheapest

Not "which gate should be tightened" — gates were tightened before and blocked
100% of top gainers. The narrow question is: **holding the number of alerts
fixed, does any signal available at entry time order them better than the order
the bot used?** A reordering changes no gate and loses no candidate.

Ground truth is the immutable label store: did the entry land on a coin that
finished the UTC day in `global top-20 ∩ watchlist`. Split by time. Every ratio
sits beside a 200-draw random control, because at a few hundred rows a 2pp
difference is noise and would otherwise be read as a finding.

## Baseline

```
entries on labelled days   4 372 over 139 days (31.5/day)
holdout                    865 rows over 42 days, from 2026-06-28
the bot's own precision    9.94%   base rate 2.93%   lift 3.39x
```

**The bot already selects 3.4× better than random.** That is the number any
proposal has to beat, and it is easy to forget while looking at negative alpha.

## Result — at 50% of the day's alerts

Random control: 9.88%, 95% range [8.16, 12.12].

```
daily_range              14.92%   64 winners   ABOVE
ranker_top_gainer_prob   14.22%   61           ABOVE
slope_pct                13.52%   58           ABOVE
candidate_score          13.05%   56           ABOVE
decoupling_score         12.59%   54           ABOVE
ml_proba                 12.12%   52           noise
ranker_quality_proba      8.62%   37           noise
ranker_final_score        6.76%   29           BELOW
ranker_ev                 6.06%   26           BELOW
```

### Two of the ranker's four outputs order the entries backwards

`ranker_final_score` and `ranker_ev` land **below** the random band — not
neutral, actively reversed. `final_score` is the ranker's headline output and
the one feeding the hard veto.

**Selection caveat, and it matters (TH-06):** these are entries that already
passed every gate including the ranker's own veto. The finding is that these
scores do not order **the survivors**, and order them backwards. It does not
show the veto is harmful, and it must not be quoted as if it did.

### The best single signal is partly tautological

`daily_range` wins, and TH-03 says to name that immediately: a coin whose
intraday range is already large is more likely to finish in the top-20 *because*
it has already moved. Like `tg_return_since_open` before it, it confirms a move
rather than predicting one. `slope_pct` carries some of the same.

The legitimate signal in this table is **`ranker_top_gainer_prob`** — a model
output whose whole job is to predict top-gainer status, scoring 14.22% against a
9.88% control, and already computed and logged at entry today. A CatBoost
reranker over all thirteen signals reaches 14.92%, which is inside the noise band
of `ranker_top_gainer_prob` alone: **no new model is justified by this evidence.**

## What this does NOT justify

Cutting the alert budget in half raises precision 9.94% → 14.22% and **drops 29%
of the winners** (86 → 61). The standing objective of this project is early
detection and coverage, not precision, and the North Star multiplies coverage
directly. Trading a quarter of the winners for precision is a decision about what
the bot is for, and it belongs to the operator, not to this spec.

Two coherent uses that do NOT cost coverage:

1. **Order the alert stream** by `ranker_top_gainer_prob` so the highest-value
   alerts arrive first — no candidate is dropped;
2. **Investigate `final_score` and `ev`**, which are live components ordering
   their own admitted population backwards. That is a defect hunt, not a tuning
   exercise, and it is the finding with the most weight behind it.

## Verification

`_backtest_entry_reranking.py` reproduces the table. It reads only entry-time
fields already written by `botlog.log_entry`, so nothing is available to the
ranking that was not available to the decision.

**Shadow/canary: не применимо** — nothing here changes behaviour yet.
