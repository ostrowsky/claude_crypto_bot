# Gate-lock evidence replay (TH-10)

- **Slug:** `gate-evidence-replay`
- **Status:** shipped 2026-08-17
- **Truth-harness invariants:** TH-05 (absence of data is not evidence), TH-10
  (evidence expiry, fail closed), TH-01 (base rate beside the ratio)
- **Rollback:** the script is read-only unless `--write`; nothing to roll back

## Problem

`do_not_touch.json` locks six gates proven not to over-block, and every
hypothesis that would relax one is rejected on sight. Its evidence was verified
**2026-05-28** against a 30-day budget, so the harness has been failing closed
for ~81 days with `TH10_EVIDENCE_EXPIRY`.

Re-running the canonical blocked-bucket counterfactual would clear the flag but
would not answer the question. The stored numbers were computed over the
**whole** `critic_dataset`, which spans several behaviour changes — the soft gate
(06-01), the trail rollback (06-05), fallback-to-trend (06-12) and the bandit
label rebuild (08-13). A gate verified across policy eras has not been verified
under the policy running now: the population it rejects today is not the one it
rejected in April.

## Change

`_replay_gate_evidence.py` computes the counterfactual over **two** windows and
prints them side by side: the maximum period (comparable with the stored
evidence) and the current policy epoch (the one that describes today). Where they
disagree the epoch figure decides, and the max-period figure explains why.

A gate counts as re-verified **only** on epoch evidence with `n >= 20`.
Max-period-only evidence leaves the lock standing and is labelled as such.
`--write` refuses to refresh the timestamp unless every lock re-verified —
a blanket refresh would convert "we did not measure" into "we measured and it
was fine", which is the exact failure TH-10 exists to prevent.

## Result — the finding is that it cannot be fully cleared

```
stored evidence  last_verified=2026-05-28  budget=30d
take baseline    epoch avg_r5=-0.883 (n=64)   max-period avg_r5=-0.073 (n=4895)

gate                 n max     miss  Sharpe   n epoch     miss  Sharpe  verdict
mtf                    262   -0.218   -2.27         2        -       -  lock stands
trend_1h_chop         1382   +0.055   -0.36       100   +0.490   -3.13  confirmed
open_cluster_cap       356   -0.070   -1.66         0        -       -  lock stands
mode_range_quality    3512   -0.004   -5.18       263   +0.789   -2.32  confirmed
cooldown              2698   -0.045   -2.78        63   +0.526   -1.81  confirmed
ml_proba_zone         6848   +0.007   -2.93        24   -0.354   -2.01  confirmed
```

**Four of six re-verify under the current policy. Two cannot** — `mtf` has 2
epoch events and `open_cluster_cap` has none. Their locks stand, unrefreshed,
and `TH10_EVIDENCE_EXPIRY` stays red until they accumulate evidence. That is the
correct outcome, not a failure of the replay.

**The positive `miss` figures do not mean the gates are over-blocking.** `miss`
is `blocked_avg − take_avg`, and the epoch take baseline is **−0.883%**: a
positive miss here means the blocked candidates lost *less* than the entries the
bot actually made. Every epoch Sharpe is negative, so no bucket was profitable.
The over-blocking rule requires `miss > 0` **and** `Sharpe > 1.5` for exactly
this reason — a positive average against a bad baseline is not a signal to open
a gate.

**The baseline itself is the more alarming number.** The bot's own entries
average **−0.883%** over 64 events in the current epoch against −0.073% across
the max period. n=64 is thin and this is a diagnostic, not a verdict, but it says
the recent entry population is worse than the historical one — which is a
question about the entry path, not about these gates.

## Findings from the review

**The epoch filter matched nothing and looked like a real answer.** The first
version read `ts` / `ts_ms` / `decision.ts`; `critic_dataset` rows carry
`ts_signal` (ISO) and `bar_ts` (ms). Every row fell out of the window, so the
epoch column read `n=0` across the board — indistinguishable from "the bot
blocked nothing recently".

**The verdict taxonomy contradicted the script's own caveat.** Max-period-only
evidence was being counted as `confirmed` while the footer printed that a gate
without epoch evidence is not re-verified. Both were on screen at once and only
the table would have been read.

**`miss` was unreadable without its baseline** (TH-01). Added to the header.

## Verification

`test_gate_evidence_replay.py`:

1. the day extractor reads `ts_signal` and falls back to `bar_ts`;
2. a gate with fewer than 20 epoch events is never marked confirmed;
3. max-period-only evidence leaves the lock standing;
4. a bucket needs positive `miss` **and** Sharpe above the floor to be called
   over-blocking;
5. `--write` refuses while any lock is unverified.
