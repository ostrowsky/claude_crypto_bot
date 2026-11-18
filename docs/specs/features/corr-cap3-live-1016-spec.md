# correlation_guard cluster cap 2 → 3, live with a readout (2026-10-16)

- **Slug:** `corr-cap3-live-1016`
- **Status:** behaviour change, operator-approved in chat on 2026-10-16. The
  operator chose: switch on with a readout, one cap for entry AND prune.
- **Change:** `CORR_MAX_PER_CLUSTER: 2 → 3`.
- **Rollback:** set it back to 2 and restart.
- **Evidence (maximum-period backtest):**
  [`corr-guard-compromise-1016`](corr-guard-compromise-1016-spec.md), over the
  whole enforcement period 2026-07-07 .. 10-16 (293 winner-days):
  - early winner-days +1.08 pp [+0.27, +2.43];
  - per trade −0.020 [−0.067, +0.028];
  - max drawdown +9.9% against the 10% bound, so marginal.
- **Shadow:** not applicable. The would-be admits (blocks at cluster size 2/2)
  are already logged, so a shadow adds no information. The live readout decides
  instead (TH-07).
- **Truth harness:** TH-04, TH-06, TH-07, TH-10, TH-11, TH-12

## What changes

- **Entry.** `correlation_guard.check_entry` blocks only when 3 or more open
  positions share the candidate's cluster (it was 2).
- **Prune — NOT in the backtest.** `prune_candidates` now closes legs only in
  clusters larger than 3, and trims them to 3 instead of 2. From the 24 logged
  enforced prunes since 2026-09-29, 8 were in clusters of exactly 3; those stop.
  Larger clusters keep one more position. Neither backtest arm modelled
  pruning, so this effect is measured live only.

## New logging (no decision reads it)

- `corr_cap_relaxed`: a candidate passed the guard with cluster size ≥ 2, so the
  old cap would have blocked it. Fields: `sym`, `tf`, `price`, `cluster_size`,
  `max_rho`, `peer_sym`. Logged before the portfolio limits; a later block is
  possible.
- `corr_prune`: an enforced prune. Fields: `sym`, `reason`. Until now prunes were
  only in stderr, and the log files are rotated away.

## Readout `CORR-CAP3` — fixed before the data exists

- **Window:** 2026-10-17 .. due 2026-11-13. Needs ≥ 21 comparable days and
  ≥ 10 relaxed trades.
- **Relaxed trade:** an entry whose coin and timeframe logged `corr_cap_relaxed`
  for the same bar (within 30 minutes before the entry), paired with its exit.
- **Goal.** Winner-days (comparable days) whose early entries (before the
  crossing) are all relaxed trades are counted as **days gained**. Days lost
  cannot be observed directly. The prune and portfolio interactions are reported
  as context.
- **Per trade.** Realized `pnl_pct` of all trades minus the same without the
  relaxed trades, i.e. "combined minus current" with the relaxed trades removed.
  The 95% bootstrap resamples trades.
- **Tail.** Daily summed realized P&L with vs without the relaxed trades: max
  drawdown of the cumulative sum, and the worst day.

| verdict | condition |
|---|---|
| **KEEP** | days gained ≥ 1 AND per-trade lower 95% ≥ −0.10 pp AND max drawdown and worst day each no more than 10% deeper with the relaxed trades |
| **ROLLBACK_SUGGESTED** | per-trade upper 95% < −0.10 pp, OR max drawdown or worst day more than 10% deeper |
| **INCONCLUSIVE** | otherwise |

**Known limits.** "Without the relaxed trades" is approximate:
- their slots might have gone to other candidates;
- the prune change alters exits of positions that are not relaxed.

Prune counts by cluster size are reported beside the verdict.

## Tests

`files/test_corr_cap3_live_1016.py`:
- the config value;
- check_entry allows a cluster of 2 and blocks 3;
- the monitor and prune logging hooks;
- the botlog event fields;
- the readout helpers on synthetic events;
- the registry entry and the docs.

## Readout result and rollback (2026-11-18)

**Readout `CORR-CAP3`, final 2026-11-17: ROLLBACK_SUGGESTED**
(`.runtime/pipeline/readouts/CORR-CAP3.json`). 21 days, 346 trades, 91 of them relaxed, 187
`corr_cap_relaxed` events, 32 prunes (clusters of 4–8).

| criterion | result | bound |
|---|---|---|
| winner-days gained | **0** of 37 | ≥ 1 |
| per trade, combined − without | −0.117 pp [−0.281, +0.028] | lower ≥ −0.10 |
| max drawdown with / without relaxed | −208.1 / −123.6 | ≤ 10% deeper |
| worst day with / without | −51.5 / −23.4 | ≤ 10% deeper |

Relaxed trades averaged −0.929% against −0.485% for the rest. The backtest's +1.08 pp of early
winner-days did not appear live, and the tail it called marginal (+9.9%) came out +68% deeper.
The relaxed entries are the loss-clustering ones.

**Operator decision, chat 2026-11-18: roll back.**
- `CORR_MAX_PER_CLUSTER` goes back to 2, for entry and prune, and the bot is restarted.
- `CORR_CAP_RELAXED_FROM` stays 2. With the cap at 2 no candidate passes at cluster size 2, so
  `corr_cap_relaxed` stops by itself.
- With the cap at 2, the prune again closes the weakest position of a cluster of 3 or more. A
  position above `CORR_PRUNE_PROFIT_PROTECT_PCT` profit is never closed this way. That is the
  pre-10-16 behaviour.
- Recorded as a `rolled_back` decision superseding `d-2026-10-16T133755Z-cap3`.
- The readout is final and is not recomputed.

**Do not re-test cap 3 on these days.** A new relaxation of this guard needs a hypothesis that
explains the clustered losses. The pre-crossing exemption was already refuted.

Tests (`files/test_corr_cap3_live_1016.py`, changed before the config):
- the config value is 2;
- with the live config a cluster of 2 blocks;
- `check_entry` still honours a patched cap of 3, so the code path stays covered.
