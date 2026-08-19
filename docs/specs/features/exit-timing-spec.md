# Exit timing: no fixed trail width helps (goal 3)

- **Slug:** `exit-timing`
- **Status:** evidence, NEGATIVE result — no change proposed
- **Created:** 2026-08-19
- **Truth-harness invariants:** TH-06 (validate on the population the policy
  applies to), TH-08 (negative results are committed with the numbers that
  killed them), TH-01 (no ratio without its denominator)

## Which goal

**Goal 3 — signal exit only just before the uptrend ends.** Capture is the North
Star factor with the most headroom: 0.29 against a ceiling of 1.0, where
coverage is already 0.80. Doubling capture reaches the acceptable floor; no
combination of the other two factors can.

Goals 1 and 2: untouched.

## What was measured

Every entry/exit pair, joined to the immutable labels, with the forward price
path from the kline cache. For each exit: what the coin did afterwards, both to
the UTC close and inside the matched ZigZag uptrend.

On winner-days (423 trades):

```
exit class        n  realized%  n_day  left in day%  inside a trend
rsi             176      2.29%    149         4.48%            43%
atr_trail       147     -0.16%    116         8.38%            22%
ema20_weakness   56     -1.25%     39         7.04%            30%
time_max_hold    25      3.42%     14         6.66%            36%

OVERALL realized 777%, left after exit 3153%  ->  19.8% of what was still ahead
```

`atr_trail` is both the largest class and the one that leaves the most.

## The result that looked like a finding and was not

Replaying "hold past the exit and trail at X% instead" **on winner-days**:

```
policy          n  median pnl%   capture   beats actual
actual exit   423        0.78%     19.8%              -
trail 8.0%    337        4.78%     58.5%            76%
```

Capture tripled and P&L improved. It is worthless, and it contradicts the live
2026-06-05 rollback of exactly that change (the 8% impulse_speed trail, −54.9%
over five days) — which is what made it worth distrusting rather than
publishing.

**The sample was conditioned on the outcome.** Winner-days are days the coin
finished in the global top-20. Of course holding longer wins there. The policy
cannot know that at exit time, and it applies to every trade.

## On the population the policy actually applies to

All 4 174 trades, 3 584 with a replayable forward path:

```
policy          n  median pnl%   mean pnl%   win rate   beats actual
actual exit  4174       -0.28%      -0.12%        39%              -
trail 1.5%   3584       -0.60%      -0.12%        38%            37%
trail 3.0%   3584       -0.71%      -0.31%        37%            38%
trail 5.0%   3584       -0.68%      -0.28%        38%            40%
trail 8.0%   3584       -0.61%      -0.18%        39%            42%
```

**No trail width beats the current exits on any column.** Every one is worse on
median P&L; none reaches parity on win rate; the best "beats actual" figure is
42%, meaning it loses on the other 58%.

This reproduces the June rollback from an independent direction, and it was
reached by the criterion this project actually cares about rather than by P&L
alone.

The capture column reads `n/a` here on purpose: with a negative realized total
the ratio `realized / (realized + left)` flips sign and prints things like
"−5.8% capture", which is not a smaller capture but an undefined one.

## What this rules out, and what it leaves

**Ruled out:** widening, tightening or replacing the trail with any fixed width.
The 71% of the move left on the table on winner-days cannot be recovered by a
blanket policy, because the same policy pays for it on everything else.

**Left standing:** an exit that distinguishes a continuing move from a finished
one *at exit time*. That is a prediction problem with the same shape as the
entry problem, not a threshold to tune — and it should not be attempted until
there is a signal with evidence behind it, or this becomes the fourth version of
the same tuning exercise.

## Verification

`_backtest_exit_timing.py`, read-only. `--all-trades` selects the honest
population; the default is winner-days and exists only to show the biased view
that produced the false finding. Per-column `n` throughout, because the medians
are taken over the rows that had forward data, not over the class total.
