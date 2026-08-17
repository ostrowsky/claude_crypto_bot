# Canonical EX1 in ZigZag mode (TH-02 / TH-11)

- **Slug:** `canonical-ex1-zigzag`
- **Status:** shipped 2026-08-17 — caller fixed, metric still `unknown` on purpose
- **Created:** 2026-08-17
- **Truth-harness invariants:** TH-02 (one canonical metric per question),
  TH-11 (proxy is not the business outcome), TH-03 (provenance travels with the
  number)
- **Rollback:** drop the `--use-zigzag` argument from the aggregator's script
  table; both modes remain in the script

## Problem

`realized_potential` — how much of an available move the bot actually captured —
is the exit-side half of the North Star, and the scorecard has read **unknown**
for it. The reason is mechanical and was invisible from the outside:

```python
SCRIPTS = ["_backtest_ex1_realized_potential.py", ...]
subprocess.run([str(PYEMBED), str(p)], ...)     # no arguments, ever
```

`report_metrics_daily` invokes every metric script with **no arguments**, so
`--use-zigzag` could not be set no matter what anyone intended. The daily number
has always been the deprecated proxy, which takes the intraday high as the
potential. The canonical mode (`zigzag_labeler.detect_uptrends`) uses the matched
uptrend's `gain_pct` — the move that actually existed, not the day's extreme.

## The two modes disagree, and in the direction that matters

Same 30-day window, same trades:

```
top20 (n=27)      median     mean    share EX1 >= 0.5
proxy             0.0027   -0.0063        0.0%
zigzag            0.0032   +0.0681       11.1%
```

The proxy systematically **understates** capture: the day's high is larger than
the matched uptrend, so every ratio is smaller. Under the canonical measure 3 of
27 winners captured at least half their move; under the proxy, none did.

That is not "the bot got better". It is the difference between measuring against
a move that existed and a move that never had to be catchable in one trade.

## Change

1. **The aggregator can pass arguments.** `SCRIPTS` becomes `(filename, args)`
   pairs, and EX1 is invoked with `--use-zigzag`. A metric that can only be
   computed one way because the caller cannot express the other way is a caller
   defect, not a metric defect.
2. **The mode travels with the number.** `METRIC_JSON` gains
   `potential_source: "zigzag" | "proxy"` plus the per-trade counts of each. Two
   materially different numbers under one metric name is exactly what TH-02
   exists to prevent, and today they are indistinguishable downstream.
3. **The scorecard refuses a proxy value.** `realized_potential` reports a value
   only when the mode is canonical; a proxy run leaves it `unknown` **with the
   reason**, rather than quietly publishing the smaller number as if it were the
   answer.

## What must be reported, not hidden

**n = 27 winners in 30 days.** `share_ex1_ge_05 = 11.1%` is three trades. The
share is reported with its denominator and must not be read as a rate that can
move week to week — a single trade shifts it by 3.7pp.

**ZigZag needs klines per trade and can fail to match.** When no uptrend matches,
the row falls back to the proxy rather than being dropped, so a mixed population
is possible; the counts make the mix visible instead of averaging it away.

## Verification

`test_canonical_ex1.py`:

1. the aggregator's script table carries arguments and EX1 is invoked with
   `--use-zigzag`;
2. `METRIC_JSON` names its `potential_source`, and the proxy and zigzag runs are
   distinguishable from the payload alone;
3. the scorecard leaves `realized_potential` unknown for a proxy payload and
   populates it for a canonical one;
4. the emitted share carries its denominator (TH-01).

**Shadow/canary: не применимо.** This changes a measurement, not behaviour. No
gate, model or threshold reads `realized_potential`.

## Result — the caller defect is fixed and the metric is still not publishable

`--use-zigzag` now reaches the script. What it revealed is the more useful
finding: **the canonical measure covers 36% of trades and 9 of 27 top-20 rows.**

```
mode      potential_source   n_zigzag   n_proxy   top20 median   share >= 0.5
proxy     proxy                     0       404        0.0027           0.0%
zigzag    mixed                   145       259        0.0032          11.1%
```

When no uptrend matches, the row falls back to the proxy rather than being
dropped — so the "canonical" run is `mixed`, and its average is half one
definition and half another. Publishing it would be TH-02 exactly: two
definitions wearing one name. Publishing the clean subset instead means **n=9**,
where a single trade moves everything.

So the scorecard publishes only above a stated bar — `top20_zigzag_n >= 20` and
`zigzag_coverage >= 0.60`, chosen (about a month of winners; an average that is
not half-and-half) rather than fitted — and otherwise reports `unknown` **with
the coverage and the cause**. TH-11 stays red, honestly, and now names what would
clear it.

### The cause is a data gap, not the metric

`_zigzag_potential_for_trade` loads `history/<SYM>_<tf>.csv`. That directory
holds **854 files at 1h and 98 at 15m**; the `_90d` variants do not match the
name the loader builds. Most trades are on 15m, so most lookups find nothing.
Clearing TH-11 means fetching 15m history for the traded symbols — a
data-collection task with its own change, not a tweak here.

## Findings from the review

**The counts were computed over the wrong collection.** The first version read a
stray `rows` that happened to exist at module scope and reported 1 zigzag / 1
proxy out of 404. Python raised nothing, and the `mixed` verdict printed beside
it looked entirely plausible. Counts are now taken over the rows that actually
enter the metric.

**The test made the whole daily aggregation run.** `report_metrics_daily`
executes every metric script at *module* level, so `import report_metrics_daily`
inside a test spawned all of them — the test took 299s. `SCRIPTS` is now read
with `ast.literal_eval` instead of importing.

**A second loop still iterated the old flat table** and raised
`TypeError: unhashable type: 'list'` — caught by the same test run.

## Not in scope

Improving capture itself. This change makes the number honest and available; it
does not move it. The exit-side work it unblocks is a separate change.
