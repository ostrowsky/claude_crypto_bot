# Weakness exits vs holding on the trail — REJECTED by the non-inferiority margin (2026-10-05)

Status: unconditional rule **rejected** under the pre-registered rule. Narrowed rule — **ignore a WEAK
RSI-divergence exit when profit at the signal ≥ +0.5 %, hold on the trail** — chosen on Mar–Jun and
**accepted on Jul–Oct** (rocket-days +0.89 pp [+0.27, +1.65], all +0.011 [−0.075, +0.105]).
Nothing deployed: shadow readout first, after the X-9c readout (2026-10-12).
Script: `files/_backtest_weak_exits.py` · result: `.runtime/weak_exits_backtest.json` (+ `.log`).
Truth-harness rules: TH-01, TH-04, TH-06 (bot's own trades), TH-08, TH-10.

## Why

Capture is the North Star's weak factor (0.22–0.25, 240 d). The rocket audit
(`rocket-segment-spec.md`) found the bot keeps a median +1.3 % of a 14.1 % rocket day; after exits on
WEAK RSI divergence, RSI overbought and similar signals the coin rose another +5.6..+8.2 % the same
day. H5 (trail-only above break-even) deliberately leaves WEAK / RSI / MACD exits alone.

Exit reasons of 4 332 15m exits since 2026-03-01: ATR trail 45 %, WEAK RSI divergence 17.5 % (mean
trade +1.47 %), EMA20 patterns ~15 %, time limit 3.9 %, RSI overbought 2.0 % (+6.16 %), quality
recheck price_edge 2.0 % / MACD 1.2 %, WEAK EMA fan 1.2 %, WEAK volume exhaustion 0.9 %, leader mode 1.5 %.

## Design (fixed before the run)

- Trades: every real 15m trade with an exit since 2026-03-01 (exit_validator grid and pairing).
- Weakness exits: RSI divergence, RSI overbought, quality recheck (price_edge / MACD), EMA fan, volume
  exhaustion — 1 078 trades.
- Current arm: exit at the close of the weakness bar (grid close, same basis as the variant).
- Variant: no weakness exit — the calibrated P-1 plain trail (trade's trail_k, the mode's floor)
  ratcheted from the entry bar, continued up to 96 bars after the weakness bar; trail crossings before
  that bar are ignored (the live trade did not exit there).
- One primary verdict (all categories together) with `exit_validator.decide`: rocket lower95 > 0 AND
  all lower95 ≥ −0.10 pp AND ≥ 60 % months ≥ 0. Per-category numbers are descriptive only.

## Result

| primary: all weakness exits held on the trail | value |
|---|---|
| trades changed | 1 049 of 4 313 |
| rocket-day trades (n=375), pp | **+1.10 [+0.62, +1.63]**, 8 of 8 months ≥ 0 (+0.19 .. +3.03) |
| winner-day trades (n=446), pp | +1.04 [+0.52, +1.61] |
| all trades, pp | −0.035 [**−0.105**, +0.039] |
| verdict | **reject** — the all-trade lower bound misses the −0.10 pp margin by 0.005 |

Per category (descriptive; mean exit at signal → mean held on trail; paired difference):

| category | n | exit → hold | all trades, pp | rocket-days, pp |
|---|---|---|---|---|
| RSI divergence | 776 | +1.47 → +1.27 | −0.19 [−0.48, +0.07] | +2.85 [+1.45, +4.47] (n=95) |
| RSI overbought | 82 | +6.07 → +6.48 | +0.41 [−1.46, +2.67] | +1.15 [−1.26, +3.85] (n=34) |
| quality price_edge | 83 | +1.62 → +1.42 | −0.20 [−1.40, +1.27] | +4.71 [+2.87, +7.15] (n=14) |
| quality MACD | 49 | −0.04 → −0.59 | −0.55 [−1.86, +0.83] | n=4 |
| EMA fan | 48 | −1.75 → −2.16 | −0.40 [−0.86, +0.11] | n=2 |
| volume exhaustion | 40 | +0.71 → +1.36 | +0.66 [−0.38, +1.79] | n=4 |

## Reading

Holding through weakness signals is exactly where the rocket capture leaks: +1.10 pp per rocket-day
trade on a backtest mean of ~3.3 % is about a third more of the move, consistent across all months.
The price is paid on ordinary trades, where the signal often was right — small on average, but its
interval reaches −0.105 pp. The rule says reject, and the margin is not relaxed after the fact.

Limits: the variant uses the P-1 plain trail, not the full live exit stack (H5, leader mode since
09-28, other soft exits after the signal); rocket-days are labelled by the day's own outcome
(the comparison is paired per trade, but the subset is known only at day end).

## Follow-up 1 run: time split, choose on Mar–Jun, judge on Jul–Oct (2026-10-05)

Script `files/_backtest_weak_exits_split.py`, result `.runtime/weak_exits_split.json` (+ `.log`).
Fixed before either half was looked at: 12 candidates = {all weakness, RSI divergence only} ×
{no condition, profit at the signal ≥ +0.5 %, ≥ +1 %, coin in the day's top-10, day return ≥ +3 %,
≥ +5 %}; choose on TRAIN (days < 2026-07-01; 2 680 trades) the passing candidate with the largest
rocket-day gain, stop if none passes; judge once on TEST (≥ 2026-07-01; 1 633 trades).

- TRAIN: exactly one candidate passes — **RSI divergence only, profit at the signal ≥ +0.5 %**:
  rocket-days +0.39 pp [+0.04, +0.81], all −0.049 [−0.091, −0.002], 3 of 4 months. (The
  unconditional rule on train: rocket +0.52 [+0.07, +1.04] but all −0.103 [−0.168, −0.034] — reject.)
- **TEST verdict: ACCEPT.** 245 trades changed. Rocket-days **+0.89 pp [+0.27, +1.65]**, all trades
  +0.011 [−0.075, +0.105], winner-days +0.62 [−0.03, +1.49], 4 of 4 months ≥ 0 (+0.09 .. +1.04).
- Reference only (not a selection): on TEST every one of the 12 candidates except "all weakness,
  top-10" passes, the unconditional rule included (+1.86 [+0.97, +2.84], all +0.077) — Jul–Oct favoured
  holding more than Mar–Jun did.

Disclosure: the candidate grid (in particular "RSI divergence only") was written after the
full-period per-category table above had been seen, and that table includes Jul–Oct. The condition
itself was chosen on Mar–Jun only, but the split is therefore not a fully blind confirmation. The
clean confirmation is the shadow readout (follow-up 2), scheduled after the X-9c readout.

**Implementation note.** The tested rule uses a FIXED +0.5 % profit at the signal for every mode.
Live H5 (`_h5_break_even_pct`) uses per-mode thresholds (0.3 % impulse_speed / impulse / breakout,
0.5 % trend / alignment / retest, 0.7 % strong_trend). Extending H5 with its own helper would ship
something else than what was tested; the RSI-divergence condition must use 0.5 % for all modes.

## Proposed follow-up (not run)

Confirm on data the replay did not use, one of:
1. **Shadow** — log, at every live weakness exit, what holding on the trail would have returned
   (computed after the fact), with the same criterion pre-registered as a readout; or
2. **Time split** — choose a narrowing condition (e.g. profit at the signal, coin's day rank) on
   2026-03..06 only and judge it on 2026-07..10 with the same criterion.
No live change before the X-9c readout (2026-10-12): both touch the exit stack of the same trades.
