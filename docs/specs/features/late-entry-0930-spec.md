# Late entry — anatomy and the crossing-entry fix (2026-09-30): REFUTED

- **Slug:** `late-entry-0930`
- **Status:** research, negative result — no behaviour change
- **Owner:** operator + Claude
- **Truth harness:** TH-01 (base rate and lift beside every precision), TH-03
  (time split for the regime filter), TH-06 (the bot's own entries as the
  comparator), TH-08 (negative result committed with its numbers), TH-10
  ("refuted" / "рано судить" rather than an invented trend), TH-12 (every extra
  look at the same data is counted), TH-13
- **Rollback:** nothing shipped; no flag, no shadow — the refuted rule never ran live
- **Scripts:** `files/_research_late_entry.py`, `files/_backtest_crossing_entry.py`,
  `files/_backtest_crossing_entry_wide.py`, `files/_backtest_crossing_entry_regime.py`

## Why

Operator, 2026-09-30: «найди решение, устраняющее главную потерю — поздний вход».
Incident reports 09-25..29 showed most winner-days entered after the first +2.5%
crossing (the North Star deadline).

## 1. Anatomy (maximum period: 531 winner-days, 2026-03-01 .. 09-28)

Immutable later-EOD top-20 on the watchlist, bot up all day, crossing known.

| where the winner-day ends up | n | share |
|---|---|---|
| entered before the crossing | 60 | 11.3% |
| entered after it (late) | 295 | 55.6% |
| never entered | 139 | 26.2% |
| held from before the day | 37 | 7.0% |

- Late entries come **3.1 h after the crossing** (median; p25 1.1, p75 8.2) with
  **46% of the move done** (move = low since D-1 00:00 → day high). Early entries:
  30% done. Modes: impulse_speed/15m 119, impulse_speed/1h 67, alignment/15m 32.
- Moves are fast: 28% of winners cross +2.5% within 1 h of the UTC open, 53%
  within 4 h. At the UTC open a median 20% of the move is already done (≥ 30% on
  27% of winner-days) — part of "late" is structural to the UTC anchor.
- Before the crossing, on the **late** days: **no entry rule fired in 70%**
  (detection, not gates), a gate blocked a candidate in 19% (first gate:
  trend_quality forecast-0 13 — already lifted by E-4 on 09-25 — ml_proba_zone
  11, mode_range_quality 9), a rule fired with no candidate in 9%. On the
  never-entered days: 58% / 32% / 11%.

So gates can reach at most ~20–30% of the lateness, and forecast-0 is already
done; the rest is detection, which five earlier attacks on price/volume could not
solve (trend-start research). The remaining lever tested here: the **delay** —
act AT the crossing instead of ~3 h after it.

## 2. Crossing entry — pre-registered criterion

Enter at the close of the first 15m bar ≥ +2.5% above the UTC open, one per coin
per day, if the coin's rank by return since the open (watchlist) ≤ K. Exit: the
live 15m stack (calibrated trail k = 2.50, impulse_speed floor 1.5%, leader-mode
switch X-9c). **Supported** only if precision (share on winner-days) ≥ the bot's
own 15m entries, pnl lower 95% ≥ bot mean − 0.10 pp, and it reaches ≥ 10% of the
434 late/never winner-days.

Base: 6 829 crossing coin-days (39.9/day), 8.1% on winner-days. Bot 15m entries,
same days: 4 026 (24.1/day), precision 0.102, mean −0.046%.

| variant | per day | precision (lift) | rockets | mean pnl [95%] | late/none reached | move done at entry |
|---|---|---|---|---|---|---|
| all crossings | 39.9 | 0.081 (1.00) | 0.076 | −0.139 [−0.239, −0.028] | 433/434 | 37% vs bot 46% |
| rank ≤ 10 | 17.9 | 0.145 (1.80) | 0.103 | −0.251 [−0.434, −0.053] | 348 | 38% vs 47% |
| rank ≤ 5 | 10.7 | 0.178 (2.22) | 0.112 | −0.406 [−0.646, −0.141] | 257 | 38% vs 47% |
| rank ≤ 3 | 6.9 | **0.202 (2.50)** | 0.125 | −0.373 [−0.732, +0.024] | 191 | 37% vs 48% |

**Refuted** — every variant fails the pnl guard. By month rank ≤ 3: −1.21 /
−0.50 / −0.16 / −0.63 / −0.66 / −0.84 / **+1.12** (Sep). What it does show: at
the crossing the leaders are told apart twice as well as the bot's entries
(precision 0.20 vs 0.10), and the entry is ~10 points of the move earlier — but
those trades lose in every month except September.

## 3. Second look — leader-mode exit from the entry bar

Same data, one variant chosen for a reason stated before the run (the leader exit
is the only exit with a measured rocket-day gain). rank ≤ 3: −0.312
[−0.686, +0.102]; rank ≤ 5: −0.340 [−0.594, −0.048]. **Refuted**; Sep again the
only positive month.

## 4. Third look — regime filter chosen on one period, judged on another

Family fixed before the run: breadth (share of watchlist above its UTC open)
≥ 0.4/0.5/0.6/0.7 × BTC since open ≥ any/0/+1%, × rank ≤ 3/5. Chosen on
2026-03-01..06-30: all 16 combinations negative; best rank ≤ 3, breadth ≥ 0.5,
BTC ≥ 0 (−0.371%). Judged on 07-01..09-29: n = 218 (2.9/day), precision 0.156 vs
bot 0.089, mean +0.454% [−0.656, +2.173] vs bot −0.079%, 30 of 198 late/none days
reached — **refuted**: July −0.70%, August −0.65%, September +3.15%; the mean is
September alone.

## Verdict and what is left

No price/volume/regime rule found here fixes the late entry without paying for it
in losing trades. Three looks at the same data were taken and are counted; none
passed. Do not re-test crossing entries on price features.

What remains, in order:
1. **Positioning at the crossing** — the rank ≤ 3 crossing has twice the bot's
   precision; if derivatives positioning (OI, funding, taker flow) separates the
   winners among those 6.9/day, the pnl could turn. Added to the scheduled
   2026-10-18 test (Question D) with this spec's criterion and baselines; the
   positioning history reaches 45 days around 10-05.
2. Gate share of the lateness (~20% of late days): forecast-0 is live (E-4
   readout 10-09); ml_proba_zone / mode_range_quality together ~5% of late days.
3. The UTC anchor: 28% of winners cross within the first hour — the North Star
   deadline counts those as late for any entry that is not already in the coin at
   the open. A move-relative earliness measure (share of the move done at entry)
   is reported here beside it and should travel with future readouts.
