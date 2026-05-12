# Metric definitions

Formal definitions of every metric in `report.json.summary`. Read this whenever the user disputes a number or asks "what does X mean exactly?".

## Per-trade metrics

### `buy_lateness_bars`
```
buy_lateness_bars = buy_signal_bar_index - matched_trend.true_start_bar_index
```
Positive = late. Zero = perfect (bought on the swing low bar). Negative is theoretically possible if the bot bought before the swing low; in practice this means the bot bought during a downtrend and got lucky.

### `buy_lateness_pct_of_move`
```
buy_lateness_pct_of_move = (buy_price - true_start_price) / (true_end_price - true_start_price) * 100
```
The fraction of the trend's total upside that was already gone by the time the bot bought. 0% = bought at the bottom; 100% = bought at the top (no upside left).

### `sell_lateness_bars`
```
sell_lateness_bars = sell_signal_bar_index - matched_trend.true_end_bar_index
```
Positive = sold after the peak (gave profit back). Negative = sold before the peak (premature exit).

### `sell_lateness_pct_of_move`
```
if sell_signal_bar_index > matched_trend.true_end_bar_index:
  # sold late: how much of the unrealized peak did we give back?
  sell_lateness_pct_of_move = (true_end_price - sell_price) / (true_end_price - buy_price) * 100
else:
  # sold early: how much of the remaining upside did we miss?
  sell_lateness_pct_of_move = -(true_end_price - sell_price) / (true_end_price - buy_price) * 100
```
Sign matches `sell_lateness_bars`.

### `captured_pnl_pct`
```
captured_pnl_pct = (sell_price * (1 - fee_pct/100) - buy_price * (1 + fee_pct/100)) / buy_price * 100
```
Net realized P&L on the trade, after taker fees on both sides.

### `available_pnl_pct`
```
available_pnl_pct = (true_end_price - true_start_price) / true_start_price * 100
```
The trend's total upside, gross of fees. This is what the bot *could* have captured if it had bought at the swing low and sold at the swing high.

### `capture_ratio`
```
capture_ratio = captured_pnl_pct / available_pnl_pct
```
The skill's headline efficiency metric. Range: typically 0 to 1, but can be negative (losing trade) or > 1 (the bot held through a bigger move than the labeler tagged — happens when adjacent trends merge in real trading but the labeler split them). The user wants this as close to 1 as fees allow.

### `verdict`

A categorical label combining lateness signs:
- `optimal` — buy_lateness_bars <= 1 AND sell_lateness_bars in [-1, 1]
- `late_entry_optimal_exit` — buy_lateness_bars > 1 AND sell_lateness_bars in [-1, 1]
- `optimal_entry_late_exit` — buy_lateness_bars <= 1 AND sell_lateness_bars > 1
- `optimal_entry_premature_exit` — buy_lateness_bars <= 1 AND sell_lateness_bars < -1
- `late_entry_late_exit` — buy_lateness_bars > 1 AND sell_lateness_bars > 1
- `late_entry_premature_exit` — buy_lateness_bars > 1 AND sell_lateness_bars < -1
- `losing_trade` — captured_pnl_pct < 0

## Aggregate metrics

### `miss_rate`
```
miss_rate = trends_missed / total_trends_in_period
```
Where `trends_missed` = number of labeled uptrends with no BUY signal in `[true_start_ts - 2 bars, true_end_ts]`.

### `false_positive_rate`
```
false_positive_rate = false_positive_buys / total_buy_signals
```
A BUY is a false positive if, looking forward from the BUY for `max_lookahead = 2 * min_trend_duration_bars` bars, the price never rose by `swing_threshold_pct` before dropping by `max_intratrend_drawdown_pct`.

### `median_capture_ratio`, `p25_capture_ratio`, `p75_capture_ratio`
Standard percentiles over the population of completed trades (BUY matched to a SELL). Excludes still-open trades and false positives (which have no matched trend).

### `alpha_vs_buy_and_hold_pct`
```
buy_and_hold_pnl_pct = (last_price_in_window - first_price_in_window) / first_price_in_window * 100
                       (averaged across symbols if multi-symbol)
alpha_vs_buy_and_hold_pct = total_realized_pnl_pct - buy_and_hold_pnl_pct
```
Positive = the bot beat just sitting on the asset. This is the most important number for "is the bot worth running at all".

### `win_rate`
```
win_rate = trades_with_captured_pnl_gt_0 / total_completed_trades
```

### `profit_factor`
```
profit_factor = sum(captured_pnl_pct for winning trades) / abs(sum(captured_pnl_pct for losing trades))
```
> 1 = winners outweigh losers. Industry rule of thumb: > 1.5 is good, > 2.0 is excellent.

### `max_drawdown_pct`
The largest peak-to-trough decline in cumulative realized P&L over the window. Computed on the equity curve assuming each trade is sized identically.

## What good looks like

For comparison when telling the user how the bot is doing:

| Metric | Poor | Acceptable | Good | Excellent |
|---|---|---|---|---|
| `median_buy_lateness_pct_of_move` | > 40% | 25–40% | 10–25% | < 10% |
| `median_capture_ratio` | < 0.4 | 0.4–0.6 | 0.6–0.8 | > 0.8 |
| `miss_rate` | > 0.7 | 0.5–0.7 | 0.3–0.5 | < 0.3 |
| `false_positive_rate` | > 0.5 | 0.3–0.5 | 0.15–0.3 | < 0.15 |
| `alpha_vs_buy_and_hold_pct` (weekly) | < 0 | 0–1% | 1–3% | > 3% |
| `profit_factor` | < 1.0 | 1.0–1.5 | 1.5–2.0 | > 2.0 |
| `win_rate` | < 0.45 | 0.45–0.55 | 0.55–0.65 | > 0.65 |

These bands are calibrated for 5-minute crypto on majors. Adjust upward (stricter) for higher timeframes where trends are cleaner.
