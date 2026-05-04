# Trend definition (ground truth labeler)

## Why we need a labeler at all

To grade the bot's BUY/SELL signals against the user's stated goal — "buy as early as possible into a sustainable uptrend, sell at the top, maximize profit" — we need an objective definition of what a sustainable uptrend **is**, computed with full hindsight. Without that label, "lateness" and "capture ratio" are undefined.

The labeler produces a list of `(swing_low_time, swing_low_price, swing_high_time, swing_high_price)` tuples. Each tuple is a "true uptrend" the bot ideally should have caught.

## The algorithm: swing-point detection with minimum-move threshold

This is the ZigZag indicator approach, used widely in technical analysis precisely because it strips out noise and keeps only moves a trader would care about.

### Pseudocode

```
inputs:
  prices: list of (timestamp, high, low, close) bars
  swing_threshold_pct: minimum move to register a swing (e.g., 3.0)
  max_intratrend_drawdown_pct: counter-move that breaks a trend (e.g., 1.5)
  min_trend_duration_bars: minimum trend length in bars (e.g., 3)

state:
  direction = 'up' or 'down' (start by guessing up)
  swing_low = (index_0, price_0)
  swing_high = (index_0, price_0)
  trends = []

for i, bar in enumerate(prices):
  if direction == 'up':
    if bar.high > swing_high.price:
      swing_high = (i, bar.high)
      continue
    # check for trend break
    drawdown_pct = (swing_high.price - bar.low) / swing_high.price * 100
    if drawdown_pct >= max_intratrend_drawdown_pct:
      # trend ended at swing_high
      gain_pct = (swing_high.price - swing_low.price) / swing_low.price * 100
      duration = swing_high.index - swing_low.index
      if gain_pct >= swing_threshold_pct and duration >= min_trend_duration_bars:
        trends.append(Trend(
          start_idx=swing_low.index,
          start_price=swing_low.price,
          end_idx=swing_high.index,
          end_price=swing_high.price,
          gain_pct=gain_pct,
          duration_bars=duration,
        ))
      direction = 'down'
      swing_low = (i, bar.low)
  else:  # direction == 'down'
    if bar.low < swing_low.price:
      swing_low = (i, bar.low)
      continue
    rebound_pct = (bar.high - swing_low.price) / swing_low.price * 100
    if rebound_pct >= max_intratrend_drawdown_pct:
      direction = 'up'
      swing_high = (i, bar.high)

return trends
```

## Visual example (ASCII)

```
price
  │            ╱╲          ╱╲
  │           ╱  ╲        ╱  ╲___       ← swing high (trend end)
  │          ╱    ╲      ╱       
  │         ╱      ╲    ╱        
  │        ╱        ╲╱╲╱         
  │       ╱                       
  │  ___ ╱                        ← swing low (trend start)
  │ ╱                              
  └────────────────────────────── time
       ↑                ↑
       T_start          T_end

  This is one labeled "sustainable uptrend".
  swing_threshold_pct = (T_end_price - T_start_price) / T_start_price * 100
  must be >= 3.0%
  Any pullback during the trend must stay below max_intratrend_drawdown_pct.
```

## Why these defaults

For 5-minute crypto bars on majors (BTC, ETH, top alts):

| Parameter | Default | Reasoning |
|---|---|---|
| `swing_threshold_pct` | 3.0 | Below 2% the result is dominated by microstructure noise; above 5% you only catch a few trends per week and the sample size is too small to grade the bot. |
| `max_intratrend_drawdown_pct` | 1.5 | About half the trend size — typical in real markets. If you set it as high as the trend size, every pullback becomes a "trend continuation" and you over-count. |
| `min_trend_duration_bars` | 3 | A 3-bar 3% spike is real for crypto but borderline tradeable. Shorter than 3 is essentially a single candle — the bot would need to act inside one bar, which isn't the regime it's designed for. |

For other timeframes, scale roughly:
- 1m bars: `swing_threshold_pct=1.0`, `max_intratrend_drawdown=0.5`, `min_duration=5`
- 1h bars: `swing_threshold_pct=5.0`, `max_intratrend_drawdown=2.5`, `min_duration=2`
- 1d bars: `swing_threshold_pct=8.0`, `max_intratrend_drawdown=4.0`, `min_duration=2`

## Edge cases the labeler handles

1. **Open trend at end of window** — if direction is still "up" when the data ends, the trend is recorded only if the unrealized gain already exceeds the threshold; otherwise it's dropped. This prevents counting incomplete moves.
2. **Multiple consecutive trends** — back-to-back uptrends separated only by a small pullback are both kept. Each is graded independently.
3. **Gaps / missing data** — bars are forward-filled by the loader before labeling. If a gap exceeds 5 bars, the labeler resets state at the gap edge.
4. **Wick-only moves** — the labeler uses `high` for swing highs and `low` for swing lows, so a single long wick can register a swing. This is intentional: the bot can in principle act on a wick, and pretending it can't would let real trends slip through.

## What this labeler is NOT

- It is **not** a live trading signal. It uses future bars. Never call it from `bot.py`.
- It is **not** the only valid definition of "sustainable trend". If the user has their own preferred definition (e.g., "EMA-50 slope positive for N bars"), the script accepts a `--custom-labeler` Python file path. The default labeler is meant to be reasonable, transparent, and easy to defend.

## Tunable for sensitivity analysis

Run the evaluator twice with different thresholds (e.g., 2% vs 4%) to see whether the bot's grade is robust or threshold-dependent. If the grade flips dramatically, the bot is overfitting to a specific market regime.
