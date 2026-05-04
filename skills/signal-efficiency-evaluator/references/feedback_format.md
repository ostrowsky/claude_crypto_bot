# Feedback format

Two outputs. The JSON is for the bot / RL pipeline; the Markdown is for the human.

## `report.json`

```json
{
  "evaluation_run_id": "eval_2025-04-30T14:00:00Z_BTCUSDT_5m",
  "generated_at": "2025-04-30T14:00:00Z",
  "config": {
    "window_start": "2025-04-23T00:00:00Z",
    "window_end": "2025-04-30T00:00:00Z",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframe": "5m",
    "swing_threshold_pct": 3.0,
    "max_intratrend_drawdown_pct": 1.5,
    "min_trend_duration_bars": 3,
    "fee_pct": 0.075,
    "labeler_version": "swing_v1"
  },
  "summary": {
    "total_trends_in_period": 42,
    "trends_caught_with_buy": 18,
    "trends_missed": 24,
    "miss_rate": 0.571,
    "total_buy_signals": 30,
    "buys_into_real_trend": 18,
    "false_positive_buys": 12,
    "false_positive_rate": 0.400,
    "total_sell_signals": 22,
    "sell_lateness_too_late_count": 14,
    "sell_lateness_premature_count": 5,
    "sell_lateness_optimal_count": 3,
    "median_buy_lateness_bars": 4,
    "median_buy_lateness_pct_of_move": 22.4,
    "median_sell_lateness_bars": 6,
    "median_capture_ratio": 0.62,
    "p25_capture_ratio": 0.41,
    "p75_capture_ratio": 0.79,
    "total_realized_pnl_pct": 14.3,
    "buy_and_hold_pnl_pct": 8.1,
    "alpha_vs_buy_and_hold_pct": 6.2,
    "win_rate": 0.667,
    "profit_factor": 2.1,
    "max_drawdown_pct": 4.7
  },
  "trade_verdicts": [
    {
      "trade_id": "t_001",
      "symbol": "BTCUSDT",
      "buy_signal": {
        "ts": "2025-04-24T13:05:00Z",
        "price": 64210.5,
        "confidence": 0.71
      },
      "sell_signal": {
        "ts": "2025-04-24T15:30:00Z",
        "price": 65880.0,
        "confidence": 0.58
      },
      "matched_trend": {
        "true_start_ts": "2025-04-24T12:35:00Z",
        "true_start_price": 63540.0,
        "true_end_ts": "2025-04-24T15:00:00Z",
        "true_end_price": 66120.0
      },
      "buy_lateness_bars": 6,
      "buy_lateness_pct_of_move": 25.9,
      "sell_lateness_bars": 6,
      "sell_lateness_pct_of_move": 9.3,
      "captured_pnl_pct": 2.60,
      "available_pnl_pct": 4.06,
      "capture_ratio": 0.640,
      "verdict": "late_entry_late_exit",
      "verdict_summary": "Bought 6 bars after the swing low (missing 25.9% of the move), sold 6 bars after the swing high (giving back 9.3% of unrealized peak). Net captured 64% of available profit."
    }
  ],
  "missed_opportunities": [
    {
      "trend_id": "trend_005",
      "symbol": "ETHUSDT",
      "true_start_ts": "...",
      "true_end_ts": "...",
      "available_pnl_pct": 4.8,
      "duration_bars": 7,
      "context": {
        "rsi_at_start": 28,
        "adx_at_start": 18,
        "volume_ratio_vs_24h_avg": 1.4
      },
      "why_likely_missed": "ADX was 18 at trend start; current entry filter requires ADX >= 20. Consider lowering threshold to 17 in clear-trend regimes."
    }
  ],
  "false_positives": [
    {
      "buy_signal_ts": "...",
      "symbol": "BTCUSDT",
      "buy_price": 64500.0,
      "max_favorable_move_pct": 0.4,
      "max_adverse_move_pct": -2.1,
      "subsequent_outcome": "stopped_out_after_3_bars",
      "context": {
        "rsi": 52,
        "adx": 22,
        "regime": "ranging"
      },
      "why_likely_false": "Ranging regime (no swing high in prior 20 bars >2% above swing low). Consider adding a regime filter that suppresses BUY when 20-bar range < 2x ATR."
    }
  ],
  "patterns": [
    {
      "pattern_id": "consistent_late_entry",
      "evidence": "Median BUY lateness = 4 bars; 73% of caught trends had lateness >= 3 bars.",
      "estimated_impact_pct": 4.2,
      "recommendation": "Investigate the entry filter chain. ADX confirmation is the most likely culprit — try lowering ADX threshold from 25 to 22, or replace ADX with a faster trend confirmation (e.g., 5-bar EMA slope)."
    }
  ],
  "coaching_examples": [
    {
      "title": "Example 1 — A clean trend the bot caught late",
      "narrative": "On 2025-04-24, BTCUSDT moved from 63540 to 66120 over 30 minutes (4.06% gain, 6 bars). The bot bought at 64210 (6 bars late, after 25.9% of the move was already gone) and sold at 65880 (6 bars after the peak, giving back 9.3%). To catch this earlier, look for the volume spike at 12:35 — it preceded the move by 5 bars and was 2.3x the 24h average."
    }
  ]
}
```

## `rl_memory.jsonl` append record

When `--append-to-rl-memory` is set, one line per evaluation is appended. Compact intentionally — `rl_memory.jsonl` is meant to be read repeatedly and shouldn't bloat.

```json
{"type": "evaluation_feedback", "ts": "2025-04-30T14:00:00Z", "run_id": "eval_2025-04-30T14:00:00Z_BTCUSDT_5m", "summary": {"miss_rate": 0.571, "false_positive_rate": 0.400, "median_buy_lateness_bars": 4, "median_capture_ratio": 0.62, "alpha_vs_buy_and_hold_pct": 6.2}, "top_pattern": "consistent_late_entry", "top_recommendation": "Lower ADX threshold from 25 to 22", "report_path": "evaluation_output/report.json"}
```

The bot can `tail`-read this and feed `top_pattern` + `top_recommendation` into its next prompt (or `rl_optimizer.py` can use the numeric fields as a reward signal).

## `report.md` structure

Always follow this template, in this order:

```markdown
# Signal Efficiency Report — <symbol(s)>, <window>

## Executive summary
<one paragraph, 3 sentences max, leading with the three headline numbers>

## Headline metrics
<table: metric | value | what good looks like>

## What the bot did well
<bullet points, concrete>

## What the bot did wrong
<bullet points, concrete, ranked by estimated_impact_pct>

## Coaching examples
<3 narrated case studies with timestamps and prices>

## Recommended adjustments
<numbered, specific, includes parameter names from config.py where applicable>

## Methodology
<one paragraph: which thresholds were used, why, and how to re-run with different ones>
```

Keep it under 2 pages printed. Anything more belongs in the JSON.
