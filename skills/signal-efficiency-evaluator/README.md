# signal-efficiency-evaluator

A skill for `claude_crypto_bot` that grades the bot's BUY/SELL signals against ground-truth uptrends and produces feedback that flows back into the existing RL pipeline.

## Drop-in usage

1. Place this folder where Claude can see it (e.g., `<repo>/skills/signal-efficiency-evaluator/` or wherever your skill loader expects).
2. Tell Claude something like: "Grade the bot's signals from the last 7 days."
3. Claude reads `SKILL.md`, runs `scripts/evaluate_signals.py`, and shows you a summary plus the path to the full JSON + Markdown report.

## Direct CLI usage (without Claude)

```bash
python scripts/evaluate_signals.py \
  --project-root /path/to/claude_crypto_bot \
  --window-days 7 \
  --timeframe 5m \
  --output-dir ./evaluation_output \
  --append-to-rl-memory
```

Outputs:
- `evaluation_output/report.json` — machine-readable, full per-trade detail
- `evaluation_output/report.md` — one-page human-readable report
- `<repo>/rl_memory.jsonl` — appended with one compact feedback record (only with `--append-to-rl-memory`)

## Where to look

- `SKILL.md` — the entry point Claude reads. Workflow, defaults, when to ask the user vs. when to use defaults.
- `references/trend_definition.md` — exactly how "sustainable uptrend" is detected (ZigZag-style swing detection). Read this if a result looks wrong or a user disputes whether something was a "real trend".
- `references/feedback_format.md` — JSON schemas for `report.json` and the `rl_memory.jsonl` record.
- `references/metrics.md` — formal definitions of every metric. Calibration table for "what good looks like".
- `scripts/evaluate_signals.py` — the implementation. Self-contained Python; only deps are pandas + requests.

## Assumptions verified on first run

The script makes assumptions about your `bot_events.jsonl` schema and falls back gracefully:

| Field needed | Accepted aliases |
|---|---|
| timestamp | `ts`, `timestamp`, `time`, `datetime` |
| symbol | `symbol`, `pair`, `ticker`, `asset` |
| action | `event`, `action`, `signal`, `signal_type`, `side`, `type` |
| price | `price`, `close`, `entry_price`, `px` |
| confidence | `confidence`, `score`, `prob`, `probability`, `conf` (optional) |

If your fields differ, pass `--schema-map '{"action": ["my_field_name"]}'`.

## Threshold sensitivity

Defaults are calibrated for **5-minute crypto on majors**. If your bot runs on a different timeframe, adjust:

| Timeframe | swing_threshold_pct | max_intratrend_drawdown_pct | min_trend_duration_bars |
|---|---|---|---|
| 1m | 1.0 | 0.5 | 5 |
| 5m | 3.0 (default) | 1.5 (default) | 3 (default) |
| 1h | 5.0 | 2.5 | 2 |
| 1d | 8.0 | 4.0 | 2 |

Always run twice with different thresholds (e.g., 2.5% and 3.5%) to see whether the bot's grade is robust or threshold-dependent. If the grade flips, the bot is overfitting to a regime.

## Integration with existing RL pipeline

The skill is designed to **complement** what's already in the repo, not replace it:

- `rl_critic.py` already scores something — this skill writes a higher-level evaluation summary that `rl_critic.py` (or a future revision) can read from `rl_memory.jsonl`.
- `rl_optimizer.py` consumes RL signals — give it the numeric fields from the appended record (`miss_rate`, `false_positive_rate`, `median_capture_ratio`, etc.) as additional reward terms.
- `offline_backtest.py` / `replay_backtest.py` handle backtesting — the evaluator runs **after** signals are produced (live or replayed) and grades them. Don't replace your backtester with this; let them complement.

The `top_pattern` and `top_recommendation` fields in the appended `rl_memory.jsonl` record are designed to be fed directly into the bot's next prompt as a system-message-level reflection: "Based on last week's evaluation: {top_pattern}. Recommendation: {top_recommendation}."

## Limitations to be honest about

1. **Hindsight labeling**. Ground-truth trends are computed with full hindsight. This is correct for grading but means the labels can never be used as features for live signal generation.
2. **Threshold dependence**. Different `swing_threshold_pct` values produce different grades. Always do a sensitivity check.
3. **Multi-symbol equity curve is naive**. Total realized P&L is the simple sum across symbols. If you trade with capital constraints (only N positions at a time, position sizing), build a proper portfolio simulator on top.
4. **Fee model is one-size-fits-all**. Default 0.075% per side (Binance taker). Pass `--fee-pct` to match your actual fees, including funding for futures.
5. **The pattern detector is heuristic**. It flags common issues with reasonable thresholds, but it does not run statistical tests. Treat its recommendations as starting hypotheses, not conclusions.
