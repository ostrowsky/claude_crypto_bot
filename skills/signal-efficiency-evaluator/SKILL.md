---
name: signal-efficiency-evaluator
description: Evaluate the efficiency of BUY/SELL signals produced by the claude_crypto_bot and generate structured feedback that reinforces the bot's ability to (a) emit a BUY as early as possible when a sustainable uptrend begins and (b) emit a SELL when the trend ends, in order to maximize realized profit per trade. Use this skill whenever the user asks to "score the bot's signals", "grade the bot's last week", "compute capture ratio", "find missed trades", "find false-positive BUYs", "tell the bot what it did wrong", "produce RL feedback", "rate the bot's earliness/lateness", or asks any question of the form "how good were the signals from <date> to <date>". Also use this skill when the user wants to update rl_memory.jsonl with verdicts or wants per-trade coaching examples derived from bot_events.jsonl. Even if the user does not say "skill" or "evaluate" explicitly, trigger this whenever the conversation is about measuring or improving the trading performance of claude_crypto_bot's signals.
---

# Signal Efficiency Evaluator

This skill turns raw bot output into a structured grade card and a feedback file the bot can learn from. It does **not** retrain the model directly — it produces the judgment the existing `rl_critic.py` / `rl_optimizer.py` pipeline can consume, plus a human-readable report.

## What you're being asked to do

The user wants their bot to:
1. Emit a BUY **as early as possible** once a sustainable uptrend begins.
2. Emit a SELL **when the uptrend ends**.
3. Maximize realized profit per trade (i.e., capture as much of each trend as possible).

Your job, when this skill triggers, is to:
1. Read the bot's signal log and the matching price history.
2. Use hindsight to label every "sustainable uptrend" in the evaluation window — this is the **ground truth**.
3. Match each BUY/SELL signal to those trends and compute concrete metrics (lateness, capture ratio, false-positive/miss rates, P&L vs. buy-and-hold).
4. Write two artifacts: a machine-readable JSON feedback file, and a human-readable Markdown report.
5. Optionally append a compact verdict record to `rl_memory.jsonl` so the existing RL pipeline picks it up.

## Workflow

Run these steps in order. Each step is implemented by `scripts/evaluate_signals.py`; you generally don't need to write evaluation logic yourself — call the script. Read the script if you need to debug or extend.

### Step 1 — Confirm scope with the user

Before running, confirm or infer:
- **Evaluation window** (defaults to last 7 days if the user didn't say).
- **Symbol(s)** (default: all symbols present in `bot_events.jsonl` for the window).
- **Timeframe** (default: read from `config.py` — usually `5m` or `15m`).
- **Trend definition thresholds** (defaults below; see `references/trend_definition.md` for the full rationale):
  - `swing_threshold_pct = 3.0` — minimum move from swing low to swing high to count as a sustainable uptrend
  - `max_intratrend_drawdown_pct = 1.5` — counter-move that breaks the trend
  - `min_trend_duration_bars = 3` — trends shorter than this are noise
- **Fee model** (default: 0.075% per side, taker fee on Binance spot).

If the user just said "evaluate the last week", use the defaults and tell them what defaults you used in the report.

### Step 2 — Locate inputs

The script expects a project root. Default discovery order:
1. The current working directory if it contains `bot_events.jsonl`.
2. `D:\Projects\claude_crypto_bot` (Windows path the user mentioned earlier).
3. Whatever path the user passed via `--project-root`.

Inputs the script reads:
- `bot_events.jsonl` — the signal log. **Assumed schema** (verify on first run; the script will print the schema it sees and ask you to confirm if anything looks off):
  ```json
  {"ts": "2025-04-30T13:05:00Z", "symbol": "BTCUSDT", "event": "BUY",
   "price": 64210.5, "confidence": 0.71, "features": {...}}
  ```
  Acceptable variants: `event` may be `signal_type` or `action`; `BUY/SELL/HOLD` may be lowercase. The script normalizes them.
- **Price history** — needed for ground-truth labeling. Discovery order:
  1. A local cache (e.g., `history/<symbol>_<tf>.parquet` or `.csv`) if `backfill_history.py` has produced one.
  2. Live fetch from Binance public REST (no API key needed for klines) if cache is missing.

### Step 3 — Run the evaluator

```bash
python scripts/evaluate_signals.py \
  --project-root <repo-path> \
  --window-days 7 \
  --timeframe 5m \
  --swing-threshold-pct 3.0 \
  --max-intratrend-drawdown-pct 1.5 \
  --min-trend-duration-bars 3 \
  --fee-pct 0.075 \
  --output-dir ./evaluation_output
```

The script will:
1. Load events and price data.
2. Detect ground-truth uptrends via swing-point labeling.
3. Match BUY/SELL signals to trends.
4. Compute all metrics.
5. Write `evaluation_output/report.json` and `evaluation_output/report.md`.
6. Print a one-screen summary to stdout.

### Step 4 — Show the user the summary

Read the printed summary, then call out the **three numbers that matter most for their stated goal**:
- **Median BUY lateness** (in bars and in % of move missed) — directly measures "as early as possible".
- **Median capture ratio** — directly measures "maximize trade profit".
- **Miss rate** (sustainable uptrends with no BUY) — measures opportunity cost.

Then briefly highlight the top 1–2 patterns the report identifies (e.g., "consistent late entry", "frequent premature SELL", "false positives concentrated in low-ADX regimes"). Don't dump the whole JSON inline — point them at the files.

### Step 5 — Decide on the feedback record

Ask the user (or infer from the task): do they want to append a feedback record to `rl_memory.jsonl` for the bot to consume on its next run? If yes:

```bash
python scripts/evaluate_signals.py --append-to-rl-memory --rl-memory-path <repo>/rl_memory.jsonl ...
```

The appended record uses the schema documented in `references/feedback_format.md`. It is intentionally compact (one JSON line per evaluation run) so it interleaves cleanly with whatever `rl_critic.py` is already writing.

### Step 6 — Iterate

The first run almost always reveals a mismatch between assumed and actual schemas, or thresholds that don't match the user's intuition for "sustainable trend". When it does:
- Re-run with adjusted thresholds and show how the metrics change.
- If the schema is different, the script's `--schema-map` flag accepts a JSON dict mapping bot fields to evaluator fields.

## Key concepts (read these before debugging output)

The reference files are kept short and load-on-demand:

- **`references/trend_definition.md`** — exactly how "sustainable uptrend" is detected, with diagrams in ASCII and worked examples. Read this if the user disputes whether something was a "real trend".
- **`references/feedback_format.md`** — the JSON schemas for `report.json` and the `rl_memory.jsonl` record. Read this if you need to edit or extend the output.
- **`references/metrics.md`** — formal definitions of every metric in the report (capture ratio, lateness, alpha vs. buy-and-hold, etc.).

## Output format

Two files in `./evaluation_output/` (path configurable):

1. **`report.json`** — machine-readable. Top-level keys: `evaluation_run_id`, `config`, `summary`, `trade_verdicts`, `missed_opportunities`, `false_positives`, `patterns`, `coaching_examples`. Schema in `references/feedback_format.md`.

2. **`report.md`** — human-readable. Sections in this order:
   - **Executive summary** (3 numbers + verdict in one paragraph)
   - **Headline metrics** (table)
   - **What the bot did well**
   - **What the bot did wrong** (the most important section)
   - **Concrete coaching examples** (3 case studies with timestamps and prices)
   - **Recommended adjustments** (specific, actionable — e.g., "lower ADX threshold from 25 to 22 for early entry")
   - **Methodology footnote** (which thresholds were used and why)

## When NOT to use this skill

- The user is asking the bot to *generate* a new signal — that's `bot.py`, not this skill.
- The user wants to optimize hyperparameters via search — that's `rl_optimizer.py`.
- The user wants to retrain the ML ranker — that's `ml_signal_model.py`.

This skill *grades* signals and produces *feedback*. Other tools act on that feedback.

## Important caveats

- **The trend labeler uses hindsight.** That's correct for grading — you can only judge "was this BUY early?" by knowing where the trend actually started, which requires looking at what happened next. Do not let the bot's live signal logic use the ground-truth labels; those are for evaluation only.
- **Threshold sensitivity.** If you set `swing_threshold_pct` too low, every wiggle becomes a "trend" and the bot looks great by accident. Too high and you miss real opportunities the bot legitimately caught. The defaults are calibrated for 5m crypto on majors; widen them for daily timeframes, narrow them for 1m scalping.
- **Fees matter.** A capture ratio of 1.0 is impossible after fees. Don't let the user be alarmed by a median capture ratio of ~0.85 — that's actually excellent.
