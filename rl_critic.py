"""
rl_critic.py — Critic Agent: Professional Trader Evaluator

Architecture: Actor-Critic RL where:
  - Actor  = rule-based bot (monitor.py / strategy.py)
  - Critic = this module — Claude API acting as senior trader

The critic evaluates every completed trade and produces:
  1. Quantitative score  (-1.0 … +1.0)
  2. Structured breakdown (entry quality, timing, exit quality, context)
  3. Natural language feedback for logging / future prompting
  4. Recommended config parameter adjustments

The reward signal fed back into the RL optimizer is:
  R = w_pnl   × normalized_pnl
    + w_critic × critic_score
    + w_fwd    × forward_accuracy_bonus
    - w_risk   × drawdown_penalty

This separates "was PnL good?" from "was the DECISION good?" —
a correct entry can lose money due to market noise, and the critic
can still give a positive signal for the entry quality.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

# ── Reward weights ─────────────────────────────────────────────────────────────
W_PNL    = 0.45   # realized PnL (most important)
W_CRITIC = 0.30   # Claude critic qualitative score
W_FWD    = 0.15   # T+3/5 forward accuracy
W_RISK   = 0.10   # MAE (max adverse excursion) penalty

# ── Normalization anchors ──────────────────────────────────────────────────────
PNL_TARGET_PCT  = 1.5    # +1.5% = "good" → reward +1.0
PNL_STOP_PCT    = -2.0   # -2.0% = "bad"  → reward -1.0
MAE_PENALTY_PCT = 1.0    # every 1% adverse excursion → -0.1 penalty


@dataclass
class TradeExperience:
    """
    Complete record of one completed trade — the unit of RL training.

    Stored in rl_memory.jsonl for replay and optimizer.
    """
    # Identity
    trade_id:     str
    sym:          str
    tf:           str
    mode:         str
    ts_entry:     str
    ts_exit:      str

    # Market context at entry
    btc_vs_ema50: float = 0.0
    is_bull_day:  bool  = False
    market_regime: str  = "neutral"

    # State vector at entry (actor input)
    state: dict = field(default_factory=dict)   # all indicators at entry bar

    # Action taken
    trail_k:       float = 2.0
    max_hold_bars: int   = 16

    # Outcome
    entry_price:  float = 0.0
    exit_price:   float = 0.0
    pnl_pct:      float = 0.0
    bars_held:    int   = 0
    exit_reason:  str   = ""
    mae_pct:      float = 0.0   # max adverse excursion (if tracked)

    # Forward accuracy (filled by ml_dataset)
    t3_correct:   Optional[bool]  = None
    t5_correct:   Optional[bool]  = None
    t10_correct:  Optional[bool]  = None
    t3_ret:       Optional[float] = None
    t5_ret:       Optional[float] = None
    t10_ret:      Optional[float] = None

    # Critic evaluation (filled after Claude call)
    critic_score:    Optional[float] = None   # -1.0 … +1.0
    critic_entry:    Optional[float] = None   # entry quality sub-score
    critic_exit:     Optional[float] = None   # exit quality sub-score
    critic_timing:   Optional[float] = None   # signal timing sub-score
    critic_feedback: Optional[str]   = None   # natural language
    critic_params:   Optional[dict]  = None   # suggested param adjustments

    # Composite reward
    reward:          Optional[float] = None   # final RL reward signal


def compute_reward(exp: TradeExperience) -> float:
    """
    Composite reward combining PnL, critic score, forward accuracy, and risk.

    Range: roughly -1.5 … +1.5 (unbounded at extremes)
    """
    # 1. PnL component (linear, clipped)
    pnl_norm = exp.pnl_pct / PNL_TARGET_PCT
    pnl_norm = max(-2.0, min(2.0, pnl_norm))

    # 2. Critic score
    critic = exp.critic_score if exp.critic_score is not None else 0.0

    # 3. Forward accuracy bonus
    fwd_scores = []
    for correct, ret, weight in [
        (exp.t3_correct, exp.t3_ret, 0.3),
        (exp.t5_correct, exp.t5_ret, 0.4),
        (exp.t10_correct, exp.t10_ret, 0.3),
    ]:
        if correct is not None:
            fwd_scores.append(weight * (1.0 if correct else -0.5))
        elif ret is not None:
            fwd_scores.append(weight * max(-1.0, min(1.0, ret / PNL_TARGET_PCT)))
    fwd = sum(fwd_scores) / max(1, len(fwd_scores)) if fwd_scores else 0.0

    # 4. Risk / MAE penalty
    risk_pen = -min(2.0, exp.mae_pct / MAE_PENALTY_PCT) * W_RISK if exp.mae_pct > 0 else 0.0

    reward = W_PNL * pnl_norm + W_CRITIC * critic + W_FWD * fwd + risk_pen
    return round(reward, 4)


# ── Critic prompt builder ──────────────────────────────────────────────────────

def _build_critic_prompt(exp: TradeExperience) -> str:
    """
    Builds the system + user prompt for Claude critic call.

    The critic persona: experienced quant trader, 10+ years crypto,
    focus on momentum + trend-following strategies.
    Evaluates the DECISION quality, not just the outcome.
    """

    fwd_summary = []
    if exp.t3_ret is not None:
        fwd_summary.append(f"T+3: {exp.t3_ret:+.2f}% ({'✓' if exp.t3_correct else '✗'})")
    if exp.t5_ret is not None:
        fwd_summary.append(f"T+5: {exp.t5_ret:+.2f}% ({'✓' if exp.t5_correct else '✗'})")
    if exp.t10_ret is not None:
        fwd_summary.append(f"T+10: {exp.t10_ret:+.2f}% ({'✓' if exp.t10_correct else '✗'})")

    state = exp.state
    return f"""You are a senior quantitative crypto trader with 10+ years specializing in momentum and trend-following on Binance spot markets.

You must evaluate this completed trade with clinical objectivity. A good DECISION can still lose money due to noise; a bad DECISION can still profit by luck. Judge the decision quality, not just the PnL.

TRADE RECORD:
  Symbol: {exp.sym}  [{exp.tf}]
  Mode: {exp.mode}
  Entry: {exp.ts_entry[:16]} UTC  Exit: {exp.ts_exit[:16]} UTC
  PnL: {exp.pnl_pct:+.2f}%  Bars held: {exp.bars_held}
  Exit reason: {exp.exit_reason}

MARKET CONTEXT AT ENTRY:
  BTC vs EMA50: {exp.btc_vs_ema50:+.2f}%  (Bull day: {exp.is_bull_day})
  Market regime: {exp.market_regime}

ENTRY INDICATORS:
  Price vs EMA20:  {state.get('price_vs_ema20', 0):+.2f}%
  EMA20 slope:     {state.get('slope_pct', 0):+.3f}%/bar
  ADX:             {state.get('adx', 0):.1f}
  RSI:             {state.get('rsi', 0):.1f}
  Volume×:         {state.get('vol_x', 0):.2f}
  MACD hist:       {state.get('macd_hist', 0):.4e}
  Daily range:     {state.get('daily_range', 0):.1f}%
  ML proba:        {state.get('ml_proba', 0):.3f}

FORWARD RETURNS (what market did after entry):
  {" | ".join(fwd_summary) if fwd_summary else "Not yet available"}

ATR trail K: {exp.trail_k}  Max hold bars: {exp.max_hold_bars}

Respond with ONLY valid JSON, no markdown, no explanation outside JSON:
{{
  "entry_quality": <float -1 to +1>,
  "exit_quality":  <float -1 to +1>,
  "timing":        <float -1 to +1>,
  "overall":       <float -1 to +1>,
  "verdict":       "<one of: excellent | good | acceptable | poor | terrible>",
  "entry_diagnosis": "<what was right/wrong about the entry conditions>",
  "exit_diagnosis":  "<was exit too early, too late, or correct?>",
  "main_error":      "<the single most important mistake, or null>",
  "param_hints": {{
    "description": "<brief explanation of what to adjust>",
    "adjustments": [
      {{"param": "<CONFIG_PARAM_NAME>", "direction": "<increase|decrease|keep>", "magnitude": "<small|medium|large>", "reason": "<why>"}}
    ]
  }}
}}"""


# ── Claude API critic call ─────────────────────────────────────────────────────

async def call_critic(
    session: aiohttp.ClientSession,
    exp: TradeExperience,
    *,
    model: str = "claude-sonnet-4-5-20251022",
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Calls Claude API with the critic prompt for a completed trade.
    Returns parsed JSON dict or None on failure.
    """
    prompt = _build_critic_prompt(exp)

    payload = {
        "model": model,
        "max_tokens": 800,
        "temperature": 0.1,   # low temperature for consistent structured output
        "system": (
            "You are a professional crypto trading evaluator. "
            "You always respond with valid JSON only, no prose outside the JSON structure."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(max_retries):
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = data["content"][0]["text"].strip()
                # Strip potential markdown fences
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)
        except json.JSONDecodeError as e:
            log.warning("Critic JSON parse error (attempt %d): %s", attempt + 1, e)
        except aiohttp.ClientError as e:
            log.warning("Critic API error (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return None


async def evaluate_trade(
    session: aiohttp.ClientSession,
    exp: TradeExperience,
) -> TradeExperience:
    """
    Main entry point: call critic and fill in exp.critic_* fields + reward.
    Returns the enriched TradeExperience.
    """
    result = await call_critic(session, exp)

    if result:
        exp.critic_score    = float(result.get("overall", 0.0))
        exp.critic_entry    = float(result.get("entry_quality", 0.0))
        exp.critic_exit     = float(result.get("exit_quality", 0.0))
        exp.critic_timing   = float(result.get("timing", 0.0))
        exp.critic_feedback = (
            f"[{result.get('verdict','?').upper()}] "
            f"Entry: {result.get('entry_diagnosis','')}. "
            f"Exit: {result.get('exit_diagnosis','')}. "
            f"Error: {result.get('main_error','none')}."
        )
        hints = result.get("param_hints", {})
        if hints and hints.get("adjustments"):
            exp.critic_params = hints
    else:
        exp.critic_score = 0.0
        exp.critic_feedback = "Critic unavailable — reward based on PnL only"

    exp.reward = compute_reward(exp)
    return exp


# ── State extraction helper ────────────────────────────────────────────────────

def extract_state(entry_event: dict) -> dict:
    """
    Extract normalized state vector from a bot entry event.
    This is the input to both critic and optimizer.
    """
    price = entry_event.get("price", 0.0)
    ema20 = entry_event.get("ema20", price)
    return {
        "price_vs_ema20": round((price / ema20 - 1) * 100, 4) if ema20 else 0.0,
        "slope_pct":  entry_event.get("slope_pct", 0.0),
        "adx":        entry_event.get("adx", 0.0),
        "rsi":        entry_event.get("rsi", 50.0),
        "vol_x":      entry_event.get("vol_x", 1.0),
        "macd_hist":  entry_event.get("macd_hist", 0.0),
        "daily_range": entry_event.get("daily_range", 0.0),
        "ml_proba":   entry_event.get("ml_proba", 0.5),
        "trail_k":    entry_event.get("trail_k", 2.0),
        "max_hold_bars": entry_event.get("max_hold_bars", 16),
        "tf_1h":      1.0 if entry_event.get("tf") == "1h" else 0.0,
        "mode_code":  {
            "trend": 0, "strong_trend": 1, "impulse_speed": 2,
            "retest": 3, "breakout": 4, "alignment": 5, "impulse": 6,
        }.get(entry_event.get("mode", "trend"), 0),
    }
