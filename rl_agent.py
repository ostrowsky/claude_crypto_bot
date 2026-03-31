"""
rl_agent.py — Actor-Critic RL Orchestrator

Integrates into the existing bot lifecycle:

  1. After each trade CLOSES → critic evaluates → experience stored
  2. Every N trades OR on schedule → optimizer runs → best params extracted
  3. Params applied to config → bot uses improved policy next cycle

Integration points:
  monitor.py → call rl_agent.on_trade_closed(exit_event, entry_event)
  bot.py     → call rl_agent.apply_optimized_params() on startup / schedule

Architecture diagram:
                    ┌─────────────────────────────────────┐
                    │           ACTOR (Bot)               │
                    │  rule-based signals + config params │
                    └──────────────┬──────────────────────┘
                                   │ trade completes
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         CRITIC (Claude API)         │
                    │  evaluates decision quality         │
                    │  outputs: score + param hints       │
                    └──────────────┬──────────────────────┘
                                   │ reward signal
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      EXPERIENCE REPLAY BUFFER       │
                    │  rl_memory.jsonl (ring buffer)      │
                    └──────────────┬──────────────────────┘
                                   │ batch sample
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      OPTIMIZER (CMA-ES)             │
                    │  searches param space               │
                    │  maximizes expected reward          │
                    └──────────────┬──────────────────────┘
                                   │ best params
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      CONFIG UPDATER                 │
                    │  writes rl_params.json              │
                    │  loaded at next scan cycle          │
                    └─────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import aiohttp

from rl_critic import (
    TradeExperience,
    evaluate_trade,
    extract_state,
    compute_reward,
)
from rl_memory import save_experience, load_experiences
from rl_optimizer import (
    run_optimization_step,
    load_optimizer_state,
    vector_to_params,
)

log = logging.getLogger(__name__)

RL_PARAMS_FILE = Path("rl_params.json")
RL_LOG_FILE    = Path("rl_log.jsonl")

# Trigger optimization after this many new experiences
OPTIMIZE_EVERY_N_TRADES = 20


class RLAgent:
    """
    Actor-Critic RL Agent.

    Singleton (use rl_agent global instance).
    Thread-safe via asyncio (single event loop in bot).
    """

    def __init__(self):
        self._trades_since_optimize = 0
        self._pending_entries: Dict[str, dict] = {}   # sym → entry event
        self._best_params: Dict[str, float] = {}
        self._load_params()

    def _load_params(self) -> None:
        """Load last optimized params from rl_params.json."""
        if RL_PARAMS_FILE.exists():
            try:
                self._best_params = json.loads(RL_PARAMS_FILE.read_text())
                log.info("RL: loaded %d optimized params", len(self._best_params))
            except Exception as e:
                log.warning("RL: failed to load params: %s", e)

    def register_entry(self, sym: str, entry_event: dict) -> None:
        """
        Called when bot OPENS a position.
        Stores the entry event keyed by symbol for later pairing.
        """
        self._pending_entries[sym] = entry_event

    async def on_trade_closed(
        self,
        exit_event: dict,
        *,
        session: aiohttp.ClientSession,
        btc_vs_ema50: float = 0.0,
        is_bull_day: bool = False,
        market_regime: str = "neutral",
        forward_data: Optional[dict] = None,
    ) -> None:
        """
        Called when a position CLOSES.
        Builds TradeExperience, calls critic, stores in memory.
        Triggers optimization if enough new trades have accumulated.

        Args:
            exit_event:     The exit event dict from botlog
            session:        aiohttp session for Claude API calls
            btc_vs_ema50:   BTC distance from EMA50 at exit time
            is_bull_day:    Bull day flag at exit time
            market_regime:  Market regime string
            forward_data:   Optional dict with T+3/5/10 actual returns
        """
        sym  = exit_event.get("sym", "?")
        mode = exit_event.get("mode", "?")
        tf   = exit_event.get("tf", "15m")

        # Retrieve paired entry event
        entry_event = self._pending_entries.pop(sym, None)
        state = extract_state(entry_event) if entry_event else {}

        # Build experience
        exp = TradeExperience(
            trade_id  = str(uuid.uuid4())[:8],
            sym       = sym,
            tf        = tf,
            mode      = mode,
            ts_entry  = (entry_event or {}).get("ts", ""),
            ts_exit   = exit_event.get("ts", ""),
            btc_vs_ema50   = btc_vs_ema50,
            is_bull_day    = is_bull_day,
            market_regime  = market_regime,
            state          = state,
            trail_k        = float((entry_event or {}).get("trail_k", 2.0)),
            max_hold_bars  = int((entry_event or {}).get("max_hold_bars", 16)),
            entry_price    = float(exit_event.get("entry_price", 0.0)),
            exit_price     = float(exit_event.get("exit_price", 0.0)),
            pnl_pct        = float(exit_event.get("pnl_pct", 0.0)),
            bars_held      = int(exit_event.get("bars_held", 0)),
            exit_reason    = exit_event.get("reason", ""),
        )

        # Fill forward accuracy if available
        if forward_data:
            exp.t3_correct  = forward_data.get("t3_correct")
            exp.t5_correct  = forward_data.get("t5_correct")
            exp.t10_correct = forward_data.get("t10_correct")
            exp.t3_ret      = forward_data.get("t3_ret")
            exp.t5_ret      = forward_data.get("t5_ret")
            exp.t10_ret     = forward_data.get("t10_ret")

        # Call critic asynchronously (non-blocking)
        try:
            exp = await evaluate_trade(session, exp)
            log.info(
                "RL critic: %s [%s] PnL=%.2f%% reward=%.3f score=%.2f | %s",
                sym, mode, exp.pnl_pct, exp.reward or 0,
                exp.critic_score or 0, (exp.critic_feedback or "")[:80],
            )
        except Exception as e:
            log.warning("RL critic failed for %s: %s", sym, e)
            exp.reward = compute_reward(exp)

        # Store experience
        save_experience(exp)
        self._log_rl_event(exp)
        self._trades_since_optimize += 1

        # Apply critic param hints immediately (light, online adjustment)
        if exp.critic_params:
            self._apply_critic_hints(exp.critic_params)

        # Trigger heavy optimization if enough new trades
        if self._trades_since_optimize >= OPTIMIZE_EVERY_N_TRADES:
            asyncio.create_task(self._run_optimization_async())
            self._trades_since_optimize = 0

    async def _run_optimization_async(self) -> None:
        """Run CMA-ES optimization in a thread pool to not block event loop."""
        try:
            log.info("RL: starting optimization cycle...")
            best = await asyncio.to_thread(
                run_optimization_step,
                batch_size=128,
                n_generations=10,
                population=20,
            )
            if best:
                self._best_params.update(best)
                RL_PARAMS_FILE.write_text(json.dumps(self._best_params, indent=2))
                log.info("RL: optimization complete, updated %d params", len(best))
        except Exception as e:
            log.error("RL optimization failed: %s", e)

    def _apply_critic_hints(self, hints: dict) -> None:
        """
        Apply lightweight online adjustment from critic param hints.
        Small nudges to avoid over-reacting to individual trades.

        The "direction" and "magnitude" from the critic are translated
        to small increments on the current best_params.
        """
        adjustments = hints.get("adjustments", [])
        MAGNITUDE_MAP = {"small": 0.02, "medium": 0.05, "large": 0.10}

        for adj in adjustments:
            param     = adj.get("param", "")
            direction = adj.get("direction", "keep")
            magnitude = MAGNITUDE_MAP.get(adj.get("magnitude", "small"), 0.02)

            if direction == "keep" or param not in self._best_params:
                continue

            current = self._best_params.get(param, 0.0)
            delta   = current * magnitude
            if direction == "increase":
                self._best_params[param] = current + delta
            elif direction == "decrease":
                self._best_params[param] = max(0.001, current - delta)

    def apply_optimized_params(self) -> bool:
        """
        Apply optimized params to the live config module.
        Called on bot startup and after each optimization cycle.

        Returns True if params were applied, False if none available.
        """
        if not self._best_params:
            return False

        try:
            import config
            applied = 0
            for name, value in self._best_params.items():
                if hasattr(config, name):
                    # Type preservation: int params stay int
                    current = getattr(config, name)
                    if isinstance(current, int):
                        value = int(round(value))
                    setattr(config, name, value)
                    applied += 1
            log.info("RL: applied %d optimized params to config", applied)
            return True
        except Exception as e:
            log.error("RL: failed to apply params: %s", e)
            return False

    def get_param_override(self, param_name: str) -> Optional[float]:
        """
        Get the RL-optimized value for a specific config parameter.
        Returns None if not optimized yet (use config default).
        """
        return self._best_params.get(param_name)

    def _log_rl_event(self, exp: TradeExperience) -> None:
        """Append a summary line to rl_log.jsonl for dashboard."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trade_id": exp.trade_id,
            "sym": exp.sym,
            "mode": exp.mode,
            "pnl_pct": round(exp.pnl_pct, 3),
            "reward": round(exp.reward, 4) if exp.reward else None,
            "critic_score": round(exp.critic_score, 3) if exp.critic_score else None,
            "verdict": (exp.critic_feedback or "")[:30],
        }
        with RL_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def status(self) -> dict:
        """Current RL agent status for bot dashboard."""
        state = load_optimizer_state()
        exps  = load_experiences(n=500)
        return {
            "optimizer_generation": state.generation,
            "best_reward": round(state.best_reward, 4),
            "experiences_total": len(load_experiences()),
            "experiences_sample": len(exps),
            "trades_since_optimize": self._trades_since_optimize,
            "params_optimized": len(self._best_params),
            "top_params": {
                k: round(v, 3) for k, v in list(self._best_params.items())[:5]
            },
        }


# Singleton instance
rl_agent = RLAgent()
