"""
exit_rl.py — RL-based Exit Policy

Learns when to exit positions using Q-learning with linear function
approximation.  Runs alongside existing rule-based exits as an
additional intelligence layer.

State:   position context (pnl, bars_held ratio, indicators, mode, regime)
Actions: hold (0), tighten_stop (1), exit (2)

Architecture:
  Q(s, a) = W[a] · φ(s)   — linear in features, separate weights per action

The policy runs alongside existing rule-based exits:
  - Rule-based exits remain as safety net (RSI overbought, ATR trail, max_hold)
  - RL policy can trigger EARLIER exits or recommend holding through soft signals
  - Tighten action reduces effective trail_k for the current bar

Integration:
  monitor.py    → recommend_exit_action() at each bar check
  rl_agent.py   → policy updates after trade closes
  offline_rl.py → batch training from accumulated data
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Actions ───────────────────────────────────────────────────────────────────
ACTION_HOLD = 0
ACTION_TIGHTEN = 1
ACTION_EXIT = 2
N_ACTIONS = 3
ACTION_NAMES = ["hold", "tighten", "exit"]

# ── State feature extraction ──────────────────────────────────────────────────

EXIT_FEATURE_NAMES = [
    "pnl_norm",        # current PnL / target PnL
    "bars_ratio",      # bars_held / max_hold_bars
    "rsi_norm",        # current RSI / 100
    "slope_norm",      # current EMA slope normalized
    "adx_norm",        # current ADX / 50
    "adx_change",      # (current_adx - entry_adx) / entry_adx
    "vol_x_norm",      # current volume ratio normalized
    "macd_trend",      # sign of MACD change
    "trail_distance",  # (close - trail_stop) / close, scaled
    "is_bull_day",     # binary
    "regime_code",     # encoded market regime
    "mode_code",       # encoded signal mode
    "bias",            # constant 1.0
]
N_EXIT_FEATURES = len(EXIT_FEATURE_NAMES)

STATE_FILE = Path("exit_policy.json")

MODE_MAP = {
    "trend": 0.0, "strong_trend": 0.2, "impulse_speed": 0.4,
    "retest": 0.6, "breakout": 0.8, "alignment": 0.9, "impulse": 1.0,
}
REGIME_MAP = {
    "bull_trend": 1.0, "bull": 0.7, "neutral": 0.0,
    "consolidation": -0.3, "bear": -0.7, "bear_trend": -1.0,
}


def extract_exit_state(
    *,
    current_pnl: float,
    bars_held: int,
    max_hold_bars: int,
    rsi: float = 50.0,
    slope: float = 0.0,
    adx: float = 20.0,
    entry_adx: float = 20.0,
    vol_x: float = 1.0,
    macd_hist: float = 0.0,
    prev_macd_hist: float = 0.0,
    close: float = 0.0,
    trail_stop: float = 0.0,
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    mode: str = "trend",
) -> np.ndarray:
    """Extract normalized state vector for exit policy."""
    pnl_norm = np.clip(current_pnl / 1.5, -3, 3)
    bars_ratio = np.clip(bars_held / max(1, max_hold_bars), 0, 2)
    rsi_norm = np.clip(rsi / 100.0, 0, 1)
    slope_norm = np.clip(slope / 0.5, -2, 2)
    adx_norm = np.clip(adx / 50.0, 0, 1)
    adx_change = (
        np.clip((adx - entry_adx) / max(1.0, entry_adx), -1, 1)
        if entry_adx > 0 else 0.0
    )
    vol_x_norm = np.clip(vol_x / 3.0, 0, 2)
    macd_trend = (
        1.0 if macd_hist > prev_macd_hist
        else (-1.0 if macd_hist < prev_macd_hist else 0.0)
    )
    trail_dist = 0.1
    if close > 0 and trail_stop > 0:
        trail_dist = np.clip((close - trail_stop) / close, -0.1, 0.2)

    return np.array([
        pnl_norm,
        bars_ratio,
        rsi_norm,
        slope_norm,
        adx_norm,
        adx_change,
        vol_x_norm,
        macd_trend,
        trail_dist * 10.0,   # scale up for better gradient signal
        1.0 if is_bull_day else 0.0,
        REGIME_MAP.get(market_regime, 0.0),
        MODE_MAP.get(mode, 0.0),
        1.0,                  # bias
    ], dtype=np.float64)


# ── Linear Q-Network ─────────────────────────────────────────────────────────

class ExitPolicy:
    """
    Linear Q-function for exit decisions.
    Q(s, a) = W[a] · s        W ∈ R^{N_ACTIONS × N_EXIT_FEATURES}
    """

    def __init__(self, learning_rate: float = 0.01, discount: float = 0.95):
        self.lr = learning_rate
        self.gamma = discount
        self.W = np.zeros((N_ACTIONS, N_EXIT_FEATURES), dtype=np.float64)
        self.n_updates = 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q(s, a) for all actions.  Shape: (N_ACTIONS,)"""
        return self.W @ state

    def select_action(
        self,
        state: np.ndarray,
        *,
        exit_threshold: float = 0.3,
        tighten_threshold: float = 0.15,
    ) -> Tuple[int, dict]:
        """
        Select action based on Q-values.
        Returns action only if exit/tighten is clearly better than holding.
        """
        q = self.q_values(state)
        q_hold = float(q[ACTION_HOLD])
        q_tighten = float(q[ACTION_TIGHTEN])
        q_exit = float(q[ACTION_EXIT])

        action = ACTION_HOLD
        if q_exit - q_hold > exit_threshold:
            action = ACTION_EXIT
        elif q_tighten - q_hold > tighten_threshold:
            action = ACTION_TIGHTEN

        return action, {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "q_hold": round(q_hold, 4),
            "q_tighten": round(q_tighten, 4),
            "q_exit": round(q_exit, 4),
            "adv_exit": round(q_exit - q_hold, 4),
            "adv_tighten": round(q_tighten - q_hold, 4),
        }

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        done: bool = True,
    ) -> float:
        """
        Q-learning update.
        Terminal:     target = reward
        Non-terminal: target = reward + γ · max_a' Q(s', a')
        """
        q_current = float(self.W[action] @ state)

        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.W @ next_state))

        td_error = target - q_current
        self.W[action] += self.lr * td_error * state
        self.W = np.clip(self.W, -10.0, 10.0)
        self.n_updates += 1
        return td_error

    def batch_update(
        self,
        samples: List[Tuple[np.ndarray, int, float, Optional[np.ndarray], bool]],
    ) -> dict:
        """Batch update.  Returns training statistics."""
        if not samples:
            return {"n": 0}
        td_errors = []
        for state, action, reward, next_state, done in samples:
            if 0 <= action < N_ACTIONS:
                err = self.update(state, action, reward, next_state, done)
                td_errors.append(err)
        return {
            "n": len(td_errors),
            "mean_td_error": round(float(np.mean(td_errors)), 5) if td_errors else 0,
            "max_td_error": round(float(np.max(np.abs(td_errors))), 5) if td_errors else 0,
            "total_updates": self.n_updates,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or STATE_FILE
        data = {
            "W": self.W.tolist(),
            "lr": self.lr,
            "gamma": self.gamma,
            "n_updates": self.n_updates,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        log.info("Exit policy saved: %d updates", self.n_updates)

    @classmethod
    def load(cls, path: Optional[Path] = None, **kwargs) -> "ExitPolicy":
        path = path or STATE_FILE
        policy = cls(**kwargs)
        if not path.exists():
            return policy
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            W = np.array(data["W"], dtype=np.float64)
            if W.shape == (N_ACTIONS, N_EXIT_FEATURES):
                policy.W = W
            policy.lr = data.get("lr", policy.lr)
            policy.gamma = data.get("gamma", policy.gamma)
            policy.n_updates = data.get("n_updates", 0)
            log.info("Exit policy loaded: %d updates", policy.n_updates)
        except Exception as e:
            log.warning("Failed to load exit policy: %s", e)
        return policy

    def weight_summary(self) -> dict:
        """Summary of learned weights for reporting."""
        result = {
            "n_updates": self.n_updates,
            "weight_norms": {},
            "bias_values": {},
            "top_features": {},
        }
        for a in range(N_ACTIONS):
            name = ACTION_NAMES[a]
            w = self.W[a]
            result["weight_norms"][name] = round(float(np.linalg.norm(w)), 4)
            result["bias_values"][name] = round(float(w[-1]), 4)
            top_idx = np.argsort(np.abs(w))[-3:][::-1]
            result["top_features"][name] = [
                {"feature": EXIT_FEATURE_NAMES[j], "weight": round(float(w[j]), 4)}
                for j in top_idx
            ]
        return result


# ── Singleton & convenience ───────────────────────────────────────────────────

_policy_instance: Optional[ExitPolicy] = None


def get_exit_policy(**kwargs) -> ExitPolicy:
    global _policy_instance
    if _policy_instance is None:
        _policy_instance = ExitPolicy.load(**kwargs)
    return _policy_instance


def recommend_exit_action(
    *,
    current_pnl: float,
    bars_held: int,
    max_hold_bars: int,
    rsi: float = 50.0,
    slope: float = 0.0,
    adx: float = 20.0,
    entry_adx: float = 20.0,
    vol_x: float = 1.0,
    macd_hist: float = 0.0,
    prev_macd_hist: float = 0.0,
    close: float = 0.0,
    trail_stop: float = 0.0,
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    mode: str = "trend",
    exit_threshold: float = 0.3,
    tighten_threshold: float = 0.15,
) -> Tuple[int, dict]:
    """
    Main entry point: get exit action recommendation for a position.
    Returns (action, info) where action is ACTION_HOLD / ACTION_TIGHTEN / ACTION_EXIT.
    """
    policy = get_exit_policy()
    state = extract_exit_state(
        current_pnl=current_pnl,
        bars_held=bars_held,
        max_hold_bars=max_hold_bars,
        rsi=rsi, slope=slope, adx=adx,
        entry_adx=entry_adx, vol_x=vol_x,
        macd_hist=macd_hist, prev_macd_hist=prev_macd_hist,
        close=close, trail_stop=trail_stop,
        is_bull_day=is_bull_day,
        market_regime=market_regime,
        mode=mode,
    )
    return policy.select_action(
        state,
        exit_threshold=exit_threshold,
        tighten_threshold=tighten_threshold,
    )
