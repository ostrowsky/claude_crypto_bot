"""
contextual_bandit.py — LinUCB Contextual Bandits

Two bandits:
  1. Entry Bandit — enter/skip decision per coin candidate
     Arms: 0=SKIP, 1=ENTER
     Reward: +1 if coin in top20 EOD gainer, 0 otherwise

  2. Trail Bandit — trail_k / max_hold_bars selection
     Arms: 5 multiplier profiles (very_tight...very_wide)
     Reward: trade PnL composite

Algorithm: LinUCB (Li et al., 2010) with disjoint linear models per arm.

Integration:
  monitor.py    -> should_enter() gate + select_entry_profile() at entry
  rl_agent.py   -> feedback_entry() after trade closes (trail bandit)
  offline_rl.py -> batch training for both bandits
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import config as config  # type: ignore
except ImportError:
    config = None  # type: ignore

log = logging.getLogger(__name__)

# ── Entry bandit arms ─────────────────────────────────────────────────────────

ENTRY_ARMS = [
    {"name": "skip",  "action": 0},
    {"name": "enter", "action": 1},
]
N_ENTRY_ARMS = len(ENTRY_ARMS)

# ── Trail bandit arms ────────────────────────────────────────────────────────

TRAIL_ARMS = [
    {"name": "very_tight", "trail_k_mult": 0.70, "hold_mult": 0.50},
    {"name": "tight",      "trail_k_mult": 0.85, "hold_mult": 0.75},
    {"name": "default",    "trail_k_mult": 1.00, "hold_mult": 1.00},
    {"name": "wide",       "trail_k_mult": 1.20, "hold_mult": 1.25},
    {"name": "very_wide",  "trail_k_mult": 1.40, "hold_mult": 1.50},
]
N_TRAIL_ARMS = len(TRAIL_ARMS)

# Backward compatibility aliases
ARMS = TRAIL_ARMS
N_ARMS = N_TRAIL_ARMS

# ── Context feature extraction ────────────────────────────────────────────────

FEATURE_NAMES = [
    "slope_norm", "adx_norm", "rsi_norm", "vol_x_norm",
    "ml_proba", "btc_vs_ema50_norm", "daily_range_norm",
    "macd_sign", "is_bull_day", "regime_bull", "regime_bear",
    "tf_1h", "mode_trend", "mode_retest", "mode_breakout",
    "mode_impulse", "mode_alignment", "bias",
]
N_FEATURES = len(FEATURE_NAMES)

# State files
ENTRY_STATE_FILE = Path("bandit_entry_state.json")
STATE_FILE = Path("bandit_state.json")  # trail bandit
PENDING_FILE = Path("bandit_pending.json")


def extract_context(
    state: dict,
    *,
    mode: str = "trend",
    tf: str = "15m",
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    btc_vs_ema50: float = 0.0,
) -> np.ndarray:
    """Extract normalized context vector from entry state dict."""
    slope = state.get("slope_pct", 0.0)
    adx = state.get("adx", 20.0)
    rsi = state.get("rsi", 50.0)
    vol_x = state.get("vol_x", 1.0)
    ml_proba = state.get("ml_proba", 0.5)
    daily_range = state.get("daily_range", 3.0)
    macd_hist = state.get("macd_hist", 0.0)

    return np.array([
        np.clip(slope / 0.5, -2, 2),
        np.clip(adx / 50.0, 0, 1),
        np.clip(rsi / 100.0, 0, 1),
        np.clip(vol_x / 3.0, 0, 2),
        np.clip(ml_proba, 0, 1),
        np.clip(btc_vs_ema50 / 5.0, -2, 2),
        np.clip(daily_range / 10.0, 0, 2),
        1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0),
        1.0 if is_bull_day else 0.0,
        1.0 if "bull" in market_regime else 0.0,
        1.0 if "bear" in market_regime else 0.0,
        1.0 if tf == "1h" else 0.0,
        1.0 if mode in ("trend", "strong_trend") else 0.0,
        1.0 if mode == "retest" else 0.0,
        1.0 if mode == "breakout" else 0.0,
        1.0 if mode in ("impulse", "impulse_speed") else 0.0,
        1.0 if mode == "alignment" else 0.0,
        1.0,  # bias
    ], dtype=np.float64)


# ── LinUCB Algorithm ──────────────────────────────────────────────────────────

class LinUCBBandit:
    """
    LinUCB contextual bandit with disjoint linear models.

    Per arm a:
      A_a in R^{d x d}   (regularized design matrix, init = I)
      b_a in R^d          (reward-weighted feature sum)
      theta_a = A_a^{-1} . b_a  (estimated parameter vector)

    UCB_a(x) = theta_a . x + alpha * sqrt(x' . A_a^{-1} . x)
    """

    def __init__(self, n_arms: int, n_features: int = N_FEATURES, alpha: float = 1.5):
        self.alpha = alpha
        self.d = n_features
        self.k = n_arms
        self.A: List[np.ndarray] = [np.eye(self.d) for _ in range(self.k)]
        self.b: List[np.ndarray] = [np.zeros(self.d) for _ in range(self.k)]
        self.n_updates: List[int] = [0] * self.k
        self.total_updates = 0

    def select_arm(self, x: np.ndarray) -> Tuple[int, dict]:
        """Select best arm given context x.  Returns (arm_index, info)."""
        ucbs = np.zeros(self.k)
        for a in range(self.k):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)
            var = float(x @ A_inv @ x)
            ucbs[a] = mean + self.alpha * math.sqrt(max(0.0, var))

        best = int(np.argmax(ucbs))
        return best, {
            "arm": best,
            "ucb": round(float(ucbs[best]), 4),
            "ucbs": [round(float(u), 4) for u in ucbs],
        }

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        """Update model for arm after observing reward."""
        if arm < 0 or arm >= self.k:
            return
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.n_updates[arm] += 1
        self.total_updates += 1

    def batch_update(self, samples: List[Tuple[np.ndarray, int, float]]) -> int:
        """Batch update from (context, arm, reward) tuples."""
        count = 0
        for x, arm, reward in samples:
            if 0 <= arm < self.k:
                self.update(x, arm, reward)
                count += 1
        return count

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {
            "alpha": self.alpha,
            "n_arms": self.k,
            "total_updates": self.total_updates,
            "n_updates": self.n_updates,
            "arms": [
                {"A": self.A[a].tolist(), "b": self.b[a].tolist()}
                for a in range(self.k)
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        log.info("Bandit saved (%d arms, %d updates) -> %s",
                 self.k, self.total_updates, path.name)

    @classmethod
    def load(
        cls, path: Path, n_arms: int,
        n_features: int = N_FEATURES, alpha: float = 1.5,
    ) -> "LinUCBBandit":
        bandit = cls(n_arms, n_features, alpha)
        if not path.exists():
            return bandit
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            saved_n_arms = data.get("n_arms", len(data.get("arms", [])))
            if saved_n_arms != n_arms:
                log.warning("Bandit arm count mismatch: saved=%d, expected=%d — reset",
                            saved_n_arms, n_arms)
                return bandit
            bandit.alpha = data.get("alpha", alpha)
            bandit.total_updates = data.get("total_updates", 0)
            bandit.n_updates = data.get("n_updates", [0] * n_arms)
            for a, arm_data in enumerate(data.get("arms", [])):
                if a < n_arms:
                    bandit.A[a] = np.array(arm_data["A"], dtype=np.float64)
                    bandit.b[a] = np.array(arm_data["b"], dtype=np.float64)
            log.info("Bandit loaded (%d arms, %d updates) <- %s",
                     n_arms, bandit.total_updates, path.name)
        except Exception as e:
            log.warning("Failed to load bandit from %s: %s", path.name, e)
        return bandit

    def arm_stats(self, arm_names: Optional[List[str]] = None) -> List[dict]:
        """Per-arm statistics for reporting."""
        stats = []
        for a in range(self.k):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            name = arm_names[a] if arm_names and a < len(arm_names) else f"arm_{a}"
            stats.append({
                "arm": a,
                "name": name,
                "n_updates": self.n_updates[a],
                "theta_norm": round(float(np.linalg.norm(theta)), 4),
                "bias_est": round(float(theta[-1]), 4),
            })
        return stats


# ── Singletons ───────────────────────────────────────────────────────────────

_entry_bandit: Optional[LinUCBBandit] = None
_trail_bandit: Optional[LinUCBBandit] = None


def get_entry_bandit(alpha: float = 2.0) -> LinUCBBandit:
    """Get/create the enter/skip bandit (2 arms)."""
    global _entry_bandit
    if _entry_bandit is None:
        _entry_bandit = LinUCBBandit.load(
            ENTRY_STATE_FILE, N_ENTRY_ARMS, alpha=alpha,
        )
    return _entry_bandit


def get_trail_bandit(alpha: float = 1.5) -> LinUCBBandit:
    """Get/create the trail_k/hold bandit (5 arms)."""
    global _trail_bandit
    if _trail_bandit is None:
        _trail_bandit = LinUCBBandit.load(
            STATE_FILE, N_TRAIL_ARMS, alpha=alpha,
        )
    return _trail_bandit


def get_bandit(alpha: float = 1.5) -> LinUCBBandit:
    """Backward compat: returns trail bandit."""
    return get_trail_bandit(alpha)


# ── Enter/Skip API (NEW — main bandit) ──────────────────────────────────────

def should_enter(
    state: dict,
    *,
    sym: str = "",
    mode: str = "trend",
    tf: str = "15m",
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    btc_vs_ema50: float = 0.0,
    alpha: float = 2.0,
) -> Tuple[bool, dict]:
    """
    Decide whether to enter a coin candidate.

    Returns (should_enter_bool, info_dict).
    Stores decision in pending buffer for delayed EOD reward.
    """
    bandit = get_entry_bandit(alpha)

    # Until bandit has enough training data, default to ENTER
    MIN_TRAINING_UPDATES = 50
    if bandit.total_updates < MIN_TRAINING_UPDATES:
        return True, {
            "arm": 1, "arm_name": "enter", "enter": True,
            "ucbs": [0, 0], "sym": sym,
            "reason": f"untrained ({bandit.total_updates}/{MIN_TRAINING_UPDATES})",
        }

    x = extract_context(
        state, mode=mode, tf=tf,
        is_bull_day=is_bull_day,
        market_regime=market_regime,
        btc_vs_ema50=btc_vs_ema50,
    )
    arm, info = bandit.select_arm(x)

    enter = (arm == 1)  # arm 1 = ENTER
    info["enter"] = enter
    info["arm_name"] = ENTRY_ARMS[arm]["name"]
    info["sym"] = sym

    # Store for delayed reward resolution at EOD
    if sym:
        _store_pending_decision(sym, mode, tf, arm, x.tolist())

    return enter, info


def feedback_entry_decision(
    sym: str,
    is_top_gainer: bool,
    arm: int,
    context: Optional[np.ndarray] = None,
    *,
    state: Optional[dict] = None,
    mode: str = "trend",
    tf: str = "15m",
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    btc_vs_ema50: float = 0.0,
) -> None:
    """
    Feed top-gainer reward to the enter/skip bandit.

    Reward scheme (asymmetric — penalizes missed winners):
      ENTER + top gainer -> +1.0  (correct entry)
      ENTER + not top    -> -0.05 (small cost for wasted entry)
      SKIP  + top gainer -> -1.0  (missed opportunity — costly!)
      SKIP  + not top    ->  0.0  (neutral)
    """
    bandit = get_entry_bandit()

    if context is None and state is not None:
        context = extract_context(
            state, mode=mode, tf=tf,
            is_bull_day=is_bull_day,
            market_regime=market_regime,
            btc_vs_ema50=btc_vs_ema50,
        )
    if context is None:
        return

    if arm == 1:  # ENTER
        reward = 1.0 if is_top_gainer else -0.05
    else:  # SKIP
        reward = -1.0 if is_top_gainer else 0.0

    bandit.update(context, arm, reward)
    if bandit.total_updates % 10 == 0:
        bandit.save(ENTRY_STATE_FILE)


def resolve_pending_decisions(top_gainer_syms: List[str]) -> int:
    """
    Resolve all pending entry/skip decisions with top gainer data.
    Called at EOD when the top gainer list is available.

    Args:
        top_gainer_syms: list of symbol names (e.g. ["TRUUSDT", "AXLUSDT"])

    Returns number of resolved decisions.
    """
    pending = _load_pending_decisions()
    if not pending:
        return 0

    # Normalize: accept both "TRU" and "TRUUSDT"
    top_set = set()
    for s in top_gainer_syms:
        top_set.add(s.upper())
        top_set.add(s.upper().replace("USDT", ""))

    bandit = get_entry_bandit()
    resolved = 0

    for dec in pending:
        sym = dec.get("sym", "")
        arm = dec.get("arm", 1)
        ctx = dec.get("context")
        if ctx is None:
            continue

        x = np.array(ctx, dtype=np.float64)
        is_top = sym.upper() in top_set or sym.upper().replace("USDT", "") in top_set

        if arm == 1:  # ENTER
            reward = 1.0 if is_top else -0.05
        else:  # SKIP
            reward = -1.0 if is_top else 0.0

        bandit.update(x, arm, reward)
        resolved += 1

    bandit.save(ENTRY_STATE_FILE)
    _clear_pending_decisions()
    log.info("Resolved %d pending bandit decisions (%d top gainers)",
             resolved, len(top_gainer_syms))
    return resolved


# ── Pending decisions buffer ─────────────────────────────────────────────────

def _store_pending_decision(
    sym: str, mode: str, tf: str, arm: int, context_list: list,
) -> None:
    pending = _load_pending_decisions()
    pending.append({
        "sym": sym, "mode": mode, "tf": tf, "arm": arm,
        "context": context_list,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    PENDING_FILE.write_text(json.dumps(pending), encoding="utf-8")


def _load_pending_decisions() -> list:
    if PENDING_FILE.exists():
        try:
            return json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _clear_pending_decisions() -> None:
    if PENDING_FILE.exists():
        PENDING_FILE.write_text("[]", encoding="utf-8")


# ── Trail K selection API (legacy, still used) ──────────────────────────────

def select_entry_profile(
    state: dict,
    *,
    base_trail_k: float = 2.0,
    base_max_hold: int = 16,
    mode: str = "trend",
    tf: str = "15m",
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    btc_vs_ema50: float = 0.0,
    alpha: float = 1.5,
) -> Tuple[float, int, dict]:
    """
    Select trail_k and max_hold_bars for a new entry.
    Returns (adjusted_trail_k, adjusted_max_hold, bandit_info).
    """
    bandit = get_trail_bandit(alpha)
    x = extract_context(
        state, mode=mode, tf=tf,
        is_bull_day=is_bull_day,
        market_regime=market_regime,
        btc_vs_ema50=btc_vs_ema50,
    )
    arm, info = bandit.select_arm(x)

    trail_k = round(base_trail_k * TRAIL_ARMS[arm]["trail_k_mult"], 2)
    # Apply minimum trail_k floor: bandit learned to use 1.05 (very_tight arm) which causes
    # instant stop-outs on normal candle volatility. Enforce a sane minimum.
    trail_k_min = getattr(config, "BANDIT_TRAIL_K_MIN", 0.0) if config is not None else 0.0
    if trail_k_min > 0 and trail_k < trail_k_min:
        trail_k = round(trail_k_min, 2)
    max_hold = max(4, int(round(base_max_hold * TRAIL_ARMS[arm]["hold_mult"])))

    info["arm_name"] = TRAIL_ARMS[arm]["name"]
    info["trail_k_mult"] = TRAIL_ARMS[arm]["trail_k_mult"]
    info["hold_mult"] = TRAIL_ARMS[arm]["hold_mult"]
    info["base_trail_k"] = base_trail_k
    info["base_max_hold"] = base_max_hold
    info["final_trail_k"] = trail_k
    info["final_max_hold"] = max_hold

    return trail_k, max_hold, info


def feedback_entry(
    state: dict,
    arm: int,
    reward: float,
    *,
    mode: str = "trend",
    tf: str = "15m",
    is_bull_day: bool = False,
    market_regime: str = "neutral",
    btc_vs_ema50: float = 0.0,
    alpha: float = 1.5,
) -> None:
    """Feed reward back to trail bandit after trade closes."""
    bandit = get_trail_bandit(alpha)
    x = extract_context(
        state, mode=mode, tf=tf,
        is_bull_day=is_bull_day,
        market_regime=market_regime,
        btc_vs_ema50=btc_vs_ema50,
    )
    bandit.update(x, arm, reward)
    if bandit.total_updates % 10 == 0:
        bandit.save(STATE_FILE)


def map_trail_k_to_arm(trail_k_mult: float) -> int:
    """Map a trail_k multiplier to the closest arm index."""
    dists = [abs(trail_k_mult - arm["trail_k_mult"]) for arm in TRAIL_ARMS]
    return int(np.argmin(dists))
