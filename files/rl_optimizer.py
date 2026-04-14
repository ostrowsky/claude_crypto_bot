"""
rl_optimizer.py — Policy Parameter Optimizer

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) —
a gradient-free black-box optimizer ideal for:
  - Non-differentiable reward functions (rule-based bot)
  - Noisy environments (crypto markets)
  - ~10-50 parameter dimensions
  - Small populations (works with 50-200 experiences)

The "policy" is a dict of config.py parameters.
The optimizer searches for parameter values that maximize
the expected reward across the experience replay buffer.

Parameter space: selected subset of config.py numeric params,
grouped by mode to allow mode-specific optimization.

Why CMA-ES over gradient descent:
  - The actor (bot) is not differentiable — can't backprop through rules
  - CMA-ES naturally handles discrete params (MACD_CONFIRM_BARS etc)
  - Robust to local optima via covariance matrix adaptation
  - Works with small datasets (50+ samples)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rl_critic import TradeExperience
from rl_memory import load_experiences, sample_batch, memory_stats

log = logging.getLogger(__name__)

OPTIMIZER_STATE_FILE = Path("rl_optimizer_state.json")

# ── Parameter space definition ─────────────────────────────────────────────────
# Each param: (config_name, min, max, step, mode_specific)
# mode_specific=True → separate value per signal mode
PARAM_SPACE: List[Tuple[str, float, float, float, bool]] = [
    # Entry quality filters
    ("EMA_SLOPE_MIN",              0.05,  0.40, 0.01, False),
    ("ADX_MIN",                    15.0,  35.0, 0.5,  False),
    ("VOL_MULT",                   0.80,  2.50, 0.05, False),
    ("ENTRY_VOL_HARD_FLOOR",       0.20,  0.80, 0.05, False),
    ("ENTRY_VOL_FLOOR_NONBULL",    0.50,  1.50, 0.05, False),
    # Mode-specific thresholds
    ("RETEST_SLOPE_MIN",           0.05,  0.30, 0.01, False),
    ("RETEST_RSI_MAX",             55.0,  75.0, 1.0,  False),
    ("RETEST_DAILY_RANGE_HARD_CAP",6.0,  15.0, 0.5,  False),
    ("ALIGNMENT_SLOPE_MIN",        0.03,  0.20, 0.01, False),
    ("ALIGNMENT_MACD_BARS",        2.0,   8.0,  1.0,  False),
    ("TREND_SLOPE_MIN",            0.10,  0.50, 0.02, False),
    ("TREND_ADX_MIN",              18.0,  35.0, 0.5,  False),
    ("IMPULSE_SPEED_ADX_MIN",      14.0,  28.0, 0.5,  False),
    # Exit / risk management
    ("ATR_TRAIL_K",                1.5,   3.5,  0.1,  False),
    ("ATR_TRAIL_K_RETEST",         1.2,   3.0,  0.1,  False),
    ("ATR_TRAIL_K_BREAKOUT",       1.2,   3.0,  0.1,  False),
    ("MAX_HOLD_BARS_RETEST",       8.0,   32.0, 2.0,  False),
    ("MAX_HOLD_BARS_BREAKOUT",     6.0,   20.0, 2.0,  False),
    ("MIN_BARS_BEFORE_ADX_EXIT",   2.0,   10.0, 1.0,  False),
    # Portfolio / position sizing
    ("MAX_OPEN_POSITIONS_NONBULL", 2.0,   8.0,  1.0,  False),
    ("ENTRY_VOL_FLOOR_IMPULSE",    0.80,  2.50, 0.10, False),
    # ML gating
    ("ML_TREND_NONBULL_MIN_PROBA", 0.35,  0.75, 0.02, False),
    ("ML_GENERAL_SCORE_WEIGHT",    0.0,  30.0,  1.0,  False),
]

PARAM_NAMES = [p[0] for p in PARAM_SPACE]
PARAM_MINS  = [p[1] for p in PARAM_SPACE]
PARAM_MAXS  = [p[2] for p in PARAM_SPACE]
PARAM_STEPS = [p[3] for p in PARAM_SPACE]
N_PARAMS    = len(PARAM_SPACE)


@dataclass
class OptimizerState:
    """Persistent state of the CMA-ES optimizer."""
    generation:     int   = 0
    mean:           List[float] = field(default_factory=lambda: [
        (mn + mx) / 2 for mn, mx in zip(PARAM_MINS, PARAM_MAXS)
    ])
    sigma:          float = 0.3           # step size
    path_c:         List[float] = field(default_factory=lambda: [0.0] * N_PARAMS)
    path_sigma:     float = 0.0
    C:              List[List[float]] = field(default_factory=lambda:
                        [[1.0 if i==j else 0.0 for j in range(N_PARAMS)]
                         for i in range(N_PARAMS)])
    best_params:    Dict[str, float] = field(default_factory=dict)
    best_reward:    float = -999.0
    history:        List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "mean": self.mean,
            "sigma": self.sigma,
            "path_c": self.path_c,
            "path_sigma": self.path_sigma,
            "best_reward": self.best_reward,
            "best_params": self.best_params,
            "history": self.history[-20:],   # keep last 20
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerState":
        s = cls()
        s.generation  = d.get("generation", 0)
        s.mean        = d.get("mean", s.mean)
        s.sigma       = d.get("sigma", 0.3)
        s.path_c      = d.get("path_c", s.path_c)
        s.path_sigma  = d.get("path_sigma", 0.0)
        s.best_reward = d.get("best_reward", -999.0)
        s.best_params = d.get("best_params", {})
        s.history     = d.get("history", [])
        return s


def load_optimizer_state() -> OptimizerState:
    if OPTIMIZER_STATE_FILE.exists():
        try:
            d = json.loads(OPTIMIZER_STATE_FILE.read_text())
            return OptimizerState.from_dict(d)
        except Exception:
            pass
    return OptimizerState()


def save_optimizer_state(state: OptimizerState) -> None:
    OPTIMIZER_STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


def _clip_and_round(val: float, idx: int) -> float:
    """Clip to [min, max] and round to nearest step."""
    mn, mx, step = PARAM_MINS[idx], PARAM_MAXS[idx], PARAM_STEPS[idx]
    val = max(mn, min(mx, val))
    if step > 0:
        val = round(val / step) * step
    return round(val, 4)


def vector_to_params(vec: List[float]) -> Dict[str, float]:
    """Convert raw vector to named config params (clipped + rounded)."""
    return {
        PARAM_NAMES[i]: _clip_and_round(vec[i], i)
        for i in range(N_PARAMS)
    }


def params_to_vector(params: Dict[str, float]) -> List[float]:
    """Convert named params dict back to normalized vector."""
    mean = [(mn + mx) / 2 for mn, mx in zip(PARAM_MINS, PARAM_MAXS)]
    return [params.get(PARAM_NAMES[i], mean[i]) for i in range(N_PARAMS)]


# ── Reward simulation ──────────────────────────────────────────────────────────

def simulate_reward_with_params(
    params: Dict[str, float],
    experiences: List[TradeExperience],
) -> float:
    """
    Estimate expected reward if the bot had used `params` on the given experiences.

    This is a lightweight simulation:
    - For each experience, check if the entry would have been ALLOWED by these params
    - If blocked, reward = 0 (neutral — no loss, no gain)
    - If allowed, use the actual recorded reward
    - Then add a diversity bonus for not over-filtering

    This is the objective function for CMA-ES.
    """
    if not experiences:
        return -1.0

    total_reward = 0.0
    n_allowed = 0
    n_blocked = 0

    for exp in experiences:
        state = exp.state
        allowed = _would_entry_be_allowed(params, exp)

        if allowed:
            r = exp.reward if exp.reward is not None else (exp.pnl_pct / 1.5)
            total_reward += r
            n_allowed += 1
        else:
            # Blocking a bad trade is good; blocking a good one is bad
            if exp.pnl_pct < -0.3:
                total_reward += 0.15   # correctly avoided bad entry
            else:
                total_reward -= 0.05   # missed a good entry
            n_blocked += 1

    # Over-filtering penalty: if we block >80% of entries, that's too conservative
    if n_allowed + n_blocked > 0:
        block_rate = n_blocked / (n_allowed + n_blocked)
        if block_rate > 0.80:
            total_reward -= (block_rate - 0.80) * len(experiences) * 0.1

    return total_reward / max(1, len(experiences))


def _would_entry_be_allowed(params: Dict[str, float], exp: TradeExperience) -> bool:
    """
    Lightweight rule check: would this entry have been taken with `params`?
    Checks the most important entry filters.
    """
    s = exp.state
    vol   = s.get("vol_x", 1.0)
    adx   = s.get("adx", 20.0)
    slope = s.get("slope_pct", 0.1)
    rsi   = s.get("rsi", 55.0)
    dr    = s.get("daily_range", 3.0)
    ml    = s.get("ml_proba", 0.5)
    mode  = exp.mode

    # Global vol floor
    if vol < params.get("ENTRY_VOL_HARD_FLOOR", 0.30):
        return False

    # Non-bull vol floor
    if not exp.is_bull_day:
        nb_floor = params.get("ENTRY_VOL_FLOOR_NONBULL", 0.70)
        if mode == "impulse_speed":
            nb_floor = max(nb_floor, params.get("ENTRY_VOL_FLOOR_IMPULSE", 1.20))
        if vol < nb_floor:
            return False

    # ADX minimum
    if adx < params.get("ADX_MIN", 20.0):
        if mode not in ("alignment", "retest"):  # these have own ADX thresholds
            return False

    # Slope minimum
    if mode == "trend":
        if slope < params.get("TREND_SLOPE_MIN", 0.20):
            return False
        if adx < params.get("TREND_ADX_MIN", 24.0):
            return False

    elif mode in ("impulse_speed",):
        if adx < params.get("IMPULSE_SPEED_ADX_MIN", 18.0):
            return False

    elif mode == "retest":
        if slope < params.get("RETEST_SLOPE_MIN", 0.10):
            return False
        if rsi > params.get("RETEST_RSI_MAX", 65.0):
            return False
        if dr > params.get("RETEST_DAILY_RANGE_HARD_CAP", 10.0):
            return False

    elif mode == "alignment":
        if slope < params.get("ALIGNMENT_SLOPE_MIN", 0.05):
            return False

    return True


# ── Simplified CMA-ES ──────────────────────────────────────────────────────────

class SimpleCMAES:
    """
    Simplified (μ, λ)-CMA-ES for policy optimization.

    Uses diagonal covariance approximation for efficiency with N_PARAMS~20.
    Full covariance would be more powerful but requires more samples.
    """

    def __init__(
        self,
        state: OptimizerState,
        *,
        lam: int = 20,     # population size
        mu: int  = 10,     # elite fraction
    ):
        self.state = state
        self.lam = lam
        self.mu  = mu
        self.n   = N_PARAMS

        # CMA-ES adaptation weights
        self.w = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        w_sum = sum(self.w)
        self.w = [wi / w_sum for wi in self.w]
        self.mueff = 1.0 / sum(wi**2 for wi in self.w)

        # Learning rates
        self.cc    = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs    = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1    = 2 / ((self.n + 1.3)**2 + self.mueff)
        self.cmu   = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n+2)**2 + self.mueff))
        self.damps = 1 + 2*max(0, math.sqrt((self.mueff-1)/(self.n+1)) - 1) + self.cs

        # Using diagonal covariance: just the variance per dimension
        self.var = [1.0] * self.n   # variance per dimension (diagonal of C)

    def ask(self) -> List[List[float]]:
        """Sample λ candidate solutions from current distribution."""
        import random
        candidates = []
        for _ in range(self.lam):
            z = [random.gauss(0, 1) for _ in range(self.n)]
            x = [
                self.state.mean[i] + self.state.sigma * math.sqrt(self.var[i]) * z[i]
                for i in range(self.n)
            ]
            # Clip to bounds
            x = [_clip_and_round(x[i], i) for i in range(self.n)]
            candidates.append(x)
        return candidates

    def tell(self, candidates: List[List[float]], fitnesses: List[float]) -> None:
        """Update distribution based on evaluated candidates (higher = better)."""
        # Sort by fitness (descending)
        ranked = sorted(zip(fitnesses, candidates), reverse=True)
        elite  = [c for _, c in ranked[:self.mu]]

        # New mean
        new_mean = [
            sum(self.w[j] * elite[j][i] for j in range(self.mu))
            for i in range(self.n)
        ]

        # Update path_sigma
        step = [(new_mean[i] - self.state.mean[i]) / self.state.sigma for i in range(self.n)]
        path_s = self.state.path_sigma
        path_s = (1 - self.cs) * path_s + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * math.sqrt(sum(s**2 for s in step) / self.n)

        # Update sigma
        sigma = self.state.sigma * math.exp(self.cs / self.damps * (path_s - 1))
        sigma = max(0.01, min(1.0, sigma))

        # Update diagonal variance (simplified covariance)
        for i in range(self.n):
            elite_centered = [(e[i] - self.state.mean[i]) / self.state.sigma for e in elite]
            var_update = sum(self.w[j] * elite_centered[j]**2 for j in range(self.mu))
            self.var[i] = (1 - self.c1 - self.cmu) * self.var[i] + self.cmu * var_update
            self.var[i] = max(0.01, self.var[i])

        self.state.mean = new_mean
        self.state.sigma = sigma
        self.state.path_sigma = path_s
        self.state.generation += 1


# ── Main optimization entry point ──────────────────────────────────────────────

def run_optimization_step(
    *,
    batch_size: int = 128,
    n_generations: int = 5,
    population: int = 20,
    verbose: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Run one optimization cycle:
    1. Load experience replay buffer
    2. Sample batch
    3. Run CMA-ES for n_generations
    4. Return best params found

    Returns None if insufficient data.
    """
    experiences = load_experiences()
    if len(experiences) < 50:
        log.info("RL optimizer: only %d experiences, need ≥50", len(experiences))
        return None

    batch = sample_batch(experiences, batch_size, mode_balanced=True)
    state = load_optimizer_state()
    cmaes = SimpleCMAES(state, lam=population, mu=population//2)

    best_reward = state.best_reward
    best_params = dict(state.best_params)

    for gen in range(n_generations):
        candidates = cmaes.ask()
        fitnesses  = [
            simulate_reward_with_params(vector_to_params(c), batch)
            for c in candidates
        ]
        cmaes.tell(candidates, fitnesses)

        gen_best_f = max(fitnesses)
        gen_best_p = vector_to_params(candidates[fitnesses.index(gen_best_f)])

        if gen_best_f > best_reward:
            best_reward = gen_best_f
            best_params = gen_best_p

        if verbose:
            log.info(
                "RL gen %d/%d: best_reward=%.4f  sigma=%.4f",
                gen + 1, n_generations, gen_best_f, state.sigma,
            )

    state.best_reward = best_reward
    state.best_params = best_params
    state.history.append({
        "generation": state.generation,
        "best_reward": best_reward,
        "n_experiences": len(experiences),
    })

    save_optimizer_state(state)

    if verbose:
        mem = memory_stats(experiences)
        log.info(
            "RL optimization complete: gen=%d reward=%.4f WR=%.1f%% n=%d",
            state.generation, best_reward, mem.get("win_rate",0)*100, mem.get("total",0),
        )

    return best_params
