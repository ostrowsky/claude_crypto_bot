"""
rl_memory.py — Experience Replay Buffer

Stores completed trade experiences (TradeExperience objects) in
rl_memory.jsonl for offline replay and batch optimization.

Design:
  - Ring buffer with configurable capacity
  - Priority sampling: recent + high-|reward| experiences more likely
  - Stratified by mode (prevents trend-dominated training)
  - Market regime tagging for context-conditional policy
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from rl_critic import TradeExperience

MEMORY_FILE = Path("rl_memory.jsonl")
MAX_MEMORY   = 5000   # ring buffer size
MIN_REPLAY   = 50     # minimum experiences before optimizer runs


def save_experience(exp: TradeExperience) -> None:
    """Append one experience to the persistent memory file."""
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(exp.__dict__) + "\n")


def load_experiences(n: Optional[int] = None) -> List[TradeExperience]:
    """Load all (or last n) experiences from memory."""
    if not MEMORY_FILE.exists():
        return []
    lines = MEMORY_FILE.read_text(encoding="utf-8").strip().splitlines()
    if n:
        lines = lines[-n:]
    result = []
    for line in lines:
        try:
            d = json.loads(line)
            result.append(TradeExperience(**{k: v for k, v in d.items()
                                             if k in TradeExperience.__dataclass_fields__}))
        except Exception:
            pass
    return result


def sample_batch(
    experiences: List[TradeExperience],
    batch_size: int = 64,
    *,
    priority_recent: float = 0.7,
    priority_reward: float = 0.3,
    mode_balanced: bool = True,
) -> List[TradeExperience]:
    """
    Priority-weighted sampling for batch optimization.

    Probability of sampling experience i:
      p_i ∝ priority_recent × recency_weight(i)
           + priority_reward × |reward_i|

    With mode balancing: each mode contributes proportionally.
    """
    if len(experiences) <= batch_size:
        return experiences

    # Compute sampling weights
    n = len(experiences)
    weights = []
    for idx, exp in enumerate(experiences):
        recency = (idx + 1) / n   # 0→1, newer = higher
        rew_mag = abs(exp.reward) if exp.reward is not None else 0.5
        w = priority_recent * recency + priority_reward * rew_mag
        weights.append(w)

    if not mode_balanced:
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(experiences, weights=probs, k=batch_size)

    # Mode-balanced: sample equally across modes
    by_mode = defaultdict(list)
    for i, exp in enumerate(experiences):
        by_mode[exp.mode].append((i, weights[i]))

    modes = list(by_mode.keys())
    per_mode = max(1, batch_size // len(modes))
    sampled_indices = set()
    for mode in modes:
        bucket = by_mode[mode]
        bucket_weights = [w for _, w in bucket]
        total_w = sum(bucket_weights)
        bucket_probs = [w / total_w for w in bucket_weights]
        k = min(per_mode, len(bucket))
        chosen = random.choices(
            [i for i, _ in bucket],
            weights=bucket_probs,
            k=k,
        )
        sampled_indices.update(chosen)

    # Fill remaining slots from full pool
    remaining = batch_size - len(sampled_indices)
    if remaining > 0:
        pool = [(i, weights[i]) for i in range(n) if i not in sampled_indices]
        if pool:
            pw = [w for _, w in pool]
            extra = random.choices([i for i, _ in pool], weights=pw, k=min(remaining, len(pool)))
            sampled_indices.update(extra)

    return [experiences[i] for i in sorted(sampled_indices)]


def memory_stats(experiences: List[TradeExperience]) -> dict:
    """Summary statistics for the replay buffer."""
    if not experiences:
        return {}
    rewards = [e.reward for e in experiences if e.reward is not None]
    pnls    = [e.pnl_pct for e in experiences]
    by_mode = defaultdict(list)
    for e in experiences:
        by_mode[e.mode].append(e.pnl_pct)
    return {
        "total": len(experiences),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "avg_pnl": sum(pnls) / len(pnls),
        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
        "by_mode": {m: {
            "n": len(v),
            "avg_pnl": sum(v) / len(v),
        } for m, v in by_mode.items()},
    }
