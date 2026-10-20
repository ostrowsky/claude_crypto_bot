"""Overfitting controls for the trials registry (trader-learning-loop-f-1020-spec.md §2, §6.5).

cscv_pbo        probability of backtest overfitting by combinatorially symmetric cross-validation
                (Bailey, Borwein, Lopez de Prado, Zhu 2015): share of IS/OOS splits where the IS-best strategy
                ranks at or below the OOS median.
deflated_sharpe probability that the true Sharpe exceeds the expected maximum of n_trials null Sharpes
                (Bailey & Lopez de Prado 2014). Sharpe per observation, not annualised.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _sharpe(x):
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    return x.mean(axis=0) / sd


def cscv_pbo(R, n_splits: int = 8) -> float:
    """R: T x N matrix of per-period returns of N strategies."""
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    if n_splits % 2 or n_splits < 2 or N < 2:
        raise ValueError("n_splits must be even and >= 2, N >= 2")
    blocks = np.array_split(np.arange(T), n_splits)
    lam = []
    for is_ids in itertools.combinations(range(n_splits), n_splits // 2):
        oos_ids = [i for i in range(n_splits) if i not in is_ids]
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in oos_ids])
        best = int(np.argmax(_sharpe(R[is_rows])))
        oos = _sharpe(R[oos_rows])
        rank = 1 + int(np.sum(oos < oos[best])) + 0.5 * int(np.sum(oos == oos[best]) - 1)   # 1..N
        w = rank / (N + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def deflated_sharpe(sr: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                    sr_var: float | None = None) -> float:
    """sr per observation; sr_var = variance of the trials' Sharpes (default 1 / n_obs, the null)."""
    if n_trials < 1 or n_obs < 2:
        raise ValueError("n_trials >= 1, n_obs >= 2")
    v = sr_var if sr_var is not None else 1.0 / n_obs
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                              + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / den))
