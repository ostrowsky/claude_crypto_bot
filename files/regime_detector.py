from __future__ import annotations

"""
Market Regime Detector — HMM-based classification with 6 states.

Replaces binary bull/nonbull with nuanced regime detection:
  1. strong_bull    — BTC+alts trending up, high ADX, expanding OI
  2. weak_bull      — Grinding up, low conviction, selective entries only
  3. ranging        — Sideways, mean-reversion works, trend signals fail
  4. volatile_chop  — High ATR but no direction, worst for trend-following
  5. weak_bear      — Gradual decline, short-term bounces possible
  6. strong_bear    — Capitulation, longs get destroyed, cash is king

Each regime maps to:
  - Entry mode whitelist (which signal types to allow)
  - Parameter overrides (RSI bounds, vol×, ATR multipliers)
  - Position sizing multiplier
  - Max open positions
  - Signal score adjustments

Architecture:
  - Gaussian HMM trained offline on BTC 4h data (features: returns, vol, ADX, correlation)
  - Online Viterbi decoding updates regime every 15 minutes
  - Coin-level micro-regime overlaid on top of market regime
  - Transition probabilities inform regime change predictions

Usage:
    detector = RegimeDetector()
    detector.load_model("regime_model.json")     # pre-trained HMM
    regime = await detector.detect(btc_data, alt_data)
    params = regime.get_params()
    allowed_modes = regime.allowed_entry_modes
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Regime definitions ───────────────────────────────────────────────────────

REGIME_NAMES = [
    "strong_bull",
    "weak_bull",
    "ranging",
    "volatile_chop",
    "weak_bear",
    "strong_bear",
]

REGIME_PARAMS: Dict[str, Dict[str, Any]] = {
    "strong_bull": {
        "allowed_modes": ("trend", "strong_trend", "impulse_speed", "impulse", "breakout", "retest", "alignment"),
        "rsi_hi": 80.0,
        "rsi_lo": 40.0,
        "vol_mult": 1.10,
        "slope_min": 0.08,
        "adx_min": 18.0,
        "range_max": 10.0,
        "atr_trail_k_mult": 1.0,       # standard trailing
        "max_positions": 8,
        "position_size_mult": 1.2,
        "score_bonus": 5.0,
    },
    "weak_bull": {
        "allowed_modes": ("trend", "strong_trend", "impulse_speed", "breakout", "retest"),
        "rsi_hi": 72.0,
        "rsi_lo": 45.0,
        "vol_mult": 1.30,
        "slope_min": 0.10,
        "adx_min": 20.0,
        "range_max": 7.0,
        "atr_trail_k_mult": 0.9,       # tighter stops
        "max_positions": 6,
        "position_size_mult": 1.0,
        "score_bonus": 0.0,
    },
    "ranging": {
        "allowed_modes": ("breakout", "retest"),
        "rsi_hi": 65.0,
        "rsi_lo": 35.0,
        "vol_mult": 1.80,              # need strong vol to break range
        "slope_min": 0.15,
        "adx_min": 25.0,               # only strong trends
        "range_max": 5.0,
        "atr_trail_k_mult": 0.8,       # tight stops
        "max_positions": 3,
        "position_size_mult": 0.7,
        "score_bonus": -5.0,
    },
    "volatile_chop": {
        "allowed_modes": ("breakout",),  # only high-conviction breakouts
        "rsi_hi": 60.0,
        "rsi_lo": 40.0,
        "vol_mult": 2.50,              # very high vol required
        "slope_min": 0.20,
        "adx_min": 30.0,
        "range_max": 4.0,
        "atr_trail_k_mult": 0.7,       # very tight
        "max_positions": 2,
        "position_size_mult": 0.5,
        "score_bonus": -10.0,
    },
    "weak_bear": {
        "allowed_modes": ("breakout", "retest", "impulse_speed"),
        "rsi_hi": 65.0,
        "rsi_lo": 35.0,
        "vol_mult": 1.80,
        "slope_min": 0.15,
        "adx_min": 22.0,
        "range_max": 5.0,
        "atr_trail_k_mult": 0.75,
        "max_positions": 3,
        "position_size_mult": 0.6,
        "score_bonus": -8.0,
    },
    "strong_bear": {
        "allowed_modes": (),            # no longs in strong bear
        "rsi_hi": 55.0,
        "rsi_lo": 30.0,
        "vol_mult": 3.00,
        "slope_min": 0.30,
        "adx_min": 35.0,
        "range_max": 3.0,
        "atr_trail_k_mult": 0.5,
        "max_positions": 0,
        "position_size_mult": 0.0,
        "score_bonus": -20.0,
    },
}


@dataclass
class RegimeState:
    """Current market regime with metadata."""
    name: str
    probability: float              # Confidence in this regime [0, 1]
    probabilities: Dict[str, float] # All regime probabilities
    timestamp_ms: int
    features_used: Dict[str, float] = field(default_factory=dict)

    # Transition prediction
    transition_probs: Dict[str, float] = field(default_factory=dict)
    most_likely_next: str = ""
    regime_age_bars: int = 0        # How long we've been in this regime

    @property
    def params(self) -> Dict[str, Any]:
        return REGIME_PARAMS.get(self.name, REGIME_PARAMS["weak_bull"])

    @property
    def allowed_entry_modes(self) -> Tuple[str, ...]:
        return self.params.get("allowed_modes", ())

    @property
    def score_bonus(self) -> float:
        return self.params.get("score_bonus", 0.0)

    @property
    def max_positions(self) -> int:
        return self.params.get("max_positions", 6)

    @property
    def position_size_mult(self) -> float:
        return self.params.get("position_size_mult", 1.0)

    @property
    def is_risk_off(self) -> bool:
        return self.name in ("strong_bear", "volatile_chop")

    def to_features(self) -> Dict[str, float]:
        """Flat features for ML models."""
        f = {}
        for rname in REGIME_NAMES:
            f[f"regime_{rname}"] = self.probabilities.get(rname, 0.0)
        f["regime_score_bonus"] = self.score_bonus
        f["regime_max_positions"] = float(self.max_positions)
        f["regime_age_bars"] = float(self.regime_age_bars)
        f["regime_is_risk_off"] = 1.0 if self.is_risk_off else 0.0
        return f

    @property
    def icon(self) -> str:
        icons = {
            "strong_bull": "🟢🐂",
            "weak_bull": "🟡🐂",
            "ranging": "↔️",
            "volatile_chop": "🌀",
            "weak_bear": "🟡🐻",
            "strong_bear": "🔴🐻",
        }
        return icons.get(self.name, "?")

    def __str__(self) -> str:
        return f"{self.icon} {self.name} ({self.probability:.0%})"


class GaussianHMM:
    """
    Minimal Gaussian Hidden Markov Model implementation.

    No external dependency (hmmlearn). Pure numpy.
    Supports: fit(), predict(), score().
    """

    def __init__(self, n_states: int = 6, n_features: int = 6):
        self.n_states = n_states
        self.n_features = n_features

        # Model parameters (initialized, trained offline)
        self.pi = np.ones(n_states) / n_states             # Initial state probs
        self.A = np.ones((n_states, n_states)) / n_states   # Transition matrix
        self.means = np.zeros((n_states, n_features))       # Emission means
        self.covars = np.array([np.eye(n_features) for _ in range(n_states)])  # Emission covars
        self._fitted = False

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """Log P(x | state) for all states. Shape: (T, n_states)."""
        T = x.shape[0]
        log_probs = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            diff = x - self.means[s]
            cov = self.covars[s]
            try:
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                solved = np.linalg.solve(L, diff.T).T
                mahal = np.sum(solved ** 2, axis=1)
            except np.linalg.LinAlgError:
                # Fallback: diagonal
                diag = np.diag(cov).clip(1e-6)
                log_det = np.sum(np.log(diag))
                mahal = np.sum(diff ** 2 / diag, axis=1)
            log_probs[:, s] = -0.5 * (
                self.n_features * np.log(2 * np.pi) + log_det + mahal
            )
        return log_probs

    def viterbi(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Viterbi decoding. Returns (state_sequence, log_probability)."""
        T = x.shape[0]
        log_em = self._log_emission(x)
        log_A = np.log(self.A.clip(1e-10))
        log_pi = np.log(self.pi.clip(1e-10))

        # Forward pass
        V = np.zeros((T, self.n_states))
        ptr = np.zeros((T, self.n_states), dtype=int)
        V[0] = log_pi + log_em[0]

        for t in range(1, T):
            for s in range(self.n_states):
                trans_prob = V[t - 1] + log_A[:, s]
                ptr[t, s] = int(np.argmax(trans_prob))
                V[t, s] = trans_prob[ptr[t, s]] + log_em[t, s]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]

        return states, float(V[-1, states[-1]])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Forward algorithm to get state probabilities. Shape: (T, n_states)."""
        T = x.shape[0]
        log_em = self._log_emission(x)
        log_A = np.log(self.A.clip(1e-10))
        log_pi = np.log(self.pi.clip(1e-10))

        # Forward (log-space)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = log_pi + log_em[0]

        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t, s] = _logsumexp(alpha[t - 1] + log_A[:, s]) + log_em[t, s]

        # Normalize to probabilities
        probs = np.zeros((T, self.n_states))
        for t in range(T):
            log_norm = _logsumexp(alpha[t])
            probs[t] = np.exp(alpha[t] - log_norm)

        return probs

    def fit(self, X: np.ndarray, n_iter: int = 50) -> None:
        """Baum-Welch EM training on observation sequence X (T, n_features)."""
        T, D = X.shape
        assert D == self.n_features

        for iteration in range(n_iter):
            # E-step: forward-backward
            log_em = self._log_emission(X)
            log_A = np.log(self.A.clip(1e-10))
            log_pi = np.log(self.pi.clip(1e-10))

            # Forward
            alpha = np.zeros((T, self.n_states))
            alpha[0] = log_pi + log_em[0]
            for t in range(1, T):
                for s in range(self.n_states):
                    alpha[t, s] = _logsumexp(alpha[t - 1] + log_A[:, s]) + log_em[t, s]

            # Backward
            beta = np.zeros((T, self.n_states))
            for t in range(T - 2, -1, -1):
                for s in range(self.n_states):
                    beta[t, s] = _logsumexp(log_A[s, :] + log_em[t + 1, :] + beta[t + 1, :])

            # Posteriors
            gamma = alpha + beta
            for t in range(T):
                gamma[t] -= _logsumexp(gamma[t])
            gamma = np.exp(gamma)

            # M-step
            self.pi = gamma[0] / gamma[0].sum()

            for s in range(self.n_states):
                w = gamma[:, s]
                w_sum = w.sum()
                if w_sum < 1e-10:
                    continue
                self.means[s] = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.means[s]
                self.covars[s] = (diff * w[:, None]).T @ diff / w_sum
                # Regularize
                self.covars[s] += np.eye(D) * 1e-3

            # Transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    num = 0.0
                    for t in range(T - 1):
                        num += np.exp(
                            alpha[t, i] + log_A[i, j] + log_em[t + 1, j] + beta[t + 1, j]
                            - _logsumexp(alpha[-1])
                        )
                    self.A[i, j] = num
                row_sum = self.A[i].sum()
                if row_sum > 0:
                    self.A[i] /= row_sum

        self._fitted = True

    def save(self, path: str) -> None:
        """Save model to JSON."""
        payload = {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "means": self.means.tolist(),
            "covars": [c.tolist() for c in self.covars],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load(self, path: str) -> None:
        """Load model from JSON."""
        data = json.loads(Path(path).read_text())
        self.n_states = data["n_states"]
        self.n_features = data["n_features"]
        self.pi = np.array(data["pi"])
        self.A = np.array(data["A"])
        self.means = np.array(data["means"])
        self.covars = np.array(data["covars"])
        self._fitted = True


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    m = float(np.max(x))
    if not np.isfinite(m):
        return float("-inf")
    return m + float(np.log(np.sum(np.exp(x - m))))


class RegimeDetector:
    """
    Detects market regime using HMM + rule-based fallback.

    Features used:
      1. BTC 4h returns (%)
      2. BTC 4h volatility (rolling std of returns)
      3. BTC ADX
      4. BTC RSI
      5. Alt-BTC correlation (rolling 20-bar)
      6. Volume regime (current vol / 20-bar avg)

    If HMM model not available, falls back to rule-based classification
    using BTC indicators (improved version of existing MarketRegime).
    """

    def __init__(self, model_path: Optional[str] = None):
        self._hmm: Optional[GaussianHMM] = None
        self._model_path = model_path
        self._last_regime: Optional[RegimeState] = None
        self._regime_start_bar: int = 0
        self._bar_count: int = 0

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, path: str) -> bool:
        """Load pre-trained HMM model."""
        try:
            self._hmm = GaussianHMM()
            self._hmm.load(path)
            log.info("RegimeDetector: loaded HMM from %s", path)
            return True
        except Exception:
            log.warning("RegimeDetector: failed to load HMM, using rule-based")
            self._hmm = None
            return False

    def detect(
        self,
        btc_data: np.ndarray,
        btc_features: dict,
        alt_returns: Optional[np.ndarray] = None,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            btc_data: BTC OHLCV structured array (same format as klines)
            btc_features: BTC computed features dict (from indicators.compute_features)
            alt_returns: Optional array of alt returns for correlation calc

        Returns:
            RegimeState with regime name, probabilities, and parameters
        """
        self._bar_count += 1

        if self._hmm and self._hmm._fitted:
            return self._detect_hmm(btc_data, btc_features, alt_returns)
        return self._detect_rules(btc_data, btc_features)

    def _detect_hmm(
        self,
        btc_data: np.ndarray,
        btc_features: dict,
        alt_returns: Optional[np.ndarray],
    ) -> RegimeState:
        """HMM-based regime detection."""
        features = self._extract_features(btc_data, btc_features, alt_returns)
        if features is None or len(features) < 5:
            return self._detect_rules(btc_data, btc_features)

        probs = self._hmm.predict_proba(features)
        current_probs = probs[-1]

        # Map HMM states to regime names
        regime_probs = {}
        for i, name in enumerate(REGIME_NAMES[:self._hmm.n_states]):
            regime_probs[name] = float(current_probs[i])

        best_idx = int(np.argmax(current_probs))
        best_name = REGIME_NAMES[best_idx] if best_idx < len(REGIME_NAMES) else "weak_bull"
        best_prob = float(current_probs[best_idx])

        # Transition probabilities
        trans_probs = {}
        for j, name in enumerate(REGIME_NAMES[:self._hmm.n_states]):
            trans_probs[name] = float(self._hmm.A[best_idx, j])
        most_likely_next = max(trans_probs, key=trans_probs.get)

        # Regime age
        if self._last_regime and self._last_regime.name == best_name:
            age = self._bar_count - self._regime_start_bar
        else:
            self._regime_start_bar = self._bar_count
            age = 0

        state = RegimeState(
            name=best_name,
            probability=best_prob,
            probabilities=regime_probs,
            timestamp_ms=int(time.time() * 1000),
            transition_probs=trans_probs,
            most_likely_next=most_likely_next,
            regime_age_bars=age,
        )
        self._last_regime = state
        return state

    def _detect_rules(
        self,
        btc_data: np.ndarray,
        btc_features: dict,
    ) -> RegimeState:
        """
        Rule-based fallback regime detection (improved over binary bull/nonbull).

        Uses BTC price vs EMAs, ADX, RSI, volatility.
        """
        i = len(btc_data) - 1
        if i < 0:
            return self._make_state("weak_bull", 0.5)

        close = float(btc_data["c"][i])

        # Extract indicators
        ema20 = _safe_get(btc_features, "ema_fast", i, close)
        ema50 = _safe_get(btc_features, "ema_slow", i, close)
        ema200 = _safe_get(btc_features, "ema200", i, close)
        adx = _safe_get(btc_features, "adx", i, 25.0)
        rsi = _safe_get(btc_features, "rsi", i, 50.0)
        vol_x = _safe_get(btc_features, "vol_x", i, 1.0)
        slope = _safe_get(btc_features, "slope", i, 0.0)
        atr = _safe_get(btc_features, "atr", i, 0.0)

        # Volatility metric: ATR / close
        atr_pct = (atr / close * 100.0) if close > 0 else 1.0

        # ── Classification logic ─────────────────────────────────────────────
        above_ema20 = close > ema20
        above_ema50 = close > ema50
        above_ema200 = close > ema200
        ema_stacked_bull = ema20 > ema50 > ema200
        ema_stacked_bear = ema20 < ema50 < ema200
        strong_trend = adx > 28
        weak_trend = adx < 18

        # Score each regime
        scores: Dict[str, float] = {name: 0.0 for name in REGIME_NAMES}

        # Strong bull: everything aligned, strong trend
        if ema_stacked_bull and above_ema20 and strong_trend and rsi > 55:
            scores["strong_bull"] = 0.8 + (adx - 28) / 50.0 + (0.1 if slope > 0.15 else 0.0)

        # Weak bull: above key EMAs but not strong
        if above_ema50 and above_ema200 and not strong_trend:
            scores["weak_bull"] = 0.5 + (0.2 if above_ema20 else 0.0) + (0.1 if slope > 0 else 0.0)
        elif above_ema200 and above_ema20:
            scores["weak_bull"] = 0.4

        # Ranging: low ADX, price oscillating around EMAs
        if weak_trend and atr_pct < 2.0:
            scores["ranging"] = 0.5 + (18 - adx) / 20.0

        # Volatile chop: high ATR but low ADX
        if weak_trend and atr_pct > 2.5:
            scores["volatile_chop"] = 0.5 + atr_pct / 10.0

        # Weak bear: below EMA50, moderate trend down
        if not above_ema50 and above_ema200 and slope < 0:
            scores["weak_bear"] = 0.5 + abs(slope) * 2.0

        # Strong bear: below all EMAs, strong downtrend
        if ema_stacked_bear and not above_ema200 and rsi < 40:
            scores["strong_bear"] = 0.7 + (0.2 if strong_trend else 0.0)

        # Normalize to probabilities
        total = sum(scores.values())
        if total < 0.01:
            scores["weak_bull"] = 1.0
            total = 1.0

        probs = {name: s / total for name, s in scores.items()}
        best = max(probs, key=probs.get)

        return self._make_state(best, probs[best], probs)

    def _extract_features(
        self,
        btc_data: np.ndarray,
        btc_features: dict,
        alt_returns: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract HMM feature matrix from BTC data."""
        n = len(btc_data)
        if n < 20:
            return None

        close = btc_data["c"].astype(float)
        returns = np.diff(np.log(close.clip(1e-8))) * 100.0  # % log returns

        # Rolling volatility (20-bar)
        vol = np.full(n, np.nan)
        for i in range(20, n):
            vol[i] = float(np.std(returns[i - 20:i]))

        adx_arr = btc_features.get("adx", np.full(n, 25.0))
        rsi_arr = btc_features.get("rsi", np.full(n, 50.0))
        vol_x_arr = btc_features.get("vol_x", np.ones(n))

        # Alt correlation (if available)
        corr_arr = np.full(n, 0.8)
        if alt_returns is not None and len(alt_returns) == n - 1:
            for i in range(20, n):
                btc_ret_window = returns[i - 20:i]
                alt_ret_window = alt_returns[i - 20:i]
                if len(btc_ret_window) == len(alt_ret_window):
                    cc = np.corrcoef(btc_ret_window, alt_ret_window)
                    corr_arr[i] = float(cc[0, 1]) if np.isfinite(cc[0, 1]) else 0.8

        # Stack features (skip first 20 bars for warmup)
        start = 20
        T = n - start
        if T < 5:
            return None

        features = np.column_stack([
            np.pad(returns, (1, 0))[start:n],
            vol[start:n],
            adx_arr[start:n],
            rsi_arr[start:n],
            corr_arr[start:n],
            vol_x_arr[start:n],
        ])

        # Replace NaN with column means
        for col in range(features.shape[1]):
            mask = ~np.isfinite(features[:, col])
            if mask.any():
                col_mean = np.nanmean(features[:, col])
                features[mask, col] = col_mean if np.isfinite(col_mean) else 0.0

        return features

    def _make_state(
        self,
        name: str,
        prob: float,
        probs: Optional[Dict[str, float]] = None,
    ) -> RegimeState:
        """Create RegimeState with defaults."""
        if probs is None:
            probs = {n: (prob if n == name else (1 - prob) / max(1, len(REGIME_NAMES) - 1))
                     for n in REGIME_NAMES}

        if self._last_regime and self._last_regime.name == name:
            age = self._bar_count - self._regime_start_bar
        else:
            self._regime_start_bar = self._bar_count
            age = 0

        state = RegimeState(
            name=name,
            probability=prob,
            probabilities=probs,
            timestamp_ms=int(time.time() * 1000),
            regime_age_bars=age,
        )
        self._last_regime = state
        return state

    def train_offline(
        self,
        btc_data: np.ndarray,
        btc_features: dict,
        alt_returns: Optional[np.ndarray] = None,
        n_iter: int = 50,
        save_path: Optional[str] = None,
    ) -> bool:
        """
        Train HMM on historical BTC data.

        Call once with 6+ months of 4h BTC data for good regime separation.
        """
        features = self._extract_features(btc_data, btc_features, alt_returns)
        if features is None or len(features) < 100:
            log.error("Insufficient data for HMM training: need 100+ bars")
            return False

        self._hmm = GaussianHMM(
            n_states=len(REGIME_NAMES),
            n_features=features.shape[1],
        )
        self._hmm.fit(features, n_iter=n_iter)

        # Label states by characteristics (sort by mean return)
        mean_returns = self._hmm.means[:, 0]  # First feature = returns
        order = np.argsort(mean_returns)[::-1]  # Highest return first

        # Reorder model parameters
        self._hmm.pi = self._hmm.pi[order]
        self._hmm.A = self._hmm.A[order][:, order]
        self._hmm.means = self._hmm.means[order]
        self._hmm.covars = self._hmm.covars[order]

        if save_path:
            self._hmm.save(save_path)
            log.info("HMM trained and saved to %s", save_path)

        return True


def _safe_get(feat: dict, key: str, i: int, default: float) -> float:
    """Safely get feature value at index i."""
    arr = feat.get(key)
    if arr is not None and i < len(arr) and np.isfinite(arr[i]):
        return float(arr[i])
    return default
