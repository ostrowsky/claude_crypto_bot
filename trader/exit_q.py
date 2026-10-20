"""F2 -- a learned exit as an optimal-stopping problem (trader-learning-loop-f-1020-spec.md §4, §6.4).

Fitted value iteration in the Longstaff-Schwartz form: for each bar t of a position the continuation value
C(s_t) = E[V_{t+1} / v_t | s_t] is regressed on the state, using the realised value of the position under the
current policy; the policy is improved and the regression refitted (n_iter rounds).

Exit rule (conservative, the CQL-like correction of the spec): exit at the close of bar t only if
    (1 - cost) - C(s_t) > delta + Z * se(s_t)
where se is the standard error of the leaf, clustered by position. The regression tree is HONEST
(Athey & Imbens): splits are chosen on half of the positions, leaf means and errors are estimated on the
other half, so a split found in noise does not also certify itself.

Bounds: the agent can only exit EARLIER than the X4 book -- never later than the X4 exit bar, never after
max_bars; the exchange catastrophe stop stays in place regardless.
"""
from __future__ import annotations

import math

import numpy as np

Z = 2.0
N_STATE = 7


def state(close, t: int) -> np.ndarray:
    """Features of bar t from closes up to t only: unrealised, MFE, drawdown from MFE, bars held,
    bars since the peak, last bar return, volatility of the last 16 bar returns."""
    c = np.asarray(close[: t + 1], dtype=float)
    e, last = c[0], c[-1]
    peak_i = int(np.argmax(c))
    peak = c[peak_i]
    r1 = last / c[-2] - 1 if t >= 1 else 0.0
    lr = np.diff(np.log(c[-17:])) if t >= 2 else np.zeros(1)
    return np.array([last / e - 1, peak / e - 1, last / peak - 1, float(t), float(t - peak_i), r1,
                     float(lr.std()) if lr.size > 1 else 0.0])


class TreeRegressor:
    """Small deterministic honest regression tree (numpy). fit(X, y, groups) -> predict / predict_se."""

    def __init__(self, max_depth: int = 4, min_leaf: int = 20, n_thresholds: int = 32, honest: bool = True):
        self.max_depth, self.min_leaf, self.n_thr, self.honest = max_depth, min_leaf, n_thresholds, honest

    def _split(self, X, y, idx, depth):
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return None
        best = None
        yy = y[idx]
        base = ((yy - yy.mean()) ** 2).sum()
        for f in range(X.shape[1]):
            x = X[idx, f]
            qs = np.unique(np.quantile(x, np.linspace(0, 1, self.n_thr + 2)[1:-1]))
            for q in qs:
                left = x <= q
                nl = int(left.sum())
                if nl < self.min_leaf or len(idx) - nl < self.min_leaf:
                    continue
                yl, yr = yy[left], yy[~left]
                sse = ((yl - yl.mean()) ** 2).sum() + ((yr - yr.mean()) ** 2).sum()
                if best is None or sse < best[0] - 1e-15:
                    best = (sse, f, float(q))
        if best is None or best[0] >= base - 1e-15:
            return None
        return best[1], best[2]

    def _grow(self, X, y, idx, depth):
        s = self._split(X, y, idx, depth)
        if s is None:
            self._leaves.append(None)
            return {"leaf": len(self._leaves) - 1}
        f, q = s
        left = idx[X[idx, f] <= q]
        right = idx[X[idx, f] > q]
        return {"f": f, "q": q, "l": self._grow(X, y, left, depth + 1), "r": self._grow(X, y, right, depth + 1)}

    def _leaf_of(self, x):
        n = self.tree
        while "leaf" not in n:
            n = n["l"] if x[n["f"]] <= n["q"] else n["r"]
        return n["leaf"]

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        g = np.arange(len(y)) if groups is None else np.asarray(groups)
        if self.honest:
            gi = np.unique(g, return_inverse=True)[1]
            build, est = gi % 2 == 0, gi % 2 == 1
            if not build.any() or not est.any():
                build = est = np.ones(len(y), dtype=bool)
        else:
            build = est = np.ones(len(y), dtype=bool)
        self._leaves = []
        self.tree = self._grow(X, y, np.flatnonzero(build), 0)
        leaf = self.leaves(X[est]) if est.any() else np.empty(0, dtype=np.int64)
        ye, ge = y[est], g[est]
        stats = []
        for k in range(len(self._leaves)):
            m = leaf == k
            if not m.any():
                stats.append((float(y[build].mean()), math.inf))
                continue
            # cluster by position: one mean per group, then the error of the mean of the group means
            gm = {}
            for gg, v in zip(ge[m], ye[m]):
                gm.setdefault(gg, []).append(v)
            means = np.array([np.mean(v) for v in gm.values()])
            se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) >= 2 else math.inf
            stats.append((float(ye[m].mean()), se))
        self.stats = stats
        return self

    def leaves(self, X):
        """Leaf id of every row, vectorised."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty(len(X), dtype=np.int64)
        stack = [(self.tree, np.arange(len(X)))]
        while stack:
            n, idx = stack.pop()
            if "leaf" in n:
                out[idx] = n["leaf"]
                continue
            m = X[idx, n["f"]] <= n["q"]
            stack.append((n["l"], idx[m]))
            stack.append((n["r"], idx[~m]))
        return out

    def predict(self, X):
        mean = np.array([m for m, _ in self.stats])
        return mean[self.leaves(X)]

    def predict_se(self, X):
        se = np.array([v for _, v in self.stats])
        return se[self.leaves(X)]


def default_model():
    return TreeRegressor(max_depth=6, min_leaf=200)


class ExitPolicy:
    def __init__(self, model_factory=None, delta: float = 0.001, max_bars: int = 96, n_iter: int = 3, z: float = Z):
        self.factory = model_factory or default_model
        self.delta, self.max_bars, self.n_iter, self.z = delta, max_bars, n_iter, z
        self.model = None

    def _horizon(self, p) -> int:
        return max(1, min(int(p["x4_exit"]), self.max_bars, len(p["close"]) - 1))

    def _exits_at(self, p, t, s=None) -> bool:
        if self.model is None:
            return False
        s = state(p["close"], t) if s is None else s
        c = float(self.model.predict(s[None, :])[0])
        se = float(self.model.predict_se(s[None, :])[0])
        return (1 - p["cost"]) - c > self.delta + self.z * se

    def _states(self, paths):
        """All decision states t = 1..T-1 of all paths, stacked: (X, path index, t)."""
        X, P, Tt = [], [], []
        for i, p in enumerate(paths):
            for t in range(1, self._horizon(p)):
                X.append(state(p["close"], t))
                P.append(i)
                Tt.append(t)
        if not X:
            return np.zeros((0, N_STATE)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.array(X), np.array(P, dtype=np.int64), np.array(Tt, dtype=np.int64)

    def _exit_mask(self, X, cost):
        if self.model is None or not len(X):
            return np.zeros(len(X), dtype=bool)
        return (1 - cost) - self.model.predict(X) > self.delta + self.z * self.model.predict_se(X)

    def fit(self, paths):
        X, P, Tt = self._states(paths)
        cost = np.array([p["cost"] for p in paths])[P] if len(P) else np.zeros(0)
        for _ in range(max(1, self.n_iter)):
            ex = self._exit_mask(X, cost)
            Y = np.empty(len(X))
            k = 0
            for i, p in enumerate(paths):                            # value under the current policy, backward
                close, T = np.asarray(p["close"], dtype=float), self._horizon(p)
                n = T - 1
                if n <= 0:
                    continue
                e0, c = close[0], p["cost"]
                V = close[T] / e0 * (1 - c)                          # forced exit at the X4 bar / horizon
                for t in range(T - 1, 0, -1):
                    r = k + t - 1                                    # row of (path i, bar t)
                    Y[r] = V / (close[t] / e0)                       # continuation ratio seen at bar t
                    if ex[r]:
                        V = close[t] / e0 * (1 - c)
                k += n
            if not len(X):
                break
            self.model = self.factory().fit(X, Y, groups=P)
        return self

    def exit_bars(self, paths) -> list:
        X, P, Tt = self._states(paths)
        cost = np.array([p["cost"] for p in paths])[P] if len(P) else np.zeros(0)
        ex = self._exit_mask(X, cost)
        out = [self._horizon(p) for p in paths]
        for r in np.flatnonzero(ex):                                 # rows are in (path, t) order: first hit wins
            i = int(P[r])
            if Tt[r] < out[i]:
                out[i] = int(Tt[r])
        return out

    def exit_bar(self, p) -> int:
        T = self._horizon(p)
        for t in range(1, T):
            if self._exits_at(p, t):
                return t
        return T
