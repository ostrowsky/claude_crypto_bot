"""Trading policy -- the ONE decision function shared by the E1 replay and the phase-C executor.

execution-replay-e1-1010-spec.md §2. A policy = admission x sizing x exits x risk limits,
frozen and hashed (policy_epoch). The replay and the live executor differ only in where bars
come from and who fills orders (fill_model vs exchange_client); decide_entry / exit_plan are
the same code.

Exit rules in v1 are path functions of the bars after entry (real logged exit, P-1 plain
trail, OCO TP/SL in ATR). For those, computing the exit at admission time over the forward
bars is identical to deciding bar by bar -- the replay uses that; the live executor will call
the same rule bar by bar. Rules that depend on portfolio state (rotation) are out of v1.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Admission:
    kind: str = "real_entries"          # real_entries | all_candidates
    min_score: float | None = None      # admit only if candidate[score_key] >= min_score
    score_key: str = "ml_proba"
    max_per_day: int | None = None      # first N admitted per UTC day (real time, no look-ahead)
    modes: tuple | None = None          # restrict to these entry modes
    random_frac: float | None = None    # reference control (h2 spec §8): admit if hash(seed, sym, ts) < frac
    random_seed: int = 0
    regime: str | None = None           # x4-family-money-1015-spec.md §3.1: R1..R4, no new entries in a weak market


@dataclass(frozen=True)
class Sizing:
    kind: str = "equal_slots"           # equal_slots | risk
    risk_pct: float = 0.5               # for "risk": % of equity lost at the stop
    max_position_pct: float = 10.0
    cash_reserve_pct: float = 0.0


@dataclass(frozen=True)
class Exits:
    kind: str = "real"                  # real | p1_trail | oco_atr
    tp_atr: float = 2.0                 # oco_atr
    sl_atr: float = 1.5                 # oco_atr
    max_bars: int = 96                  # p1_trail / oco_atr horizon (time exit at the close)
    floor_override: float | None = None # p1_trail: minimum stop buffer for every mode (h3 X4: 0.08)
    leader: tuple | None = None         # leader-mode params as sorted (key, value) pairs (h3 X2/X3)
    rsi_min_profit: float = 0.005       # rsi_hold: profit at the signal bar needed to hold (fixed, all modes)
    hold_after_bars: int = 96           # rsi_hold: latest exit = signal bar + this


@dataclass(frozen=True)
class Risk:
    max_open: int = 10
    daily_loss_limit_pct: float | None = None     # no new entries for the rest of the UTC day
    cooldown_bars: int = 0                        # after an exit of the same symbol (all_candidates)
    one_per_symbol: bool = False                  # no second position in a symbol already held (h2 spec §7.1)


@dataclass(frozen=True)
class TradingPolicy:
    name: str = "champion"
    admission: Admission = field(default_factory=Admission)
    sizing: Sizing = field(default_factory=Sizing)
    exits: Exits = field(default_factory=Exits)
    risk: Risk = field(default_factory=Risk)

    def epoch(self) -> str:
        d = asdict(self)
        if d["admission"].get("regime") is None:   # field added 2026-10-15: running policies keep their epoch
            d["admission"].pop("regime", None)
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


TP_TICK = 1e-4
CHAMPION = TradingPolicy()        # every bot entry, its own logged exit, 10 equal slots


def decide_entry_explained(policy: TradingPolicy, cand: dict, state: dict):
    """-> (fraction of equity or None, reason). reason is 'enter' or one code of
    trader-audit-log-1012-spec.md §3. state: open, open_syms, day_count, day_return_pct,
    cooldown_until (bar index per sym), equity, bar."""
    A, R, S = policy.admission, policy.risk, policy.sizing
    if A.kind == "real_entries" and cand.get("kind") != "entry":
        return None, "not_bot_entry"
    if A.modes and cand.get("mode") not in A.modes:
        return None, "mode_filtered"
    if A.min_score is not None:
        v = cand.get(A.score_key)
        if not isinstance(v, (int, float)):
            return None, "score_missing"
        if v < A.min_score:
            return None, "score_below_min"
    if A.random_frac is not None:
        h = hashlib.sha256(f"{A.random_seed}|{cand.get('sym')}|{cand.get('ts')}".encode()).digest()
        if int.from_bytes(h[:8], "big") / 2 ** 64 >= A.random_frac:
            return None, "random_excluded"
    if A.regime is not None:
        import regime as RG
        if RG.blocks(A.regime, state.get("regime")):
            return None, "regime_block"
    if A.max_per_day is not None and state.get("day_count", 0) >= A.max_per_day:
        return None, "daily_cap"
    if state.get("open", 0) >= R.max_open:
        return None, "slots_full"
    if R.one_per_symbol and cand.get("sym") in state.get("open_syms", ()):
        return None, "symbol_held"
    if R.daily_loss_limit_pct is not None and state.get("day_return_pct", 0.0) <= -abs(R.daily_loss_limit_pct):
        return None, "daily_loss_limit"
    if R.cooldown_bars and state.get("cooldown_until", {}).get(cand.get("sym"), -1) > state.get("bar", 0):
        return None, "cooldown"
    if S.kind == "equal_slots":
        frac = 1.0 / R.max_open
    else:
        stop_dist = cand.get("stop_dist")              # fraction of price, from the exit plan
        if not stop_dist or stop_dist <= 0:
            return None, "no_stop_distance"
        frac = (S.risk_pct / 100.0) / stop_dist
    frac = min(frac, S.max_position_pct / 100.0)
    return (frac, "enter") if frac > 0 else (None, "zero_size")


def decide_entry(policy: TradingPolicy, cand: dict, state: dict):
    """-> fraction of equity to commit, or None (behaviour unchanged; reasons via decide_entry_explained)."""
    return decide_entry_explained(policy, cand, state)[0]


def exit_plan(policy: TradingPolicy, grid, cand: dict, floor_of=None):
    """(exit_bar, exit_price_before_costs, reason, stop_dist_at_entry) from the bars after the entry bar,
    or None when the path cannot be evaluated. Uses only bars > cand['ie'] (no look-ahead into the
    entry decision: the entry bar itself is the last closed bar before the signal)."""
    E = policy.exits
    j, ie = cand.get("j"), cand.get("ie")
    if E.kind == "real":
        if cand.get("exit_price") is None or cand.get("ix") is None:
            return None
        return cand["ix"], float(cand["exit_price"]), "real", None
    if E.kind in ("rsi_hold", "leader", "leader_rsi_hold"):
        if cand.get("exit_price") is None or cand.get("ix") is None:
            return None
        real = (cand["ix"], float(cand["exit_price"]), "real", None)
        if cand.get("tf") != "15m" or j is None or ie is None:
            return real                           # h3 spec §2: new exits act on 15m trades only
        if E.kind in ("leader", "leader_rsi_hold") and E.leader:
            r = _leader_exit(grid, cand, dict(E.leader))
            if r is not None:
                return r
        if E.kind in ("rsi_hold", "leader_rsi_hold"):
            r = _rsi_hold(grid, cand, floor_of(cand.get("mode")) if floor_of else 0.0, E)
            if r is not None:
                return r
        return real
    if j is None or ie is None:
        return None
    C, H, L, O, A = grid.C, grid.H, grid.L, grid.O, grid.ATR
    c0, a0 = C[ie, j], A[ie, j]
    if not (math.isfinite(c0) and math.isfinite(a0)):
        return None
    last = min(grid.N - 1, ie + E.max_bars)
    hourly = cand.get("tf") == "1h"
    if E.kind == "p1_trail":
        k = float(cand.get("trail_k") or 2.0)
        floor = E.floor_override if E.floor_override is not None else (floor_of(cand.get("mode")) if floor_of else 0.0)
        stop = c0 - max(k * a0, floor * c0)
        dist = (c0 - stop) / c0
        for q in range(ie + 1, last + 1):
            cq = C[q, j]
            if not math.isfinite(cq):
                continue
            if hourly and (q + 1) % 4:
                continue                       # 1h positions are judged on hour closes
            if math.isfinite(A[q, j]):
                stop = max(stop, cq - max(k * A[q, j], floor * cq))
            if cq < stop:
                return q, float(cq), "trail", dist
        return (last, float(C[last, j]), "time", dist) if math.isfinite(C[last, j]) else None
    if E.kind == "oco_atr":
        tp, sl = c0 + E.tp_atr * a0, c0 - E.sl_atr * a0
        dist = (c0 - sl) / c0
        for q in range(ie + 1, last + 1):
            lo, hi, op = L[q, j], H[q, j], O[q, j]
            if not (math.isfinite(lo) and math.isfinite(hi)):
                continue
            if lo <= sl:                          # SL first when both are touched (pessimistic)
                return q, float(min(sl, op if math.isfinite(op) else sl)), "sl", dist
            if hi >= tp * (1 + TP_TICK):          # trade-through, no queue optimism
                return q, float(tp), "tp", dist
        return (last, float(C[last, j]), "time", dist) if math.isfinite(C[last, j]) else None
    raise ValueError(E.kind)


def _is_rsi_divergence(reason) -> bool:
    r = str(reason or "").lower()
    return "rsi" in r and ("дивергенц" in r or "diverg" in r)


def _rsi_hold(grid, cand, floor: float, E: Exits):
    """X1 (weak-exits-1005-spec.md): a WEAK RSI-divergence exit with profit >= E.rsi_min_profit at the
    signal bar is replaced by holding on the P-1 trail ratcheted from the entry bar; exits only at or
    after the signal bar, at most E.hold_after_bars later. Mirrors _backtest_weak_exits.hold_on_trail."""
    if not _is_rsi_divergence(cand.get("exit_reason")):
        return None
    j, ie, ix, ep = cand["j"], cand["ie"], cand["ix"], float(cand["price"])
    C, A = grid.C, grid.ATR
    cx = C[ix, j]
    if not math.isfinite(cx) or cx / ep - 1 < E.rsi_min_profit:
        return None
    k = float(cand.get("trail_k") or 2.0)
    c0 = C[ie, j]
    if not math.isfinite(c0):
        return None
    stop = c0 - max(k * A[ie, j] if math.isfinite(A[ie, j]) else 0.0, floor * c0)
    last = min(grid.N - 1, ix + E.hold_after_bars)
    for q in range(ie + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], floor * cq))
        if q >= ix and cq < stop:
            return q, float(cq), "rsi_hold", None
    q = last
    while q > ix and not math.isfinite(C[q, j]):
        q -= 1
    return (q, float(C[q, j]), "rsi_hold", None) if math.isfinite(C[q, j]) else None


def _leader_exit(grid, cand, P: dict):
    """X2 (leader-exit-x6-spec.md, X-9c): switch on the first bar in [ie, ix) where the coin ranks
    <= rank_max of the day with day return >= min_ret; then a wide trail (floor P['floor']) and an exit
    on losing rank > lost_rank after lost_min_bars. Mirrors exit_validator.leader_arm / wide_from."""
    j, ie, ix = cand["j"], cand["ie"], cand["ix"]
    RANK, RET, C, A = grid.RANK, grid.RET, grid.C, grid.ATR
    s = None
    for q in range(ie, ix):
        rt = RET[q, j]
        if RANK[q, j] <= P["rank_max"] and math.isfinite(rt) and rt >= P["min_ret"]:
            s = q
            break
    if s is None:
        return None
    k = float(cand.get("trail_k") or 2.0)
    cs = C[s, j]
    stop = cs - max(k * A[s, j] if math.isfinite(A[s, j]) else 0.0, P["floor"] * cs)
    last = min(grid.N - 1, s + P["max_hold_bars"])
    for q in range(s + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], P["floor"] * cq))
        if cq < stop or (q - s >= P["lost_min_bars"] and RANK[q, j] > P["lost_rank"]):
            return q, float(cq), "leader", None
    q = last
    while q > s and not math.isfinite(C[q, j]):
        q -= 1
    return (q, float(C[q, j]), "leader", None) if math.isfinite(C[q, j]) else None
