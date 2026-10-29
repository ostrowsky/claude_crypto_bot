"""EX-RL1 phase 0 -- entry execution on the public aggTrades tape (execution-bandit-exrl1-1029-spec.md §3-§5).

tape = {"T": ms, "p": price, "q": qty, "m": isBuyerMaker, "a": aggregate id}; m True = the SELLER was the taker.

  a0_price   market buy: the first buyer-taker trade at or after t + lag
  limit_ref  the resting bid proxy: the last seller-taker price at or before t + lag
  a1_price   limit at limit_ref, filled only if a trade prints STRICTLY below it in (t+lag, t+lag+timeout],
             else a market buy after the timeout; no reference -> same as A0
  a2_fill    the same limit, no market fallback: (filled, price)
  pair_diff_bp, calibration, fetch_tape (GET /api/v3/aggTrades, <= 1 h per request, deduped by id), store
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

TAPE = TC.RUNTIME / "tape"
HOUR = 3_600_000
CALIB_MAX_BP = 15.0


def _arr(tape):
    return (np.asarray(tape["T"], dtype=np.int64), np.asarray(tape["p"], float), np.asarray(tape["m"], bool))


def _first_buy_at_or_after(T, p, m, t):
    i = np.flatnonzero((T >= t) & ~m)
    return float(p[i[0]]) if len(i) else None


def a0_price(tape, t, lag):
    T, p, m = _arr(tape)
    return _first_buy_at_or_after(T, p, m, t + lag)


def limit_ref(tape, t, lag):
    T, p, m = _arr(tape)
    i = np.flatnonzero((T <= t + lag) & m)
    return float(p[i[-1]]) if len(i) else None


def _limit_filled(T, p, ref, t0, t1) -> bool:
    return bool(np.any((T > t0) & (T <= t1) & (p < ref)))


def a1_price(tape, t, lag, timeout):
    T, p, m = _arr(tape)
    ref = limit_ref(tape, t, lag)
    if ref is None:
        return _first_buy_at_or_after(T, p, m, t + lag)
    if _limit_filled(T, p, ref, t + lag, t + lag + timeout):
        return ref
    return _first_buy_at_or_after(T, p, m, t + lag + timeout)


def a2_fill(tape, t, lag, timeout):
    T, p, _ = _arr(tape)
    ref = limit_ref(tape, t, lag)
    if ref is not None and _limit_filled(T, p, ref, t + lag, t + lag + timeout):
        return True, ref
    return False, None


def pair_diff_bp(a0, arm, ret_pct, skipped) -> float:
    """Money of the arm minus A0 in bp of the trade's capital; ret_pct is the A0 trade's net result in %."""
    if skipped:
        return -float(ret_pct) * 100.0
    return (float(a0) / float(arm) - 1.0) * (1.0 + float(ret_pct) / 100.0) * 1e4


def calibration(sim, actual) -> dict:
    d = [abs(s - a) / s * 1e4 for s, a in zip(sim, actual) if s and a]
    med = float(np.median(d)) if d else None
    return {"n": len(d), "median_abs_bp": med, "p90_abs_bp": float(np.percentile(d, 90)) if d else None,
            "ok": med is not None and med <= CALIB_MAX_BP}


# ------------------------------------------------------------------ download + store
def _empty():
    return {"T": np.zeros(0, dtype=np.int64), "p": np.zeros(0), "q": np.zeros(0), "m": np.zeros(0, bool),
            "a": np.zeros(0, dtype=np.int64)}


def fetch_tape(client, sym, t0, t1, limit=1000) -> dict:
    by = {}
    start = int(t0)
    while start <= t1:
        end = min(start + HOUR - 1, int(t1))
        rows = client.agg_trades(sym, start, end, limit) or []
        for r in rows:
            by[int(r["a"])] = (int(r["T"]), float(r["p"]), float(r["q"]), bool(r["m"]))
        if len(rows) >= limit:
            last = int(rows[-1]["T"])
            start = last if last > start else start + 1     # same-ms trades: dedupe by id, never loop forever
        else:
            start = end + 1
    if not by:
        return _empty()
    ids = sorted(by, key=lambda a: (by[a][0], a))
    v = [by[a] for a in ids]
    return {"T": np.array([x[0] for x in v], dtype=np.int64), "p": np.array([x[1] for x in v]),
            "q": np.array([x[2] for x in v]), "m": np.array([x[3] for x in v], bool), "a": np.array(ids, dtype=np.int64)}


def tape_path(sym, t_ms, root: Path | None = None) -> Path:
    return (root or TAPE) / sym / f"{int(t_ms)}.npz"


def have_tape(sym, t_ms, root: Path | None = None) -> bool:
    return tape_path(sym, t_ms, root).exists()


def save_tape(path: Path, tape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in tape.items()})


def load_tape(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
