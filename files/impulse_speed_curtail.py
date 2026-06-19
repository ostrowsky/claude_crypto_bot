"""Regime curtailment with auto-revive for the impulse_speed mode.

Why: four independent analyses (lateness, extension gate, extension-in-bandit,
full multivariate logistic — OOS AUC ~0.50) proved there is NO learnable
entry-time signal that separates impulse_speed winners from losers. The mode's
profitability is regime-driven (profitable Mar-early May, negative mid-May-Jun).
So the only evidence-based lever is mode-level: pause the mode while its own
recent realized pnl is negative, and auto-revive when it turns positive again.

Backtest (_backtest_impulse_speed_curtail.py, full critic_dataset, robust
across a window/threshold grid): disabling impulse_speed when its trailing
14-day mean realized pnl < 0 lifts kept avg/trade +0.025 -> +0.162 and removes
~-93 of realized losses, at the cost of ~27% of its (already late, low-capture)
big-mover catches on bad-regime days.

Two entry points:
  - compute_and_write(): run daily (daily_learning). Reads recent realized pnl
    from critic_dataset, decides curtail on/off, writes the state file.
  - is_curtailed(): cheap, cached read for the live bot's entry path. Fails
    OPEN (returns False = mode active) on any error or missing/stale state, so a
    broken state file never silently kills the mode.

State file: .runtime/impulse_speed_curtail.json
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = _ROOT / ".runtime" / "impulse_speed_curtail.json"
CRITIC_DATASET = _ROOT / "files" / "critic_dataset.jsonl"

# Caching for the hot is_curtailed() path
_CACHE: dict = {"ts": 0.0, "val": False}
_CACHE_TTL_S = 300.0
# A state snapshot older than this is ignored (fail open) so a stale daily job
# can't pin the mode off forever.
_STATE_MAX_AGE_H = 48.0


def _cfg(name, default):
    try:
        import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_and_write(window_days: Optional[int] = None,
                      threshold: Optional[float] = None) -> dict:
    """Compute trailing-window impulse_speed mean realized pnl and write the
    curtail state. Returns the record written (or an error dict)."""
    window = int(window_days if window_days is not None
                 else _cfg("IMPULSE_SPEED_CURTAIL_WINDOW_DAYS", 14))
    thr = float(threshold if threshold is not None
                else _cfg("IMPULSE_SPEED_CURTAIL_PNL_THRESHOLD", 0.0))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window)).strftime("%Y-%m-%d")

    pnls = []
    try:
        with open(CRITIC_DATASET, encoding="utf-8", errors="replace") as f:
            for ln in f:
                if "impulse_speed" not in ln:
                    continue
                try:
                    e = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if e.get("signal_type") != "impulse_speed":
                    continue
                if str((e.get("decision", {}) or {}).get("action", "")) != "take":
                    continue
                if str(e.get("ts_signal", ""))[:10] < cutoff:
                    continue
                pnl = _f((e.get("labels", {}) or {}).get("trade_exit_pnl"))
                if pnl is not None:
                    pnls.append(pnl)
    except OSError as ex:
        return {"error": f"read failed: {ex}"}

    n = len(pnls)
    mean_pnl = (sum(pnls) / n) if n else None
    # Need a minimum of resolved trades to judge a regime; below that, stay
    # ACTIVE (fail open) rather than curtail on noise.
    min_n = int(_cfg("IMPULSE_SPEED_CURTAIL_MIN_TRADES", 8))
    curtailed = bool(n >= min_n and mean_pnl is not None and mean_pnl < thr)

    rec = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window,
        "threshold": thr,
        "n_trades": n,
        "trailing_mean_pnl": mean_pnl,
        "curtailed": curtailed,
        "min_trades": min_n,
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(rec, indent=2, default=str),
                              encoding="utf-8")
    except OSError as ex:
        rec["write_error"] = str(ex)
    return rec


def is_curtailed() -> bool:
    """Cheap cached read for the live entry path. Fails OPEN (False) on any
    problem so a broken/missing/stale state never silently disables the mode."""
    if not bool(_cfg("IMPULSE_SPEED_REGIME_CURTAIL_ENABLED", False)):
        return False
    now = time.time()
    if now - _CACHE["ts"] < _CACHE_TTL_S:
        return _CACHE["val"]
    val = False
    try:
        if STATE_FILE.exists():
            rec = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            ts = rec.get("computed_at")
            fresh = True
            if ts:
                age_h = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(ts)).total_seconds() / 3600.0
                fresh = age_h <= _STATE_MAX_AGE_H
            if fresh:
                val = bool(rec.get("curtailed", False))
    except Exception:
        val = False
    _CACHE["ts"] = now
    _CACHE["val"] = val
    return val


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = compute_and_write()
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print("is_curtailed() ->", is_curtailed())
