"""L3 in money -- verdicts of execution-replay-e1-1010-spec.md §7 (used by H2 and later families).

  improves    paired daily-revenue difference vs the base, on the TEST half:
              lower bound > 0, max drawdown not worse than the base by > 5 pp,
              >= 60% of months with difference >= 0, >= 200 trades
  profitable  mean daily R > 0 AND lower 95% of the per-trade net return > 0
  trials      every candidate opened on test is appended to the trials ledger (k for Bonferroni)
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import trader_config as TC

TRIALS = TC.RUNTIME / "experiments" / "trials.jsonl"
DD_MARGIN_PP = 5.0
MONTH_AGREE = 0.60
MIN_TRADES = 200


def improves(paired: dict, cand_dd: float, base_dd: float, month_diffs: dict, n_trades: int) -> dict:
    checks = {
        "paired_lo_gt_0": bool(paired.get("lo95") is not None and paired["lo95"] > 0),
        "dd_not_worse": bool(cand_dd >= base_dd - DD_MARGIN_PP),        # drawdowns are negative numbers
        "months_agree": bool(month_diffs) and sum(1 for v in month_diffs.values() if v >= 0) / len(month_diffs) >= MONTH_AGREE,
        "enough_trades": n_trades >= MIN_TRADES,
    }
    return {"verdict": all(checks.values()), "checks": checks}


def trade_ci(rets: list, seed: int = 29, boot: int = 2000, alpha: float = 0.05) -> dict:
    n = len(rets)
    if n == 0:
        return {"n": 0, "mean": None, "lo95": None, "hi95": None}
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choices(rets, k=n)) / n for _ in range(boot))
    return {"n": n, "mean": round(sum(rets) / n, 4), "lo95": round(bs[int(alpha / 2 * boot)], 4),
            "hi95": round(bs[min(boot - 1, int((1 - alpha / 2) * boot))], 4)}


def profitable(day_agg: dict, trade_ci: dict) -> dict:
    checks = {"mean_day_gt_0": bool((day_agg.get("mean_pct") or 0) > 0),
              "trade_lo_gt_0": bool(trade_ci.get("lo95") is not None and trade_ci["lo95"] > 0)}
    return {"verdict": all(checks.values()), "checks": checks}


def log_trial(rec: dict, path: Path | None = None) -> None:
    path = path or TRIALS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec},
                            ensure_ascii=False, default=str) + "\n")
