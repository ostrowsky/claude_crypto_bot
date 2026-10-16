"""E7 readout (e7-shadow-portfolio-1012-spec.md §6): P4 (X4) vs P0 on shadow days.

KEEP_X4       lower 95% of the paired daily difference (P4 - P0) > 0
REJECT_X4     upper 95% < 0
INCONCLUSIVE  otherwise, or < 28 days, or < 150 closed P4 trades (window extended, criteria never loosened)
x4_profitable mean daily R of P4 > 0 AND lower 95% of its net per-trade result > 0

  (from trader/) ../pyembed/python.exe shadow_readout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import daily_revenue as DR        # noqa: E402
import money_validator as MV      # noqa: E402

MIN_DAYS = 28
MIN_TRADES = 150
OUT = TC.RUNTIME / "shadow" / "READOUT_E7.json"


def series_from_marks(marks: dict, started_utc: str | None) -> list:
    """Daily R from 00:00 UTC equity marks; the start day is dropped unless the shadow started at 00:00."""
    m = sorted(marks.items())
    out = [(a, (eb / ea - 1) * 100) for (a, ea), (_, eb) in zip(m, m[1:])]
    if started_utc:
        st = str(started_utc)
        first_full = st[:10] if st[11:19] == "00:00:00" else None
        if first_full is None:
            out = [(d, v) for d, v in out if d > st[:10]]
    return out


def verdict(p0_series, p4_series, p4_trade_rets, min_days: int = MIN_DAYS, min_trades: int = MIN_TRADES) -> dict:
    a0, a4 = dict(p0_series), dict(p4_series)
    days = sorted(set(a0) & set(a4))
    paired = DR.paired(p4_series, p0_series)
    tci = MV.trade_ci(list(p4_trade_rets))
    agg4, agg0 = DR.aggregate([(d, a4[d]) for d in days]), DR.aggregate([(d, a0[d]) for d in days])
    out = {"days": len(days), "p4_trades": len(p4_trade_rets), "paired": paired, "p4": agg4, "p0": agg0, "trade_ci": tci,
           "x4_profitable": bool(agg4.get("mean_pct") is not None and agg4["mean_pct"] > 0
                                 and tci.get("lo95") is not None and tci["lo95"] > 0)}
    if len(days) < min_days or len(p4_trade_rets) < min_trades:
        out["status"] = "INCONCLUSIVE"
        out["reason"] = f"need >= {min_days} days and >= {min_trades} P4 trades"
    elif paired["lo95"] > 0:
        out["status"] = "KEEP_X4"
    elif paired["hi95"] < 0:
        out["status"] = "REJECT_X4"
    else:
        out["status"] = "INCONCLUSIVE"
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    st = json.loads((TC.RUNTIME / "shadow" / "state.json").read_text(encoding="utf-8"))
    books = st["books"]

    def series(name):
        return series_from_marks(books[name]["day_start_eq"], st.get("started_utc"))
    rets = []
    led = TC.RUNTIME / "shadow" / "ledger.jsonl"
    if led.exists():
        for line in led.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("type") == "exit" and r.get("book") == "x4_floor8":
                rets.append(float(r["ret_pct"]))
    v = verdict(series("champion"), series("x4_floor8"), rets)
    v["started_utc"] = st.get("started_utc")
    OUT.write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v[k] for k in ("status", "days", "p4_trades", "x4_profitable")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
