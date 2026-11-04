"""PX -- pairs trading (pairs-trading-px-1104-spec.md).

  train    2023-03-01..2025-12-31 on the FC market cache: 4 candidates, alpha 0.05 / 4, 50 random-pair controls on
           the same entry days and holds -> runtime/experiments/PX_TRAIN.json
  holdout  the train winner only, ONE look, 2020-02-01..2022-12-31 on archive_bh.load_wide_market(), alpha 0.05
           -> runtime/experiments/PX_HOLDOUT.json (refuses to run twice)

  (from trader/) ../pyembed/python.exe -u experiment_px.py train | holdout
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import px                         # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN = ("2023-03-01", "2026-01-01")
HOLDOUT = ("2020-02-01", "2023-01-01")
ALPHA_TRAIN = 0.05 / 4
N_RANDOM = 50
MIN_TRADES = 100
YEARS_AGREE = 2
TRAIN_OUT = TC.RUNTIME / "experiments" / "PX_TRAIN.json"
HOLDOUT_OUT = TC.RUNTIME / "experiments" / "PX_HOLDOUT.json"
SPEC = "pairs-trading-px-1104-spec.md"


def candidates() -> dict:
    return {f"PX_F{F}_Z{int(z * 10)}": {"F": F, "z_in": z} for F in (30, 60) for z in (2.0, 2.5)}


def admit(res: dict):
    a = res.get("agg") or {}
    years = res.get("r_by_year") or {}
    checks = {"mean_lo_gt_0": a.get("lo95") is not None and a["lo95"] > 0,
              "beats_random_p95": res.get("random_p95") is not None and a.get("mean_pct") is not None
              and a["mean_pct"] > res["random_p95"],
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
              "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES,
              "no_liquidation": res.get("liquidations", 1) == 0}
    return all(checks.values()), checks


def evaluate(market, cand, fees, lo, hi, n_random, alpha) -> dict:
    import daily_revenue as DR
    r = px.simulate(market, cand, fees, lo, hi)
    plan = [(t["entry_day"], t["exit_day"]) for t in r["trades"] if t["reason"] != "window_end"]
    rand = []
    for s in range(n_random):
        q = px.simulate(market, cand, fees, lo, hi, plan=plan, seed=100 + s)
        if q["series"]:
            rand.append(float(np.mean([v for _, v in q["series"]])))
    by_year = {}
    for d, v in r["series"]:
        by_year.setdefault(d[:4], []).append(v)
    tr = r["trades"]
    reasons = {}
    for t in tr:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    gross = [t["gross_price_pct"] for t in tr]                 # price move of both legs only: no fee, slip, funding
    return {"config": cand, "window": [lo, hi], "agg": DR.aggregate(r["series"], alpha=alpha),
            "random_means": rand, "random_p95": float(np.percentile(rand, 95)) if rand else None,
            "n_trades": len(tr), "liquidations": r["liquidations"], "exit_reasons": reasons,
            "trade_mean_pct": float(np.mean([t["ret_pct"] for t in tr])) if tr else None,
            "trade_mean_before_fees_pct": float(np.mean(gross)) if tr else None,
            "trade_mean_funding_pct": float(np.mean([t["funding_pct"] for t in tr])) if tr else None,
            "funding_share_of_pnl": (sum(t["funding"] for t in tr) / sum(t["pnl"] for t in tr)) if tr and sum(t["pnl"] for t in tr) else None,
            "converged_share": reasons.get("converged", 0) / len(tr) if tr else None,
            "r_by_year": {y: sum(v) / len(v) for y, v in sorted(by_year.items())},
            "conservation_ok": r["conservation_ok"], "_series": r["series"]}


def _line(name, e):
    a = e["agg"]
    return (f"{name}: R {a.get('mean_pct')} [{a.get('lo95')}, {a.get('hi95')}] dd {a.get('max_dd_pct')} | random p95 "
            f"{e['random_p95']} | trades {e['n_trades']} mean {e['trade_mean_pct']} (before fees {e['trade_mean_before_fees_pct']}) "
            f"| converged {e['converged_share']} | reasons {e['exit_reasons']} | liq {e['liquidations']} "
            f"| years { {k: round(v, 4) for k, v in e['r_by_year'].items()} }")


def run_train() -> dict:
    import carry as C
    import experiment_fc as EF
    import pbo as PB
    t0 = time.time()
    fees = C.fees(1)
    market = EF.load_market().cut(EF.TEST_FROM)
    res, series = {}, {}
    for name, cand in candidates().items():
        e = evaluate(market, cand, fees, *TRAIN, N_RANDOM, ALPHA_TRAIN)
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e.pop("_series"))
        e.pop("random_means")
        res[name] = e
        print(_line(name, e) + f" | {'ADMIT' if ok else '-'} | {(time.time() - t0) / 60:.1f} min", flush=True)
    days = sorted(set().union(*[set(v) for v in series.values()]))
    R = np.array([[series[n].get(d, 0.0) for n in res] for d in days])
    pbo = float(PB.cscv_pbo(R, 8)) if len(days) >= 16 else None
    adm = [n for n in res if res[n]["admitted"]]
    winner = max(adm, key=lambda n: res[n]["agg"]["mean_pct"]) if adm else None
    out = {"spec": SPEC, "window": TRAIN, "alpha": ALPHA_TRAIN, "candidates": res, "pbo": pbo, "winner": winner,
           "verdict": "winner -> one holdout look 2020-2022" if winner else "STOP: no winner",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pbo": pbo, "winner": winner, "verdict": out["verdict"]}))
    return out


def run_holdout() -> dict:
    import archive_bh as A
    import carry as C
    if HOLDOUT_OUT.exists():
        raise SystemExit("PX holdout already looked at once: %s" % HOLDOUT_OUT)
    tr = json.loads(TRAIN_OUT.read_text(encoding="utf-8"))
    if not tr.get("winner"):
        raise SystemExit("no train winner: the holdout is not opened")
    name = tr["winner"]
    e = evaluate(A.load_wide_market(), candidates()[name], C.fees(1), *HOLDOUT, N_RANDOM, 0.05)
    e.pop("_series")
    e.pop("random_means")
    checks = {"mean_lo_gt_0": (e["agg"].get("lo95") or -1) > 0,
              "beats_random_p95": e["random_p95"] is not None and e["agg"]["mean_pct"] > e["random_p95"]}
    out = {"spec": SPEC, "window": HOLDOUT, "winner": name, "result": e, "checks": checks,
           "accepted": all(checks.values()), "utc": datetime.now(timezone.utc).isoformat()}
    HOLDOUT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(_line(name, e), "|", checks, "| accepted", out["accepted"])
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["train"]:
        run_train()
    elif argv[:1] == ["holdout"]:
        run_holdout()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
