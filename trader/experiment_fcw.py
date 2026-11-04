"""FC-W -- passive funding carry on the N most liquid pairs (fc-wide-carry-fcw-1104-spec.md).

  train    2023-03-01..2025-12-31 on the FC market cache: 3 candidates, alpha 0.05 / 3, vs FC_PASSIVE BTC+ETH on the
           same days, 50 matched random controls -> runtime/experiments/FCW_TRAIN.json
  holdout  the train winner only, ONE look, 2020-02-01..2022-12-31 on the BH wide market (archive_bh.build_wide),
           alpha 0.05 -> runtime/experiments/FCW_HOLDOUT.json (refuses to run twice)

  (from trader/) ../pyembed/python.exe -u experiment_fcw.py train | holdout
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

import fcw                        # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN = ("2023-03-01", "2026-01-01")
HOLDOUT = ("2020-02-01", "2023-01-01")
ALPHA_TRAIN = 0.05 / 3
ALPHA_HOLDOUT = 0.05
N_RANDOM = 50
YEARS_AGREE = 2
TRAIN_OUT = TC.RUNTIME / "experiments" / "FCW_TRAIN.json"
HOLDOUT_OUT = TC.RUNTIME / "experiments" / "FCW_HOLDOUT.json"
SPEC = "fc-wide-carry-fcw-1104-spec.md"


def admit(res: dict):
    a = res.get("agg") or {}
    years = res.get("r_by_year") or {}
    checks = {"mean_lo_gt_0": a.get("lo95") is not None and a["lo95"] > 0,
              "no_liquidation": res.get("liquidations", 1) == 0,
              "beats_passive": (res.get("paired_passive") or {}).get("lo95") is not None and res["paired_passive"]["lo95"] > 0,
              "beats_random_p95": res.get("random_p95") is not None and a.get("mean_pct") is not None
              and a["mean_pct"] > res["random_p95"],
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def evaluate(market, cfg, fm, lo, hi, n_random, passive_series, alpha) -> dict:
    import carry as C
    import daily_revenue as DR
    r = C.simulate(market, cfg, fm, lo, hi, chooser=fcw.chooser(cfg.K))
    rand = C.random_control(market, cfg, fm, lo, hi, r["moves"], n_random) if n_random else []
    rand = [v for v in rand if v is not None]
    by_year = {}
    for d, v in r["series"]:
        by_year.setdefault(d[:4], []).append(v)
    reasons = {}
    for t in r["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"config": {"K": cfg.K, "lev": cfg.lev, "margin_stop": cfg.margin_stop}, "window": [lo, hi],
            "agg": DR.aggregate(r["series"], alpha=alpha), "paired_passive": DR.paired(r["series"], passive_series, alpha=alpha),
            "random_means": rand, "random_p95": float(np.percentile(rand, 95)) if rand else None,
            "liquidations": r["liquidations"], "margin_stops": r["margin_stops"], "n_trades": len(r["trades"]),
            "exit_reasons": reasons, "funding_income": r["funding_income"], "turnover_per_day": r["turnover_per_day"],
            "exposure_share": r["exposure_share"], "conservation_ok": r["conservation_ok"],
            "r_by_year": {y: sum(v) / len(v) for y, v in sorted(by_year.items())},
            "pairs_held": sorted({t["sym"] for t in r["trades"]} | set(r.get("open_syms", []))) if r["trades"] else []}


def _passive(market, fm, lo, hi, alpha):
    import carry as C
    import daily_revenue as DR
    import experiment_fc as EF
    p = C.simulate(market, EF.passive_config(), fm, lo, hi)
    return p["series"], {"agg": DR.aggregate(p["series"], alpha=alpha), "liquidations": p["liquidations"]}


def _line(name, r):
    a, pp = r["agg"], r["paired_passive"]
    return (f"{name}: R {a.get('mean_pct')} [{a.get('lo95')}, {a.get('hi95')}] dd {a.get('max_dd_pct')} | vs passive "
            f"{pp.get('mean_pct')} [{pp.get('lo95')}, {pp.get('hi95')}] | random p95 {r['random_p95']} | liq {r['liquidations']} "
            f"stops {r['margin_stops']} | years { {k: round(v, 4) for k, v in r['r_by_year'].items()} }")


def run_train() -> dict:
    import carry as C
    import experiment_fc as EF
    import pbo as PB
    t0 = time.time()
    fm = C.fees(1)
    market = EF.load_market().cut(EF.TEST_FROM)
    ps, passive = _passive(market, fm, *TRAIN, ALPHA_TRAIN)
    print(f"FC_PASSIVE: R {passive['agg'].get('mean_pct')} [{passive['agg'].get('lo95')}, {passive['agg'].get('hi95')}] "
          f"liq {passive['liquidations']}", flush=True)
    res, series = {}, {}
    for name, cfg in fcw.candidates().items():
        r = evaluate(market, cfg, fm, *TRAIN, N_RANDOM, ps, ALPHA_TRAIN)
        ok, checks = admit(r)
        r["admitted"], r["checks"] = ok, checks
        res[name] = {k: v for k, v in r.items() if k != "random_means"}
        print(_line(name, r) + f" | {'ADMIT' if ok else '-'} | {(time.time() - t0) / 60:.1f} min", flush=True)
    adm = [n for n in res if res[n]["admitted"]]
    winner = max(adm, key=lambda n: res[n]["agg"]["mean_pct"]) if adm else None
    out = {"spec": SPEC, "window": TRAIN, "alpha": ALPHA_TRAIN, "passive": passive, "candidates": res, "winner": winner,
           "verdict": "winner -> one holdout look 2020-2022" if winner else "STOP: no winner",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"winner": winner, "verdict": out["verdict"]}))
    return out


def run_holdout() -> dict:
    import archive_bh as A
    import carry as C
    if HOLDOUT_OUT.exists():
        raise SystemExit("FC-W holdout already looked at once: %s" % HOLDOUT_OUT)
    tr = json.loads(TRAIN_OUT.read_text(encoding="utf-8"))
    if not tr.get("winner"):
        raise SystemExit("no train winner: the holdout is not opened")
    name = tr["winner"]
    cfg = fcw.candidates()[name]
    fm = C.fees(1)
    market = A.load_wide_market()
    ps, passive = _passive(market, fm, *HOLDOUT, ALPHA_HOLDOUT)
    r = evaluate(market, cfg, fm, *HOLDOUT, N_RANDOM, ps, ALPHA_HOLDOUT)
    checks = {"mean_lo_gt_0": (r["agg"].get("lo95") or -1) > 0, "no_liquidation": r["liquidations"] == 0,
              "beats_passive": (r["paired_passive"].get("lo95") or -1) > 0}
    out = {"spec": SPEC, "window": HOLDOUT, "alpha": ALPHA_HOLDOUT, "winner": name, "passive": passive,
           "result": {k: v for k, v in r.items() if k != "random_means"}, "checks": checks,
           "accepted": all(checks.values()), "utc": datetime.now(timezone.utc).isoformat()}
    HOLDOUT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(_line(name, r), "|", checks, "| accepted", out["accepted"])
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
