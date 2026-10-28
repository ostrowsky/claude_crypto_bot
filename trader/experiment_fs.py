"""FS -- train-only check of BTC+ETH carry switched on by funding (fc-switch-n4-1028-spec.md §5).

  train   9 candidates FS_W{W}_T{theta%} on decision days 2023-03-01 .. 2025-12-31 on the BTC+ETH carry market
          (fc_market.npz restricted to the two pairs, cut before 2026); per candidate: R, paired vs FC_PASSIVE, years,
          50 circular-shift random schedules, liquidations, switches; PBO over the 9 -> runtime/experiments/FS_TRAIN.json
  There is NO test mode: the 2026 window was already opened for FC (2026-10-22), so an accepted candidate only
  becomes a proposal for a shadow book next to FC_PASSIVE (spec §2).

  (from trader/) ../pyembed/python.exe -u experiment_fs.py train
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

import carry as C                 # noqa: E402
import fc_switch as FSW           # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_SWITCHES = 6
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
OUT_DIR = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT_DIR / "FS_TRAIN.json"
SPEC = "fc-switch-n4-1028-spec.md"


def candidates() -> dict:
    out = {}
    for W in (7, 14, 30):
        for T in (5, 10, 15):
            out[f"FS_W{W}_T{T}"] = {"W": W, "theta_in": T / 100}
    return out


def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("paired_passive_by_year") or {}
    checks = {
        "paired_passive_gt_0": (res.get("paired_passive") or {}).get("mean_pct") is not None
        and res["paired_passive"]["mean_pct"] > 0,
        "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
        "beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
        "no_liquidation": res.get("liquidations", 1) == 0,
        "enough_switches": (res.get("n_switches") or 0) >= MIN_SWITCHES,
    }
    return all(checks.values()), checks


def select_winner(res: dict, pbo):
    if pbo is None or pbo >= PBO_MAX:
        return None
    adm = [n for n, v in res.items() if v.get("admitted")]
    return max(adm, key=lambda n: (res[n]["paired_passive"]["mean_pct"], n)) if adm else None


def train_market(m, test_from: str = TEST_FROM):
    return m.cut(test_from)


def load_market():
    import experiment_fc as EF
    return FSW.subset_market(EF.load_market(), FSW.PAIRS)


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def evaluate(market, cand: dict, fees, day_from, day_to, n_random, passive_series) -> dict:
    import daily_revenue as DR
    import experiment_fc as EF
    import timing as TM
    days = FSW.days_between(day_from, day_to)
    sched = FSW.schedule(market.funding, FSW.PAIRS, days, cand["W"], cand["theta_in"])
    cfg = EF.passive_config()
    r = C.simulate(market, cfg, fees, day_from, day_to, chooser=FSW.schedule_chooser(sched, days))
    pm = dict(passive_series)
    diffs = [(d, v - pm[d]) for d, v in r["series"] if d in pm]
    rand = []
    for seed in range(n_random):
        sh = TM.shift_targets({s: v.astype(float) for s, v in sched.items()}, seed)
        rand.append(C.simulate(market, cfg, fees, day_from, day_to,
                               chooser=FSW.schedule_chooser(sh, days))["agg"].get("mean_pct"))
    return {"name": None, "config": cand, "window": [day_from, day_to], "mean_pct": r["agg"].get("mean_pct"),
            "agg": r["agg"], "paired_passive": DR.paired(r["series"], passive_series),
            "paired_passive_by_year": _year_means(diffs), "r_by_year": _year_means(r["series"]),
            "random_means": rand, "random_p95": _p95(rand), "liquidations": r["liquidations"],
            "margin_stops": r["margin_stops"], "funding_income": r["funding_income"],
            "n_switches": FSW.n_switches(sched), "on_share": {s: float(np.mean(v)) for s, v in sched.items()},
            "n_trades": len(r["trades"]), "conservation_ok": r["conservation_ok"], "_series": r["series"]}


def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import experiment_fc as EF
    import experiment_xsmom as XE
    import pbo as PB
    market = train_market(load_market())
    fees = C.fees(1)
    t0 = time.time()
    pas = C.simulate(market, EF.passive_config(), fees, TRAIN_FROM, TEST_FROM)
    print("FC_PASSIVE (BTC+ETH market) R %+.4f [%+.4f, %+.4f] | trades %d | liq %d | stops %d"
          % (pas["agg"].get("mean_pct") or 0, pas["agg"].get("lo95") or 0, pas["agg"].get("hi95") or 0,
             len(pas["trades"]), pas["liquidations"], pas["margin_stops"]), flush=True)
    res, series = {}, {}
    cands = candidates()
    for i, (name, cand) in enumerate(cands.items(), 1):
        e = evaluate(market, cand, fees, TRAIN_FROM, TEST_FROM, n_random, pas["series"])
        e["name"] = name
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e.pop("_series"))
        res[name] = e
        print("%d/%d %-12s R %+.4f | vs passive %+.4f [%+.4f, %+.4f] | years %s | rnd p95 %+.4f | on %s | switches %d | "
              "liq %d | %s | %.1f min"
              % (i, len(cands), name, e["mean_pct"] or 0, e["paired_passive"].get("mean_pct") or 0,
                 e["paired_passive"].get("lo95") or 0, e["paired_passive"].get("hi95") or 0,
                 {y: round(v, 4) for y, v in e["paired_passive_by_year"].items()}, e["random_p95"] or 0,
                 {s[:3]: round(v, 2) for s, v in e["on_share"].items()}, e["n_switches"], e["liquidations"],
                 "ADMIT" if ok else "-", (time.time() - t0) / 60), flush=True)
    names = list(cands)
    days = sorted(set.intersection(*[set(series[n]) for n in names]))
    R = np.array([[series[n][d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= PBO_SPLITS * 2 else None
    winner = select_winner(res, pbo)
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "pairs": list(FSW.PAIRS),
           "passive": {"agg": pas["agg"], "liquidations": pas["liquidations"], "margin_stops": pas["margin_stops"]},
           "candidates": res, "pbo": pbo, "winner": winner,
           "deflated_sharpe": XE._deflated(list(series[winner].values()), len(cands)) if winner else None,
           "next": "shadow book proposal next to FC_PASSIVE" if winner else "N4 rejected on train",
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        r = run_train()
        print(json.dumps({"winner": r["winner"], "pbo": r["pbo"],
                          "admitted": [n for n, v in r["candidates"].items() if v["admitted"]]}, indent=1))
        return 0
    if cmd == "test":
        print("FS: no test mode -- the 2026 window was opened for FC on 2026-10-22; an accepted candidate goes to a "
              "shadow book (fc-switch-n4-1028-spec.md §2)")
        return 2
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
