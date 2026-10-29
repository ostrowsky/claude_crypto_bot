"""RV -- train of the BTC <-> ETH rotation (btc-eth-rotation-rv-1028-spec.md §4).

  train  6 candidates on 2023-03-01 .. 2025-12-31 (panel cut before 2026): R, paired vs 50/50 buy-and-hold, years,
         50 circular-shift random rotations, BH BTC / ETH; PBO -> runtime/experiments/RV_TRAIN.json
  test   RV_mom_56 on 2026-01-01 .. last full day, ONCE (marker RV_TEST_OPENED.json); needs --operator-approved

  (from trader/) ../pyembed/python.exe -u experiment_rv.py train
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

import rotation_rv as RV          # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_SWITCHES = 12
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
TRAIN_OUT = TC.RUNTIME / "experiments" / "RV_TRAIN.json"
TEST_OUT = TC.RUNTIME / "experiments" / "RV_TEST.json"
TEST_MARKER = TC.RUNTIME / "experiments" / "RV_TEST_OPENED.json"
WINNER = "RV_mom_56"
SPEC = "btc-eth-rotation-rv-1028-spec.md"


def candidates() -> dict:
    return RV.candidates()


def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("paired_5050_by_year") or {}
    checks = {"beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
              "paired_5050_gt_0": (res.get("paired_5050") or {}).get("mean_pct") is not None
              and res["paired_5050"]["mean_pct"] > 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
              "enough_switches": (res.get("n_switches") or 0) >= MIN_SWITCHES}
    return all(checks.values()), checks


def decide(agg: dict, paired_5050: dict, random_means) -> dict:
    p95 = _p95(random_means)
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0,
              "paired_5050_lo_gt_0": paired_5050.get("lo95") is not None and paired_5050["lo95"] > 0,
              "beats_random_p95": p95 is not None and agg.get("mean_pct") is not None and agg["mean_pct"] > p95}
    return {"accepted": all(checks.values()), "checks": checks, "random_p95": p95}


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("RV: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def evaluate(panel, cand: dict, fm, day_from: str, day_to: str, n_random: int = N_RANDOM) -> dict:
    import daily_revenue as DR
    import timing as TM
    W = panel.cut(day_to)
    n = W.row(day_to) - W.row(day_from)
    half = TM.simulate(panel, {s: np.ones(n) for s in RV.PAIR}, fm, day_from, day_to)
    bh = {s: RV.simulate(panel, np.full(n, i), fm, day_from, day_to)["agg"] for i, s in enumerate(RV.PAIR)}
    r = RV.run(panel, cand, fm, day_from, day_to)
    hm = dict(half["series"])
    diffs = [(d, v - hm[d]) for d, v in r["series"] if d in hm]
    rand = [RV.simulate(panel, RV.shift(r["holdings"], s), fm, day_from, day_to)["agg"].get("mean_pct")
            for s in range(n_random)]
    return {"window": [day_from, day_to], "agg": r["agg"], "half_agg": half["agg"], "bh": bh,
            "paired_5050": DR.paired(r["series"], half["series"]), "paired_5050_by_month": DR.month_means(diffs),
            "random_means": rand, "random_p95": _p95(rand), "n_switches": r["n_switches"], "eth_share": r["eth_share"],
            "conservation_ok": r["conservation_ok"]}


def run_test(marker=TEST_MARKER, out_path=TEST_OUT, n_random=N_RANDOM) -> dict:
    import experiment_n8 as EN
    import experiment_tm as ETM
    import fill_model as FM
    import money_validator as MV
    import xsmom as XS
    panel = ETM.load_panel()
    day_to = XS.day_str(XS.day_num(EN.last_full_day(panel)) + 1)
    cand = candidates()[WINNER]
    open_test(marker, {"spec": SPEC, "rule": WINNER, "config": cand, "window": [TEST_FROM, day_to], "k": 1})
    fm = FM.model("taker", 1)
    t = evaluate(panel, cand, fm, TEST_FROM, day_to, n_random)
    v = decide(t["agg"], t["paired_5050"], t["random_means"])
    p = t["paired_5050"]
    print("TEST %s..%s %s R %+.4f [%+.4f, %+.4f] dd %+.1f%% | 50/50 %+.4f | vs 50/50 %+.4f [%+.4f, %+.4f] | "
          "BH BTC %+.4f BH ETH %+.4f | rnd p95 %+.4f | ETH share %.2f switches %d | %s"
          % (TEST_FROM, day_to, WINNER, t["agg"]["mean_pct"], t["agg"]["lo95"], t["agg"]["hi95"], t["agg"]["max_dd_pct"],
             t["half_agg"]["mean_pct"], p["mean_pct"], p["lo95"], p["hi95"], t["bh"]["BTCUSDT"]["mean_pct"],
             t["bh"]["ETHUSDT"]["mean_pct"], v["random_p95"], t["eth_share"], t["n_switches"],
             "ACCEPTED" if v["accepted"] else "REJECTED"), flush=True)
    MV.log_trial({"family": "RV", "name": WINNER, "k_opened_on_test": 1, **v, "test_R": t["agg"], "paired_5050": p})
    out = {"spec": SPEC, "rule": WINNER, "config": cand, "fill": "taker + 8 bps per side", "verdict": v, "test": t,
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def run_train(n_random=N_RANDOM) -> dict:
    import daily_revenue as DR
    import experiment_tm as ETM
    import fill_model as FM
    import pbo as PB
    import timing as TM
    panel = ETM.load_panel().cut(TEST_FROM)
    fm = FM.model("taker", 1)
    n = panel.row(TEST_FROM) - panel.row(TRAIN_FROM)
    half = TM.simulate(panel, {s: np.ones(n) for s in RV.PAIR}, fm, TRAIN_FROM, TEST_FROM)
    bh = {s: RV.simulate(panel, np.full(n, i), fm, TRAIN_FROM, TEST_FROM)["agg"] for i, s in enumerate(RV.PAIR)}
    print("50/50 %+.4f | BH BTC %+.4f | BH ETH %+.4f" % (half["agg"]["mean_pct"], bh["BTCUSDT"]["mean_pct"],
                                                        bh["ETHUSDT"]["mean_pct"]), flush=True)
    res, series, t0 = {}, {}, time.time()
    hm = dict(half["series"])
    for name, cand in candidates().items():
        r = RV.run(panel, cand, fm, TRAIN_FROM, TEST_FROM)
        diffs = [(d, v - hm[d]) for d, v in r["series"] if d in hm]
        rand = [RV.simulate(panel, RV.shift(r["holdings"], s), fm, TRAIN_FROM, TEST_FROM)["agg"].get("mean_pct")
                for s in range(n_random)]
        e = {"config": cand, "mean_pct": r["agg"]["mean_pct"], "agg": r["agg"],
             "paired_5050": DR.paired(r["series"], half["series"]), "paired_5050_by_year": _year_means(diffs),
             "random_means": rand, "random_p95": _p95(rand), "n_switches": r["n_switches"], "eth_share": r["eth_share"],
             "conservation_ok": r["conservation_ok"]}
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        res[name], series[name] = e, dict(r["series"])
        p = e["paired_5050"]
        print("%-10s R %+.4f | vs 50/50 %+.4f [%+.4f, %+.4f] | years %s | rnd p95 %+.4f | ETH share %.2f | switches %d | %s | %.1f min"
              % (name, e["mean_pct"], p["mean_pct"], p["lo95"], p["hi95"],
                 {y: round(v, 3) for y, v in e["paired_5050_by_year"].items()}, e["random_p95"], e["eth_share"],
                 e["n_switches"], "ADMIT" if ok else "-", (time.time() - t0) / 60), flush=True)
    names = list(res)
    days = sorted(set.intersection(*[set(series[n]) for n in names]))
    R = np.array([[series[n][d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS))
    adm = [n for n in names if res[n]["admitted"]]
    winner = max(adm, key=lambda n: res[n]["paired_5050"]["mean_pct"]) if (adm and pbo < PBO_MAX) else None
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "half_agg": half["agg"], "bh": bh,
           "candidates": res, "pbo": pbo, "winner": winner,
           "verdict": "winner -> test needs the operator" if winner else "STOP: no winner, test not opened",
           "utc": datetime.now(timezone.utc).isoformat()}
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        r = run_train()
        print(json.dumps({"winner": r["winner"], "pbo": r["pbo"], "verdict": r["verdict"]}, indent=1))
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("RV: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        run_test()
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
