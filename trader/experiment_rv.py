"""RV -- train of the BTC <-> ETH rotation (btc-eth-rotation-rv-1028-spec.md §4).

  train  6 candidates on 2023-03-01 .. 2025-12-31 (panel cut before 2026): R, paired vs 50/50 buy-and-hold, years,
         50 circular-shift random rotations, BH BTC / ETH; PBO -> runtime/experiments/RV_TRAIN.json
  test   needs --operator-approved; written only if train produces a winner

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
        raise SystemExit("RV: the test runner is written only if train produces a winner (spec §4)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
