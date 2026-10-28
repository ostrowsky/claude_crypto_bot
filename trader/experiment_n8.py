"""N8 -- one frozen low-volatility alt basket, judged ONCE on 2026 (lowvol-alts-n8-1028-spec.md §2-§5).

  test   marker N8_TEST_OPENED.json first, then: rule vs equal-weight universe (same H = 7 rotation), 50 matched-turnover
         random rotations, BTC buy-and-hold reference; train 2023-03..2025-12 computed beside it for reference only
         -> runtime/experiments/N8_TEST.json. Needs --operator-approved.

  (from trader/) ../pyembed/python.exe -u experiment_n8.py test --operator-approved
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lowvol as LV               # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
VOL_DAYS = 28
N_RANDOM = 50
START_EQUITY = 1_000_000.0
OUT_DIR = TC.RUNTIME / "experiments"
TEST_MARKER = OUT_DIR / "N8_TEST_OPENED.json"
TEST_OUT = OUT_DIR / "N8_TEST.json"
SPEC = "lowvol-alts-n8-1028-spec.md"


def config() -> XS.Config:
    return XS.Config(L=1, H=7, K=10, min_age_days=60)


def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def decide(agg: dict, paired_ew: dict, random_means) -> dict:
    p95 = _p95(random_means)
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0,
              "paired_ew_lo_gt_0": paired_ew.get("lo95") is not None and paired_ew["lo95"] > 0,
              "beats_random_p95": p95 is not None and agg.get("mean_pct") is not None and agg["mean_pct"] > p95}
    return {"accepted": all(checks.values()), "checks": checks, "random_p95": p95}


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("N8: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def last_full_day(panel) -> str:
    j = panel.col["BTCUSDT"]
    rows = np.flatnonzero(panel.complete[:, j])
    return XS.day_str(panel.d0 + int(rows[-1]))


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def evaluate(panel, day_from, day_to, fm, n_random) -> dict:
    import daily_revenue as DR
    cfg = config()
    ch = LV.lowvol_chooser(panel, cfg.K, VOL_DAYS)
    r = XS.rotate(panel, cfg, fm, day_from, day_to, chooser=ch, start_equity=START_EQUITY)
    ew = XS.rotate(panel, cfg, fm, day_from, day_to, chooser=XS.ewu_chooser, start_equity=START_EQUITY)
    btc = XS.rotate(panel, cfg, fm, day_from, day_to, chooser=XS.btc_chooser, start_equity=START_EQUITY)
    paired = DR.paired(r["series"], ew["series"])
    em = dict(ew["series"])
    rand = XS.random_control(panel, cfg, fm, day_from, day_to, r["moves"], n_random, start_equity=START_EQUITY)
    syms = {}
    for t in r["trades"]:
        syms[t["sym"]] = syms.get(t["sym"], 0) + 1
    return {"window": [day_from, day_to], "agg": r["agg"], "ew_agg": ew["agg"], "btc_agg": btc["agg"],
            "paired_ew": paired, "paired_ew_by_month": DR.month_means([(d, v - em[d]) for d, v in r["series"] if d in em]),
            "paired_ew_by_year": _year_means([(d, v - em[d]) for d, v in r["series"] if d in em]),
            "random_means": rand, "random_p95": _p95(rand), "n_trades": len(r["trades"]),
            "turnover_per_day": r["turnover_per_day"], "conservation_ok": r["conservation_ok"],
            "top_held": sorted(syms.items(), key=lambda kv: -kv[1])[:15]}


def run_test(marker=TEST_MARKER, out_path=TEST_OUT, n_random=N_RANDOM) -> dict:
    import experiment_xsmom as XE
    import fill_model as FM
    import money_validator as MV
    panel = XE.load_panel()
    last = last_full_day(panel)
    day_to = XS.day_str(XS.day_num(last) + 1)
    open_test(marker, {"spec": SPEC, "rule": asdict(config()), "vol_days": VOL_DAYS, "window": [TEST_FROM, day_to]})
    fm = FM.model("taker", 1)
    t0 = time.time()
    test = evaluate(panel, TEST_FROM, day_to, fm, n_random)
    v = decide(test["agg"], test["paired_ew"], test["random_means"])
    print("TEST %s..%s R %+.4f [%+.4f, %+.4f] | EW %+.4f | paired %+.4f [%+.4f, %+.4f] | rnd p95 %+.4f | BTC %+.4f | %s | %.0f min"
          % (TEST_FROM, day_to, test["agg"]["mean_pct"], test["agg"]["lo95"], test["agg"]["hi95"],
             test["ew_agg"]["mean_pct"], test["paired_ew"]["mean_pct"], test["paired_ew"]["lo95"],
             test["paired_ew"]["hi95"], v["random_p95"], test["btc_agg"]["mean_pct"],
             "ACCEPTED" if v["accepted"] else "REJECTED", (time.time() - t0) / 60), flush=True)
    MV.log_trial({"family": "N8", "name": "LOWVOL_K10_H7_V28", "k_opened_on_test": 1, **v, "test_R": test["agg"],
                  "paired_ew": test["paired_ew"]})
    train = evaluate(panel.cut(TEST_FROM), TRAIN_FROM, TEST_FROM, fm, n_random)
    print("TRAIN (reference) R %+.4f | EW %+.4f | paired %+.4f [%+.4f, %+.4f] | rnd p95 %+.4f | BTC %+.4f"
          % (train["agg"]["mean_pct"], train["ew_agg"]["mean_pct"], train["paired_ew"]["mean_pct"],
             train["paired_ew"]["lo95"], train["paired_ew"]["hi95"], train["random_p95"], train["btc_agg"]["mean_pct"]),
          flush=True)
    out = {"spec": SPEC, "rule": asdict(config()), "vol_days": VOL_DAYS, "fill": "taker + 8 bps per side",
           "verdict": v, "test": test, "train_reference": train, "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["test"]:
        if "--operator-approved" not in argv:
            print("N8: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        r = run_test()
        print(json.dumps(r["verdict"], indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
