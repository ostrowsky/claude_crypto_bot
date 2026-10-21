"""S1 -- train / one-time test of the reversal rotation across the whole spot market (s1-reversal-universe-1021-spec.md).

  train   16 candidates (K = 10 worst by L in {1,7,14,28} days, H 1/7, with/without the market filter, pair age >= 60
          days) on decision days 2023-03-01 .. 2025-12-31 (panel cut before 2026). Admission = the XSMOM rule; a row
          closes at PBO >= 0.5; ONE winner (largest paired difference with the equal-weight market); STOP unless its
          deflated Sharpe >= 0.90 at n_trials = 40 (24 XSMOM + 16 S1) -> runtime/experiments/S1_TRAIN.json
  test    the 2026 window, opened ONCE for the whole family: refused if S1_TEST_OPENED.json or XSMOM_TEST_OPENED.json
          exists; needs --operator-approved -> runtime/experiments/S1_TEST.json

The train window is not fresh (the hypothesis came from the XSMOM look, spec §0): train selects, the test decides.

  (from trader/) ../pyembed/python.exe -u experiment_xsmom.py panel      (the shared daily panel)
  (from trader/) ../pyembed/python.exe -u experiment_s1.py train
  (from trader/) ../pyembed/python.exe -u experiment_s1.py test --operator-approved
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import experiment_xsmom as XE     # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

TRAIN_FROM = XE.TRAIN_FROM
TEST_FROM = XE.TEST_FROM
N_RANDOM = 50
MIN_TRADES = XE.MIN_TRADES
PBO_MAX = 0.5
PBO_SPLITS = 8
DSR_MIN = 0.90
MIN_AGE_DAYS = 60
K = 10
START_EQUITY = XE.START_EQUITY
OUT_DIR = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT_DIR / "S1_TRAIN.json"
TEST_OUT = OUT_DIR / "S1_TEST.json"
TEST_MARKER = OUT_DIR / "S1_TEST_OPENED.json"
FAMILY_MARKERS = (TEST_MARKER, XE.TEST_MARKER)     # the 2026 window opens once for ranking-by-past-return
SPEC = "s1-reversal-universe-1021-spec.md"


# ------------------------------------------------------------------ candidates (spec §4)
def candidates() -> dict:
    out = {}
    for mf in (False, True):
        for L in (1, 7, 14, 28):
            for H in (1, 7):
                c = XS.Config(L=L, H=H, K=K, market_filter=mf, side="reversal", min_age_days=MIN_AGE_DAYS)
                out[c.name] = c
    return out


ROWS = {"R": [n for n in candidates() if n.startswith("R_")], "RF": [n for n in candidates() if n.startswith("RF_")]}
N_TRIALS_FAMILY = len(XE.candidates()) + len(candidates())


# ------------------------------------------------------------------ admission, PBO, selection, DSR (spec §6)
admit = XE.admit                     # the same rule as XSMOM (spec §6.2)


def row_open(pbo) -> bool:
    return pbo is not None and pbo < PBO_MAX


def select_winner(res: dict, pbo_by_row: dict):
    """One winner (k = 1): the largest paired difference with ewu among admitted candidates of open rows."""
    pool = [n for row, names in ROWS.items() if row_open(pbo_by_row.get(row))
            for n in names if n in res and res[n].get("admitted")]
    if not pool:
        return None
    return max(pool, key=lambda n: (res[n]["paired_ewu"]["mean_pct"], n))


def dsr_ok(dsr) -> bool:
    return dsr is not None and dsr >= DSR_MIN


def open_test(marker, info: dict, family_markers=FAMILY_MARKERS) -> None:
    for m in {Path(marker), *[Path(x) for x in family_markers]}:
        if m.exists():
            raise RuntimeError("S1: the 2026 window is already opened for this family (%s); a second look is not allowed" % m)
    Path(marker).parent.mkdir(parents=True, exist_ok=True)
    Path(marker).write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                            encoding="utf-8")


def pipeline(train_fn, test_fn, marker=TEST_MARKER, family_markers=FAMILY_MARKERS) -> dict:
    winner = train_fn()
    if not winner:
        return {"stop": True, "winner": None}
    open_test(marker, {"spec": SPEC, "winner": winner}, family_markers)
    alpha = 0.05
    return {"stop": False, "winner": winner, "alpha": alpha, "test": test_fn(winner, alpha)}


def train_panel(panel):
    return panel.cut(TEST_FROM)


def verdict(mean_pct, random_means, paired_ewu, day_agg) -> dict:
    p95 = XE._p95(random_means)
    v = {"beats_random": p95 is not None and mean_pct is not None and mean_pct > p95,
         "beats_market": paired_ewu.get("lo95") is not None and paired_ewu["lo95"] > 0,
         "profitable": day_agg.get("lo95") is not None and day_agg["lo95"] > 0, "random_p95": p95}
    if v["beats_random"] and v["beats_market"] and v["profitable"]:
        v["outcome"] = "accepted"
    elif v["beats_random"] and v["beats_market"]:
        v["outcome"] = "shadow_candidate"
    else:
        v["outcome"] = "rejected"
    return v


# ------------------------------------------------------------------ runners
def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import fill_model as FM
    import pbo as PB
    full = XE.load_panel()
    panel = train_panel(full)
    fm = FM.model("taker", 1)
    res, series, ewu_cache = {}, {}, {}
    cands = candidates()
    t0 = time.time()
    for i, (name, cfg) in enumerate(cands.items(), 1):
        e = XE.evaluate(panel, cfg, fm, TRAIN_FROM, TEST_FROM, n_random, ewu_cache)
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e["_series"])
        res[name] = XE._slim(e)
        g = e["agg"]
        print("%2d/%d %-14s R %+.4f [%+.4f, %+.4f] | ewu %+.4f | paired %+.4f | rnd p95 %+.4f | trades %d | turnover %.3f | %s | %.0f min"
              % (i, len(cands), name, g.get("mean_pct") or 0, g.get("lo95") or 0, g.get("hi95") or 0,
                 e["ewu_agg"].get("mean_pct") or 0, e["paired_ewu"].get("mean_pct") or 0, e["random_p95"] or 0,
                 e["n_trades"], e["turnover_per_day"], "ADMIT" if ok else "-", (time.time() - t0) / 60), flush=True)
    pbo_by_row = {}
    for row, names in ROWS.items():
        days = sorted(set.intersection(*[set(series[n]) for n in names]))
        R = np.array([[series[n][d] for n in names] for d in days])
        pbo_by_row[row] = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= PBO_SPLITS * 2 else None
    best = select_winner(res, pbo_by_row)
    dsr = XE._deflated(list(series[best].values()), N_TRIALS_FAMILY) if best else None
    winner = best if (best and dsr_ok((dsr or {}).get("dsr"))) else None
    stop_reason = None if winner else ("no admitted candidate in an open row" if not best else "deflated Sharpe below %.2f" % DSR_MIN)
    test_days = max(1, (full.d0 + full.n) - XS.day_num(TEST_FROM))
    power = {n: {"sd_r": r.get("sd_r"), "sd_paired": r.get("sd_paired"),
                 "halfwidth_r_test": (1.96 * r["sd_r"] / math.sqrt(test_days)) if r.get("sd_r") else None,
                 "halfwidth_paired_test": (1.96 * r["sd_paired"] / math.sqrt(test_days)) if r.get("sd_paired") else None}
             for n, r in res.items()}
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "pairs_in_panel": len(full.syms),
           "panel_last_day": XS.day_str(full.d0 + full.n - 1), "start_equity": START_EQUITY,
           "fill": "taker + 8 bps per side (fill_model level 1)", "candidates": res, "pbo_by_row": pbo_by_row,
           "best": best, "deflated_sharpe": dsr, "n_trials_family": N_TRIALS_FAMILY, "winner": winner,
           "stop_reason": stop_reason, "power_test_days_estimate": test_days, "power": power,
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def run_test(train_path=TRAIN_OUT, out_path=TEST_OUT, marker=TEST_MARKER, n_random=N_RANDOM) -> dict:
    import daily_revenue as DR
    import fill_model as FM
    import money_validator as MV
    XE.require_train(train_path)
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    panel = XE.load_panel()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    last = XS.day_str(panel.d0 + panel.n - 1)
    if last < yesterday:
        raise RuntimeError("S1: the panel ends %s, before %s; run history_store.py backfill and `experiment_xsmom.py panel` first"
                           % (last, yesterday))

    def test_fn(name, alpha):
        fm = FM.model("taker", 1)
        cfg = candidates()[name]
        e = XE.evaluate(panel, cfg, fm, TEST_FROM, today, n_random, {}, alpha)
        v = verdict(e["mean_pct"], e["random_means"], e["paired_ewu"], e["agg"])
        r = XS.rotate(panel, cfg, fm, TEST_FROM, today, start_equity=START_EQUITY)
        refs = {"btc": XS.rotate(panel, XS.Config(L=1, H=1, K=1), fm, TEST_FROM, today, chooser=XS.btc_chooser,
                                 start_equity=START_EQUITY)["agg"],
                "cash": {"mean_pct": 0.0}, "delisted_later": XS.delisted_later_share(r["trades"], panel)}
        for tag, fmx in (("slip0", FM.model("taker", 0)), ("slip2", FM.model("taker", 2)), ("bnb", FM.model("bnb", 1))):
            refs[tag] = XS.rotate(panel, cfg, fmx, TEST_FROM, today, start_equity=START_EQUITY)["agg"]
        for h in (0.0, 1.0):
            refs[f"haircut{h:g}"] = XS.rotate(panel, replace(cfg, delist_haircut=h), fm, TEST_FROM, today,
                                              start_equity=START_EQUITY)["agg"]
        try:
            x4 = XE._x4_reference(TEST_FROM)
            refs["vs_x4"] = {"paired": DR.paired(e["_series"], x4["series"], alpha=alpha), "x4_agg": x4["agg"]}
        except Exception as ex:  # reference only
            refs["vs_x4"] = {"error": repr(ex)[:300]}
        MV.log_trial({"family": "S1", "name": name, "k_opened_on_test": 1, **v, "test_R": e["agg"], "paired_ewu": e["paired_ewu"]})
        return {"winner": name, "alpha": alpha, "verdict": v, "result": XE._slim(e), "reference": refs}

    res = pipeline(lambda: train.get("winner"), test_fn, marker)
    res.update({"spec": SPEC, "stage": "test", "window": [TEST_FROM, today], "utc": datetime.now(timezone.utc).isoformat()})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        r = run_train()
        print(json.dumps({k: r.get(k) for k in ("best", "winner", "stop_reason", "pbo_by_row", "deflated_sharpe")}, indent=1,
                         default=str))
        print("admitted:", [n for n, v in r["candidates"].items() if v["admitted"]])
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("S1: the 2026 window is opened once for the family and only on the operator's decision; add --operator-approved")
            return 2
        r = run_test()
        print(json.dumps({k: r.get(k) for k in ("stop", "winner", "alpha")}, indent=1, default=str))
        if r.get("test"):
            print(json.dumps(r["test"]["verdict"], indent=1, default=str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
