"""TM -- train / one-time test of BTC/ETH spot timing (btc-eth-timing-tm-1028-spec.md §6-§7).

  train   20 candidates on decision days 2023-03-01 .. 2025-12-31 (panel and funding cut before 2026);
          per candidate: R, CE (constant mean share), 50 circular-shift random timings, BH; admission (§6),
          PBO per row T / V / F, winner per row -> runtime/experiments/TM_TRAIN.json
  test    opened ONCE (marker TM_TEST_OPENED.json), winners only, alpha = 0.05 / k; needs --operator-approved

  (from trader/) ../pyembed/python.exe -u experiment_tm.py train
  (from trader/) ../pyembed/python.exe -u experiment_tm.py test --operator-approved
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import timing as TM               # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_REBAL = 12
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
OUT_DIR = TC.RUNTIME / "experiments"
TEST_MARKER = OUT_DIR / "TM_TEST_OPENED.json"
TRAIN_OUT = OUT_DIR / "TM_TRAIN.json"
TEST_OUT = OUT_DIR / "TM_TEST.json"
SPEC = "btc-eth-timing-tm-1028-spec.md"


def candidates() -> dict:
    return TM.candidates()


ROWS = {"T": [n for n in candidates() if n.startswith("T_")], "V": [n for n in candidates() if n.startswith("V_")],
        "F": [n for n in candidates() if n.startswith("F_")]}


# ------------------------------------------------------------------ admission, selection, decision
def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("paired_ce_by_year") or {}
    checks = {
        "beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
        "paired_ce_gt_0": (res.get("paired_ce") or {}).get("mean_pct") is not None and res["paired_ce"]["mean_pct"] > 0,
        "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
        "enough_rebal": (res.get("n_rebal") or 0) >= MIN_REBAL,
    }
    return all(checks.values()), checks


def select_by_row(res: dict, pbo_by_row: dict) -> dict:
    out = {}
    for row, names in ROWS.items():
        pbo = pbo_by_row.get(row)
        if pbo is None or pbo >= PBO_MAX:
            continue
        adm = [n for n in names if res.get(n, {}).get("admitted")]
        if adm:
            out[row] = max(adm, key=lambda n: (res[n]["paired_ce"]["mean_pct"], n))
    return out


def alpha_for(k: int) -> float:
    return 0.05 / k


def decide(test_agg: dict, paired_ce: dict, alpha: float, random_means) -> dict:
    p95 = _p95(random_means)
    checks = {"R_lo_gt_0": test_agg.get("lo95") is not None and test_agg["lo95"] > 0,
              "paired_ce_lo_gt_0": paired_ce.get("lo95") is not None and paired_ce["lo95"] > 0,
              "beats_random_p95": p95 is not None and test_agg.get("mean_pct") is not None and test_agg["mean_pct"] > p95}
    return {"accepted": all(checks.values()), "checks": checks, "alpha": alpha, "random_p95": p95}


def require_train(path) -> None:
    if not Path(path).exists():
        raise RuntimeError("TM: the train result %s is missing; the test half is not opened before train" % path)


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("TM: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(train_fn, test_fn, marker) -> dict:
    winners = train_fn() or {}
    if not winners:
        return {"stop": True, "winners": {}}
    open_test(marker, {"spec": SPEC, "winners": winners})
    alpha = alpha_for(len(winners))
    return {"stop": False, "winners": winners, "alpha": alpha, "test": test_fn(winners, alpha)}


def train_panel(panel):
    return panel.cut(TEST_FROM)


def train_funding(funding: dict) -> dict:
    hi = XS.day_num(TEST_FROM) * TM.DAY_MS
    out = {}
    for s, f in funding.items():
        t = np.asarray(f["t"], dtype=np.int64)
        k = int(np.searchsorted(t, hi, "left"))
        out[s] = {"t": t[:k], "rate": np.asarray(f["rate"], dtype=float)[:k]}
    return out


# ------------------------------------------------------------------ data
def load_panel():
    import history_store as HS
    daily = {}
    for s in TM.SYMS:
        a = HS.load(HS.path_of(s))
        if a is None or not len(a["t"]):
            raise RuntimeError("TM: no 15m history for %s" % s)
        daily[s] = XS.daily_from_15m(a)
    return XS.Panel.from_daily(daily)


def load_funding() -> dict:
    import funding_store as FS
    out = {}
    for s in TM.SYMS:
        f = FS.load_funding(FS.funding_path(s))
        if f is None:
            raise RuntimeError("TM: no funding for %s" % s)
        o = np.argsort(f["t"], kind="stable")
        out[s] = {"t": np.asarray(f["t"], dtype=np.int64)[o], "rate": np.asarray(f["rate"], dtype=float)[o]}
    return out


# ------------------------------------------------------------------ evaluation of one candidate
def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def evaluate(panel, funding, cand, fm, day_from, day_to, n_random, alpha=0.05) -> dict:
    import daily_revenue as DR
    r = TM.run(panel, funding, cand, fm, day_from, day_to)
    tg = r["targets"]
    ce = TM.simulate(panel, TM.ce_targets(tg), fm, day_from, day_to)
    bh = TM.simulate(panel, {s: np.ones(len(w)) for s, w in tg.items()}, fm, day_from, day_to)
    paired = DR.paired(r["series"], ce["series"], alpha=alpha)
    pbh = DR.paired(r["series"], bh["series"], alpha=alpha)
    cm = dict(ce["series"])
    diffs = [(d, v - cm[d]) for d, v in r["series"] if d in cm]
    rand = [TM.simulate(panel, TM.shift_targets(tg, seed), fm, day_from, day_to)["agg"].get("mean_pct")
            for seed in range(n_random)]
    vals = [v for _, v in r["series"]]
    return {"name": cand.name, "config": asdict(cand), "window": [day_from, day_to], "mean_pct": r["agg"].get("mean_pct"),
            "agg": r["agg"], "n_rebal": r["n_rebal"], "exposure_share": r["exposure_share"],
            "turnover_per_day": r["turnover_per_day"], "conservation_ok": r["conservation_ok"],
            "ce_agg": ce["agg"], "bh_agg": bh["agg"], "paired_ce": paired, "paired_bh": pbh,
            "paired_ce_by_year": _year_means(diffs), "r_by_year": _year_means(r["series"]),
            "random_means": rand, "random_p95": _p95(rand),
            "sd_r": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "sd_paired": float(np.std([v for _, v in diffs], ddof=1)) if len(diffs) > 1 else None,
            "_series": r["series"]}


def _slim(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def _deflated(vals, n_trials):
    import pbo as PB
    from scipy.stats import kurtosis, skew
    x = np.asarray(vals, dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    sr = float(x.mean() / x.std(ddof=1))
    return {"sharpe_per_day": sr, "dsr": PB.deflated_sharpe(sr, n_trials, len(x), float(skew(x)), float(kurtosis(x, fisher=False)))}


# ------------------------------------------------------------------ runners
def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import fill_model as FM
    import pbo as PB
    full = load_panel()
    panel, funding = train_panel(full), train_funding(load_funding())
    fm = FM.model("taker", 1)
    res, series = {}, {}
    cands = candidates()
    t0 = time.time()
    for i, (name, cand) in enumerate(cands.items(), 1):
        e = evaluate(panel, funding, cand, fm, TRAIN_FROM, TEST_FROM, n_random)
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e["_series"])
        res[name] = _slim(e)
        g = e["agg"]
        print("%2d/%d %-11s R %+.4f | CE %+.4f | paired CE %+.4f [%+.4f, %+.4f] | BH %+.4f | rnd p95 %+.4f | exp %.2f | "
              "rebal %3d | years %s | %s | %.1f min"
              % (i, len(cands), name, g.get("mean_pct") or 0, e["ce_agg"].get("mean_pct") or 0,
                 e["paired_ce"].get("mean_pct") or 0, e["paired_ce"].get("lo95") or 0, e["paired_ce"].get("hi95") or 0,
                 e["bh_agg"].get("mean_pct") or 0, e["random_p95"] or 0, e["exposure_share"], e["n_rebal"],
                 {y: round(v, 4) for y, v in e["paired_ce_by_year"].items()}, "ADMIT" if ok else "-",
                 (time.time() - t0) / 60), flush=True)
    pbo_by_row = {}
    for row, names in ROWS.items():
        days = sorted(set.intersection(*[set(series[n]) for n in names]))
        R = np.array([[series[n][d] for n in names] for d in days])
        pbo_by_row[row] = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= PBO_SPLITS * 2 else None
    winners = select_by_row(res, pbo_by_row)
    test_days = max(1, (full.d0 + full.n) - XS.day_num(TEST_FROM))
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "syms": list(TM.SYMS),
           "start_equity": TM.START_EQUITY, "band": TM.BAND, "fill": "taker + 8 bps per side (fill_model level 1)",
           "panel_last_day": XS.day_str(full.d0 + full.n - 1), "candidates": res, "pbo_by_row": pbo_by_row,
           "winners": winners,
           "deflated_sharpe": {row: _deflated(list(series[n].values()), len(cands)) for row, n in winners.items()},
           "power": {n: {"halfwidth_paired_test": (1.96 * r["sd_paired"] / math.sqrt(test_days)) if r.get("sd_paired") else None}
                     for n, r in res.items()},
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def run_test(train_path=TRAIN_OUT, out_path=TEST_OUT, marker=TEST_MARKER, n_random=N_RANDOM) -> dict:
    import fill_model as FM
    import money_validator as MV
    require_train(train_path)
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    panel, funding = load_panel(), load_funding()
    last = XS.day_str(panel.d0 + panel.n - 1)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if last < yesterday:
        raise RuntimeError("TM: the panel ends %s, before %s; refresh history_store for BTCUSDT/ETHUSDT first" % (last, yesterday))
    day_to = XS.day_str(panel.d0 + panel.n)

    def test_fn(winners, alpha):
        fm = FM.model("taker", 1)
        cands, out = candidates(), {}
        for row, name in winners.items():
            e = evaluate(panel, funding, cands[name], fm, TEST_FROM, day_to, n_random, alpha)
            v = decide(e["agg"], e["paired_ce"], alpha, e["random_means"])
            refs = {tag: TM.run(panel, funding, cands[name], fmx, TEST_FROM, day_to)["agg"]
                    for tag, fmx in (("bnb", FM.model("bnb", 1)), ("slip2", FM.model("taker", 2)))}
            out[row] = {"winner": name, "verdict": v, "result": _slim(e), "reference": refs}
            MV.log_trial({"family": "TM", "name": name, "k_opened_on_test": len(winners), **v, "test_R": e["agg"],
                          "paired_ce": e["paired_ce"]})
        return out

    res = pipeline(lambda: train.get("winners") or {}, test_fn, marker)
    res.update({"spec": SPEC, "stage": "test", "window": [TEST_FROM, day_to], "utc": datetime.now(timezone.utc).isoformat()})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        r = run_train()
        print(json.dumps({"winners": r["winners"], "pbo_by_row": r["pbo_by_row"],
                          "admitted": [n for n, v in r["candidates"].items() if v["admitted"]]}, indent=1))
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("TM: the test half is opened once and only on the operator's decision; add --operator-approved")
            return 2
        r = run_test()
        print(json.dumps({k: r.get(k) for k in ("stop", "winners", "alpha")}, indent=1, default=str))
        for row, v in (r.get("test") or {}).items():
            print(row, v["winner"], json.dumps(v["verdict"]))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
