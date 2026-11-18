"""TF-ENS -- train / one-time test of the Donchian trend ensemble (tf-ensemble-tfens-1118-spec.md §4-§6).

  train   2 candidates on 2024-01-01 .. 2025-12-31 (panel cut before 2026; 2023 is warm-up only); per candidate: R,
          ALWAYS (e = 1), 50 circular-shift random timings, BTC buy-and-hold and the equal-weight universe for reference;
          admission (§6), winner -> runtime/experiments/TFENS_TRAIN.json
  test    opened ONCE (marker TFENS_TEST_OPENED.json), the winner only, k = 1; needs --operator-approved and a fresh panel

  (from trader/) ../pyembed/python.exe -u experiment_tfens.py train
  (from trader/) ../pyembed/python.exe -u experiment_tfens.py test --operator-approved
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tfens as T                 # noqa: E402
import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

SPEC = "tf-ensemble-tfens-1118-spec.md"
TRAIN_FROM = "2024-01-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_REBAL = 12
YEARS = ("2024", "2025")
OUT_DIR = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT_DIR / "TFENS_TRAIN.json"
TEST_OUT = OUT_DIR / "TFENS_TEST.json"
TEST_MARKER = OUT_DIR / "TFENS_TEST_OPENED.json"


@dataclass(frozen=True)
class Cand:
    name: str
    band: float


def candidates() -> dict:
    return {"TF_B10": Cand("TF_B10", 0.10), "TF_B50": Cand("TF_B50", 0.50)}


# ------------------------------------------------------------------ pure parts (tested)
def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    years = res.get("paired_always_by_year") or {}
    m, p95 = res.get("mean_pct"), res.get("random_p95")
    pa = (res.get("paired_always") or {}).get("mean_pct")
    checks = {"R_gt_0": m is not None and m > 0,
              "beats_random_p95": m is not None and p95 is not None and m > p95,
              "paired_always_gt_0": pa is not None and pa > 0,
              "both_years_ge_0": len(years) >= 2 and all(v is not None and v >= 0 for v in years.values()),
              "enough_rebal": (res.get("n_rebal") or 0) >= MIN_REBAL}
    return all(checks.values()), checks


def select(train: dict):
    adm = [n for n, r in train.items() if r.get("admitted")]
    return max(adm, key=lambda n: (train[n]["mean_pct"], n)) if adm else None


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("TFENS: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(validate, select_fn, test, marker=TEST_MARKER) -> dict:
    tr = validate()
    w = select_fn(tr)
    if not w:
        return {"train": tr, "winner": None, "test": None, "verdict": "STOP: no candidate passes on train; the test half was not opened"}
    open_test(marker, {"spec": SPEC, "winner": w})
    return {"train": tr, "winner": w, "test": test(w), "verdict": "tested"}


def decide(test_agg: dict, paired_always: dict, random_means) -> dict:
    p95 = _p95(random_means)
    checks = {"R_lo_gt_0": test_agg.get("lo95") is not None and test_agg["lo95"] > 0,
              "paired_always_lo_gt_0": paired_always.get("lo95") is not None and paired_always["lo95"] > 0,
              "beats_random_p95": p95 is not None and test_agg.get("mean_pct") is not None and test_agg["mean_pct"] > p95}
    return {"accepted": all(checks.values()), "checks": checks, "random_p95": p95}


# ------------------------------------------------------------------ evaluation
def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def union_universe(panel, day_from: str, day_to: str) -> list:
    out, d = set(), XS.day_num(day_from)
    while d < XS.day_num(day_to):
        day = XS.day_str(d)
        out |= set(T.universe(panel, day))
        y, m = int(day[:4]), int(day[5:7])
        nxt = "%04d-%02d-01" % (y + (m == 12), 1 if m == 12 else m + 1)
        d = XS.day_num(nxt)
    return sorted(out)


def evaluate(panel, sig, cand: Cand, fm, day_from: str, day_to: str, n_random: int, alpha: float = 0.05) -> dict:
    import daily_revenue as DR
    tf = T.simulate(panel, lambda d: sig.weights(d), fm, day_from, day_to, cand.band)
    al = T.simulate(panel, lambda d: sig.weights(d, mode="always"), fm, day_from, day_to, cand.band)
    syms = union_universe(panel, day_from, day_to)
    L = XS.day_num(day_to) - XS.day_num(day_from)
    rand = []
    for seed in range(n_random):
        k = T.shifts(syms, L, seed)
        r = T.simulate(panel, lambda d, k=k: sig.weights(d, shift=k, window=(day_from, day_to)), fm, day_from, day_to, cand.band)
        rand.append(DR.aggregate(r["series"]).get("mean_pct"))
    agg = DR.aggregate(tf["series"], alpha=alpha)
    pa = DR.paired(tf["series"], al["series"], alpha=alpha)
    am = dict(al["series"])
    diffs = [(d, v - am[d]) for d, v in tf["series"] if d in am]
    exposure = float(np.mean([sum(sig.weights(XS.day_str(x)).values()) for x in range(XS.day_num(day_from), XS.day_num(day_to), 7)]))
    return {"name": cand.name, "config": asdict(cand), "window": [day_from, day_to], "mean_pct": agg.get("mean_pct"), "agg": agg,
            "n_rebal": tf["n_rebal"], "n_trades": len(tf["trades"]), "turnover_per_day": tf["turnover_per_day"],
            "cash_ok": tf["cash_ok"], "mean_exposure_weekly_sample": exposure,
            "delisted_sells": sum(1 for t in tf["trades"] if t["reason"] == "delisted"),
            "always_agg": DR.aggregate(al["series"], alpha=alpha), "paired_always": pa,
            "paired_always_by_year": {y: v for y, v in _year_means(diffs).items() if y in YEARS} if day_from < TEST_FROM else _year_means(diffs),
            "r_by_year": _year_means(tf["series"]), "random_means": rand, "random_p95": _p95(rand),
            "universe_union": len(syms), "_series": tf["series"]}


def references(panel, sig, fm, day_from: str, day_to: str) -> dict:
    import daily_revenue as DR
    btc = T.simulate(panel, lambda d: {"BTCUSDT": 1.0}, fm, day_from, day_to, 0.10)
    ew = T.simulate(panel, lambda d: {s: 1.0 / T.N_U for s in T.universe(panel, d)}, fm, day_from, day_to, 0.50)
    return {"btc_buy_hold": DR.aggregate(btc["series"]), "ew_universe": DR.aggregate(ew["series"])}


def _slim(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def load_panel():
    import experiment_xsmom as EX
    return EX.load_panel()


def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import fill_model as FM
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    full = load_panel()
    panel = full.cut(TEST_FROM)
    sig = T.Signals(panel)
    fm = FM.model("taker", 1)
    res = {}
    for name, cand in candidates().items():
        e = evaluate(panel, sig, cand, fm, TRAIN_FROM, TEST_FROM, n_random)
        ok, checks = admit(e)
        e.update(admitted=ok, checks=checks)
        res[name] = _slim(e)
        a = e["agg"]
        print(f"  {name}: R {a.get('mean_pct')} [{a.get('lo95')}, {a.get('hi95')}] dd {a.get('max_dd_pct')} | ALWAYS "
              f"{e['always_agg'].get('mean_pct')} | paired {e['paired_always'].get('mean_pct')} "
              f"[{e['paired_always'].get('lo95')}, {e['paired_always'].get('hi95')}] by year {e['paired_always_by_year']} | "
              f"random p95 {e['random_p95']} | rebal {e['n_rebal']} turnover/day {e['turnover_per_day']} | {checks}", flush=True)
    winner = select(res)
    refs = references(panel, sig, fm, TRAIN_FROM, TEST_FROM)
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "panel_last_day": XS.day_str(full.d0 + full.n - 1),
           "pairs_in_panel": len(full.syms), "fill": "taker + 8 bps per side (fill_model level 1)", "candidates": res,
           "references": refs, "winner": winner,
           "verdict": ("winner %s: propose the single 2026 look to the operator" % winner) if winner else
                      "STOP: no candidate passes on train; the test half was not opened",
           "provenance": PV.block(), "minutes": round((time.time() - t0) / 60, 1),
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("references", json.dumps({k: {x: v.get(x) for x in ("mean_pct", "lo95", "hi95", "max_dd_pct")} for k, v in refs.items()}))
    print("winner:", winner, "|", out["verdict"])
    return out


def run_test(train_path=TRAIN_OUT, out_path=TEST_OUT, marker=TEST_MARKER, n_random=N_RANDOM) -> dict:
    import daily_revenue as DR  # noqa: F401
    import fill_model as FM
    import money_validator as MV
    import provenance as PV
    PV.require_clean()
    if not Path(train_path).exists():
        raise RuntimeError("TFENS: no train result; the test half is not opened before train")
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    panel = load_panel()
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    last = XS.day_str(panel.d0 + panel.n - 1)
    if last < yesterday:
        raise RuntimeError("TFENS: the panel ends %s, before %s; rebuild it (experiment_xsmom.py panel) first" % (last, yesterday))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sig = T.Signals(panel)
    fm = FM.model("taker", 1)

    def test(name):
        e = evaluate(panel, sig, candidates()[name], fm, TEST_FROM, today, n_random)
        v = decide(e["agg"], e["paired_always"], e["random_means"])
        MV.log_trial({"family": "TFENS", "name": name, "k_opened_on_test": 1, **v, "test_R": e["agg"]})
        return {"result": _slim(e), "decision": v, "references": references(panel, sig, fm, TEST_FROM, today)}

    res = pipeline(lambda: train["candidates"], lambda t: train.get("winner"), test, marker)
    res.update({"spec": SPEC, "stage": "test", "window": [TEST_FROM, today], "provenance": PV.block(),
                "utc": datetime.now(timezone.utc).isoformat()})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["train"]:
        run_train()
    elif argv[:1] == ["test"]:
        if "--operator-approved" not in argv:
            print("TFENS: the test half is opened only by the operator: add --operator-approved")
            return 2
        r = run_test()
        print(json.dumps({k: r.get(k) for k in ("winner", "verdict")}, indent=1, default=str))
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
