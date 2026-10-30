"""BH19 -- frozen rules on the unseen 2018-2022 backward holdout (backward-holdout-bh19-1030-spec.md §2).

  DIR_1d      CatBoost (DIR params) fitted ONCE on all 2023-03-01..2025-12-31 rows of the live panel, scored on the BH
              panel rows 2019-03-01..2022-12-31; top decile after costs and vs the universe (DIR §5)
  TM          T_sma_56 on 2018-01-01..2022-12-31 (experiment_tm.evaluate / decide)
  N8          LOWVOL_K10_H7_V28 on 2019-03-01..2022-12-31 (experiment_n8.evaluate / decide)
  RV          RV_mom_56 on 2018-01-01..2022-12-31 (experiment_rv.evaluate / decide)
  FC_PASSIVE  carry.simulate passive on 2020-02-01..2022-12-31 (experiment_fc.evaluate / judge "passive")
  alpha = 0.05 / 5 = 0.01 for every interval; one look; -> runtime/experiments/BH19_RESULT.json

  (from trader/) ../pyembed/python.exe -u experiment_bh19.py run [--only RULE ...]   (--only: rules whose run crashed)
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

ALPHA = 0.05 / 5
N_RANDOM = 50
OUT = TC.RUNTIME / "experiments" / "BH19_RESULT.json"
SPEC = "backward-holdout-bh19-1030-spec.md"
WIN = {"DIR_1d": ("2019-03-01", "2023-01-01"), "TM": ("2018-01-01", "2023-01-01"), "N8": ("2019-03-01", "2023-01-01"),
       "RV": ("2018-01-01", "2023-01-01"), "FC_PASSIVE": ("2020-02-01", "2023-01-01")}


def train_mask(days, lo: str, hi: str) -> np.ndarray:
    d = np.asarray(days).astype(str)
    return (d >= lo) & (d < hi)


def overall(results: dict) -> dict:
    return {"confirmed": sorted(k for k, v in results.items() if v.get("accepted")),
            "not_confirmed": sorted(k for k, v in results.items() if not v.get("accepted"))}


@contextlib.contextmanager
def alpha_default(alpha: float):
    """Intervals that callers compute without an explicit alpha are computed at `alpha` (one look, k = 5)."""
    import daily_revenue as DR
    a0, p0 = DR.aggregate, DR.paired

    def agg(series, seed=29, excluded=0, alpha=None):
        return a0(series, seed, excluded, alpha=ALPHA_NOW if alpha is None else alpha)

    def paired(a, b, seed=29, alpha=None):
        return p0(a, b, seed, alpha=ALPHA_NOW if alpha is None else alpha)
    global ALPHA_NOW
    ALPHA_NOW = alpha
    DR.aggregate, DR.paired = agg, paired
    try:
        yield
    finally:
        DR.aggregate, DR.paired = a0, p0


ALPHA_NOW = ALPHA


# ------------------------------------------------------------------ the five rules
def run_dir(bh_panel) -> dict:
    import experiment_dir as ED
    import experiment_xsmom as EX
    import xsmom as XS
    from catboost import CatBoostClassifier
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=60)

    def rows(P, lo, hi):
        unis = {d: u for d in range(max(P.row(lo), 0), min(P.row(hi), P.n))
                if (u := XS.universe(P, XS.day_str(P.d0 + d), cfg))}
        rw, times = ED.build_1d(P, cfg, unis)
        tidx, sym, r, Xb = rw.arrays()
        X = np.column_stack([Xb[:, :len(ED.D.FEATURES) + 2], ED._cross_section(tidx, Xb), Xb[:, -1:]])
        return tidx, sym, r, X, times
    live = EX.load_panel().cut(ED.TEST_FROM)
    t_tr, _, r_tr, X_tr, times_tr = rows(live, ED.TRAIN_FROM, ED.TEST_FROM)
    days_tr = np.array([times_tr[int(t)]["t0"].strftime("%Y-%m-%d") for t in t_tr])
    m = train_mask(days_tr, ED.TRAIN_FROM, ED.TEST_FROM)
    model = CatBoostClassifier(depth=4, iterations=300, learning_rate=0.05, random_seed=29, verbose=False,
                               thread_count=4, loss_function="Logloss")
    model.fit(X_tr[m], (r_tr[m] > 0).astype(int))
    lo, hi = WIN["DIR_1d"]
    t_bh, s_bh, r_bh, X_bh, times_bh = rows(bh_panel, lo, hi)
    p = model.predict_proba(X_bh)[:, 1]
    old = ED.ALPHA
    ED.ALPHA = ALPHA
    try:
        res = ED.evaluate(t_bh, s_bh, r_bh, p, times_bh)
    finally:
        ED.ALPHA = old
    res["n_train_rows"], res["n_bh_rows"] = int(m.sum()), int(len(r_bh))
    res["checks"] = {"sel_net_lo_gt_0": res["sel"]["lo95"] > 0, "paired_lo_gt_0": res["paired"]["lo95"] > 0}
    res["accepted"] = all(res["checks"].values())
    return res


def run_tm(bh_panel, funding) -> dict:
    import daily_revenue as DR
    import experiment_tm as ETM
    import fill_model as FM
    cand = ETM.candidates()["T_sma_56"]
    lo, hi = WIN["TM"]
    e = ETM.evaluate(bh_panel, funding, cand, FM.model("taker", 1), lo, hi, N_RANDOM, alpha=ALPHA)
    agg = DR.aggregate(e["_series"], alpha=ALPHA)
    v = ETM.decide(agg, e["paired_ce"], ALPHA, e["random_means"])
    return {"agg": agg, "paired_ce": e["paired_ce"], "paired_bh": e["paired_bh"], "bh_agg": e["bh_agg"],
            "random_p95": v["random_p95"], "exposure_share": e["exposure_share"], **v}


def run_n8(bh_panel) -> dict:
    import experiment_n8 as N8
    import fill_model as FM
    lo, hi = WIN["N8"]
    with alpha_default(ALPHA):
        t = N8.evaluate(bh_panel, lo, hi, FM.model("taker", 1), N_RANDOM)
    v = N8.decide(t["agg"], t["paired_ew"], t["random_means"])
    return {"agg": t["agg"], "paired_ew": t["paired_ew"], "ew_agg": t["ew_agg"], "btc_agg": t["btc_agg"],
            "paired_ew_by_year": t["paired_ew_by_year"], "n_trades": t["n_trades"], **v}


def run_rv(bh_panel) -> dict:
    import experiment_rv as RVX
    import fill_model as FM
    lo, hi = WIN["RV"]
    with alpha_default(ALPHA):
        t = RVX.evaluate(bh_panel, RVX.candidates()[RVX.WINNER], FM.model("taker", 1), lo, hi, N_RANDOM)
    v = RVX.decide(t["agg"], t["paired_5050"], t["random_means"])
    return {"agg": t["agg"], "paired_5050": t["paired_5050"], "half_agg": t["half_agg"], "bh": t["bh"],
            "eth_share": t["eth_share"], "n_switches": t["n_switches"], **v}


def fc_costs():
    """FC is judged with the cost model of its own train and test (experiment_fc.run_train / run_test)."""
    import carry as C
    return C.fees(1)


def run_fc(market) -> dict:
    import experiment_fc as EF
    lo, hi = WIN["FC_PASSIVE"]
    e = EF.evaluate(market, EF.passive_config(), fc_costs(), lo, hi, 0, alpha=ALPHA)
    v = EF.judge("passive", e, ALPHA)
    return {"agg": e.get("agg"), "liquidations": e.get("liquidations"), "r_by_year": e.get("r_by_year"),
            "accepted": v["outcome"] == "accepted", **{k: v[k] for k in v if k != "outcome"}, "outcome": v["outcome"]}


def run(only=None) -> dict:
    """only: rules to (re)compute after a crash that produced no number; the others are kept from BH19_RESULT.json."""
    import archive_bh as A
    t0 = time.time()
    panel = A.load_panel()
    market = A.load_market()
    res = {}
    if only:
        res = json.loads(OUT.read_text(encoding="utf-8"))["results"]
        for k in only:
            if k in res and "error" not in res[k]:
                raise SystemExit(f"{k} already has a result; one look only")
    for name, fn in (("FC_PASSIVE", lambda: run_fc(market)), ("RV", lambda: run_rv(panel)),
                     ("TM", lambda: run_tm(panel, market.funding)), ("N8", lambda: run_n8(panel)),
                     ("DIR_1d", lambda: run_dir(panel))):
        if only and name not in only:
            continue
        try:
            res[name] = fn()
        except Exception as e:                                     # noqa: BLE001
            res[name] = {"accepted": False, "error": f"{type(e).__name__}: {e}"}
        r = res[name]
        a = r.get("agg") or r.get("sel") or {}
        print(f"{name}: accepted={r.get('accepted')} | R/sel {a.get('mean_pct')} [{a.get('lo95')}, {a.get('hi95')}] "
              f"| {r.get('checks')} {r.get('error', '')} | {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {"spec": SPEC, "alpha": ALPHA, "windows": WIN, "results": res, "overall": overall(res),
           "utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["overall"]))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run(argv[2:] if argv[1:2] == ["--only"] else None)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
