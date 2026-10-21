"""XSMOM -- train / one-time test of the daily cross-sectional momentum rotation (xsmom-daily-money-1021-spec.md §5-§8).

  panel   daily panel from trader/runtime/history (668+ pairs, delisted included) -> runtime/experiments/xsmom_panel.npz
  train   24 candidates on decision days 2023-03-01 .. 2025-12-31 (the panel is cut before 2026: no 2026 bar is read);
          per candidate: R, matched-turnover random control (50 seeds), paired vs the equal-weight universe, years;
          admission (§7.2), PBO per row (§7.3), winner per row (§7.4) -> runtime/experiments/XSMOM_TRAIN.json
  test    opened ONCE (marker XSMOM_TEST_OPENED.json), winners only, alpha = 0.05 / k; needs --operator-approved

  (from trader/) ../pyembed/python.exe -u experiment_xsmom.py panel
  (from trader/) ../pyembed/python.exe -u experiment_xsmom.py train
  (from trader/) ../pyembed/python.exe -u experiment_xsmom.py test --operator-approved
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import xsmom as XS                # noqa: E402

TRAIN_FROM = "2023-03-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 50
MIN_TRADES = 200
PBO_MAX = 0.5
PBO_SPLITS = 8
YEARS_AGREE = 2
OUT_DIR = TC.RUNTIME / "experiments"
TEST_MARKER = OUT_DIR / "XSMOM_TEST_OPENED.json"
TRAIN_OUT = OUT_DIR / "XSMOM_TRAIN.json"
TEST_OUT = OUT_DIR / "XSMOM_TEST.json"
PANEL_CACHE = OUT_DIR / "xsmom_panel.npz"
SPEC = "xsmom-daily-money-1021-spec.md"


# ------------------------------------------------------------------ candidates (spec §5)
def candidates() -> dict:
    out = {}
    for mf in (False, True):
        for L in (7, 14, 28):
            for H in (1, 7):
                for K in (5, 10):
                    c = XS.Config(L=L, H=H, K=K, market_filter=mf)
                    out[c.name] = c
    return out


ROWS = {"M": [n for n in candidates() if n.startswith("M_")], "MF": [n for n in candidates() if n.startswith("MF_")]}


# ------------------------------------------------------------------ admission, PBO, selection (spec §7)
def _p95(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("paired_ewu_by_year") or {}
    checks = {
        "beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
        "paired_ewu_gt_0": (res.get("paired_ewu") or {}).get("mean_pct") is not None and res["paired_ewu"]["mean_pct"] > 0,
        "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
        "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES,
    }
    return all(checks.values()), checks


def row_open(pbo) -> bool:
    return pbo is not None and pbo < PBO_MAX


def select_by_row(res: dict, pbo_by_row: dict) -> dict:
    out = {}
    for row, names in ROWS.items():
        if not row_open(pbo_by_row.get(row)):
            continue
        adm = [n for n in names if n in res and res[n].get("admitted")]
        if adm:
            out[row] = max(adm, key=lambda n: (res[n]["paired_ewu"]["mean_pct"], n))
    return out


def alpha_for(k: int) -> float:
    return 0.05 / k


def require_train(path) -> None:
    if not Path(path).exists():
        raise RuntimeError("XSMOM: the train result %s is missing; the test half is not opened before train" % path)


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("XSMOM: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(train_fn, test_fn, marker) -> dict:
    """Winners from train; none -> STOP without opening the test; otherwise the marker first, then the test once."""
    winners = train_fn() or {}
    if not winners:
        return {"stop": True, "winners": {}}
    open_test(marker, {"spec": SPEC, "winners": winners})
    alpha = alpha_for(len(winners))
    return {"stop": False, "winners": winners, "alpha": alpha, "test": test_fn(winners, alpha)}


def train_panel(panel):
    return panel.cut(TEST_FROM)


def verdict(mean_pct, random_means, paired_ewu, day_agg, trade_ci) -> dict:
    import money_validator as MV
    p95 = _p95(random_means)
    return {"beats_random": p95 is not None and mean_pct is not None and mean_pct > p95,
            "beats_market": paired_ewu.get("lo95") is not None and paired_ewu["lo95"] > 0,
            "profitable": MV.profitable(day_agg, trade_ci)["verdict"], "random_p95": p95}


# ------------------------------------------------------------------ panel
def build_panel():
    import history_store as HS
    daily = {}
    files = sorted(HS.HIST_DIR.glob("*_15m.npz"))
    for i, f in enumerate(files, 1):
        a = HS.load(f)
        if a is not None and len(a["t"]):
            daily[f.name[:-len("_15m.npz")]] = XS.daily_from_15m(a)
        if i % 100 == 0:
            print("daily bars %d/%d pairs" % (i, len(files)), flush=True)
    p = XS.Panel.from_daily(daily)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PANEL_CACHE, days=p.days, syms=np.array(p.syms), close=p.close, first_close=p.first_close,
                        qv=p.qv, complete=p.complete, has_first=p.has_first, has_bar=p.has_bar)
    return p


def load_panel():
    if not PANEL_CACHE.exists():
        raise RuntimeError("XSMOM: %s missing; run `experiment_xsmom.py panel` first" % PANEL_CACHE)
    z = np.load(PANEL_CACHE, allow_pickle=False)
    return XS.Panel(z["days"], [str(s) for s in z["syms"]], z["close"], z["first_close"], z["qv"], z["complete"],
                    z["has_first"], z["has_bar"])


# ------------------------------------------------------------------ evaluation of one candidate
def evaluate(panel, cfg, fm, day_from, day_to, n_random, ewu_cache: dict, alpha: float = 0.05) -> dict:
    import daily_revenue as DR
    import money_validator as MV
    r = XS.rotate(panel, cfg, fm, day_from, day_to)
    key = (cfg.L, cfg.H, cfg.market_filter, day_from, day_to)
    if key not in ewu_cache:
        ewu_cache[key] = XS.rotate(panel, cfg, fm, day_from, day_to, chooser=XS.ewu_chooser)
    ewu = ewu_cache[key]
    paired = DR.paired(r["series"], ewu["series"], alpha=alpha)
    em = dict(ewu["series"])
    diffs = [(d, v - em[d]) for d, v in r["series"] if d in em]
    by_year, r_year = {}, {}
    for d, v in diffs:
        by_year.setdefault(d[:4], []).append(v)
    for d, v in r["series"]:
        r_year.setdefault(d[:4], []).append(v)
    rand = XS.random_control(panel, cfg, fm, day_from, day_to, r["moves"], n_random)
    rets = [t["ret_pct"] for t in r["trades"]]
    vals = [v for _, v in r["series"]]
    reasons = {}
    for t in r["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"name": cfg.name, "config": asdict(cfg), "window": [day_from, day_to], "mean_pct": r["agg"].get("mean_pct"),
            "agg": r["agg"], "n_trades": len(rets), "trade_ci": MV.trade_ci(rets), "exit_reasons": reasons,
            "random_means": rand, "random_p95": _p95(rand), "paired_ewu": paired,
            "paired_ewu_by_year": {y: sum(v) / len(v) for y, v in sorted(by_year.items())},
            "r_by_year": {y: sum(v) / len(v) for y, v in sorted(r_year.items())},
            "ewu_agg": ewu["agg"], "sd_r": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "sd_paired": float(np.std([v for _, v in diffs], ddof=1)) if len(diffs) > 1 else None,
            "turnover_per_day": r["turnover_per_day"], "exposure_share": r["exposure_share"],
            "conservation_ok": r["conservation_ok"], "_series": r["series"]}


def _slim(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def _deflated(series_vals, n_trials):
    import pbo as PB
    from scipy.stats import kurtosis, skew
    x = np.asarray(series_vals, dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    sr = float(x.mean() / x.std(ddof=1))
    return {"sharpe_per_day": sr, "dsr": PB.deflated_sharpe(sr, n_trials, len(x), float(skew(x)), float(kurtosis(x, fisher=False)))}


# ------------------------------------------------------------------ train / test runners
def run_train(out_path=TRAIN_OUT, n_random=N_RANDOM) -> dict:
    import fill_model as FM
    import pbo as PB
    full = load_panel()
    panel = train_panel(full)
    fm = FM.model("taker", 1)
    res, series, ewu_cache = {}, {}, {}
    cands = candidates()
    t0 = time.time()
    for i, (name, cfg) in enumerate(cands.items(), 1):
        e = evaluate(panel, cfg, fm, TRAIN_FROM, TEST_FROM, n_random, ewu_cache)
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e["_series"])
        res[name] = _slim(e)
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
    winners = select_by_row(res, pbo_by_row)
    test_days = max(1, (full.d0 + full.n) - XS.day_num(TEST_FROM))
    power = {n: {"sd_r": r.get("sd_r"), "sd_paired": r.get("sd_paired"),
                 "halfwidth_r_test": (1.96 * r["sd_r"] / math.sqrt(test_days)) if r.get("sd_r") else None,
                 "halfwidth_paired_test": (1.96 * r["sd_paired"] / math.sqrt(test_days)) if r.get("sd_paired") else None}
             for n, r in res.items()}
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "pairs_in_panel": len(full.syms),
           "panel_last_day": XS.day_str(full.d0 + full.n - 1), "fill": "taker + 8 bps per side (fill_model level 1)",
           "candidates": res, "pbo_by_row": pbo_by_row, "winners": winners,
           "deflated_sharpe": {row: _deflated(list(series[n].values()), len(cands)) for row, n in winners.items()},
           "power_test_days_estimate": test_days, "power": power, "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def _x4_reference(day_from: str) -> dict:
    """Series of the X4 book on the bot's entries (execution_replay), for the reference comparison (spec §6)."""
    import execution_replay as R
    import fill_model as FM
    import trading_policy as TP
    x4 = replace(TP.CHAMPION, name="x4_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))
    trades, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(trades, grid)
    r = R.run(x4, FM.model("taker", 1), grid, trades, floor_of=R._floor(), day_from=day_from, plan_cache={})
    return {"series": r["series"], "agg": r["agg"]}


def run_test(train_path=TRAIN_OUT, out_path=TEST_OUT, marker=TEST_MARKER, n_random=N_RANDOM) -> dict:
    import daily_revenue as DR
    import fill_model as FM
    import money_validator as MV
    require_train(train_path)
    train = json.loads(Path(train_path).read_text(encoding="utf-8"))
    panel = load_panel()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last = XS.day_str(panel.d0 + panel.n - 1)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if last < yesterday:
        raise RuntimeError("XSMOM: the panel ends %s, before %s; run history_store.py backfill and `panel` first" % (last, yesterday))
    day_to = today

    def test_fn(winners, alpha):
        fm = FM.model("taker", 1)
        cands, ewu_cache, out = candidates(), {}, {}
        try:
            x4 = _x4_reference(TEST_FROM)
        except Exception as e:  # reference only: its absence does not change the verdicts
            x4 = {"error": repr(e)[:300]}
        for row, name in winners.items():
            cfg = cands[name]
            e = evaluate(panel, cfg, fm, TEST_FROM, day_to, n_random, ewu_cache, alpha)
            v = verdict(e["mean_pct"], e["random_means"], e["paired_ewu"], e["agg"], e["trade_ci"])
            refs = {}
            for tag, fmx in (("slip0", FM.model("taker", 0)), ("slip2", FM.model("taker", 2)), ("bnb", FM.model("bnb", 1))):
                refs[tag] = XS.rotate(panel, cfg, fmx, TEST_FROM, day_to)["agg"]
            for h in (0.0, 1.0):
                refs[f"haircut{h:g}"] = XS.rotate(panel, replace(cfg, delist_haircut=h), fm, TEST_FROM, day_to)["agg"]
            if "series" in x4:
                pr = DR.paired(e["_series"], x4["series"], alpha=alpha)
                bm = dict(x4["series"])
                months = DR.month_means([(d, val - bm[d]) for d, val in e["_series"] if d in bm])
                refs["vs_x4"] = {"paired": pr, "improves": MV.improves(pr, e["agg"].get("max_dd_pct", 0),
                                                                         x4["agg"].get("max_dd_pct", 0), months, e["n_trades"])}
            else:
                refs["vs_x4"] = x4
            out[row] = {"winner": name, "alpha": alpha, "verdict": v, "result": _slim(e), "reference": refs}
            MV.log_trial({"family": "XSMOM", "name": name, "k_opened_on_test": len(winners), **v,
                          "test_R": e["agg"], "paired_ewu": e["paired_ewu"]})
        return out

    res = pipeline(lambda: train.get("winners") or {}, test_fn, marker)
    res.update({"spec": SPEC, "stage": "test", "window": [TEST_FROM, day_to], "utc": datetime.now(timezone.utc).isoformat()})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "panel":
        p = build_panel()
        print(json.dumps({"pairs": len(p.syms), "days": int(p.n), "from": XS.day_str(p.d0),
                          "to": XS.day_str(p.d0 + p.n - 1), "file": str(PANEL_CACHE)}, indent=1))
        return 0
    if cmd == "train":
        r = run_train()
        print(json.dumps({"winners": r["winners"], "pbo_by_row": r["pbo_by_row"],
                          "admitted": [n for n, v in r["candidates"].items() if v["admitted"]]}, indent=1))
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("XSMOM: the test half is opened once and only on the operator's decision; add --operator-approved")
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
