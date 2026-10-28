"""MD -- multi-day spot trading, train / one-time test (multiday-spot-md-1028-spec.md).

  train   row B: the bot's entries on the X4 book, P-1 trail floor {8, 12, 20}% x horizon {3, 5, 10} days vs X4,
                 entries before 2026-06-21 (10-day purge before the X4F split)
          row U: event entries (BRK / DIP / VOL) on the top-100 spot universe held {2, 5, 10} days, 2023-03..2025-12,
                 vs 50 matched random entries and the equal-weight universe; PBO over the 9
          -> runtime/experiments/MD_TRAIN.json
  test    opened ONCE for the winners (marker MD_TEST_OPENED.json), alpha = 0.05 / k; needs --operator-approved

  (from trader/) ../pyembed/python.exe -u experiment_md.py train
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import multiday as MD             # noqa: E402
import trader_config as TC        # noqa: E402
import trading_policy as TP       # noqa: E402
import xsmom as XS                # noqa: E402

U_TRAIN_FROM = "2023-03-01"
U_TEST_FROM = "2026-01-01"
B_SPLIT = "2026-07-01"
B_PURGE_DAYS = 10
N_RANDOM = 50
MIN_TRADES_U = 200
MIN_CHANGED_B = 100
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
K_SLOTS = 10
START_EQUITY = 1_000_000.0
OUT_DIR = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT_DIR / "MD_TRAIN.json"
TEST_MARKER = OUT_DIR / "MD_TEST_OPENED.json"
SPEC = "multiday-spot-md-1028-spec.md"


def candidates_u() -> dict:
    return {f"U_{k}_H{h}": {"kind": k, "H": h, "K": K_SLOTS} for k in ("BRK", "DIP", "VOL") for h in (2, 5, 10)}


def candidates_b() -> dict:
    import experiment_x4f as X4F
    out = {}
    for f in (0.08, 0.12, 0.20):
        for d in (3, 5, 10):
            n = f"B_f{round(f * 100):02d}_d{d}"
            out[n] = replace(X4F.BASE, name=n, exits=TP.Exits(kind="p1_trail", floor_override=f, max_bars=96 * d))
    return out


def b_windows(grid_last_day: str):
    train_to = XS.day_str(XS.day_num(B_SPLIT) - B_PURGE_DAYS)
    test_to = XS.day_str(XS.day_num(grid_last_day) - B_PURGE_DAYS)
    return train_to, B_SPLIT, test_to


def ucfg() -> XS.Config:
    return XS.Config(L=1, H=1, K=K_SLOTS, min_age_days=60)


def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit_u(res: dict):
    p95 = _p95(res.get("random_means") or [])
    years = res.get("paired_ew_by_year") or {}
    checks = {"beats_random_p95": p95 is not None and res.get("mean_pct") is not None and res["mean_pct"] > p95,
              "paired_ew_gt_0": (res.get("paired_ew") or {}).get("mean_pct") is not None and res["paired_ew"]["mean_pct"] > 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE,
              "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES_U}
    return all(checks.values()), checks


def admit_b(res: dict) -> bool:
    lo = (res.get("paired") or {}).get("lo95")
    m = (res.get("agg") or {}).get("mean_pct")
    return res.get("changed", 0) >= MIN_CHANGED_B and lo is not None and lo > 0 and m is not None and m > 0


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("MD: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


# ------------------------------------------------------------------ row U
def evaluate_u(panel, cand, fm, day_from, day_to, n_random, ew_series) -> dict:
    import daily_revenue as DR
    r = MD.simulate(panel, cand, fm, day_from, day_to, ucfg(), start_equity=START_EQUITY)
    em = dict(ew_series)
    diffs = [(d, v - em[d]) for d, v in r["series"] if d in em]
    rand = [MD.simulate(panel, cand, fm, day_from, day_to, ucfg(), start_equity=START_EQUITY,
                        chooser=MD.random_chooser(r["moves"], seed))["agg"].get("mean_pct") for seed in range(n_random)]
    rets = [t["ret_pct"] for t in r["trades"]]
    reasons = {}
    for t in r["trades"]:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"config": cand, "window": [day_from, day_to], "mean_pct": r["agg"].get("mean_pct"), "agg": r["agg"],
            "paired_ew": DR.paired(r["series"], ew_series), "paired_ew_by_year": _year_means(diffs),
            "r_by_year": _year_means(r["series"]), "random_means": rand, "random_p95": _p95(rand),
            "n_trades": len(rets), "trade_mean_pct": float(np.mean(rets)) if rets else None,
            "trade_win_share": float(np.mean([x > 0 for x in rets])) if rets else None, "exit_reasons": reasons,
            "conservation_ok": r["conservation_ok"], "_series": r["series"]}


def run_train_u(n_random=N_RANDOM) -> dict:
    import experiment_xsmom as XE
    import fill_model as FM
    import pbo as PB
    panel = XE.load_panel().cut(U_TEST_FROM)
    fm = FM.model("taker", 1)
    ew = XS.rotate(panel, ucfg(), fm, U_TRAIN_FROM, U_TEST_FROM, chooser=XS.ewu_chooser, start_equity=START_EQUITY)
    btc = XS.rotate(panel, ucfg(), fm, U_TRAIN_FROM, U_TEST_FROM, chooser=XS.btc_chooser, start_equity=START_EQUITY)
    print("U: EW %+.4f %%/day | BTC %+.4f" % (ew["agg"]["mean_pct"], btc["agg"]["mean_pct"]), flush=True)
    res, series, t0 = {}, {}, time.time()
    cands = candidates_u()
    for i, (name, cand) in enumerate(cands.items(), 1):
        e = evaluate_u(panel, cand, fm, U_TRAIN_FROM, U_TEST_FROM, n_random, ew["series"])
        ok, checks = admit_u(e)
        e["admitted"], e["checks"] = ok, checks
        series[name] = dict(e.pop("_series"))
        res[name] = e
        p = e["paired_ew"]
        print("U %d/%d %-11s R %+.4f | vs EW %+.4f [%+.4f, %+.4f] | years %s | rnd p95 %+.4f | trades %d | trade %+.2f%% win %.2f | %s | %.1f min"
              % (i, len(cands), name, e["mean_pct"] or 0, p.get("mean_pct") or 0, p.get("lo95") or 0, p.get("hi95") or 0,
                 {y: round(v, 3) for y, v in e["paired_ew_by_year"].items()}, e["random_p95"] or 0, e["n_trades"],
                 e["trade_mean_pct"] or 0, e["trade_win_share"] or 0, "ADMIT" if ok else "-", (time.time() - t0) / 60),
              flush=True)
    names = list(cands)
    days = sorted(set.intersection(*[set(series[n]) for n in names]))
    R = np.array([[series[n][d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= PBO_SPLITS * 2 else None
    adm = [n for n, v in res.items() if v["admitted"]]
    winner = max(adm, key=lambda n: res[n]["paired_ew"]["mean_pct"]) if (adm and pbo is not None and pbo < PBO_MAX) else None
    return {"ew_agg": ew["agg"], "btc_agg": btc["agg"], "candidates": res, "pbo": pbo, "winner": winner}


# ------------------------------------------------------------------ row B
def run_train_b() -> dict:
    import daily_revenue as DR
    import experiment_x4f as X4F
    R, FM, go, thin_for, trades, grid, share = X4F._setup()
    last_bar = grid.t0 + timedelta(minutes=15 * (grid.N - 1))
    grid_last_day = last_bar.strftime("%Y-%m-%d")
    train_to, test_from, test_to = b_windows(grid_last_day)
    base = go(X4F.BASE, None, train_to)
    print("B: grid ends %s; train entries < %s; X4 R %+.4f %%/day, trades %d"
          % (grid_last_day, train_to, base["agg"]["mean_pct"], len(base["trades"])), flush=True)
    res = {}
    for name, pol in candidates_b().items():
        r = go(pol, None, train_to)
        pr = DR.paired(r["series"], base["series"])
        reasons = {}
        for t in r["trades"]:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        e = {"agg": r["agg"], "paired": pr, "changed": X4F.changed("W", base["trades"], r["trades"]),
             "trades": len(r["trades"]), "exit_reasons": reasons,
             "trade_mean_pct": float(np.mean([t["ret_pct"] for t in r["trades"]])) if r["trades"] else None}
        e["admitted"] = admit_b(e)
        res[name] = e
        print("B %-10s R %+.4f | vs X4 %+.4f [%+.4f, %+.4f] | trades %d changed %d | trade %+.2f%% | exits %s | %s"
              % (name, r["agg"]["mean_pct"], pr["mean_pct"], pr["lo95"], pr["hi95"], e["trades"], e["changed"],
                 e["trade_mean_pct"] or 0, reasons, "ADMIT" if e["admitted"] else "-"), flush=True)
    adm = [n for n, v in res.items() if v["admitted"]]
    winner = max(adm, key=lambda n: res[n]["paired"]["mean_pct"]) if adm else None
    return {"grid_last_day": grid_last_day, "windows": {"train_to": train_to, "test_from": test_from, "test_to": test_to},
            "base_agg": base["agg"], "base_trades": len(base["trades"]), "candidates": res, "winner": winner}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        t0 = time.time()
        b = run_train_b()
        u = run_train_u()
        winners = {k: v for k, v in (("B", b["winner"]), ("U", u["winner"])) if v}
        out = {"spec": SPEC, "stage": "train", "B": b, "U": u, "winners": winners,
               "verdict": "winners -> test needs the operator" if winners else "STOP: no winner, test not opened",
               "minutes": (time.time() - t0) / 60, "utc": datetime.now(timezone.utc).isoformat()}
        TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(json.dumps({"winners": winners, "pbo_U": u["pbo"], "verdict": out["verdict"]}, indent=1))
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("MD: the test half is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("MD: the test runner is written only if train produces a winner (spec §3)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
