"""CAP -- market-wide capitulation basket (capitulation-basket-cap-1104-spec.md).

  plan   decision window 2018-03-01..2022-12-31: event counts per candidate and the MDE from random-bar baskets,
         BEFORE any event return is read -> runtime/experiments/CAP_PLAN.json
  test   the decision: §4 on 2018-2022, plus the same numbers on 2023-03..2025-12 as the exploratory origin of the idea
         -> runtime/experiments/CAP_TEST.json

  look2026  operator 2026-11-04: ONE look at 2026 for the frozen CAP_W4_X50_H24 only, alpha 0.05, marker
            runtime/experiments/CAP_2026_OPENED -> runtime/experiments/CAP_2026.json

  (from trader/) ../pyembed/python.exe -u experiment_cap.py plan | test | look2026
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

import cap                        # noqa: E402
import trader_config as TC        # noqa: E402

BH_WIN = ("2018-03-01", "2023-01-01")
ORIGIN_WIN = ("2023-03-01", "2026-01-01")
ALPHA = 0.05 / 8
N_RANDOM = 200
MIN_EVENTS = 10
PERIODS_AGREE = 2
PBO_MAX = 0.5
PLAN_OUT = TC.RUNTIME / "experiments" / "CAP_PLAN.json"
TEST_OUT = TC.RUNTIME / "experiments" / "CAP_TEST.json"
SPEC = "capitulation-basket-cap-1104-spec.md"
RULE_2026 = "CAP_W4_X50_H24"
MARKER_2026 = TC.RUNTIME / "experiments" / "CAP_2026_OPENED"
OUT_2026 = TC.RUNTIME / "experiments" / "CAP_2026.json"
MIN_EVENTS_2026 = 5
Z = {0.05 / 8: 2.7344}           # two-sided normal quantile at 1 - alpha/2


def candidates() -> dict:
    out = {}
    for wh, W, Y in ((1, 4, 0.05), (4, 16, 0.10)):
        for X in (25, 50):
            for Hh, H in ((24, 96), (72, 288)):
                out[f"CAP_W{wh}_X{X}_H{Hh}"] = {"W": W, "Y": Y, "X": X / 100, "H": H}
    return out


def period_of(year: str) -> str:
    return "2018-2019" if year <= "2019" else ("2020-2021" if year <= "2021" else ("2022" if year == "2022" else year))


def admit(res: dict):
    a = res.get("agg") or {}
    per = res.get("by_period") or {}
    checks = {"mean_lo_gt_0": a.get("lo95") is not None and a["lo95"] > 0,
              "beats_random_p95": res.get("rand_p95") is not None and a.get("mean_pct") is not None
              and a["mean_pct"] > res["rand_p95"],
              "periods_agree": sum(1 for v in per.values() if v is not None and v >= 0) >= PERIODS_AGREE,
              "enough_events": (res.get("n_events") or 0) >= MIN_EVENTS}
    return all(checks.values()), checks


def _ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _year(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y")


# ------------------------------------------------------------------ data
def load_window(kind: str, lo: str, hi: str, cut: str | None = "2026-01-01"):
    """(grid, pairs, uni) for one window; pairs only for symbols that are in some day's universe."""
    import xsmom as XS
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=60)
    if kind == "bh":
        import archive_bh as A
        P = A.load_panel()
        load = lambda s: A._load(A.STORE / f"{s}_15m.npz", ("t", "c")) if (A.STORE / f"{s}_15m.npz").exists() else None
    else:
        import experiment_xsmom as EX
        import history_store as HS
        P = EX.load_panel()
        P = P.cut(cut) if cut else P
        load = lambda s: HS.load(HS.path_of(s))
    uni = {}
    for r in range(max(P.row(lo), 0), min(P.row(hi), P.n)):
        day = XS.day_str(P.d0 + r)
        u = XS.universe(P, day, cfg)
        if u:
            uni[XS.day_num(day)] = set(u)
    syms = sorted(set().union(*uni.values())) if uni else []
    lo_ms, hi_ms = _ms(lo), _ms(hi)
    pairs = {}
    for s in syms:
        a = load(s)
        if a is None or not len(a["t"]):
            continue
        t = np.asarray(a["t"], dtype=np.int64)
        m = (t >= lo_ms - 2 * cap.DAY) & (t < hi_ms + 4 * cap.DAY)       # hold may run past the window edge
        pairs[s] = (t[m], np.asarray(a["c"], dtype=np.float32)[m])
    grid = np.arange(lo_ms, hi_ms, cap.BAR, dtype=np.int64)
    return grid, pairs, uni


def _syms_at(grid, uni, i):
    return sorted(uni.get(int(grid[i] // cap.DAY), ()))


def baskets(grid, idx, hold, pairs, uni, fm) -> list:
    out = []
    for i in idx:
        b = cap.basket(grid, i, hold, pairs, _syms_at(grid, uni, i), fm)
        if b is not None:
            out.append(b)
    return out


def breadths(grid, pairs, uni) -> dict:
    out = {}
    for c in candidates().values():
        if (c["W"], c["Y"]) not in out:
            out[(c["W"], c["Y"])] = cap.breadth(grid, pairs, uni, c["W"], c["Y"])
    return out


# ------------------------------------------------------------------ plan (no event return is read)
def run_plan() -> dict:
    import fill_model as FM
    import lq
    t0 = time.time()
    fm = FM.model("taker", 1)
    grid, pairs, uni = load_window("bh", *BH_WIN)
    br = breadths(grid, pairs, uni)
    res = {}
    for name, c in candidates().items():
        ev = cap.events(br[(c["W"], c["Y"])], c["X"], c["H"])
        rnd = baskets(grid, lq.random_bars(grid, list(range(0, len(grid) - c["H"] - 2, 96 * 7)), 260, 11),
                      c["H"], pairs, uni, fm)                           # ~one random bar a week: the spread only
        sd = float(np.std([b["ret_pct"] for b in rnd], ddof=1)) if len(rnd) > 2 else None
        n = len(ev)
        res[name] = {"n_events": n, "random_basket_sd_pct": sd,
                     "mde_pct": round(Z[ALPHA] * sd / np.sqrt(n), 3) if (sd and n) else None,
                     "events_by_period": {p: sum(1 for i in ev if period_of(_year(int(grid[i]))) == p)
                                          for p in ("2018-2019", "2020-2021", "2022")}}
        print(f"{name}: events {n} {res[name]['events_by_period']} | random basket sd {sd} | MDE {res[name]['mde_pct']}%",
              flush=True)
    out = {"spec": SPEC, "window": BH_WIN, "pairs": len(pairs), "days_with_universe": len(uni), "alpha": ALPHA,
           "candidates": res, "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    PLAN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ test
def evaluate(kind, lo, hi, fm, n_random=N_RANDOM, log=print) -> dict:
    import daily_revenue as DR
    import lq
    import pbo as PB
    grid, pairs, uni = load_window(kind, lo, hi)
    br = breadths(grid, pairs, uni)
    res, monthly = {}, {}
    for name, c in candidates().items():
        ev = cap.events(br[(c["W"], c["Y"])], c["X"], c["H"])
        bs = baskets(grid, ev, c["H"], pairs, uni, fm)
        rets = [b["ret_pct"] for b in bs]
        rmeans = []
        for s in range(n_random):
            rb = baskets(grid, lq.random_bars(grid, ev, len(ev), 9000 + s), c["H"], pairs, uni, fm)
            if rb:
                rmeans.append(float(np.mean([b["ret_pct"] for b in rb])))
        by_p, by_m, daily = {}, {}, {}
        for b in bs:
            y = _year(b["entry_close_ms"])
            by_p.setdefault(period_of(y), []).append(b["ret_pct"])
            mo = datetime.fromtimestamp(b["entry_close_ms"] / 1000, timezone.utc).strftime("%Y-%m")
            by_m[mo] = (1 + by_m.get(mo, 0.0) / 100) * (1 + b["ret_pct"] / 100) * 100 - 100
            d = datetime.fromtimestamp(b["exit_close_ms"] / 1000, timezone.utc).strftime("%Y-%m-%d")
            daily[d] = daily.get(d, 0.0) + b["ret_pct"]
        days = []
        d = datetime.strptime(lo, "%Y-%m-%d")
        while d.strftime("%Y-%m-%d") < hi:
            days.append(d.strftime("%Y-%m-%d"))
            d = d.fromordinal(d.toordinal() + 1)
        book = DR.aggregate([(x, daily.get(x, 0.0)) for x in days], alpha=ALPHA)
        r = {"config": c, "n_events": len(bs), "agg": cap.mean_ci(rets, alpha=ALPHA),
             "rand_p95": float(np.percentile(rmeans, 95)) if rmeans else None,
             "rand_mean": float(np.mean(rmeans)) if rmeans else None,
             "by_period": {p: float(np.mean(v)) for p, v in sorted(by_p.items())},
             "n_by_period": {p: len(v) for p, v in sorted(by_p.items())},
             "mean_pairs_in_basket": float(np.mean([b["n_pairs"] for b in bs])) if bs else None,
             "days_in_market_share": round(len(bs) * c["H"] / 96 / max(1, len(days)), 4),
             "book_R": book, "worst_event_pct": min(rets) if rets else None, "best_event_pct": max(rets) if rets else None,
             "book_marked_agg": DR.aggregate(cap.book_marked(bs, days), alpha=ALPHA)}      # protocol-fixes P5
        ok, checks = admit(r)
        r["admitted"], r["checks"] = ok, checks
        res[name], monthly[name] = r, by_m
        a = r["agg"]
        log(f"[{kind}] {name}: events {r['n_events']} | mean {a.get('mean_pct')} [{a.get('lo95')}, {a.get('hi95')}] "
            f"| random p95 {r['rand_p95']} mean {r['rand_mean']} | periods { {k: round(v, 2) for k, v in r['by_period'].items()} } "
            f"| book {book.get('mean_pct')} %/day dd {book.get('max_dd_pct')} | {'ADMIT' if ok else '-'}")
    months = sorted(set().union(*[set(v) for v in monthly.values()]))
    R = np.array([[monthly[n].get(m, 0.0) for n in res] for m in months]) if months else np.zeros((0, len(res)))
    pbo = float(PB.cscv_pbo(R, 8)) if len(months) >= 16 else None
    return {"window": [lo, hi], "pairs": len(pairs), "candidates": res, "pbo": pbo}


def run_test() -> dict:
    import fill_model as FM
    if not PLAN_OUT.exists():
        raise SystemExit("plan first: the MDE goes into the journal before the test")
    t0 = time.time()
    fm = FM.model("taker", 1)
    log = lambda s: print(s, flush=True)
    bh = evaluate("bh", *BH_WIN, fm, log=log)
    adm = [n for n, r in bh["candidates"].items() if r["admitted"]]
    winner = (max(adm, key=lambda n: bh["candidates"][n]["agg"]["mean_pct"])
              if adm and bh["pbo"] is not None and bh["pbo"] < PBO_MAX else None)
    origin = evaluate("live", *ORIGIN_WIN, fm, log=log)
    out = {"spec": SPEC, "alpha": ALPHA, "decision_window": bh, "origin_exploratory": origin, "winner": winner,
           "verdict": "winner -> 2026 by the operator" if winner else "STOP: no winner",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    TEST_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pbo": bh["pbo"], "winner": winner, "verdict": out["verdict"]}))
    return out


# ------------------------------------------------------------------ one look at 2026 (operator 2026-11-04)
def verdict_2026(agg: dict, rand_p95) -> str:
    if (agg.get("n") or 0) < MIN_EVENTS_2026:
        return "TOO_FEW"
    if agg.get("lo95") is not None and agg["lo95"] > 0 and rand_p95 is not None and agg["mean_pct"] > rand_p95:
        return "CONFIRMED_2026"
    return "NOT_SHOWN_2026"


def open_2026(marker=MARKER_2026) -> None:
    marker = Path(marker)
    if marker.exists():
        raise SystemExit("CAP 2026 already looked at once: %s" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def run_2026() -> dict:
    import daily_revenue as DR
    import experiment_xsmom as EX
    import fill_model as FM
    import lq
    import xsmom as XS
    P = EX.load_panel()
    last = XS.day_str(P.d0 + P.n - 1)                      # last full day of the panel
    lo, hi = "2026-01-01", last                            # events end a day before: the 24 h hold must finish
    open_2026()
    t0 = time.time()
    fm = FM.model("taker", 1)
    c = candidates()[RULE_2026]
    grid, pairs, uni = load_window("live", lo, hi, cut=None)
    share = cap.breadth(grid, pairs, uni, c["W"], c["Y"])
    ev = [i for i in cap.events(share, c["X"], c["H"]) if i + 1 + c["H"] < len(grid)]
    bs = baskets(grid, ev, c["H"], pairs, uni, fm)
    rets = [b["ret_pct"] for b in bs]
    rmeans = []
    for sd in range(N_RANDOM):
        rb = baskets(grid, [i for i in lq.random_bars(grid, ev, len(ev), 9000 + sd) if i + 1 + c["H"] < len(grid)],
                     c["H"], pairs, uni, fm)
        if rb:
            rmeans.append(float(np.mean([b["ret_pct"] for b in rb])))
    agg = cap.mean_ci(rets, alpha=0.05)
    p95 = float(np.percentile(rmeans, 95)) if rmeans else None
    daily = {}
    for b in bs:
        d = datetime.fromtimestamp(b["exit_close_ms"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        daily[d] = daily.get(d, 0.0) + b["ret_pct"]
    days = [XS.day_str(n) for n in range(XS.day_num(lo), XS.day_num(hi))]
    out = {"spec": SPEC, "rule": RULE_2026, "config": c, "window": [lo, hi], "alpha": 0.05, "pairs": len(pairs),
           "events": [{"entry": datetime.fromtimestamp(b["entry_close_ms"] / 1000, timezone.utc).isoformat(),
                       "ret_pct": round(b["ret_pct"], 3), "n_pairs": b["n_pairs"]} for b in bs],
           "agg": agg, "rand_p95": p95, "rand_mean": float(np.mean(rmeans)) if rmeans else None,
           "book_R": DR.aggregate([(d, daily.get(d, 0.0)) for d in days]),
           "book_marked_agg": DR.aggregate(cap.book_marked(bs, days)), "verdict": verdict_2026(agg, p95),
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    OUT_2026.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("window", "agg", "rand_p95", "verdict")}, default=str))
    for e in out["events"]:
        print(e)
    return out

def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["plan"]:
        run_plan()
    elif argv[:1] == ["test"]:
        run_test()
    elif argv[:1] == ["look2026"]:
        run_2026()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
