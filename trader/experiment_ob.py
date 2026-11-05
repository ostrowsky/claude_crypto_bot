"""OB step 1 -- does futures order-book imbalance carry information? (orderbook-depth-ob-1105-spec.md §3-§4)

  features   6 imbalance features per hour close H (bookdepth hourly summaries)
  targets    perp log return H -> H + 4 h / H + 24 h minus the cross-sectional mean of the hour
  stats      rank IC across pairs per moment (4 h grid / 00:00 UTC), partial IC net of the past return of the same
             horizon, 5-day block bootstrap of the daily mean IC at alpha 0.05 / 12, halves, top-bottom quintile
  run        window 2023-03-01..2025-12-31 (H + 24 h stays inside it) -> runtime/experiments/OB_STEP1.json

  (from trader/) ../pyembed/python.exe -u experiment_ob.py run
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

import bookdepth as B             # noqa: E402
import trader_config as TC        # noqa: E402

H = B.HOUR
FEATURES = ("imb_02", "imb_1", "imb_2", "imb_5", "imb_1_mean1h", "d_imb_1_4h")
TARGETS = {"ret_4h": 4, "ret_24h": 24}
WINDOW = ("2023-03-01", "2026-01-01")
HALF_SPLIT = "2024-08-01"
ALPHA = 0.05 / 12
MIN_N = 10
COST_PCT = 0.36
OUT = TC.RUNTIME / "experiments" / "OB_STEP1.json"
SPEC = "orderbook-depth-ob-1105-spec.md"


def features(h: dict) -> dict:
    t = np.asarray(h["t"], dtype=np.int64)
    lv = {k: B.LEVELS.index(k) for k in (0.2, 1, 2, 5)}
    i1 = B.imb(h["bid"][:, lv[1]], h["ask"][:, lv[1]])
    back = t - 4 * H
    j = np.searchsorted(t, back)
    jc = np.minimum(j, len(t) - 1)
    ok = (j < len(t)) & (t[jc] == back)
    d4 = np.full(len(t), np.nan)
    d4[ok] = i1[ok] - i1[jc[ok]]
    return {"imb_02": B.imb(h["bid"][:, lv[0.2]], h["ask"][:, lv[0.2]]), "imb_1": i1,
            "imb_2": B.imb(h["bid"][:, lv[2]], h["ask"][:, lv[2]]), "imb_5": B.imb(h["bid"][:, lv[5]], h["ask"][:, lv[5]]),
            "imb_1_mean1h": np.asarray(h["imb1_mean"], dtype=float), "d_imb_1_4h": d4}


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def ic(f, y, min_n: int = MIN_N):
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    a, b = _rank(f[m]), _rank(y[m])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def partial_ic(f, past, y, min_n: int = MIN_N):
    f, past, y = (np.asarray(v, dtype=float) for v in (f, past, y))
    m = np.isfinite(f) & np.isfinite(past) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    X = np.column_stack([np.ones(m.sum()), past[m]])
    beta, *_ = np.linalg.lstsq(X, f[m], rcond=None)
    res = f[m] - X @ beta
    if res.std() < 1e-12 * max(f[m].std(), 1e-300):
        return 0.0                                                 # the feature is fully explained by the past return
    out = np.full(len(f), np.nan)
    out[m] = res
    return ic(out, y, min_n)


def quintile_spread(f, y):
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < 5:
        return None
    o = np.argsort(f[m], kind="stable")
    q = max(1, int(m.sum() // 5))
    yy = y[m][o]
    return float(yy[-q:].mean() - yy[:q].mean())


def moment_hours(t_ms, step_h: int) -> np.ndarray:
    return (np.asarray(t_ms, dtype=np.int64) // H) % step_h == 0


def admit(res: dict):
    checks = {"ic_lo_gt_0": (res.get("ic") or {}).get("lo95") is not None and res["ic"]["lo95"] > 0,
              "partial_lo_gt_0": (res.get("partial") or {}).get("lo95") is not None and res["partial"]["lo95"] > 0,
              "both_halves_positive": len(res.get("halves") or []) == 2 and all(v is not None and v > 0 for v in res["halves"]),
              "spread_gt_cost": res.get("spread_pct") is not None and res["spread_pct"] > COST_PCT}
    return all(checks.values()), checks


# ------------------------------------------------------------------ run
def _ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def build(log=print):
    import funding_store as FS
    import history_store as HS
    pairs = json.loads(B.PAIRS.read_text(encoding="utf-8"))
    lo, hi = _ms(WINDOW[0]), _ms(WINDOW[1])
    grid = np.arange(lo, hi - 24 * H + 1, H, dtype=np.int64)          # H + 24 h stays inside the window
    P = len(pairs["perp"])
    F = {k: np.full((len(grid), P), np.nan) for k in FEATURES}
    C = np.full((len(grid) + 25, P), np.nan)                            # perp close at grid hours (+ 24 h ahead)
    Cb = np.full((len(grid), P), np.nan)                                # close 24 h before (for the past return)
    tgrid_ext = np.arange(lo, lo + (len(grid) + 25) * H, H, dtype=np.int64)
    for j, (spot, perp) in enumerate(zip(pairs["spot"], pairs["perp"])):
        h = B.load(perp)
        if h is not None and len(h["t"]):
            f = features(h)
            pos = np.searchsorted(grid, h["t"])
            ok = (pos < len(grid)) & (grid[np.minimum(pos, len(grid) - 1)] == h["t"])
            for k in FEATURES:
                F[k][pos[ok], j] = f[k][ok]
        a = HS.load(FS.perp_path(spot))
        if a is not None and len(a["t"]):
            tc = np.asarray(a["t"], dtype=np.int64) + 15 * 60 * 1000       # bar close time
            c = np.asarray(a["c"], dtype=float)
            for arr, tg in ((C, tgrid_ext), (Cb, grid - 24 * H)):
                p = np.searchsorted(tc, tg)
                okc = (p < len(tc)) & (tc[np.minimum(p, len(tc) - 1)] == tg)
                arr[np.flatnonzero(okc), j] = c[p[okc]]
        log(f"  {perp}: depth hours {0 if h is None else len(h['t'])}")
    return grid, F, C, Cb, pairs


def run() -> dict:
    import daily_revenue as DR
    t0 = time.time()
    grid, F, C, Cb, pairs = build(lambda s: print(s, flush=True))
    n = len(grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(C)
        lcb = np.log(Cb)
    res = {}
    split = _ms(HALF_SPLIT)
    for tname, hz in TARGETS.items():
        y = lc[hz:hz + n] - lc[:n]
        y = y - np.nanmean(y, axis=1, keepdims=True)
        if hz == 4:
            past = np.full((n, y.shape[1]), np.nan)
            past[4:] = lc[4:n] - lc[:n - 4]
        else:
            past = lc[:n] - lcb
        mom = np.flatnonzero(moment_hours(grid, hz))
        for fname in FEATURES:
            f = F[fname]
            rows = []
            for i in mom:
                a = ic(f[i], y[i])
                if a is None:
                    continue
                rows.append((int(grid[i]), a, partial_ic(f[i], past[i], y[i]), quintile_spread(f[i], y[i]),
                             int(np.isfinite(f[i]).sum())))
            by_day, by_day_p = {}, {}
            for t, a, p, _, _ in rows:
                d = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")
                by_day.setdefault(d, []).append(a)
                if p is not None:
                    by_day_p.setdefault(d, []).append(p)
            agg = DR.aggregate(sorted((d, float(np.mean(v))) for d, v in by_day.items()), alpha=ALPHA)
            aggp = DR.aggregate(sorted((d, float(np.mean(v))) for d, v in by_day_p.items()), alpha=ALPHA)
            h1 = [a for t, a, *_ in rows if t < split]
            h2 = [a for t, a, *_ in rows if t >= split]
            sp = [s for *_, s, _ in rows if s is not None]
            r = {"n_moments": len(rows), "mean_pairs": float(np.mean([x[4] for x in rows])) if rows else None,
                 "ic": {"mean": agg.get("mean_pct"), "lo95": agg.get("lo95"), "hi95": agg.get("hi95")},
                 "partial": {"mean": aggp.get("mean_pct"), "lo95": aggp.get("lo95"), "hi95": aggp.get("hi95")},
                 "halves": [float(np.mean(h1)) if h1 else None, float(np.mean(h2)) if h2 else None],
                 "spread_pct": float(np.mean(sp)) * 100 if sp else None}
            ok, checks = admit(r)
            r["passed"], r["checks"] = ok, checks
            res[f"{fname}|{tname}"] = r
            print(f"{fname:13s} {tname:7s} IC {r['ic']['mean']} [{r['ic']['lo95']}, {r['ic']['hi95']}] | partial "
                  f"{r['partial']['mean']} [{r['partial']['lo95']}, {r['partial']['hi95']}] | halves "
                  f"{[None if v is None else round(v, 4) for v in r['halves']]} | spread {r['spread_pct']} % | "
                  f"moments {r['n_moments']} pairs {r['mean_pairs']} | {'PASS' if ok else '-'}", flush=True)
    passed = [k for k, v in res.items() if v["passed"]]
    out = {"spec": SPEC, "window": WINDOW, "alpha": ALPHA, "pairs": pairs["perp"], "results": res, "passed": passed,
           "verdict": "step 2 (money book, separate pre-registration)" if passed else "STOP: no information",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"passed": passed, "verdict": out["verdict"]}))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        run()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
