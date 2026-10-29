"""DIR -- train of the supervised direction forecast at 4h / 1d (direction-forecast-dir-1029-spec.md §3-§6).

  rows    top-100 liquid spot pairs (xsmom.universe, age >= 60 d) at every decision time 2023-03-01 .. 2025-12-31;
          17 causal features, label = next-period return > 0 (entry/exit at the first 15m close of the period)
  model   CatBoostClassifier(depth 4, 300 it, lr 0.05, seed 29), no tuning; purged walk-forward 4 folds, embargo 2 d
  stats   OOS AUC (day-block bootstrap), top-decile long after 36 bp round trip with turnover-aware costs, paired vs
          the universe, years; alpha = 0.025 per horizon (two horizons)
  test    the 2026 window is opened once and only on the operator's decision (marker DIR_TEST_OPENED.json)

  (from trader/) ../pyembed/python.exe -u experiment_dir.py train
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import direction_dir as D         # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM, TEST_FROM = "2023-03-01", "2026-01-01"
COST_PCT = 0.36                   # 2 x (taker 0.10% + 8 bp), fill_model.model("taker", 1)
ALPHA = 0.025                     # 0.05 / 2 horizons
BOOT = 2000
YEARS_AGREE = 2
H4 = 4 * 3600 * 1000
DAY = 86_400_000
OUT = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT / "DIR_TRAIN.json"
TEST_MARKER = OUT / "DIR_TEST_OPENED.json"
SPEC = "direction-forecast-dir-1029-spec.md"
COLS = list(D.FEATURES) + ["btc_r1", "btc_r6", "mkt_r1", "mkt_r6", "xr1", "xr6", "log_age"]


def candidates() -> dict:
    return {"DIR_4h": {"h": "4h"}, "DIR_1d": {"h": "1d"}}


def admit(res: dict):
    years = res.get("sel_by_year") or {}
    checks = {"auc_lo_gt_05": res.get("auc_lo") is not None and res["auc_lo"] > 0.5,
              "sel_net_lo_gt_0": (res.get("sel") or {}).get("lo95") is not None and res["sel"]["lo95"] > 0,
              "paired_lo_gt_0": (res.get("paired") or {}).get("lo95") is not None and res["paired"]["lo95"] > 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("DIR: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


# ------------------------------------------------------------------ rows
def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class Rows:
    def __init__(self):
        self.tidx, self.sym, self.r, self.X = [], [], [], []

    def add(self, tidx, sym, r, x):
        self.tidx.append(tidx)
        self.sym.append(sym)
        self.r.append(r)
        self.X.append(x)

    def arrays(self):
        return (np.asarray(self.tidx, dtype=np.int64), np.asarray(self.sym), np.asarray(self.r, float),
                np.asarray(self.X, float).reshape(-1, len(D.FEATURES) + 3))


def _cross_section(tidx, X):
    """Adds mkt_r1, mkt_r6 (median over the universe) and xr1, xr6 (percentile rank) per decision time."""
    i1, i6 = D.FEATURES.index("r1"), D.FEATURES.index("r6")
    out = np.full((len(tidx), 4), np.nan)
    order = np.argsort(tidx, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tidx[order][1:] != tidx[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        for col, k in ((0, i1), (1, i6)):
            v = X[idx, k]
            ok = np.isfinite(v)
            if ok.sum():
                out[idx, col] = np.median(v[ok])
            if ok.sum() > 1:
                rk = np.empty(ok.sum())
                rk[np.argsort(v[ok], kind="stable")] = np.arange(ok.sum()) / (ok.sum() - 1)
                tmp = np.full(len(idx), np.nan)
                tmp[ok] = rk
                out[idx, col + 2] = tmp
    return out


def build_1d(P, cfg, unis):
    import xsmom as XS
    first = P.first_bar_rows()
    btc = P.col["BTCUSDT"]
    need = sorted({s for u in unis.values() for s in u})
    feats = {s: D.pair_features(P.close[:, P.col[s]], P.qv[:, P.col[s]]) for s in need + ["BTCUSDT"]}
    rows, times = Rows(), []
    for d in sorted(unis):
        if d + 1 >= P.n:
            continue
        ti = len(times)
        times.append({"t0": _dt((P.d0 + d) * DAY), "t1": _dt((P.d0 + d + 1) * DAY + XS.BAR_MS)})
        i = d - 1
        bt = feats["BTCUSDT"]
        for s in unis[d]:
            j = P.col[s]
            ex = P.first_close[d + 1, j]
            delisted = d + 3 < P.n and not P.has_bar[d + 1:d + 4, j].any()
            r = D.label(float(P.first_close[d, j]), float(ex) if math.isfinite(ex) else None, delisted)
            if r is None:
                continue
            f = feats[s]
            rows.add(ti, s, r, [f[k][i] for k in D.FEATURES] + [bt["r1"][i], bt["r6"][i],
                                                                 math.log1p(max(0, d - int(first[j])))])
    return rows, times


def _blocks_4h(a, b0, nb):
    t = np.asarray(a["t"], dtype=np.int64)
    c = np.asarray(a["c"], dtype=float)
    v = np.asarray(a["v"], dtype=float)
    blk = t // H4 - b0
    ok = (blk >= 0) & (blk < nb)
    t, c, v, blk = t[ok], c[ok], v[ok], blk[ok]
    close, qv, first = (np.full(nb, np.nan) for _ in range(3))
    if len(t):
        cnt = np.bincount(blk, minlength=nb)
        last = np.searchsorted(blk, np.arange(nb), side="right") - 1
        has = cnt > 0
        close[has] = c[last[has]]
        close[cnt < 16] = np.nan
        np.add.at(qv, blk, 0)
        qv = np.where(has, 0.0, np.nan)
        np.add.at(qv, blk, c * v)
        st = (t % H4) == 0
        first[blk[st]] = c[st]
    return close, qv, first, t


def build_4h(P, unis):
    import history_store as HS
    b0 = _ms("2023-01-01") // H4
    nb = _ms(TEST_FROM) // H4 - b0
    first_row = P.first_bar_rows()
    need = sorted({s for u in unis.values() for s in u})
    mem = {s: np.zeros(P.n, dtype=bool) for s in need}
    for d, u in unis.items():
        for s in u:
            mem[s][d] = True
    btc_close, btc_qv, _, _ = _blocks_4h(HS.load(HS.path_of("BTCUSDT")), b0, nb)
    bt = D.pair_features(btc_close, btc_qv)
    ks = np.arange(nb - 2)                                    # decision at close of block k; exit block k+2 < TEST_FROM
    entry_ms = (b0 + ks + 1) * H4
    day_row = entry_ms // DAY - P.d0
    in_win = (day_row >= P.row(TRAIN_FROM)) & (day_row < P.n)
    times = [{"t0": _dt(int((b0 + k + 1) * H4)), "t1": _dt(int((b0 + k + 2) * H4 + 15 * 60 * 1000))} for k in ks]
    rows = Rows()
    test_ms = _ms(TEST_FROM)
    for n_s, s in enumerate(need, 1):
        a = HS.load(HS.path_of(s))
        if a is None:
            continue
        close, qv, first, t15 = _blocks_4h(a, b0, nb)
        f = D.pair_features(close, qv)
        j = P.col[s]
        m = in_win.copy()
        m[in_win] &= mem[s][day_row[in_win]]
        for k in np.flatnonzero(m):
            en, ex = first[k + 1], first[k + 2]
            if not math.isfinite(en):
                continue
            em = int(entry_ms[k])
            nxt = int(np.searchsorted(t15, em, side="right"))
            delisted = em + 3 * DAY < test_ms and (nxt >= len(t15) or int(t15[nxt]) > em + 3 * DAY)
            r = D.label(float(en), float(ex) if math.isfinite(ex) else None, delisted)
            if r is None:
                continue
            rows.add(int(k), s, r, [f[x][k] for x in D.FEATURES] + [bt["r1"][k], bt["r6"][k],
                                                                    math.log1p(max(0, int(day_row[k]) - int(first_row[j])))])
        if n_s % 50 == 0:
            print(f"  4h rows: {n_s}/{len(need)} pairs, {len(rows.r)} rows", flush=True)
    return rows, times


# ------------------------------------------------------------------ fit + stats
def fit_oos(tidx, X, y, times):
    from catboost import CatBoostClassifier
    uniq = np.unique(tidx)                                    # only decision times that carry rows
    fs = [(uniq[tr], uniq[te]) for tr, te in D.folds([times[int(t)] for t in uniq])]
    p = np.full(len(y), np.nan)
    for n_f, (tr, te) in enumerate(fs, 1):
        trm, tem = np.isin(tidx, tr), np.isin(tidx, te)
        m = CatBoostClassifier(depth=4, iterations=300, learning_rate=0.05, random_seed=29, verbose=False,
                               thread_count=4, loss_function="Logloss")
        m.fit(X[trm], y[trm])
        p[tem] = m.predict_proba(X[tem])[:, 1]
        print(f"  fold {n_f}: train {int(trm.sum())} rows, test {int(tem.sum())} rows, "
              f"{times[min(te)]['t0']:%Y-%m-%d}..{times[max(te)]['t0']:%Y-%m-%d}", flush=True)
    return p


def evaluate(tidx, sym, r, p, times):
    import daily_revenue as DR
    ok = np.isfinite(p)
    tidx, sym, r, p = tidx[ok], sym[ok], r[ok], p[ok]
    y = r > 0
    day = np.array([times[t]["t0"].date().toordinal() for t in tidx])
    a, lo, hi = D.auc_ci(y, p, day, alpha=ALPHA, seed=29, boot=BOOT)
    order = np.argsort(tidx, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tidx[order][1:] != tidx[order][:-1], True])
    groups = [order[x:z] for x, z in zip(bounds[:-1], bounds[1:])]
    sels = [D.select({str(sym[i]): float(p[i]) for i in g}) for g in groups]
    costs = D.turnover_costs(sels, COST_PCT)
    by_day, pair_day, ew_day = {}, {}, {}
    hits, absr, n_sel = 0, [], 0
    for g, sel, cs in zip(groups, sels, costs):
        dkey = times[int(tidx[g[0]])]["t0"].strftime("%Y-%m-%d")
        rr = {str(sym[i]): float(r[i]) * 100 for i in g}
        all_mean = float(np.mean(list(rr.values())))
        ew_day.setdefault(dkey, []).append(all_mean)
        if not sel:
            continue
        rs = [rr[s] for s in sel]
        by_day.setdefault(dkey, []).append(float(np.mean(rs)) - sum(cs.values()) / len(sel))
        pair_day.setdefault(dkey, []).append(float(np.mean(rs)) - all_mean)
        hits += sum(1 for v in rs if v > 0)
        absr += [abs(v) for v in rs]
        n_sel += len(rs)
    ser = sorted((d, float(np.mean(v))) for d, v in by_day.items())
    pser = sorted((d, float(np.mean(v))) for d, v in pair_day.items())
    yr = {}
    for d, v in ser:
        yr.setdefault(d[:4], []).append(v)
    m_abs = float(np.mean(absr)) if absr else None
    return {"n_rows": int(len(r)), "n_times": len(groups), "base_up": float(y.mean()),
            "accuracy": float(((p > 0.5) == y).mean()), "auc": a, "auc_lo": lo, "auc_hi": hi,
            "sel": DR.aggregate(ser, alpha=ALPHA), "paired": DR.aggregate(pser, alpha=ALPHA),
            "ew_universe_gross": DR.aggregate(sorted((d, float(np.mean(v))) for d, v in ew_day.items()), alpha=ALPHA),
            "sel_by_year": {k: float(np.mean(v)) for k, v in sorted(yr.items())},
            "sel_hit_rate": hits / n_sel if n_sel else None, "sel_mean_abs_pct": m_abs,
            "sel_breakeven_p": 0.5 + COST_PCT / (2 * m_abs) if m_abs else None, "n_selected": n_sel,
            "turnover_cost_per_period_pct": float(np.mean([sum(c.values()) / len(c) for c in costs if c])) if n_sel else None}


def run_train() -> dict:
    import experiment_xsmom as EX
    import xsmom as XS
    t0 = time.time()
    P = EX.load_panel().cut(TEST_FROM)
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=60)
    unis = {}
    for d in range(P.row(TRAIN_FROM), P.n):
        u = XS.universe(P, XS.day_str(P.d0 + d), cfg)
        if u:
            unis[d] = u
    print(f"panel {P.n} days x {len(P.syms)} pairs; universe days {len(unis)}, union "
          f"{len({s for u in unis.values() for s in u})} pairs; {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "cost_pct": COST_PCT, "alpha": ALPHA,
           "features": COLS, "candidates": {}}
    for name, cd in candidates().items():
        rows, times = build_1d(P, cfg, unis) if cd["h"] == "1d" else build_4h(P, unis)
        tidx, sym, r, Xb = rows.arrays()
        X = np.column_stack([Xb[:, :len(D.FEATURES) + 2], _cross_section(tidx, Xb), Xb[:, -1:]])
        print(f"{name}: {len(r)} rows, {len(set(tidx.tolist()))} decision times, up share {np.mean(r > 0):.3f}; "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        p = fit_oos(tidx, X, (r > 0).astype(int), times)
        res = evaluate(tidx, sym, r, p, times)
        ok, checks = admit(res)
        res["admitted"], res["checks"] = ok, checks
        out["candidates"][name] = res
        s, q = res["sel"], res["paired"]
        print(f"{name}: AUC {res['auc']:.4f} [{res['auc_lo']:.4f}, {res['auc_hi']:.4f}] acc {res['accuracy']:.4f} "
              f"base {res['base_up']:.4f} | top-decile net {s['mean_pct']:+.4f} [{s['lo95']:+.4f}, {s['hi95']:+.4f}] %/period "
              f"| vs universe {q['mean_pct']:+.4f} [{q['lo95']:+.4f}, {q['hi95']:+.4f}] | EW gross "
              f"{res['ew_universe_gross']['mean_pct']:+.4f} | hit {res['sel_hit_rate']:.4f} vs p* {res['sel_breakeven_p']:.4f} "
              f"| cost/period {res['turnover_cost_per_period_pct']:.3f}% | years "
              f"{ {k: round(v, 4) for k, v in res['sel_by_year'].items()} } | {'ADMIT' if ok else '-'} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    adm = [n for n, v in out["candidates"].items() if v["admitted"]]
    out["admitted"] = adm
    out["verdict"] = "stage 2 needs its own spec" if adm else "STOP: no horizon admitted, test not opened"
    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat()
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"admitted": adm, "verdict": out["verdict"]}))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        run_train()
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("DIR: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("DIR: the test runner is written only after stage 2 (spec §6)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
