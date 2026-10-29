"""DPOS -- derivatives positioning added to the 4h / 1d direction forecast (dir-positioning-dpos-1029-spec.md §3-§4).

  pos_features(_many)   from the futures metrics archive: a row with create_time T is known at T + 5 min; the last
                        known row must be <= 30 min old, else every feature is NaN
  funding_features      last rate <= t, mean rate of events in (t - 3 d, t]
  paired_auc_ci         AUC(B) - AUC(A) on the same rows, moving-block bootstrap over decision days (binned scores)
  pit                   archive vs the bot's live positioning snapshots 2026-08-21..10-28 (data check only)
  train                 the DIR rows (experiment_dir) restricted to rows with metrics; A = 17 DIR features,
                        B = 17 + 10 positioning; same purged folds, statistics of DIR §5; outcome per §4

  (from trader/) ../pyembed/python.exe -u experiment_dpos.py pit
  (from trader/) ../pyembed/python.exe -u experiment_dpos.py train
"""
from __future__ import annotations

import json
import math
import random
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

MIN5 = 5 * 60 * 1000
H1 = 3600 * 1000
H4 = 4 * H1
DAY = 24 * H1
LAG = MIN5                         # a row is known 5 min after its create_time
STALE = 30 * 60 * 1000
POS_KEYS = ("oi_c1", "oi_c6", "oi_qv", "ls_top", "ls_all", "tk")
EXTRA = POS_KEYS + ("fund_last", "fund_3d", "x_oi_c1", "x_fund")
PIT_SPEARMAN_MIN = 0.9
PIT_TOP_REL_MAX = 0.02
YEARS_AGREE = 2
OUT = TC.RUNTIME / "experiments"
PIT_OUT = OUT / "DPOS_PIT.json"
TRAIN_OUT = OUT / "DPOS_TRAIN.json"
TEST_MARKER = OUT / "DPOS_TEST_OPENED.json"
SPEC = "dir-positioning-dpos-1029-spec.md"


# ------------------------------------------------------------------ features
def pos_features_many(m, ts, h_ms, qv_med) -> dict:
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    out = {k: np.full(n, np.nan) for k in POS_KEYS}
    if m is None or len(m["t"]) == 0 or n == 0:
        return out
    av = np.asarray(m["t"], dtype=np.int64) + LAG
    oi, oiv = np.asarray(m["oi"], float), np.asarray(m["oi_val"], float)
    tk = np.asarray(m["taker"], float)

    def last(t):
        i = np.searchsorted(av, t, side="right") - 1
        ok = (i >= 0) & (av[np.clip(i, 0, None)] >= t - STALE)
        return i, ok

    i, ok = last(ts)
    j1, ok1 = last(ts - h_ms)
    j6, ok6 = last(ts - 6 * h_ms)
    ic, j1c, j6c = np.clip(i, 0, None), np.clip(j1, 0, None), np.clip(j6, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["oi_c1"] = np.where(ok & ok1, np.log(oi[ic] / oi[j1c]), np.nan)
        out["oi_c6"] = np.where(ok & ok6, np.log(oi[ic] / oi[j6c]), np.nan)
        q = np.asarray(qv_med, float) * np.ones(n)
        out["oi_qv"] = np.where(ok & (q > 0), np.log(oiv[ic] / q), np.nan)
        C = np.r_[0.0, np.cumsum(np.nan_to_num(tk))]
        N = np.r_[0, np.cumsum(np.isfinite(tk))]
        cnt = N[i + 1] - N[j1 + 1]
        out["tk"] = np.where(ok & (cnt > 0), (C[i + 1] - C[j1 + 1]) / np.maximum(cnt, 1), np.nan)
    out["ls_top"] = np.where(ok, np.asarray(m["top_pos"], float)[ic], np.nan)
    out["ls_all"] = np.where(ok, np.asarray(m["ls_all"], float)[ic], np.nan)
    for k in POS_KEYS:
        out[k] = np.where(ok & np.isfinite(out[k]), out[k], np.nan)
    return out


def pos_features(m, t_ms, h_ms, spot_qv_med) -> dict:
    f = pos_features_many(m, [t_ms], h_ms, [spot_qv_med])
    return {k: float(v[0]) for k, v in f.items()}


def funding_features_many(f, ts) -> dict:
    ts = np.asarray(ts, dtype=np.int64)
    out = {"fund_last": np.full(len(ts), np.nan), "fund_3d": np.full(len(ts), np.nan)}
    if f is None or len(f["t"]) == 0:
        return out
    o = np.argsort(np.asarray(f["t"]), kind="stable")
    ft, fr = np.asarray(f["t"], dtype=np.int64)[o], np.asarray(f["rate"], float)[o]
    hi = np.searchsorted(ft, ts, side="right")
    lo = np.searchsorted(ft, ts - 3 * DAY, side="right")
    C = np.r_[0.0, np.cumsum(fr)]
    out["fund_last"] = np.where(hi > 0, fr[np.clip(hi - 1, 0, None)], np.nan)
    out["fund_3d"] = np.where(hi > lo, (C[hi] - C[lo]) / np.maximum(hi - lo, 1), np.nan)
    return out


def funding_features(f, t_ms) -> dict:
    r = funding_features_many(f, [t_ms])
    return {k: float(v[0]) for k, v in r.items()}


def has_pos(P) -> np.ndarray:
    return np.isfinite(np.asarray(P, float)).any(axis=1)


def xrank(tidx, v) -> np.ndarray:
    out = np.full(len(v), np.nan)
    order = np.argsort(tidx, kind="stable")
    b = np.flatnonzero(np.r_[True, tidx[order][1:] != tidx[order][:-1], True])
    for a, z in zip(b[:-1], b[1:]):
        idx = order[a:z]
        x = v[idx]
        ok = np.isfinite(x)
        if ok.sum() > 1:
            rk = np.empty(ok.sum())
            rk[np.argsort(x[ok], kind="stable")] = np.arange(ok.sum()) / (ok.sum() - 1)
            t = np.full(len(idx), np.nan)
            t[ok] = rk
            out[idx] = t
    return out


# ------------------------------------------------------------------ statistics
def _hist(y, s, di, D_, nbins):
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nbins + 1)[1:-1]))
    b = np.searchsorted(edges, s, side="right")
    nb = len(edges) + 1
    pos, neg = np.zeros((D_, nb)), np.zeros((D_, nb))
    np.add.at(pos, (di[y], b[y]), 1)
    np.add.at(neg, (di[~y], b[~y]), 1)
    return (np.cumsum(np.vstack([np.zeros(nb), pos]), axis=0), np.cumsum(np.vstack([np.zeros(nb), neg]), axis=0))


def paired_auc_ci(y, sa, sb, days, alpha: float = 0.05, seed: int = 29, boot: int = 2000):
    y = np.asarray(y).astype(bool)
    sa, sb = np.asarray(sa, float), np.asarray(sb, float)
    delta = D.auc(y, sb) - D.auc(y, sa)
    ud, di = np.unique(np.asarray(days), return_inverse=True)
    D_ = len(ud)
    pa, na = _hist(y, sa, di, D_, D.N_BINS)
    pb, nb_ = _hist(y, sb, di, D_, D.N_BINS)
    blk = max(1, min(D.BLOCK_DAYS, D_))
    starts, nblocks = D_ - blk + 1, int(math.ceil(D_ / blk))
    rnd = random.Random(seed)
    vals = []
    for _ in range(boot):
        st = [rnd.randrange(starts) for _ in range(nblocks)]
        A = D._auc_hist(sum(pa[s + blk] - pa[s] for s in st), sum(na[s + blk] - na[s] for s in st))
        B = D._auc_hist(sum(pb[s + blk] - pb[s] for s in st), sum(nb_[s + blk] - nb_[s] for s in st))
        vals.append(B - A)
    vals = np.sort(np.asarray(vals))
    return (float(delta), float(vals[int(alpha / 2 * len(vals))]),
            float(vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]))


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pit_check(pairs) -> dict:
    p = [r for r in pairs if all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                                 for k in ("bot_oi_1h", "arc_oi_1h", "bot_top", "arc_top")) and r["bot_top"] > 0]
    if len(p) < 30:
        return {"n": len(p), "spearman_oi": None, "med_rel_top": None, "ok": False}
    sp = _spearman(np.array([r["bot_oi_1h"] for r in p]), np.array([r["arc_oi_1h"] for r in p]))
    rel = float(np.median([abs(r["arc_top"] - r["bot_top"]) / r["bot_top"] for r in p]))
    return {"n": len(p), "spearman_oi": sp, "med_rel_top": rel,
            "ok": sp >= PIT_SPEARMAN_MIN and rel <= PIT_TOP_REL_MAX}


PIT2_FROM, PIT2_TO = "2026-10-21", "2026-10-28"
PIT2_N_PERPS = 25
PIT2_COVERAGE_MIN = 0.90
PIT2_EXACT_MIN = 0.99
PIT2_TOL = 1e-6
PIT2_OUT = OUT / "DPOS_PIT2.json"
REST_SHIFT = MIN5          # archive create_time T == REST 5m timestamp T + 5 min (BTCUSDT 10-27..28: 500/500 exact, 1/500 at 0)
PIT2_PATHS = {"oi": ("/futures/data/openInterestHist", "sumOpenInterest"),
              "top": ("/futures/data/topLongShortPositionRatio", "longShortRatio"),
              "glob": ("/futures/data/globalLongShortAccountRatio", "longShortRatio")}


def pit2_compare(arc, rest: dict) -> dict:
    """Archive rows vs REST 5m values at the same timestamps (spec journal, PIT-2)."""
    t = [int(x) + REST_SHIFT for x in arc["t"]]                      # archive T <-> REST T + 5 min
    hit = [i for i, x in enumerate(t) if x in rest.get("oi", {})]
    cov = len(hit) / len(t) if t else 0.0

    def share(key, col, rel):
        ok = n = 0
        for i in hit:
            r = rest.get(key, {}).get(t[i])
            a = float(arc[col][i])
            if r is None or not np.isfinite(a):
                continue
            n += 1
            ok += (abs(a / r - 1) if rel else abs(a - r)) <= PIT2_TOL if (r or not rel) else 0
        return ok / n if n else 0.0
    s = {"oi": share("oi", "oi", True), "top": share("top", "top_pos", False), "glob": share("glob", "ls_all", False)}
    return {"n_archive": len(t), "n_matched": len(hit), "coverage": cov, "exact_share": s,
            "ok": cov >= PIT2_COVERAGE_MIN and all(v >= PIT2_EXACT_MIN for v in s.values())}


def rest_fields(client, path, fields, perp, t0, t1, limit=500) -> dict:
    """{field: {timestamp: value}} over [t0, t1]. The endpoint returns the LATEST `limit` rows of the range: page back."""
    vals, end = {f: {} for f in fields}, t1
    while end >= t0:
        rows = client.futures_data(path, perp, "5m", t0, end, limit) or []
        ts = []
        for r in rows:
            try:
                t = int(r["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            ts.append(t)
            for f in fields:
                try:
                    vals[f][t] = float(r[f])
                except (KeyError, TypeError, ValueError):
                    pass
        if len(rows) < limit or not ts:
            break
        end = min(ts) - 1
        time.sleep(0.3)
    return vals


def rest_series(client, path, field, perp, t0, t1, limit=500) -> dict:
    return rest_fields(client, path, (field,), perp, t0, t1, limit)[field]


def pit2b_compare(arc, rest: dict) -> dict:
    """PIT-2b (spec journal): OI exact; top ratio == archive rounded to 4 decimals; global ratio within the error of a
    ratio computed from 4-decimal account shares."""
    t = [int(x) + REST_SHIFT for x in arc["t"]]
    hit = [i for i, x in enumerate(t) if x in rest.get("oi", {})]
    cov = len(hit) / len(t) if t else 0.0
    cnt = {"oi": [0, 0], "top": [0, 0], "glob": [0, 0]}
    for i in hit:
        T = t[i]
        o = rest["oi"].get(T)
        if o and np.isfinite(arc["oi"][i]):
            cnt["oi"][1] += 1
            cnt["oi"][0] += abs(float(arc["oi"][i]) / o - 1) <= PIT2_TOL
        tp = rest.get("top", {}).get(T)
        if tp is not None and np.isfinite(arc["top_pos"][i]):
            cnt["top"][1] += 1
            cnt["top"][0] += abs(round(float(arc["top_pos"][i]), 4) - tp) <= 1e-9
        g, la, sa = (rest.get(k, {}).get(T) for k in ("glob", "glob_long", "glob_short"))
        if g is not None and la and sa and np.isfinite(arc["ls_all"][i]):
            cnt["glob"][1] += 1
            cnt["glob"][0] += abs(float(arc["ls_all"][i]) - g) <= g * (5e-5 / la + 5e-5 / sa) + 5e-5
    s = {k: (a / n if n else 0.0) for k, (a, n) in cnt.items()}
    return {"n_archive": len(t), "n_matched": len(hit), "coverage": cov, "share": s,
            "ok": cov >= PIT2_COVERAGE_MIN and all(v >= PIT2_EXACT_MIN for v in s.values())}


def pit2_perps(counts: dict, n: int = PIT2_N_PERPS) -> list:
    return [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def train_allowed(pit1, pit2) -> bool:
    """PIT-1 failed (definition mismatch, disclosed); the operator replaced it by PIT-2, which alone gates the train."""
    return bool(pit2 and pit2.get("ok"))


def run_pit2(client=None) -> dict:
    import exchange_client as X
    import metrics_store as MS
    client = client or X.ExchangeClient(key="", secret="")
    t0 = int(datetime.strptime(PIT2_FROM, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime.strptime(PIT2_TO, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + DAY - MIN5
    counts, arcs = {}, {}
    for perp in MS.jobs_pit():
        m = MS.load(perp)
        if m is None:
            continue
        sel = (m["t"] >= t0) & (m["t"] <= t1)
        if sel.sum():
            counts[perp] = int(sel.sum())
            arcs[perp] = {k: v[sel] for k, v in m.items()}
    perps = pit2_perps(counts)
    per, tot_ok = {}, True
    for perp in perps:
        rest = {}
        for key, (path, field) in PIT2_PATHS.items():
            extra = ("longAccount", "shortAccount") if key == "glob" else ()
            got = rest_fields(client, path, (field,) + extra, perp, t0 + REST_SHIFT, t1 + REST_SHIFT)
            rest[key] = got[field]
            if extra:
                rest["glob_long"], rest["glob_short"] = got["longAccount"], got["shortAccount"]
        per[perp] = {"pit2": pit2_compare(arcs[perp], rest), "pit2b": pit2b_compare(arcs[perp], rest)}
        per[perp]["ok"] = per[perp]["pit2b"]["ok"]
        tot_ok &= per[perp]["ok"]
        print(f"  {perp}: {json.dumps(per[perp])}", flush=True)
    agg = {"n_archive": sum(v["pit2b"]["n_archive"] for v in per.values()),
           "n_matched": sum(v["pit2b"]["n_matched"] for v in per.values())}
    agg["coverage"] = agg["n_matched"] / agg["n_archive"] if agg["n_archive"] else 0.0
    agg["pit2_exact_share_min"] = {k: min(v["pit2"]["exact_share"][k] for v in per.values()) for k in PIT2_PATHS} if per else {}
    agg["pit2b_share_min"] = {k: min(v["pit2b"]["share"][k] for v in per.values()) for k in PIT2_PATHS} if per else {}
    agg["pit2_original_ok"] = bool(per) and all(v["pit2"]["ok"] for v in per.values())
    res = {"window": [PIT2_FROM, PIT2_TO], "perps": perps, "per_perp": per, "aggregate": agg, "criterion": "PIT-2b",
           "ok": bool(per) and len(perps) == PIT2_N_PERPS and all(v["ok"] for v in per.values()),
           "utc": datetime.now(timezone.utc).isoformat()}
    PIT2_OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"ok": res["ok"], "aggregate": agg}))
    return res


def outcome(res: dict) -> str:
    B = res.get("B") or {}
    years = B.get("sel_by_year") or {}
    money = ((B.get("sel") or {}).get("lo95") is not None and B["sel"]["lo95"] > 0
             and (B.get("paired") or {}).get("lo95") is not None and B["paired"]["lo95"] > 0
             and sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE)
    info = res.get("d_auc_lo") is not None and res["d_auc_lo"] > 0
    if info and money:
        return "admit"
    return "info_only" if info else "stop"


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("DPOS: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


# ------------------------------------------------------------------ PIT check (§2.4)
def run_pit() -> dict:
    import metrics_store as MS
    mp = MS.mapping()
    by_perp, pairs = {}, []
    with open(TC.POSITIONING, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = str(r.get("ts", ""))
            if not (MS.PIT_FROM <= ts[:10] <= MS.PIT_TO):
                continue
            perp = (mp.get(r.get("sym")) or {}).get("perp", r.get("sym"))
            by_perp.setdefault(perp, []).append(r)
    for perp, rows in by_perp.items():
        m = MS.load(perp)
        if m is None or not len(m["t"]):
            continue
        idx = {int(t): k for k, t in enumerate(m["t"])}
        for r in rows:
            t = int(datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp() * 1000)
            H = (t - LAG) // H1 * H1                                   # last full hour <= snapshot - 5 min
            a, b = idx.get(H - H1), idx.get(H)
            if a is None or b is None or not m["oi"][a] > 0:
                continue
            pairs.append({"perp": perp, "bot_oi_1h": r.get("oi_1h"), "arc_oi_1h": (m["oi"][b] / m["oi"][a] - 1) * 100,
                          "bot_top": r.get("top"), "arc_top": float(m["top_pos"][b]),
                          "bot_retail": r.get("retail"), "arc_retail": float(m["ls_all"][b])})
    res = pit_check(pairs)
    rr = [abs(p["arc_retail"] - p["bot_retail"]) / p["bot_retail"] for p in pairs
          if isinstance(p.get("bot_retail"), (int, float)) and p["bot_retail"] > 0 and math.isfinite(p["arc_retail"])]
    res.update({"perps": len(by_perp), "med_rel_retail": float(np.median(rr)) if rr else None,
                "window": [MS.PIT_FROM, MS.PIT_TO], "utc": datetime.now(timezone.utc).isoformat()})
    OUT.mkdir(parents=True, exist_ok=True)
    PIT_OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res


# ------------------------------------------------------------------ train (§3.3)
def _roll_median_prior(x, w=24):
    """value at i = nanmedian of x[i - w + 1 .. i] (bars <= i)."""
    import warnings
    W = D._window(np.asarray(x, float), w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(W, axis=1)


def _extra_features(name, P, tidx, sym, times):
    import experiment_dir as ED
    import funding_store as FS
    import history_store as HS
    import metrics_store as MS
    mp = MS.mapping()
    n = len(tidx)
    X = np.full((n, len(EXTRA)), np.nan)
    t0 = np.array([int(times[int(t)]["t0"].timestamp() * 1000) for t in tidx], dtype=np.int64)
    h = DAY if name == "DIR_1d" else H4
    b0 = ED._ms("2023-01-01") // H4
    nb = ED._ms(ED.TEST_FROM) // H4 - b0
    order = np.argsort(sym, kind="stable")
    bounds = np.flatnonzero(np.r_[True, sym[order][1:] != sym[order][:-1], True])
    for k, (a, z) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        idx = order[a:z]
        s = str(sym[idx[0]])
        if name == "DIR_1d":
            q = _roll_median_prior(P.qv[:, P.col[s]])
            qm = q[np.array([int(times[int(t)]["t0"].timestamp() * 1000) // DAY - P.d0 - 1 for t in tidx[idx]])]
        else:
            a15 = HS.load(HS.path_of(s))
            _, qv4, _, _ = ED._blocks_4h(a15, b0, nb)
            qm = _roll_median_prior(qv4)[tidx[idx]]
        perp = (mp.get(s) or {}).get("perp")
        m = MS.load(perp) if perp else None
        pf = pos_features_many(m, t0[idx], h, qm)
        ff = funding_features_many(FS.load_funding(FS.funding_path(s)), t0[idx])
        for c, key in enumerate(POS_KEYS):
            X[idx, c] = pf[key]
        X[idx, 6], X[idx, 7] = ff["fund_last"], ff["fund_3d"]
        if k % 50 == 0:
            print(f"  {name} positioning: {k}/{len(bounds) - 1} pairs", flush=True)
    X[:, 8] = xrank(tidx, X[:, 0])
    X[:, 9] = xrank(tidx, X[:, 6])
    return X, t0


def run_train() -> dict:
    import experiment_dir as ED
    import experiment_xsmom as EX
    import xsmom as XS
    pit = json.loads(PIT_OUT.read_text(encoding="utf-8")) if PIT_OUT.exists() else {}
    pit2 = json.loads(PIT2_OUT.read_text(encoding="utf-8")) if PIT2_OUT.exists() else None
    if not train_allowed(pit, pit2):
        raise SystemExit("DPOS: the PIT-2 check has not passed (spec journal); train is not run")
    pit = {"pit1": pit, "pit2": {k: pit2[k] for k in ("window", "aggregate", "ok")}}
    t_start = time.time()
    P = EX.load_panel().cut(ED.TEST_FROM)
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=60)
    unis = {d: u for d in range(P.row(ED.TRAIN_FROM), P.n) if (u := XS.universe(P, XS.day_str(P.d0 + d), cfg))}
    out = {"spec": SPEC, "stage": "train", "pit": pit, "extra_features": list(EXTRA), "candidates": {}}
    for name, cd in ED.candidates().items():
        rows, times = ED.build_1d(P, cfg, unis) if cd["h"] == "1d" else ED.build_4h(P, unis)
        tidx, sym, r, Xb = rows.arrays()
        X = np.column_stack([Xb[:, :len(D.FEATURES) + 2], ED._cross_section(tidx, Xb), Xb[:, -1:]])
        E, t0 = _extra_features(name, P, tidx, sym, times)
        mask = has_pos(E[:, :len(POS_KEYS)])
        y = (r > 0).astype(int)
        print(f"{name}: DIR rows {len(r)}, with metrics {int(mask.sum())} ({mask.mean():.1%}); "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
        tm, sm, rm, ym = tidx[mask], sym[mask], r[mask], y[mask]
        pA = ED.fit_oos(tm, X[mask], ym, times)
        pB = ED.fit_oos(tm, np.column_stack([X[mask], E[mask]]), ym, times)
        A, B = ED.evaluate(tm, sm, rm, pA, times), ED.evaluate(tm, sm, rm, pB, times)
        ok = np.isfinite(pA) & np.isfinite(pB)
        days = np.array([times[int(t)]["t0"].date().toordinal() for t in tm])
        d, lo, hi = paired_auc_ci(ym[ok], pA[ok], pB[ok], days[ok], alpha=ED.ALPHA, seed=29, boot=ED.BOOT)
        res = {"n_rows_dir": int(len(r)), "n_rows": int(mask.sum()), "coverage": float(mask.mean()),
               "feature_nan_share": {k: float(np.mean(~np.isfinite(E[mask, c]))) for c, k in enumerate(EXTRA)},
               "A": A, "B": B, "d_auc": d, "d_auc_lo": lo, "d_auc_hi": hi}
        res["outcome"] = outcome(res)
        out["candidates"][name] = res
        print(f"{name}: AUC A {A['auc']:.4f} B {B['auc']:.4f} | dAUC {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
              f"B top-decile net {B['sel']['mean_pct']:+.4f} [{B['sel']['lo95']:+.4f}, {B['sel']['hi95']:+.4f}] "
              f"(A {A['sel']['mean_pct']:+.4f}) | B vs universe {B['paired']['mean_pct']:+.4f} "
              f"[{B['paired']['lo95']:+.4f}, {B['paired']['hi95']:+.4f}] | B hit {B['sel_hit_rate']:.4f} vs p* "
              f"{B['sel_breakeven_p']:.4f} | years B { {k: round(v, 4) for k, v in B['sel_by_year'].items()} } | "
              f"{res['outcome']} | {(time.time() - t_start) / 60:.1f} min", flush=True)
    out["verdict"] = {n: v["outcome"] for n, v in out["candidates"].items()}
    out["minutes"] = round((time.time() - t_start) / 60, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat()
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["verdict"]))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "pit2":
        return 0 if run_pit2().get("ok") else 1
    if cmd == "pit":
        return 0 if run_pit().get("ok") else 1
    if cmd == "train":
        run_train()
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("DPOS: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("DPOS: the test runner is written only after stage 2 (spec §4)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
