"""E2 -- a model trained on MONEY: enter only when the predicted net result after costs is positive.

e2-money-target-1010-spec.md. Offline, trader app; the signal bot is read only.

  rows      all candidates (entry + blocked) on the 15m watchlist grid since 2026-03-01
  label y   net % of the trade E1 would make if it admitted the candidate: P-1 trail exit,
            taker 0.10% + 8 bps per side (identical to the engine's trade result)
  features  logged at decision time + grid features from bars <= ie (no look-ahead)
  split     train < 2026-05-31 | 1-day embargo | validation June | test >= 2026-07-01 opened ONCE,
            model refit monthly on everything before the month minus a 1-day embargo
  families  (a) numpy logistic, (b) CatBoost regression on y, (c) CatBoost classifier on y > 0
  policies  EV > t (t in 0 / 0.10 / 0.25 %) x max_per_day (none / 3 / 10) x family = 27
  choose    largest mean daily R on validation with >= 30 trades; R <= 0 -> STOP (test not opened)

  (from trader/) ../pyembed/python.exe experiment_e2.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import daily_revenue as DR        # noqa: E402
import execution_replay as R      # noqa: E402
import fill_model as FM           # noqa: E402
import money_validator as MV      # noqa: E402
import trading_policy as TP       # noqa: E402

TRAIN_END = "2026-05-31"          # exclusive; validation starts 06-01 -> 1-day embargo
VAL_FROM, VAL_TO = "2026-06-01", "2026-07-01"
TEST_FROM = "2026-07-01"
HORIZON = 96
EVENT_NUM = ("ml_proba", "candidate_score", "ranker_final_score", "ranker_ev", "ranker_top_gainer_prob",
             "adx", "rsi", "vol_x", "slope_pct", "daily_range", "btc_vs_ema50", "is_bull_day", "macd_hist")
GRID_NUM = ("day_ret", "day_rank", "atr_pct", "ret4", "ret16", "ret96", "vol_rel", "hour")
CATS = ("mode", "tf", "kind", "gate")
THRESHOLDS = (0.0, 0.10, 0.25)
CAPS = (None, 3, 10)
FAMILIES = ("a", "b", "c")
MIN_VAL_TRADES = 30
OUT = TC.RUNTIME / "experiments"
P1 = replace(TP.CHAMPION, name="p1", admission=TP.Admission(kind="all_candidates"), exits=TP.Exits(kind="p1_trail"))


def _day(c):
    return c["ts"].strftime("%Y-%m-%d")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------- label and features

def label(grid, c, fm, floor_of=None):
    """Net % of the trade the E1 engine makes when it admits `c` (spec §8.1); None if not evaluable."""
    plan = TP.exit_plan(P1, grid, c, floor_of)
    if plan is None or not c.get("price"):
        return None
    _, exit_raw, reason, _ = plan
    fill = fm.buy(c["price"], 0.0, None)
    px = exit_raw if reason == "tp" else fm.sell(exit_raw, 0.0, None)
    fee_exit = fm.maker_fee if reason == "tp" else fm.taker_fee
    return (px * (1 - fee_exit) * (1 - fm.taker_fee) / fill - 1) * 100


def row_features(c, grid) -> dict:
    """Decision-time features only: logged fields + grid features from bars <= ie (spec §2, §8.2)."""
    f = {k: _num(c.get(k)) for k in EVENT_NUM}
    j, ie = c.get("j"), c.get("ie")
    g = dict.fromkeys(GRID_NUM)
    if j is not None and ie is not None:
        C, V = grid.C[:, j], grid.V[:, j]
        cl = C[ie]
        g["day_ret"] = _num(grid.RET[ie, j])
        g["day_rank"] = _num(grid.RANK[ie, j])
        g["atr_pct"] = _num(grid.ATR[ie, j] / cl) if cl else None
        for n in (4, 16, 96):
            prev = C[ie - n] if ie - n >= 0 else float("nan")
            g[f"ret{n}"] = _num(cl / prev - 1) if prev else None
        win = V[max(0, ie - 95): ie + 1]
        win = win[np.isfinite(win)]
        med = float(np.median(win)) if len(win) else 0.0
        g["vol_rel"] = _num(V[ie] / med) if med > 0 else None
    g["hour"] = float(c["ts"].hour)
    f.update(g)
    for k in CATS:
        f[k] = str(c.get(k)) if c.get(k) is not None else "none"
    return f


# ---------------------------------------------------------------- splits

def embargo_cut(rows, start_day: str, days: int = 1) -> list:
    """Training rows for a period starting at `start_day`: dated before start - `days` (spec §8.3)."""
    cut = (datetime.strptime(start_day, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    return [r for r in rows if r["ts"].strftime("%Y-%m-%d") < cut]


def monthly_refit_plan(test_from: str, test_to: str) -> list:
    """[(YYYY-MM, first day of that month inside the test period)] (spec §3, §8.8)."""
    out, d = [], datetime.strptime(test_from, "%Y-%m-%d")
    end = datetime.strptime(test_to, "%Y-%m-%d")
    while d <= end:
        out.append((d.strftime("%Y-%m"), d.strftime("%Y-%m-%d")))
        d = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
    return out


def select_on_validation(val_results: dict):
    """Largest mean daily R with >= MIN_VAL_TRADES trades; None (STOP) if that R <= 0 (spec §4, §8.4–8.5)."""
    ok = {n: r for n, r in val_results.items() if r["trades"] >= MIN_VAL_TRADES and r.get("R") is not None}
    if not ok:
        return None
    best = max(ok, key=lambda n: ok[n]["R"])
    return best if ok[best]["R"] > 0 else None


def pipeline(validate, select, test):
    """validate -> select -> test, and test only when select returns a winner."""
    val = validate()
    w = select(val)
    if w is None:
        return {"validation": val, "winner": None, "test": None}
    return {"validation": val, "winner": w, "test": test(w)}


# ---------------------------------------------------------------- models

class LogReg:
    """L2 logistic regression by Newton steps (numpy only)."""

    def __init__(self, l2: float = 1.0, iters: int = 50):
        self.l2, self.iters = l2, iters
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        reg = np.full(d + 1, self.l2)
        reg[-1] = 0.0
        for _ in range(self.iters):
            z = np.clip(Xb @ w, -35, 35)
            p = 1 / (1 + np.exp(-z))
            g = Xb.T @ (p - y) + reg * w
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg + 1e-9)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w, self.b = w[:-1], w[-1]
        return self

    def predict_proba(self, X):
        z = np.clip(np.asarray(X, float) @ self.w + self.b, -35, 35)
        return 1 / (1 + np.exp(-z))


def auc(y, s) -> float | None:
    y, s = np.asarray(y, float), np.asarray(s, float)
    pos, neg = y > 0.5, y <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ev_from_proba(p, mean_pos, mean_neg_abs):
    return p * mean_pos - (1 - p) * mean_neg_abs


class Design:
    """Feature matrix: numeric with train medians + missing flags (logistic) and raw + cat (CatBoost)."""

    def __init__(self, train_rows):
        F = [r["f"] for r in train_rows]
        self.num = list(EVENT_NUM) + list(GRID_NUM)
        self.med = {k: float(np.median([v[k] for v in F if v[k] is not None] or [0.0])) for k in self.num}
        cols = np.array([[v[k] if v[k] is not None else self.med[k] for k in self.num] for v in F], float)
        self.mu, self.sd = cols.mean(0), cols.std(0) + 1e-9
        self.levels = {k: sorted({v[k] for v in F}) for k in CATS}

    def logit_X(self, rows):
        out = []
        for r in rows:
            v = r["f"]
            x = [((v[k] if v[k] is not None else self.med[k]) - self.mu[i]) / self.sd[i] for i, k in enumerate(self.num)]
            x += [1.0 if v[k] is None else 0.0 for k in self.num]
            for k in CATS:
                x += [1.0 if v[k] == lv else 0.0 for lv in self.levels[k]]
            out.append(x)
        return np.array(out, float)

    def cb_X(self, rows):
        return [[(r["f"][k] if r["f"][k] is not None else float("nan")) for k in self.num] + [r["f"][k] for k in CATS]
                for r in rows]

    def cat_idx(self):
        return list(range(len(self.num), len(self.num) + len(CATS)))


def fit_family(fam, train_rows):
    """-> function rows -> EV (% net) per row."""
    D = Design(train_rows)
    y = np.array([r["y"] for r in train_rows], float)
    yb = (y > 0).astype(float)
    mp = float(y[y > 0].mean()) if (y > 0).any() else 0.0
    mn = float(-y[y <= 0].mean()) if (y <= 0).any() else 0.0
    if fam == "a":
        m = LogReg(l2=1.0).fit(D.logit_X(train_rows), yb)
        return lambda rows: ev_from_proba(m.predict_proba(D.logit_X(rows)), mp, mn)
    from catboost import CatBoostClassifier, CatBoostRegressor
    kw = dict(depth=4, iterations=300, learning_rate=0.05, random_seed=29, verbose=False, thread_count=4,
              cat_features=D.cat_idx())
    if fam == "b":
        m = CatBoostRegressor(loss_function="RMSE", **kw).fit(D.cb_X(train_rows), y)
        return lambda rows: np.asarray(m.predict(D.cb_X(rows)), float)
    m = CatBoostClassifier(loss_function="Logloss", **kw).fit(D.cb_X(train_rows), yb)
    return lambda rows: ev_from_proba(np.asarray(m.predict_proba(D.cb_X(rows)))[:, 1], mp, mn)


def diagnostics(rows, ev) -> dict:
    y = np.array([r["y"] for r in rows], float)
    ev = np.asarray(ev, float)
    dec = {}
    if len(rows) >= 50:
        q = np.quantile(ev, np.linspace(0, 1, 11))
        for i in range(10):
            m = (ev >= q[i]) & ((ev <= q[i + 1]) if i == 9 else (ev < q[i + 1]))
            if m.any():
                dec[i + 1] = {"pred": round(float(ev[m].mean()), 4), "fact": round(float(y[m].mean()), 4), "n": int(m.sum())}
    return {"n": len(rows), "base_rate_y_gt_0": round(float((y > 0).mean()), 4) if len(y) else None,
            "auc": auc((y > 0).astype(float), ev), "ic_spearman": spearman(ev, y), "deciles": dec}


def _policy(thr, cap, fam, cooldown):
    return replace(TP.CHAMPION, name=f"ev_{fam}|t{thr}|cap{cap}",
                   admission=TP.Admission(kind="all_candidates", min_score=thr + 1e-12, score_key=f"ev_{fam}", max_per_day=cap),
                   exits=TP.Exits(kind="p1_trail"),
                   risk=TP.Risk(max_open=TP.CHAMPION.risk.max_open, one_per_symbol=True, cooldown_bars=cooldown))


# ---------------------------------------------------------------- main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    TC.bot_readonly()
    import config as BOT
    trades, blocked = R.load_events(include_blocked=True)
    grid = R._grid()
    R.attach_grid(trades, grid)
    R.attach_grid(blocked, grid)
    fl, fm, cache = R._floor(), FM.model("taker", 1), {}
    rows = []
    for c in sorted([c for c in trades + blocked if c["ie"] is not None], key=lambda c: c["ts"]):
        if c["ie"] + HORIZON >= grid.N:
            continue                                   # label not mature
        y = label(grid, c, fm, fl)
        if y is None:
            continue
        c["y"], c["f"] = y, row_features(c, grid)
        rows.append(c)
    print(f"rows {len(rows)} ({_day(rows[0])}..{_day(rows[-1])}), y>0 share {np.mean([r['y'] > 0 for r in rows]):.3f}, "
          f"mean y {np.mean([r['y'] for r in rows]):+.3f}%  [{time.time() - t0:.0f}s]", flush=True)
    train = [r for r in rows if _day(r) < TRAIN_END]
    val = [r for r in rows if VAL_FROM <= _day(r) < VAL_TO]
    cooldown = int(getattr(BOT, "COOLDOWN_BARS", 19))
    report = {"spec": "e2-money-target-1010-spec.md", "rows": len(rows), "train_rows": len(train), "val_rows": len(val),
              "costs": "taker 0.10% + 8 bps per side", "diagnostics": {}}

    def validate():
        res = {}
        for fam in FAMILIES:
            f = fit_family(fam, train)
            ev_tr, ev_va = f(train), f(val)
            for r, e in zip(val, ev_va):
                r[f"ev_{fam}"] = float(e)
            report["diagnostics"][f"{fam}_train_in_sample"] = diagnostics(train, ev_tr)
            report["diagnostics"][f"{fam}_validation"] = diagnostics(val, ev_va)
            d = report["diagnostics"][f"{fam}_validation"]
            print(f"  family {fam}: val AUC {d['auc']} IC {d['ic_spearman']} (train AUC "
                  f"{report['diagnostics'][f'{fam}_train_in_sample']['auc']})", flush=True)
            for thr in THRESHOLDS:
                for cap in CAPS:
                    pol = _policy(thr, cap, fam, cooldown)
                    r = R.run(pol, fm, grid, val, floor_of=fl, day_from=VAL_FROM, day_to=VAL_TO, plan_cache=cache)
                    res[pol.name] = {"R": r["agg"].get("mean_pct"), "trades": len(r["trades"]), "agg": r["agg"],
                                     "net_per_trade": R.decompose(r["trades"]).get("net_mean_pct")}
                    print(f"    {pol.name:22s} trades {len(r['trades']):4d} R {r['agg'].get('mean_pct') or 0:+.3f}%/d "
                          f"net/trade {res[pol.name]['net_per_trade'] or 0:+.3f}", flush=True)
        return res

    def test(winner):
        fam = winner.split("|")[0][3:]
        thr = float(winner.split("|")[1][1:])
        capt = winner.split("|")[2][3:]
        cap = None if capt == "None" else int(capt)
        test_rows = [r for r in rows if _day(r) >= TEST_FROM]
        plan = monthly_refit_plan(TEST_FROM, _day(rows[-1]))
        evs, refits = [], []
        for i, (month, start) in enumerate(plan):
            end = plan[i + 1][1] if i + 1 < len(plan) else "9999-99-99"
            tr = embargo_cut(rows, start)
            f = fit_family(fam, tr)
            mrows = [r for r in test_rows if start <= _day(r) < end]
            for r, e in zip(mrows, f(mrows)):
                r[f"ev_{fam}"] = float(e)
            evs += [r[f"ev_{fam}"] for r in mrows]
            refits.append({"month": month, "train_rows": len(tr), "train_last_day": _day(tr[-1]), "test_rows": len(mrows)})
        report["diagnostics"][f"{fam}_test_monthly_refit"] = diagnostics(test_rows, evs)
        pol = _policy(thr, cap, fam, cooldown)
        w = R.run(pol, fm, grid, test_rows, floor_of=fl, day_from=TEST_FROM, plan_cache=cache)
        base_pol = replace(TP.CHAMPION, name="base_p1_real", exits=TP.Exits(kind="p1_trail"))
        real = [c for c in trades if c["ie"] is not None]
        b = R.run(base_pol, fm, grid, real, floor_of=fl, day_from=TEST_FROM, plan_cache=cache)
        tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
        # same intensity as the winner: the share of test candidates passing its EV threshold, same cap
        share = sum(1 for r in test_rows if r.get(f"ev_{fam}", float("-inf")) > thr) / max(1, len(test_rows))
        ctrl = []
        for seed in range(20):
            cp = replace(pol, name=f"random{seed}", admission=TP.Admission(kind="all_candidates", max_per_day=cap,
                                                                           random_frac=share, random_seed=seed))
            rr = R.run(cp, fm, grid, test_rows, floor_of=fl, day_from=TEST_FROM, plan_cache=cache)
            ctrl.append({"seed": seed, "R": rr["agg"].get("mean_pct"), "trades": len(rr["trades"]),
                         "net_per_trade": MV.trade_ci([t["ret_pct"] for t in rr["trades"]])["mean"]})
        rs = sorted(c["R"] for c in ctrl if c["R"] is not None)
        ref = {f"slip{lvl}": R.run(pol, FM.model("taker", lvl), grid, test_rows, floor_of=fl, day_from=TEST_FROM)["agg"]
               for lvl in (0, 2)}
        prof = MV.profitable(w["agg"], tci)
        out = {"policy": asdict(pol), "epoch": pol.epoch(), "refits": refits, "winner": w["agg"], "trades": len(w["trades"]),
               "trade_ci": tci, "decomposition": R.decompose(w["trades"]), "profitable": prof,
               "beats_cash": bool(w["agg"].get("lo95") is not None and w["agg"]["lo95"] > 0),
               "base": b["agg"], "paired_vs_base_reference": DR.paired(w["series"], b["series"]),
               "random_control": {"frac": share, "runs": ctrl, "R_median": rs[len(rs) // 2] if rs else None,
                                  "winner_R_percentile": (sum(1 for v in rs if v < (w["agg"].get("mean_pct") or 0)) / len(rs)) if rs else None},
               "slip_reference": ref}
        MV.log_trial({"family": "E2", "name": winner, "policy_epoch": pol.epoch(), "k_opened_on_test": 1,
                      "profitable": prof["verdict"], "beats_cash": out["beats_cash"], "test_R": w["agg"]})
        return out

    res = pipeline(validate, select_on_validation, test)
    report.update(validation=res["validation"], winner=res["winner"], test=res["test"])
    if res["winner"] is None:
        report["verdict"] = "STOP: no policy has mean daily R > 0 on validation; the test half was not opened"
        print("\n" + report["verdict"])
    else:
        t = res["test"]
        print(f"\nWINNER on validation: {res['winner']}  {res['validation'][res['winner']]['R']:+.3f}%/d")
        print(f"TEST: R {t['winner']['mean_pct']:+.3f}%/d [{t['winner']['lo95']:+.3f},{t['winner']['hi95']:+.3f}] "
              f"trades {t['trades']} net/trade {t['trade_ci']['mean']:+.3f} [{t['trade_ci']['lo95']:+.3f},{t['trade_ci']['hi95']:+.3f}] "
              f"profitable {t['profitable']['verdict']} beats_cash {t['beats_cash']} | random median R {t['random_control']['R_median']} "
              f"winner beats {t['random_control']['winner_R_percentile']}")
        print("test diagnostics:", {k: v for k, v in report["diagnostics"].items() if "test" in k})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "e2_money_target_1011.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"saved {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
