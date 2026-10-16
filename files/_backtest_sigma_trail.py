"""Volatility-forecast trail (sigma-hat) vs the live ATR trail, on the bot's own 15m trades.

WHY (2026-10-05, prod-tools plan, step 1)
The North Star decomposes as coverage x capture x lead. Over the last 240 days
coverage is 0.73 but capture only 0.22: the bot is in most winners and keeps a
fifth of their move. The trail decides capture. The live buffer is
max(k * ATR, floor * price): ATR is a 14-bar average of PAST ranges (~3.5 h). A
realized-volatility forecast (HAR: 15m / 1h / 4h / 1d / 1w windows + intraday
profile) forecasts the NEXT hour's variability instead. On BTC 1m data a HAR of
this kind reached R^2 0.75 for the next hour's log volatility (research notebook,
2026-10-05) -- which is why it is worth replaying here.

DESIGN, FIXED BEFORE THE REPLAY
  * One change only: ATR in the stop formula is replaced by ATR_hat = kappa *
    sigma_hat * close. trail_k of each trade and the per-mode floor are untouched.
  * Same WIDTH by construction: kappa = median(ATR / (sigma_hat * close)) over
    every coin and bar BEFORE 2026-03-01. What is compared is the SHAPE of the
    buffer (forecast vs trailing average), not wider vs narrower.
  * The sigma model is a pooled OLS fitted only on bars before 2026-03-01; every
    replayed trade is from 2026-03-01 on (exit_validator.SINCE_DEFAULT).
  * Primary arm: sigma_hat for the next 4 bars (1 h). Secondary: next 16 bars (4 h).
  * Engine, labels, statistics and the verdict rule are exit_validator's (plain
    P-1 trail from the entry bar, 96 bars; rocket-days from the immutable store):
      accept = rocket lower95 > 0 AND all lower95 >= -0.10 pp AND >= 60% months >= 0
  * Diagnostic: does sigma_hat forecast the next hour's realized volatility better
    than ATR does, after March? If not, the trail has no reason to improve.

Run from files/:  ../pyembed/python.exe _backtest_sigma_trail.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                                   # noqa: E402
import exit_validator as EV                      # noqa: E402
import _backtest_trend_start_detector as TD      # noqa: E402

SPLIT = datetime(2026, 3, 1, tzinfo=timezone.utc)
WINDOWS = (1, 4, 16, 96, 672)                    # 15m bars: 15m, 1h, 4h, 1d, 1w
HORIZONS = {"1h": 4, "4h": 16}
PRIMARY = "1h"
OUT = HERE.parent / ".runtime" / "sigma_trail_backtest.json"


def lows_aligned(grid):
    """15m lows on the grid (exit_validator.Grid keeps O/H/C only)."""
    L = np.full_like(grid.C, np.nan)
    for s, j in grid.col.items():
        for x in TD.bars_15m(s):
            k = int((x[0] - grid.t0) / EV.STEP)
            if 0 <= k < L.shape[0]:
                L[k, j] = x[3]
    return L


def window_mean(a, w):
    """Trailing mean over w bars including the current one; NaN unless >= 90% present."""
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0)
    cs, cn = np.cumsum(v, axis=0), np.cumsum(ok, axis=0)
    s = cs.copy(); n = cn.copy().astype(float)
    s[w:] -= cs[:-w]; n[w:] -= cn[:-w]
    out = s / np.where(n > 0, n, np.nan)
    out[n < 0.9 * w] = np.nan
    out[: w - 1] = np.nan
    return out


def features(grid, L):
    C, H = grid.C, grid.H
    r = np.diff(np.log(C), axis=0, prepend=np.nan)
    r2 = r ** 2
    pk = np.log(H / L) ** 2 / (4 * math.log(2))
    floor = 1e-10
    X = [np.log(np.maximum(window_mean(r2, w), floor)) for w in WINDOWS]
    X += [np.log(np.maximum(window_mean(pk, w), floor)) for w in (1, 4)]
    tod = (np.arange(C.shape[0]) % 96) / 96.0
    for k in (1, 2, 3):
        X += [np.repeat(np.sin(2 * np.pi * k * tod)[:, None], C.shape[1], 1),
              np.repeat(np.cos(2 * np.pi * k * tod)[:, None], C.shape[1], 1)]
    return np.stack([x.astype(np.float32) for x in X], axis=-1), r2


def target(r2, K):
    """log mean squared 15m return over the NEXT K bars (strictly after bar q)."""
    fut = window_mean(r2, K)                      # mean over bars q-K+1..q
    y = np.full_like(fut, np.nan)
    y[:-K] = np.log(np.maximum(fut[K:], 1e-10))  # shifted so row q holds bars q+1..q+K
    return y


def fit_sigma(X, y, train_rows):
    """Pooled OLS on train rows; returns per-bar sigma_hat (return units) for every row."""
    F = X.reshape(-1, X.shape[-1]); Y = y.reshape(-1)
    tr = train_rows.reshape(-1) & np.isfinite(F).all(1) & np.isfinite(Y)
    idx = np.flatnonzero(tr)[::2]
    A = np.column_stack([np.ones(len(idx)), F[idx]])
    beta = np.linalg.lstsq(A, Y[idx], rcond=None)[0]
    pred = F @ beta[1:].astype(np.float32) + beta[0]
    smear = float(np.mean(np.exp(Y[idx] - A @ beta)))     # Duan: E[r^2] from log fit
    sig = np.sqrt(np.exp(pred) * smear).reshape(y.shape)
    return sig, beta, smear, len(idx)


def trail_with(grid, t, A, floor, max_bars=EV.P1_MAX_BARS):
    """exit_validator.plain_trail with the volatility array passed in."""
    j, ie, k = t["j"], t["ie"], t["k"]
    C = grid.C
    ep = C[ie, j]
    if not math.isfinite(ep):
        return None
    stop = ep - max(k * A[ie, j] if math.isfinite(A[ie, j]) else 0.0, floor * ep)
    last = min(grid.N - 1, ie + max_bars)
    for q in range(ie + 1, last + 1):
        cq = C[q, j]
        if not math.isfinite(cq):
            continue
        if math.isfinite(A[q, j]):
            stop = max(stop, cq - max(k * A[q, j], floor * cq))
        if cq < stop:
            return (cq / ep - 1) * 100
    return (C[last, j] / ep - 1) * 100 if math.isfinite(C[last, j]) else None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    grid = EV.load_grid(int(getattr(config, "ATR_PERIOD", 14)))
    n = grid.C.shape[0]
    print(f"grid: {len(grid.syms)} symbols, {n} bars from {grid.t0:%Y-%m-%d} to {grid.last_bar:%Y-%m-%d}", flush=True)
    L = lows_aligned(grid)
    X, r2 = features(grid, L)
    times = np.array([grid.t0 + i * EV.STEP for i in range(n)])
    pre = np.repeat((times < SPLIT)[:, None], grid.C.shape[1], 1)
    post = ~pre

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": SPLIT.isoformat(),
              "symbols": len(grid.syms), "arms": {}}
    A_hat = {}
    for name, K in HORIZONS.items():
        y = target(r2, K)
        sig, beta, smear, nfit = fit_sigma(X, y, pre)
        atr_rel = grid.ATR[:n] / grid.C
        ratio = atr_rel / sig
        kappa = float(np.nanmedian(np.where(pre & np.isfinite(ratio), ratio, np.nan)))
        A_hat[name] = kappa * sig * grid.C
        # diagnostic: out-of-sample predictive power for the next K bars' log realized vol
        ok = post & np.isfinite(y) & np.isfinite(sig) & np.isfinite(atr_rel) & (atr_rel > 0)
        yy, ps, pa = y[ok], np.log(sig[ok] ** 2), np.log(atr_rel[ok] ** 2)
        def r2_cal(p):                            # R^2 after a linear calibration fitted on PRE rows
            okp = pre & np.isfinite(y) & np.isfinite(sig) & np.isfinite(atr_rel) & (atr_rel > 0)
            pp = (np.log(sig[okp] ** 2) if p is ps else np.log(atr_rel[okp] ** 2))
            b1, b0 = np.polyfit(pp[::7], y[okp][::7], 1)
            e = yy - (b0 + b1 * p)
            return float(1 - np.mean(e ** 2) / np.var(yy))
        report["arms"][name] = {"bars_ahead": K, "fit_rows": nfit, "smear": round(smear, 4), "kappa": round(kappa, 4),
                                "oos_r2_log_rv_sigma_hat": round(r2_cal(ps), 4), "oos_r2_log_rv_atr": round(r2_cal(pa), 4),
                                "oos_rows": int(ok.sum())}
        print(name, report["arms"][name], flush=True)

    trades = EV.load_trades(grid, EV.SINCE_DEFAULT)
    rockets, winners = EV.day_labels()
    print(f"trades since {EV.SINCE_DEFAULT}: {len(trades)}", flush=True)
    for name in HORIZONS:
        rows, width_now, width_new = [], [], []
        by_mode = {}
        for t in trades:
            fl = EV.floor_of(t["mode"], config)
            x = trail_with(grid, t, grid.ATR, fl)
            yv = trail_with(grid, t, A_hat[name], fl)
            if x is None or yv is None:
                continue
            rows.append((t, x, yv))
            ie, j = t["ie"], t["j"]
            if math.isfinite(grid.ATR[ie, j]) and math.isfinite(A_hat[name][ie, j]):
                width_now.append(max(t["k"] * grid.ATR[ie, j], fl * grid.C[ie, j]) / grid.C[ie, j] * 100)
                width_new.append(max(t["k"] * A_hat[name][ie, j], fl * grid.C[ie, j]) / grid.C[ie, j] * 100)
            by_mode.setdefault(t["mode"] or "?", []).append(yv - x)
        rnd = random.Random(29)
        d_all = [y_ - x for _, x, y_ in rows]
        d_rk = [y_ - x for t, x, y_ in rows if (t["day"], t["sym"]) in rockets]
        d_wn = [y_ - x for t, x, y_ in rows if (t["day"], t["sym"]) in winners]
        changed = sum(1 for v in d_all if abs(v) > 1e-9)
        all_s, rk_s, wn_s = EV.paired(d_all, rnd), EV.paired(d_rk, rnd), EV.paired(d_wn, rnd)
        bym = {}
        for t, x, y_ in rows:
            if (t["day"], t["sym"]) in rockets:
                bym.setdefault(t["day"][:7], []).append(y_ - x)
        months = {m: {"mean": round(sum(v) / len(v), 3), "n": len(v)} for m, v in sorted(bym.items())}
        scored = [m for m, v in months.items() if v["n"] >= EV.MONTH_MIN]
        agree = (sum(1 for m in scored if months[m]["mean"] >= 0) / len(scored)) if scored else 0.0
        verdict = EV.decide(all_s, rk_s, agree, changed)
        arm = report["arms"][name]
        arm.update({"trades": len(rows), "changed_trades": changed,
                    "median_width_pct_now": round(float(np.median(width_now)), 3),
                    "median_width_pct_sigma": round(float(np.median(width_new)), 3),
                    "mean_pnl_now": round(sum(x for _, x, _ in rows) / len(rows), 3),
                    "mean_pnl_sigma": round(sum(y_ for _, _, y_ in rows) / len(rows), 3),
                    "all_trades_diff_pp": all_s, "rocket_day_diff_pp": rk_s, "winner_day_diff_pp": wn_s,
                    "rocket_months": months, "month_agreement": round(agree, 3), "months_scored": len(scored),
                    "by_mode_mean_diff_pp": {m: {"n": len(v), "mean": round(sum(v) / len(v), 3)} for m, v in sorted(by_mode.items())},
                    "verdict": verdict, "primary": name == PRIMARY})
        print(f"\n[{name}{' PRIMARY' if name == PRIMARY else ''}] verdict={verdict}  trades={len(rows)} changed={changed}  "
              f"width now {arm['median_width_pct_now']}% vs sigma {arm['median_width_pct_sigma']}%\n"
              f"  rocket-days {rk_s}\n  all {all_s}\n  winners {wn_s}\n  months agree {agree:.0%} of {len(scored)}: {months}\n"
              f"  by mode: {arm['by_mode_mean_diff_pp']}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
