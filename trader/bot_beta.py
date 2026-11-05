"""H4 -- how much of the X4 book's loss is market beta and how much the bot's entries (bot-beta-h4-1105-spec.md).

  factors   daily returns from the bot's 15m grid: ALT = equal weight of the watchlist without BTC/ETH, BTC
  model     R_d = alpha + b_alt * ALT_d + b_btc * BTC_d, OLS; 5-day block bootstrap over days (rows), 95%
  windows   main 2026-03-01..06-30; reference 07-01..10-10 (H3 test half, already used) and 10-13..11-05 (E7)
  hedged    reference only: R - b_alt(main) * ALT - b_btc(main) * BTC on the reference windows, hedge costs not counted

  (from trader/) ../pyembed/python.exe -u bot_beta.py run
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "bot-beta-h4-1105-spec.md"
OUT = TC.RUNTIME / "experiments" / "H4_BETA.json"
BARS_PER_DAY = 96
BLOCK, BOOT, SEED = 5, 2000, 29
MAIN = ("main", "2026-03-01", "2026-07-01")
REFERENCE = (("h3_test", "2026-07-01", "2026-10-11"), ("e7_window", "2026-10-13", "2026-11-05"))
START_EQUITY = 10_000.0


def factor_returns(grid, alt_syms, btc_sym) -> dict:
    days = grid.N // BARS_PER_DAY
    cols = [grid.col[s] for s in alt_syms if s in grid.col]
    jb = grid.col.get(btc_sym)
    out = {}
    for k in range(1, days):
        i0, i1 = k * BARS_PER_DAY - 1, (k + 1) * BARS_PER_DAY - 1
        c0, c1 = grid.C[i0, cols], grid.C[i1, cols]
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0)
        alt = float(np.mean(c1[ok] / c0[ok] - 1) * 100) if ok.any() else None
        btc = None
        if jb is not None and np.isfinite(grid.C[i0, jb]) and np.isfinite(grid.C[i1, jb]) and grid.C[i0, jb] > 0:
            btc = float((grid.C[i1, jb] / grid.C[i0, jb] - 1) * 100)
        if alt is not None and btc is not None:
            out[(grid.t0 + timedelta(days=k)).strftime("%Y-%m-%d")] = (alt, btc)
    return out


def ols(y, X) -> list:
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(c) for c in coef]


def boot_ols(y, X, block: int = BLOCK, boot: int = BOOT, seed: int = SEED) -> dict:
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    n = len(y)
    b = max(1, min(block, n))
    rnd = random.Random(seed)
    draws = []
    for _ in range(boot):
        idx = []
        while len(idx) < n:
            s = rnd.randrange(n - b + 1)
            idx.extend(range(s, s + b))
        idx = idx[:n]
        draws.append(ols(y[idx], X[idx]))
    d = np.array(draws)
    return {"coef": ols(y, X), "lo": [float(v) for v in np.percentile(d, 2.5, axis=0)],
            "hi": [float(v) for v in np.percentile(d, 97.5, axis=0)], "ci_level": 0.95, "n_days": n,
            "block": b, "boot": boot}


def hedged(series, factors: dict, b_alt: float, b_btc: float) -> list:
    return [(d, r - b_alt * factors[d][0] - b_btc * factors[d][1]) for d, r in series if d in factors]


def exposure(trades, series, start_equity: float) -> float:
    days = [d for d, _ in series]
    eq, e_day = start_equity, {}
    for d, r in series:
        e_day[d] = eq
        eq *= 1 + r / 100
    held = {d: 0.0 for d in days}
    for t in trades:
        a, b = t["entry_ts"], t["exit_time"]
        for d in days:
            d0 = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            ov = (min(b, d0 + timedelta(days=1)) - max(a, d0)).total_seconds()
            if ov > 0:
                held[d] += t["cost"] * ov / 86400
    return float(np.mean([held[d] / e_day[d] for d in days])) if days else 0.0


def _fit(series, factors) -> dict:
    rows = [(r, factors[d][0], factors[d][1]) for d, r in series if d in factors]
    y = [r[0] for r in rows]
    X = [[r[1], r[2]] for r in rows]
    out = boot_ols(y, X)
    out["names"] = ["alpha_pct_per_day", "b_alt", "b_btc"]
    out["mean_R"] = float(np.mean(y)) if y else None
    out["mean_ALT"] = float(np.mean([x[0] for x in X])) if X else None
    out["mean_BTC"] = float(np.mean([x[1] for x in X])) if X else None
    return out


def reading(fit: dict) -> str:
    """spec §3, on the main window."""
    a_lo, a_hi = fit["lo"][0], fit["hi"][0]
    if a_hi < 0:
        return "anti-skill: entries worse than their beta -> close bot-signal trading"
    if a_lo > 0:
        return "skill: entries better than their beta -> separate spec for a hedged book, forward only"
    if fit["lo"][1] > 0:
        return "beta: the loss is exposure to alts -> separate spec for a beta-hedged book, forward only"
    return "too early: alpha and beta both include 0"


def run() -> dict:
    import execution_replay as ER
    import fill_model as FM
    import provenance as PV
    import daily_revenue as DR
    import _audit_k1_replay_windows as K
    t0 = time.time()
    trades, _ = ER.load_events()
    grid = ER._grid()
    ER.attach_grid(trades, grid)
    alts = [s for s in grid.syms if s not in ("BTCUSDT", "ETHUSDT")]
    f = factor_returns(grid, alts, "BTCUSDT")
    pol, fm, floor = K.x4_policy(), FM.model("taker", 1), ER._floor()
    out = {"spec": SPEC, "n_alt": len(alts), "windows": {},
           "provenance": PV.block(events=PV.events_digest(ER.EVENTS, grid.last_bar.strftime("%Y-%m-%dT%H:%M")),
                                  grid=PV.grid_digest(grid, 0, grid.N), grid_last_bar=str(grid.last_bar))}
    main_fit = None
    for name, lo, hi in (MAIN,) + REFERENCE:
        r = ER.run(pol, fm, grid, trades, floor_of=floor, day_from=lo, day_to=hi)
        fit = _fit(r["series"], f)
        w = {"window": [lo, hi], "fit": fit, "agg": r["agg"], "n_trades": len(r["trades"]),
             "n_censored": r["n_censored"], "exposure": exposure(r["trades"], r["series"], START_EQUITY)}
        if name == "main":
            main_fit = fit
            w["reading"] = reading(fit)
        else:
            h = hedged(r["series"], f, main_fit["coef"][1], main_fit["coef"][2])
            w["hedged_with_main_betas"] = DR.aggregate(h)
        out["windows"][name] = w
        c, l, u = fit["coef"], fit["lo"], fit["hi"]
        print(f"{name} {lo}..{hi}: days {fit['n_days']} R {fit['mean_R']:.3f} ALT {fit['mean_ALT']:.3f} "
              f"BTC {fit['mean_BTC']:.3f} | alpha {c[0]:+.3f} [{l[0]:+.3f}, {u[0]:+.3f}] b_alt {c[1]:+.3f} "
              f"[{l[1]:+.3f}, {u[1]:+.3f}] b_btc {c[2]:+.3f} [{l[2]:+.3f}, {u[2]:+.3f}] | exposure "
              f"{w['exposure']:.3f} trades {w['n_trades']} censored {w['n_censored']}"
              + (f" | {w['reading']}" if "reading" in w else
                 f" | hedged R {w['hedged_with_main_betas'].get('mean_pct')} "
                 f"[{w['hedged_with_main_betas'].get('lo95')}, {w['hedged_with_main_betas'].get('hi95')}]"),
              flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
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
