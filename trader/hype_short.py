"""Hype-short family (hype-short-family-1105-spec.md): H-A shorts yesterday's leaders against the equal-weight market,
H-B shorts the 2-60 day cohort of new perps against BTC. Daily book on USD-M perps, marked at the day's last 15m close.

  Panel      daily close, quote volume, funding per perp (day numbers; first / last day with data)
  universe   age >= 30 days, top-100 by turnover over [i - 30, i), >= 50 names, else nothing is held
  leaders    top return close[i-1-L] -> close[i-1]: today's move is not in the signal; trades at close[i]
  run_ha     7 weekly tranches; per tranche short K leaders and long the universe, half the tranche each; a name's short
             across tranches <= 1/20 of equity; seed -> random K names (control)
  run_hb     short each perp at age 2 (1/30 of equity, total shorts <= 1/2), exit at age 60 or delisting; BTC long at
             beta * shorts, beta from past days only; seed -> random perps older than 180 days on the same days (control)
  pnl        qty * price change (perps: no cash outlay), funding: short receives rate * notional, long pays it;
             cost per side = costs(col, i) * |traded notional|

  (from trader/) ../pyembed/python.exe -u hype_short.py explore | legs      -- 2023-03..2025-12, exploration only
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "hype-short-family-1105-spec.md"
DAY = 86_400_000
AGE_MIN, TOP_N, MIN_UNIVERSE, TURN_DAYS = 30, 100, 50, 30
K, N_TRANCHES, NAME_CAP = 10, 7, 1 / 20
HB_ENTRY_AGE, HB_EXIT_AGE, HB_SHARE, HB_CAP, HB_OLD_AGE = 2, 60, 1 / 30, 0.5, 180
BETA_DAYS, BETA_MIN_DAYS, BETA_CLIP = 60, 30, (0.5, 2.0)
PERP_TAKER = 0.0005
EXPLORE = ("2023-03-01", "2026-01-01")
DECISIVE = ("2020-07-01", "2023-01-01")
OUT_EXPLORE = TC.RUNTIME / "experiments" / "HYPE_SHORT_EXPLORE.json"
N_RANDOM = 50


@dataclass
class Panel:
    days: list
    syms: list
    close: np.ndarray
    qvol: np.ndarray
    first: np.ndarray
    last: np.ndarray
    fund: np.ndarray


# ------------------------------------------------------------------ signal
def universe(p: Panel, i: int) -> list:
    if i < TURN_DAYS:
        return []
    live = [j for j in range(len(p.syms)) if p.first[j] <= i - AGE_MIN and p.last[j] >= i
            and np.isfinite(p.close[i - 1, j])]
    if len(live) < MIN_UNIVERSE:
        return []
    turn = {j: float(np.nansum(p.qvol[i - TURN_DAYS:i, j])) for j in live}
    top = sorted(live, key=lambda j: (-turn[j], j))[:TOP_N]
    return sorted(top) if len(top) >= MIN_UNIVERSE else []


def leaders(p: Panel, i: int, uni, L: int, k: int, blocked) -> list:
    if i - 1 - L < 0:
        return []
    r = []
    for j in uni:
        a, b = p.close[i - 1 - L, j], p.close[i - 1, j]
        if j not in blocked and np.isfinite(a) and np.isfinite(b) and a > 0:
            r.append((b / a - 1, j))
    r.sort(key=lambda x: (-x[0], x[1]))
    return [j for _, j in r[:k]]


def funding_pnl(side: str, notional: float, rate: float) -> float:
    return notional * rate if side == "short" else -notional * rate


def beta_past(cohort_rets, btc_rets) -> float:
    x = np.asarray(btc_rets[-BETA_DAYS:], dtype=float)
    y = np.asarray(cohort_rets[-BETA_DAYS:], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < BETA_MIN_DAYS or np.var(x[m]) == 0:
        return 1.0
    b = float(np.cov(y[m], x[m], ddof=1)[0, 1] / np.var(x[m], ddof=1))
    return float(min(BETA_CLIP[1], max(BETA_CLIP[0], b)))


# ------------------------------------------------------------------ book helpers
class _Book:
    """Signed quantities per key; marking, funding and costs accumulate separately for the conservation check."""

    def __init__(self, p: Panel, costs, equity: float = 1.0):
        self.p, self.costs, self.eq0, self.equity = p, costs, equity, equity
        self.pos = {}                  # key -> [col, qty, last price]
        self.pnl = self.fund = self.cost = 0.0
        self.fund_long = self.fund_short = 0.0   # by the sign of the net position of the name in its tranche

    def px(self, col, i, fallback):
        c = self.p.close[i, col]
        return float(c) if np.isfinite(c) else fallback

    def mark(self, i):
        for k, v in self.pos.items():
            col, q, last = v
            pr = self.px(col, i, last)
            d = q * (pr - last)
            self.pnl += d
            self.equity += d
            f = -q * pr * float(self.p.fund[i, col])        # short (q < 0) receives a positive rate
            self.fund += f
            if q < 0:
                self.fund_short += f
            else:
                self.fund_long += f
            self.equity += f
            v[2] = pr

    def trade(self, key, col, new_q, i):
        old = self.pos.get(key)
        oq = old[1] if old else 0.0
        pr = self.px(col, i, old[2] if old else float("nan"))
        if not np.isfinite(pr):
            return
        c = abs(new_q - oq) * pr * float(self.costs(col, i))
        self.cost += c
        self.equity -= c
        if abs(new_q) < 1e-15:
            self.pos.pop(key, None)
        else:
            self.pos[key] = [col, new_q, pr]

    def notional(self, pred):
        return sum(abs(v[1]) * v[2] for k, v in self.pos.items() if pred(k, v))

    def conservation_ok(self):
        return abs(self.equity - (self.eq0 + self.pnl + self.fund - self.cost)) < 1e-9 * max(1.0, self.eq0)


# ------------------------------------------------------------------ H-A
def run_ha(p: Panel, L: int, costs, seed=None, k: int = K, start: int = 0, end=None) -> dict:
    end = len(p.days) if end is None else end
    rnd = random.Random(seed) if seed is not None else None
    b = _Book(p, costs)
    series, eq_min, prev = [], b.equity, b.equity
    for i in range(start, end):
        b.mark(i)
        for key in [key for key, v in b.pos.items() if i >= p.last[v[0]]]:      # delisting: close at the last close
            b.trade(key, b.pos[key][0], 0.0, i)
        t = (i - start) % N_TRANCHES
        uni = universe(p, i)
        target = {}
        if uni:
            E = b.equity
            leg = E / N_TRANCHES / 2
            per_s = leg / k
            other_short = {}
            for key, v in b.pos.items():
                if key[0] != t and v[1] < 0:
                    other_short[v[0]] = other_short.get(v[0], 0.0) + abs(v[1]) * v[2]
            blocked = {j for j in uni if other_short.get(j, 0.0) + per_s > NAME_CAP * E + 1e-12}
            if rnd is None:
                shorts = leaders(p, i, uni, L, k, blocked)
            else:
                pool = [j for j in uni if j not in blocked]
                shorts = rnd.sample(pool, min(k, len(pool)))
            per_l = leg / len(uni)
            for j in uni:
                target[j] = target.get(j, 0.0) + per_l / p.close[i, j] if np.isfinite(p.close[i, j]) else target.get(j, 0.0)
            for j in shorts:
                if np.isfinite(p.close[i, j]):
                    target[j] = target.get(j, 0.0) - per_s / p.close[i, j]
        old = {key[1]: key for key in b.pos if key[0] == t}
        for col in set(old) | set(target):
            b.trade((t, col), col, target.get(col, 0.0), i)
        eq_min = min(eq_min, b.equity)
        series.append((i, (b.equity / prev - 1) * 100 if prev > 0 else 0.0))
        prev = b.equity
    return {"series": series, "equity_min": eq_min, "final_equity": b.equity, "pnl": b.pnl, "funding": b.fund,
            "fund_long": b.fund_long, "fund_short": b.fund_short, "costs": b.cost,
            "conservation_ok": b.conservation_ok()}


# ------------------------------------------------------------------ H-B
def run_hb(p: Panel, btc: int, costs, seed=None, start: int = 0, end=None) -> dict:
    end = len(p.days) if end is None else end
    rnd = random.Random(seed) if seed is not None else None
    b = _Book(p, costs)
    series, entries, shares, daily_share = [], [], [], []
    open_ = {}                                       # col -> entry record
    cohort_r, btc_r = [], []
    prev = b.equity
    for i in range(start, end):
        held = [c for c in open_ if np.isfinite(p.close[i, c]) and np.isfinite(p.close[i - 1, c])] if i > 0 else []
        if held:
            cohort_r.append(float(np.mean([p.close[i, c] / p.close[i - 1, c] - 1 for c in held])))
            btc_r.append(float(p.close[i, btc] / p.close[i - 1, btc] - 1) if np.isfinite(p.close[i - 1, btc]) else np.nan)
        b.mark(i)
        for c, e in list(open_.items()):
            if i >= e["exit_due"] or i >= p.last[c]:
                b.trade(("s", c), c, 0.0, i)
                e["exit_i"] = i
                del open_[c]
        news = [c for c in range(len(p.syms)) if c != btc and i == p.first[c] + HB_ENTRY_AGE
                and p.last[c] > i and np.isfinite(p.close[i, c])]
        for c in news:
            if rnd is not None:
                pool = [x for x in range(len(p.syms)) if x != btc and x not in open_ and i - p.first[x] > HB_OLD_AGE
                        and p.last[x] > i and np.isfinite(p.close[i, x])]
                if not pool:
                    continue
                c = rnd.choice(pool)
            if c in open_:
                continue
            E = b.equity
            if b.notional(lambda k, v: k[0] == "s") + HB_SHARE * E > HB_CAP * E + 1e-12:
                continue
            b.trade(("s", c), c, -HB_SHARE * E / p.close[i, c], i)
            due = i + (HB_EXIT_AGE - HB_ENTRY_AGE)
            rec = {"col": c, "sym": p.syms[c], "entry_i": i, "exit_i": None, "exit_due": due}
            open_[c] = rec
            entries.append(rec)
            shares.append(b.notional(lambda k, v: k[0] == "s") / b.equity)
        shorts = b.notional(lambda k, v: k[0] == "s")
        beta = beta_past(cohort_r, btc_r)
        if np.isfinite(p.close[i, btc]):
            b.trade(("h", btc), btc, beta * shorts / p.close[i, btc], i)
        daily_share.append(shorts / b.equity if b.equity > 0 else 0.0)
        series.append((i, (b.equity / prev - 1) * 100 if prev > 0 else 0.0))
        prev = b.equity
    return {"series": series, "entries": entries, "short_share_after_entry": shares, "final_equity": b.equity,
            "pnl": b.pnl, "funding": b.fund, "fund_long": b.fund_long, "fund_short": b.fund_short, "costs": b.cost,
            "short_share_daily": daily_share, "conservation_ok": b.conservation_ok()}


# ------------------------------------------------------------------ data
def _day(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) // DAY


def build_panel(kind: str, lo: str, hi: str) -> Panel:
    """kind 'recent': runtime/perp_history + runtime/funding (2023+); 'bh': runtime/history_bh/perp + funding (2020-22).
    A perp whose first bar is the store's first day existed before the store: its age is unknown and large, so it is
    never a 'new' perp for H-B and never too young for H-A. A perp whose last bar is the store's last day is live."""
    if kind == "recent":
        pdir, fdir, fname = TC.RUNTIME / "perp_history", TC.RUNTIME / "funding", "{}_funding.npz"
    else:
        pdir, fdir, fname = TC.RUNTIME / "history_bh" / "perp", TC.RUNTIME / "history_bh" / "funding", "{}_funding.npz"
    d0, d1 = _day(lo) - TURN_DAYS - AGE_MIN - 5, _day(hi)
    files = sorted(pdir.glob("*_15m.npz"))
    raw, gmin, gmax = {}, None, None
    for f in files:
        sym = f.name[:-len("_15m.npz")]
        if not sym.isascii():
            continue
        with np.load(f) as z:
            t, c, v = z["t"].astype(np.int64), z["c"].astype(float), z["v"].astype(float)
        if not len(t):
            continue
        dd = t // DAY
        raw[sym] = (dd, c, v)
        gmin = dd[0] if gmin is None else min(gmin, dd[0])
        gmax = dd[-1] if gmax is None else max(gmax, dd[-1])
    syms = sorted(raw)
    D, S = d1 - d0, len(syms)
    close, qvol, fund = np.full((D, S), np.nan), np.zeros((D, S)), np.zeros((D, S))
    first, last = np.zeros(S, dtype=np.int64), np.zeros(S, dtype=np.int64)
    for j, sym in enumerate(syms):
        dd, c, v = raw[sym]
        idx = dd - d0
        m = (idx >= 0) & (idx < D)
        if m.any():
            ii = idx[m]
            close[ii, j] = c[m]                         # sorted bars: the last write of a day is its last close
            np.add.at(qvol[:, j], ii, v[m] * c[m])
        first[j] = -10 ** 6 if dd[0] <= gmin else int(dd[0] - d0)
        last[j] = 10 ** 6 if dd[-1] >= gmax - 1 else int(dd[-1] - d0)
        fp = fdir / fname.format(sym)
        if fp.exists():
            with np.load(fp) as z:
                ft, fr = z["t"].astype(np.int64), z["rate"].astype(float)
            fi = ft // DAY - d0
            fm = (fi >= 0) & (fi < D) & np.isfinite(fr)
            np.add.at(fund[:, j], fi[fm], fr[fm])
    # a day without a bar keeps the previous close for marking; liveness still uses first / last
    for j in range(S):
        col = close[:, j]
        okv = np.isfinite(col)
        if okv.any():
            idx = np.where(okv, np.arange(D), -1)
            np.maximum.accumulate(idx, out=idx)
            filled = np.where(idx >= 0, col[np.maximum(idx, 0)], np.nan)
            lo_i, hi_i = max(0, first[j]), min(D - 1, last[j])
            seg = np.full(D, np.nan)
            if hi_i >= lo_i:
                seg[lo_i:hi_i + 1] = filled[lo_i:hi_i + 1]
            close[:, j] = seg
    return Panel(list(range(d0, d1)), syms, close, qvol, first, last, fund)


def cost_fn(p: Panel, flat_bps=None):
    """Per side: perp taker + slippage. Default: COST_CAL bucket of the day's turnover rank (1 000 USDT medians)."""
    import cost_calibration as CC
    table = json.loads(CC.OUT.read_text(encoding="utf-8"))["tables"]["live"]
    bps = {b: (table[b]["buy"]["1000"]["median"] + table[b]["sell"]["1000"]["median"]) / 2 for b in table}
    cache = {}

    def rank_of(i):
        if i not in cache:
            turn = np.nansum(p.qvol[max(0, i - TURN_DAYS):i], axis=0)
            order = np.argsort(-turn, kind="stable")
            r = np.empty(len(order), dtype=np.int64)
            r[order] = np.arange(1, len(order) + 1)
            cache[i] = r
        return cache[i]

    def f(col, i):
        if flat_bps is not None:
            return PERP_TAKER + flat_bps / 1e4
        return PERP_TAKER + bps[CC.bucket(p.syms[col], int(rank_of(i)[col]))] / 1e4
    return f


def _window(p: Panel, lo: str, hi: str):
    a, b = _day(lo) - p.days[0], _day(hi) - p.days[0]
    return max(0, a), min(len(p.days), b)


def _summary(res, p: Panel, alpha: float) -> dict:
    import daily_revenue as DR
    s = [(datetime.fromtimestamp(p.days[i] * DAY / 1000, timezone.utc).strftime("%Y-%m-%d"), v) for i, v in res["series"]]
    agg = DR.aggregate(s, alpha=alpha, block=20)
    by_year, eq, peak, dd_year = {}, 1.0, 1.0, {}
    for d, v in s:
        by_year.setdefault(d[:4], []).append(v)
        eq *= 1 + v / 100
        peak = max(peak, eq)
        dd_year[d[:4]] = min(dd_year.get(d[:4], 0.0), eq / peak - 1)
    months = {}
    for d, v in s:
        months[d[:7]] = (1 + months.get(d[:7], 0.0) / 100) * (1 + v / 100) * 100 - 100
    return {"agg": agg, "sd": float(np.std([v for _, v in s], ddof=1)) if len(s) > 1 else None,
            "r_by_year": {y: float(np.mean(v)) for y, v in sorted(by_year.items())},
            "max_dd_by_year": {y: round(v * 100, 2) for y, v in sorted(dd_year.items())},
            "worst_month_pct": min(months.values()) if months else None, "equity_min": res.get("equity_min"),
            "funding": res.get("funding"), "costs": res.get("costs"), "conservation_ok": res.get("conservation_ok")}


def explore(n_random: int = N_RANDOM) -> dict:
    """Exploration window only (2023-03..2025-12, survivorship disclosed): sigma, MDE for the decisive window, code check."""
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    p = build_panel("recent", *EXPLORE)
    a, b = _window(p, *EXPLORE)
    costs = cost_fn(p)
    btc = p.syms.index("BTCUSDT")
    out = {"spec": SPEC, "stage": "exploration", "window": EXPLORE, "n_perps": len(p.syms), "candidates": {}}
    runs = {"HA_L1": (lambda sd, cf=costs: run_ha(p, 1, cf, seed=sd, start=a, end=b), 0.0125),
            "HA_L7": (lambda sd, cf=costs: run_ha(p, 7, cf, seed=sd, start=a, end=b), 0.0125),
            "HB": (lambda sd, cf=costs: run_hb(p, btc, cf, seed=sd, start=a, end=b), 0.025)}
    for name, (fn, alpha) in runs.items():
        base = _summary(fn(None), p, alpha)
        rand = [_summary(fn(sd), p, alpha)["agg"]["mean_pct"] for sd in range(n_random)]
        sens = {f"flat_{x}bp": _summary(fn(None, cost_fn(p, x)), p, alpha)["agg"]["mean_pct"] for x in (8, 20)}
        base.update(rand_p95=float(np.percentile(rand, 95)), rand_p50=float(np.median(rand)), cost_sensitivity=sens)
        out["candidates"][name] = base
        g = base["agg"]
        print(f"{name}: R {g['mean_pct']} [{g['lo95']}, {g['hi95']}] ({g['ci_method']}, block 20) sd {base['sd']:.3f} | "
              f"random p50/p95 {base['rand_p50']:.4f}/{base['rand_p95']:.4f} | years {base['r_by_year']} | "
              f"dd {base['max_dd_by_year']} | worst month {base['worst_month_pct']:.1f}% | sens {sens} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    out["provenance"] = PV.block()
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT_EXPLORE.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def legs(n_random: int = N_RANDOM) -> dict:
    """fourth-review-1105-spec.md §3: the base runs on committed code, funding by leg, H-B exposure, and H-B against
    its control day by day (the paired series sets the power of the forward look). Exploration window only."""
    import daily_revenue as DR
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    p = build_panel("recent", *EXPLORE)
    a, b = _window(p, *EXPLORE)
    costs = cost_fn(p)
    btc = p.syms.index("BTCUSDT")
    day = lambda i: datetime.fromtimestamp(p.days[i] * DAY / 1000, timezone.utc).strftime("%Y-%m-%d")      # noqa: E731
    out = {"spec": "fourth-review-1105-spec.md", "stage": "exploration", "window": EXPLORE, "runs": {}}
    base = {"HA_L1": run_ha(p, 1, costs, start=a, end=b), "HA_L7": run_ha(p, 7, costs, start=a, end=b),
            "HB": run_hb(p, btc, costs, start=a, end=b)}
    n = b - a
    for name, r in base.items():
        out["runs"][name] = {"mean_pct": float(np.mean([v for _, v in r["series"]])),
                             "fund_long_pct_day": r["fund_long"] * 100 / n, "fund_short_pct_day": r["fund_short"] * 100 / n,
                             "funding_pct_day": r["funding"] * 100 / n, "costs_pct_day": r["costs"] * 100 / n,
                             "conservation_ok": r["conservation_ok"]}
        print(name, json.dumps(out["runs"][name]), flush=True)
    hb = base["HB"]
    out["hb_mean_short_share"] = float(np.mean(hb["short_share_daily"]))
    out["hb_entries"] = len(hb["entries"])
    ctrl = [run_hb(p, btc, costs, seed=sd, start=a, end=b)["series"] for sd in range(n_random)]
    cm = np.mean([[v for _, v in c] for c in ctrl], axis=0)
    paired = [(day(i), v - float(m)) for (i, v), m in zip(hb["series"], cm)]
    agg = DR.aggregate(paired, block=20)
    sd_d = float(np.std([v for _, v in paired], ddof=1))
    se_day = (agg["hi95"] - agg["lo95"]) / 2 / 1.96 * n ** 0.5                       # per sqrt(day), block-adjusted
    edge = agg["mean_pct"]
    from statistics import NormalDist
    N = NormalDist()
    out["hb_paired_vs_control"] = {"agg": agg, "sd_daily": sd_d, "se_scale_per_sqrt_day": se_day,
                                   "days_for_80pct_power": ((1.96 + 0.8416) * se_day / edge) ** 2 if edge > 0 else None,
                                   "power_180d": N.cdf(edge / (se_day / 180 ** 0.5) - 1.96) if edge > 0 else 0.0,
                                   "power_309d": N.cdf(edge / (se_day / 309 ** 0.5) - 1.96) if edge > 0 else 0.0}
    print(json.dumps({k: out[k] for k in ("hb_mean_short_share", "hb_entries", "hb_paired_vs_control")}, default=str), flush=True)
    out["provenance"] = PV.block()
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (TC.RUNTIME / "experiments" / "HYPE_SHORT_LEGS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                                     encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["explore"]:
        explore(int(argv[1]) if len(argv) > 1 else N_RANDOM)
    elif argv[:1] == ["legs"]:
        legs(int(argv[1]) if len(argv) > 1 else N_RANDOM)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
