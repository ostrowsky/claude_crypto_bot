"""LA-3 -- money test of exchange announcements on train (listing-announcements-la-1028-spec.md §4).

  events    announcements of a coin ALREADY on Binance spot at t_pub (BN_PERP_NEW, OKX/BYBIT spot and perp listings),
            one per (type, coin) within 24 h, train t_pub in [2023-03-01, 2025-12-31)
  trade     buy at the close of the first 15m bar opening >= t_pub + 1 min; exit after 1 h / 4 h / 24 h or on the
            X4 trail (P-1, k = 2, 8% floor, 96 bars); optional skip when the coin is already >= +10% since the UTC open
  book      10 slots of equity/10, fill_model level 1, 1 000 000 USDT, marks at 00:00 UTC
  controls  50 seeds each: the same coin at a random time of the month without any announcement within 48 h; a
            random top-100 liquid coin without announcements at the same time
  leak      the 4 h / 24 h return BEFORE t_pub, and the share of events already >= +2.5% before it
  -> runtime/experiments/LA_TRAIN.json. The 2026 test is opened once, by the operator only (spec §4).

  (from trader/) ../pyembed/python.exe -u experiment_la.py train
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import announcements as AN        # noqa: E402
import trader_config as TC        # noqa: E402

BAR = 15 * 60 * 1000
HOUR = 3_600_000
DAY = 86_400_000
TRADE_TYPES = ("BN_PERP_NEW", "OKX_SPOT_NEW", "OKX_PERP_NEW", "BYBIT_SPOT_NEW", "BYBIT_PERP_NEW")
EXIT_BARS = {"h1": 4, "h4": 16, "h24": 96}
X4_BARS, X4_FLOOR, X4_K = 96, 0.08, 2.0
SKIP_EXT = 0.10
DELAY_MS = 60_000
DEDUP_MS = DAY
TRAIN_FROM, TEST_FROM = "2023-03-01", "2026-01-01"
N_RANDOM = 50
MIN_TRADES = 100
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
SLOTS = 10
START_EQUITY = 1_000_000.0
OUT = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT / "LA_TRAIN.json"
TEST_MARKER = OUT / "LA_TEST_OPENED.json"
SPEC = "listing-announcements-la-1028-spec.md"


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def candidates() -> dict:
    return {f"{x}{'_skipext' if s else ''}": {"exit": x, "skip_ext": s} for x in ("h1", "h4", "h24", "x4") for s in (False, True)}


# ------------------------------------------------------------------ events
def mentions(records, spot_syms) -> dict:
    """sym -> sorted announcement times, any type (used to keep controls away from announcements)."""
    out = {}
    for r in records:
        et = AN.classify(r)
        for s in AN.to_spot(AN.extract_tickers(r["title"], et), et, spot_syms):
            out.setdefault(s, []).append(r["t_pub"])
    return {s: sorted(v) for s, v in out.items()}


def build_events(records, spot_first_bar: dict, t_from: int, t_to: int) -> list:
    last = {}
    out = []
    for r in sorted(records, key=lambda r: r["t_pub"]):
        if not (t_from <= r["t_pub"] < t_to - DAY):
            continue
        et = AN.classify(r)
        if et not in TRADE_TYPES:
            continue
        for s in AN.to_spot(AN.extract_tickers(r["title"], et), et, spot_first_bar):
            if not AN.tradable(s, r["t_pub"], spot_first_bar):
                continue
            k = (et, s)
            if k in last and r["t_pub"] - last[k] < DEDUP_MS:
                continue
            last[k] = r["t_pub"]
            out.append({"sym": s, "t_pub": r["t_pub"], "etype": et, "title": r["title"]})
    return out


# ------------------------------------------------------------------ one trade
def atr_of(arr, n: int = 14) -> np.ndarray:
    c, h, l = (np.asarray(arr[k], dtype=float) for k in ("c", "h", "l"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    cs = np.cumsum(np.r_[0.0, tr])
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _ret_since_open(arr, i: int):
    t = np.asarray(arr["t"], dtype=np.int64)
    d0 = int(t[i]) // DAY * DAY
    j = int(np.searchsorted(t, d0))
    if j >= len(t) or int(t[j]) != d0 or j > i:
        return None
    o = float(arr["o"][j])
    return float(arr["c"][i]) / o - 1 if o > 0 else None


def trade(arr, t_pub: int, cand: dict, fm, atr):
    t = np.asarray(arr["t"], dtype=np.int64)
    c = np.asarray(arr["c"], dtype=float)
    e = AN.entry_bar_open(t_pub, DELAY_MS)
    i = int(np.searchsorted(t, e))
    if i >= len(t) or int(t[i]) != e or not math.isfinite(c[i]) or c[i] <= 0:
        return None
    if cand.get("skip_ext"):
        r = _ret_since_open(arr, i)
        if r is not None and r >= SKIP_EXT:
            return None
    c0 = float(c[i])
    if cand["exit"] in EXIT_BARS:
        j = i + EXIT_BARS[cand["exit"]]
        if j >= len(c) or int(t[j]) != int(t[i]) + EXIT_BARS[cand["exit"]] * BAR:
            return None
        reason = "time"
    else:
        last = min(len(c) - 1, i + X4_BARS)
        if last < i + X4_BARS:
            return None
        a0 = atr[i] if atr is not None and math.isfinite(atr[i]) else 0.0
        stop = c0 - max(X4_K * a0, X4_FLOOR * c0)
        j, reason = last, "time"
        for q in range(i + 1, last + 1):
            cq = c[q]
            if not math.isfinite(cq):
                continue
            aq = atr[q] if atr is not None and math.isfinite(atr[q]) else 0.0
            stop = max(stop, cq - max(X4_K * aq, X4_FLOOR * cq))
            if cq < stop:
                j, reason = q, "trail"
                break
    cx = float(c[j])
    fill = fm.buy(c0, 0.0, None)
    sell = fm.sell(cx, 0.0, None)
    gross = (1 - fm.taker_fee) / fill * sell * (1 - fm.taker_fee)       # proceeds per 1 USDT spent
    entry_close_ms, exit_close_ms = int(t[i]) + BAR, int(t[j]) + BAR
    marks = {}
    d = (entry_close_ms // DAY + 1) * DAY
    while d < exit_close_ms:
        k = int(np.searchsorted(t, d - BAR, "right")) - 1                # last bar closed by the mark
        marks[d] = (float(c[k]) * (1 - fm.taker_fee) / fill) if k >= i else 1.0
        d += DAY
    return {"entry_ms": int(t[i]), "exit_ms": int(t[j]), "entry_close_ms": entry_close_ms, "exit_close_ms": exit_close_ms,
            "entry_raw": c0, "exit_raw": cx, "ret_pct": (gross - 1) * 100, "gross_mult": gross, "marks": marks,
            "reason": reason}


# ------------------------------------------------------------------ book
def book(trades, day_from: str, day_to: str, slots: int = SLOTS, start_equity: float = START_EQUITY) -> dict:
    """trades: [{sym, ...trade()}]; each trade's own marks value the open position at 00:00."""
    d0, d1 = _ms(day_from), _ms(day_to)
    tr = sorted((x for x in trades if d0 <= x["entry_close_ms"] < d1), key=lambda x: (x["entry_close_ms"], x["sym"]))
    cash, opn, done, marks = float(start_equity), [], [], {}
    skipped = 0
    k = 0

    def close_until(ts):
        nonlocal cash
        for p in sorted([p for p in opn if p["exit_close_ms"] <= ts], key=lambda p: p["exit_close_ms"]):
            opn.remove(p)
            proceeds = p["cost"] * p["gross_mult"]
            cash += proceeds
            done.append({**{k2: v for k2, v in p.items() if k2 != "marks"}, "pnl": proceeds - p["cost"]})

    def equity_at(ts):
        v = cash
        for p in opn:
            m = [mv for mt, mv in p["marks"].items() if mt <= ts]
            v += p["cost"] * (m[-1] if m else 1.0)
        return v

    for day_ms in range(d0, d1 + 1, DAY):
        while k < len(tr) and tr[k]["entry_close_ms"] < day_ms:
            x = tr[k]
            k += 1
            close_until(x["entry_close_ms"])
            if len(opn) >= slots:
                skipped += 1
                continue
            notional = min(equity_at(x["entry_close_ms"]) / slots, cash)
            if notional <= 0:
                skipped += 1
                continue
            cash -= notional
            opn.append({**x, "cost": notional})
        close_until(day_ms)
        marks[_day(day_ms)] = equity_at(day_ms)
    for p in list(opn):                                     # window edge: sold at the last mark value
        m = [mv for mt, mv in p["marks"].items() if mt <= d1]
        p["gross_mult"] = m[-1] if m else 1.0
        p["exit_close_ms"] = d1
    close_until(d1 + 1)
    days = sorted(marks)
    series = [(a, (marks[b] / marks[a] - 1) * 100) for a, b in zip(days, days[1:]) if marks[a] > 0]
    pnl = sum(x["pnl"] for x in done)
    return {"series": series, "trades": done, "skipped_full": skipped, "final_equity": cash,
            "conservation_ok": abs((cash - start_equity) - pnl) < 1e-6 * max(1.0, start_equity) and not opn}


def simulate(events, bars: dict, cand: dict, fm, day_from: str, day_to: str, slots: int = SLOTS,
             start_equity: float = START_EQUITY) -> dict:
    import daily_revenue as DR
    trades = []
    for e in events:
        a = bars.get(e["sym"])
        if a is None:
            continue
        x = trade(a, e["t_pub"], cand, fm, atr_of(a) if cand["exit"] == "x4" else None)
        if x is not None:
            trades.append({**x, "sym": e["sym"], "etype": e.get("etype")})
    r = book(trades, day_from, day_to, slots, start_equity)
    r["agg"] = DR.aggregate(r["series"])
    return r


# ------------------------------------------------------------------ controls
def control_events(events, bars: dict, kind: str, seed: int, all_events_by_sym: dict, coins_at=None) -> list:
    rnd = random.Random(seed)
    out = []
    for e in events:
        if kind == "time":
            t = AN.control_time(e["sym"], e["t_pub"], all_events_by_sym, rnd)
            if t is not None:
                out.append({**e, "t_pub": int(t)})
        else:
            pool = coins_at(e["t_pub"]) if coins_at else sorted(bars)
            pool = [s for s in pool if s in bars]
            s = AN.control_coin(e["t_pub"], pool, all_events_by_sym, rnd)
            if s is not None:
                out.append({**e, "sym": s})
    return out


# ------------------------------------------------------------------ admission
def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    m = (res.get("agg") or {}).get("mean_pct")
    pt, pc = _p95(res.get("rand_time") or []), _p95(res.get("rand_coin") or [])
    years = res.get("r_by_year") or {}
    checks = {"beats_random_time_p95": m is not None and pt is not None and m > pt,
              "beats_random_coin_p95": m is not None and pc is not None and m > pc,
              "R_lo_gt_0": (res.get("agg") or {}).get("lo95") is not None and res["agg"]["lo95"] > 0,
              "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def select(res: dict, pbo):
    if pbo is None or pbo >= PBO_MAX:
        return None
    adm = [n for n, v in res.items() if v.get("admitted")]
    return max(adm, key=lambda n: (res[n]["agg"]["mean_pct"], n)) if adm else None


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("LA: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


# ------------------------------------------------------------------ train run
def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def run_train(n_random: int = N_RANDOM) -> dict:
    import daily_revenue as DR
    import experiment_xsmom as XE
    import fill_model as FM
    import history_store as HS
    import pbo as PB
    import xsmom as XS
    fm = FM.model("taker", 1)
    t0 = time.time()
    st = AN.AnnouncementStore()
    records = AN.drop_migrated([r for s in ("binance", "okx", "bybit") for r in st.load(s)])
    panel = XE.load_panel()
    first_bar = {}
    for s, j in panel.col.items():
        rows = np.flatnonzero(panel.has_bar[:, j])
        if len(rows):
            first_bar[s] = (int(panel.d0) + int(rows[0])) * DAY
    spot = set(first_bar)
    all_ev = mentions(records, spot)
    events = build_events(records, first_bar, _ms(TRAIN_FROM), _ms(TEST_FROM))
    ucfg = XS.Config(L=1, H=1, K=10, min_age_days=60)
    uni_cache = {}

    def coins_at(t):
        d = _day(t)
        if d not in uni_cache:
            uni_cache[d] = XS.universe(panel, d, ucfg)
        return uni_cache[d]

    cands = candidates()
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "n_records": len(records),
           "fill": "taker + 8 bps per side", "types": {}}
    for et in TRADE_TYPES:
        ev = [e for e in events if e["etype"] == et]
        print(f"{et}: {len(ev)} events", flush=True)
        if len(ev) < 10:
            out["types"][et] = {"n_events": len(ev), "skipped": "too few events"}
            continue
        ctrl = {("time", s): control_events(ev, {x: True for x in spot}, "time", s, all_ev) for s in range(n_random)}
        ctrl.update({("coin", s): control_events(ev, {x: True for x in spot}, "coin", 1000 + s, all_ev, coins_at)
                     for s in range(n_random)})
        need = {}                                           # sym -> set of t_pub
        for e in ev:
            need.setdefault(e["sym"], set()).add(e["t_pub"])
        for evs in ctrl.values():
            for e in evs:
                need.setdefault(e["sym"], set()).add(e["t_pub"])
        trades = {}                                         # (sym, t, cand) -> trade
        leak = []
        ev_keys = {(e["sym"], e["t_pub"]) for e in ev}
        for sym, ts in need.items():
            a = HS.load(HS.path_of(sym))
            if a is None or not len(a["t"]):
                continue
            atr = atr_of(a)
            for t in ts:
                for name, cand in cands.items():
                    x = trade(a, t, cand, fm, atr if cand["exit"] == "x4" else None)
                    if x is not None:
                        trades[(sym, t, name)] = {**x, "sym": sym}
                if (sym, t) in ev_keys:
                    p4, p24 = AN.pre_return(a, t, 4), AN.pre_return(a, t, 24)
                    leak.append({"pre4h": p4, "pre24h": p24})
        res, series = {}, {}
        for name in cands:
            sel = lambda evs: [trades[(e["sym"], e["t_pub"], name)] for e in evs if (e["sym"], e["t_pub"], name) in trades]  # noqa: E731
            main = book(sel(ev), TRAIN_FROM, TEST_FROM)
            agg = DR.aggregate(main["series"])

            def mean_of(evs):
                s = book(sel(evs), TRAIN_FROM, TEST_FROM)["series"]
                return sum(v for _, v in s) / len(s) if s else None
            rt = [mean_of(ctrl[("time", s)]) for s in range(n_random)]
            rc = [mean_of(ctrl[("coin", s)]) for s in range(n_random)]
            rets = [x["ret_pct"] for x in main["trades"]]
            r = {"agg": agg, "n_trades": len(rets), "skipped_full": main["skipped_full"],
                 "trade_mean_pct": float(np.mean(rets)) if rets else None,
                 "trade_median_pct": float(np.median(rets)) if rets else None,
                 "win_share": float(np.mean([x > 0 for x in rets])) if rets else None,
                 "rand_time": rt, "rand_coin": rc, "rand_time_p95": _p95(rt), "rand_coin_p95": _p95(rc),
                 "r_by_year": _year_means(main["series"]), "conservation_ok": main["conservation_ok"]}
            ok, checks = admit(r)
            r["admitted"], r["checks"] = ok, checks
            res[name], series[name] = r, dict(main["series"])
            print(f"  {name:12s} R {agg.get('mean_pct', 0):+.4f} [{agg.get('lo95', 0):+.4f}, {agg.get('hi95', 0):+.4f}] | "
                  f"trades {r['n_trades']} trade {r['trade_mean_pct'] or 0:+.2f}% win {r['win_share'] or 0:.2f} | "
                  f"p95 time {r['rand_time_p95'] or 0:+.4f} coin {r['rand_coin_p95'] or 0:+.4f} | years "
                  f"{ {y: round(v, 4) for y, v in r['r_by_year'].items()} } | {'ADMIT' if ok else '-'} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        names = list(cands)
        days = sorted(set.intersection(*[set(series[n]) for n in names]))
        R = np.array([[series[n][d] for n in names] for d in days])
        pbo = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= 2 * PBO_SPLITS else None
        p4 = [x["pre4h"] for x in leak if x["pre4h"] is not None]
        p24 = [x["pre24h"] for x in leak if x["pre24h"] is not None]
        out["types"][et] = {"n_events": len(ev), "candidates": res, "pbo": pbo, "winner": select(res, pbo),
                            "leak": {"pre4h_mean_pct": float(np.mean(p4)) * 100 if p4 else None,
                                     "pre24h_mean_pct": float(np.mean(p24)) * 100 if p24 else None,
                                     "share_pre4h_ge_2_5": float(np.mean([x >= 0.025 for x in p4])) if p4 else None,
                                     "share_pre24h_ge_2_5": float(np.mean([x >= 0.025 for x in p24])) if p24 else None}}
        print(f"  -> pbo {pbo}, winner {out['types'][et]['winner']}, leak {out['types'][et]['leak']}", flush=True)
    out["winners"] = {et: v["winner"] for et, v in out["types"].items() if v.get("winner")}
    out["verdict"] = "winners -> test needs the operator" if out["winners"] else "STOP: no winner, test not opened"
    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat()
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "train":
        r = run_train()
        print(json.dumps({"winners": r["winners"], "verdict": r["verdict"]}, indent=1))
        return 0
    if cmd == "test":
        if "--operator-approved" not in argv:
            print("LA: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("LA: the test runner is written only if train produces a winner (spec §4)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
