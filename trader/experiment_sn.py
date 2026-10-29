"""SN -- train / one-time test of shorting new Binance perps after launch (short-new-perps-n6-1029-spec.md §6-§7).

  train  8 candidates SN_d{0|96}_H{1|7|14|30} on launches 2023-01-08 .. 2025-12-31 with exits before 2026; 10-slot
         book (experiment_la.book), carry.fees(1), funding; 50 seeds of the same shorts on random OLD perps; PBO
         -> runtime/experiments/SN_TRAIN.json
  test   opened ONCE by the operator (marker SN_TEST_OPENED.json, k = 1); fewer than 15 trades -> TOO_EARLY

  (from trader/) ../pyembed/python.exe -u experiment_sn.py train
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

import shortperp as S             # noqa: E402
import trader_config as TC        # noqa: E402

CUTOFF = "2023-01-08"
TRAIN_FROM, TEST_FROM = "2023-01-08", "2026-01-01"
N_RANDOM = 50
MIN_TRADES = 100
LIQ_SHARE_MAX = 0.05
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
MIN_TEST_TRADES = 15
START_EQUITY = 1_000_000.0
OUT = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT / "SN_TRAIN.json"
TEST_MARKER = OUT / "SN_TEST_OPENED.json"
SPEC = "short-new-perps-n6-1029-spec.md"


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def candidates() -> dict:
    return {f"SN_d{d}_H{h}": {"d": d, "H": h} for d in (0, 96) for h in (1, 7, 14, 30)}


def _p95(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 95)) if xs else None


def admit(res: dict):
    agg = res.get("agg") or {}
    p95 = _p95(res.get("rand") or [])
    years = res.get("r_by_year") or {}
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0,
              "beats_random_old_p95": p95 is not None and agg.get("mean_pct") is not None and agg["mean_pct"] > p95,
              "enough_trades": (res.get("n_trades") or 0) >= MIN_TRADES,
              "liq_share_ok": (res.get("liq_share") if res.get("liq_share") is not None else 1.0) <= LIQ_SHARE_MAX,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def select(res: dict, pbo):
    if pbo is None or pbo >= PBO_MAX:
        return None
    adm = [n for n, v in res.items() if v.get("admitted")]
    return max(adm, key=lambda n: (res[n]["agg"]["mean_pct"], n)) if adm else None


def decide(agg: dict, rand, liq_share: float, n_trades: int) -> dict:
    if n_trades < MIN_TEST_TRADES:
        return {"verdict": "TOO_EARLY", "n_trades": n_trades}
    p95 = _p95(rand)
    checks = {"R_lo_gt_0": agg.get("lo95") is not None and agg["lo95"] > 0,
              "beats_random_old_p95": p95 is not None and agg.get("mean_pct") is not None and agg["mean_pct"] > p95,
              "liq_share_ok": liq_share <= LIQ_SHARE_MAX}
    return {"verdict": "ACCEPTED" if all(checks.values()) else "REJECTED", "checks": checks, "random_p95": p95,
            "n_trades": n_trades}


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("SN: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def _year_means(series):
    by = {}
    for d, v in series:
        by.setdefault(d[:4], []).append(v)
    return {y: sum(v) / len(v) for y, v in sorted(by.items())}


def _first_bars() -> dict:
    import funding_store as FS
    out = {}
    for p in sorted(FS.PERP_DIR.glob("*_15m.npz")):
        with np.load(p) as z:
            t = z["t"]
            if len(t):
                out[p.name[:-len("_15m.npz")]] = int(t.min())
    return out


def run_train(n_random: int = N_RANDOM) -> dict:
    import carry as C
    import daily_revenue as DR
    import funding_store as FS
    import history_store as HS
    import pbo as PB
    fees = C.fees(1)
    t0 = time.time()
    first = _first_bars()
    lau = S.launches(first, _ms(CUTOFF))
    test_from = _ms(TEST_FROM)
    lau_train = {s: t for s, t in lau.items() if t < test_from}
    print(f"perps {len(first)}, launches after {CUTOFF}: {len(lau)}, before {TEST_FROM}: {len(lau_train)}", flush=True)
    cands = candidates()
    # entries per d (launch + d bars), then the controls per (d, seed): same entry time, random old perp
    bars_cache = {}

    def load(sym):
        if sym not in bars_cache:
            a = HS.load(FS.perp_path(sym))
            f = FS.load_funding(FS.funding_path(sym))
            if f is not None:
                o = np.argsort(f["t"], kind="stable")
                f = {"t": np.asarray(f["t"], dtype=np.int64)[o], "rate": np.asarray(f["rate"], dtype=float)[o]}
            bars_cache[sym] = (a, f)
        return bars_cache[sym]

    entries = {}
    for d in (0, 96):
        ev = []
        for s, lt in sorted(lau_train.items()):
            a, _ = load(s)
            if a is None:
                continue
            em = S.entry_ms_for(a, lt, d)
            if em is not None:
                ev.append({"sym": s, "entry_ms": em, "launch_ms": lt})
            bars_cache.clear()
        entries[d] = ev
    ctrl = {(d, k): S.control(entries[d], first, lau, 7000 + 100 * d + k) for d in (0, 96) for k in range(n_random)}
    need = {}                                                 # sym -> set of entry_ms
    for evs in list(entries.values()) + list(ctrl.values()):
        for e in evs:
            need.setdefault(e["sym"], set()).add(e["entry_ms"])
    trades = {}                                               # (sym, entry_ms, H) -> trade
    for n_s, (sym, ems) in enumerate(sorted(need.items()), 1):
        a, f = load(sym)
        if a is not None:
            for em in ems:
                for H in (1, 7, 14, 30):
                    x = S.trade(a, f, em, H, fees)
                    if x is not None and S.in_train(x, test_from):
                        trades[(sym, em, H)] = {**x, "sym": sym}
        bars_cache.clear()
        if n_s % 100 == 0:
            print(f"  trades: {n_s}/{len(need)} perps, {len(trades)} trades, {(time.time() - t0) / 60:.1f} min", flush=True)
    res, series = {}, {}
    for name, cd in cands.items():
        sel = lambda evs: [trades[(e["sym"], e["entry_ms"], cd["H"])] for e in evs  # noqa: E731
                           if (e["sym"], e["entry_ms"], cd["H"]) in trades]
        main = S.book(sel(entries[cd["d"]]), TRAIN_FROM, TEST_FROM)
        agg = DR.aggregate(main["series"])
        rand = []
        for k in range(n_random):
            s_ = S.book(sel(ctrl[(cd["d"], k)]), TRAIN_FROM, TEST_FROM)["series"]
            rand.append(sum(v for _, v in s_) / len(s_) if s_ else None)
        tr = main["trades"]
        liq = sum(1 for x in tr if x["reason"] == "liquidation")
        r = {"config": cd, "agg": agg, "n_trades": len(tr), "liquidations": liq, "liq_share": liq / len(tr) if tr else None,
             "skipped_full": main["skipped_full"], "trade_mean_pct": float(np.mean([x["ret_pct"] for x in tr])) if tr else None,
             "trade_median_pct": float(np.median([x["ret_pct"] for x in tr])) if tr else None,
             "win_share": float(np.mean([x["ret_pct"] > 0 for x in tr])) if tr else None,
             "funding_mean_pct": float(np.mean([x["funding"] for x in tr])) * 100 if tr else None,
             "rand": rand, "random_p95": _p95(rand), "random_mean": float(np.mean([x for x in rand if x is not None])),
             "r_by_year": _year_means(main["series"]), "conservation_ok": main["conservation_ok"]}
        ok, checks = admit(r)
        r["admitted"], r["checks"] = ok, checks
        res[name], series[name] = r, dict(main["series"])
        print(f"{name:11s} R {agg['mean_pct']:+.4f} [{agg['lo95']:+.4f}, {agg['hi95']:+.4f}] dd {agg['max_dd_pct']:+.1f}% | "
              f"trades {r['n_trades']} liq {liq} | trade mean {r['trade_mean_pct']:+.2f}% med {r['trade_median_pct']:+.2f}% "
              f"win {r['win_share']:.2f} fund {r['funding_mean_pct']:+.2f}% | old-perp mean {r['random_mean']:+.4f} "
              f"p95 {r['random_p95']:+.4f} | years { {y: round(v, 4) for y, v in r['r_by_year'].items()} } | "
              f"{'ADMIT' if ok else '-'}", flush=True)
    names = list(cands)
    days = sorted(set.intersection(*[set(series[n]) for n in names]))
    R = np.array([[series[n][d] for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, PBO_SPLITS)) if len(days) >= 2 * PBO_SPLITS else None
    winner = select(res, pbo)
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "n_perps": len(first),
           "n_launches_train": len(lau_train), "entries": {d: len(v) for d, v in entries.items()}, "candidates": res,
           "pbo": pbo, "winner": winner,
           "verdict": "winner -> test needs the operator" if winner else "STOP: no winner, test not opened",
           "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pbo": pbo, "winner": winner, "verdict": out["verdict"]}))
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
            print("SN: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("SN: the test runner is written only if train produces a winner (spec §7)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
