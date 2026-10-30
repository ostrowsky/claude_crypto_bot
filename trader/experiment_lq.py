"""LQ -- train of the long-liquidation-flush rebound (liquidation-flush-lq-1030-spec.md §3-§5).

  8 candidates PO_o{5|10}_p{3|6}_H{4|24} on 2023-03-01..2025-12-31; spot pairs with a perp in the metrics archive.
  Per candidate: mean trade (lq.trade_mean_ci: day-block bootstrap, alpha 0.05/8; the daily-mean series is kept beside it), random same-coin same-month bars (50 seeds),
  paired monthly difference vs price-only drops, years, 10-slot book; PBO over the 8 -> runtime/experiments/LQ_TRAIN.json

  (from trader/) ../pyembed/python.exe -u experiment_lq.py train
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

import lq                         # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM, TRAIN_TO = "2023-03-01", "2026-01-01"
ALPHA = 0.05 / 8
N_RANDOM = 50
MIN_EVENTS = 100
YEARS_AGREE = 2
PBO_MAX = 0.5
OUT = TC.RUNTIME / "experiments" / "LQ_TRAIN.json"
TRADES = TC.RUNTIME / "experiments" / "lq_trades.json"
SPEC = "liquidation-flush-lq-1030-spec.md"


def candidates() -> dict:
    return {f"PO_o{o}_p{p}_H{h}": {"oi": -o / 100, "p": -p / 100, "H": h} for o in (5, 10) for p in (3, 6) for h in (4, 24)}


def admit(res: dict):
    years = res.get("by_year") or {}
    a = res.get("agg") or {}
    checks = {"mean_lo_gt_0": a.get("lo95") is not None and a["lo95"] > 0,
              "beats_random_p95": res.get("rand_p95") is not None and a.get("mean_pct") is not None
              and a["mean_pct"] > res["rand_p95"],
              "beats_price_only": (res.get("vs_p") or {}).get("lo95") is not None and res["vs_p"]["lo95"] > 0,
              "enough_events": (res.get("n_events") or 0) >= MIN_EVENTS,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def _ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def run_train() -> dict:
    import daily_revenue as DR
    import experiment_la as EL
    import fill_model as FM
    import history_store as HS
    import metrics_store as MS
    import pbo as PB
    t0 = time.time()
    fm = FM.model("taker", 1)
    lo_ms, hi_ms = _ms(TRAIN_FROM), _ms(TRAIN_TO)
    cands = candidates()
    trades = {n: {"PO": [], "P": [], "R": [[] for _ in range(N_RANDOM)]} for n in cands}
    mp = MS.mapping()
    pairs = 0
    for k, (spot, info) in enumerate(sorted(mp.items()), 1):
        m = MS.load(info["perp"])
        a = HS.load(HS.path_of(spot))
        if m is None or a is None or not len(m["t"]) or not len(a["t"]):
            continue
        t15, c15 = np.asarray(a["t"], dtype=np.int64), np.asarray(a["c"], float)
        sel = (t15 >= lo_ms - 2 * lq.DAY) & (t15 < hi_ms)
        t15, c15 = t15[sel], c15[sel]
        if len(t15) < 200:
            continue
        pairs += 1
        delisted = int(np.asarray(a["t"]).max()) < hi_ms - lq.DAY
        dp = lq.price_change_1h(c15)
        doi = lq.oi_change_1h(m, t15 + lq.BAR)
        win = t15 >= lo_ms
        for name, cd in cands.items():
            hold = cd["H"] * 4
            for kind in ("PO", "P"):
                for i in lq.events(t15, np.where(win, dp, np.nan), doi, cd["oi"], cd["p"], kind):
                    x = lq.trade(t15, c15, i, hold, fm, delisted)
                    if x is not None and x["exit_close_ms"] <= hi_ms + lq.DAY:
                        trades[name][kind].append({**x, "sym": spot})
            po_idx = lq.events(t15, np.where(win, dp, np.nan), doi, cd["oi"], cd["p"], "PO")
            for s in range(N_RANDOM):
                for i in lq.random_bars(t15, po_idx, len(po_idx), 7000 + s):
                    x = lq.trade(t15, c15, i, hold, fm, delisted)
                    if x is not None:
                        trades[name]["R"][s].append(x["ret_pct"])
        if k % 50 == 0:
            print(f"  pairs {k}/{len(mp)} ({pairs} used), {(time.time() - t0) / 60:.1f} min", flush=True)
    TRADES.write_text(json.dumps({n: {"PO": [{k: x[k] for k in ("sym", "ret_pct", "entry_close_ms", "exit_close_ms")}
                                               for x in tr["PO"]],
                                        "P": [{k: x[k] for k in ("sym", "ret_pct", "entry_close_ms")} for x in tr["P"]],
                                        "R": tr["R"]} for n, tr in trades.items()}), encoding="utf-8")
    res, daily = {}, {}
    for name, tr in trades.items():
        po, p = tr["PO"], tr["P"]
        by_day, by_month_po, by_month_p, by_year = {}, {}, {}, {}
        for x in po:
            d = _day(x["entry_close_ms"])
            by_day.setdefault(d, []).append(x["ret_pct"])
            by_month_po.setdefault(d[:7], []).append(x["ret_pct"])
            by_year.setdefault(d[:4], []).append(x["ret_pct"])
        for x in p:
            by_month_p.setdefault(_day(x["entry_close_ms"])[:7], []).append(x["ret_pct"])
        ser = sorted((d, float(np.mean(v))) for d, v in by_day.items())
        vs = sorted((mo, float(np.mean(by_month_po[mo]) - np.mean(by_month_p[mo]))) for mo in by_month_po if mo in by_month_p)
        rand_means = [float(np.mean(r)) for r in tr["R"] if r]
        book = EL.book(po, TRAIN_FROM, TRAIN_TO)
        r = {"config": cands[name], "n_events": len(po), "n_price_only": len(p),
             "agg": lq.trade_mean_ci(po, alpha=ALPHA), "daily_mean_agg": DR.aggregate(ser, alpha=ALPHA),
             "price_only_agg": lq.trade_mean_ci(p, alpha=ALPHA),
             "trade_mean_pct": float(np.mean([x["ret_pct"] for x in po])) if po else None,
             "price_only_mean_pct": float(np.mean([x["ret_pct"] for x in p])) if p else None,
             "vs_p": DR.aggregate(vs, alpha=ALPHA), "rand_p95": float(np.percentile(rand_means, 95)) if rand_means else None,
             "rand_mean": float(np.mean(rand_means)) if rand_means else None,
             "by_year": {y: float(np.mean(v)) for y, v in sorted(by_year.items())},
             "book_R": book["series"] and DR.aggregate(book["series"]), "book_conservation_ok": book["conservation_ok"]}
        ok, checks = admit(r)
        r["admitted"], r["checks"] = ok, checks
        res[name], daily[name] = r, dict(ser)
        a = r["agg"]
        dm = r["daily_mean_agg"]
        print(f"{name}: events {r['n_events']} (price-only {r['n_price_only']}) | trade mean {a.get('mean_pct')} "
              f"[{a.get('lo95')}, {a.get('hi95')}] top5 days {a.get('top5_days_share_of_sum')} | daily-mean {dm.get('mean_pct')} "
              f"| price-only {r['price_only_mean_pct']} "
              f"vs_p {r['vs_p'].get('mean_pct')} [{r['vs_p'].get('lo95')}, {r['vs_p'].get('hi95')}] | rand p95 {r['rand_p95']} "
              f"| years { {y: round(v, 3) for y, v in r['by_year'].items()} } | {'ADMIT' if ok else '-'}", flush=True)
    names = list(res)
    days = sorted(set().union(*[set(daily[n]) for n in names]))
    R = np.array([[daily[n].get(d, 0.0) for n in names] for d in days])
    pbo = float(PB.cscv_pbo(R, 8)) if len(days) >= 16 else None
    adm = [n for n in names if res[n]["admitted"]]
    winner = max(adm, key=lambda n: res[n]["agg"]["mean_pct"]) if (adm and pbo is not None and pbo < PBO_MAX) else None
    out = {"spec": SPEC, "window": [TRAIN_FROM, TRAIN_TO], "pairs": pairs, "alpha": ALPHA, "candidates": res, "pbo": pbo,
           "winner": winner, "verdict": "winner -> 2020-2022 holdout, then 2026 by the operator" if winner else
           "STOP: no winner", "minutes": round((time.time() - t0) / 60, 1), "utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pbo": pbo, "winner": winner, "verdict": out["verdict"]}))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["train"]:
        run_train()
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
