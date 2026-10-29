"""RB -- train of periodic rebalancing vs holding the same initial weights (rebalancing-rb-1029-spec.md §2-§3).

  train  6 candidates (BTC/ETH, BTC/USDT, ETH/USDT 50/50 x daily / weekly) on 2023-03-01 .. 2025-12-31 (panel cut
         before 2026); paired RB - BH of the same mix, alpha 0.05/6, years, PBO -> runtime/experiments/RB_TRAIN.json
  test   opened once and only on the operator's decision (marker RB_TEST_OPENED.json)

  (from trader/) ../pyembed/python.exe -u experiment_rb.py train
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rebalance as RB            # noqa: E402
import trader_config as TC        # noqa: E402

TRAIN_FROM, TEST_FROM = "2023-03-01", "2026-01-01"
ALPHA = 0.05 / 6
YEARS_AGREE = 2
PBO_MAX = 0.5
PBO_SPLITS = 8
OUT = TC.RUNTIME / "experiments"
TRAIN_OUT = OUT / "RB_TRAIN.json"
TEST_MARKER = OUT / "RB_TEST_OPENED.json"
SPEC = "rebalancing-rb-1029-spec.md"


def candidates() -> dict:
    mixes = {"BE": {"BTCUSDT": 0.5, "ETHUSDT": 0.5}, "BU": {"BTCUSDT": 0.5, "USDT": 0.5},
             "EU": {"ETHUSDT": 0.5, "USDT": 0.5}}
    return {f"RB_{m}_{'d1' if p == 1 else 'd7'}": {"weights": w, "period": p} for m, w in mixes.items() for p in (1, 7)}


def admit(res: dict):
    years = res.get("paired_by_year") or {}
    checks = {"paired_lo_gt_0": (res.get("paired") or {}).get("lo95") is not None and res["paired"]["lo95"] > 0,
              "years_agree": sum(1 for v in years.values() if v is not None and v >= 0) >= YEARS_AGREE}
    return all(checks.values()), checks


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("RB: test already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def _series(eq, days):
    return [(days[i], (eq[i + 1] / eq[i] - 1) * 100) for i in range(len(eq) - 1) if eq[i] > 0]


def run_train() -> dict:
    import daily_revenue as DR
    import experiment_xsmom as EX
    import fill_model as FM
    import pbo as PB
    import xsmom as XS
    P = EX.load_panel().cut(TEST_FROM)
    r0, r1 = P.row(TRAIN_FROM), P.n
    days = [XS.day_str(P.d0 + r) for r in range(r0, r1)]
    px = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        j = P.col[s]
        px[s] = np.where(P.has_first[r0:r1, j], P.first_close[r0:r1, j], np.nan)
    fm = FM.model("taker", 1)
    res, diffs = {}, {}
    for name, cd in candidates().items():
        w = cd["weights"]
        sub = {s: px[s] for s in w if s != "USDT"}
        rb, bh = RB.simulate(sub, w, cd["period"], fm), RB.simulate(sub, w, None, fm)
        sr, sb = _series(rb["equity"], days), _series(bh["equity"], days)
        hb = dict(sb)
        dd = [(d, v - hb[d]) for d, v in sr if d in hb]
        by = {}
        for d, v in dd:
            by.setdefault(d[:4], []).append(v)
        e = {"config": cd, "R": DR.aggregate(sr), "BH": DR.aggregate(sb), "paired": DR.paired(sr, sb, alpha=ALPHA),
             "paired_by_year": {y: float(np.mean(v)) for y, v in sorted(by.items())}, "fees_pct_of_start": rb["fees"] / 1e4,
             "turnover_x": rb["turnover"] / 1e6, "n_rebalances": len(rb["trade_days"])}
        ok, checks = admit(e)
        e["admitted"], e["checks"] = ok, checks
        res[name], diffs[name] = e, dict(dd)
        p = e["paired"]
        print(f"{name}: R {e['R']['mean_pct']:+.4f} BH {e['BH']['mean_pct']:+.4f} | RB-BH {p['mean_pct']:+.4f} "
              f"[{p['lo95']:+.4f}, {p['hi95']:+.4f}] %/day | years { {y: round(v, 4) for y, v in e['paired_by_year'].items()} } "
              f"| rebalances {e['n_rebalances']} turnover {e['turnover_x']:.1f}x fees {e['fees_pct_of_start']:.2f}% | "
              f"{'ADMIT' if ok else '-'}", flush=True)
    names = list(res)
    common = sorted(set.intersection(*[set(diffs[n]) for n in names]))
    pbo = float(PB.cscv_pbo(np.array([[diffs[n][d] for n in names] for d in common]), PBO_SPLITS))
    adm = [n for n in names if res[n]["admitted"]]
    winner = max(adm, key=lambda n: res[n]["paired"]["mean_pct"]) if (adm and pbo < PBO_MAX) else None
    out = {"spec": SPEC, "stage": "train", "window": [TRAIN_FROM, TEST_FROM], "alpha": ALPHA, "candidates": res,
           "pbo": pbo, "winner": winner,
           "verdict": "winner -> test needs the operator" if winner else "STOP: no winner, test not opened",
           "utc": datetime.now(timezone.utc).isoformat()}
    OUT.mkdir(parents=True, exist_ok=True)
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
            print("RB: the 2026 window is opened once and only on the operator's decision; add --operator-approved")
            return 2
        raise SystemExit("RB: the test runner is written only if train produces a winner (spec §3)")
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
