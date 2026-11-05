"""LA sign check on train (fourth-review-1105-spec.md §2): is "the fall after a perp-listing announcement came from coins
without a Binance perp" true, event by event? Gross returns only; no 2026 price is read.

  gross_pair   spot long and perp short on the same event, each on its own 15m closes
  split_diff   spot long of perp_live events vs the rest, bootstrap over UTC days
  run          OKX_PERP_NEW, BYBIT_PERP_NEW x 1 / 4 / 24 h -> runtime/experiments/LA_SIGN_CHECK.json

  (from trader/) ../pyembed/python.exe -u la_sign_check.py run
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "fourth-review-1105-spec.md"
OUT = TC.RUNTIME / "experiments" / "LA_SIGN_CHECK.json"
BAR = 15 * 60 * 1000
DAY = 86_400_000
TYPES = ("OKX_PERP_NEW", "BYBIT_PERP_NEW")
HOLDS = {"h1": 4, "h4": 16, "h24": 96}
ALPHA = 0.05 / 6
BOOT, SEED = 4000, 29


def _leg(b, t_pub: int, hold: int):
    t, c = b["t"], b["c"]
    e = int(np.searchsorted(t, t_pub + 60_000))
    x = e + int(hold)
    if x >= len(t) or int(t[x]) != int(t[e]) + hold * BAR or not (np.isfinite(c[e]) and np.isfinite(c[x])) or c[e] <= 0:
        return None
    return float(c[e]), float(c[x]), int(t[e])


def gross_pair(spot, perp, t_pub: int, hold_bars: int):
    a, b = _leg(spot, t_pub, hold_bars), _leg(perp, t_pub, hold_bars)
    if a is None or b is None:
        return None
    return {"long_pct": (a[1] / a[0] - 1) * 100, "short_pct": (b[0] - b[1]) / b[0] * 100, "day": a[2] // DAY}


def split_diff(rows, alpha: float = ALPHA, boot: int = BOOT, seed: int = SEED) -> dict:
    live = [r for r in rows if r["group"] == "live"]
    rest = [r for r in rows if r["group"] == "rest"]
    out = {"n_live": len(live), "n_rest": len(rest), "live": float(np.mean([r["long_pct"] for r in live])) if live else None,
           "rest": float(np.mean([r["long_pct"] for r in rest])) if rest else None, "diff": None, "lo": None, "hi": None,
           "ci_level": 1 - alpha}
    if not live or not rest:
        return out
    out["diff"] = out["rest"] - out["live"]
    by = {}
    for r in rows:
        by.setdefault(r["day"], []).append(r)
    days = list(by.values())
    rnd = random.Random(seed)
    ds = []
    for _ in range(boot):
        s = {"live": [0.0, 0], "rest": [0.0, 0]}
        for _ in range(len(days)):
            for r in days[rnd.randrange(len(days))]:
                s[r["group"]][0] += r["long_pct"]
                s[r["group"]][1] += 1
        if s["live"][1] and s["rest"][1]:
            ds.append(s["rest"][0] / s["rest"][1] - s["live"][0] / s["live"][1])
    ds.sort()
    out["lo"], out["hi"] = ds[int(alpha / 2 * len(ds))], ds[min(len(ds) - 1, int((1 - alpha / 2) * len(ds)))]
    return out


def claim_holds(d: dict) -> bool:
    return d.get("hi") is not None and d["hi"] < 0


def _spot(sym):
    import history_store as HS
    a = HS.load(HS.path_of(sym))
    if a is None or not len(a["t"]):
        return None
    return {"t": np.asarray(a["t"], dtype=np.int64), "c": np.asarray(a["c"], dtype=float)}


def _perp(sym):
    import funding_store as FS
    p = FS.perp_path(sym)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {"t": z["t"].astype(np.int64), "c": z["c"].astype(float)}


def run() -> dict:
    import la_short_feasibility as F
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    rows = [r for r in json.loads(F.OUT.read_text(encoding="utf-8"))["rows"] if r["etype"] in TYPES]
    spots, perps = {}, {}
    out = {"spec": SPEC, "n_events": len(rows), "alpha": ALPHA, "cells": {}}
    for et in TYPES:
        for hn, hb in HOLDS.items():
            pairs, split = [], []
            for r in (x for x in rows if x["etype"] == et):
                s = r["sym"]
                if s not in spots:
                    spots[s], perps[s] = _spot(s), _perp(s)
                if spots[s] is None:
                    continue
                if r["status"] == "perp_live" and perps[s] is not None:
                    g = gross_pair(spots[s], perps[s], r["t_pub"], hb)
                    if g:
                        pairs.append(g)
                        split.append({"group": "live", "long_pct": g["long_pct"], "day": g["day"]})
                elif r["status"] != "perp_live":
                    a = _leg(spots[s], r["t_pub"], hb)
                    if a:
                        split.append({"group": "rest", "long_pct": (a[1] / a[0] - 1) * 100, "day": a[2] // DAY})
            lp = np.array([g["long_pct"] for g in pairs])
            sp = np.array([g["short_pct"] for g in pairs])
            d = split_diff(split)
            cell = {"n_pairs": len(pairs), "mirror": {"long_mean": float(lp.mean()), "short_mean": float(sp.mean()),
                                                     "sum_mean": float((lp + sp).mean()),
                                                     "corr_long_vs_minus_short": float(np.corrcoef(lp, -sp)[0, 1])},
                    "split": d, "claim_holds": claim_holds(d)}
            out["cells"][f"{et}|{hn}"] = cell
            print(f"{et} {hn}: pairs {len(pairs)} long {lp.mean():+.3f} short {sp.mean():+.3f} sum {(lp + sp).mean():+.3f} "
                  f"corr {cell['mirror']['corr_long_vs_minus_short']:.4f} | spot long live {d['live']:+.3f} (n {d['n_live']}) "
                  f"rest {d['rest']:+.3f} (n {d['n_rest']}) diff {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] "
                  f"-> {'HOLDS' if cell['claim_holds'] else 'not confirmed'}", flush=True)
    out["claim_holds_anywhere"] = [k for k, v in out["cells"].items() if v["claim_holds"]]
    out["provenance"] = PV.block()
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
