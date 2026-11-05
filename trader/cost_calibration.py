"""What a market order really costs on the MAIN market, by liquidity bucket (cost-calibration-mainnet-1105-spec.md).

  data       runtime/microstructure/*.jsonl (microstructure_recorder: bps from mid of a market order of
             100/500/1000/5000 USDT walking 20 book levels, hosts live and demo; 24h quote volume hourly)
  buckets    B1 BTC+ETH | B2 volume ranks 3-20 | B3 21-100 | B4 101+ (rank by the median live 24h quote volume)
  drift      demo entries: slippage from the signal price minus the demo book cost at the nearest snapshot (+-30 min)
  sensitivity  first-order per-trade adjustment vs the model's 8 bps per side (spec §4); no verdict is re-decided

  (from trader/) ../pyembed/python.exe -u cost_calibration.py run
"""
from __future__ import annotations

import bisect
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "cost-calibration-mainnet-1105-spec.md"
DATA = TC.RUNTIME / "microstructure"
OUT = TC.RUNTIME / "experiments" / "COST_CAL.json"
WINDOW = ("2026-10-10", "2026-11-04")          # full days only (§2)
SIZES = (100, 500, 1000, 5000)
B1_PAIRS = ("BTCUSDT", "ETHUSDT")
MIN_N = 50
MODEL_BPS = 8.0
DRIFT_WINDOW_SEC = 1800
SLOT_SHARE = 0.10                               # X4 equal slots: 10% of equity per position
BOOT, SEED = 2000, 29
X4_WINDOWS = (("h3_test", "2026-07-01", "2026-10-11"), ("e7_window", "2026-10-13", "2026-11-05"))


def bucket(sym: str, rank: int) -> str:
    if sym in B1_PAIRS:
        return "B1"
    if rank <= 20:
        return "B2"
    if rank <= 100:
        return "B3"
    return "B4"


def pair_stats(values, min_n: int = MIN_N):
    if len(values) < min_n:
        return None
    x = np.array([v for v in values if v is not None], dtype=float)
    return {"median": float(np.median(x)) if len(x) else None,
            "p90": float(np.percentile(x, 90)) if len(x) else None,
            "none_share": round(1 - len(x) / len(values), 4), "n": len(values)}


def bucket_stats(stats_list) -> dict:
    med = [s["median"] for s in stats_list if s and s["median"] is not None]
    p90 = [s["p90"] for s in stats_list if s and s["p90"] is not None]
    return {"median": float(np.median(med)) if med else None, "p90": float(np.percentile(p90, 90)) if p90 else None,
            "none_share": float(np.mean([s["none_share"] for s in stats_list])) if stats_list else None,
            "n_pairs": len(stats_list)}


def nearest_size(notional: float) -> int:
    return min(SIZES, key=lambda s: (abs(s - float(notional)), s))


def drift(entries, book_index: dict, window_sec: float = DRIFT_WINDOW_SEC):
    out, excluded = [], 0
    for e in entries:
        snaps = book_index.get(e["sym"]) or []
        ts = [s[0] for s in snaps]
        i = bisect.bisect_left(ts, e["ts"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(snaps) and abs(snaps[j][0] - e["ts"]) <= window_sec:
                if best is None or abs(snaps[j][0] - e["ts"]) < abs(best[0] - e["ts"]):
                    best = snaps[j]
        cost = None if best is None else best[1].get(nearest_size(e["notional"]))
        if cost is None or e.get("slippage_bps") is None:
            excluded += 1
            continue
        out.append(float(e["slippage_bps"]) - float(cost))
    return out, excluded


def trade_delta_pp(entry_bps: float, exit_bps: float, model_bps: float = MODEL_BPS) -> float:
    """Extra cost of one round trip versus the model, in percentage points (positive = dearer)."""
    return (float(entry_bps) + float(exit_bps) - 2 * model_bps) / 100


def iter_records(data_dir: Path, first_day: str, last_day: str):
    for f in sorted(Path(data_dir).glob("2*.jsonl")):
        if not (first_day <= f.stem <= last_day):
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def _mean_ci(x, boot: int = BOOT, seed: int = SEED):
    if len(x) < 10:
        return {"n": len(x), "mean": float(np.mean(x)) if x else None}
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choice(x) for _ in x) / len(x) for _ in range(boot))
    return {"n": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "lo95": bs[int(0.025 * boot)], "hi95": bs[int(0.975 * boot)], "ci_level": 0.95}


# ------------------------------------------------------------------ run
def collect(data_dir: Path = DATA, window=WINDOW):
    vals = {}                         # (host, sym, side, size) -> [bps | None]
    vol = {}                          # sym -> [quote volume]
    demo_index = {}                   # sym -> [(ts, {size: buy bps})]
    n = 0
    for r in iter_records(data_dir, *window):
        n += 1
        ts = _epoch(r["ts"])
        for host, d in (r.get("depth") or {}).items():
            for sym, sides in d.items():
                for side in ("buy", "sell"):
                    for sz, v in (sides.get(side) or {}).items():
                        vals.setdefault((host, sym, side, int(sz)), []).append(v)
                if host == "demo":
                    demo_index.setdefault(sym, []).append(
                        (ts, {int(k): v for k, v in (sides.get("buy") or {}).items()}))
        for sym, v in ((r.get("vol24h") or {}).get("live") or {}).items():
            vol.setdefault(sym, []).append(float(v[0]))
    for s in demo_index:
        demo_index[s].sort(key=lambda x: x[0])
    return vals, vol, demo_index, n


def tables(vals: dict, ranks: dict) -> dict:
    out = {}
    for host in ("live", "demo"):
        for side in ("buy", "sell"):
            for sz in SIZES:
                per_b = {}
                for (h, sym, sd, s), v in vals.items():
                    if (h, sd, s) != (host, side, sz) or sym not in ranks:
                        continue
                    st = pair_stats(v)
                    if st:
                        per_b.setdefault(bucket(sym, ranks[sym]), []).append(st)
                for b, lst in sorted(per_b.items()):
                    out.setdefault(host, {}).setdefault(b, {}).setdefault(side, {})[sz] = bucket_stats(lst)
    return out


def pair_medians(vals: dict, host: str, size: int) -> dict:
    out = {}
    for side in ("buy", "sell"):
        for (h, sym, sd, s), v in vals.items():
            if (h, sd, s) == (host, side, size):
                st = pair_stats(v)
                if st and st["median"] is not None:
                    out.setdefault(sym, {})[side] = st["median"]
    return out


def x4_sensitivity(bucket_cost, ranks, drift_bps: float) -> dict:
    import execution_replay as ER
    import fill_model as FM
    import _audit_k1_replay_windows as K
    trades, _ = ER.load_events()
    grid = ER._grid()
    ER.attach_grid(trades, grid)
    pol, fm, floor = K.x4_policy(), FM.model("taker", 1), ER._floor()
    res = {}
    for name, lo, hi in X4_WINDOWS:
        r = ER.run(pol, fm, grid, trades, floor_of=floor, day_from=lo, day_to=hi)
        d, by_b = [], {}
        for t in r["trades"]:
            b = bucket(t["sym"], ranks.get(t["sym"], 10 ** 6))
            buy, sell = bucket_cost[b]
            x = trade_delta_pp(buy + drift_bps, sell)
            d.append(x)
            by_b[b] = by_b.get(b, 0) + 1
        nd = r["agg"].get("n_days") or 1
        dc = ER.decompose(r["trades"])
        res[name] = {"window": [lo, hi], "n_days": nd, "trades": len(d), "by_bucket": by_b,
                     "net_per_trade": dc.get("net_mean_pct"), "gross_per_trade": dc.get("gross_mean_pct"),
                     "delta_per_trade_pp": float(np.mean(d)) if d else None,
                     "net_per_trade_new": (dc.get("net_mean_pct") or 0) - (float(np.mean(d)) if d else 0),
                     "R": r["agg"].get("mean_pct"),
                     "R_shift_approx": -sum(d) * SLOT_SHARE / nd,
                     "not_ranked": sum(1 for t in r["trades"] if t["sym"] not in ranks),
                     "n_censored": r.get("n_censored")}
    return res


def run() -> dict:
    import parity as P
    t0 = time.time()
    vals, vol, demo_index, n = collect()
    med_vol = {s: float(np.median(v)) for s, v in vol.items() if v}
    ranks = {s: i + 1 for i, (s, _) in enumerate(sorted(med_vol.items(), key=lambda x: -x[1]))}
    tab = tables(vals, ranks)
    rows = []
    for f in sorted((TC.RUNTIME / "audit" / "decisions").glob("*.jsonl")):
        rows += [json.loads(x) for x in f.open(encoding="utf-8") if x.strip()]
    entries = [{"sym": p["sym"], "ts": _epoch(p["entry_ts"]), "slippage_bps": p["slippage_bps"],
                "notional": p.get("cost") or 100.0}
               for p in P.pairs(rows) if p.get("slippage_bps") is not None and p.get("entry_ts")]
    d, excl = drift(entries, demo_index)
    dr = _mean_ci(d)
    drift_used = dr["mean"] if dr.get("lo95") is not None and (dr["lo95"] > 0 or dr["hi95"] < 0) else 0.0
    live = tab.get("live", {})
    cost1000 = {b: (live[b]["buy"][1000]["median"], live[b]["sell"][1000]["median"]) for b in live}
    pm = pair_medians(vals, "live", 1000)
    top100 = [s for s, r in ranks.items() if r <= 100 and s in pm and "buy" in pm[s] and "sell" in pm[s]]
    cap_delta = trade_delta_pp(float(np.mean([pm[s]["buy"] for s in top100])),
                               float(np.mean([pm[s]["sell"] for s in top100]))) if top100 else None
    out = {"spec": SPEC, "window": WINDOW, "snapshots": n, "pairs_ranked": len(ranks), "model_bps": MODEL_BPS,
           "tables": tab, "drift": {**dr, "excluded": excl, "used_bps": drift_used},
           "cost_1000": cost1000,
           "fc_b1_5000": {"buy": live.get("B1", {}).get("buy", {}).get(5000), "sell": live.get("B1", {}).get("sell", {}).get(5000)},
           "cap_top100_delta_pp": cap_delta, "cap_top100_pairs": len(top100)}
    print(json.dumps({k: out[k] for k in ("snapshots", "pairs_ranked", "drift", "cost_1000", "cap_top100_delta_pp")},
                     default=str), flush=True)
    import provenance as PV
    out["provenance"] = PV.block(data_files={f.name: PV.file_digest(f) for f in sorted(DATA.glob("2*.jsonl"))
                                             if WINDOW[0] <= f.stem <= WINDOW[1]})
    out["x4"] = x4_sensitivity(cost1000, ranks, drift_used)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["x4"], default=str))
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
