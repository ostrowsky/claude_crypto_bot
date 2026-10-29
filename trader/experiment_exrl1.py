"""EX-RL1 phase 0 -- market vs limit-at-bid entry for the X4 book on the public aggTrades tape
(execution-bandit-exrl1-1029-spec.md §2-§4). Offline; nothing trades.

  fetch  X4 replay entries since 2026-03-01 + the live demo entries -> tapes [t, t + 17 min] in runtime/tape/
  run    calibration on the demo fills, arms A0 / A1 (limit 2 min then market) / A2 (limit 15 min else no trade),
         paired money difference vs A0 per trade, daily series, months, admission (§4.3) -> EXRL1_PHASE0.json

  (from trader/) ../pyembed/python.exe -u experiment_exrl1.py fetch
  (from trader/) ../pyembed/python.exe -u experiment_exrl1.py run
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exec_tape as ET            # noqa: E402
import trader_config as TC        # noqa: E402

S = 1000
LAG = 30 * S
A1_TIMEOUT = 120 * S
A2_TIMEOUT = 900 * S
WINDOW = 17 * 60 * S
DAY_FROM, DAY_TO = "2026-03-01", "2026-10-29"
ALPHA = 0.025
MIN_CHANGED = 100
MONTHS_AGREE = 5
RATE = 5.0
OUT = TC.RUNTIME / "experiments" / "EXRL1_PHASE0.json"
SPEC = "execution-bandit-exrl1-1029-spec.md"


def admit(res: dict):
    months = res.get("by_month") or {}
    checks = {"calibration_ok": bool(res.get("calibration_ok")),
              "lo_gt_0": (res.get("agg") or {}).get("lo95") is not None and res["agg"]["lo95"] > 0,
              "enough_changed": (res.get("n_changed") or 0) >= MIN_CHANGED,
              "months_agree": sum(1 for v in months.values() if v is not None and v >= 0) >= MONTHS_AGREE}
    return all(checks.values()), checks


def _ms(dt) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def x4_trades() -> list:
    import execution_replay as R
    import fill_model as FM
    import trading_policy as TP
    pol = replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))
    cands, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(cands, grid)
    r = R.run(pol, FM.model("taker", 1), grid, cands, floor_of=R._floor(), day_from=DAY_FROM, day_to=DAY_TO)
    return [{"sym": t["sym"], "t": _ms(t["entry_ts"]), "ret_pct": float(t["ret_pct"]),
             "day": datetime.fromtimestamp(_ms(t["entry_ts"]) / 1000, timezone.utc).strftime("%Y-%m-%d")}
            for t in r["trades"]]


def demo_entries() -> list:
    import audit_report as AR
    out = []
    for r in AR.load("decisions", 400):
        if (r.get("book") == "exec_x4" and r.get("mode") == "demo" and r.get("kind") == "real_entry"
                and r.get("action") == "enter" and isinstance(r.get("fill_price"), (int, float))):
            sig = str(r.get("key", "")).split("|")[-1]
            try:
                out.append({"sym": r["sym"], "t": _ms(sig), "fill": float(r["fill_price"]),
                            "lag": int(float(r.get("latency_sec") or 30) * S)})
            except Exception:
                continue
    return out


class _Limiter:
    def __init__(self, per_sec):
        self.gap, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def fetch(jobs, workers: int = 4) -> dict:
    import exchange_client as X
    lim, local = _Limiter(RATE), threading.local()

    class Limited:
        def agg_trades(self, *a, **k):
            lim.wait()
            if not hasattr(local, "c"):
                local.c = X.ExchangeClient(key="", secret="")
            return local.c.agg_trades(*a, **k)
    todo = sorted({(j["sym"], j["t"]) for j in jobs if not ET.have_tape(j["sym"], j["t"])})
    stat = {"todo": len(todo), "ok": 0, "empty": 0, "errors": 0}
    t0 = time.time()

    def one(st):
        sym, t = st
        try:
            tp = ET.fetch_tape(Limited(), sym, t, t + WINDOW)
            ET.save_tape(ET.tape_path(sym, t), tp)
            return "ok" if len(tp["T"]) else "empty"
        except Exception:                                          # noqa: BLE001
            return "errors"
    with ThreadPoolExecutor(workers) as ex:
        for n, k in enumerate(ex.map(one, todo), 1):
            stat[k] += 1
            if n % 200 == 0:
                print(f"  tapes {n}/{len(todo)} {stat} {(time.time() - t0) / 60:.1f} min", flush=True)
    return stat


def run() -> dict:
    import daily_revenue as DR
    trades, demo = x4_trades(), demo_entries()
    sim, act = [], []
    for d in demo:
        p = ET.tape_path(d["sym"], d["t"])
        if p.exists():
            a = ET.a0_price(ET.load_tape(p), d["t"], d["lag"])
            if a:
                sim.append(a)
                act.append(d["fill"])
    cal = ET.calibration(sim, act)
    rows, missing = [], 0
    for tr in trades:
        p = ET.tape_path(tr["sym"], tr["t"])
        if not p.exists():
            missing += 1
            continue
        tp = ET.load_tape(p)
        a0 = ET.a0_price(tp, tr["t"], LAG)
        if a0 is None:
            missing += 1
            continue
        a1 = ET.a1_price(tp, tr["t"], LAG, A1_TIMEOUT)
        f2, p2 = ET.a2_fill(tp, tr["t"], LAG, A2_TIMEOUT)
        ref = ET.limit_ref(tp, tr["t"], LAG)
        rows.append({**tr, "a0": a0, "a1": a1, "a2_filled": f2, "a2": p2, "ref": ref,
                     "d1": ET.pair_diff_bp(a0, a1, tr["ret_pct"], False) if a1 else None,
                     "d2": ET.pair_diff_bp(a0, p2, tr["ret_pct"], not f2)})
    out = {"spec": SPEC, "window": [DAY_FROM, DAY_TO], "lag_s": LAG / S, "n_trades_x4": len(trades),
           "n_with_tape": len(rows), "missing": missing, "calibration": cal, "arms": {}}
    for arm, key in (("A1", "d1"), ("A2", "d2")):
        rr = [r for r in rows if r[key] is not None]
        by_day, by_month = {}, {}
        for r in rr:
            by_day.setdefault(r["day"], []).append(r[key])
            by_month.setdefault(r["day"][:7], []).append(r[key])
        ser = sorted((d, float(np.mean(v))) for d, v in by_day.items())
        changed = sum(1 for r in rr if abs(r[key]) > 1e-9)
        res = {"n": len(rr), "n_changed": changed, "mean_bp": float(np.mean([r[key] for r in rr])) if rr else None,
               "agg": DR.aggregate(ser, alpha=ALPHA), "by_month": {m: float(np.mean(v)) for m, v in sorted(by_month.items())},
               "calibration_ok": cal["ok"]}
        if arm == "A1":
            res["limit_filled_share"] = float(np.mean([r["a1"] == r["ref"] and r["ref"] is not None for r in rr])) if rr else None
        else:
            f = [r for r in rr if r["a2_filled"]]
            u = [r for r in rr if not r["a2_filled"]]
            res["fill_share"] = len(f) / len(rr) if rr else None
            res["ret_filled_pct"] = float(np.mean([r["ret_pct"] for r in f])) if f else None
            res["ret_unfilled_pct"] = float(np.mean([r["ret_pct"] for r in u])) if u else None
        ok, checks = admit(res)
        res["admitted"], res["checks"] = ok, checks
        out["arms"][arm] = res
        a = res["agg"]
        print(f"{arm}: n {res['n']} changed {changed} mean {res['mean_bp']:+.2f} bp/trade | by day {a['mean_pct']:+.2f} "
              f"[{a['lo95']:+.2f}, {a['hi95']:+.2f}] bp | months { {m: round(v, 1) for m, v in res['by_month'].items()} } | "
              + (f"limit filled {res['limit_filled_share']:.2f}" if arm == "A1" else
                 f"filled {res['fill_share']:.2f}, ret filled {res['ret_filled_pct']:+.3f}% vs unfilled {res['ret_unfilled_pct']:+.3f}%")
              + f" | {'ADMIT' if ok else '-'}", flush=True)
    out["utc"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"calibration": cal, "admitted": {k: v["admitted"] for k, v in out["arms"].items()}}))
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        tr, de = x4_trades(), demo_entries()
        print(f"X4 trades {len(tr)}, demo entries {len(de)}", flush=True)
        print(json.dumps(fetch(de + tr)))
        return 0
    if cmd == "run":
        run()
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
