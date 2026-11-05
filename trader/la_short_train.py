"""LA narrow short, train step (la-short-feasibility-1105-spec.md, entry of 2026-11-05).

  events    OKX_PERP_NEW / BYBIT_PERP_NEW announcements on the LA train window whose coin already had a live Binance
            USD-M perp at t_pub (LA_SHORT_FEASIBILITY.json, status perp_live)
  trade     short the perp at the close of the first 15m perp bar opening >= t_pub + 1 min; exit 1 h / 4 h / 24 h later;
            perp taker 0.05% + 8 bps per side; funding received (positive rate) or paid over the hold; L = 1: a high
            >= entry * 1.99 inside the hold is a liquidation, -100%
  stats     mean trade, bootstrap over UTC days (4 000 draws, alpha 0.05/6); control: the same coin at a random time of
            the same month without an announcement within 48 h (50 seeds)
  power     2026 events are only COUNTED (no prices): one-sided power at the train effect and sd
  -> runtime/experiments/LA_SHORT_TRAIN.json. No 2026 price is read.

  (from trader/) ../pyembed/python.exe -u la_short_train.py run
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

SPEC = "la-short-feasibility-1105-spec.md"
OUT = TC.RUNTIME / "experiments" / "LA_SHORT_TRAIN.json"
BAR = 15 * 60 * 1000
DAY = 86_400_000
COST_SIDE = 0.0005 + 0.0008
LIQ_MULT = 1 + 1 - 0.01
HOLDS = {"h1": 4, "h4": 16, "h24": 96}
TYPES = ("OKX_PERP_NEW", "BYBIT_PERP_NEW")
K = len(TYPES) * len(HOLDS)
ALPHA = 0.05 / K
BOOT, SEED, N_RANDOM = 4000, 29, 50


def short_trade(bars: dict, t_pub: int, hold_bars: int, funding):
    t, c, h = bars["t"], bars["c"], bars["h"]
    e = int(np.searchsorted(t, t_pub + 60_000))
    x = e + int(hold_bars)
    if x >= len(t) or int(t[x]) != int(t[e]) + hold_bars * BAR or not (np.isfinite(c[e]) and np.isfinite(c[x])):
        return None
    entry, ex = float(c[e]), float(c[x])
    entry_close, exit_close = int(t[e]) + BAR, int(t[x]) + BAR
    fund = 0.0
    if funding is not None and len(funding["t"]):
        ft = np.asarray(funding["t"], dtype=np.int64)
        m = (ft > entry_close) & (ft <= exit_close)
        fund = float(np.sum(np.asarray(funding["rate"], dtype=float)[m]))
    hi = np.asarray(h[e + 1:x + 1], dtype=float)
    liq = bool(len(hi) and np.nanmax(hi) >= entry * LIQ_MULT)
    ret = -100.0 if liq else ((entry - ex) / entry - 2 * COST_SIDE + fund) * 100
    return {"ret_pct": ret, "funding_pct": fund * 100, "liquidated": liq, "entry_close_ms": entry_close}


def select(rows) -> list:
    return [r for r in rows if r.get("status") == "perp_live" and r.get("etype") in TYPES]


def day_cluster_ci(trades, alpha: float = ALPHA, boot: int = BOOT, seed: int = SEED) -> dict:
    if not trades:
        return {"n": 0}
    by = {}
    for t in trades:
        by.setdefault(int(t["entry_close_ms"]) // DAY, []).append(float(t["ret_pct"]))
    days = list(by.values())
    rnd = random.Random(seed)
    ms = []
    for _ in range(boot):
        s = n = 0.0
        for _ in range(len(days)):
            d = days[rnd.randrange(len(days))]
            s += sum(d)
            n += len(d)
        ms.append(s / n)
    ms.sort()
    return {"n": len(trades), "n_days": len(days), "mean": float(np.mean([t["ret_pct"] for t in trades])),
            "lo": ms[int(alpha / 2 * boot)], "hi": ms[min(boot - 1, int((1 - alpha / 2) * boot))],
            "ci_level": 1 - alpha, "tail_draws": int(alpha / 2 * boot)}


def power(mu: float, sd: float, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or sd <= 0:
        return 0.0
    return NormalDist().cdf(mu / (sd / math.sqrt(n)) - NormalDist().inv_cdf(1 - alpha))


def _bars(sym):
    import funding_store as FS
    p = FS.perp_path(sym)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {"t": z["t"].astype(np.int64), "c": z["c"].astype(float), "h": z["h"].astype(float)}


def run() -> dict:
    import announcements as AN
    import experiment_la as LA
    import experiment_xsmom as XE
    import funding_store as FS
    import la_short_feasibility as F
    import provenance as PV
    t0 = time.time()
    feas = json.loads(F.OUT.read_text(encoding="utf-8"))
    ev = select(feas["rows"])
    st = AN.AnnouncementStore()
    records = AN.drop_migrated([r for s in ("binance", "okx", "bybit") for r in st.load(s)])
    panel = XE.load_panel()
    first_bar = {}
    for s, j in panel.col.items():
        rows = np.flatnonzero(panel.has_bar[:, j])
        if len(rows):
            first_bar[s] = (int(panel.d0) + int(rows[0])) * LA.DAY
    all_ev = LA.mentions(records, set(first_bar))
    bars, fund = {}, {}

    def data(sym):
        if sym not in bars:
            bars[sym] = _bars(sym)
            fund[sym] = FS.load_funding(FS.funding_path(sym))
        return bars[sym], fund[sym]

    out = {"spec": SPEC, "stage": "train", "alpha": ALPHA, "k": K, "n_events": len(ev), "candidates": {}}
    for et in TYPES:
        e_t = [e for e in ev if e["etype"] == et]
        ctrl = {s: LA.control_events(e_t, {x["sym"]: True for x in e_t}, "time", s, all_ev) for s in range(N_RANDOM)}
        for hn, hb in HOLDS.items():
            tr = []
            for e in e_t:
                b, f = data(e["sym"])
                r = short_trade(b, e["t_pub"], hb, f) if b is not None else None
                if r is not None:
                    tr.append(r)
            rand = []
            for s, ce in ctrl.items():
                rr = []
                for e in ce:
                    b, f = data(e["sym"])
                    r = short_trade(b, e["t_pub"], hb, f) if b is not None else None
                    if r is not None:
                        rr.append(r["ret_pct"])
                if rr:
                    rand.append(float(np.mean(rr)))
            ci = day_cluster_ci(tr)
            p95 = float(np.percentile(rand, 95)) if rand else None
            liq = sum(1 for t in tr if t["liquidated"])
            rets = [t["ret_pct"] for t in tr]
            res = {"ci": ci, "rand_p95": p95, "rand_mean": float(np.mean(rand)) if rand else None,
                   "liquidations": liq, "sd": float(np.std(rets, ddof=1)) if len(rets) > 1 else None,
                   "median": float(np.median(rets)) if rets else None,
                   "pos_share": float(np.mean([x > 0 for x in rets])) if rets else None,
                   "funding_mean_pct": float(np.mean([t["funding_pct"] for t in tr])) if tr else None}
            res["admitted"] = bool(ci.get("lo") is not None and ci["lo"] > 0 and p95 is not None
                                   and ci["mean"] > p95 and liq == 0)
            out["candidates"][f"{et}|{hn}"] = res
            print(f"{et} {hn}: n {ci.get('n')} mean {ci.get('mean'):+.3f} [{ci.get('lo'):+.3f}, {ci.get('hi'):+.3f}] "
                  f"median {res['median']:+.3f} pos {res['pos_share']:.2f} | control mean {res['rand_mean']:+.3f} "
                  f"p95 {p95:+.3f} | liq {liq} | {'ADMIT' if res['admitted'] else '-'}", flush=True)
    # 2026 events are only counted: which coins had a live Binance perp at t_pub (no prices are read)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    ev26 = [e for e in LA.build_events(records, first_bar, LA._ms("2026-01-01"), now) if e["etype"] in TYPES]
    n26 = {}
    for e in ev26:
        b = bars.get(e["sym"]) if e["sym"] in bars else _bars(e["sym"])
        first = int(b["t"][0]) if b is not None and len(b["t"]) else None
        if F.status(e["t_pub"], first)[0] == "perp_live":
            n26[e["etype"]] = n26.get(e["etype"], 0) + 1
    out["n_2026_perp_live"] = n26
    out["power_2026"] = {}
    for name, res in out["candidates"].items():
        et = name.split("|")[0]
        if res["sd"]:
            out["power_2026"][name] = power(res["ci"]["mean"], res["sd"], n26.get(et, 0))
    out["admitted"] = [k for k, v in out["candidates"].items() if v["admitted"]]
    out["verdict"] = ("train: admitted " + ", ".join(out["admitted"])) if out["admitted"] else "STOP: 0 admitted on train"
    out["provenance"] = PV.block()
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"n_2026_perp_live": n26, "power_2026": out["power_2026"], "verdict": out["verdict"]}, indent=1))
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
