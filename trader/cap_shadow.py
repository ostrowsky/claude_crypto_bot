"""CAP-SHADOW -- forward paper book of the frozen CAP_W4_X50_H24 (cap-shadow-1104-spec.md).

  step       pure: the closed bars after the last processed one -> events (cap.breadth / cap.events rules) and, once
             the exit bar has closed, results (cap.basket) with a matched random control of 50 bars of the same window
  run_step   IO: candles of the W1 scanner (runtime/scan/bars), universe from their daily panel (xsmom.universe, age
             >= 60 days from history_store, no stablecoins), state + append-only ledger in runtime/cap_shadow/
  readout    COLLECTING / KEEP / REJECT / INCONCLUSIVE / TOO_FEW (§4) -> runtime/cap_shadow/READOUT_CAP.json
Paper only: no key, no order. Rollback: trader_config.CAP_SHADOW_ENABLED = False.

  (from trader/) ../pyembed/python.exe cap_shadow.py step
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cap                        # noqa: E402
import trader_config as TC        # noqa: E402

BAR, DAY = cap.BAR, cap.DAY
W, Y, X, HOLD = 16, 0.10, 0.50, 96          # CAP_W4_X50_H24, frozen
N_RAND = 50
MIN_AGE_DAYS = 60
MIN_RESULTS = 15
TOO_FEW_DAY = "2027-11-04"
DIR = TC.RUNTIME / "cap_shadow"
SPEC = "cap-shadow-1104-spec.md"


def _grid(bars: dict, now_ms: int) -> np.ndarray:
    starts = [int(t[0]) for t, _ in bars.values() if len(t)]
    if not starts:
        return np.zeros(0, dtype=np.int64)
    last_open = (int(now_ms) // BAR) * BAR - BAR           # the last bar whose close is <= now
    return np.arange(min(starts) // BAR * BAR, last_open + 1, BAR, dtype=np.int64)


def _syms(uni, t_ms) -> list:
    return sorted(uni.get(int(t_ms) // DAY, ()))


def step(bars: dict, uni: dict, state: dict, now_ms: int, fm):
    st = {"last_bar_ms": None, "next_allowed_ms": None, "open": []}
    st.update(json.loads(json.dumps(state or {})))
    grid = _grid(bars, now_ms)
    rows = []
    if len(grid) < W + 2:
        return st, rows
    share = cap.breadth(grid, bars, uni, W, Y)
    for i in range(len(grid) - 1):                         # bar i is decided once bar i + 1 has closed
        t = int(grid[i])
        if st["last_bar_ms"] is not None and t <= st["last_bar_ms"]:
            continue
        if np.isfinite(share[i]) and share[i] >= X and (st["next_allowed_ms"] is None or t >= st["next_allowed_ms"]):
            rows.append({"kind": "event", "t_ms": t, "share": float(share[i]), "n_universe": len(_syms(uni, t))})
            st["open"].append(t)
            st["next_allowed_ms"] = t + (1 + HOLD) * BAR
        st["last_bar_ms"] = t
    keep = []
    for t in st["open"]:
        i = int(np.searchsorted(grid, t))
        if i >= len(grid) or int(grid[i]) != t:
            rows.append({"kind": "lost", "t_ms": t, "why": "event bar left the scanner window"})
            continue
        if i + 1 + HOLD > len(grid) - 1:
            keep.append(t)
            continue
        b = cap.basket(grid, i, HOLD, bars, _syms(uni, t), fm)
        rnd = random.Random(t)
        idx = [rnd.randint(0, len(grid) - 2 - HOLD) for _ in range(N_RAND)]
        rm = [cap.basket(grid, j, HOLD, bars, _syms(uni, grid[j]), fm) for j in idx]
        rm = [x["ret_pct"] for x in rm if x is not None]
        rows.append({"kind": "result", "t_ms": t, "ret_pct": b["ret_pct"] if b else None,
                     "n_pairs": b["n_pairs"] if b else 0, "rand_mean_pct": float(np.mean(rm)) if rm else None,
                     "exit_close_ms": b["exit_close_ms"] if b else None})
    st["open"] = keep
    return st, rows


def readout(results, today: str) -> dict:
    r = [x for x in results if x.get("kind") == "result" and x.get("ret_pct") is not None]
    n = len(r)
    out = {"spec": SPEC, "today": today, "n_results": n, "min_results": MIN_RESULTS}
    if n < MIN_RESULTS:
        out["status"] = "TOO_FEW" if today >= TOO_FEW_DAY else "COLLECTING"
        return out
    agg = cap.mean_ci([x["ret_pct"] for x in r], alpha=0.05)
    paired = cap.mean_ci([x["ret_pct"] - x["rand_mean_pct"] for x in r if x.get("rand_mean_pct") is not None], alpha=0.05)
    out.update(agg=agg, paired_vs_random=paired)
    if agg["lo95"] > 0 and paired.get("lo95") is not None and paired["lo95"] > 0:
        out["status"] = "KEEP"
    elif agg["hi95"] < 0:
        out["status"] = "REJECT"
    else:
        out["status"] = "INCONCLUSIVE"
    return out


# ------------------------------------------------------------------ IO
def _first_bars(out_dir: Path, syms) -> dict:
    """First 15m bar of each pair in history_store (read only), cached for a day."""
    import history_store as HS
    f = out_dir / "first_bar.json"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if cache.get("_day") != today:
        cache = {"_day": today}
    for s in syms:
        if s in cache:
            continue
        a = HS.load(HS.path_of(s)) if HS.path_of(s).exists() else None
        cache[s] = int(a["t"][0]) if a is not None and len(a["t"]) else None
    f.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def load_live(out_dir: Path):
    import trading_universe as TU
    import xsmom as XS
    bars, daily = {}, {}
    for p in sorted((TC.RUNTIME / "scan" / "bars").glob("*_15m.npz")):
        s = p.name[:-len("_15m.npz")]
        if not s.endswith("USDT") or s[:-4] in TU.STABLE_BASES:
            continue
        with np.load(p) as z:
            a = {k: z[k] for k in ("t", "o", "h", "l", "c", "v")}
        if not len(a["t"]):
            continue
        bars[s] = (np.asarray(a["t"], dtype=np.int64), np.asarray(a["c"], dtype=float))
        daily[s] = XS.daily_from_15m(a)
    P = XS.Panel.from_daily(daily)
    first = _first_bars(out_dir, sorted(bars))
    cfg = XS.Config(L=1, H=1, K=10, min_age_days=0)
    uni = {}
    for r in range(P.n + 1):
        day = XS.day_str(P.d0 + r)
        dn = XS.day_num(day)
        u = [s for s in XS.universe(P, day, cfg)
             if first.get(s) is not None and first[s] <= (dn - MIN_AGE_DAYS) * DAY]
        if u:
            uni[dn] = set(u)
    return bars, uni


def run_step(enabled=None, out_dir: Path | None = None, now_ms: int | None = None) -> dict:
    enabled = getattr(TC, "CAP_SHADOW_ENABLED", False) if enabled is None else enabled
    if not enabled:
        return {"enabled": False}
    import fill_model as FM
    out_dir = Path(out_dir or DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    bars, uni = load_live(out_dir)
    sp, lp = out_dir / "state.json", out_dir / "ledger.jsonl"
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
    else:                                                  # start now: no event before the first step (§3)
        g = _grid(bars, now_ms)
        state = {"last_bar_ms": int(g[-2]) if len(g) > 1 else None, "next_allowed_ms": None, "open": [],
                 "started_ms": now_ms}
    state, rows = step(bars, uni, state, now_ms, FM.model("taker", 1))
    if rows:
        with lp.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "logged_ms": now_ms}) + "\n")
    sp.write_text(json.dumps(state), encoding="utf-8")
    results = [json.loads(x) for x in lp.read_text(encoding="utf-8").splitlines()] if lp.exists() else []
    ro = readout(results, datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y-%m-%d"))
    ro.update(open_events=len(state["open"]), universe_today=len(uni.get(now_ms // DAY, ())),
              last_bar=datetime.fromtimestamp((state["last_bar_ms"] or 0) / 1000, timezone.utc).isoformat())
    (out_dir / "READOUT_CAP.json").write_text(json.dumps(ro, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "status": ro["status"], "n_results": ro["n_results"],
                      "open": ro["open_events"], "universe_today": ro["universe_today"], "last_bar": ro["last_bar"]}))
    return ro


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["step"]:
        run_step()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
