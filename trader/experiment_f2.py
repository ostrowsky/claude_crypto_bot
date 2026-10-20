"""F2 runner -- learned exit vs X4 in money (trader-learning-loop-f-1020-spec.md §4, "F2 runner -- контракт").

  train      paths of generator-B entries on W93 before TRAIN_TO (b3 cache + lazy grid) -> ExitPolicy
  validate   the bot's real entries VALID_FROM..VALID_TO in the E1 engine, one run per delta in DELTAS
  test       the bot's real entries from TEST_FROM, ONCE (marker), against X4 and a random early-exit control

  (from trader/) ../pyembed/python.exe -u experiment_f2.py run  -> runtime/experiments/F2_RESULT.json
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import trader_config as TC  # noqa: E402
import learned_exits as LE  # noqa: E402

TRAIN_TO = "2026-01-01"
VALID_FROM, VALID_TO = "2026-03-01", "2026-07-01"
TEST_FROM = "2026-07-01"
DELTAS = (0.0, 0.002, 0.005, 0.01)
N_RANDOM = 20
MAX_PATHS = 30000
ROUNDTRIP = LE.ROUNDTRIP
OUT = TC.RUNTIME / "experiments"
MARKER = OUT / "F2_TEST_OPENED.json"
RESULT = OUT / "F2_RESULT.json"

register = LE.register


def path_from_grid(grid, cand, x4_plan=None, max_bars: int = 96):
    return LE.path_from_grid(grid, cand, x4_plan=x4_plan, max_bars=max_bars)


def _h(*parts) -> float:
    d = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(d[:8], "big") / 2 ** 64


class RandomEarly:
    """Control: with probability p a trade exits at a uniformly random bar 1..x4_exit-1, else at the X4 bar."""

    def __init__(self, p: float, seed: int):
        self.p, self.seed = float(p), int(seed)

    def exit_bar(self, path) -> int:
        x4 = int(path["x4_exit"])
        if x4 < 2 or _h(self.seed, "early", path["key"]) >= self.p:
            return x4
        return 1 + int(_h(self.seed, "bar", path["key"]) * (x4 - 1))


def sample_paths(items, n: int) -> list:
    """Deterministic, order-free: the n items with the smallest hash of their key, returned sorted by key."""
    pick = sorted(items, key=lambda x: hashlib.sha256(str(x["key"]).encode()).hexdigest())[:n]
    return sorted(pick, key=lambda x: x["key"])


def select_delta(valid: dict):
    ok = {d: r["paired"]["mean_pct"] for d, r in valid.items()
          if isinstance(r.get("paired", {}).get("mean_pct"), (int, float)) and r["paired"]["mean_pct"] > 0}
    return max(ok, key=lambda d: ok[d]) if ok else None


def verdict(improves: bool, beats_random: bool, profitable: bool) -> str:
    if improves and beats_random:
        return "accepted" if profitable else "shadow_candidate"
    return "rejected"


def open_test(marker: Path = MARKER, info: dict | None = None) -> None:
    """The test half is opened once: a second call raises."""
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("F2: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**(info or {}), "opened_utc": datetime.now(timezone.utc).isoformat()},
                                 ensure_ascii=False), encoding="utf-8")


def pipeline(train, validate, select, test, marker: Path = MARKER) -> dict:
    pol = train()
    v = validate(pol)
    d = select(v)
    if d is None:
        return {"validation": v, "delta": None, "test": None,
                "verdict": "STOP: no delta beats X4 on validation; the test half was not opened"}
    open_test(marker, {"spec": "trader-learning-loop-f-1020-spec.md F2", "delta": d})
    return {"validation": v, "delta": d, "test": test(pol, d), "verdict": "tested"}


# ------------------------------------------------------------------ the real run
def _x4():
    import trading_policy as TP
    return replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def _learned_policy(lid: str):
    import trading_policy as TP
    return replace(_x4(), name="F2_" + lid, exits=TP.Exits(kind="learned", floor_override=0.08, learned_id=lid))


def train_paths(log=print) -> list:
    import experiment_b3 as B
    import lazy_grid as LG
    import execution_replay as R
    wl = set(json.loads((TC.BOT_FILES / "watchlist.json").read_text(encoding="utf-8")))
    cands = [c for c in B.load_cands() if c["sym"] in wl and c["tf"] == "15m" and c["ts"].strftime("%Y-%m-%d") < TRAIN_TO]
    log(f"train candidates W93 15m < {TRAIN_TO}: {len(cands)}")
    keyed = [{"key": f"{c['sym']}|{c['tf']}|{c['ts'].isoformat()}", "c": c} for c in cands]
    pick = sample_paths(keyed, MAX_PATHS * 2)                       # spare for paths lost to gaps
    grid = LG.LazyGrid.load(LG.CACHE_DIR, t0=B.T0_GRID)
    cc = [x["c"].copy() if hasattr(x["c"], "copy") else dict(x["c"]) for x in pick]
    R.attach_grid(cc, grid)
    paths = []
    for c in cc:
        p = path_from_grid(grid, c)
        if p is not None:
            paths.append(p)
        if len(paths) >= MAX_PATHS:
            break
    grid.close()
    log(f"train paths: {len(paths)}")
    return paths


def main(argv=None) -> int:
    import daily_revenue as DR
    import execution_replay as R
    import exit_q as XQ
    import fill_model as FM
    import money_validator as MV
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "run":
        print(__doc__)
        return 0
    t0 = time.time()
    TC.bot_readonly()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    trades, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(trades, grid)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}
    X4 = _x4()

    def go(pol, lo, hi, f=fm):
        return R.run(pol, f, grid, trades, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache)

    state = {}

    def train():
        paths = train_paths(log)
        state["n_paths"] = len(paths)
        return paths

    def agent_for(paths, delta):
        a = XQ.ExitPolicy(delta=delta).fit(paths)
        return register(a, f"f2_d{delta}")

    def validate(paths):
        base = go(X4, VALID_FROM, VALID_TO)
        out = {}
        for d in DELTAS:
            lid = agent_for(paths, d)
            r = go(_learned_policy(lid), VALID_FROM, VALID_TO)
            pr = DR.paired(r["series"], base["series"])
            out[d] = {"paired": pr, "agg": r["agg"], "trades": len(r["trades"]),
                      "learned_exits": sum(1 for t in r["trades"] if t["reason"] == "learned")}
            log(f"valid delta {d}: R {r['agg'].get('mean_pct')} paired {pr.get('mean_pct')} "
                f"[{pr.get('lo95')}, {pr.get('hi95')}] learned exits {out[d]['learned_exits']}/{len(r['trades'])}")
        state["valid_base"] = base["agg"]
        return out

    def test(paths, d):
        lid = f"f2_d{d}"                                              # registered during validation, same fit
        w, b = go(_learned_policy(lid), TEST_FROM, None), go(X4, TEST_FROM, None)
        pr = DR.paired(w["series"], b["series"])
        bmap = dict(b["series"])
        months = DR.month_means([(dd, v - bmap[dd]) for dd, v in w["series"] if dd in bmap])
        imp = MV.improves(pr, w["agg"].get("max_dd_pct", 0), b["agg"].get("max_dd_pct", 0), months, len(w["trades"]))
        tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
        prof = MV.profitable(w["agg"], tci)
        p_early = sum(1 for t in w["trades"] if t["reason"] == "learned") / max(1, len(w["trades"]))
        rand = []
        for s in range(N_RANDOM):
            rid = register(RandomEarly(p_early, s), f"f2_random_{s}")
            rand.append(go(_learned_policy(rid), TEST_FROM, None)["agg"].get("mean_pct"))
        med = statistics.median([x for x in rand if x is not None]) if rand else None
        beats = med is not None and (w["agg"].get("mean_pct") or -1e9) > med
        v = verdict(imp["verdict"], beats, prof["verdict"])
        MV.log_trial({"family": "F2", "name": lid, "k_opened_on_test": 1, "improves": imp["verdict"],
                      "beats_random_exit": beats, "profitable": prof["verdict"], "verdict": v, "paired_test": pr})
        log(f"TEST {lid}: R {w['agg'].get('mean_pct')} [{w['agg'].get('lo95')}, {w['agg'].get('hi95')}] "
            f"X4 {b['agg'].get('mean_pct')} paired {pr.get('mean_pct')} [{pr.get('lo95')}, {pr.get('hi95')}] "
            f"early share {p_early:.2f} random median {med} -> {v}")
        return {"winner": w["agg"], "base": b["agg"], "paired_vs_x4": pr, "months_paired": months, "improves": imp,
                "trade_ci": tci, "profitable": prof, "early_share": p_early, "random_means": rand, "random_median": med,
                "beats_random_exit": beats, "verdict": v, "trades": len(w["trades"]),
                "decomposition": R.decompose(w["trades"]), "base_decomposition": R.decompose(b["trades"])}

    res = pipeline(train, validate, select_delta, test)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"spec": "trader-learning-loop-f-1020-spec.md F2", "windows": {"train_to": TRAIN_TO, "valid": [VALID_FROM, VALID_TO],
           "test_from": TEST_FROM}, "n_train_paths": state.get("n_paths"), "valid_base": state.get("valid_base"),
           "validation": {str(k): v for k, v in (res["validation"] or {}).items()}, "delta": res["delta"],
           "test": res["test"], "verdict": res["verdict"], "policy_x4": asdict(X4),
           "utc": datetime.now(timezone.utc).isoformat()}
    RESULT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"{res['verdict']}; saved {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
