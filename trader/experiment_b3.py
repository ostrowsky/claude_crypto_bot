"""Phase B3 -- does trading the whole spot universe beat the 93-coin watchlist, in money? (wide-universe-b-1014-spec.md §4)

Arms on the SAME candidate generator and the SAME policy (X4 exits, 10 slots, taker + 8 bps per side):
W93 = watchlist coins, WALL = all pairs (incl. delisted on the days they traded), WNEW = only the new pairs.
Time split fixed before any run: train 2023-01-01..2025-12-31, test 2026-01-01..last full day; the test half
is opened ONCE (marker file + money_validator.log_trial).
Verdicts on the test half: improves = paired WALL − W93 lo95 > 0; profitable = WALL lo95 > 0;
beats_random = WALL mean > median of 20 random thinnings of the WALL candidate pool to the same trade count.
"""
from __future__ import annotations

import json
import random
from collections.abc import MutableMapping
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

TRAIN_FROM = "2023-01-01"
TEST_FROM = "2026-01-01"
N_RANDOM = 20
OUT_DIR = TC.RUNTIME / "experiments"
TEST_MARKER = OUT_DIR / "B3_TEST_OPENED.json"


def split_arms(cands, watchlist) -> dict:
    wl = set(watchlist)
    return {"W93": [c for c in cands if c["sym"] in wl], "WALL": list(cands), "WNEW": [c for c in cands if c["sym"] not in wl]}


def thin(cands, n, seed) -> list:
    """n candidates drawn at random (reproducible by seed), returned in time order."""
    rnd = random.Random(seed)
    pick = rnd.sample(range(len(cands)), min(n, len(cands)))
    return [cands[i] for i in sorted(pick, key=lambda i: (cands[i]["ts"], i))]


def verdict(paired, wall, random_means) -> dict:
    med = statistics.median(random_means) if random_means else None
    return {"improves": paired.get("lo95") is not None and paired["lo95"] > 0,
            "profitable": wall.get("lo95") is not None and wall["lo95"] > 0,
            "beats_random": med is not None and wall.get("mean_pct") is not None and wall["mean_pct"] > med,
            "random_median": med}


def open_test(marker: Path, info: dict) -> None:
    """The test half is opened once: the second call raises."""
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("B3: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ runner (spec §4b)
CANDS_DIR = OUT_DIR / "b3_cands"


class Cand(MutableMapping):
    """Compact replay candidate (spec §4b.7): the fields execution_replay.run / attach_grid / exit_plan read or write,
    behind the Mapping interface, so ~2 million of them fit in memory. Unknown keys -> KeyError (.get() -> None)."""
    __slots__ = ("kind", "sym", "tf", "ts", "price", "mode", "trail_k", "exit_ts", "exit_price", "j", "ie", "ix")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __getitem__(self, k):
        if k in self.__slots__:
            return getattr(self, k)
        raise KeyError(k)

    def __setitem__(self, k, v):
        if k not in self.__slots__:
            raise KeyError(k)
        setattr(self, k, v)

    def __delitem__(self, k):
        raise TypeError("Cand fields cannot be deleted")

    def __iter__(self):
        return iter(self.__slots__)

    def __len__(self):
        return len(self.__slots__)

    def copy(self):
        return Cand(**{k: getattr(self, k) for k in self.__slots__})


def to_replay(events) -> list:
    """Generator events -> compact candidates of execution_replay.run (kind entry, ts as datetime), in time order."""
    out = []
    for e in events:
        ts = e["ts"] if isinstance(e["ts"], datetime) else datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
        out.append(Cand(kind="entry", sym=sys.intern(e["sym"]), tf=sys.intern(e["tf"]), ts=ts, price=float(e["price"]),
                        mode=sys.intern(e.get("mode") or ""), trail_k=e.get("trail_k")))
    out.sort(key=lambda c: (c["ts"], c["sym"], c["tf"]))
    return out


def b3_policy(cooldown_bars: int):
    from dataclasses import replace
    import trading_policy as TP
    return replace(TP.CHAMPION, name="b3_x4", exits=TP.Exits(kind="p1_trail", floor_override=0.08),
                   risk=TP.Risk(max_open=10, one_per_symbol=True, cooldown_bars=int(cooldown_bars)))


def cands_for(sym, arr, cache_dir, gen_fn, sha) -> list:
    """Generator events of one pair, cached per pair; recomputed when the code/config sha differs."""
    cache_dir = Path(cache_dir)
    meta_p, data_p = cache_dir / f"{sym}.meta.json", cache_dir / f"{sym}.jsonl"
    try:
        if json.loads(meta_p.read_text(encoding="utf-8")).get("sha") == sha and data_p.exists():
            return [json.loads(x) for x in data_p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        pass
    ev = gen_fn(sym, arr)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_p.write_text("".join(json.dumps(e) + "\n" for e in ev), encoding="utf-8")
    meta_p.write_text(json.dumps({"sym": sym, "sha": sha, "n": len(ev), "utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    return ev


def _run(policy, fm, grid, cands, day_from, day_to) -> dict:
    import execution_replay as R
    cc = [c.copy() if isinstance(c, Cand) else dict(c) for c in cands]
    R.attach_grid(cc, grid)
    r = R.run(policy, fm, grid, cc, floor_of=lambda m: 0.0, day_from=day_from, day_to=day_to, plan_cache={})
    trades = r.get("trades", [])
    return {"agg": r.get("agg", {}), "series": r.get("series", []), "n_trades": len(trades), "decomposition": R.decompose(trades),
            "trade_days": sorted({(t["entry_ts"] if isinstance(t["entry_ts"], datetime) else datetime.fromisoformat(str(t["entry_ts"]))).strftime("%Y-%m-%d")
                                  for t in trades}),
            "v3_conservation_ok": r.get("v3_conservation_ok")}


def _defaults(fm, policy):
    import fill_model as FM
    if policy is None:
        TC.bot_readonly()
        import config as BOT
        policy = b3_policy(int(getattr(BOT, "COOLDOWN_BARS", 19)))
    return fm or FM.model("taker", 1), policy


def arms_run(cands, grid, watchlist, day_from, day_to, fm=None, policy=None) -> dict:
    fm, policy = _defaults(fm, policy)
    return {name: _run(policy, fm, grid, pool, day_from, day_to) for name, pool in split_arms(cands, watchlist).items()}


def _slim(arms):
    return {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in arms.items()}


def run_train(cands, grid, watchlist, out_path, fm=None, policy=None) -> dict:
    fm, policy = _defaults(fm, policy)
    arms = arms_run([c for c in cands if c["ts"].strftime("%Y-%m-%d") < TEST_FROM], grid, watchlist, TRAIN_FROM, TEST_FROM, fm, policy)
    res = {"spec": "wide-universe-b-1014-spec.md §4b", "window": [TRAIN_FROM, TEST_FROM], "policy": policy.name, "arms": _slim(arms),
           "n_candidates": {k: len(v) for k, v in split_arms(cands, watchlist).items()},
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def random_control(pool, n_size, grid, day_from, day_to, n=N_RANDOM, fm=None, policy=None) -> list:
    """Mean R of n random subsets (seeds 0..n-1) of `pool`, each of n_size candidates."""
    fm, policy = _defaults(fm, policy)
    out = []
    for seed in range(n):
        r = _run(policy, fm, grid, thin(pool, n_size, seed), day_from, day_to)
        out.append(r["agg"].get("mean_pct"))
    return out


def run_test(cands, grid, watchlist, train_path, out_path, marker=TEST_MARKER, n_random=N_RANDOM, trial_log=None, fm=None, policy=None) -> dict:
    import daily_revenue as DR
    import money_validator as MV
    if not Path(train_path).exists():
        raise RuntimeError("B3: the train result %s is missing; the test half is not opened before train" % train_path)
    fm, policy = _defaults(fm, policy)
    open_test(marker, {"spec": "wide-universe-b-1014-spec.md §4b", "train_file": str(train_path)})
    test_c = [c for c in cands if c["ts"].strftime("%Y-%m-%d") >= TEST_FROM]
    arms = arms_run(test_c, grid, watchlist, TEST_FROM, None, fm, policy)
    paired = DR.paired(arms["WALL"]["series"], arms["W93"]["series"])
    pools = split_arms(test_c, watchlist)
    rand = random_control(pools["WALL"], len(pools["W93"]), grid, TEST_FROM, None, n_random, fm, policy)
    v = verdict(paired, arms["WALL"]["agg"], [x for x in rand if x is not None])
    w93 = arms["W93"]["agg"].get("mean_pct")
    v["w93_vs_random_median"] = None if (w93 is None or v["random_median"] is None) else w93 - v["random_median"]
    res = {"spec": "wide-universe-b-1014-spec.md §4b", "window": [TEST_FROM, None], "policy": policy.name, "arms": _slim(arms),
           "paired_WALL_minus_W93": paired, "random_means": rand, "verdict": v, "n_candidates": {k: len(x) for k, x in pools.items()},
           "utc": datetime.now(timezone.utc).isoformat()}
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    MV.log_trial({"rule": "b3_wide_universe", "stage": "test", "verdict": v,
                  "wall_mean": arms["WALL"]["agg"].get("mean_pct"), "w93_mean": w93}, trial_log)
    return res


# ------------------------------------------------------------------ CLI
T0_GRID = datetime(2023, 1, 1, tzinfo=timezone.utc)


def code_sha() -> str:
    """Generator code + the bot's rule code and config (+ runtime overrides): a change invalidates the candidate cache."""
    import hashlib
    h = hashlib.sha256()
    for p in (HERE / "candidate_gen.py", TC.BOT_FILES / "strategy.py", TC.BOT_FILES / "indicators.py", TC.BOT_FILES / "config.py",
              TC.BOT_FILES / "runtime_overrides.json"):
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def _gen_full(sym, arr):
    import candidate_gen as CG
    import history_store as HS
    if arr is None or not len(arr["t"]):
        return []
    return CG.generate(sym, "15m", arr) + CG.generate(sym, "1h", HS.agg_1h(arr, complete_only=True))


def _gen_job(args):
    sym, sha = args
    import history_store as HS
    try:
        ev = cands_for(sym, HS.load(HS.path_of(sym)), CANDS_DIR, _gen_full, sha)
        return sym, len(ev), None
    except Exception as e:  # one pair must not stop the run
        return sym, 0, repr(e)[:300]


def gen_all(processes: int = 3) -> dict:
    import time
    from multiprocessing import Pool
    import history_store as HS
    sha = code_sha()
    syms = sorted(p.name[:-len("_15m.npz")] for p in HS.HIST_DIR.glob("*_15m.npz"))
    t0, n, errs = time.time(), 0, []
    with Pool(processes) as pool:
        for k, (sym, cnt, err) in enumerate(pool.imap_unordered(_gen_job, [(s, sha) for s in syms]), 1):
            n += cnt
            if err:
                errs.append((sym, err))
            if k % 10 == 0 or k == len(syms):
                print("gen %d/%d pairs | %d candidates | %.0f min" % (k, len(syms), n, (time.time() - t0) / 60), flush=True)
    return {"pairs": len(syms), "candidates": n, "errors": errs[:20], "sha": sha}


def load_cands() -> list:
    ev = []
    for p in sorted(CANDS_DIR.glob("*.jsonl")):
        ev += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return to_replay(ev)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd == "gen":
        print(json.dumps(gen_all(int(argv[argv.index("--processes") + 1]) if "--processes" in argv else 3), indent=1))
        return 0
    if cmd == "grid":
        import lazy_grid as LG
        print(json.dumps(LG.build_all(), indent=1))
        return 0
    if cmd in ("train", "test"):
        import lazy_grid as LG
        wl = set(json.loads((TC.BOT_FILES / "watchlist.json").read_text(encoding="utf-8")))
        cands = load_cands()
        grid = LG.LazyGrid.load(LG.CACHE_DIR, t0=T0_GRID)
        print("candidates %d, pairs in grid %d" % (len(cands), len(grid.syms)), flush=True)
        if cmd == "train":
            r = run_train(cands, grid, wl, OUT_DIR / "B3_TRAIN.json")
            print(json.dumps({k: {kk: v.get(kk) for kk in ("n_trades",)} | {"agg": v["agg"]} for k, v in r["arms"].items()}, indent=1, default=str))
        else:
            r = run_test(cands, grid, wl, OUT_DIR / "B3_TRAIN.json", OUT_DIR / "B3_TEST.json")
            print(json.dumps({"verdict": r["verdict"], "paired": r["paired_WALL_minus_W93"]}, indent=1, default=str))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
