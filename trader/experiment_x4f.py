"""X4 family in money (x4-family-money-1015-spec.md): changes on top of the X4 book, the bot's own entries.

Rows: W stop width / horizon | O exchange OCO (H4) | R market regime (H5) | L daily loss limit (H6).
Train on days < SPLIT: paired daily difference vs X4; R and L must also beat random thinning of X4 to the
same trade count. One winner per row; the test half is opened ONCE for the winners, k = their number,
alpha = 0.05 / k.

  (from trader/) ../pyembed/python.exe experiment_x4f.py train   -> runtime/experiments/X4F_TRAIN.json
  (from trader/) ../pyembed/python.exe experiment_x4f.py test    -> runtime/experiments/X4F_TEST.json (once)
"""
from __future__ import annotations

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

import trader_config as TC        # noqa: E402
import daily_revenue as DR        # noqa: E402
import money_validator as MV      # noqa: E402
import trading_policy as TP       # noqa: E402

SPLIT = "2026-07-01"
MIN_CHANGED = 100
N_THIN = 20
ROWS = ("W", "O", "R", "L")
FILTER_ROWS = ("R", "L")
OUT = TC.RUNTIME / "experiments"
TRAIN_PATH = OUT / "X4F_TRAIN.json"
TEST_PATH = OUT / "X4F_TEST.json"
MARKER = OUT / "X4F_TEST_OPENED.json"

BASE = replace(TP.CHAMPION, name="X4_p1_floor8", exits=TP.Exits(kind="p1_trail", floor_override=0.08))


def candidates() -> dict:
    """{name: (row, policy)} -- registered before the run (spec §3)."""
    c = {}
    for f in (0.04, 0.06, 0.10, 0.12):
        n = "W_f%02d" % round(f * 100)
        c[n] = ("W", replace(BASE, name=n, exits=TP.Exits(kind="p1_trail", floor_override=f, max_bars=96)))
    for h in (48, 192):
        n = "W_h%d" % h
        c[n] = ("W", replace(BASE, name=n, exits=TP.Exits(kind="p1_trail", floor_override=0.08, max_bars=h)))
    for tp in (1.5, 2, 3):
        for sl in (1, 1.5):
            n = "O_t%s_s%s" % (tp, sl)
            c[n] = ("O", replace(BASE, name=n, exits=TP.Exits(kind="oco_atr", tp_atr=tp, sl_atr=sl, max_bars=96)))
    for r in ("R1", "R2", "R3", "R4"):
        c[r] = ("R", replace(BASE, name=r, admission=replace(BASE.admission, regime=r)))
    for lim in (1, 2, 3):
        n = "L%d" % lim
        c[n] = ("L", replace(BASE, name=n, risk=replace(BASE.risk, daily_loss_limit_pct=lim)))
    return c


# ------------------------------------------------------------------ pure parts (tested)
def _key(t):
    return t["sym"], t["entry_ts"]


def changed(row: str, base: list, cand: list) -> int:
    """W/O: candidate trades with another exit bar or price than X4 (or absent from X4);
    R/L: X4 trades whose entry the candidate skipped."""
    bm = {_key(t): t for t in base}
    if row in FILTER_ROWS:
        ck = {_key(t) for t in cand}
        return sum(1 for k in bm if k not in ck)
    n = 0
    for t in cand:
        b = bm.get(_key(t))
        if b is None or b["exit_time"] != t["exit_time"] or abs(b["gross_ret_pct"] - t["gross_ret_pct"]) > 1e-9:
            n += 1
    return n


def admit(row: str, res: dict) -> bool:
    lo = res["paired"].get("lo95")
    if res["changed"] < MIN_CHANGED or lo is None or not lo > 0:
        return False
    if row in FILTER_ROWS:
        th, m = res.get("thin_median"), res["agg"].get("mean_pct")
        return th is not None and m is not None and m > th
    return True


def select_by_row(train_results: dict) -> dict:
    """{row: winner name or None}: largest train paired mean among the admitted of each row."""
    out = {}
    for row in ROWS:
        ok = {n: r for n, (rw, r) in train_results.items() if rw == row and admit(row, r)}
        out[row] = max(ok, key=lambda n: ok[n]["paired"]["mean_pct"]) if ok else None
    return out


def alpha_for(k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return 0.05 / k


def paired_k(a, b, k: int) -> dict:
    return DR.paired(a, b, alpha=alpha_for(k))


def thinning_median(run, target: int, n: int = N_THIN) -> dict:
    """run(frac, seed) -> (n_trades, mean R). frac matched on seed 0 to `target` trades (+-5%)."""
    import experiment_h3 as H
    frac = H.match_frac(lambda f: run(f, 0)[0], target)
    res = [run(frac, s) for s in range(n)]
    means = [m for _, m in res if m is not None]
    return {"median": statistics.median(means) if means else None, "frac": frac, "n_trades": res[0][0], "means": means}


def validate(go, cands: dict, thin) -> dict:
    """Train half only. go(policy, day_from, day_to) -> replay result; thin(target_trades) -> thinning_median."""
    base = go(BASE, None, SPLIT)
    out = {}
    for name, (row, pol) in cands.items():
        r = go(pol, None, SPLIT)
        pr = DR.paired(r["series"], base["series"])
        res = {"paired": pr, "changed": changed(row, base["trades"], r["trades"]), "agg": r["agg"],
               "trades": len(r["trades"]), "thin_median": None}
        if row in FILTER_ROWS:
            res["thin"] = thin(len(r["trades"]))
            res["thin_median"] = res["thin"]["median"]
        out[name] = (row, res)
        print(f"  {name:12s} {row} trades {res['trades']:5d} changed {res['changed']:5d} R {r['agg'].get('mean_pct')}"
              f" paired {pr.get('mean_pct')} [{pr.get('lo95')}, {pr.get('hi95')}] thin {res['thin_median']}", flush=True)
    return out


def open_test(marker: Path, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("X4F: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(validate, select, test, marker: Path = MARKER) -> dict:
    tr = validate()
    w = select(tr)
    winners = {name: row for row, name in w.items() if name}
    if not winners:
        return {"train": tr, "winners": w, "test": None,
                "verdict": "STOP: no row passes on train; the test half was not opened"}
    open_test(marker, {"spec": "x4-family-money-1015-spec.md", "winners": winners})
    return {"train": tr, "winners": w, "test": test(winners, len(winners)), "verdict": "tested"}


def require_train(path: Path) -> None:
    if not Path(path).exists():
        raise RuntimeError("X4F: the train result %s is missing; the test half is not opened before train" % path)


def row_verdict(row: str, improves: bool, beats_thin) -> str:
    if not improves:
        return "rejected"
    if row in FILTER_ROWS and not beats_thin:
        return "less_trading_only"
    return "accepted"


# ------------------------------------------------------------------ run on the bot's entries
def _setup():
    import execution_replay as R
    import fill_model as FM
    import regime as RG
    trades, _ = R.load_events()
    grid = R._grid()
    R.attach_grid(trades, grid)
    flags = RG.flags(grid)
    fm, fl, cache = FM.model("taker", 1), R._floor(), {}

    def go(pol, lo, hi, f=fm):
        return R.run(pol, f, grid, trades, floor_of=fl, day_from=lo, day_to=hi, plan_cache=cache, regime_flags=flags)

    def thin_for(lo, hi):
        def run(frac, seed):
            r = go(replace(BASE, name="thin", admission=replace(BASE.admission, random_frac=frac, random_seed=seed)), lo, hi)
            return len(r["trades"]), r["agg"].get("mean_pct")
        return lambda target: thinning_median(run, target)

    share = {k: round(float(v.mean()), 4) for k, v in flags.items()}
    print(f"bot entries {len(trades)}, grid bars {grid.N}, pairs {len(grid.syms)}, BTC in grid {'BTCUSDT' in grid.col}, "
          f"flag share {share}", flush=True)
    return R, FM, go, thin_for, trades, grid, share


def _jsonable_train(tr: dict) -> dict:
    return {n: {"row": row, **res} for n, (row, res) in tr.items()}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd not in ("train", "test"):
        print(__doc__)
        return 0
    t0 = time.time()
    TC.bot_readonly()
    R, FM, go, thin_for, trades, grid, share = _setup()
    cands = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "train":
        tr = validate(go, cands, thin_for(None, SPLIT))
        base = go(BASE, None, SPLIT)
        w = select_by_row(tr)
        rep = {"spec": "x4-family-money-1015-spec.md", "split": SPLIT, "base": BASE.name, "base_train": base["agg"],
               "base_trades": len(base["trades"]), "base_decomposition": R.decompose(base["trades"]),
               "regime_flag_share": share, "train": _jsonable_train(tr), "winners": w,
               "utc": datetime.now(timezone.utc).isoformat()}
        TRAIN_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\nBASE X4 train R {base['agg'].get('mean_pct')} trades {len(base['trades'])}")
        print("winners by row:", json.dumps(w))
        print(f"saved {TRAIN_PATH} ({time.time() - t0:.0f}s)")
        return 0

    require_train(TRAIN_PATH)
    saved = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))

    def test(winners, k):
        b = go(BASE, SPLIT, None)
        bmap = dict(b["series"])
        thin = thin_for(SPLIT, None)
        out = {"k": k, "alpha": alpha_for(k), "base": b["agg"], "base_trades": len(b["trades"]), "rows": {}}
        for name, row in winners.items():
            pol = cands[name][1]
            w = go(pol, SPLIT, None)
            pr = paired_k(w["series"], b["series"], k)
            months = DR.month_means([(d, v - bmap[d]) for d, v in w["series"] if d in bmap])
            imp = MV.improves(pr, w["agg"].get("max_dd_pct", 0), b["agg"].get("max_dd_pct", 0), months, len(w["trades"]))
            tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
            prof = MV.profitable(w["agg"], tci)
            th = thin(len(w["trades"])) if row in FILTER_ROWS else None
            beats = None if th is None else (th["median"] is not None and (w["agg"].get("mean_pct") or -1e9) > th["median"])
            ref = {f"slip{lvl}": go(pol, SPLIT, None, FM.model("taker", lvl))["agg"] for lvl in (0, 2)}
            verdict = row_verdict(row, imp["verdict"], beats)
            out["rows"][name] = {"row": row, "policy": asdict(pol), "epoch": pol.epoch(), "agg": w["agg"],
                                 "trades": len(w["trades"]), "changed": changed(row, b["trades"], w["trades"]),
                                 "paired_vs_x4": pr, "months_paired": months, "improves": imp, "trade_ci": tci,
                                 "profitable": prof, "thinning": th, "beats_thinning": beats, "verdict": verdict,
                                 "decomposition": R.decompose(w["trades"]), "slip_reference": ref}
            MV.log_trial({"family": "X4F", "name": name, "row": row, "policy_epoch": pol.epoch(), "k_opened_on_test": k,
                          "improves": imp["verdict"], "profitable": prof["verdict"], "beats_thinning": beats,
                          "verdict": verdict, "paired_test": pr})
            print(f"TEST {name} ({row}): R {w['agg'].get('mean_pct')} [{w['agg'].get('lo95')}, {w['agg'].get('hi95')}] "
                  f"paired vs X4 {pr.get('mean_pct')} [{pr.get('lo95')}, {pr.get('hi95')}] (alpha {alpha_for(k)}) "
                  f"improves {imp['verdict']} profitable {prof['verdict']} thin {beats} -> {verdict}", flush=True)
        return out

    res = pipeline(validate=lambda: saved["train"], select=lambda t: saved["winners"], test=test)
    rep = {"spec": "x4-family-money-1015-spec.md", "split": SPLIT, "verdict": res["verdict"], "winners": res["winners"],
           "test": res["test"], "utc": datetime.now(timezone.utc).isoformat(),
           "disclosure": "test window 07-01.. was opened by H3 on 2026-10-11 for X0/X4; X4 is the base here"}
    TEST_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(res["verdict"], f"saved {TEST_PATH} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
