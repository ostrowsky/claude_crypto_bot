"""E4 scale-out on top of X4 in money (e4-scale-out-1118-spec.md): half the position off at +8/15/25%, the rest on the
X4 trail. The bot's own entries, the X4 book, same fill model, slots and sizing.

Train on days < SPLIT: paired daily difference vs X4, >= MIN_CHANGED changed trades. One winner; the test half is opened
ONCE and only by the operator (--operator-approved), k = 1.

  (from trader/) ../pyembed/python.exe -u experiment_e4s.py train
  (from trader/) ../pyembed/python.exe -u experiment_e4s.py test --operator-approved   (once)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402
import daily_revenue as DR        # noqa: E402
import experiment_x4f as X4F      # noqa: E402
import trading_policy as TP       # noqa: E402

SPEC = "e4-scale-out-1118-spec.md"
SPLIT = X4F.SPLIT
MIN_CHANGED = 100
BASE = X4F.BASE
TAKES = (("S_t08", 0.08), ("S_t15", 0.15), ("S_t25", 0.25))
OUT = TC.RUNTIME / "experiments"
TRAIN_PATH = OUT / "E4S_TRAIN.json"
TEST_PATH = OUT / "E4S_TEST.json"
MARKER = OUT / "E4S_TEST_OPENED.json"


def candidates() -> dict:
    """{name: policy} -- registered before the run (spec §3)."""
    return {n: replace(BASE, name=n, exits=TP.Exits(kind="p1_scale", floor_override=0.08, max_bars=96, take_pct=t, take_frac=0.5))
            for n, t in TAKES}


# ------------------------------------------------------------------ pure parts (tested)
def admit(res: dict) -> bool:
    lo = (res.get("paired") or {}).get("lo95")
    return res.get("changed", 0) >= MIN_CHANGED and lo is not None and lo > 0


def select(train: dict):
    ok = {n: r for n, r in train.items() if admit(r)}
    return max(ok, key=lambda n: ok[n]["paired"]["mean_pct"]) if ok else None


def open_test(marker, info: dict) -> None:
    marker = Path(marker)
    if marker.exists():
        raise RuntimeError("E4S: test half already opened (%s); a second look is not allowed" % marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**info, "opened_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                      encoding="utf-8")


def pipeline(validate, select_fn, test, marker=MARKER) -> dict:
    tr = validate()
    w = select_fn(tr)
    if not w:
        return {"train": tr, "winner": None, "test": None, "verdict": "STOP: no candidate passes on train; the test half was not opened"}
    open_test(marker, {"spec": SPEC, "winner": w})
    return {"train": tr, "winner": w, "test": test(w), "verdict": "tested"}


# ------------------------------------------------------------------ run on the bot's entries
def _changed_detail(base_trades: list, cand_trades: list) -> dict:
    bm = {X4F._key(t): t for t in base_trades}
    diffs, months = [], Counter()
    for t in cand_trades:
        b = bm.get(X4F._key(t))
        if b is not None and (b["exit_time"] != t["exit_time"] or abs(b["gross_ret_pct"] - t["gross_ret_pct"]) > 1e-9):
            diffs.append(t["ret_pct"] - b["ret_pct"])
            months[str(t["entry_ts"])[:7]] += 1
    n = len(diffs)
    return {"n": n, "mean_ret_diff_pp": (sum(diffs) / n) if n else None, "by_month": dict(sorted(months.items()))}


def train_run(go) -> dict:
    base = go(BASE, None, SPLIT)
    out = {}
    for name, pol in candidates().items():
        r = go(pol, None, SPLIT)
        pr = DR.paired(r["series"], base["series"])
        res = {"paired": pr, "changed": X4F.changed("W", base["trades"], r["trades"]), "agg": r["agg"],
               "trades": len(r["trades"]), "scale_exits": sum(1 for t in r["trades"] if t.get("reason") == "scale"),
               "changed_detail": _changed_detail(base["trades"], r["trades"]),
               "months_paired": DR.month_means([(d, v - dict(base["series"])[d]) for d, v in r["series"] if d in dict(base["series"])])}
        out[name] = res
        print(f"  {name}: trades {res['trades']} scale {res['scale_exits']} changed {res['changed']} R {r['agg'].get('mean_pct')} "
              f"paired {pr.get('mean_pct')} [{pr.get('lo95')}, {pr.get('hi95')}]", flush=True)
    return {"base": base, "cands": out}


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ""
    if cmd not in ("train", "test"):
        print(__doc__)
        return 0
    import provenance as PV
    PV.require_clean()
    t0 = time.time()
    TC.bot_readonly()
    R, FM, go, _thin, trades, grid, share = X4F._setup()
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "train":
        tr = train_run(go)
        w = select(tr["cands"])
        base = tr["base"]
        rep = {"spec": SPEC, "split": SPLIT, "base": BASE.name, "base_epoch": BASE.epoch(), "base_train": base["agg"],
               "base_trades": len(base["trades"]), "train": tr["cands"], "winner": w,
               "verdict": "winner %s: propose the single test look to the operator" % w if w else
                          "STOP: no candidate passes on train; the test half was not opened",
               "provenance": PV.block(), "utc": datetime.now(timezone.utc).isoformat()}
        TRAIN_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\nBASE X4 train R {base['agg'].get('mean_pct')} trades {len(base['trades'])}")
        print("winner:", w, "|", rep["verdict"])
        print(f"saved {TRAIN_PATH} ({time.time() - t0:.0f}s)")
        return 0

    if "--operator-approved" not in argv:
        print("E4S: the test half is opened only by the operator: add --operator-approved")
        return 2
    if not TRAIN_PATH.exists():
        raise RuntimeError("E4S: no train result; the test half is not opened before train")
    saved = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    import money_validator as MV

    def test(name):
        pol = candidates()[name]
        b = go(BASE, SPLIT, None)
        w = go(pol, SPLIT, None)
        bmap = dict(b["series"])
        pr = DR.paired(w["series"], b["series"], alpha=0.05)
        months = DR.month_means([(d, v - bmap[d]) for d, v in w["series"] if d in bmap])
        imp = MV.improves(pr, w["agg"].get("max_dd_pct", 0), b["agg"].get("max_dd_pct", 0), months, len(w["trades"]))
        tci = MV.trade_ci([t["ret_pct"] for t in w["trades"]])
        prof = MV.profitable(w["agg"], tci)
        out = {"k": 1, "alpha": 0.05, "policy": asdict(pol), "epoch": pol.epoch(), "agg": w["agg"], "base": b["agg"],
               "paired_vs_x4": pr, "months_paired": months, "improves": imp, "profitable": prof,
               "changed": X4F.changed("W", b["trades"], w["trades"]), "verdict": "accepted" if imp["verdict"] else "rejected"}
        MV.log_trial({"family": "E4S", "name": name, "policy_epoch": pol.epoch(), "k_opened_on_test": 1,
                      "improves": imp["verdict"], "profitable": prof["verdict"], "verdict": out["verdict"], "paired_test": pr})
        return out

    res = pipeline(validate=lambda: saved["train"], select_fn=lambda t: saved["winner"], test=test)
    rep = {"spec": SPEC, "split": SPLIT, **{k: res[k] for k in ("winner", "test", "verdict")}, "provenance": PV.block(),
           "utc": datetime.now(timezone.utc).isoformat(),
           "disclosure": "test window 07-01.. was opened by H3 on 2026-10-11 for X0/X4; X4 is the base here"}
    TEST_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(res["verdict"], f"saved {TEST_PATH} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
