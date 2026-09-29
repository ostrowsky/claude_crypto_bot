"""Calibrate the ML gate floor to each model -- does it hold the operating point
where the fixed floor drifts, and which operating point serves the goal?

Why (2026-09-29): the nightly model's score scale moves while the floor is fixed
at 0.15 -- on 08-20 a scale shift blocked all 4 486 candidates; on 09-29 a
candidate model would have admitted 82% of rows instead of 62% (precision 0.308
-> 0.271). Proposal: the trainer stores `calibrated_floor` = the score at which
the model admits a target share R of the positive rows (peak label) on its own
held-out rows, and the live gate uses it.

Walk-forward, never in-sample (TH-03): for each cut, a model is trained with the
production trainer (ml_signal_model.train_and_evaluate) on the rows BEFORE the
cut only; its floor is calibrated on its own test split (inside the prefix); both
the fixed floor and the calibrated floors are then applied to the NEXT block of
rows, which the model never saw.

Per fold and pooled:
  recall     share of positive rows (peak >= 2% within 5 bars) admitted
  admitted   share of all rows admitted
  precision  share of admitted rows that are positive (base rate beside it)
  goal reach winner-days (immutable top-20 on the watchlist, crossing known) in
             the forward block with >= 1 row before the first +2.5% crossing:
             share of them with >= 1 such row admitted

Decision, fixed before the run:
  R candidates 0.80, 0.85, 0.90, 0.95
  eligible     pooled goal reach >= the fixed floor's pooled goal reach
  choose       among eligible R, the highest pooled precision (least junk)
  none eligible -> calibration refuted; the fixed floor stays
  stability    reported: max |recall - R| over folds vs the fixed floor's recall range
"""
import io
import json
import sys
import tempfile
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import ml_signal_model as MS  # noqa: E402

CUTS = (0.50, 0.60, 0.70, 0.80)
BLOCK = 0.10
RS = (0.80, 0.85, 0.90, 0.95)
FIXED = float(cfg.ML_GENERAL_HARD_BLOCK_MIN)


def labels_of(rows):
    b = MS.build_dataset(rows)
    import numpy as np
    return [float(v) for v in np.concatenate([b.y_train, b.y_val, b.y_test])]


def admit(scores, floor):
    return [s >= floor for s in scores]


def stats(y, adm):
    pos = sum(1 for t in y if t > 0.5)
    a = sum(adm)
    tp = sum(1 for t, x in zip(y, adm) if x and t > 0.5)
    return {"recall": tp / pos if pos else None, "admitted": a / len(y) if y else None,
            "precision": tp / a if a else None, "n": len(y), "pos": pos, "adm": a, "tp": tp}


def winner_reach(rows, adm, winners, deadlines):
    """(days reached, days with a pre-crossing row) in this block."""
    has, got = set(), set()
    for r, x in zip(rows, adm):
        day = r["_dt"].strftime("%Y-%m-%d")
        k = (day, r.get("sym"))
        if k not in winners or k not in deadlines:
            continue
        if r["_dt"] < deadlines[k]:
            has.add(k)
            if x:
                got.add(k)
    return len(got), len(has)


def main():
    import immutable_labels as IL
    import label_store as LS
    import _compute_early_capture as E
    rows = MS.load_training_rows(MS.DATASET_FILE)
    n = len(rows)
    print("rows %d, %s .. %s, fixed floor %.3f" % (n, rows[0]["_dt"].date(), rows[-1]["_dt"].date(), FIXED))
    y_all = labels_of(rows)
    print("peak-label base rate %.3f" % (sum(y_all) / n))
    win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    winners = set(win)
    deadlines = {k: v[1] for k, v in LS.intraday_deadlines().items() if v[1] is not None}

    pooled = {name: {"y": [], "adm": [], "reach": [0, 0]} for name in ["fixed"] + ["R%.2f" % r for r in RS]}
    for c in CUTS:
        i0, i1 = int(n * c), int(n * (c + BLOCK))
        prefix, fwd, y_fwd = rows[:i0], rows[i0:i1], y_all[i0:i1]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prefix.jsonl"
            with io.open(p, "w", encoding="utf-8") as fh:
                for r in prefix:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != "_dt"}, ensure_ascii=False, default=str) + "\n")
            rep = MS.train_and_evaluate(p)
        payload = MS.build_live_model_payload(rep)
        ho = rep["_holdout"]
        s_ho = [MS.predict_proba_from_payload(payload, r) for r in ho["rows"]]
        s_fwd = [MS.predict_proba_from_payload(payload, r) for r in fwd]
        print("\ncut %.2f: train rows %d (to %s), forward %d rows %s .. %s, model %s" % (
            c, len(prefix), prefix[-1]["_dt"].date(), len(fwd), fwd[0]["_dt"].date(), fwd[-1]["_dt"].date(),
            payload.get("model_name")))
        variants = [("fixed", FIXED)] + [("R%.2f" % r, MS.calibrate_floor(ho["y"], s_ho, r)) for r in RS]
        for name, fl in variants:
            adm = admit(s_fwd, fl)
            st = stats(y_fwd, adm)
            g, h = winner_reach(fwd, adm, winners, deadlines)
            pooled[name]["y"] += y_fwd
            pooled[name]["adm"] += adm
            pooled[name]["reach"][0] += g
            pooled[name]["reach"][1] += h
            print("  %-6s floor %.4f  recall %.3f  admitted %.3f  precision %.3f (base %.3f)  winner-days reached %d/%d" % (
                name, fl, st["recall"] or 0, st["admitted"] or 0, st["precision"] or 0, st["pos"] / st["n"], g, h))
        pooled.setdefault("_folds", []).append({name: stats(y_fwd, admit(s_fwd, fl))["recall"] for name, fl in variants})

    print("\n=== POOLED over %d forward blocks ===" % len(CUTS))
    res = {}
    for name, v in pooled.items():
        if name.startswith("_"):
            continue
        st = stats(v["y"], v["adm"])
        g, h = v["reach"]
        res[name] = dict(st, reach=g / h if h else None, reach_n=(g, h))
        print("  %-6s recall %.3f  admitted %.3f  precision %.3f  winner-days reached %d/%d = %.1f%%" % (
            name, st["recall"], st["admitted"], st["precision"], g, h, 100 * g / h if h else 0))
    folds = pooled["_folds"]
    fr = [f["fixed"] for f in folds]
    print("\nstability: fixed floor recall range %.3f .. %.3f" % (min(fr), max(fr)))
    for r in RS:
        name = "R%.2f" % r
        dev = max(abs(f[name] - r) for f in folds)
        print("  %s: max |recall - target| over folds %.3f" % (name, dev))
    base = res["fixed"]["reach"]
    eligible = [n_ for n_ in res if n_ != "fixed" and res[n_]["reach"] is not None and res[n_]["reach"] >= base]
    if not eligible:
        print("\nVERDICT: no R reaches as many winner-days as the fixed floor -> calibration refuted")
    else:
        best = max(eligible, key=lambda n_: res[n_]["precision"])
        print("\nVERDICT: eligible %s; chosen %s (precision %.3f vs fixed %.3f, reach %.1f%% vs %.1f%%)" % (
            eligible, best, res[best]["precision"], res["fixed"]["precision"], 100 * res[best]["reach"], 100 * base))


if __name__ == "__main__":
    main()
