"""Matched-admission promotion over the maximum period -- walk-forward, nothing deployed.

WHY (2026-10-09, ml-admission-match-1009-spec.md)
The single-split check (_backtest_ml_promotion_matched.py) used one test split.
This replays the NIGHTLY situation over the whole ml_dataset: at each cut a new
model is trained by the production trainer on the rows before the cut, the gate
decides against the model currently in use, and the model in use is applied to the
NEXT block of rows (which no model saw). Two policies run side by side:

  fixed    today's gate: compare at the fixed floor, live gate uses the fixed floor
  matched  compare at matched admission; the promoted model brings its own floor,
           the score at which it admits the incumbent's share of the holdout

Fixed before the run:
  cuts     0.35 0.45 0.55 0.65 0.75 0.85 of rows (by time); block = up to the next cut / the end
  first    the first model is promoted ungated in both policies at the fixed floor
  report   per block and pooled: admitted share, precision, recall (peak label),
           winner-days reached before the +2.5% crossing (same definition as
           _backtest_ml_floor_calibration.py); promotions per policy; and the TRANSFER
           question per cut: the new model at its matched floor vs the model in use,
           admitted-share difference on the forward block
  reading  matched passes if pooled |admitted share - fixed policy's share at the
           first block| does not drift more than the fixed policy's, precision is not
           lower by more than 0.03, and winner-days reached are not fewer.
           This is a description for the operator, not an auto-switch.
Run from files/:  ../pyembed/python.exe _backtest_ml_admission_walkforward.py
"""
import io
import json
import random
import sys
import tempfile
from pathlib import Path

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import ml_signal_model as MS  # noqa: E402
import ml_promotion_gate as MPG  # noqa: E402
from _backtest_ml_floor_calibration import labels_of, stats, winner_reach  # noqa: E402

CUTS = tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
OUT = FILES.parent / ".runtime" / ("ml_admission_walkforward%s.json" % ("_" + sys.argv[1].replace(",", "_") if len(sys.argv) > 1 else ""))


def train(prefix):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prefix.jsonl"
        with io.open(p, "w", encoding="utf-8") as fh:
            for r in prefix:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "_dt"}, ensure_ascii=False, default=str) + "\n")
        rep = MS.train_and_evaluate(p)
    return MS.build_live_model_payload(rep), rep["_holdout"]


def floors_of(rows, fl):
    """fl = None -> config floors; else {"default", "bull_day"}."""
    if fl is None:
        return MPG.floors_for(rows, cfg)
    return [fl["bull_day"] if r.get("is_bull_day") else fl["default"] for r in rows]


def main():
    import immutable_labels as IL
    import label_store as LS
    import _compute_early_capture as E
    rows = MS.load_training_rows(MS.DATASET_FILE)
    n = len(rows)
    y_all = labels_of(rows)
    print("rows %d, %s .. %s, base %.3f" % (n, rows[0]["_dt"].date(), rows[-1]["_dt"].date(), sum(y_all) / n))
    win, _ = IL.winners_by_day(top_n=20, watchlist=E.load_watchlist(), rank_before_filter=True)
    winners = set(win)
    deadlines = {k: v[1] for k, v in LS.intraday_deadlines().items() if v[1] is not None}

    pol = {"fixed": {"model": None, "floor": None, "promotions": 0},
           "matched": {"model": None, "floor": None, "promotions": 0}}
    pooled = {k: {"y": [], "adm": [], "reach": [0, 0], "blocks": []} for k in pol}
    transfer = []
    for ci, c in enumerate(CUTS):
        i0 = int(n * c)
        i1 = int(n * CUTS[ci + 1]) if ci + 1 < len(CUTS) else n
        prefix, fwd, y_fwd = rows[:i0], rows[i0:i1], y_all[i0:i1]
        cand, ho = train(prefix)
        hr, hy = ho["rows"], [float(v) for v in ho["y"]]
        s_c = [MS.predict_proba_from_payload(cand, r) for r in hr]
        line = "cut %.2f: train to %s, forward %d rows %s..%s, candidate %s" % (
            c, prefix[-1]["_dt"].date(), len(fwd), fwd[0]["_dt"].date(), fwd[-1]["_dt"].date(), cand.get("model_name"))
        print("\n" + line)
        for name, P in pol.items():
            if P["model"] is None:
                P.update(model=cand, floor=None, promotions=1)
                print("  %-7s first model promoted ungated" % name)
                continue
            s_i = [MS.predict_proba_from_payload(P["model"], r) for r in hr]
            f_i = floors_of(hr, P["floor"])
            if name == "fixed":
                cm = MPG.compare(hy, s_i, s_c, f_i, random.Random(29))
                new_floor = None
            else:
                mf = MPG.matched_floors(hr, s_i, s_c, f_i)
                new_floor = {"default": mf["default"], "bull_day": mf["bull_day"]}
                cm = MPG.compare(hy, s_i, s_c, f_i, random.Random(29), floors_new=floors_of(hr, new_floor))
                # transfer: the new model at its matched floor vs the model in use, on the forward block
                s_if = [MS.predict_proba_from_payload(P["model"], r) for r in fwd]
                s_cf = [MS.predict_proba_from_payload(cand, r) for r in fwd]
                a_i = sum(v >= f for v, f in zip(s_if, floors_of(fwd, P["floor"]))) / len(fwd)
                a_c = sum(v >= f for v, f in zip(s_cf, floors_of(fwd, new_floor))) / len(fwd)
                transfer.append({"cut": c, "in_use": P["model"].get("model_name"), "candidate": cand.get("model_name"),
                                 "matched_floor": new_floor, "share_holdout": mf["incumbent_share"],
                                 "share_fwd_in_use": round(a_i, 4), "share_fwd_candidate": round(a_c, 4)})
                print("  transfer: in use admits %.3f of the forward block, candidate at matched floor %.3f (floor %.4f)" % (
                    a_i, a_c, new_floor["default"]))
            d, why = MPG.decide(cm)
            print("  %-7s %s -- %s" % (name, d, why))
            if d == "promote":
                P.update(model=cand, floor=new_floor)
                P["promotions"] += 1
        for name, P in pol.items():
            s = [MS.predict_proba_from_payload(P["model"], r) for r in fwd]
            adm = [v >= f for v, f in zip(s, floors_of(fwd, P["floor"]))]
            st = stats(y_fwd, adm)
            g, h = winner_reach(fwd, adm, winners, deadlines)
            pooled[name]["y"] += y_fwd
            pooled[name]["adm"] += adm
            pooled[name]["reach"][0] += g
            pooled[name]["reach"][1] += h
            blk = {"cut": c, "model": P["model"].get("model_name"), "floor": P["floor"], **st, "reach": [g, h]}
            pooled[name]["blocks"].append(blk)
            print("  %-7s in use %-9s admitted %.3f precision %.3f (base %.3f) recall %.3f winner-days %d/%d" % (
                name, P["model"].get("model_name"), st["admitted"], st["precision"] or 0, st["pos"] / st["n"],
                st["recall"] or 0, g, h))

    print("\n=== POOLED over %d forward blocks ===" % len(CUTS))
    res = {"cuts": CUTS, "transfer": transfer}
    for name, v in pooled.items():
        st = stats(v["y"], v["adm"])
        g, h = v["reach"]
        sh = [b["admitted"] for b in v["blocks"]]
        res[name] = {**st, "reach": [g, h], "promotions": pol[name]["promotions"], "blocks": v["blocks"],
                     "admitted_range": [min(sh), max(sh)]}
        print("  %-7s promotions %d  admitted %.3f (blocks %.3f..%.3f)  precision %.3f  recall %.3f  winner-days %d/%d" % (
            name, pol[name]["promotions"], st["admitted"], min(sh), max(sh), st["precision"], st["recall"], g, h))
    if transfer:
        dif = [t["share_fwd_candidate"] - t["share_fwd_in_use"] for t in transfer]
        print("  transfer share diff (candidate at matched floor - in use), per cut: %s" % [round(x, 3) for x in dif])
        res["transfer_share_diff"] = dif
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
