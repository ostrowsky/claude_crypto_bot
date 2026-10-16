"""Is the frozen ML gate a model problem or a floor problem? -- offline, nothing deployed.

WHY (2026-10-09)
ml_promotion_gate kept the 10-01 incumbent (catboost) seven nights in a row
(10-02..10-08), each time "clearly worse on precision at the live floor". In
every one of those nights the candidate was a LOGISTIC model with the same AUC
(0.655 vs 0.654 on 10-08) whose scores sit higher, so the fixed floor 0.15
admits ~84% of rows instead of ~66%. Precision at a fixed floor then falls
mechanically. Question: at an EQUAL admitted share, is the candidate worse?

DESIGN, FIXED BEFORE THE RUN
  rows    the trainer's own test split (newest 15% of ml_dataset rows, rebuilt
          exactly as ml_signal_model.build_dataset does); neither model trained on it.
  models  incumbent = ml_signal_model.json, candidate = ml_signal_model.candidate.json
  A       fixed live floor (what the gate compares today)
  B       matched admission: candidate floor = the score at which it admits the same
          share of rows the incumbent admits at the live floor
  B'      transfer check: matched floor chosen on the FIRST half of the test rows
          (by time), admitted share and precision measured on the SECOND half
  metrics precision, recall (peak label), share of winner-day rows admitted
          (immutable later-EOD top-20, exit_validator.day_labels), paired bootstrap
          95% interval of (candidate - incumbent), 500 resamples, seed 29
Run from files/:  ../pyembed/python.exe _backtest_ml_promotion_matched.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                          # noqa: E402
import ml_signal_model as M            # noqa: E402
import ml_promotion_gate as G          # noqa: E402

OUT = HERE.parent / ".runtime" / "ml_promotion_matched.json"


def floor_for_share(scores, share):
    s = sorted(scores, reverse=True)
    k = max(1, min(len(s), int(round(share * len(s)))))
    return s[k - 1]


def metrics(y, s, fl, win):
    adm = [v >= f for v, f in zip(s, fl)]
    n_adm = sum(adm)
    pos = sum(1 for t in y if t > 0.5)
    tp = sum(1 for a, t in zip(adm, y) if a and t > 0.5)
    nw = sum(win)
    return {"admitted": n_adm / len(y), "precision": tp / n_adm if n_adm else None,
            "recall": tp / pos if pos else None,
            "winner_rows_admitted": (sum(1 for a, w in zip(adm, win) if a and w) / nw) if nw else None}


def boot(y, so, sn, fo, fn, win, rnd, B=500):
    keys = ("precision", "recall", "winner_rows_admitted")
    d = {k: [] for k in keys}
    n = len(y)
    for _ in range(B):
        idx = [rnd.randrange(n) for _ in range(n)]
        sub = lambda a: [a[i] for i in idx]  # noqa: E731
        mo = metrics(sub(y), sub(so), sub(fo), sub(win))
        mn = metrics(sub(y), sub(sn), sub(fn), sub(win))
        for k in keys:
            if mo[k] is not None and mn[k] is not None:
                d[k].append(mn[k] - mo[k])
    out = {}
    for k, v in d.items():
        v.sort()
        out[k] = [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)] if v else None
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inc = json.loads((HERE / "ml_signal_model.json").read_text(encoding="utf-8"))
    cand = json.loads((HERE / "ml_signal_model.candidate.json").read_text(encoding="utf-8"))
    print("incumbent", inc.get("model_name"), "candidate", cand.get("model_name"),
          "labels", inc.get("label_version"), cand.get("label_version"))
    rows = M.load_training_rows(M.DATASET_FILE)
    b = M.build_dataset(rows)
    test, y = b.meta_test, [float(v) for v in b.y_test]
    so = [M.predict_proba_from_payload(inc, r) for r in test]
    sn = [M.predict_proba_from_payload(cand, r) for r in test]
    live = G.floors_for(test, config)
    try:
        import exit_validator as EV
        _, winners = EV.day_labels()
    except Exception as e:  # labels are a secondary metric
        print("winner labels unavailable:", e)
        winners = set()
    win = [(r["_dt"].strftime("%Y-%m-%d"), r.get("sym")) in winners for r in test]
    print(f"test rows {len(y)}  {test[0]['_dt']:%Y-%m-%d}..{test[-1]['_dt']:%Y-%m-%d}  base {sum(y)/len(y):.3f}  "
          f"winner-day rows {sum(win)}  AUC old {G.auc(y, so):.4f} new {G.auc(y, sn):.4f}")
    rep = {"rows": len(y), "from": str(test[0]["_dt"]), "to": str(test[-1]["_dt"]), "base_rate": sum(y) / len(y),
           "auc_old": G.auc(y, so), "auc_new": G.auc(y, sn), "winner_rows": sum(win)}

    # A: fixed live floor
    mA_o, mA_n = metrics(y, so, live, win), metrics(y, sn, live, win)
    ciA = boot(y, so, sn, live, live, win, random.Random(29))
    # B: matched admission
    share = mA_o["admitted"]
    fm = floor_for_share(sn, share)
    fB = [fm] * len(y)
    mB_n = metrics(y, sn, fB, win)
    ciB = boot(y, so, sn, live, fB, win, random.Random(29))
    rep.update(A={"old": mA_o, "new": mA_n, "ci95_new_minus_old": ciA},
               B={"matched_floor": fm, "share": share, "old": mA_o, "new": mB_n, "ci95_new_minus_old": ciB})
    # B': transfer -- choose on first half, judge on second half
    h = len(y) // 2
    fm1 = floor_for_share(sn[:h], metrics(y[:h], so[:h], live[:h], win[:h])["admitted"])
    y2, so2, sn2, l2, w2 = y[h:], so[h:], sn[h:], live[h:], win[h:]
    mT_o, mT_n = metrics(y2, so2, l2, w2), metrics(y2, sn2, [fm1] * len(y2), w2)
    ciT = boot(y2, so2, sn2, l2, [fm1] * len(y2), w2, random.Random(29))
    rep["B_transfer"] = {"floor_from_first_half": fm1, "old": mT_o, "new": mT_n, "ci95_new_minus_old": ciT}

    fmt = lambda m: "adm %.3f prec %.3f rec %.3f win %s" % (m["admitted"], m["precision"], m["recall"],  # noqa: E731
                                                             "-" if m["winner_rows_admitted"] is None else "%.3f" % m["winner_rows_admitted"])
    print(f"\nA fixed floor {live[0]:.3f}:\n  old {fmt(mA_o)}\n  new {fmt(mA_n)}\n  ci {ciA}")
    print(f"\nB matched admission (new floor {fm:.4f}):\n  old {fmt(mA_o)}\n  new {fmt(mB_n)}\n  ci {ciB}")
    print(f"\nB' transfer (floor {fm1:.4f} from first half, judged on second half n={len(y2)}):\n"
          f"  old {fmt(mT_o)}\n  new {fmt(mT_n)}\n  ci {ciT}")
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
