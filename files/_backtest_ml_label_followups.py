"""Two follow-ups to goal-aligned-ml-label-1012 (criteria: ml-label-followups-1012-spec.md).

Test 1 (EXPLORATORY, same data as the parent): the GOAL-label model with trades
counted the way the bot takes them -- one position per coin plus the cooldown --
instead of every admitted candidate. It decides only whether a forward shadow is
worth building; it cannot reverse the parent's verdict.

Test 2 (pre-registered): the live model trains on ml_dataset.jsonl signal rows
(~85% from March-April); the gate scores the bot's candidates. Same label, same
estimator, two training populations, graded on the goal.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _backtest_goal_label as G  # noqa: E402

CI = 0.95
BOOT = 2000
MIN_MLDS_TRAIN = 500
ARMS2 = ("SRC_MLDS", "SRC_CRITIC")


def dedup_trades(rows, cooldown):
    """Indices of rows that open a trade: one position per coin, then `cooldown` bars of the row's tf."""
    busy = {}
    taken = []
    for i in sorted(range(len(rows)), key=lambda k: rows[k]["dec"]):
        r = rows[i]
        if r.get("exit") is None or not np.isfinite(r["pnl"]):
            continue
        b = busy.get(r["sym"])
        if b is not None and r["dec"] < b:
            continue
        taken.append(i)
        busy[r["sym"]] = r["exit"] + cooldown * G.TF_LEN[r["tf"]]
    return sorted(taken)


def split_changed(arm_mask, ctrl_mask):
    return np.where(arm_mask & ~ctrl_mask)[0], np.where(ctrl_mask & ~arm_mask)[0]


def cluster_delta(days_a, vals_a, days_b, vals_b, rnd, boot=BOOT):
    """mean(a) - mean(b) with a day-cluster bootstrap (both arms resample the same days)."""
    ud = sorted(set(days_a) | set(days_b))
    pos = {d: k for k, d in enumerate(ud)}
    sa, na, sb, nb = (np.zeros(len(ud)) for _ in range(4))
    for d, v in zip(days_a, vals_a):
        sa[pos[d]] += v
        na[pos[d]] += 1
    for d, v in zip(days_b, vals_b):
        sb[pos[d]] += v
        nb[pos[d]] += 1
    pt = sa.sum() / max(1, na.sum()) - sb.sum() / max(1, nb.sum())
    bs = []
    for _ in range(boot):
        pick = [rnd.randrange(len(ud)) for _ in range(len(ud))]
        bs.append(sa[pick].sum() / max(1, na[pick].sum()) - sb[pick].sum() / max(1, nb[pick].sum()))
    bs.sort()
    a = (1 - CI) / 2
    return float(pt), float(bs[int(a * boot)]), float(bs[int((1 - a) * boot) - 1])


def outcome_shadow(lo, hi):
    if lo >= G.NONINF_PP:
        return "shadow-worthy"
    if hi < 0:
        return "not worth a shadow"
    return "undetermined"


def _mlds_keep(r):
    return str(r.get("signal_type", "none")) != "none" and (r.get("labels") or {}).get("ret_5") is not None


def main():
    P = G.prepare()
    meta, X, lab, pnl, M, GV, cfg = P.meta, P.X, P.lab, P.pnl, P.M, P.GV, P.cfg
    cooldown = int(getattr(cfg, "COOLDOWN_BARS", 19))
    print("cooldown: %d bars" % cooldown)

    # ---- the live training population
    mmeta, mX = G.load_rows(HERE / "ml_dataset.jsonl", P.names, M, keep=_mlds_keep)
    my, mt = [], []
    for j, m in enumerate(mmeta):
        if m["sym"] not in P.coins:
            P.coins[m["sym"]] = G.Coin(m["sym"], cfg.ATR_PERIOD)
        co = P.coins[m["sym"]]
        i = G.decision_index(co.idx, m["ts"], m["tf"])
        p5 = G.peak_after(co.h, co.c, i, G.PEAK5_BARS[m["tf"]]) if i is not None else None
        if p5 is not None:
            my.append((j, int(p5 >= 2.0)))
    mX = mX[[j for j, _ in my]]
    mt = [mmeta[j]["dec"] for j, _ in my]
    my = np.array([y for _, y in my], dtype=float)
    bym = collections.Counter(t.strftime("%Y-%m") for t in mt)
    print("ml_dataset signal rows: %d, resolvable with the live label: %d (base %.1f%%); by month %s"
          % (len(mmeta), len(mt), 100 * my.mean(), dict(sorted(bym.items()))))

    times = [m["dec"] for m in meta]
    months = sorted({t.strftime("%Y-%m") for t in times if t.strftime("%Y-%m") >= G.TEST_FROM})
    sc = {k: np.full(len(meta), np.nan) for k in ("C_peak5", "GOAL", "SRC_MLDS")}
    mts = np.array([t.timestamp() for t in mt])
    print("\nfolds (month, critic train, ml_dataset train, test):")
    for mo, tr, te in G.folds(times, months):
        tr = np.array([i for i in tr if lab[i] is not None])
        start = datetime.strptime(mo + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        mtr = np.where(mts < (start - G.EMBARGO).timestamp())[0]
        ok = len(tr) >= 2000 and len(mtr) >= MIN_MLDS_TRAIN and len(te) > 0
        print("  %s  %6d  %6d  %6d  %s" % (mo, len(tr), len(mtr), len(te), "ok" if ok else "skipped"))
        if not ok:
            continue
        s1 = M.StandardScaler().fit(X[tr])
        for a in ("C_peak5", "GOAL"):
            y = np.array([lab[i][a] for i in tr], dtype=float)
            sc[a][te] = M.CatBoostModel(X.shape[1]).fit(s1.transform(X[tr]), y).predict_proba(s1.transform(X[te]))
        s2 = M.StandardScaler().fit(mX[mtr])
        sc["SRC_MLDS"][te] = M.CatBoostModel(X.shape[1]).fit(s2.transform(mX[mtr]), my[mtr]).predict_proba(
            s2.transform(X[te]))

    test = np.array([i for i in range(len(meta)) if all(np.isfinite(v[i]) for v in sc.values())])
    tmonths = sorted({times[i].strftime("%Y-%m") for i in test})
    tmo = np.array([times[i].strftime("%Y-%m") for i in test])
    W = [w for w in GV.winner_days(G.TEST_FROM + "-01") if w[0][:7] in tmonths]
    print("out-of-sample rows %d, months %s, winner-days %d" % (len(test), ", ".join(tmonths), len(W)))
    rnd = random.Random(1012)

    def admitted(a, share):
        mask = np.zeros(len(test), dtype=bool)
        for mo in tmonths:
            sel = np.where(tmo == mo)[0]
            mask[sel] = G.admit_top(sc[a][test[sel]], share)
        return mask

    def grade(arm, ctrl, share):
        ma, mc = admitted(arm, share), admitted(ctrl, share)
        ra = np.array(G.reachable([(meta[test[k]]["sym"], times[test[k]]) for k in np.where(ma)[0]], W), float)
        rc = np.array(G.reachable([(meta[test[k]]["sym"], times[test[k]]) for k in np.where(mc)[0]], W), float)
        d = ra - rc
        n = len(W)
        gb = sorted(100 * float(np.mean(d[[rnd.randrange(n) for _ in range(n)]])) for _ in range(BOOT))
        al = (1 - CI) / 2
        ms = []
        for mo in tmonths:
            sel = [k for k, w in enumerate(W) if w[0][:7] == mo]
            if sel:
                ms.append(float(np.mean(d[sel])) >= 0)
        out = {"goal_arm": 100 * ra.mean(), "goal_ctrl": 100 * rc.mean(), "goal": 100 * float(d.mean()),
               "goal_lo": gb[int(al * BOOT)], "goal_hi": gb[int((1 - al) * BOOT) - 1],
               "months": float(np.mean(ms)) if ms else 0.0}

        def rows_of(mask):
            ks = [test[k] for k in np.where(mask)[0]]
            return ks, [{"sym": meta[i]["sym"], "tf": meta[i]["tf"], "dec": times[i], "exit": meta[i].get("exit"),
                         "pnl": pnl[i]} for i in ks]

        def day(i):
            return times[i].strftime("%Y-%m-%d")
        ka, kc = [i for i in rows_of(ma)[0] if np.isfinite(pnl[i])], [i for i in rows_of(mc)[0] if np.isfinite(pnl[i])]
        out["row"] = cluster_delta([day(i) for i in ka], pnl[ka], [day(i) for i in kc], pnl[kc], rnd)
        out["row_means"] = (float(np.mean(pnl[ka])), float(np.mean(pnl[kc])), len(ka), len(kc))
        ia, la = rows_of(ma)
        ic, lc = rows_of(mc)
        ta = [ia[k] for k in dedup_trades(la, cooldown)]
        tc = [ic[k] for k in dedup_trades(lc, cooldown)]
        out["trade"] = cluster_delta([day(i) for i in ta], pnl[ta], [day(i) for i in tc], pnl[tc], rnd)
        out["trade_means"] = (float(np.mean(pnl[ta])), float(np.mean(pnl[tc])), len(ta), len(tc))
        add, rem = split_changed(ma, mc)
        pa = [pnl[test[k]] for k in add if np.isfinite(pnl[test[k]])]
        pr = [pnl[test[k]] for k in rem if np.isfinite(pnl[test[k]])]
        out["changed"] = (len(pa), float(np.mean(pa)) if pa else float("nan"),
                          len(pr), float(np.mean(pr)) if pr else float("nan"))
        return out

    def show(name, g):
        print("  goal: %s %.1f%% vs control %.1f%% -> %+.2f pp [%+.2f, %+.2f], months >= 0: %.0f%%" % (
            name, g["goal_arm"], g["goal_ctrl"], g["goal"], g["goal_lo"], g["goal_hi"], 100 * g["months"]))
        a, c, na, nc = g["trade_means"]
        print("  trades as the bot takes them: arm %+.3f%% (n=%d) vs control %+.3f%% (n=%d) -> %+.3f pp [%+.3f, %+.3f]"
              % (a, na, c, nc, *g["trade"]))
        a, c, na, nc = g["row_means"]
        print("  every admitted row (parent's estimand): arm %+.3f%% (n=%d) vs %+.3f%% (n=%d) -> %+.3f pp [%+.3f, %+.3f]"
              % (a, na, c, nc, *g["row"]))
        print("  changed rows: only the arm admits %d (mean %+.3f%%), only the control admits %d (mean %+.3f%%)"
              % g["changed"])

    print("\n" + "=" * 100)
    print("TEST 1 (EXPLORATORY, same data as the parent): GOAL label vs live label, both trained on candidates")
    print("=" * 100)
    res1 = {}
    for share in (G.A_PRIMARY, G.A_SECONDARY):
        print("admission %.0f%%:" % (100 * share))
        res1[share] = grade("GOAL", "C_peak5", share)
        show("GOAL", res1[share])
    g = res1[G.A_PRIMARY]
    print("TEST 1 OUTCOME at 60%%: %s" % outcome_shadow(g["trade"][1], g["trade"][2]))

    print("\n" + "=" * 100)
    print("TEST 2 (pre-registered): live label, trained on candidates (SRC_CRITIC) vs on ml_dataset (SRC_MLDS)")
    print("=" * 100)
    sc["SRC_CRITIC"] = sc["C_peak5"]
    res2 = {}
    for share in (G.A_PRIMARY, G.A_SECONDARY):
        print("admission %.0f%%:" % (100 * share))
        res2[share] = grade("SRC_CRITIC", "SRC_MLDS", share)
        show("SRC_CRITIC", res2[share])
    g = res2[G.A_PRIMARY]
    v2 = G.verdict(g["goal"], g["goal_lo"], g["trade"][1], g["months"])
    print("TEST 2 VERDICT at 60%%: %s" % v2)

    print("\ncontext:")
    y5 = np.array([lab[i]["C_peak5"] if lab[i] else -1 for i in test])
    ok = y5 >= 0
    for a in ("SRC_MLDS", "SRC_CRITIC", "GOAL"):
        print("  %-10s AUC vs live label %.3f" % (a, M.roc_auc_score_np(y5[ok].astype(float), sc[a][test][ok]) or float("nan")))
    for k in range(len(test)):
        i = test[k]
        if meta[i]["sym"] == "ICPUSDT" and meta[i]["ts"] == datetime(2026, 10, 11, 13, 0, tzinfo=timezone.utc) \
                and meta[i]["tf"] == "15m":
            sel = np.where(tmo == tmo[k])[0]
            for a in ("SRC_MLDS", "SRC_CRITIC", "GOAL"):
                s = sc[a][test[sel]]
                print("  ICP 10-11 13:00 %-10s score %.3f, rank percentile in its month %.0f%%"
                      % (a, sc[a][i], 100 * float(np.mean(s <= sc[a][i]))))
    print(json.dumps({"test1": outcome_shadow(res1[G.A_PRIMARY]["trade"][1], res1[G.A_PRIMARY]["trade"][2]),
                      "test2": v2}))


if __name__ == "__main__":
    main()
