"""Time-held-out ranking at a fixed alert budget (report step P1, 2026-10-01).

Today: ~31 entry messages/day at 13.6% precision (share on immutable top-20
winner-days); the canonical targets are <= 10 messages/day and 35% precision,
and the operator wants 1-3 leaders without junk. Question: can a score known at
entry time decide WHICH of the bot's own entries to send, at a fixed budget,
without losing the early winner catches?

Population: the bot's live entry events, 2026-05-01 .. last labelled day, days
with the bot up all day; one message = one entry event; precision and pnl per
first entry of a coin-day. Scores available at entry (pre-registered family):
  ranker_top_gainer_prob, ranker_final_score, ranker_ev, ml_proba,
  candidate_score, day rank (return since the UTC open, watchlist, at the last
  closed 15m bar; lower is better), and a logistic combination of all of them
  fitted on TRAIN only.
Online rule: send if score >= T, T set on TRAIN so the mean messages/day = B
(B = 3, 5, 10). The score with the best TRAIN precision at B is chosen; nothing
is chosen on TEST.

TRAIN = 2026-05-01 .. 07-31, TEST = 2026-08-01 .. last day.
Pre-registered success at B = 10, on TEST, against all entries on TEST:
  precision >= 2x the current, with its Wilson 95% lower bound above the current;
  messages/day <= 10;
  early winner catches (entered before the +2.5% crossing) kept >= 50%;
  mean realised pnl of the sent >= current mean - 0.10 pp (bootstrap lower bound).
"""
import collections
import io
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as cfg  # noqa: E402
import exit_validator as EV  # noqa: E402
import goal_validator as GV  # noqa: E402

START, SPLIT = "2026-05-01", "2026-08-01"
BUDGETS = (3, 5, 10)
SCORES = ("ranker_top_gainer_prob", "ranker_final_score", "ranker_ev", "ml_proba", "candidate_score",
          "neg_day_rank", "logit_all")
UTC = timezone.utc


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_entries(g):
    out, exits = [], collections.defaultdict(list)
    with io.open(FILES / "bot_events.jsonl", "rb") as fh:
        for raw in fh:
            if b'"entry"' not in raw and b'"exit"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            if e.get("event") not in ("entry", "exit") or str(e.get("ts", "")) < START:
                continue
            d = GV._dt(e["ts"])
            if e["event"] == "exit":
                exits[(e.get("sym"), e.get("tf"))].append((d, e.get("pnl_pct")))
                continue
            r = {"sym": e.get("sym"), "tf": e.get("tf"), "dt": d, "day": d.strftime("%Y-%m-%d")}
            for k in SCORES[:5]:
                v = e.get(k)
                r[k] = float(v) if isinstance(v, (int, float)) else None
            j = g.col.get(r["sym"])
            i = g.idx(GV.bar_open(d, "15m")) - 1
            if j is not None and 0 <= i < g.N and math.isfinite(g.RET[i, j]):
                r["neg_day_rank"] = -float(g.RANK[i, j])
                r["ret_open"] = float(g.RET[i, j])
            else:
                r["neg_day_rank"] = None
                r["ret_open"] = None
            out.append(r)
    for r in out:
        xs = sorted(exits.get((r["sym"], r["tf"]), []), key=lambda x: x[0])
        x = next((x for x in xs if x[0] > r["dt"]), None)
        r["pnl"] = float(x[1]) if x and isinstance(x[1], (int, float)) else None
    return out


def fit_logit(rows, feats):
    X = np.array([[r[f] if r[f] is not None else 0.0 for f in feats] for r in rows], dtype=float)
    y = np.array([r["win"] for r in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(len(feats)), 0.0
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        gw, gb = Z.T @ (p - y) / len(y) + 1e-3 * w, float(np.mean(p - y))
        w -= 0.5 * gw
        b -= 0.5 * gb
    return lambda r: float(((np.array([r[f] if r[f] is not None else 0.0 for f in feats]) - mu) / sd) @ w + b)


def main():
    g = EV.load_grid(cfg.ATR_PERIOD)
    winners_list = GV.winner_days(START)
    winners = {(d, s) for d, s, _, _ in winners_list}
    deadline = {(d, s): dd for d, s, _, dd in winners_list}
    full = {d for d, _, _, _ in winners_list}
    rows = [r for r in load_entries(g) if r["day"] in full]
    for r in rows:
        r["win"] = 1.0 if (r["day"], r["sym"]) in winners else 0.0
        dd = deadline.get((r["day"], r["sym"]))
        r["early_win"] = bool(dd and r["dt"] < dd)
    tr = [r for r in rows if r["day"] < SPLIT]
    te = [r for r in rows if r["day"] >= SPLIT]
    days_tr = len({r["day"] for r in tr}) or 1
    days_te = len({r["day"] for r in te}) or 1
    feats = list(SCORES[:6]) + ["ret_open"]
    logit = fit_logit(tr, feats)
    for r in rows:
        r["logit_all"] = logit(r)

    def metrics(sel, all_rows, days):
        first = {}
        for r in sorted(sel, key=lambda r: r["dt"]):
            first.setdefault((r["day"], r["sym"]), r)
        n = len(first)
        k = sum(1 for r in first.values() if r["win"])
        early = {(r["day"], r["sym"]) for r in sel if r["early_win"]}
        pnl = [r["pnl"] for r in first.values() if r["pnl"] is not None]
        return {"msgs_day": len(sel) / days, "coin_days": n, "precision": k / n if n else 0.0, "k": k,
                "early_winner_days": len(early), "pnl": pnl}

    cur_tr, cur_te = metrics(tr, tr, days_tr), metrics(te, te, days_te)
    print("entries: train %d (%s..07-31, %d days), test %d (08-01.., %d days)" % (len(tr), START, days_tr, len(te), days_te))
    print("CURRENT  train: %.1f msg/day, precision %.3f (%d/%d), early winner-days %d | test: %.1f msg/day, "
          "precision %.3f (%d/%d), early winner-days %d, mean pnl %+.3f%%" % (
              cur_tr["msgs_day"], cur_tr["precision"], cur_tr["k"], cur_tr["coin_days"], cur_tr["early_winner_days"],
              cur_te["msgs_day"], cur_te["precision"], cur_te["k"], cur_te["coin_days"], cur_te["early_winner_days"],
              np.mean(cur_te["pnl"])))
    rnd = random.Random(11)
    result = {"date": datetime.now(UTC).strftime("%Y-%m-%d"), "train": [START, SPLIT], "budgets": {},
              "current_test": {"precision": cur_te["precision"], "msgs_day": cur_te["msgs_day"],
                               "early_winner_days": cur_te["early_winner_days"]}}
    for B in BUDGETS:
        best = None
        print("\nBUDGET %d msg/day -- TRAIN choice:" % B)
        for s in SCORES:
            vals = sorted((r[s] for r in tr if r[s] is not None), reverse=True)
            if len(vals) < B * days_tr:
                continue
            T = vals[B * days_tr - 1]
            sel = [r for r in tr if r[s] is not None and r[s] >= T]
            m = metrics(sel, tr, days_tr)
            print("   %-24s T=%.4f  precision %.3f (%d/%d)  early %d" % (s, T, m["precision"], m["k"], m["coin_days"],
                                                                        m["early_winner_days"]))
            if best is None or m["precision"] > best[2]:
                best = (s, T, m["precision"])
        s, T, _ = best
        sel = [r for r in te if r[s] is not None and r[s] >= T]
        m = metrics(sel, te, days_te)
        lo, hi = wilson(m["k"], m["coin_days"])
        bs = sorted(np.mean(rnd.choices(m["pnl"], k=len(m["pnl"]))) for _ in range(1000)) if m["pnl"] else [0] * 1000
        keep = m["early_winner_days"] / max(1, cur_te["early_winner_days"])
        ok = (m["precision"] >= 2 * cur_te["precision"] and lo > cur_te["precision"] and m["msgs_day"] <= 10
              and keep >= 0.5 and bs[25] >= np.mean(cur_te["pnl"]) - 0.10)
        print("  TEST with %s >= %.4f: %.1f msg/day, precision %.3f [%.3f, %.3f] (%d/%d) vs current %.3f; "
              "early winner-days %d of %d (%.0f%%); mean pnl %+.3f%% [%+.3f, %+.3f] vs %+.3f%% -> %s" % (
                  s, T, m["msgs_day"], m["precision"], lo, hi, m["k"], m["coin_days"], cur_te["precision"],
                  m["early_winner_days"], cur_te["early_winner_days"], 100 * keep, np.mean(m["pnl"]) if m["pnl"] else 0,
                  bs[25], bs[-26], np.mean(cur_te["pnl"]), "SUPPORTED" if ok else "refuted"))
        result["budgets"][str(B)] = {"score": s, "threshold": T, "msgs_day": m["msgs_day"],
                                     "precision": m["precision"], "wilson95": [lo, hi],
                                     "early_winner_days": m["early_winner_days"], "verdict": "supported" if ok else "refuted"}
    out = FILES.parent / ".runtime" / "backtests" / "alert_budget_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("written %s" % out)


if __name__ == "__main__":
    main()
