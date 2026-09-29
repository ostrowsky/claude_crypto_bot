"""Promotion gate for the nightly ML signal model: the new model replaces the
live one unless it is CLEARLY worse on rows neither model trained on.

WHY (2026-09-29, agent-tasks-0929-spec.md §6)

daily_learning retrains ml_signal_model.json every night and overwrote the live
file unconditionally. The trainer picks the best of three families on its own
validation split, but nothing compared the new model with the one it replaces.
A bad night (thin labels, a regime shift in the last 15% of rows) would ship
straight into the live ML gate.

WHAT IS COMPARED

  rows     the trainer's own TEST split -- the newest 15% of rows by time. The
           candidate never trained on them (train = first 70%), and neither did
           the incumbent: it trained on the first 70% of yesterday's rows, which
           end before today's 85% mark.
  metric A AUC against the label the model is trained for (the peak label while
           ML_PEAK_LABEL_ENABLED) -- the same exam the trainer selects by
  metric B recall of positive rows at the LIVE floor (ML_GENERAL_HARD_BLOCK_MIN,
           the bull-day floor on bull days): how many future movers the gate
           would let through
  metric C precision at the live floor: share of admitted rows that are movers
           -- added after the first run (2026-09-29), which showed a candidate
           admitting 82% of rows instead of 62%: recall rose only because the
           score scale moved against a FIXED floor, and precision fell (~0.31 ->
           ~0.27, base 0.24). Recall alone rewards "admit everything".
  paired bootstrap over rows, 500 resamples, 95% interval of (new - old)

DECISION, fixed before the first run

  keep the incumbent  if the upper bound of the AUC, recall or precision
                      difference < 0 (clearly worse on any of the three)
  promote             otherwise -- including "no better": a model is not frozen
                      because the new one merely ties (CLAUDE.md §0: learning is P0)
  promote, ungated    no incumbent, unreadable incumbent, or a different label
                      version (the two answer different questions)

Every decision is appended to .runtime/ml_promotion_log.jsonl; a kept incumbent
leaves the candidate at ml_signal_model.candidate.json. Rollback:
ML_PROMOTION_GATE_ENABLED = False (overwrite every night, as before).
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE.parent / ".runtime" / "ml_promotion_log.jsonl"
BOOT = 500


def auc(y, s) -> float | None:
    pos = [v for v, t in zip(s, y) if t > 0.5]
    neg = [v for v, t in zip(s, y) if t <= 0.5]
    if not pos or not neg:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rp = sum(r for r, t in zip(ranks, y) if t > 0.5)
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def recall_at_floor(y, s, floors) -> float | None:
    pos = [(v, f) for v, t, f in zip(s, y, floors) if t > 0.5]
    if not pos:
        return None
    return sum(1 for v, f in pos if v >= f) / len(pos)


def precision_at_floor(y, s, floors) -> float | None:
    adm = [t for v, t, f in zip(s, y, floors) if v >= f]
    if not adm:
        return None
    return sum(1 for t in adm if t > 0.5) / len(adm)


def floors_for(rows, cfg) -> list:
    lo = float(getattr(cfg, "ML_GENERAL_HARD_BLOCK_MIN", 0.0))
    bull = float(getattr(cfg, "ML_GENERAL_HARD_BLOCK_BULL_DAY_MIN", lo))
    return [bull if r.get("is_bull_day") else lo for r in rows]


def compare(y, s_old, s_new, floors, rnd: random.Random) -> dict:
    n = len(y)
    out = {"n": n, "positives": int(sum(1 for t in y if t > 0.5))}
    a_old, a_new = auc(y, s_old), auc(y, s_new)
    r_old, r_new = recall_at_floor(y, s_old, floors), recall_at_floor(y, s_new, floors)
    p_old, p_new = precision_at_floor(y, s_old, floors), precision_at_floor(y, s_new, floors)
    out.update(auc_old=a_old, auc_new=a_new, recall_old=r_old, recall_new=r_new,
               precision_old=p_old, precision_new=p_new, base_rate=out["positives"] / n if n else None,
               admitted_old=sum(1 for v, f in zip(s_old, floors) if v >= f) / n if n else None,
               admitted_new=sum(1 for v, f in zip(s_new, floors) if v >= f) / n if n else None)
    da, dr, dp = [], [], []
    for _ in range(BOOT):
        idx = [rnd.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        so, sn, ff = [s_old[i] for i in idx], [s_new[i] for i in idx], [floors[i] for i in idx]
        ao, an = auc(yy, so), auc(yy, sn)
        ro, rn = recall_at_floor(yy, so, ff), recall_at_floor(yy, sn, ff)
        if None not in (ao, an):
            da.append(an - ao)
        if None not in (ro, rn):
            dr.append(rn - ro)
        po, pn = precision_at_floor(yy, so, ff), precision_at_floor(yy, sn, ff)
        if None not in (po, pn):
            dp.append(pn - po)

    def ci(v):
        if not v:
            return None
        v.sort()
        return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]
    out["auc_diff_ci95"], out["recall_diff_ci95"], out["precision_diff_ci95"] = ci(da), ci(dr), ci(dp)
    return out


def decide(cmp_: dict) -> tuple:
    a, r = cmp_.get("auc_diff_ci95"), cmp_.get("recall_diff_ci95")
    if a and a[1] < 0:
        return "keep_incumbent", "new model clearly worse on AUC (upper bound %+.4f)" % a[1]
    if r and r[1] < 0:
        return "keep_incumbent", "new model clearly worse on recall at the live floor (upper bound %+.4f)" % r[1]
    p = cmp_.get("precision_diff_ci95")
    if p and p[1] < 0:
        return "keep_incumbent", "new model clearly worse on precision at the live floor (upper bound %+.4f)" % p[1]
    return "promote", "not clearly worse on AUC, recall or precision at the live floor"


def evaluate(incumbent_path: Path, candidate_payload: dict, holdout: dict, cfg=None, predict=None) -> dict:
    """Decision for tonight's candidate. `holdout` = {"rows": [...], "y": [...]}."""
    if cfg is None:
        import config as cfg
    if predict is None:
        from ml_signal_model import predict_proba_from_payload as predict
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "candidate_model": candidate_payload.get("model_name"),
           "label_version": candidate_payload.get("label_version")}
    try:
        inc = json.loads(Path(incumbent_path).read_text(encoding="utf-8"))
    except Exception as e:
        return {**rec, "decision": "promote", "ungated": True, "reason": f"no readable incumbent: {e!r}"[:200]}
    rec["incumbent_model"] = inc.get("model_name")
    if inc.get("label_version") != candidate_payload.get("label_version"):
        return {**rec, "decision": "promote", "ungated": True,
                "reason": "label version changed (%s -> %s): not comparable"
                          % (inc.get("label_version"), candidate_payload.get("label_version"))}
    rows, y = holdout.get("rows") or [], list(holdout.get("y") or [])
    if len(rows) < 100 or len(rows) != len(y):
        return {**rec, "decision": "promote", "ungated": True, "reason": f"holdout too small ({len(rows)})"}
    s_old = [predict(inc, r) for r in rows]
    s_new = [predict(candidate_payload, r) for r in rows]
    cmp_ = compare(y, s_old, s_new, floors_for(rows, cfg), random.Random(29))
    d, why = decide(cmp_)
    return {**rec, "decision": d, "reason": why, "comparison": cmp_}


def log_decision(rec: dict, path: Path | None = None) -> None:
    path = path or LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def consecutive_keeps(path: Path | None = None) -> int:
    """How many nights in a row the incumbent was kept (a stuck model is a finding)."""
    path = path or LOG
    try:
        recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    except Exception:
        return 0
    n = 0
    for r in reversed(recs):
        if r.get("decision") != "keep_incumbent":
            break
        n += 1
    return n
