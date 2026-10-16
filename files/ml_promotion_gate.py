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

Every decision is appended to .runtime/ml_promotion_log.jsonl; every candidate,
kept or promoted, is left at ml_signal_model.candidate.json (since 2026-10-09). Rollback:
ML_PROMOTION_GATE_ENABLED = False (overwrite every night, as before).

MATCHED ADMISSION (2026-10-09, ml-admission-match-1009-spec.md)

The incumbent was kept seven nights running (10-02..10-08), each time "clearly
worse on precision at the live floor". Every candidate was a logistic model with
the SAME AUC whose scores sit higher, so the fixed floor 0.15 admitted ~84% of
rows instead of ~66% and precision fell mechanically. At an equal admitted share
the candidate was neither better nor worse (_backtest_ml_promotion_matched.py).
A fixed floor compares score scales, not models -- learning could not reach the
gate whenever the winning family changed.

So every night the gate also computes the MATCHED comparison: the candidate gets
its own floor, the score at which it admits the same share of tonight's held-out
rows as the incumbent admits at its effective floor (its own admission_floor if
it carries one, else the live config floor), separately for bull and other days.
The candidate payload always carries that `admission_floor`. And for the rows
that arrived after yesterday's candidate was built, it records how yesterday's
candidate at ITS stored floor behaved against the incumbent -- an out-of-sample
check that the matched floor keeps the admitted share on new data (readout
ML-ADMISSION-MATCH).

ML_PROMOTION_MATCHED_ADMISSION_ENABLED = False (SHADOW): decisions still use the
fixed floor; the matched decision is only logged. True: the decision uses the
matched comparison and the live gate reads the model's admission_floor
(monitor._ml_floor_admission) -- both together, never one without the other,
because promoting a model at matched admission while the bot keeps the fixed
floor would widen the gate. Rollback: False + restart.
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


def compare(y, s_old, s_new, floors, rnd: random.Random, floors_new=None) -> dict:
    """floors_new: the candidate's own per-row floors (matched admission); default = floors."""
    floors_old = floors
    floors_new = floors if floors_new is None else floors_new
    n = len(y)
    out = {"n": n, "positives": int(sum(1 for t in y if t > 0.5))}
    a_old, a_new = auc(y, s_old), auc(y, s_new)
    r_old, r_new = recall_at_floor(y, s_old, floors_old), recall_at_floor(y, s_new, floors_new)
    p_old, p_new = precision_at_floor(y, s_old, floors_old), precision_at_floor(y, s_new, floors_new)
    out.update(auc_old=a_old, auc_new=a_new, recall_old=r_old, recall_new=r_new,
               precision_old=p_old, precision_new=p_new, base_rate=out["positives"] / n if n else None,
               admitted_old=sum(1 for v, f in zip(s_old, floors_old) if v >= f) / n if n else None,
               admitted_new=sum(1 for v, f in zip(s_new, floors_new) if v >= f) / n if n else None)
    da, dr, dp = [], [], []
    for _ in range(BOOT):
        idx = [rnd.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        so, sn = [s_old[i] for i in idx], [s_new[i] for i in idx]
        ff, fn = [floors_old[i] for i in idx], [floors_new[i] for i in idx]
        ao, an = auc(yy, so), auc(yy, sn)
        ro, rn = recall_at_floor(yy, so, ff), recall_at_floor(yy, sn, fn)
        if None not in (ao, an):
            da.append(an - ao)
        if None not in (ro, rn):
            dr.append(rn - ro)
        po, pn = precision_at_floor(yy, so, ff), precision_at_floor(yy, sn, fn)
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


def matched_enabled(cfg=None) -> bool:
    if cfg is None:
        try:
            import config as cfg
        except Exception:
            return False
    return bool(getattr(cfg, "ML_PROMOTION_MATCHED_ADMISSION_ENABLED", False))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def effective_floors(payload: dict, rows, cfg) -> list:
    """Per-row floor a model is judged at: its own admission_floor if it carries a
    complete one, else the live config floor."""
    live = floors_for(rows, cfg)
    af = (payload or {}).get("admission_floor") or {}
    d, b = af.get("default"), af.get("bull_day")
    if not (_num(d) and _num(b)):
        return live
    return [float(b) if r.get("is_bull_day") else float(d) for r in rows]


def floor_for_share(scores, share: float):
    """Lowest score among the top `share` of `scores`: admitting v >= floor lets
    through that share (ties may add a little). None for no scores."""
    if not scores:
        return None
    s = sorted(scores, reverse=True)
    k = max(1, min(len(s), int(round(share * len(s)))))
    return float(s[k - 1])


def matched_floors(rows, s_old, s_new, floors_old) -> dict:
    """Candidate floors that admit the incumbent's share, per day type.
    A day type absent from the rows inherits the other one's floor."""
    out, shares = {}, {}
    for key, bull in (("default", False), ("bull_day", True)):
        idx = [i for i, r in enumerate(rows) if bool(r.get("is_bull_day")) == bull]
        if not idx:
            continue
        share = sum(1 for i in idx if s_old[i] >= floors_old[i]) / len(idx)
        shares[key] = round(share, 4)
        out[key] = floor_for_share([s_new[i] for i in idx], share)
    if not out:
        return {}
    out.setdefault("default", out.get("bull_day"))
    out.setdefault("bull_day", out.get("default"))
    return {"default": round(out["default"], 6), "bull_day": round(out["bull_day"], 6),
            "incumbent_share": shares, "rows": len(rows)}


def _row_dt(r) -> str:
    dt = r.get("_dt")
    return dt.isoformat() if hasattr(dt, "isoformat") else str(r.get("ts_signal") or "")


def transfer_rows(prev_candidate, incumbent: dict, rows, y, cfg, predict):
    """Yesterday's candidate at ITS stored floor vs the incumbent, on rows newer than
    yesterday's holdout -- rows that floor was not fitted on. None if not applicable."""
    af = (prev_candidate or {}).get("admission_floor") or {}
    since = af.get("holdout_to")
    if not since or not (_num(af.get("default")) and _num(af.get("bull_day"))):
        return None
    fresh = [(r, t) for r, t in zip(rows, y) if _row_dt(r) > since]
    out = {"since": since, "candidate_model": prev_candidate.get("model_name"), "rows": []}
    if not fresh:
        return out
    fr = [r for r, _ in fresh]
    f_old = effective_floors(incumbent, fr, cfg) if matched_enabled(cfg) else floors_for(fr, cfg)
    f_new = effective_floors(prev_candidate, fr, cfg)
    for (r, t), fo, fn in zip(fresh, f_old, f_new):
        out["rows"].append([_row_dt(r)[:19], r.get("sym"), int(t > 0.5),
                            int(predict(incumbent, r) >= fo), int(predict(prev_candidate, r) >= fn)])
    return out


def evaluate(incumbent_path: Path, candidate_payload: dict, holdout: dict, cfg=None, predict=None,
             prev_candidate=None) -> dict:
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
    live_matched = matched_enabled(cfg)
    # the incumbent is judged at the floor the bot applies to it: its own
    # admission_floor only while the live gate reads it (flag on), else the config floor
    f_old = effective_floors(inc, rows, cfg) if live_matched else floors_for(rows, cfg)
    cmp_ = compare(y, s_old, s_new, f_old, random.Random(29))
    d, why = decide(cmp_)
    out = {**rec, "decision": d, "reason": why, "comparison": cmp_, "matched_live": live_matched}
    # matched admission: computed every night, decisive only behind the flag
    try:
        mf = matched_floors(rows, s_old, s_new, f_old)
        if mf:
            mf["holdout_to"] = _row_dt(rows[-1])
            f_new = [mf["bull_day"] if r.get("is_bull_day") else mf["default"] for r in rows]
            cm = compare(y, s_old, s_new, f_old, random.Random(29), floors_new=f_new)
            dm, whym = decide(cm)
            out["matched"] = {"decision": dm, "reason": whym.replace("live floor", "matched admission"),
                              "comparison": cm}
            out["admission_floor"] = mf
            if out["matched_live"]:
                out["fixed_floor_decision"] = {"decision": d, "reason": why}
                out["decision"], out["reason"] = dm, out["matched"]["reason"]
        tr = transfer_rows(prev_candidate, inc, rows, y, cfg, predict)
        if tr is not None:
            out["transfer"] = tr
    except Exception as e:  # the shadow must never break the nightly decision
        out["matched_error"] = repr(e)[:200]
    return out


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
