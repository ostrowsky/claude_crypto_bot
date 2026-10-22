"""F1 -- the trader's own decisions as a learning dataset (trader-learning-loop-f-1020-spec.md §3, §6.1-6.2).

One row per executor decision (book exec_x4: entry -> enter / skip). The outcome of a live entry comes from
the executor's exit row of the same key (the exchange's answer); reconciliation with myTrades is phase D.
The file is append-only: an entry stored while still open gets its outcome later as a separate 'outcome'
row, never by rewriting. Idempotent by decision_id. Features are attached only if known at the entry time.

  rows_from_decisions(decisions)        -> dataset rows (entries, with outcome when the exit is there)
  snapshot_at(snaps, sym, t, max_age)   -> latest snapshot <= t, not older than max_age, or None
  append_new(path, rows)                -> number of lines appended (rows + late outcomes)
  counterfactual_rows(skips, replay)    -> 'if we had entered' rows from the replay
  live_vs_replay(live, replay)          -> per-trade difference live - replay (spec §0 rule 3 diagnostic)
"""
from __future__ import annotations

import bisect
import json
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                  # the embedded Python does not add the script's folder
    sys.path.insert(0, str(HERE))
import trader_config as TC  # noqa: E402

DATASET = TC.RUNTIME / "learning" / "trades.jsonl"
BOOK = "exec_x4"
ENTRY_FIELDS = ("fill_price", "paper_price", "slippage_bps", "latency_sec", "cost", "qty", "quote", "reason",
                "policy_stop", "stop_price", "protected", "status")


def _dt(v):
    if isinstance(v, datetime):
        d = v
    else:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _did(r) -> str:
    return r.get("decision_id") or f"{r.get('kind')}|{r.get('key')}|{r.get('action')}"


def rows_from_decisions(decisions) -> list:
    """Entries (enter / skip) of the executor book, in journal order; the exit row of a key becomes the
    outcome of its enter row. Other books and other kinds are not dataset rows."""
    exits, entries = {}, []
    for r in decisions:
        if r.get("book") != BOOK:
            continue
        k = r.get("kind")
        if k == "real_exit" and r.get("action") == "exit":
            exits[r.get("key")] = {"ret_pct": r.get("ret_pct"), "pnl": r.get("pnl"), "exit_reason": r.get("reason"),
                                   "exit_ts": r.get("ts")}
        elif k == "real_entry" and r.get("action") in ("enter", "skip"):
            entries.append(r)
    out = []
    for r in entries:
        row = {"kind": "entry", "action": r["action"], "key": r.get("key"), "sym": r.get("sym"), "ts": r.get("ts"),
               "mode": r.get("mode"), "decision_id": _did(r), "source": "live",
               **{f: r.get(f) for f in ENTRY_FIELDS if f in r}}
        row["outcome"] = exits.get(r.get("key")) if r["action"] == "enter" else None
        out.append(row)
    return out


def snapshot_at(snaps: dict, sym: str, t, max_age_sec: float):
    """snaps: {sym: [(ts, {feature: value}), ...]} sorted or not. The latest snapshot at or before t, not older
    than max_age_sec, as a copy with age_sec; None otherwise (never a snapshot from after the decision)."""
    lst = snaps.get(sym)
    if not lst:
        return None
    t = _dt(t)
    ts = sorted((_dt(a), b) for a, b in lst)
    keys = [a for a, _ in ts]
    i = bisect.bisect_right(keys, t) - 1
    if i < 0:
        return None
    age = (t - keys[i]).total_seconds()
    if age > max_age_sec:
        return None
    return {**ts[i][1], "age_sec": age}


def _read(path: Path):
    ids, with_outcome, outcome_keys = set(), set(), set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue                                 # a broken or partial line is skipped, not fatal
            if not isinstance(r, dict):
                continue
            ids.add(r.get("decision_id"))
            if r.get("kind") == "outcome":
                outcome_keys.add(r.get("key"))
            elif r.get("outcome") is not None:
                with_outcome.add(r.get("decision_id"))
    return ids, with_outcome, outcome_keys


def append_new(path, rows) -> int:
    path = Path(path)
    ids, with_outcome, outcome_keys = _read(path)
    new = []
    for r in rows:
        did = r["decision_id"]
        if did not in ids:
            new.append(r)
            ids.add(did)
            if r.get("outcome") is not None:
                with_outcome.add(did)
        elif (r.get("outcome") is not None and did not in with_outcome and r.get("key") not in outcome_keys):
            new.append({"kind": "outcome", "key": r.get("key"), "decision_id": "outcome|" + did,
                        "entry_decision_id": did, "outcome": r["outcome"]})
            outcome_keys.add(r.get("key"))
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        lead = ""
        if path.exists() and path.stat().st_size:
            with open(path, "rb") as fh:
                fh.seek(-1, 2)
                lead = "" if fh.read(1) == b"\n" else "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(lead + "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in new))
    return len(new)


def counterfactual_rows(skips, replay_ret) -> list:
    """For decisions not taken: the replay's 'if we had entered' outcome (source = replay)."""
    out = []
    for r in skips:
        v = replay_ret(r)
        if v is None:
            continue
        out.append({**r, "source": "replay", "decision_id": "replay|" + r["decision_id"],
                    "outcome": {"ret_pct": float(v), "exit_reason": "replay"}})
    return out


def live_vs_replay(live, replay_ret) -> dict:
    """diff = live ret - replay ret on the same trade, in pp. Trades the replay cannot price are counted."""
    rows, miss = [], 0
    for t in live:
        rv = replay_ret(t)
        src = None
        if isinstance(rv, tuple):                       # (value, source)
            rv, src = rv
        if rv is None:
            miss += 1
            continue
        rows.append({"key": t.get("key"), "mode": t.get("mode"), "live_pct": t["ret_pct"], "replay_pct": float(rv),
                     "diff_pct": float(t["ret_pct"]) - float(rv), "replay_source": src})
    n = len(rows)
    return {"n": n, "no_replay": miss, "rows": rows,
            "mean_diff_pct": (sum(r["diff_pct"] for r in rows) / n) if n else None}


# ------------------------------------------------------------------ F1 nightly (spec "F1 nightly -- контракт")
DECISIONS_DIR = TC.RUNTIME / "audit" / "decisions"
MICRO_DIR = TC.RUNTIME / "microstructure"
SHADOW_BOOK = "x4_floor8"
MICRO_MAX_AGE_SEC = 300
BOT_FEATURES = ("ml_proba", "candidate_score", "ranker_final_score", "mode", "signal_mode", "trail_k")


def _num(v):
    return float(v) if isinstance(v, (int, float)) else None


def micro_features(snap: dict, sym: str):
    """Features of one microstructure snapshot for one pair; None when the pair is not in the live book."""
    hosts = snap.get("hosts") or {}
    live = ((hosts.get("live") or {}).get("book") or {}).get(sym)
    if not live:
        return None
    demo = ((hosts.get("demo") or {}).get("book") or {}).get(sym)
    dep = ((snap.get("depth") or {}).get("live") or {}).get(sym) or {}
    vol = ((snap.get("vol24h") or {}).get("live") or {}).get(sym)
    return {"spread_bps_live": _num(live[1]) if len(live) > 1 else None,
            "spread_bps_demo": _num(demo[1]) if demo and len(demo) > 1 else None,
            "divergence_bps": _num((snap.get("divergence_bps") or {}).get(sym)),
            "depth_buy_1000_bps": _num((dep.get("buy") or {}).get("1000")),
            "depth_sell_1000_bps": _num((dep.get("sell") or {}).get("1000")),
            "qvol24h_live": _num(vol[0]) if vol else None}


def _lines(paths):
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    yield r


def load_micro(paths, syms) -> dict:
    out = {}
    for snap in _lines(paths):
        ts = snap.get("ts")
        if not ts:
            continue
        for s in syms:
            f = micro_features(snap, s)
            if f is not None:
                out.setdefault(s, []).append((_dt(ts), f))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def shadow_index(decisions, book: str = SHADOW_BOOK) -> dict:
    """{(sym, tf, event_ts to the second): {features, ret_pct, exit_reason}} of the shadow book the executor
    mirrors -- its outcome is the replay of the same entry."""
    idx = {}
    for r in decisions:
        if r.get("book") != book or not r.get("event_ts"):
            continue
        k = (r.get("sym"), r.get("tf"), str(r["event_ts"])[:19])
        if r.get("kind") == "entry_decision" and r.get("action") == "enter":
            c = r.get("candidate") or {}
            idx.setdefault(k, {"ret_pct": None, "exit_reason": None})["features"] = {f: c.get(f) for f in BOT_FEATURES}
        elif r.get("kind") == "exit_decision":
            e = idx.setdefault(k, {"features": {}})
            e["ret_pct"], e["exit_reason"] = r.get("ret_pct"), r.get("reason")
    return idx


def _skey(key: str):
    sym, tf, ts = (str(key).split("|") + [None, None, None])[:3]
    return sym, tf, str(ts)[:19]


def nightly(decisions, snaps: dict) -> dict:
    decisions = list(decisions)
    sh = shadow_index(decisions)
    rows = rows_from_decisions(decisions)
    for r in rows:
        r["micro"] = snapshot_at(snaps, r.get("sym"), r["ts"], MICRO_MAX_AGE_SEC) if r.get("ts") else None
        r["bot"] = (sh.get(_skey(r.get("key"))) or {}).get("features")

    # the executor's OWN paper result of the same trade (exit row, or the 'already_closed' skip after an exchange
    # stop) is the per-trade replay; the E7 shadow book is only the fallback -- its slots can differ (bug 10-21)
    own = {}
    for r in decisions:
        if (r.get("book") == BOOK and r.get("kind") == "real_exit" and isinstance(r.get("paper_ret_pct"), (int, float))
                and (r.get("action") == "exit" or r.get("reason") == "already_closed")):
            own[r.get("key")] = float(r["paper_ret_pct"])

    def replay(row):
        v = (sh.get(_skey(row.get("key"))) or {}).get("ret_pct")
        return float(v) if isinstance(v, (int, float)) else None

    def replay_live(row):
        if row.get("key") in own:
            return own[row["key"]], "own_paper"
        v = replay(row)
        return (v, "shadow") if v is not None else None

    cf = counterfactual_rows([r for r in rows if r["action"] == "skip"], replay)
    live = [{"key": r["key"], "ret_pct": r["outcome"]["ret_pct"], "mode": r.get("mode")} for r in rows
            if r["action"] == "enter" and r.get("outcome") and isinstance(r["outcome"].get("ret_pct"), (int, float))]
    lv = live_vs_replay(live, replay_live)
    lv["by_mode"] = {m: {k: v for k, v in live_vs_replay([t for t in live if t["mode"] == m], replay_live).items() if k != "rows"}
                     for m in sorted({t["mode"] for t in live if t["mode"]})}
    return {"rows": rows + cf, "live_vs_replay": lv}


def _day_files(d: Path, days: int, now):
    names = {(now.date() - timedelta(days=i)).isoformat() + ".jsonl" for i in range(days)}
    return sorted(p for p in Path(d).glob("*.jsonl") if p.name in names)


def build(days: int = 3, now=None, decisions_dir=None, micro_dir=None, out_dir=None) -> dict:
    now = _dt(now) if now is not None else datetime.now(timezone.utc)
    decisions_dir = Path(decisions_dir or DECISIONS_DIR)
    micro_dir = Path(micro_dir or MICRO_DIR)
    out_dir = Path(out_dir or DATASET.parent)
    files = _day_files(decisions_dir, days, now)
    dec = list(_lines(files))
    syms = {r.get("sym") for r in dec if r.get("book") == BOOK and r.get("sym")}
    snaps = load_micro(_day_files(micro_dir, days, now), syms)
    res = nightly(dec, snaps)
    appended = append_new(out_dir / "trades.jsonl", res["rows"])
    # live vs replay over every journal day, one trade per key (spec: recomputed each night over all matched trades)
    all_dec = list(_lines(sorted(decisions_dir.glob("*.jsonl"))))
    lv_all = nightly(all_dec, {})["live_vs_replay"]
    seen, rows = set(), []
    for x in lv_all["rows"]:
        if x["key"] not in seen:
            seen.add(x["key"])
            rows.append(x)
    n = len(rows)
    by_mode = {}
    for m in sorted({x.get("mode") for x in rows if x.get("mode")}):
        v = [x["diff_pct"] for x in rows if x.get("mode") == m]
        by_mode[m] = {"n": len(v), "mean_diff_pct": sum(v) / len(v)}
    lv = {"n": n, "no_replay": lv_all["no_replay"], "mean_diff_pct": (sum(x["diff_pct"] for x in rows) / n) if n else None,
          "by_mode": by_mode, "rows": rows, "utc": now.isoformat()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_vs_replay.json").write_text(json.dumps(lv, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    summary = {"utc": now.isoformat(), "days": days, "files": [p.name for p in files], "rows": len(res["rows"]),
               "appended": appended, "with_micro": sum(1 for r in res["rows"] if r.get("micro")),
               "live_vs_replay": {k: lv[k] for k in ("n", "no_replay", "mean_diff_pct", "by_mode")}}
    (out_dir / "BUILD_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    return summary


def main(argv=None) -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--days", type=int, default=3)
    a = ap.parse_args(argv)
    print(json.dumps(build(days=a.days), ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
