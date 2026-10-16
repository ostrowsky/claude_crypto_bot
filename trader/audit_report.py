"""Summary of the trader audit journals (trader-audit-log-1012-spec.md §6).

  exchange   per endpoint: requests, error share, error codes, latency p50/p95, last used weight
  decisions  per book: enters, skips by reason, exits by reason with mean ret %

  (from trader/) ../pyembed/python.exe audit_report.py [--days 7]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit as A                 # noqa: E402


def _q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def summarize(exchange_rows: list, decision_rows: list) -> dict:
    ex = collections.defaultdict(list)
    for r in exchange_rows:
        ex[r.get("path")].append(r)
    exs = {}
    for path, rs in sorted(ex.items()):
        lat = [r.get("latency_ms") for r in rs]
        errs = [r for r in rs if not r.get("ok")]
        exs[path] = {"n": len(rs), "error_share": round(len(errs) / len(rs), 4),
                     "error_codes": dict(collections.Counter(str(r.get("error_code")) for r in errs)),
                     "p50_ms": _q(lat, 0.5), "p95_ms": _q(lat, 0.95),
                     "last_used_weight_1m": next((r.get("used_weight_1m") for r in reversed(rs) if r.get("used_weight_1m")), None)}
    dec = collections.defaultdict(lambda: {"enter": 0, "skip": 0, "skip_reasons": collections.Counter(),
                                           "exits": collections.defaultdict(list), "orphan_exits": 0})
    for r in decision_rows:
        b = dec[r.get("book")]
        if r.get("kind") == "entry_decision":
            if r.get("action") == "enter":
                b["enter"] += 1
            else:
                b["skip"] += 1
                b["skip_reasons"][r.get("reason")] += 1
        elif r.get("kind") == "exit_decision":
            b["exits"][r.get("reason")].append(r.get("ret_pct"))
        elif r.get("kind") == "orphan_exit":
            b["orphan_exits"] += 1
    decs = {}
    for book, b in dec.items():
        decs[book] = {"enter": b["enter"], "skip": b["skip"], "skip_reasons": dict(b["skip_reasons"]),
                      "orphan_exits": b["orphan_exits"],
                      "exits": {k: {"n": len(v), "mean_ret_pct": round(sum(x for x in v if x is not None) / len(v), 4) if v else None}
                                for k, v in b["exits"].items()}}
    return {"exchange": exs, "decisions": decs}


def load(kind: str, days: int, root: Path | None = None) -> list:
    root = (root or A.ROOT) / kind
    cut = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    out = []
    for f in sorted(root.glob("*.jsonl")):
        if f.stem < cut:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args(argv)
    s = summarize(load("exchange", a.days), load("decisions", a.days))
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
