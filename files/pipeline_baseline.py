"""Pin a baseline snapshot of all tracked metrics BEFORE applying a change.

A baseline is the only honest reference for post-apply attribution. Without
it, the "before vs after" window in L7 monitor is naive — it samples 7 days
either side of `decision.ts` but cannot tell apart:
  - effect of the change
  - market drift in the same period
  - effect of any other concurrent change

This module solves the first half: it captures, at decision-pin time, two
windows of daily health metrics:
  - pre_window  (default 14d)  — what the system looked like BEFORE
  - ref_window  (default 30d)  — wider context used by attribution to
                                 cancel out market drift

`pipeline_attribution.py` then reads this file later, gathers post-apply
windows of the same shape, computes raw_delta, market_drift, normalised_delta,
and a bootstrap CI95.

Tracked metrics deliberately span THREE domains:
  - deployment (does the bot do its job?)
  - training   (does the model still learn well?)
  - gap        (does training quality translate to deployment quality?)

If a config change improves training metrics but widens the gap, that is
worse than not changing anything — only the cross-domain view catches that.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINES_DIR = PL.PIPELINE / "baselines"
BASELINES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PRE_DAYS = 14
DEFAULT_REF_DAYS = 30

TRACKED_METRICS: dict[str, list[str]] = {
    "deployment": [
        "watchlist_top_early_capture_pct",   # north star
        "watchlist_top_bought_pct",
        "false_positive_rate",
    ],
    "training": [
        "recall_at_20",
        "ucb_separation",
        "auc",
    ],
    "gap": [
        "training_to_live_gap",
    ],
}

ALL_METRIC_KEYS = (
    TRACKED_METRICS["deployment"]
    + TRACKED_METRICS["training"]
    + TRACKED_METRICS["gap"]
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _parse_target_date(report: dict) -> datetime | None:
    raw = report.get("target_date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract(report: dict, key: str) -> float | None:
    dh = report.get("deployment_health") or {}
    th = report.get("training_health") or {}
    if key in dh:
        v = dh.get(key)
        return float(v) if isinstance(v, (int, float)) else None
    if key in th:
        v = th.get(key)
        return float(v) if isinstance(v, (int, float)) else None
    if key == "training_to_live_gap":
        v = (report.get("training_to_live_gap") or {}).get("value")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def collect_window(start: datetime, end: datetime) -> list[dict]:
    """Return [{date, metrics{key: value or None}}] for each health-*.json
    whose target_date falls in [start, end]. Order is ascending by date."""
    out: list[dict] = []
    if not PL.HEALTH.exists():
        return out
    for p in sorted(PL.HEALTH.glob("health-*.json")):
        h = PL.read_json(p)
        if not h:
            continue
        d = _parse_target_date(h)
        if d is None or not (start <= d <= end):
            continue
        metrics = {k: _extract(h, k) for k in ALL_METRIC_KEYS}
        out.append({"date": h.get("target_date"), "metrics": metrics})
    return out


def _summarize(rows: list[dict], key: str) -> dict | None:
    vals = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
    if not vals:
        return None
    return {
        "n":      len(vals),
        "median": round(median(vals), 6),
        "min":    round(min(vals), 6),
        "max":    round(max(vals), 6),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pin_baseline(
    decision_id: str,
    decision_ts: datetime | str,
    pre_window_days: int = DEFAULT_PRE_DAYS,
    ref_window_days: int = DEFAULT_REF_DAYS,
) -> Path:
    """Write baseline snapshot to baselines/baseline_pre_<id>.json.

    Returns the path written. Always succeeds — if no health reports exist
    yet, summaries will be None and attribution will later report
    insufficient_data. That's the right behaviour: never silently fabricate
    a baseline."""
    if isinstance(decision_ts, str):
        ts = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
    else:
        ts = decision_ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    pre_rows = collect_window(ts - timedelta(days=pre_window_days), ts)
    ref_rows = collect_window(ts - timedelta(days=ref_window_days), ts)

    pre_summary = {k: _summarize(pre_rows, k) for k in ALL_METRIC_KEYS}
    ref_summary = {k: _summarize(ref_rows, k) for k in ALL_METRIC_KEYS}

    record = {
        "schema_version":  1,
        "decision_id":     decision_id,
        "pinned_at":       PL.utc_now_iso(),
        "decision_ts":     ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pre_window_days": pre_window_days,
        "ref_window_days": ref_window_days,
        "tracked_metrics": TRACKED_METRICS,
        "pre_summary":     pre_summary,
        "ref_summary":     ref_summary,
        "pre_rows":        pre_rows,   # daily granularity for bootstrap later
        "ref_rows":        ref_rows,
    }

    out = BASELINES_DIR / f"baseline_pre_{decision_id}.json"
    PL.write_json(out, record)
    return out


# ---------------------------------------------------------------------------
# CLI (manual baseline pinning, e.g. backfill)
# ---------------------------------------------------------------------------


def _load_decision(decision_id: str) -> dict | None:
    for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
        if rec.get("decision_id") == decision_id:
            return rec
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", help="decision_id to pin (uses decisions.jsonl ts)")
    ap.add_argument("--ts", help="ISO timestamp; required if --decision not in log")
    ap.add_argument("--pre-days", type=int, default=DEFAULT_PRE_DAYS)
    ap.add_argument("--ref-days", type=int, default=DEFAULT_REF_DAYS)
    ap.add_argument("--backfill-approved", action="store_true",
                    help="pin a baseline for every approved decision missing one")
    args = ap.parse_args()

    if args.backfill_approved:
        n = 0
        for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
            if rec.get("stage") != "approved":
                continue
            did = rec["decision_id"]
            if (BASELINES_DIR / f"baseline_pre_{did}.json").exists():
                continue
            out = pin_baseline(did, rec["ts"], args.pre_days, args.ref_days)
            print(f"[baseline] backfilled {did} -> {out.name}")
            n += 1
        print(f"[baseline] backfill done: {n} new")
        return

    if not args.decision:
        ap.error("--decision is required (or use --backfill-approved)")

    rec = _load_decision(args.decision)
    if rec:
        ts = rec["ts"]
    elif args.ts:
        ts = args.ts
    else:
        print(f"ERROR: decision {args.decision} not in log; pass --ts ISO_TIMESTAMP")
        sys.exit(1)

    out = pin_baseline(args.decision, ts, args.pre_days, args.ref_days)
    print(f"[baseline] wrote {out}")


if __name__ == "__main__":
    main()
