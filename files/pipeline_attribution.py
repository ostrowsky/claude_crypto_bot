"""Post-apply attribution — what did this change ACTUALLY do?

For each approved decision in `decisions.jsonl` that has a pinned baseline
(see pipeline_baseline.py) and is at least `--days-after` old, compute:

  raw_delta         = median(post_window) - median(pre_window)
  market_drift      = median(ref_post_window) - median(ref_pre_window)
                      (uses a wider 30d reference window on BOTH sides; this
                      isolates "what happens to the metric on average in
                      this regime" so we can subtract it out)
  normalised_delta  = raw_delta - market_drift

  bootstrap CI95    = 1000 resamples of (pre_vals, post_vals); difference
                      of medians, percentile interval. The CI tells us
                      whether the move is significant given the daily
                      variance, not just whether the point estimate is
                      positive.

Verdict per decision:
  hit         — for every expected metric: normalised_delta within expected
                range AND CI excludes zero on the right side (lower > 0)
  miss        — at least one expected metric does not improve significantly
  regression  — any SENSITIVE metric dropped >= 2pp with CI excluding zero
                on the wrong side (upper < 0)
  needs_data  — insufficient post-window samples (< 4 days)

Pipeline meta-metric (over all evaluated decisions):
  hit_rate    = hits / (hits + misses + regressions)
  If hit_rate < 0.60 over a quarter, the pipeline is no better than coin
  flip at picking improvements and needs review.

This is the only honest answer to "is the pipeline making the bot better?".
The "naive 7d before vs 7d after" comparison in L7 monitor cannot answer it
because it neither cancels market drift nor reports uncertainty.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import pipeline_lib as PL
import pipeline_baseline as PB

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ATTRIBUTION_DIR = PL.PIPELINE / "attribution"
ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
ATTRIBUTION_LOG = ATTRIBUTION_DIR / "attribution.jsonl"

DEFAULT_DAYS_AFTER = 14
MIN_POST_SAMPLES   = 4
BOOTSTRAP_N        = 1000
SENSITIVE_METRICS  = [
    "watchlist_top_early_capture_pct",
    "watchlist_top_bought_pct",
    "recall_at_20",
]
SENSITIVE_REGRESSION_THRESHOLD = 0.02   # 2pp drop


# ---------------------------------------------------------------------------
# Stats helpers (no scipy — keep dependency-free)
# ---------------------------------------------------------------------------


def _percentile(vals: list[float], p: float) -> float:
    """p in [0, 100]. Linear interpolation, matches numpy.percentile default."""
    if not vals:
        return float("nan")
    vs = sorted(vals)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def _bootstrap_delta_ci(
    pre: list[float], post: list[float], *, n: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for (median(post) - median(pre)). Resamples with replacement.

    We resample independently on each side because we want the sampling
    distribution of the difference of medians, not paired pre/post (the
    health-report indices on either side don't correspond to the same days)."""
    rng = random.Random(seed)
    n_pre, n_post = len(pre), len(post)
    diffs: list[float] = []
    for _ in range(n):
        a = [pre[rng.randrange(n_pre)]   for _ in range(n_pre)]
        b = [post[rng.randrange(n_post)] for _ in range(n_post)]
        diffs.append(median(b) - median(a))
    return _percentile(diffs, 100 * alpha / 2), _percentile(diffs, 100 * (1 - alpha / 2))


def _sign_test_prop(pre: list[float], post: list[float]) -> float:
    """Fraction of post values above pre median. > 0.5 => post tends higher."""
    if not pre or not post:
        return float("nan")
    pre_med = median(pre)
    return sum(1 for v in post if v > pre_med) / len(post)


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def _parse_expected_range(expected_str: str) -> tuple[float, float] | None:
    """'+0.10..+0.20' → (0.10, 0.20). '+5..+9' → (5, 9). Single value → (v, v)."""
    s = str(expected_str).strip()
    m = re.match(r"([+\-]?\d+\.?\d*)\s*\.\.\s*([+\-]?\d+\.?\d*)?", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else float("inf")
        return min(lo, hi), max(lo, hi)
    try:
        v = float(s)
        return v, v
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core attribution
# ---------------------------------------------------------------------------


def attribute(
    decision: dict,
    days_after: int = DEFAULT_DAYS_AFTER,
    ref_window_days: int = PB.DEFAULT_REF_DAYS,
) -> dict:
    decision_id = decision["decision_id"]
    ts = _parse_iso(decision.get("ts", ""))
    if ts is None:
        return {"verdict": "skip", "reason": "no parseable ts"}

    baseline_path = PB.BASELINES_DIR / f"baseline_pre_{decision_id}.json"
    baseline = PL.read_json(baseline_path)
    if not baseline:
        return {"verdict": "no_baseline",
                "reason": f"no baseline_pre_{decision_id}.json — run pipeline_baseline.py --backfill-approved"}

    age = (datetime.now(timezone.utc) - ts).days
    if age < days_after:
        return {"verdict": "needs_data",
                "reason": f"only {age}d after apply, need {days_after}"}

    # Post windows
    post_rows = PB.collect_window(ts, ts + timedelta(days=days_after))
    ref_post  = PB.collect_window(ts, ts + timedelta(days=ref_window_days))

    if len(post_rows) < MIN_POST_SAMPLES:
        return {"verdict": "needs_data",
                "reason": f"only {len(post_rows)} post-apply health reports, need >= {MIN_POST_SAMPLES}"}

    pre_rows = baseline.get("pre_rows", [])
    per_metric: dict[str, dict] = {}
    for k in PB.ALL_METRIC_KEYS:
        pre_vals  = [r["metrics"].get(k) for r in pre_rows  if r["metrics"].get(k) is not None]
        post_vals = [r["metrics"].get(k) for r in post_rows if r["metrics"].get(k) is not None]
        if len(pre_vals) < 2 or len(post_vals) < 2:
            per_metric[k] = {"status": "insufficient_data",
                             "n_pre": len(pre_vals), "n_post": len(post_vals)}
            continue
        raw_delta = median(post_vals) - median(pre_vals)

        # Market drift via 30d reference window
        ref_pre_summary = (baseline.get("ref_summary") or {}).get(k) or {}
        ref_pre_med = ref_pre_summary.get("median")
        ref_post_vals = [r["metrics"].get(k) for r in ref_post if r["metrics"].get(k) is not None]
        if ref_pre_med is not None and ref_post_vals:
            ref_post_med = median(ref_post_vals)
            market_drift = ref_post_med - ref_pre_med
        else:
            ref_post_med = None
            market_drift = 0.0

        normalised = raw_delta - market_drift
        ci_lo, ci_hi = _bootstrap_delta_ci(pre_vals, post_vals)
        ci_excludes_zero = ci_lo > 0 or ci_hi < 0
        sign_prop = _sign_test_prop(pre_vals, post_vals)

        per_metric[k] = {
            "pre_median":        round(median(pre_vals), 6),
            "post_median":       round(median(post_vals), 6),
            "raw_delta":         round(raw_delta, 6),
            "ref_pre_median":    round(ref_pre_med, 6) if ref_pre_med is not None else None,
            "ref_post_median":   round(ref_post_med, 6) if ref_post_med is not None else None,
            "market_drift":      round(market_drift, 6),
            "normalised_delta":  round(normalised, 6),
            "ci95_low":          round(ci_lo, 6),
            "ci95_high":         round(ci_hi, 6),
            "ci_excludes_zero":  ci_excludes_zero,
            "sign_test_prop":    round(sign_prop, 4),
            "n_pre":             len(pre_vals),
            "n_post":            len(post_vals),
        }

    # ---- Verdict logic --------------------------------------------------
    expected = decision.get("expected") or {}
    rationale: list[str] = []
    expected_hits = []
    expected_misses = []

    for em, em_str in expected.items():
        pm = per_metric.get(em)
        if not pm or pm.get("status") == "insufficient_data":
            expected_misses.append(em)
            rationale.append(f"{em}: insufficient_data")
            continue
        rng = _parse_expected_range(em_str)
        if rng is None:
            rationale.append(f"{em}: cannot parse expected_range={em_str!r}")
            expected_misses.append(em)
            continue
        lo_e, hi_e = rng
        actual = pm["normalised_delta"]
        ci_low = pm["ci95_low"]
        # Hit = direction matches expected range AND CI excludes zero in the right direction
        if lo_e >= 0:
            hit = (actual >= lo_e - 0.005) and (ci_low > 0)
        else:
            # expected negative delta (e.g. FPR reduction)
            hit = (actual <= hi_e + 0.005) and (pm["ci95_high"] < 0)
        if hit:
            expected_hits.append(em)
            rationale.append(
                f"{em}: normalised={actual:+.4f} in expected [{lo_e:+.4f}..{hi_e:+.4f}], "
                f"CI95=[{ci_low:+.4f}..{pm['ci95_high']:+.4f}] → HIT"
            )
        else:
            expected_misses.append(em)
            rationale.append(
                f"{em}: normalised={actual:+.4f} vs expected [{lo_e:+.4f}..{hi_e:+.4f}], "
                f"CI95=[{ci_low:+.4f}..{pm['ci95_high']:+.4f}] → MISS"
            )

    # Sensitive-metric regression check (orthogonal to expected hits)
    regressions = []
    for sk in SENSITIVE_METRICS:
        pm = per_metric.get(sk)
        if not pm or pm.get("status") == "insufficient_data":
            continue
        if (pm["normalised_delta"] < -SENSITIVE_REGRESSION_THRESHOLD
            and pm["ci95_high"] < 0):
            regressions.append({"metric": sk,
                                "normalised_delta": pm["normalised_delta"],
                                "ci95_high": pm["ci95_high"]})
            rationale.append(
                f"REGRESSION in sensitive metric {sk}: normalised={pm['normalised_delta']:+.4f}, "
                f"CI95_high={pm['ci95_high']:+.4f}"
            )

    # Final verdict
    if regressions:
        verdict_str = "regression"
    elif not expected:
        verdict_str = "skip"
        rationale.append("decision has no expected_delta — cannot evaluate")
    elif expected_misses and not expected_hits:
        verdict_str = "miss"
    elif expected_misses and expected_hits:
        verdict_str = "partial"
    else:
        verdict_str = "hit"

    return {
        "decision_id":         decision_id,
        "evaluated_at":        PL.utc_now_iso(),
        "days_after":          days_after,
        "ref_window_days":     ref_window_days,
        "bootstrap_n":         BOOTSTRAP_N,
        "verdict":             verdict_str,
        "expected_metrics":    list(expected.keys()),
        "expected_hits":       expected_hits,
        "expected_misses":     expected_misses,
        "regressions":         regressions,
        "rationale":           rationale,
        "per_metric":          per_metric,
    }


# ---------------------------------------------------------------------------
# Pipeline-level aggregation
# ---------------------------------------------------------------------------


def aggregate_pipeline(results: list[dict]) -> dict:
    """Compute the meta-metric hit_rate across decisions evaluated this run."""
    by_verdict: dict[str, int] = {}
    for r in results:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    hits = by_verdict.get("hit", 0) + by_verdict.get("partial", 0) * 0.5
    misses = by_verdict.get("miss", 0)
    regressions = by_verdict.get("regression", 0)
    total = hits + misses + regressions
    hit_rate = round(hits / total, 4) if total > 0 else None
    return {
        "n_evaluated":     len(results),
        "by_verdict":      by_verdict,
        "weighted_hits":   hits,
        "misses":          misses,
        "regressions":     regressions,
        "hit_rate":        hit_rate,
        "interpretation":  "hit_rate < 0.60 over a quarter → pipeline is no better than coin flip",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--decision", help="attribute one specific decision_id")
    g.add_argument("--all-due", action="store_true",
                   help="attribute every approved decision older than --days-after")
    ap.add_argument("--days-after", type=int, default=DEFAULT_DAYS_AFTER)
    ap.add_argument("--ref-window-days", type=int, default=PB.DEFAULT_REF_DAYS)
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    if not (args.decision or args.all_due):
        args.all_due = True

    targets: list[dict] = []
    if args.decision:
        for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
            if rec.get("decision_id") == args.decision:
                targets.append(rec)
                break
        if not targets:
            print(f"ERROR: decision {args.decision} not in log")
            sys.exit(1)
    else:
        for rec in PL.iter_jsonl(PL.DECISIONS_LOG):
            if rec.get("stage") == "approved":
                targets.append(rec)

    results: list[dict] = []
    for d in targets:
        r = attribute(d, args.days_after, args.ref_window_days)
        results.append(r)
        PL.append_jsonl(ATTRIBUTION_LOG, r)

    meta = aggregate_pipeline(results)
    report = {
        "report_id":     f"attribution-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at":  PL.utc_now_iso(),
        "days_after":    args.days_after,
        "ref_window_days": args.ref_window_days,
        "n_targets":     len(targets),
        "results":       results,
        "pipeline_meta": meta,
    }
    out = ATTRIBUTION_DIR / f"{report['report_id']}.json"
    PL.write_json(out, report)
    print(f"[attribution] wrote {out}")
    print(f"[attribution] verdicts: {meta['by_verdict']}  hit_rate={meta['hit_rate']}")

    if args.do_print:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
