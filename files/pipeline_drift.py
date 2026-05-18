"""Distribution-drift detection for ranker_proba and other key features.

Purpose
-------
The bot's ML model is retrained nightly. **Retraining doesn't detect when
the underlying market regime has shifted enough that the old model is
fundamentally wrong on the new regime.** Recall@20 will degrade with a lag
of days — by which time the bot has already missed dozens of signals.

This module compares the recent distribution of a key feature to a longer
historical reference. If they diverge significantly (Kolmogorov-Smirnov
two-sample test), we flag drift in the daily health report and (optionally)
trigger a force-retrain.

Why KS over alternatives
------------------------
- ADWIN / Page-Hinkley are streaming detectors with state; KS is stateless
  and dead-simple to test, audit, reason about
- KS works on the empirical CDF — no distributional assumption, which is
  appropriate for `ranker_proba` (bimodal-ish) and `candidate_score`
  (heavy-tailed)
- O(n log n) on a single daily run — fits in well under a second

Why two-sample, not one-sample
------------------------------
We don't know the "true" distribution — we know "what it was 30 days ago".
The hypothesis we test is "the recent window is drawn from the same
distribution as the reference window". That's exactly the two-sample
formulation.

The output is consumed by:
  - `bot_health_report.py` — emits a red flag when drift is significant
  - `pipeline_run.py` — daily orchestrator step calls this module

See also: roadmap RM-14 in `docs/specs/features/auto-improvement-loop-spec.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import pipeline_lib as PL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CRITIC_DATASET = PL.FILES_DIR / "critic_dataset.jsonl"
DRIFT_DIR = PL.PIPELINE / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)
DRIFT_LOG = DRIFT_DIR / "drift_checks.jsonl"

# Default settings — tunable from config later if needed.
DEFAULT_RECENT_DAYS    = 7
DEFAULT_REFERENCE_DAYS = 30
DEFAULT_MIN_SAMPLES    = 100
# KS critical p-value below which we declare drift. 0.01 is conservative —
# we'd rather miss a borderline shift than retrain unnecessarily on noise.
DEFAULT_P_THRESHOLD    = 0.01
# Even if p < threshold, ignore if effect size (KS statistic D) is tiny.
# D < 0.05 means CDFs are within 5pp everywhere — practically identical.
DEFAULT_D_THRESHOLD    = 0.10

# Features we track. These are the inputs that drive the bot's downstream
# decisions; if they shift, the model's policy is implicitly stale even if
# AUC on the training set looks fine.
TRACKED_FEATURES = [
    # path-into-record  ,  human label
    (("decision", "ranker_proba"),         "ranker_proba"),
    (("decision", "candidate_score"),      "candidate_score"),
    (("decision", "ml_proba"),             "ml_proba"),
    (("decision", "forecast_return_pct"),  "forecast_return_pct"),
]


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov two-sample test (stdlib-only)
# ---------------------------------------------------------------------------


def ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (D, p_value) for the two-sample KS test.

    No scipy dependency: D = max|F_a(x) - F_b(x)| computed by merge-walk over
    sorted samples; p approximated via Kolmogorov distribution asymptote
    (Smirnov 1948). Accurate enough for our daily use (n >= 100 each side).
    """
    if not a or not b:
        return 0.0, 1.0
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    # Walk over the sorted UNION of values, advancing whichever side has the
    # smaller next value (both, when equal). Sample two-sample KS = max
    # over all x of |F_a(x) - F_b(x)|; since ECDF jumps only at observed
    # values, scanning the merged sequence suffices.
    i = j = 0
    d_max = 0.0
    while i < na or j < nb:
        if i < na and (j >= nb or sa[i] < sb[j]):
            i += 1
        elif j < nb and (i >= na or sb[j] < sa[i]):
            j += 1
        else:
            # Tie: advance both so identical samples → D == 0
            i += 1
            j += 1
        d = abs(i / na - j / nb)
        if d > d_max:
            d_max = d
    # Edge case: identical empirical CDFs → no evidence of difference
    if d_max == 0.0:
        return 0.0, 1.0
    # Asymptotic p-value via Kolmogorov distribution: 2 Σ (-1)^(k-1) e^(-2 k^2 lambda^2)
    n_eff = math.sqrt(na * nb / (na + nb))
    lam = (n_eff + 0.12 + 0.11 / n_eff) * d_max
    # Truncate series at 100 terms — converges very fast
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * (k * lam) ** 2)
                  for k in range(1, 101))
    p = max(0.0, min(1.0, p))
    return d_max, p


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _parse_ts(ev: dict) -> datetime | None:
    """critic_dataset rows have bar_ts / ts_signal as either ISO strings or
    millisecond UNIX timestamps. Handle both, mirroring pipeline_shadow."""
    for k in ("bar_ts", "ts_signal"):
        v = ev.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            try:
                return datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            continue
    return None


def _get_value(ev: dict, path: tuple[str, ...]) -> float | None:
    cur: object = ev
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
        if cur is None:
            return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def collect_window(path: Path, feature_path: tuple[str, ...],
                   start: datetime, end: datetime) -> list[float]:
    """Stream critic_dataset, extract feature values whose ts falls in window.
    Returns a list (may be empty). Defensive: skips malformed rows silently."""
    out: list[float] = []
    if not path.exists():
        return out
    for ev in PL.iter_jsonl(path):
        ts = _parse_ts(ev)
        if ts is None or not (start <= ts <= end):
            continue
        v = _get_value(ev, feature_path)
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_feature(
    feature_path: tuple[str, ...],
    label: str,
    *,
    recent_days:    int   = DEFAULT_RECENT_DAYS,
    reference_days: int   = DEFAULT_REFERENCE_DAYS,
    min_samples:    int   = DEFAULT_MIN_SAMPLES,
    p_threshold:    float = DEFAULT_P_THRESHOLD,
    d_threshold:    float = DEFAULT_D_THRESHOLD,
    now:            datetime | None = None,
    dataset_path:   Path | None = None,
) -> dict:
    """Run a single drift check on `feature_path`. Returns a result dict.

    Status:
      drift_detected   — p<threshold AND D>=d_threshold (act on this)
      no_drift         — distribution similar enough
      insufficient_data — < min_samples on either side; can't conclude
    """
    ds = dataset_path if dataset_path is not None else CRITIC_DATASET
    now = now if now is not None else datetime.now(timezone.utc)

    recent_start    = now - timedelta(days=recent_days)
    reference_start = now - timedelta(days=reference_days)
    reference_end   = recent_start    # contiguous, no overlap

    recent    = collect_window(ds, feature_path, recent_start, now)
    reference = collect_window(ds, feature_path, reference_start, reference_end)

    if len(recent) < min_samples or len(reference) < min_samples:
        return {
            "feature":          label,
            "status":           "insufficient_data",
            "n_recent":         len(recent),
            "n_reference":      len(reference),
            "min_samples":      min_samples,
            "p_threshold":      p_threshold,
            "d_threshold":      d_threshold,
        }

    d, p = ks_two_sample(recent, reference)
    drift = (p < p_threshold) and (d >= d_threshold)

    return {
        "feature":          label,
        "status":           "drift_detected" if drift else "no_drift",
        "ks_statistic":     round(d, 4),
        "p_value":          round(p, 6),
        "p_threshold":      p_threshold,
        "d_threshold":      d_threshold,
        "median_recent":    round(median(recent), 4),
        "median_reference": round(median(reference), 4),
        "n_recent":         len(recent),
        "n_reference":      len(reference),
        "recent_window":    {"start": recent_start.isoformat(),    "end": now.isoformat()},
        "reference_window": {"start": reference_start.isoformat(), "end": reference_end.isoformat()},
    }


def check_all(features: list[tuple[tuple[str, ...], str]] | None = None,
              **kwargs) -> dict:
    """Run all configured drift checks, return a structured report."""
    feats = features if features is not None else TRACKED_FEATURES
    results = [check_feature(path, label, **kwargs) for path, label in feats]
    drifted = [r for r in results if r.get("status") == "drift_detected"]
    insufficient = [r for r in results if r.get("status") == "insufficient_data"]
    return {
        "report_id":    f"drift-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "generated_at": PL.utc_now_iso(),
        "n_features":   len(results),
        "n_drift":      len(drifted),
        "n_insufficient": len(insufficient),
        "verdict":      "drift_detected" if drifted else (
                         "insufficient_data" if insufficient and len(insufficient) == len(results)
                         else "no_drift"),
        "drifted_features": [r["feature"] for r in drifted],
        "results":      results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-days",    type=int, default=DEFAULT_RECENT_DAYS)
    ap.add_argument("--reference-days", type=int, default=DEFAULT_REFERENCE_DAYS)
    ap.add_argument("--min-samples",    type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--p-threshold",    type=float, default=DEFAULT_P_THRESHOLD)
    ap.add_argument("--d-threshold",    type=float, default=DEFAULT_D_THRESHOLD)
    ap.add_argument("--print",          dest="do_print", action="store_true")
    args = ap.parse_args()

    report = check_all(
        recent_days=args.recent_days,
        reference_days=args.reference_days,
        min_samples=args.min_samples,
        p_threshold=args.p_threshold,
        d_threshold=args.d_threshold,
    )
    out = DRIFT_DIR / f"{report['report_id']}.json"
    PL.write_json(out, report)
    PL.append_jsonl(DRIFT_LOG, report)
    print(f"[drift] wrote {out}")
    print(f"[drift] verdict={report['verdict']} "
          f"drifted={report['n_drift']}/{report['n_features']}")
    if args.do_print:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
