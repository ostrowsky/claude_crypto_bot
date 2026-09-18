"""trend_quality blocks on a forecast that does not exist. Does that cost moves?

WHY THIS EXISTS

`_trend_entry_quality_guard_reason` (monitor.py) admits a 15m trend candidate if
`forecast_return_pct >= TREND_15M_QUALITY_FORECAST_MIN` (0.25), otherwise only
through an alternative path (vol_x >= 1.2 AND ADX >= 24 AND slope >= 0.35).

The forecast is built in strategy.py as the best expected forward return over
THIS coin's own rule signals since 00:00 UTC -- and when fewer than
TODAY_MIN_SIGNALS (2) of them are evaluable it is not "unknown", it is 0.0.
So a coin that has just started moving, with no history today, is judged as
if its forecast were bad. It resets for every coin at UTC midnight.

On 2026-09-18 STRK rose +34.9% in 39 hours; trend_quality blocked it 61 times,
64 of its 96 "weak 15m trend" reasons carried forecast exactly 0.000, and the
alternative path failed on ADX -- which lags at the start of a move. Across
the whole log, 78% of 32 826 such blocks since 2026-04-09 had forecast 0.000.

THE QUESTION, ASKED SO IT CAN COME BACK NEGATIVE

Split the gate's forecast-path rejections into
    NO DATA     forecast exactly 0.000 -- the missing-data default
    LOW FCST    forecast computed and below 0.25 -- a real negative opinion
and compare each with 15m candidates that went PAST trend_quality. If NO DATA
moves no worse than what the gate passes, the gate is blocking on absent
evidence (CLAUDE.md 0a rule 5). If it moves worse, 0.000 is accidentally
filtering something real and that is the result to record.

METHOD

  source      bot_events.jsonl, every blocked/entry event on tf=15m.
  NO DATA / LOW FCST   blocked events whose reason is "weak 15m trend",
              split on the printed forecast value.
  PASSED      15m events stopped by a gate AFTER trend_quality in the pipeline
              (trend_chop, mode_range_quality, correlation_guard, clone_guard,
              ...) or entered. Blocked events do not record the entry mode, so
              this set contains non-trend modes the guard never judged -- the
              comparison is to "what the pipeline passes on 15m", not to a
              matched trend-only population. Stated, not hidden.
  outcome     forward PEAK over the next N 15m bars (strictly after the
              decision bar) against the event's own price, from the cached
              history. A truncated window is dropped, never scored short.
  dedup       one row per (symbol, hour, group): a gate that re-fires every
              poll must not weight its own opinion by loop frequency.

Base rate beside every ratio (TH-01), the bot's own candidates (TH-06), by
month so a single regime cannot carry the verdict.

VERDICT 2026-09-18: REFUTED as a lever. Do not re-test without new evidence.

Maximum period 2026-04-01 .. 2026-09-18, 12 594 deduplicated 15m decisions:
NO DATA 2877, LOW FCST 981, PASSED 8736 (NO DATA = 75% of forecast-path rejects).

    forward peak, 16 x 15m (4h)   n     median   mean   >=3%   >=5%   vs PASS
    PASSED trend_quality        8729    0.95%   1.62%  15.1%   5.9%     --
    REJECTED: NO DATA           2877    0.98%   1.53%  14.0%   4.4%   0.94x
    REJECTED: LOW FCST           981    0.96%   1.47%  14.3%   5.3%   0.90x

5 bars: 0.91 / 0.84 (0.92x) / 0.79 (0.86x), >=5%: 1.8 / 1.3 / 1.0.

The design IS wrong in principle -- "no forecast yet today" is scored as "bad
forecast" -- but the population it wrongly scores is not hiding big movers: its
>=5% tail is thinner than what the gate passes (4.4% vs 5.9%). Admitting it
wholesale would dilute the flow. The conflation costs nothing measurable and
slightly helps. September sits at parity (1.74% vs 1.69%), so the regime that
raised the question does not reverse it either.

The midnight reset is visible: 00-05 UTC holds a third of NO DATA rejects and is
where they are weakest (0.72% vs 0.87% at 5 bars).

A DATA DEFECT found by this file, bigger than its own question: TD.bars_15m reads
only history/<sym>_15m_419d.csv, which ends 2026-08-20. The first run lost 4104
of 12 592 rows and ALL of September to it. index_15m() merges that file with the
daily-refreshed rolling history/<sym>_15m.csv; the same gap silently degrades
the peak TRAINING label (ml_signal_model._peak_bars) -- see the spec.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# data lives next to the bot; fall back to the working directory when this file
# is run from outside files/ (analysis copies, scratch runs)
FILES = HERE if (HERE / "bot_events.jsonl").exists() else Path.cwd()
if str(FILES) not in sys.path:
    sys.path.insert(0, str(FILES))

import _backtest_trend_start_detector as TD  # noqa: E402

EVENTS = FILES / "bot_events.jsonl"
FORECAST_RE = re.compile(r"forecast (-?[\d.]+)")

# gates that run AFTER trend_quality -- a candidate stopped here passed it
LATER_GATES = ("trend_1h_chop", "trend_chop", "mode_range_quality",
               "correlation_guard", "clone_guard", "clone_signal_guard",
               "open_cluster_cap", "rotation", "bandit", "late_continuation")

_IDX: dict = {}


def _read_csv(fp):
    import csv
    out = []
    if not fp.exists():
        return out
    with io.open(fp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append((datetime.fromisoformat(r["ts"]), float(r["open"]),
                            float(r["high"]), float(r["low"]), float(r["close"]),
                            float(r.get("volume") or 0)))
            except (KeyError, ValueError):
                continue
    return out


def index_15m(sym):
    """The 419-day backfill UNION the daily-refreshed rolling file.

    TD.bars_15m reads only history/<sym>_15m_419d.csv, a one-off backfill that
    ends 2026-08-20. The daily task writes history/<sym>_15m.csv, a rolling
    30-day window. Reading only the first silently drops every decision after
    08-20 -- which on the first run of this file removed ALL of September, the
    very regime the question came from. Later bars win on overlap.
    """
    if sym not in _IDX:
        merged = {}
        for b in _read_csv(TD.HISTORY / ("%s_15m_419d.csv" % sym)):
            merged[b[0]] = b
        for b in _read_csv(TD.HISTORY / ("%s_15m.csv" % sym)):
            merged[b[0]] = b
        bars = [merged[k] for k in sorted(merged)]
        _IDX[sym] = (bars, {b[0]: i for i, b in enumerate(bars)})
    return _IDX[sym]


def bar_open(dt):
    return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)


def forward_peak(sym, dt, px, horizon):
    """Highest high over the `horizon` 15m bars after the decision bar, as %."""
    bars, idx = index_15m(sym)
    i = idx.get(bar_open(dt))
    if i is None or px <= 0:
        return None
    fut = bars[i + 1: i + 1 + horizon]
    if len(fut) < horizon:
        return None
    return (max(b[2] for b in fut) / px - 1.0) * 100.0


def classify(e):
    """Return NO DATA / LOW FCST / PASSED / None for one event."""
    if str(e.get("tf")) != "15m":
        return None
    ev = e.get("event")
    if ev == "entry":
        return "PASSED"
    if ev != "blocked":
        return None
    reason = str(e.get("reason") or "")
    if "weak 15m trend" in reason:
        m = FORECAST_RE.search(reason)
        if not m:
            return None
        return "NO DATA" if float(m.group(1)) == 0.0 else "LOW FCST"
    gate = str(e.get("reason_code") or e.get("signal_type") or e.get("gate") or "")
    if any(g in gate for g in LATER_GATES):
        return "PASSED"
    return None


def load(since):
    seen = {}
    with io.open(EVENTS, "rb") as fh:
        for raw in fh:
            if b'"15m"' not in raw:
                continue
            if b'"blocked"' not in raw and b'"entry"' not in raw:
                continue
            try:
                e = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            ts = str(e.get("ts") or "")
            if ts < since:
                continue
            grp = classify(e)
            if grp is None:
                continue
            px = e.get("price")
            if not isinstance(px, (int, float)) or px <= 0 or not e.get("sym"):
                continue
            try:
                d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            key = (e["sym"], d.replace(minute=0, second=0, microsecond=0), grp)
            if key not in seen:
                seen[key] = {"sym": e["sym"], "dt": d, "px": float(px), "grp": grp,
                             "hour_utc": d.hour}
    return list(seen.values())


def row(name, v, base_mean=None):
    if not v:
        print("%-24s%8s" % (name, "n=0"))
        return
    s = sorted(v)
    med = s[len(s) // 2]
    p75 = s[int(0.75 * (len(s) - 1))]
    mean = sum(s) / len(s)
    s3 = 100.0 * sum(1 for x in s if x >= 3) / len(s)
    s5 = 100.0 * sum(1 for x in s if x >= 5) / len(s)
    lift = ("%6.2fx" % (mean / base_mean)) if base_mean else "     --"
    print("%-24s%8d%9.2f%%%9.2f%%%9.2f%%%8.1f%%%8.1f%%%9s"
          % (name, len(s), med, p75, mean, s3, s5, lift))


HDR = "%-24s%8s%10s%10s%10s%9s%9s%9s" % (
    "group", "n", "median", "p75", "mean", ">=3%", ">=5%", "vs PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-01")
    ap.add_argument("--horizons", default="5,16",
                    help="forward windows in 15m bars (5 = label horizon, 16 = 4h)")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    rows = load(args.since)
    if not rows:
        print("no rows")
        return
    rows.sort(key=lambda r: r["dt"])
    cnt = collections.Counter(r["grp"] for r in rows)
    print("15m decisions since %s, deduplicated by symbol-hour-group: %d  (%s .. %s)"
          % (args.since, len(rows), rows[0]["dt"].date(), rows[-1]["dt"].date()))
    for g in ("NO DATA", "LOW FCST", "PASSED"):
        print("   %-9s %6d" % (g, cnt.get(g, 0)))
    nd, lf = cnt.get("NO DATA", 0), cnt.get("LOW FCST", 0)
    print("share of the forecast-path rejections that are NO DATA: %.0f%%"
          % (100.0 * nd / max(1, nd + lf)))

    for h in horizons:
        vals = collections.defaultdict(list)
        month = collections.defaultdict(lambda: collections.defaultdict(list))
        hour = collections.defaultdict(lambda: collections.defaultdict(list))
        miss = 0
        for r in rows:
            p = forward_peak(r["sym"], r["dt"], r["px"], h)
            if p is None:
                miss += 1
                continue
            vals[r["grp"]].append(p)
            month[r["dt"].strftime("%Y-%m")][r["grp"]].append(p)
            hour[r["hour_utc"] // 6][r["grp"]].append(p)
        base = vals.get("PASSED") or []
        bmean = sum(base) / len(base) if base else None
        print()
        print("=" * 96)
        print("FORWARD PEAK over %d x 15m bars (%.2gh)   unresolved: %d"
              % (h, h / 4.0, miss))
        print("=" * 96)
        print(HDR)
        print("-" * 96)
        row("PASSED trend_quality", base)
        row("REJECTED: NO DATA", vals.get("NO DATA"), bmean)
        row("REJECTED: LOW FCST", vals.get("LOW FCST"), bmean)
        print("-" * 96)
        print("'vs PASS' = mean peak relative to what the pipeline passes on 15m.")

        print()
        print("by month -- mean peak, n in brackets")
        print("%-10s%22s%22s%22s" % ("month", "PASSED", "NO DATA", "LOW FCST"))
        for m in sorted(month):
            cells = []
            for g in ("PASSED", "NO DATA", "LOW FCST"):
                v = month[m].get(g) or []
                cells.append(("%6.2f%%  [%5d]" % (sum(v) / len(v), len(v))) if v
                             else "            --")
            print("%-10s%22s%22s%22s" % (m, *cells))

        print()
        print("by UTC hour block -- NO DATA should pile up after midnight if the")
        print("reset is what produces it")
        print("%-10s%22s%22s%22s" % ("UTC", "PASSED", "NO DATA", "LOW FCST"))
        for b in sorted(hour):
            cells = []
            for g in ("PASSED", "NO DATA", "LOW FCST"):
                v = hour[b].get(g) or []
                cells.append(("%6.2f%%  [%5d]" % (sum(v) / len(v), len(v))) if v
                             else "            --")
            print("%02d-%02d     %22s%22s%22s" % (b * 6, b * 6 + 5, *cells))

    print()
    print("READ THIS")
    print("  PASSED mixes modes the guard never judged (blocked events do not record")
    print("  the entry mode). It is the pipeline's 15m pass-through, not a matched")
    print("  trend-only control -- compare NO DATA with LOW FCST first, then with it.")
    print("  These are the bot's own candidates (TH-06); nothing here speaks for")
    print("  coins the upstream gates never let reach trend_quality.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
