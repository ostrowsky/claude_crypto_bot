"""Weekly steering pair: `Coverage@move` and `Precision@alert`.

The North Star needs a change of 2.1x its own value to be visible at a weekly
comparison — it can only see the metric tripling or collapsing, which is the
same as having no measurement. The label store supplies ~16 qualifying events a
day, about 114 a week, against ~20 top-20 winners. This module turns that into
the two rates that can actually move within a week.

Both read the same `MoveEvent` and the same boundaries, so the trade between
them is visible instead of hidden: coverage alone rewards alerting on
everything, precision alone rewards alerting on nothing.

`precision_all` sits beside `precision_early` to close a gaming vector. Early
precision counts only alerts before the deadline, so on its own it would reward
alerting *late* — a late wrong alert simply leaves the denominator. When the two
diverge, the bot is buying precision with lateness, which is the opposite of the
mission.

    pyembed\\python.exe files\\weekly_steering.py --days 7
    pyembed\\python.exe files\\weekly_steering.py --compare 7

Spec: docs/specs/features/weekly-steering-pair-spec.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MIN_ACTIVE_HOURS = 18
BOOTSTRAP_DEFAULT = 2000
MIN_DAYS_TO_COMPARE = 5
WINDOW_RATIO_TOLERANCE = 0.5        # windows differing by more than 2x are not compared


# ── core computation ────────────────────────────────────────────────────────

def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _bootstrap_ci(per_day: list[tuple[int, int]], *, draws: int,
                  seed: int = 20260814) -> tuple[float, float]:
    """Day-clustered bootstrap.

    Rows are resampled by DAY, not individually: 98 symbols on one day share
    market beta and are not 98 independent facts. Resampling rows would report
    an interval far tighter than the evidence supports.
    """
    days = [d for d in per_day if d[1] > 0]
    if not days:
        return (0.0, 0.0)
    if len(days) == 1:
        # One independent observation cannot produce a narrower interval than
        # itself. Saying so is more honest than a spuriously tight range.
        value = _rate(days[0][0], days[0][1])
        return (round(value, 6), round(value, 6))
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        picked = [days[rng.randrange(len(days))] for _ in range(len(days))]
        num = sum(p[0] for p in picked)
        den = sum(p[1] for p in picked)
        samples.append(_rate(num, den))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return (round(lo, 6), round(hi, 6))


def _metric(per_day: dict[str, tuple[int, int]], *, base_rate: float | None,
            draws: int) -> dict[str, Any]:
    """`base_rate=None` means the metric has no natural base to lift against.

    Coverage is such a metric: its denominator is already the qualifying events,
    so a "lift" would be the value divided by one — the number wearing a label
    it has not earned. Reporting None is more honest than reporting 0.05x.
    """
    num = sum(v[0] for v in per_day.values())
    den = sum(v[1] for v in per_day.values())
    value = _rate(num, den)
    return {
        "value": round(value, 6),
        "n": den,
        "hits": num,
        "base_rate": round(base_rate, 6) if base_rate is not None else None,
        "lift": round(value / base_rate, 4) if base_rate else None,
        "ci": _bootstrap_ci(list(per_day.values()), draws=draws),
        "days": len(per_day),
    }


def compute(labels: Sequence[dict], alerts: Sequence[dict],
            observed_hours: dict[str, int], *,
            bootstrap: int = BOOTSTRAP_DEFAULT) -> dict[str, Any]:
    """The pair over one window.

    `labels`  — MoveEvent records (symbol, utc_day, qualifies_move5, deadline)
    `alerts`  — one per (symbol, utc_day) already, or the earliest is used
    `observed_hours` — distinct hours the bot emitted anything, per UTC day
    """
    # A day the labels know about but the event log does not is a day the bot
    # was silent for — downtime, not an absent day. Counting it nowhere would
    # quietly shrink the window instead of reporting the outage (TH-05).
    label_days = {r["utc_day"] for r in labels}
    hours = {d: observed_hours.get(d, 0) for d in label_days | set(observed_hours)}
    scored_days = {d for d, h in hours.items() if h >= MIN_ACTIVE_HOURS}
    excluded = {d for d in hours if d not in scored_days}

    earliest: dict[tuple[str, str], int] = {}
    for a in alerts:
        key = (a["symbol"], a["utc_day"])
        ts = int(a["ts_ms"])
        if key not in earliest or ts < earliest[key]:
            earliest[key] = ts

    cov: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pre: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pal: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    qualifying = total_events = 0

    for rec in labels:
        day = rec["utc_day"]
        if day not in scored_days or not rec.get("complete", True):
            continue
        total_events += 1
        qualifies = bool(rec.get("qualifies_move5"))
        qualifying += 1 if qualifies else 0
        deadline = rec.get("early_deadline_ts")
        alert_ts = earliest.get((rec["symbol"], day))
        # Strictly before: an alert inside the deadline's own hour cannot be
        # ordered against the crossing, so it is counted late.
        eligible = alert_ts is not None and (deadline is None or alert_ts < deadline)

        if qualifies:
            cov[day][1] += 1
            cov[day][0] += 1 if eligible else 0
        if eligible:
            pre[day][1] += 1
            pre[day][0] += 1 if qualifies else 0
        if alert_ts is not None:
            pal[day][1] += 1
            pal[day][0] += 1 if qualifies else 0

    base = _rate(qualifying, total_events)
    to_tuples = lambda d: {k: (v[0], v[1]) for k, v in d.items()}  # noqa: E731
    return {
        "days_scored": len(scored_days & {r["utc_day"] for r in labels}),
        "days_excluded_no_data": len(excluded),
        "events": total_events,
        "qualifying": qualifying,
        "coverage": _metric(to_tuples(cov), base_rate=None, draws=bootstrap),
        "precision_early": _metric(to_tuples(pre), base_rate=base, draws=bootstrap),
        "precision_all": _metric(to_tuples(pal), base_rate=base, draws=bootstrap),
    }


def compare(before: dict, after: dict, *, metric: str,
            practical_threshold: float) -> dict[str, Any]:
    """Two windows, judged by the observed interval against a pre-registered
    practical-significance threshold — never by MDE, which is a design-time
    power parameter and cannot be a decision rule on observed data."""
    a_days, b_days = before["days_scored"], after["days_scored"]
    if min(a_days, b_days) < MIN_DAYS_TO_COMPARE or \
            min(a_days, b_days) / max(a_days, b_days, 1) < WINDOW_RATIO_TOLERANCE:
        return {"verdict": "NOT_COMPARABLE",
                "text": (f"окна несопоставимы: {a_days} и {b_days} зачтённых дней "
                         f"— сравнение отказано (§0a правило 4)")}

    lo_a, hi_a = before[metric]["ci"]
    lo_b, hi_b = after[metric]["ci"]
    delta = after[metric]["value"] - before[metric]["value"]

    if lo_b > hi_a and delta >= practical_threshold:
        verdict, text = "IMPROVING", f"улучшение {delta:+.3f}, интервалы не пересекаются"
    elif hi_b < lo_a and -delta >= practical_threshold:
        verdict, text = "DEGRADING", f"ухудшение {delta:+.3f}, интервалы не пересекаются"
    elif abs(delta) < practical_threshold and not (hi_b < lo_a or lo_b > hi_a):
        verdict, text = "PRACTICALLY_EQUIVALENT", (
            f"разница {delta:+.3f} меньше порога практической значимости "
            f"{practical_threshold:.3f}")
    else:
        verdict, text = "INSUFFICIENT_EVIDENCE", (
            f"рано судить: разница {delta:+.3f}, интервалы [{lo_a:.3f},{hi_a:.3f}] и "
            f"[{lo_b:.3f},{hi_b:.3f}] пересекаются относительно порога "
            f"{practical_threshold:.3f}")
    return {"verdict": verdict, "text": text, "delta": round(delta, 6),
            "ci_before": [lo_a, hi_a], "ci_after": [lo_b, hi_b]}


# ── live data wiring ────────────────────────────────────────────────────────

def load_window(days: int, *, end: str | None = None) -> tuple[list, list, dict]:
    """Labels, alerts and observed hours for the last `days` UTC days."""
    import label_store
    import event_store

    end_day = end or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    start_day = (datetime.strptime(end_day, "%Y-%m-%d")
                 - timedelta(days=days - 1)).strftime("%Y-%m-%d")

    watchlist = set(json.loads((HERE / "watchlist.json").read_text(encoding="utf-8")))
    labels = [r for r in label_store.LabelStore().records()
              if start_day <= r["utc_day"] <= end_day and r["symbol"] in watchlist]

    conn = event_store._connect()
    try:
        alerts = [{"symbol": s, "utc_day": d, "ts_ms": _iso_to_ms(ts)}
                  for d, s, ts in conn.execute(
                      "SELECT day, sym, ts FROM events WHERE event='entry' "
                      "AND day BETWEEN ? AND ? AND sym IS NOT NULL",
                      (start_day, end_day)).fetchall() if ts]
        observed = {d: h for d, h in conn.execute(
            "SELECT day, COUNT(DISTINCT substr(ts, 12, 2)) FROM events "
            "WHERE day BETWEEN ? AND ? GROUP BY day", (start_day, end_day)).fetchall()}
    finally:
        conn.close()
    return labels, alerts, observed


def _iso_to_ms(ts: str) -> int:
    return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp() * 1000)


def render(res: dict, title: str) -> str:
    lines = ["=" * 74, title, "=" * 74,
             f"  зачтённых дней {res['days_scored']} · исключено без данных "
             f"{res['days_excluded_no_data']} · событий {res['events']} "
             f"(из них ≥+5%: {res['qualifying']})", ""]
    labels = {"coverage": "Coverage@move   (поймано до дедлайна)",
              "precision_early": "Precision@alert (ранние алерты)",
              "precision_all": "Precision@all   (все алерты)"}
    for key, name in labels.items():
        m = res[key]
        lo, hi = m["ci"]
        lift = f"  лифт {m['lift']:.2f}x" if m.get("lift") else "  (базы нет)"
        lines.append(f"  {name:<38}{m['value']:.3f}  [{lo:.3f}, {hi:.3f}]"
                     f"  n={m['n']}{lift}")
    lines += ["",
              "  Интервалы — бутстрап с кластеризацией по дням: монеты одного дня",
              "  делят рыночную бету и не являются независимыми наблюдениями.",
              "  Это proxy для руления, а не миссия: +5% движение не обязательно",
              "  ракета дневного top-20."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="weekly steering pair")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--compare", type=int, default=0,
                    help="compare the last N days against the N before them")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="pre-registered practical-significance threshold")
    args = ap.parse_args(argv)

    if args.compare:
        n = args.compare
        end_recent = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        end_prior = (datetime.now(timezone.utc) - timedelta(days=1 + n)).strftime("%Y-%m-%d")
        recent = compute(*load_window(n, end=end_recent))
        prior = compute(*load_window(n, end=end_prior))
        print(render(prior, f"ПРЕДЫДУЩИЕ {n} дней (до {end_prior})"))
        print()
        print(render(recent, f"ПОСЛЕДНИЕ {n} дней (до {end_recent})"))
        print()
        for metric in ("coverage", "precision_early"):
            v = compare(prior, recent, metric=metric,
                        practical_threshold=args.threshold)
            print(f"  {metric:<18}{v['verdict']:<24}{v['text']}")
        return 0

    res = compute(*load_window(args.days))
    print(render(res, f"Недельная пара · последние {args.days} дней"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
