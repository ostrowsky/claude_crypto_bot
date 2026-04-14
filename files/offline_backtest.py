from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
EVENTS_FILE = ROOT / "bot_events.jsonl"
DATASET_FILE = ROOT / "ml_dataset.jsonl"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _stats(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "n": len(values),
        "avg": round(mean(values), 4),
        "win_rate": round(sum(1 for v in values if v > 0) / len(values), 4),
    }


def load_events(path: Path, since: datetime) -> List[dict]:
    events = []
    for rec in _iter_jsonl(path):
        ts = rec.get("ts")
        if not ts:
            continue
        dt = _parse_ts(ts)
        if dt < since:
            continue
        rec["_dt"] = dt
        events.append(rec)
    return events


def load_dataset(path: Path, since: datetime) -> List[dict]:
    rows = []
    for rec in _iter_jsonl(path):
        ts = rec.get("ts_signal")
        if not ts:
            continue
        dt = _parse_ts(ts)
        if dt < since:
            continue
        rec["_dt"] = dt
        rows.append(rec)
    return rows


def summarize_events(events: List[dict]) -> dict:
    counts = Counter(rec.get("event", "unknown") for rec in events)
    exits = [rec for rec in events if rec.get("event") == "exit"]
    forwards = [rec for rec in events if rec.get("event") == "forward"]
    blocked = [rec for rec in events if rec.get("event") == "blocked"]

    exit_by_mode: Dict[str, List[float]] = defaultdict(list)
    instant_exits: List[dict] = []
    for rec in exits:
        mode = rec.get("mode", "unknown")
        pnl = rec.get("pnl_pct")
        if pnl is not None:
            exit_by_mode[mode].append(float(pnl))
        if int(rec.get("bars_held", -1)) == 0:
            instant_exits.append(rec)

    forward_by_mode_h: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in forwards:
        mode = rec.get("mode", "unknown")
        horizon = f"T+{rec.get('horizon')}"
        pnl = rec.get("pnl_pct")
        if pnl is not None:
            forward_by_mode_h[mode][horizon].append(float(pnl))

    blocked_reasons = Counter(str(rec.get("reason", "")).split(":")[0] for rec in blocked)

    return {
        "counts": dict(counts),
        "exit_by_mode": {mode: _stats(vals) for mode, vals in exit_by_mode.items()},
        "forward_by_mode": {
            mode: {h: _stats(vals) for h, vals in horizons.items()}
            for mode, horizons in forward_by_mode_h.items()
        },
        "blocked_top_reasons": blocked_reasons.most_common(10),
        "instant_exit_count": len(instant_exits),
        "instant_exit_examples": instant_exits[:10],
    }


def summarize_dataset(rows: List[dict]) -> dict:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in rows:
        signal_type = str(rec.get("signal_type", "none"))
        if signal_type == "none":
            continue
        labels = rec.get("labels") or {}
        for key in ("ret_3", "ret_5", "ret_10", "exit_pnl"):
            val = labels.get(key)
            if val is not None:
                grouped[signal_type][key].append(float(val))

    return {
        mode: {metric: _stats(vals) for metric, vals in metrics.items()}
        for mode, metrics in grouped.items()
    }


def build_suggestions(event_summary: dict, dataset_summary: dict) -> List[str]:
    suggestions: List[str] = []
    blocked_top = event_summary.get("blocked_top_reasons", [])
    if blocked_top:
        top_reason, top_count = blocked_top[0]
        if "портфель" in top_reason:
            suggestions.append(
                f"Чаще всего входы режет портфельный лимит ({top_count} случаев). "
                f"Стоит усиливать replacement policy и ранжирование кандидатов."
            )
        if "MTF" in top_reason:
            suggestions.append(
                f"MTF остаётся главным фильтром ({top_count} случаев). "
                f"Есть смысл отдельно валидировать мягкий допуск по MACD на новых данных."
            )

    if event_summary.get("instant_exit_count", 0) > 0:
        suggestions.append(
            f"Обнаружены мгновенные выходы на 0-м баре ({event_summary['instant_exit_count']} случаев). "
            f"Нужен guard против same-bar exit после входа/рестарта."
        )

    weak_modes = []
    for mode, stats_by_metric in dataset_summary.items():
        ret5 = stats_by_metric.get("ret_5")
        if ret5 and ret5["n"] >= 5 and ret5["avg"] < 0:
            weak_modes.append((mode, ret5["avg"]))
    weak_modes.sort(key=lambda x: x[1])
    for mode, avg in weak_modes[:3]:
        suggestions.append(
            f"Режим `{mode}` имеет отрицательный средний ret_5 ({avg:+.3f}%). "
            f"Стоит ужесточить entry-фильтры или сократить hold/стоп для этого режима."
        )

    strong_modes = []
    for mode, stats_by_metric in dataset_summary.items():
        ret5 = stats_by_metric.get("ret_5")
        if ret5 and ret5["n"] >= 5 and ret5["avg"] > 0:
            strong_modes.append((mode, ret5["avg"]))
    strong_modes.sort(key=lambda x: x[1], reverse=True)
    for mode, avg in strong_modes[:2]:
        suggestions.append(
            f"Режим `{mode}` стабильно силён по ret_5 ({avg:+.3f}%). "
            f"Можно дать ему чуть больший приоритет в портфельной ротации."
        )

    return suggestions


def render_text(report: dict) -> str:
    lines: List[str] = []
    lines.append("Offline Backtest")
    lines.append(f"Window: {report['window_start']} .. {report['window_end']}")
    lines.append("")
    lines.append("Events:")
    for key, value in sorted(report["event_summary"]["counts"].items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Exit by mode:")
    for mode, stats in sorted(report["event_summary"]["exit_by_mode"].items()):
        if stats:
            lines.append(f"  {mode}: n={stats['n']} avg={stats['avg']:+.4f}% wr={stats['win_rate']:.1%}")
    lines.append("")
    lines.append("Dataset by signal_type:")
    for mode, metrics in sorted(report["dataset_summary"].items()):
        parts = []
        for metric in ("ret_3", "ret_5", "ret_10", "exit_pnl"):
            stats = metrics.get(metric)
            if stats:
                parts.append(f"{metric}=n{stats['n']} avg{stats['avg']:+.4f}% wr{stats['win_rate']:.1%}")
        if parts:
            lines.append(f"  {mode}: " + " | ".join(parts))
    lines.append("")
    lines.append("Top blocked reasons:")
    for reason, count in report["event_summary"]["blocked_top_reasons"]:
        lines.append(f"  {reason}: {count}")
    lines.append("")
    lines.append("Suggestions:")
    for item in report["suggestions"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline backtest/report for local strategy logs")
    parser.add_argument("--events", type=Path, default=EVENTS_FILE)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.lookback_hours)

    events = load_events(args.events, start)
    dataset = load_dataset(args.dataset, start)
    event_summary = summarize_events(events)
    dataset_summary = summarize_dataset(dataset)
    suggestions = build_suggestions(event_summary, dataset_summary)

    report = {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "events_file": str(args.events),
        "dataset_file": str(args.dataset),
        "event_summary": event_summary,
        "dataset_summary": dataset_summary,
        "suggestions": suggestions,
    }

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
