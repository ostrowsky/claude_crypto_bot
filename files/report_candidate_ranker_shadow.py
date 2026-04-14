from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import ml_candidate_ranker
from ml_signal_model import _parse_ts, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_FILE = ROOT / "critic_dataset.jsonl"
DEFAULT_MODEL_FILE = ROOT / "ml_candidate_ranker.json"
DEFAULT_SHADOW_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _target_return(rec: dict) -> float:
    labels = rec.get("labels") or {}
    ret_5 = _safe_float(labels.get("ret_5"))
    trade_taken = bool(labels.get("trade_taken"))
    trade_exit = labels.get("trade_exit_pnl")
    if trade_taken and trade_exit is not None:
        return 0.6 * _safe_float(trade_exit) + 0.4 * ret_5
    return ret_5


def _ret5(rec: dict) -> float:
    return _safe_float((rec.get("labels") or {}).get("ret_5"))


def _exit_pnl(rec: dict) -> float:
    labels = rec.get("labels") or {}
    if labels.get("trade_exit_pnl") is None:
        return 0.0
    return _safe_float(labels.get("trade_exit_pnl"))


def _teacher_payload(rec: dict) -> dict:
    teacher = rec.get("teacher") or {}
    if isinstance(teacher.get("final"), dict):
        return teacher["final"]
    if isinstance(teacher.get("midday"), dict):
        return teacher["midday"]
    return {}


def _teacher_present(rec: dict) -> bool:
    return bool(_teacher_payload(rec))


def _teacher_top_gainer(rec: dict) -> float:
    return 1.0 if _teacher_payload(rec).get("watchlist_top_gainer") else 0.0


def _teacher_capture_ratio(rec: dict) -> float:
    payload = _teacher_payload(rec)
    value = payload.get("capture_ratio")
    return 0.0 if value is None else max(0.0, min(1.5, _safe_float(value)))


def _win_rate(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    wins = sum(1 for v in values if v > 0.0)
    return wins / len(values)


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def _selection_metrics(rows: Sequence[dict]) -> Dict[str, float]:
    targets = [_target_return(r) for r in rows]
    ret5 = [_ret5(r) for r in rows]
    teacher_rows = [r for r in rows if _teacher_present(r)]
    teacher_top = [_teacher_top_gainer(r) for r in teacher_rows]
    teacher_capture = [_teacher_capture_ratio(r) for r in teacher_rows]
    exit_pnls = [_exit_pnl(r) for r in rows if (r.get("labels") or {}).get("trade_exit_pnl") is not None]
    return {
        "count": len(rows),
        "avg_target_return": round(_avg(targets), 4),
        "avg_ret5": round(_avg(ret5), 4),
        "avg_exit_pnl": round(_avg(exit_pnls), 4) if exit_pnls else 0.0,
        "win_rate": round(_win_rate(targets), 4),
        "teacher_rows": len(teacher_rows),
        "teacher_top_gainer_rate": round(_avg(teacher_top), 4) if teacher_top else 0.0,
        "teacher_capture_ratio": round(_avg(teacher_capture), 4) if teacher_capture else 0.0,
    }


def _load_payload(src: Path | dict) -> dict:
    if isinstance(src, dict):
        return src
    return json.loads(Path(src).read_text(encoding="utf-8"))


def _bar_group_key(rec: dict) -> str:
    return str(rec.get("ts_signal") or rec.get("bar_ts") or "")


def build_shadow_report(
    dataset_path: Path,
    model_src: Path | dict,
    *,
    top_ns: Sequence[int] = (1, 3, 5),
    min_ts: datetime | None = None,
) -> Dict[str, Any]:
    rows = ml_candidate_ranker.load_training_rows(dataset_path, min_ts=min_ts)
    payload = _load_payload(model_src)
    action_counts = Counter(str(r.get("decision", {}).get("action", "")) for r in rows)
    signal_counts = Counter(str(r.get("signal_type", "")) for r in rows)
    blocked_like = sum(action_counts.get(k, 0) for k in ("blocked", "candidate", "snapshot"))

    for rec in rows:
        comps = ml_candidate_ranker.predict_components_from_candidate_payload(payload, rec)
        rec["_pred_proba"] = float(comps["quality_proba"])
        rec["_pred_ev"] = float(comps["ev_raw"])
        rec["_pred_drawdown"] = float(comps["expected_drawdown"])
        rec["_pred_final_score"] = float(comps["final_score"])
        rec["_target_return"] = _target_return(rec)
        rec["_candidate_score"] = _safe_float((rec.get("decision") or {}).get("candidate_score"))

    groups_raw: Dict[str, List[dict]] = defaultdict(list)
    for rec in rows:
        groups_raw[_bar_group_key(rec)].append(rec)

    grouped_rows = [group for group in groups_raw.values() if len(group) >= 2]
    grouped_rows.sort(key=lambda group: group[0].get("ts_signal", ""))

    top_n_reports: List[Dict[str, Any]] = []
    for top_n in top_ns:
        eligible = [group for group in grouped_rows if len(group) >= top_n]
        baseline_selected: List[dict] = []
        ranker_selected: List[dict] = []
        overlap_total = 0
        for group in eligible:
            baseline = sorted(
                group,
                key=lambda r: (
                    r.get("_candidate_score", 0.0),
                    _safe_float((r.get("decision") or {}).get("forecast_return_pct")),
                    r.get("sym", ""),
                ),
                reverse=True,
            )[:top_n]
            ranker = sorted(
                group,
                key=lambda r: (
                    r.get("_pred_final_score", 0.0),
                    r.get("_pred_proba", 0.0),
                    r.get("_candidate_score", 0.0),
                    r.get("sym", ""),
                ),
                reverse=True,
            )[:top_n]
            baseline_selected.extend(baseline)
            ranker_selected.extend(ranker)
            overlap_total += len({r["id"] for r in baseline} & {r["id"] for r in ranker})

        baseline_metrics = _selection_metrics(baseline_selected)
        ranker_metrics = _selection_metrics(ranker_selected)
        top_n_reports.append(
            {
                "top_n": int(top_n),
                "eligible_groups": len(eligible),
                "baseline": baseline_metrics,
                "ranker": ranker_metrics,
                "delta": {
                    "avg_target_return": round(ranker_metrics["avg_target_return"] - baseline_metrics["avg_target_return"], 4),
                    "avg_ret5": round(ranker_metrics["avg_ret5"] - baseline_metrics["avg_ret5"], 4),
                    "win_rate": round(ranker_metrics["win_rate"] - baseline_metrics["win_rate"], 4),
                    "teacher_top_gainer_rate": round(ranker_metrics["teacher_top_gainer_rate"] - baseline_metrics["teacher_top_gainer_rate"], 4),
                    "teacher_capture_ratio": round(ranker_metrics["teacher_capture_ratio"] - baseline_metrics["teacher_capture_ratio"], 4),
                },
                "overlap_ratio": round(overlap_total / max(1, len(eligible) * top_n), 4),
            }
        )

    head_to_head = {"wins": 0, "losses": 0, "ties": 0, "eligible_groups": 0}
    for group in grouped_rows:
        baseline = sorted(group, key=lambda r: (r.get("_candidate_score", 0.0), r.get("sym", "")), reverse=True)[0]
        ranker = sorted(
            group,
            key=lambda r: (r.get("_pred_final_score", 0.0), r.get("_pred_proba", 0.0), r.get("_candidate_score", 0.0)),
            reverse=True,
        )[0]
        delta = round(ranker["_target_return"] - baseline["_target_return"], 10)
        head_to_head["eligible_groups"] += 1
        if delta > 0:
            head_to_head["wins"] += 1
        elif delta < 0:
            head_to_head["losses"] += 1
        else:
            head_to_head["ties"] += 1

    sorted_rows = sorted(rows, key=lambda r: r.get("_pred_proba", 0.0), reverse=True)
    bucket_count = min(5, max(1, len(sorted_rows)))
    bucket_size = max(1, len(sorted_rows) // bucket_count)
    calibration: List[Dict[str, Any]] = []
    for idx in range(bucket_count):
        lo = idx * bucket_size
        hi = len(sorted_rows) if idx == bucket_count - 1 else min(len(sorted_rows), (idx + 1) * bucket_size)
        bucket = sorted_rows[lo:hi]
        if not bucket:
            continue
        probs = [r.get("_pred_proba", 0.0) for r in bucket]
        targets = [r["_target_return"] for r in bucket]
        calibration.append(
            {
                "bucket": idx + 1,
                "rows": len(bucket),
                "proba_min": round(min(probs), 4),
                "proba_max": round(max(probs), 4),
                "avg_target_return": round(_avg(targets), 4),
                "win_rate": round(_win_rate(targets), 4),
            }
        )

    warnings: List[str] = []
    if blocked_like == 0:
        warnings.append("Dataset is currently take-heavy: shadow ranking compares taken trades only, not full candidate competition.")
    if sum(1 for r in rows if (r.get("labels") or {}).get("trade_exit_pnl") is not None) < max(50, len(rows) // 10):
        warnings.append("Many rows still lack realized trade_exit_pnl; target quality depends mostly on ret_5.")

    return {
        "dataset_file": str(dataset_path),
        "model_name": payload.get("model_name", ""),
        "rows_total": len(rows),
        "payload_version": int(payload.get("payload_version", 1) or 1),
        "grouped_bar_competitions": len(grouped_rows),
        "data_health": {
            "action_counts": dict(action_counts),
            "signal_counts": dict(signal_counts),
            "blocked_like_rows": blocked_like,
        },
        "top_n": top_n_reports,
        "top1_head_to_head": head_to_head,
        "calibration": calibration,
        "warnings": warnings,
    }


def render_text(report: Dict[str, Any]) -> str:
    lines = [
        "Candidate Ranker Shadow Report",
        f"Rows: {report['rows_total']}",
        f"Grouped competitions: {report['grouped_bar_competitions']}",
    ]
    if report.get("warnings"):
        lines.append("Warnings:")
        for warning in report["warnings"]:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append("Top-N comparison:")
    for item in report["top_n"]:
        lines.append(
            f"  top-{item['top_n']} groups={item['eligible_groups']} "
            f"baseline={item['baseline']['avg_target_return']:+.4f}% "
            f"ranker={item['ranker']['avg_target_return']:+.4f}% "
            f"delta={item['delta']['avg_target_return']:+.4f}% "
            f"tg={item['delta'].get('teacher_top_gainer_rate', 0.0):+.4f} "
            f"cap={item['delta'].get('teacher_capture_ratio', 0.0):+.4f} "
            f"overlap={item['overlap_ratio']:.3f}"
        )
    h2h = report["top1_head_to_head"]
    lines.append("")
    lines.append(
        f"Top-1 head-to-head: groups={h2h['eligible_groups']} "
        f"wins={h2h['wins']} losses={h2h['losses']} ties={h2h['ties']}"
    )
    if report["calibration"]:
        lines.append("")
        lines.append("Calibration:")
        for item in report["calibration"]:
            lines.append(
                f"  bucket {item['bucket']}: rows={item['rows']} "
                f"proba={item['proba_min']:.3f}-{item['proba_max']:.3f} "
                f"target={item['avg_target_return']:+.4f}% wr={item['win_rate']:.3f}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline shadow report for the candidate ranker")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--shadow-out", type=Path, default=DEFAULT_SHADOW_FILE)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--top-n", type=str, default="1,3,5")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    top_ns = tuple(sorted({max(1, int(part.strip())) for part in args.top_n.split(",") if part.strip()}))
    min_ts = _parse_ts(args.min_date) if args.min_date else None
    report = build_shadow_report(args.dataset, args.model, top_ns=top_ns, min_ts=min_ts)
    save_json(args.shadow_out, report)

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"Shadow report saved to: {args.shadow_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
