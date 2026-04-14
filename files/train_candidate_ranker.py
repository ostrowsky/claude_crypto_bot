from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_candidate_ranker
import report_candidate_ranker_shadow
from ml_signal_model import _parse_ts, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_FILE = ROOT / "critic_dataset.jsonl"
DEFAULT_MODEL_FILE = ROOT / "ml_candidate_ranker.json"
DEFAULT_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"
DEFAULT_SHADOW_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the candidate ranker and build a shadow top-N report")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--positive-ret-threshold", type=float, default=0.0)
    parser.add_argument("--ev-lambda", type=float, default=ml_candidate_ranker.DEFAULT_EV_LAMBDA)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--shadow-out", type=Path, default=DEFAULT_SHADOW_FILE)
    parser.add_argument("--top-n", type=str, default="1,3,5")
    parser.set_defaults(require_catboost=True)
    parser.add_argument("--require-catboost", dest="require_catboost", action="store_true")
    parser.add_argument("--allow-fallback", dest="require_catboost", action="store_false")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    min_ts = _parse_ts(args.min_date) if args.min_date else None
    top_ns = tuple(sorted({max(1, int(part.strip())) for part in args.top_n.split(",") if part.strip()}))

    report = ml_candidate_ranker.train_and_evaluate(
        args.dataset,
        positive_ret_threshold=args.positive_ret_threshold,
        min_ts=min_ts,
        min_rows=args.min_rows,
        ev_lambda=args.ev_lambda,
        require_catboost=args.require_catboost,
    )
    model_payload = ml_candidate_ranker.build_live_model_payload(report)
    report_without_payload = {k: v for k, v in report.items() if k != "model_payload"}

    save_json(args.model_out, model_payload)
    save_json(args.report_out, report_without_payload)

    shadow_report = report_candidate_ranker_shadow.build_shadow_report(
        args.dataset,
        model_payload,
        top_ns=top_ns,
        min_ts=min_ts,
    )
    save_json(args.shadow_out, shadow_report)

    if args.as_json:
        print(
            json.dumps(
                {
                    "train_report": report_without_payload,
                    "shadow_report": shadow_report,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(ml_candidate_ranker.render_text(report))
        print("")
        print(report_candidate_ranker_shadow.render_text(shadow_report))
        print("")
        print(f"Model saved to: {args.model_out}")
        print(f"Train report saved to: {args.report_out}")
        print(f"Shadow report saved to: {args.shadow_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
