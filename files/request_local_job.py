from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / ".runtime" / "codex_jobs" / "queue"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a local job request for the Codex execution supervisor.")
    parser.add_argument(
        "job_type",
        choices=[
            "run_rl_headless",
            "stop_rl_headless",
            "run_rl_train_once",
            "run_top_gainer_critic",
            "check_telegram_connectivity",
            "install_catboost",
            "run_trade_bot",
            "stop_trade_bot",
        ],
    )
    parser.add_argument("--phase", choices=["midday", "final"], default="midday")
    parser.add_argument("--date", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "job_type": args.job_type,
        "requested_at": utc_now_iso(),
    }
    if args.job_type == "run_top_gainer_critic":
        payload["phase"] = args.phase
        if args.date:
            payload["date"] = args.date
    path = QUEUE_DIR / f"{job_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "job_id": job_id, "path": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
