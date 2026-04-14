from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import critic_dataset
import ml_dataset


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for rec in _iter_rows(path):
        rec_id = str(rec.get("id", "")).strip()
        if rec_id:
            ids.add(rec_id)
    return ids


def _to_critic_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    labels = rec.get("labels", {}) or {}
    return {
        "id": rec.get("id"),
        "sym": rec.get("sym"),
        "tf": rec.get("tf"),
        "ts_signal": rec.get("ts_signal"),
        "bar_ts": rec.get("bar_ts"),
        "signal_type": rec.get("signal_type"),
        "is_bull_day": rec.get("is_bull_day", False),
        "hour_utc": rec.get("hour_utc"),
        "dow": rec.get("dow"),
        "f": rec.get("f", {}),
        "seq": rec.get("seq", []),
        "seq_feature_names": rec.get("seq_feature_names", []),
        "decision": {
            "action": "take",
            "reason_code": "bootstrap_ml_dataset",
            "reason": "backfilled from ml_dataset signal history",
            "stage": "bootstrap",
            "candidate_score": 0.0,
            "base_score": 0.0,
            "score_floor": 0.0,
            "forecast_return_pct": 0.0,
            "today_change_pct": 0.0,
            "ml_proba": None,
            "mtf_soft_penalty": 0.0,
            "fresh_priority": False,
            "catchup": False,
            "continuation_profile": rec.get("signal_type") in {"impulse_speed", "impulse", "alignment"},
            "signal_flags": {},
        },
        "labels": {
            "ret_3": labels.get("ret_3"),
            "ret_5": labels.get("ret_5"),
            "ret_10": labels.get("ret_10"),
            "label_3": labels.get("label_3"),
            "label_5": labels.get("label_5"),
            "label_10": labels.get("label_10"),
            "trade_taken": True,
            "trade_exit_pnl": labels.get("exit_pnl"),
            "trade_exit_reason": labels.get("exit_reason"),
            "trade_bars_held": labels.get("bars_held"),
            "linked_ml_record_id": rec.get("id", ""),
        },
    }


def main() -> int:
    src = ml_dataset.ML_FILE
    dst = critic_dataset.CRITIC_FILE
    existing = _existing_ids(dst)
    added = 0

    rows_to_add: List[str] = []
    for rec in _iter_rows(src):
        if rec.get("signal_type") == "none":
            continue
        rec_id = str(rec.get("id", "")).strip()
        if not rec_id or rec_id in existing:
            continue
        out = _to_critic_row(rec)
        rows_to_add.append(json.dumps(out, ensure_ascii=False, cls=critic_dataset._Enc))
        existing.add(rec_id)
        added += 1

    if rows_to_add:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a", encoding="utf-8") as f:
            for line in rows_to_add:
                f.write(line + "\n")

    print(
        json.dumps(
            {
                "source": str(src),
                "target": str(dst),
                "added": added,
                "total_target_rows": len(existing),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
