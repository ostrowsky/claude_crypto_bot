from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import critic_dataset


def _load_rows(path: Path) -> List[Dict[str, Any]]:
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


def build_report(now: datetime | None = None) -> Dict[str, Any]:
    now = now or datetime.utcnow()
    path = critic_dataset.CRITIC_FILE
    rows = _load_rows(path)
    last_24h_cutoff = now - timedelta(hours=24)

    def _parse_ts(rec: Dict[str, Any]) -> datetime | None:
        raw = rec.get("ts_signal")
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    last_24h_rows = [r for r in rows if (_parse_ts(r) or datetime.min) >= last_24h_cutoff]
    actions = Counter(str(r.get("decision", {}).get("action", "")) for r in rows)
    signals = Counter(str(r.get("signal_type", "")) for r in rows)
    labeled = sum(1 for r in rows if r.get("labels", {}).get("ret_3") is not None)
    taken = sum(1 for r in rows if bool(r.get("labels", {}).get("trade_taken")))
    outcomes = sum(1 for r in rows if r.get("labels", {}).get("trade_exit_pnl") is not None)

    return {
        "path": str(path),
        "exists": path.exists(),
        "rows_total": len(rows),
        "rows_last_24h": len(last_24h_rows),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "last_write_time": path.stat().st_mtime if path.exists() else 0,
        "labeled_rows": labeled,
        "trade_taken_rows": taken,
        "trade_outcome_rows": outcomes,
        "actions": dict(actions),
        "signal_types": dict(signals),
    }


def render_text(report: Dict[str, Any]) -> str:
    if not report["exists"]:
        return "Critic dataset: file not created yet"
    lines = [
        f"Critic dataset: {report['rows_total']} rows",
        f"Last 24h: {report['rows_last_24h']}",
        f"Labeled: {report['labeled_rows']}",
        f"Trade taken: {report['trade_taken_rows']}",
        f"Trade outcomes: {report['trade_outcome_rows']}",
        f"Size: {report['size_bytes']} bytes",
    ]
    if report["actions"]:
        actions = ", ".join(f"{k}={v}" for k, v in sorted(report["actions"].items()))
        lines.append(f"Actions: {actions}")
    if report["signal_types"]:
        sigs = ", ".join(f"{k}={v}" for k, v in sorted(report["signal_types"].items()))
        lines.append(f"Signal types: {sigs}")
    return "\n".join(lines)


def main() -> int:
    print(render_text(build_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
