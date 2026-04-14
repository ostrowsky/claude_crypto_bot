from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import critic_dataset
import top_gainer_critic


REPORT_DIR = top_gainer_critic.REPORT_DIR


def _iter_reports(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for report_path in sorted(path.glob("top_gainer_critic_*.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def main() -> int:
    reports = list(_iter_reports(REPORT_DIR))
    summaries: List[Dict[str, Any]] = []
    total_rows_annotated = 0

    for report in reports:
        summary = critic_dataset.annotate_top_gainer_teacher(report)
        summaries.append(summary)
        total_rows_annotated += int(summary.get("rows_annotated", 0) or 0)

    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "reports_processed": len(reports),
                "rows_annotated_total": total_rows_annotated,
                "summaries": summaries,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
