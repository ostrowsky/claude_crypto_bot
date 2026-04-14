from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


BOT_EVENTS_FILE = Path("bot_events.jsonl")
AGENT_EVENTS_FILE = Path("agent_events.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standalone market-agent signals with bot signals.")
    parser.add_argument("--hours", type=float, default=24.0, help="Window size in hours")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args()


def _parse_ts(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _load_events(path: Path, cutoff: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(str(obj.get("ts", "")))
            if ts is None or ts < cutoff:
                continue
            if obj.get("event") not in {"entry", "exit"}:
                continue
            if not obj.get("sym") or not obj.get("tf"):
                continue
            obj["_ts"] = ts
            rows.append(obj)
    rows.sort(key=lambda x: x["_ts"])
    return rows


def _event_key(row: Dict[str, Any]) -> str:
    return f"{row.get('event')}|{row.get('sym')}|{row.get('tf')}"


def _match_events(
    agent_events: List[Dict[str, Any]],
    bot_events: List[Dict[str, Any]],
    tolerance_minutes: int = 90,
) -> Dict[str, Any]:
    tolerance = timedelta(minutes=tolerance_minutes)
    bot_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in bot_events:
        bot_by_key[_event_key(row)].append(row)

    matched: List[Dict[str, Any]] = []
    agent_only: List[Dict[str, Any]] = []
    used_bot_ids: set[int] = set()

    for agent_row in agent_events:
        key = _event_key(agent_row)
        candidates = bot_by_key.get(key, [])
        best_idx = None
        best_delta = None
        for idx, bot_row in enumerate(candidates):
            row_id = id(bot_row)
            if row_id in used_bot_ids:
                continue
            delta = abs(agent_row["_ts"] - bot_row["_ts"])
            if delta > tolerance:
                continue
            if best_delta is None or delta < best_delta:
                best_idx = idx
                best_delta = delta
        if best_idx is None:
            agent_only.append(agent_row)
            continue
        bot_row = candidates[best_idx]
        used_bot_ids.add(id(bot_row))
        matched.append(
            {
                "event": agent_row["event"],
                "sym": agent_row["sym"],
                "tf": agent_row["tf"],
                "agent_ts": agent_row["_ts"].isoformat(),
                "bot_ts": bot_row["_ts"].isoformat(),
                "lag_min": round((bot_row["_ts"] - agent_row["_ts"]).total_seconds() / 60.0, 2),
                "agent_mode": agent_row.get("mode"),
                "bot_mode": bot_row.get("mode"),
            }
        )

    bot_only = [row for row in bot_events if id(row) not in used_bot_ids]
    return {
        "matched": matched,
        "agent_only": agent_only,
        "bot_only": bot_only,
    }


def _format_row(row: Dict[str, Any]) -> str:
    ts = row["_ts"].strftime("%Y-%m-%d %H:%M")
    return f"{ts} {row['event']} {row['sym']} [{row['tf']}] mode={row.get('mode', '-')}"


def _format_report(payload: Dict[str, Any], hours: float) -> str:
    matched = payload["matched"]
    agent_only = payload["agent_only"]
    bot_only = payload["bot_only"]
    lines = [
        f"Agent vs bot comparison for last {hours:.1f}h",
        f"matched: {len(matched)}",
        f"agent_only: {len(agent_only)}",
        f"bot_only: {len(bot_only)}",
        "",
    ]
    if matched:
        lines.append("Matched:")
        for row in matched[:10]:
            lines.append(
                f"- {row['event']} {row['sym']} [{row['tf']}] lag={row['lag_min']:+.2f}m "
                f"agent={row.get('agent_mode')} bot={row.get('bot_mode')}"
            )
        lines.append("")
    if agent_only:
        lines.append("Agent only:")
        for row in agent_only[:10]:
            lines.append(f"- {_format_row(row)}")
        lines.append("")
    if bot_only:
        lines.append("Bot only:")
        for row in bot_only[:10]:
            lines.append(f"- {_format_row(row)}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    agent_events = _load_events(AGENT_EVENTS_FILE, cutoff)
    bot_events = _load_events(BOT_EVENTS_FILE, cutoff)
    payload = _match_events(agent_events, bot_events)
    if args.json:
        safe_payload = {
            "matched": payload["matched"],
            "agent_only": [
                {
                    "event": row["event"],
                    "sym": row["sym"],
                    "tf": row["tf"],
                    "mode": row.get("mode"),
                    "ts": row["_ts"].isoformat(),
                }
                for row in payload["agent_only"]
            ],
            "bot_only": [
                {
                    "event": row["event"],
                    "sym": row["sym"],
                    "tf": row["tf"],
                    "mode": row.get("mode"),
                    "ts": row["_ts"].isoformat(),
                }
                for row in payload["bot_only"]
            ],
        }
        print(json.dumps(safe_payload, ensure_ascii=False, indent=2))
        return
    print(_format_report(payload, args.hours))


if __name__ == "__main__":
    main()
