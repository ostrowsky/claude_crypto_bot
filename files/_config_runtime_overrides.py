"""Runtime config overrides — RM-4 auto-apply.

Reads `.runtime/pipeline/decisions/decisions.jsonl` and applies APPROVED
decisions that target real config.py constants. Skips:
  - decisions superseded by a later deferred/rolled_back record
    (same logic as pipeline_attribution sticky defer)
  - non-concrete diffs ("current", "+10% looser", strings) — only literal
    numeric/bool diff.to values are auto-applied. Directive diffs require
    operator-supplied concrete values; surface them in the log.

Called at the END of files/config.py so every `import config` in the bot
sees the active overrides. Failure here never blocks startup — log and
continue with defaults.

Snapshot of applied overrides written to .runtime/config_overrides_applied.json
for transparency and post-mortem.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("config_runtime_overrides")
_ROOT = Path(__file__).resolve().parent.parent
DECISIONS_LOG = _ROOT / ".runtime" / "pipeline" / "decisions" / "decisions.jsonl"
APPLIED_SNAPSHOT = _ROOT / ".runtime" / "config_overrides_applied.json"


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _superseded_hyps() -> set[str]:
    """Hypothesis-ids and decision-ids superseded by a later defer/rollback.
    Same sticky logic as pipeline_attribution.attribute()."""
    s: set[str] = set()
    for r in _iter_jsonl(DECISIONS_LOG):
        if r.get("stage") in ("deferred", "rolled_back"):
            hid = r.get("hypothesis_id")
            if hid:
                s.add(hid)
            tgt = r.get("defers") or r.get("rolling_back")
            if tgt:
                s.add(tgt)
    return s


def _is_concrete(v: Any) -> bool:
    """Auto-apply only literal values — never directive strings."""
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return True
    return False


def load_active_overrides() -> dict[str, Any]:
    """Resolve {config_key: value} from active approved decisions.

    Last-writer-wins per config_key (the most recent active approve for a
    given key takes effect; older approves of the same key are shadowed)."""
    superseded = _superseded_hyps()
    overrides: dict[str, Any] = {}
    skipped: list[dict] = []
    for r in _iter_jsonl(DECISIONS_LOG):
        if r.get("stage") != "approved":
            continue
        if r.get("hypothesis_id") in superseded:
            continue
        if r.get("decision_id") in superseded:
            continue
        key = r.get("config_key")
        diff = r.get("diff") or {}
        to_val = diff.get("to")
        if not key:
            continue
        if not _is_concrete(to_val):
            skipped.append({"decision_id": r.get("decision_id"),
                            "config_key": key, "to": to_val,
                            "reason": "non-concrete diff (directive)"})
            continue
        overrides[key] = to_val  # last-writer-wins
    if skipped:
        overrides.setdefault("__skipped__", skipped)
    return overrides


# Keys this mechanism may never set. `AUTO_APPLY_OVERRIDES_ENABLED` is the
# switch that gates this very call: a decision record targeting it could turn
# the mechanism back on — or make the audit snapshot disagree with the value the
# rest of the process reads. A kill switch reachable by the thing it kills is
# not a kill switch.
_NEVER_OVERRIDABLE = frozenset({
    "AUTO_APPLY_OVERRIDES_ENABLED",
    "DEFAULT_WATCHLIST",          # watchlist is immutable (CLAUDE.md §14)
    "TELEGRAM_BOT_TOKEN",
})


def apply_overrides(module_globals: dict) -> dict:
    """Apply active overrides onto the given config module globals.
    Returns a record of what was applied (for transparency)."""
    try:
        loaded = load_active_overrides()
    except Exception as e:
        LOG.warning("runtime override read failed: %s — using defaults", e)
        return {"error": str(e)}

    skipped = loaded.pop("__skipped__", [])
    applied: dict[str, dict] = {}
    not_in_config: list[str] = []
    refused: list[str] = []
    for k, v in loaded.items():
        if k in _NEVER_OVERRIDABLE:
            refused.append(k)
            continue
        if k not in module_globals:
            not_in_config.append(k)
            continue
        old = module_globals[k]
        if old != v:
            module_globals[k] = v
            applied[k] = {"from": old, "to": v}

    record = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied": applied,
        "skipped_non_concrete": skipped,
        "config_key_not_present": not_in_config,
        "refused_protected_key": refused,
    }
    try:
        APPLIED_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        APPLIED_SNAPSHOT.write_text(json.dumps(record, indent=2, default=str),
                                    encoding="utf-8")
    except OSError:
        pass

    if refused:
        LOG.warning("runtime override REFUSED for protected key(s): %s",
                    ", ".join(refused))
    if applied:
        # WARNING, not INFO: a file on disk is silently changing live gating
        # constants away from what config.py reads. That is never routine.
        LOG.warning("runtime config overrides ACTIVE (source: %s): %s — "
                    "disable with AUTO_APPLY_OVERRIDES_ENABLED=False",
                    DECISIONS_LOG.name,
                    ", ".join(f"{k}={v['to']} (config.py says {v['from']})"
                              for k, v in applied.items()))
    return record
