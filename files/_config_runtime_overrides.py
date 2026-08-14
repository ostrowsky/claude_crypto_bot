"""Runtime config overrides — applier for the release store.

Reads **only** `.runtime/release/runtime_overrides.json`, which
`release_overrides.py` writes from signed approvals (or explicitly-labelled
legacy entries). It no longer reads `decisions.jsonl`.

That change is the point. The decisions log was research memory *and* an
execution channel: appending an approved record changed live gating at the next
`import config`, and the newest approved record in it had been written by an
LLM. Severing the read is what makes research memory inert
(docs/specs/features/four-store-split-spec.md).

Called at the END of files/config.py so every `import config` sees the active
overrides. Failure never blocks startup — log and continue with defaults.
Snapshot written to .runtime/config_overrides_applied.json for post-mortem.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("config_runtime_overrides")
_ROOT = Path(__file__).resolve().parent.parent
# The ONLY file this module reads. It is written exclusively by
# release_overrides.py from signed approvals (or explicitly-labelled legacy
# entries). Research memory -- decisions.jsonl, the research ledger -- is never
# consulted here: that shared file was the confused-deputy path this split
# exists to sever (docs/specs/features/four-store-split-spec.md).
OVERRIDE_STORE = _ROOT / ".runtime" / "release" / "runtime_overrides.json"
APPLIED_SNAPSHOT = _ROOT / ".runtime" / "config_overrides_applied.json"


def load_active_overrides() -> dict[str, Any]:
    """Read {config_key: value} from the release store.

    No superseding logic lives here any more: the release tool resolved that
    when it materialised the store, so this module has exactly one job and one
    input. Failure is silent-and-empty by design -- a missing or unreadable
    store must never block bot startup, it just means "no overrides".
    """
    if not OVERRIDE_STORE.exists():
        return {}
    try:
        data = json.loads(OVERRIDE_STORE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("release store unreadable (%s) -- running on config.py defaults", exc)
        return {}
    out: dict[str, Any] = {}
    unsigned: list[str] = []
    for key, entry in (data.get("overrides") or {}).items():
        if not isinstance(entry, dict):
            continue
        out[key] = entry.get("value")
        if entry.get("source") == "legacy_decisions_jsonl":
            unsigned.append(key)
    if unsigned:
        out["__unsigned__"] = unsigned
    return out


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
    unsigned = loaded.pop("__unsigned__", [])
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
        "unsigned_legacy_keys": unsigned,
        "source": str(OVERRIDE_STORE),
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
                    OVERRIDE_STORE.name,
                    ", ".join(f"{k}={v['to']} (config.py says {v['from']})"
                              for k, v in applied.items()))
    if unsigned:
        # Visible on every start until re-approved or lapsed: these are in force
        # without an operator signature, carried over from the old executable
        # decisions log so the split did not move live gating.
        LOG.warning("%d override(s) in force UNSIGNED (legacy): %s — "
                    "re-approve or let them lapse (release_overrides.py --status)",
                    len(unsigned), ", ".join(sorted(unsigned)))
    return record
