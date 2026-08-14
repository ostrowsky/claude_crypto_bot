"""The only writer of the runtime override store.

Nothing else may produce behaviour. A record in research memory — appended by an
agent, a script or a person — reaches live gating only if it becomes a signed
approval and this tool materialises it.

    pyembed\\python.exe files\\release_overrides.py --status
    pyembed\\python.exe files\\release_overrides.py --migrate-legacy
    pyembed\\python.exe files\\release_overrides.py --apply

Fail-closed by design: an unsigned approval is refused, a bad signature is
refused, and a protected key is refused even with a valid signature. The one
exception is explicitly labelled — entries migrated from the old executable
`decisions.jsonl`, which predate signatures. They keep working so that live
gating does not change under the operator's feet, they are marked
`legacy_decisions_jsonl` with provenance and a review date, and they are
reported as debt on every run.

Spec: docs/specs/features/four-store-split-spec.md
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from control_plane_stores import (CONTROL_PLANE, OPERATOR_KEY, OVERRIDE_STORE,  # noqa: E402
                                  RELEASE_DIR, ROOT, ControlPlaneStores)

# Same protected set the applier enforces. Duplicated deliberately: refusing at
# release time and again at apply time means neither one is a single point of
# failure.
NEVER_OVERRIDABLE = frozenset({
    "AUTO_APPLY_OVERRIDES_ENABLED", "DEFAULT_WATCHLIST", "TELEGRAM_BOT_TOKEN",
})

LEGACY_DECISIONS = ROOT / ".runtime" / "pipeline" / "decisions" / "decisions.jsonl"
LEGACY_REVIEW_DAYS = 31


def sign(record: dict[str, Any], key: bytes) -> str:
    """HMAC over the canonical payload — signature and timestamp excluded."""
    payload = {k: v for k, v in record.items()
               if k not in ("signature", "at_utc")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      default=str).encode("utf-8")
    return hmac.new(key, blob, hashlib.sha256).hexdigest()


def load_key() -> bytes | None:
    """Read the operator key. Never logged, never printed, never committed."""
    if not OPERATOR_KEY.exists():
        return None
    data = OPERATOR_KEY.read_bytes().strip()
    return data or None


def _is_concrete(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)


def materialise(stores: ControlPlaneStores, *, key: bytes | None = None,
                dry_run: bool = False) -> dict[str, dict]:
    """Build the override set from signed approvals. Research memory is not read.

    Later approvals for the same key win, so a re-approval supersedes.
    """
    accepted: dict[str, dict] = {}
    for rec in stores.approvals.records():
        config_key = str(rec.get("config_key") or "")
        value = rec.get("value")
        source = rec.get("source") or "signed_approval"

        if not config_key or config_key in NEVER_OVERRIDABLE:
            continue
        if not _is_concrete(value):
            continue

        if source == "legacy_decisions_jsonl":
            # Grandfathered so live gating does not move, but visibly unsigned.
            if not rec.get("provenance") or not rec.get("review_by"):
                continue
        else:
            signature = rec.get("signature")
            if not signature or key is None:
                continue
            if not hmac.compare_digest(signature, sign(rec, key)):
                continue

        accepted[config_key] = {
            "value": value,
            "source": source,
            "approval_id": rec.get("approval_id"),
            "provenance": rec.get("provenance") or [],
            "review_by": rec.get("review_by"),
        }

    if not dry_run:
        RELEASE_DIR.mkdir(parents=True, exist_ok=True)
        OVERRIDE_STORE.write_text(json.dumps({
            "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "written_by": "release_overrides.py",
            "overrides": accepted,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
    return accepted


def debt_report(stores: ControlPlaneStores) -> list[dict]:
    """Unsigned overrides still in force, with their review dates."""
    out = []
    for key_name, entry in materialise(stores, key=load_key(), dry_run=True).items():
        if entry["source"] == "legacy_decisions_jsonl":
            out.append({"config_key": key_name, "value": entry["value"],
                        "review_by": entry["review_by"],
                        "provenance": entry["provenance"]})
    return out


def _effective_legacy_overrides() -> dict[str, dict]:
    """What the OLD executable path would apply, read once for migration.

    Mirrors the superseding rules the old reader used: an approved decision is
    active unless a later record for the same key rolled it back or deferred it.
    """
    if not LEGACY_DECISIONS.exists():
        return {}
    active: dict[str, dict] = {}
    for line in LEGACY_DECISIONS.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key_name = rec.get("config_key")
        if not key_name or key_name in NEVER_OVERRIDABLE:
            continue
        stage = str(rec.get("stage") or "")
        to_val = (rec.get("diff") or {}).get("to")
        if stage in ("rolled_back", "deferred", "rejected"):
            active.pop(key_name, None)
            continue
        if stage == "approved" or rec.get("applied"):
            if _is_concrete(to_val):
                active[key_name] = {"value": to_val,
                                    "provenance": [rec.get("decision_id")]}
    return active


def migrate_legacy(*, dry_run: bool = False) -> dict[str, Any]:
    """One-time: carry the currently effective overrides into signed approvals
    as explicitly-unsigned legacy entries, so live gating does not move.

    Idempotent — re-running produces the same set and appends nothing new.
    """
    stores = ControlPlaneStores.default()
    effective = _effective_legacy_overrides()
    already = {r.get("config_key") for r in stores.approvals.records()
               if r.get("source") == "legacy_decisions_jsonl"}
    review_by = (date.today() + timedelta(days=LEGACY_REVIEW_DAYS)).isoformat()

    planned = {}
    for key_name, info in sorted(effective.items()):
        planned[key_name] = info["value"]
        if key_name in already or dry_run:
            continue
        stores.approvals.append({
            "approval_id": f"legacy-{key_name}",
            "config_key": key_name,
            "value": info["value"],
            "signature": None,
            "source": "legacy_decisions_jsonl",
            "provenance": info["provenance"],
            "review_by": review_by,
            "note": ("carried over from the executable decisions log so live "
                     "gating did not change during the four-store split"),
        })
    if not dry_run:
        materialise(stores, key=load_key())
    return planned


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the only writer of the override store")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--migrate-legacy", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    stores = ControlPlaneStores.default()
    if args.migrate_legacy:
        planned = migrate_legacy()
        print(f"migrated {len(planned)} effective override(s): {planned}")

    if args.apply:
        applied = materialise(stores, key=load_key())
        print(f"release store written: {len(applied)} override(s) -> {OVERRIDE_STORE}")

    current = materialise(stores, key=load_key(), dry_run=True)
    print("=" * 70)
    print(f"runtime overrides in force: {len(current)}")
    for key_name, entry in sorted(current.items()):
        mark = "unsigned" if entry["source"] == "legacy_decisions_jsonl" else "signed"
        print(f"  {key_name:<46} = {entry['value']!r:<8} [{mark}]")
    debt = debt_report(stores)
    if debt:
        print()
        print(f"DEBT: {len(debt)} override(s) in force without a signature.")
        for item in debt:
            print(f"  {item['config_key']} — review by {item['review_by']}, "
                  f"from {item['provenance']}")
        print("  Re-approve with a signature, or let them lapse. Not a default.")
    if load_key() is None:
        print()
        print("No operator key at .runtime/release/operator.key — new signed")
        print("approvals cannot be accepted (fail closed). Legacy entries still apply.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
