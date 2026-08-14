"""Four stores, so that memory and execution stop being the same file.

`.runtime/pipeline/decisions/decisions.jsonl` was both the improvement loop's
memory and its execution channel: appending an approved record changed live
gating at the next `import config`. Two gating constants were live through it,
and the newest approved record in it had been written by an LLM.

The split:

    research_ledger.jsonl      research writes here. Never executable.
    promotion_requests.jsonl   proposals derived from verified results.
    signed_approvals.jsonl     operator authorisation. Authorises; does not act.
    ../release/runtime_overrides.json   the only file config.py reads.

The severing property is structural: **no function reachable from the research
path writes the release store.** `release_overrides.py` is the only writer, and
it accepts a signed approval or an explicitly recorded legacy migration.

Honest boundary: this is enforced by tooling, not by OS permissions — one
process, one user, one filesystem. What is guaranteed and tested is that no code
path leads from research memory to behaviour.

Spec: docs/specs/features/four-store-split-spec.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
CONTROL_PLANE = ROOT / ".runtime" / "control_plane"
RELEASE_DIR = ROOT / ".runtime" / "release"
OVERRIDE_STORE = RELEASE_DIR / "runtime_overrides.json"
OPERATOR_KEY = RELEASE_DIR / "operator.key"


class AppendOnlyStore:
    """A JSONL store that only grows. No update, no delete, by construction."""

    def __init__(self, path: Path, *, executable: bool) -> None:
        self.path = Path(path)
        # Recorded so a reader can see, from the object itself, whether this
        # store can affect behaviour. Only the release store may be executable.
        self.executable = executable

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamped = dict(record)
        stamped.setdefault("at_utc",
                           datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        return stamped

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a malformed line is skipped, never repaired
        return iter(out)

    def records(self) -> list[dict[str, Any]]:
        return list(self)


@dataclass
class ControlPlaneStores:
    """The three non-executable stores, rooted anywhere (tests use a temp root)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.research = AppendOnlyStore(self.root / "research_ledger.jsonl",
                                        executable=False)
        self.promotions = AppendOnlyStore(self.root / "promotion_requests.jsonl",
                                          executable=False)
        self.approvals = AppendOnlyStore(self.root / "signed_approvals.jsonl",
                                         executable=False)

    @classmethod
    def default(cls) -> "ControlPlaneStores":
        return cls(CONTROL_PLANE)
