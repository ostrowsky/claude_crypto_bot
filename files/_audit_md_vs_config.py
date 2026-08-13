"""Audit: does CLAUDE.md still describe the bot that is actually running?

CLAUDE.md is auto-injected into Claude's context, so when it drifts from
config.py every future session starts from a false picture. On 2026-08-13 an
audit found 10 such drifts at once: flags I had changed without updating the doc
(TREND_1H_CHOP_USE_BULL_DAY_RELAX), values rolled back in config but still
advertised as live (TRAIL_MIN_BUFFER_PCT_IMPULSE_SPEED 0.08 -> 0.015), and
dataset sizes understated 4x (critic 36 MB -> 139 MB).

Checks:
  1. every `FLAG = value` claim in CLAUDE.md against the live config.py
  2. every `(~N MB)` dataset size against the file on disk
  3. that flags named in CLAUDE.md exist at all

Blocks documenting history are skipped: a heading marked ИСТОРИЧЕСКИЙ (or
HISTORICAL) turns checking off until the next heading, so past decisions can stay
in the file without failing the audit.

Exit code 1 when anything drifted, so it can gate a commit.
  pyembed\python.exe files\_audit_md_vs_config.py
"""
from __future__ import annotations
import io, os, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "files"))
import config  # noqa: E402

MD = ROOT / "CLAUDE.md"
text = io.open(MD, encoding="utf-8", errors="replace").read()

# --- 1. flag claims, skipping historical sections -------------------------
lines = text.splitlines()
historical = False
claims: list[tuple[int, str, str]] = []
for i, ln in enumerate(lines, 1):
    if ln.startswith("#"):
        historical = ("ИСТОРИЧЕСК" in ln.upper()) or ("HISTORICAL" in ln.upper())
        continue
    if historical:
        continue
    m = re.match(r"^([A-Z][A-Z0-9_]{4,})\s*(?::\s*\w+)?\s*=\s*([^\s#]+)", ln)
    if m:
        claims.append((i, m.group(1), m.group(2).strip().rstrip(",")))


def same(claimed: str, live) -> bool:
    a, b = claimed.strip().strip('"').lower(), str(live).strip().lower()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


drift: list[str] = []
seen: set[str] = set()
for line_no, key, val in claims:
    if key in seen:
        continue
    seen.add(key)
    if not hasattr(config, key):
        drift.append(f"CLAUDE.md:{line_no} {key} — нет такого параметра в config.py")
        continue
    live = getattr(config, key)
    if not same(val, live):
        drift.append(f"CLAUDE.md:{line_no} {key}: в MD {val}, в config {live}")

# --- 2. dataset sizes ------------------------------------------------------
for m in re.finditer(r"`files/([\w.]+\.jsonl)` \(~(\d+) MB\)", text):
    name, claimed_mb = m.group(1), int(m.group(2))
    p = ROOT / "files" / name
    if not p.exists():
        drift.append(f"{name}: заявлен в CLAUDE.md, но файла нет")
        continue
    real_mb = os.path.getsize(p) / 1e6
    if abs(real_mb - claimed_mb) / max(claimed_mb, 1) > 0.3:
        drift.append(f"{name}: в MD ~{claimed_mb} MB, на диске {real_mb:.0f} MB")

print("=" * 72)
print(f"CLAUDE.md vs live config  ·  проверено флагов: {len(seen)}")
print("=" * 72)
if drift:
    for d in drift:
        print("  ✗ " + d)
    print(f"\nрасхождений: {len(drift)} — CLAUDE.md описывает не тот бот, что работает")
    sys.exit(1)
print("  расхождений нет — документация соответствует работающему боту")
