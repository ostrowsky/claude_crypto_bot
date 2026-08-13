#!/usr/bin/env bash
# All mechanical MD-compliance gates in one call. Run from anywhere:
#     bash .claude/skills/md-compliance/scripts/check.sh [--staged]
#
# --staged runs the narrow "change" profile against the staged diff, which is
# what the pre-commit hook uses. Without it you get the full audit.
#
# Exit code is what matters: 0 = clean, 1 = drift found, 2 = harness broke.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="$ROOT/pyembed/python.exe"
[ -x "$PY" ] || PY="python"

if [ "${1:-}" = "--staged" ]; then
  ARGS=(change --staged)
else
  ARGS=(full)
fi

cd "$ROOT/files" || exit 2
"$PY" -m truth_harness "${ARGS[@]}" --json "$ROOT/.runtime/harness_last.json"
RC=$?

echo
case $RC in
  0) echo "MECHANICAL: clean." ;;
  1) echo "MECHANICAL: drift found above — fix or justify each item." ;;
  *) echo "MECHANICAL: the harness itself failed (rc=$RC). Report that, do not"
     echo "read a broken harness as a pass." ;;
esac
echo "Still owed: the judgment checks in reference.md. No script can see a leaky"
echo "feature, a missing base rate, or a report claiming more than the data shows."
exit $RC
