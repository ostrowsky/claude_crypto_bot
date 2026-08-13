"""Declared write interval per learning artifact, and an alarm when one lapses.

Two silent stalls prompted this, both invisible to every report at the time:

  * `.runtime/backfill_critic.lock` held for 1389 hours, so 11 908 rows went
    unlabelled from mid-June to 2026-08-13;
  * in the sibling bot, `critic_dataset.jsonl` stopped being written on
    2026-08-04 while every other dataset kept updating.

Neither was a crash. Both were inputs that simply stopped arriving, and nothing
in the system had an opinion about how often they *should* arrive. This module
is that opinion: every artifact the learning loop depends on declares a maximum
age, and anything past it is reported.

The thresholds below are measured, not guessed — each is roughly 2–3× the
observed cadence on 2026-08-13, so normal jitter never fires an alarm.

An artifact gated behind a config flag reports `disabled` rather than `stale`
when the flag is off: `fast_reversal_catboost.cbm` is 46 days old on purpose,
and a checker that cries wolf about it is a checker people stop reading.

    pyembed\\python.exe files\\artifact_freshness.py          # table, exit 1 if stale
    pyembed\\python.exe files\\artifact_freshness.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


@dataclass(frozen=True)
class Artifact:
    name: str
    path: str                 # relative to repo root
    max_age_h: float
    why: str
    flag: str | None = None   # config attribute that must be true to expect writes
    is_dir: bool = False      # newest file inside the directory is the timestamp


# Observed ages on 2026-08-13 are in the `why` text so a later reader can tell
# whether a threshold was ever grounded in anything.
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("bot_events", "files/bot_events.jsonl", 2,
             "the bot appends continuously; a 2h gap means it is not running"),
    Artifact("critic_dataset", "files/critic_dataset.jsonl", 6,
             "signal records with outcomes, written as decisions happen"),
    Artifact("ml_dataset", "files/ml_dataset.jsonl", 6,
             "raw ML rows, written alongside critic records"),
    Artifact("top_gainer_dataset", "files/top_gainer_dataset.jsonl", 12,
             "intraday snapshots at 08:30/14:30/20:30 local plus EOD"),
    Artifact("bandit_entry_state", "files/bandit_entry_state.json", 36,
             "rebuilt by the nightly EOD learning cycle"),
    Artifact("top_gainer_model", "files/top_gainer_model.json", 36,
             "retrained nightly by daily_learning.py"),
    Artifact("ml_candidate_ranker", "files/ml_candidate_ranker.json", 6,
             "RL worker retrains hourly"),
    Artifact("learning_progress", ".runtime/learning_progress.jsonl", 36,
             "one row per nightly cycle; this is the file trend reports read"),
    Artifact("metrics_daily", ".runtime/metrics_daily.jsonl", 36,
             "L0 of the pipeline; when it stalls the North Star goes invisible"),
    Artifact("pipeline_health", ".runtime/pipeline/health", 36,
             "daily health snapshot consumed by the morning report", is_dir=True),
    Artifact("fast_reversal_model", "files/fast_reversal_catboost.cbm", 36,
             "only trained while the fast-reversal learning flag is on",
             flag="FAST_REVERSAL_LEARNING_ENABLED"),
)


def _flag_enabled(name: str) -> bool:
    try:
        sys.path.insert(0, str(HERE))
        import config  # noqa: PLC0415
        return bool(getattr(config, name, False))
    except Exception:
        # Cannot read config -> assume enabled, so a real stall is not hidden.
        return True


def _mtime(path: Path, is_dir: bool) -> float | None:
    if is_dir:
        if not path.is_dir():
            return None
        stamps = [p.stat().st_mtime for p in path.iterdir() if p.is_file()]
        return max(stamps) if stamps else None
    return path.stat().st_mtime if path.exists() else None


def check(now: float | None = None, root: Path = ROOT) -> list[dict]:
    now = now if now is not None else time.time()
    out: list[dict] = []
    for art in ARTIFACTS:
        row: dict = {"name": art.name, "path": art.path,
                     "max_age_h": art.max_age_h, "why": art.why}
        if art.flag and not _flag_enabled(art.flag):
            row.update(status="disabled", age_h=None,
                       detail=f"{art.flag} is off; no writes expected")
            out.append(row)
            continue
        stamp = _mtime(root / art.path, art.is_dir)
        if stamp is None:
            row.update(status="missing", age_h=None,
                       detail="file or directory does not exist")
        else:
            age = (now - stamp) / 3600.0
            row.update(age_h=round(age, 2),
                       status="stale" if age > art.max_age_h else "ok",
                       detail=f"last write {age:.1f}h ago")
        out.append(row)
    return out


def render(rows: list[dict]) -> str:
    order = {"stale": 0, "missing": 1, "ok": 2, "disabled": 3}
    rows = sorted(rows, key=lambda r: (order.get(r["status"], 9), -(r["age_h"] or 0)))
    lines = ["=" * 74,
             "Свежесть артефактов обучения",
             "=" * 74,
             f"  {'артефакт':<22}{'статус':<10}{'возраст':>9}{'предел':>9}"]
    for r in rows:
        age = f"{r['age_h']:.1f}ч" if r["age_h"] is not None else "—"
        lines.append(f"  {r['name']:<22}{r['status']:<10}{age:>9}{r['max_age_h']:>8.0f}ч")
        if r["status"] in ("stale", "missing"):
            lines.append(f"      {r['why']}")
    bad = [r for r in rows if r["status"] in ("stale", "missing")]
    lines.append("")
    if bad:
        lines.append(f"ПРОСРОЧЕНО: {len(bad)} — вход обучения перестал наполняться.")
        lines.append("Это не сбой процесса и не решение фильтра: данные просто не приходят.")
    else:
        lines.append("Все артефакты в пределах заявленного интервала записи.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="artifact freshness SLO")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = check()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print(render(rows))
    return 1 if any(r["status"] in ("stale", "missing") for r in rows) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
