"""Deterministic truth/compliance harness for the crypto bot.

Profiles:
  python files/truth_harness.py full
  python files/truth_harness.py change --staged

The harness is read-only. It rejects unsupported claims and missing evidence
contracts; it does not decide whether a trading hypothesis is good.

Spec: docs/specs/features/truth-harness-spec.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
AUTO_LOOP_SPEC = ROOT / "docs" / "specs" / "features" / "auto-improvement-loop-spec.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOOP_COMPONENTS = {
    "files/analyze_learning_progress.py", "files/bot_health_report.py",
    "files/daily_learning.py",
    "files/offline_rl.py", "files/pipeline_attribution.py",
    "files/pipeline_baseline.py", "files/pipeline_hypothesis.py",
    "files/pipeline_validator.py", "files/report_metrics_daily.py",
    "files/train_top_gainer.py",
}
BEHAVIOUR_COMPONENTS = {
    "config.py", "files/config.py", "files/monitor.py", "files/strategy.py",
    "files/contextual_bandit.py", "files/rotation.py",
    "files/correlation_guard.py", "files/trend_scout_rules.py",
}
METRIC_COMPONENT_MARKERS = (
    "report", "metric", "backtest", "critic", "train", "learning",
    "ranker", "model", "bandit", "pipeline", "truth_harness",
)


@dataclass(frozen=True)
class Finding:
    check_id: str
    invariant: str
    severity: str
    message: str
    evidence: str = ""
    remediation: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity == "error"


class Audit:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.findings: list[Finding] = []
        self.checks_run: list[str] = []

    def checked(self, check_id: str) -> None:
        if check_id not in self.checks_run:
            self.checks_run.append(check_id)

    def add(self, check_id: str, invariant: str, severity: str,
            message: str, evidence: str = "", remediation: str = "") -> None:
        self.checked(check_id)
        self.findings.append(Finding(
            check_id, invariant, severity, message, evidence, remediation,
        ))

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "profile": self.profile,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail" if self.blocking else "pass",
            "checks_run": self.checks_run,
            "blocking_count": len(self.blocking),
            "warning_count": sum(f.severity == "warning" for f in self.findings),
            "findings": [asdict(f) for f in self.findings],
        }


def _run(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _latest_json(directory: Path, pattern: str) -> tuple[Path | None, dict | None]:
    paths = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    for path in reversed(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return path, value
    return None, None


def audit_md_config(audit: Audit, root: Path = ROOT) -> None:
    audit.checked("TH09_MD_CONFIG")
    script = root / "files" / "_audit_md_vs_config.py"
    if not script.exists():
        audit.add("TH09_MD_CONFIG", "TH-09", "error",
                  "MD/config auditor is missing", str(script))
        return
    result = _run([sys.executable, str(script)], root)
    if result.returncode != 0:
        tail = "\n".join(((result.stdout or "") + (result.stderr or "")).splitlines()[-12:])
        audit.add("TH09_MD_CONFIG", "TH-09", "error",
                  "CLAUDE.md does not match current config/datasets", tail,
                  "Update the claim or implementation; mark history explicitly.")


def audit_enforcement(audit: Audit, root: Path = ROOT,
                      check_git_config: bool = True) -> None:
    audit.checked("TH12_ENFORCEMENT")
    required = {
        root / ".githooks" / "pre-commit": "tracked pre-commit hook",
        root / "skills" / "crypto-bot-truth-harness" / "SKILL.md": "project skill",
        root / "AGENTS.md": "agent workflow",
        root / "docs" / "specs" / "features" / "truth-harness-spec.md": "Harness spec",
    }
    for path, label in required.items():
        if not path.exists():
            audit.add("TH12_ENFORCEMENT", "TH-12", "error",
                      f"Missing {label}", str(path))
    agents = _read(root / "AGENTS.md").lower()
    if "truth_harness" not in agents and "truth-harness" not in agents:
        audit.add("TH12_ENFORCEMENT", "TH-12", "error",
                  "AGENTS.md does not require the Truth Harness for changes")
    hook = _read(root / ".githooks" / "pre-commit")
    if hook and "truth_harness.py" not in hook:
        audit.add("TH12_ENFORCEMENT", "TH-12", "error",
                  "pre-commit hook exists but does not run truth_harness.py")
    if check_git_config:
        cfg = _run(["git", "config", "--get", "core.hooksPath"], root)
        value = (cfg.stdout or "").strip().replace("\\", "/")
        if value not in (".githooks", "./.githooks"):
            audit.add("TH12_ENFORCEMENT", "TH-12", "error",
                      "This worktree does not use the tracked hooks",
                      f"core.hooksPath={value or '<unset>'}",
                      "Run: git config core.hooksPath .githooks")


def audit_model_provenance(audit: Audit, root: Path = ROOT) -> None:
    """Detect known optimistic evaluation paths from source, not model blobs."""
    audit.checked("TH03_TOP_GAINER_TARGET")
    daily = _read(root / "files" / "daily_learning.py")
    features = _read(root / "files" / "top_gainer_model.py")
    trainer = _read(root / "files" / "train_top_gainer.py")
    offline = _read(root / "files" / "offline_rl.py")
    north_star = _read(root / "files" / "_compute_early_capture.py")

    same_snapshot_label = (
        "rank_gainers(tickers)" in daily
        and '"label_top20": int(sym in top20)' in daily
    )
    if same_snapshot_label and '"tg_return_since_open"' in features:
        audit.add(
            "TH03_TOP_GAINER_TARGET", "TH-03", "error",
            "Top-gainer AUC recognizes the current leaderboard; it does not prove final-leaderboard prediction",
            "label_top20 and tg_return_since_open come from the same snapshot",
            "Attach a later immutable EOD label and publish a leaky-feature ablation.",
        )
    if same_snapshot_label and 'label_field="label_top20"' in north_star:
        audit.add(
            "TH03_NORTH_STAR_TARGET", "TH-03", "error",
            "Canonical North Star is not based on immutable later EOD ground truth",
            "_compute_early_capture.py consumes label_top20 created from the same rolling-24h snapshot",
            "Create later EOD labels, version the metric, and recompute historical baselines.",
        )

    audit.checked("TH04_DAY_GROUP_SPLIT")
    row_split = "split_idx = int(len(X) * (1 - val_ratio))" in trainer
    grouped_day_split = "split_day" in trainer or "unique_days" in trainer
    if row_split and not grouped_day_split:
        audit.add(
            "TH04_DAY_GROUP_SPLIT", "TH-04", "error",
            "Top-gainer validation can split one UTC day across train and holdout",
            "train_top_gainer.py splits sorted rows by row index",
            "Split on complete UTC days and persist date ranges.",
        )

    audit.checked("TH04_BANDIT_POST_FIT")
    trained_first = offline.find('results["entry_bandit"] = train_entry_bandit()')
    evaluated_after = offline.find('results["bandit_accuracy"] = evaluate_bandit_accuracy')
    if trained_first >= 0 and evaluated_after > trained_first:
        audit.add(
            "TH04_BANDIT_POST_FIT", "TH-04", "error",
            "Bandit recall is evaluated post-fit on records used by the bandit",
            "run_offline_training trains before evaluate_bandit_accuracy",
            "Call it diagnostic-only or use a frozen pre-fit model on later days.",
        )


def _canonical_status(report: dict) -> tuple[dict, dict]:
    metrics = ((report.get("metrics_daily_latest") or {}).get("metrics") or {})
    return metrics, metrics.get("NS_EarlyCapture_top20") or {}


def audit_health_report(audit: Audit, root: Path = ROOT,
                        today: date | None = None) -> None:
    audit.checked("TH10_HEALTH_REPORT")
    directory = root / ".runtime" / "pipeline" / "health"
    path, report = _latest_json(directory, "health-*.json")
    if report is None:
        audit.add("TH10_HEALTH_REPORT", "TH-10", "error",
                  "No readable health report", str(directory))
        return
    tg = _read(path.with_suffix(".tg.txt")) if path else ""
    metrics, ns = _canonical_status(report)

    if not ns:
        audit.add("TH10_HEALTH_REPORT", "TH-10", "error",
                  "Health report has no canonical North Star", str(path))
    else:
        missing = [k for k in ("early_capture", "n", "days_window", "days_full")
                   if ns.get(k) is None]
        if missing:
            audit.add("TH10_HEALTH_REPORT", "TH-10", "error",
                      "North Star omits denominator/window evidence",
                      f"missing={missing}")
        score_ns = ((report.get("canonical_scorecard") or {}).get("north_star") or {})
        if score_ns.get("status") not in ("verified", "provisional"):
            audit.add("TH03_NORTH_STAR_DISCLOSURE", "TH-03", "error",
                      "Health report does not disclose provisional North-Star labels",
                      f"status={score_ns.get('status') or 'missing'}")

    training = report.get("training_health") or {}
    if training.get("recall_at_20") is not None:
        required = ("evaluation_scope", "action_rate", "base_rate", "lift")
        missing = [k for k in required if training.get(k) is None]
        if missing:
            audit.add("TH01_RATIO_CONTEXT", "TH-01", "error",
                      "Training recall is published without base/action rate and lift",
                      f"missing={missing}; recall={training.get('recall_at_20')}")
        if training.get("evaluation_scope") != "out_of_sample_time_holdout":
            audit.add("TH04_REPORT_SCOPE", "TH-04", "error",
                      "Training recall is not a valid achievement metric",
                      f"evaluation_scope={training.get('evaluation_scope') or 'unknown'}")
            if "модель: находит" in tg:
                audit.add("TH02_PROXY_AS_PROGRESS", "TH-02", "error",
                          "Telegram turns an in-sample proxy into a capability claim",
                          "text contains 'модель: находит'")

    gap = report.get("training_to_live_gap") or {}
    if gap.get("available") and training.get("evaluation_scope") != "out_of_sample_time_holdout":
        audit.add("TH02_INVALID_GAP", "TH-02", "error",
                  "Training-to-live gap subtracts an in-sample proxy from live deployment",
                  f"gap={gap.get('value')}",
                  "Suppress it until the training side is an honest time holdout.")

    audit.checked("TH11_CANONICAL_COVERAGE")
    required_metrics = {
        "EX1_realized_potential": "profit leakage",
        "D1_D2_precision_msgrate": "signal precision/message rate",
        "E1_time_to_signal": "time to signal",
        "Q2_whipsaw_rate": "whipsaw",
        "Q1_Q3_fast_reversal": "fast reversal",
    }
    for key, question in required_metrics.items():
        if key not in metrics:
            audit.add("TH11_CANONICAL_COVERAGE", "TH-02", "error",
                      f"Canonical metric missing: {question}", key)
    money = ((report.get("canonical_scorecard") or {}).get("portfolio_alpha") or {})
    if money.get("value") is None:
        audit.add("TH11_CANONICAL_COVERAGE", "TH-11", "error",
                  "Portfolio alpha vs buy-and-hold is absent; profitability is unknown",
                  "per-mode/per-trade PnL is not the canonical money metric")
    ex1 = ((report.get("canonical_scorecard") or {}).get("realized_potential") or {})
    if ex1.get("value") is None:
        audit.add("TH11_CANONICAL_COVERAGE", "TH-02/TH-11", "error",
                  "Canonical ZigZag EX1 is absent; realized-potential quality is unknown",
                  str(ex1.get("reason") or "legacy proxy cannot replace canonical EX1"))

    audit.checked("TH10_EVIDENCE_EXPIRY")
    dnt = report.get("do_not_touch") or {}
    verified = dnt.get("last_verified")
    budget = int(dnt.get("verify_every_days") or 0)
    if verified and budget:
        try:
            as_of = today or datetime.now(timezone.utc).date()
            age = (as_of - date.fromisoformat(str(verified))).days
            if age > budget:
                audit.add("TH10_EVIDENCE_EXPIRY", "TH-10", "error",
                          "Gate-lock evidence expired; the protection must remain fail-closed until refreshed",
                          f"last_verified={verified}, age={age}d, budget={budget}d",
                          "Re-run the targeted maximum-period replay.")
        except ValueError:
            audit.add("TH10_EVIDENCE_EXPIRY", "TH-10", "warning",
                      "Cannot parse do_not_touch.last_verified", str(verified))


def audit_legacy_evidence_memory(audit: Audit, root: Path = ROOT) -> None:
    audit.checked("TH08_NEGATIVE_MEMORY")
    no_verdict: list[str] = []
    verdict_re = re.compile(r"VERDICT|ВЕРДИКТ|RESULT|REFUTED|FINDING", re.I)
    for path in sorted((root / "files").glob("_backtest_*.py")):
        if not verdict_re.search(_read(path)):
            no_verdict.append(path.name)
    if no_verdict:
        audit.add("TH08_NEGATIVE_MEMORY", "TH-08", "warning",
                  f"{len(no_verdict)} legacy backtests have no durable verdict",
                  ", ".join(no_verdict[:8]) + (" …" if len(no_verdict) > 8 else ""),
                  "Record period, N, metrics and verdict when each is next used.")


def _git_changed(root: Path, staged: bool) -> list[str]:
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    cmd.extend(["--name-only", "--diff-filter=ACMR"])
    result = _run(cmd, root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines()
            if line.strip()]


def _is_test(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("test_") or "/tests/" in f"/{path.lower()}"


def audit_change_set(audit: Audit, changed: Iterable[str],
                     root: Path = ROOT) -> None:
    paths = sorted(set(str(p).replace("\\", "/") for p in changed))
    audit.checked("TH12_CHANGE_EVIDENCE")
    source = [p for p in paths if p.endswith((".py", ".ps1", ".cmd", ".bat"))
              and not _is_test(p) and not p.startswith("skills/")]
    specs = [p for p in paths if p.startswith("docs/specs/features/")
             and p.endswith("-spec.md")]
    tests = [p for p in paths if _is_test(p)]

    if source and not specs:
        audit.add("TH12_CHANGE_EVIDENCE", "TH-12", "error",
                  "Source change has no staged feature specification",
                  ", ".join(source[:8]))
    if source and not tests:
        audit.add("TH12_CHANGE_EVIDENCE", "TH-12", "error",
                  "Source change has no staged focused test",
                  ", ".join(source[:8]))

    loop_changed = sorted(set(paths) & LOOP_COMPONENTS)
    auto_spec_rel = "docs/specs/features/auto-improvement-loop-spec.md"
    if loop_changed and auto_spec_rel not in paths:
        audit.add("TH12_CHANGE_EVIDENCE", "TH-12", "error",
                  "Loop component changed without its living-spec update",
                  ", ".join(loop_changed))

    behaviour_changed = sorted(set(paths) & BEHAVIOUR_COMPONENTS)
    if behaviour_changed:
        combined = "\n".join(_read(root / p) for p in specs).lower()
        requirements = {
            "rollback": ("rollback", "откат"),
            "maximum-period backtest": ("maximum", "максималь"),
            "shadow/canary decision": ("shadow", "canary", "не применимо"),
        }
        for label, markers in requirements.items():
            if not any(marker in combined for marker in markers):
                audit.add("TH07_BEHAVIOUR_SAFETY", "TH-06/TH-07", "error",
                          f"Behaviour-change spec lacks {label}",
                          ", ".join(behaviour_changed))

    metric_changed = [p for p in source
                      if any(m in Path(p).name.lower() for m in METRIC_COMPONENT_MARKERS)]
    if metric_changed:
        spec_text = "\n".join(_read(root / p) for p in specs)
        if not re.search(r"TH-0?[1-9]|TH-1[0-2]", spec_text):
            audit.add("TH12_CHANGE_EVIDENCE", "TH-12", "error",
                      "Metric/report change spec names no Truth Harness invariant",
                      ", ".join(metric_changed[:8]))


def render(audit: Audit) -> str:
    lines = [
        f"Truth Harness · profile={audit.profile} · "
        f"{'FAIL' if audit.blocking else 'PASS'}",
        f"checks={len(audit.checks_run)} blocking={len(audit.blocking)} "
        f"warnings={sum(f.severity == 'warning' for f in audit.findings)}",
    ]
    if not audit.findings:
        lines.append("No violations found in the checked scope.")
    for finding in audit.findings:
        icon = "ERROR" if finding.severity == "error" else "WARN"
        lines.append(f"\n[{icon}] {finding.check_id} ({finding.invariant})\n  {finding.message}")
        if finding.evidence:
            lines.append(f"  evidence: {finding.evidence}")
        if finding.remediation:
            lines.append(f"  next: {finding.remediation}")
    return "\n".join(lines)


def build_audit(profile: str, *, staged: bool = False,
                root: Path = ROOT) -> Audit:
    audit = Audit(profile)
    if profile == "change":
        audit_enforcement(audit, root, check_git_config=False)
        audit_change_set(audit, _git_changed(root, staged=staged), root)
        return audit
    audit_md_config(audit, root)
    audit_enforcement(audit, root, check_git_config=True)
    audit_model_provenance(audit, root)
    audit_health_report(audit, root)
    audit_legacy_evidence_memory(audit, root)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crypto-bot Truth Harness")
    parser.add_argument("profile", choices=("full", "change"), nargs="?", default="full")
    parser.add_argument("--staged", action="store_true",
                        help="audit staged diff (for pre-commit)")
    parser.add_argument("--json", dest="json_path", type=Path,
                        help="write structured findings")
    args = parser.parse_args(argv)
    try:
        audit = build_audit(args.profile, staged=args.staged)
    except Exception as exc:
        print(f"Truth Harness internal error: {exc}", file=sys.stderr)
        return 2
    print(render(audit))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(audit.payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 1 if audit.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
