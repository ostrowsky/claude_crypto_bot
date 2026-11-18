"""Strategic learning-progress analyzer for the crypto bot.

Single token-efficient entry point that answers six recurring questions
without the operator (or Claude) having to read large jsonl files:

  Q1. Is the bot improving toward its target (North Star) metric?
  Q2. Progress or degradation of the learning diagnostics?
  Q3. Are all analytical components (ML / bandit / scout / pipeline)
      actually consuming data to move toward the target?
  Q4. Is there enough data to keep learning?
  Q5. What roadmap stage are we at, and does it need correcting?
  Q6. What are the concrete next steps?

It aggregates:
  - .runtime/metrics_daily.jsonl         (canonical business metrics inc. NS)
  - .runtime/learning_progress.jsonl     (ML/bandit nightly history)
  - .runtime/pipeline/health/health-*.json (pre-aggregated daily health)
  - .runtime/pipeline/decisions/decisions.jsonl (pipeline activity)
  - .runtime/pipeline/attribution/*.json (pipeline hit-rate / effect)
  - docs/specs/features/continuous-improvement-agent-spec.md (signal bot, target)
  - docs/specs/features/autonomous-trading-roadmap-1010-spec.md (trader roadmap)
  - docs/specs/features/auto-improvement-loop-spec.md (LEGACY; reported as stale)

Trends compare only comparable, non-overlapping windows and print n for every
point; with too little evidence the verdict is INSUFFICIENT or INCONCLUSIVE
("рано судить"), never a direction. Spec:
docs/specs/features/learning-progress-comparable-1118-spec.md (TH-01, TH-02,
TH-04, TH-05, TH-10).

Output is a compact structured text report. No file writes, no network.

Usage:
    pyembed\\python.exe files\\analyze_learning_progress.py
    pyembed\\python.exe files\\analyze_learning_progress.py --json   # machine form
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Comparability rules (spec §2) ─────────────────────────────────────────────

NS_METRIC = "NS_EarlyCapture_top20_v2"
NS_LABEL_PROVENANCE = "immutable_later_eod_klines"
# Two North Star rows are comparable only if ALL of these agree.
NS_DEFINITION_KEYS = ("metric", "days_window", "label_provenance",
                      "denominator", "lead_definition")

BANDIT_OOS_SCOPE = "out_of_sample_time_holdout"

# Definition changes that left no field in learning_progress rows. Points
# before an epoch start are not compared with points after it. A new change of
# this kind must add an entry here, with its evidence.
EPOCHS: Dict[str, Tuple[str, str]] = {
    "bandit": (
        "2026-09-22",
        "label store refreshed after its 08-16..09-21 freeze (0573da3); "
        "phantom_filter judges liveness from that store, so bandit training "
        "rows fell 11 110 -> 6 816 over 08-23..09-21 and were 11 618 on 09-22",
    ),
    "model": (
        "2026-09-22",
        "label store refreshed (0573da3): the top-gainer model had retrained on "
        "frozen labels for 34 nights; AUC 0.898 -> 0.848 that night",
    ),
}

# Minimum pooled n per compared half. continuous-improvement-agent-spec §11.1:
# at ~20 winners per window the North Star's minimum detectable difference is
# 0.127, 2.1x the metric's own level; below 20 not even that is reached.
TREND_MIN_N = 20
Z95 = 1.959964

TARGET_SPEC_NAME = "continuous-improvement-agent-spec.md"
TRADER_ROADMAP_NAME = "autonomous-trading-roadmap-1010-spec.md"
LEGACY_SPEC_NAME = "auto-improvement-loop-spec.md"


# ── Path resolution ───────────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    """Locate the repo root that actually holds .runtime data.

    The bot runs from the main repo; dev worktrees do not carry .runtime
    (it is gitignored). Probe a few candidates so the script works from
    either location.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent,                       # <repo>/files/.. -> <repo>
        Path("D:/Projects/claude_crypto_bot"),    # canonical main repo
    ]
    cur = Path.cwd()
    for _ in range(5):
        candidates.append(cur)
        cur = cur.parent
    for c in candidates:
        if (c / ".runtime" / "learning_progress.jsonl").exists():
            return c
    return candidates[0]


def _find_doc(repo_root: Path, name: str) -> Optional[Path]:
    """A spec from the analyzer's own checkout first (docs match this code),
    then from the data root."""
    for base in (Path(__file__).resolve().parent.parent, repo_root):
        p = base / "docs" / "specs" / "features" / name
        if p.exists():
            return p
    return None


def _find_spec(repo_root: Path) -> Optional[Path]:
    """The legacy as-built spec (reported as stale, never as the plan)."""
    return _find_doc(repo_root, LEGACY_SPEC_NAME)


def _git_last_commit_date(path: Optional[Path]) -> Optional[str]:
    """Date of the last commit touching `path`, or None if git cannot say."""
    if path is None:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(path.parent), "log", "-1", "--format=%cs", "--", path.name],
            capture_output=True, text=True, timeout=10)
        out = (proc.stdout or "").strip()
        return out if re.fullmatch(r"\d{4}-\d{2}-\d{2}", out) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_commits_since(repo_dir: Path, since: Optional[str], rel: str) -> Optional[int]:
    """Commits touching `rel` after the day `since`, or None if git cannot say."""
    if not since:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-list", "--count",
             f"--since={since}T23:59:59", "HEAD", "--", rel],
            capture_output=True, text=True, timeout=10)
        out = (proc.stdout or "").strip()
        return int(out) if proc.returncode == 0 and out.isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


# ── Generic helpers ───────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _iso_date(ts: Any) -> Optional[str]:
    s = str(ts or "")[:10]
    try:
        date.fromisoformat(s)
        return s
    except ValueError:
        return None


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def _days_since(day: Optional[str]) -> Optional[int]:
    if not day:
        return None
    try:
        return (datetime.now(timezone.utc).date() - date.fromisoformat(day[:10])).days
    except ValueError:
        return None


def _by_ts(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda r: str(r.get("ts", "")))


# ── Comparable windows and the verdict (spec §2.2, §2.3) ─────────────────────

def non_overlapping_chain(points: List[dict]) -> List[dict]:
    """Disjoint windows, walking back from the newest point.

    `points` are chronological; each carries `span = (first_day, last_day)` or
    None. A point without a span cannot be shown to be disjoint and is skipped.
    Returned chronological.
    """
    usable = [p for p in points if p.get("span")]
    if not usable:
        return []
    chain = [usable[-1]]
    for p in reversed(usable[:-1]):
        if p["span"][1] < chain[-1]["span"][0]:
            chain.append(p)
    return list(reversed(chain))


def compare_halves(chain: List[dict], *, min_n: int = TREND_MIN_N) -> Tuple[str, Dict[str, Any]]:
    """Early half vs late half of a chain of disjoint windows.

    For a mean of per-unit values in [0, 1] (NS, recall), p(1-p) bounds the
    variance from above, so the interval is conservative in the variance; units
    are treated as independent (not day-clustered), which is optimistic.
    """
    if len(chain) < 2:
        return "INSUFFICIENT", {
            "reason": f"{len(chain)} non-overlapping comparable window(s); need 2",
            "windows": len(chain)}
    h = len(chain) // 2
    early, late = chain[:h], chain[-h:]
    if any(p.get("n") is None for p in early + late):
        return "INSUFFICIENT", {"reason": "n not logged for every window",
                                "windows": len(chain)}

    def pool(ps: List[dict]) -> Tuple[Dict[str, Any], float]:
        n = sum(int(p["n"]) for p in ps)
        v = (sum(int(p["n"]) * float(p["value"]) for p in ps) / n) if n else 0.0
        return ({"from": ps[0]["date"], "to": ps[-1]["date"], "windows": len(ps),
                 "covers": [ps[0]["span"][0], ps[-1]["span"][1]],
                 "n": n, "value": round(v, 4)}, v)

    e, pe = pool(early)
    l, pl = pool(late)
    info: Dict[str, Any] = {"early": e, "late": l, "windows": len(chain),
                            "middle_dropped": len(chain) % 2 == 1}
    if e["n"] < min_n or l["n"] < min_n:
        info["reason"] = f"pooled n {e['n']} / {l['n']} < {min_n} per half"
        return "INSUFFICIENT", info
    pe, pl = min(1.0, max(0.0, pe)), min(1.0, max(0.0, pl))
    se = math.sqrt(pe * (1 - pe) / e["n"] + pl * (1 - pl) / l["n"])
    raw = pl - pe
    lo, hi = raw - Z95 * se, raw + Z95 * se
    info.update({
        # from the two printed values, so the printed numbers agree exactly
        "delta": round(l["value"] - e["value"], 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "method": ("disjoint windows pooled by n; 95% CI with the [0,1] "
                   "variance bound p(1-p); units assumed independent "
                   "(not day-clustered)"),
    })
    if lo > 0:
        return "IMPROVING", info
    if hi < 0:
        return "DEGRADING", info
    return "INCONCLUSIVE", info


def _diagnostic(points: List[Tuple[str, float]], epoch: str) -> Tuple[str, Dict[str, Any]]:
    """First and last of a comparable epoch, no verdict (no n / variance)."""
    if not points:
        return "UNKNOWN", {"reason": f"no comparable rows since {epoch}"}
    (d0, v0), (d1, v1) = points[0], points[-1]
    return "DIAGNOSTIC", {
        "first": round(v0, 4), "first_date": d0,
        "last": round(v1, 4), "last_date": d1,
        "last_minus_first": round(round(v1, 4) - round(v0, 4), 4),
        "points": len(points), "epoch_start": epoch,
    }


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_learning_progress(repo: Path) -> List[dict]:
    return _read_jsonl(repo / ".runtime" / "learning_progress.jsonl")


def load_metrics_daily(repo: Path) -> List[dict]:
    return _read_jsonl(repo / ".runtime" / "metrics_daily.jsonl")


def load_latest_health(repo: Path) -> Optional[dict]:
    hdir = repo / ".runtime" / "pipeline" / "health"
    if not hdir.exists():
        return None
    files = sorted(hdir.glob("health-*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_decisions(repo: Path) -> List[dict]:
    return _read_jsonl(repo / ".runtime" / "pipeline" / "decisions" / "decisions.jsonl")


def load_attribution_meta(repo: Path) -> Optional[dict]:
    adir = repo / ".runtime" / "pipeline" / "attribution"
    if not adir.exists():
        return None
    files = sorted(adir.glob("attribution-*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")).get("pipeline_meta")
    except (OSError, json.JSONDecodeError):
        return None


# ── North Star points (spec §2.1) ─────────────────────────────────────────────

def ns_points(md_rows: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """Comparable North Star points, one per UTC date, and why rows were left out."""
    excluded: Dict[str, int] = {}

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    rows = _by_ts(md_rows)
    ref: Optional[tuple] = None
    for r in reversed(rows):
        ec = r.get("_compute_early_capture.py") or {}
        if (ec.get("metric") == NS_METRIC
                and ec.get("label_provenance") == NS_LABEL_PROVENANCE
                and _safe(ec.get("early_capture")) is not None):
            ref = tuple(ec.get(k) for k in NS_DEFINITION_KEYS)
            break

    by_date: Dict[str, dict] = {}
    for r in rows:
        ec = r.get("_compute_early_capture.py") or {}
        day = _iso_date(r.get("ts"))
        value = _safe(ec.get("early_capture"))
        n = _safe(ec.get("n"))
        if day is None or value is None or not n or n <= 0:
            skip("no value (failed or empty run)")
            continue
        key = tuple(ec.get(k) for k in NS_DEFINITION_KEYS)
        if ref is None or key != ref:
            diff = [k for k, a, b in zip(NS_DEFINITION_KEYS, key, ref or key) if a != b]
            skip("definition differs: " + (", ".join(diff) or "no v2 reference row"))
            continue
        if ec.get("primary_degraded"):
            skip("primary degraded to the leaky label")
            continue
        store_last = _iso_date(ec.get("immutable_store_last_day"))
        if store_last is None:
            skip("label-store coverage not recorded (rows before 2026-09-22)")
            continue
        if store_last < _shift(day, -1):
            skip("label store did not cover the window")
            continue
        window = int(ec.get("days_window") or 0)
        v3 = ec.get("move_lead_early_capture")
        by_date[day] = {
            "date": day,
            "value": value,
            "n": int(n),
            "span": (_shift(day, -window), _shift(day, -1)) if window > 0 else None,
            "days_window": window,
            "days_full": ec.get("days_full"),
            "days_down_or_partial": ec.get("days_down_or_partial"),
            "coverage": _safe(ec.get("decomp_coverage")),
            "capture": _safe(ec.get("decomp_capture_mean")),
            "lead": _safe(ec.get("decomp_time_lead_mean")),
            "lead_definition": ec.get("lead_definition"),
            "v3_metric": ec.get("move_lead_metric"),
            "v3_value": _safe(v3),
            "v3_n": ec.get("move_lead_n"),
            "v3_without_deadline": ec.get("move_lead_winners_without_deadline"),
            "store_last_day": store_last,
        }
    return [by_date[d] for d in sorted(by_date)], excluded


# ── learning_progress series (spec §2.1) ──────────────────────────────────────

def _bandit_rows(lp: List[dict]) -> Tuple[List[dict], bool]:
    oos = [r for r in _by_ts(lp) if r.get("bandit_evaluation_scope") == BANDIT_OOS_SCOPE]
    if not oos:
        return [], False
    label = oos[-1].get("bandit_label")
    start = EPOCHS["bandit"][0]
    rows = [r for r in oos if r.get("bandit_label") == label
            and (_iso_date(r.get("ts")) or "") >= start]
    return rows, True


def recall_points(rows: List[dict]) -> List[dict]:
    by_date: Dict[str, dict] = {}
    for r in rows:
        day = _iso_date(r.get("ts"))
        v = _safe(r.get("bandit_recall_top20"))
        if day is None or v is None:
            continue
        first, last = r.get("bandit_eval_first_day"), r.get("bandit_eval_last_day")
        n = r.get("bandit_n_positive")
        by_date[day] = {
            "date": day, "value": v,
            "n": int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else None,
            "span": (str(first), str(last)) if first and last else None,
            "action_rate": _safe(r.get("bandit_action_rate")),
            "base_rate": _safe(r.get("bandit_top20_base_rate")),
            "lift": _safe(r.get("bandit_recall_lift")),
            "precision": _safe(r.get("bandit_precision")),
        }
    return [by_date[d] for d in sorted(by_date)]


def _model_rows(lp: List[dict]) -> Tuple[List[dict], bool]:
    imm = [r for r in _by_ts(lp)
           if str(r.get("model_label_timing") or "").startswith("immutable_later_eod")
           and "holdout" in str(r.get("model_evaluation_scope") or "")
           and _safe(r.get("model_auc_top20")) is not None]
    if not imm:
        return [], False
    ref = (imm[-1].get("model_evaluation_scope"), imm[-1].get("model_label_timing"))
    start = EPOCHS["model"][0]
    rows = [r for r in imm
            if (r.get("model_evaluation_scope"), r.get("model_label_timing")) == ref
            and (_iso_date(r.get("ts")) or "") >= start]
    return rows, True


def _series(rows: List[dict], field: str) -> List[Tuple[str, float]]:
    by_date: Dict[str, float] = {}
    for r in rows:
        day, v = _iso_date(r.get("ts")), _safe(r.get(field))
        if day is not None and v is not None:
            by_date[day] = v
    return [(d, by_date[d]) for d in sorted(by_date)]


# ── Roadmap specs (spec §2.4, Q5) ─────────────────────────────────────────────

def _section(lines: List[str], heading_re: str) -> List[str]:
    start = None
    for i, ln in enumerate(lines):
        if start is None:
            if re.match(heading_re, ln):
                start = i + 1
        elif re.match(r"^##\s", ln):
            return lines[start:i]
    return lines[start:] if start is not None else []


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*`~]", "", s)).strip()


def parse_target_spec(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\*\*Status:\*\*\s*([^\n]+)", text)
    status = _plain(m.group(1)) if m else "unknown"
    m = re.search(r"\*\*Revised:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    revised = m.group(1) if m else None
    phases: List[Tuple[str, str]] = []
    for ln in _section(text.splitlines(), r"^##\s+\d+\.\s+Rollout"):
        if not ln.startswith("|"):
            continue
        cells = _cells(ln)
        if len(cells) < 2:
            continue
        pid = _plain(cells[0])
        if not re.fullmatch(r"-?\d+[a-z]?", pid):
            continue
        name = _plain(cells[1])
        name = re.split(r"\. |: | — ", name, maxsplit=1)[0].rstrip(".")
        if len(name) > 34:
            name = name[:34].rsplit(" ", 1)[0].rstrip(",;") + "..."
        phases.append((pid, name))
    return {"available": True, "path": str(path), "status": status,
            "revised": revised, "phases": phases,
            "last_commit": _git_last_commit_date(path)}


def parse_trader_roadmap(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: List[Dict[str, str]] = []
    for ln in _section(text.splitlines(), r"^##\s+\d+\.\s+Спеки этапов"):
        if not ln.startswith("|"):
            continue
        cells = _cells(ln)
        if len(cells) < 4:
            continue
        m = re.search(r"`([^`]+-spec\.md)`", cells[1])
        if not m:
            continue
        b = re.search(r"\*\*(.+?)\*\*", cells[3])
        verdict = _plain(b.group(1)) if b else _plain(cells[3])[:60]
        rows.append({"num": _plain(cells[0]), "spec": m.group(1),
                     "stage": _plain(cells[2]), "verdict": verdict[:90]})
    last = _git_last_commit_date(path)
    return {"available": True, "path": str(path), "rows": rows, "last_commit": last,
            # how much trader work landed after the table last changed
            "trader_commits_since": _git_commits_since(path.parents[3], last, "trader")}


def parse_spec(spec_path: Optional[Path]) -> Dict[str, Any]:
    """The LEGACY as-built spec: only its status and date, to call it stale."""
    if spec_path is None or not spec_path.exists():
        return {"available": False}
    text = spec_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\*\*Last updated:\*\*\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    s = re.search(r"\*\*Status:\*\*\s*([^\n]+)", text)
    return {"available": True, "path": str(spec_path),
            "last_updated": m.group(1) if m else None,
            "status": _plain(s.group(1)) if s else "unknown"}


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(repo: Path, spec_path: Optional[Path]) -> Dict[str, Any]:
    lp = load_learning_progress(repo)
    md = load_metrics_daily(repo)
    health = load_latest_health(repo)
    decisions = load_decisions(repo)
    attr_meta = load_attribution_meta(repo)

    out: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ── Q2: learning diagnostics on comparable epochs ─────────────────────────
    ml: Dict[str, Any] = {"epochs": {k: {"start": v[0], "why": v[1]} for k, v in EPOCHS.items()}}
    if lp:
        lps = _by_ts(lp)
        ml["span"] = (f"{str(lps[0].get('ts', '?'))[:10]} -> "
                      f"{str(lps[-1].get('ts', '?'))[:10]} ({len(lps)} snapshots)")
        brows, has_oos = _bandit_rows(lps)
        rpts = recall_points(brows)
        rchain = non_overlapping_chain(rpts)
        if not has_oos:
            ml["recall_at_20"] = ("UNKNOWN", {"reason": "no OOS temporal holdout"})
        elif not rpts:
            ml["recall_at_20"] = ("INSUFFICIENT", {
                "reason": f"no OOS rows since the bandit epoch {EPOCHS['bandit'][0]}"})
        elif not rchain:
            ml["recall_at_20"] = ("INSUFFICIENT", {
                "reason": "evaluation n and window span not logged on these rows "
                          "(daily_learning logs them from 2026-11-19)"})
        else:
            ml["recall_at_20"] = compare_halves(rchain)
        ml["recall_latest"] = rpts[-1] if rpts else None
        ml["recall_chain"] = rchain
        ml["ucb_separation"] = _diagnostic(_series(brows, "bandit_ucb_separation"),
                                           EPOCHS["bandit"][0])
        mrows, has_imm = _model_rows(lps)
        ml["auc_top20"] = (
            _diagnostic(_series(mrows, "model_auc_top20"), EPOCHS["model"][0])
            if has_imm else ("UNKNOWN", {"reason": "no immutable later-EOD OOS holdout"}))
        counters: Dict[str, List[Optional[float]]] = {}
        for field in ("bandit_n_signal", "bandit_total_updates"):
            s = _series(brows, field)
            if s:
                counters[field] = [s[0][1], s[-1][1]]
        ml["counters"] = counters
    else:
        ml["span"] = "NO DATA"
    out["ml_history"] = ml

    # ── Q1: North Star on comparable, disjoint windows ────────────────────────
    ns_block: Dict[str, Any] = {}
    if health and isinstance(health.get("north_star"), dict):
        ns = health["north_star"]
        ns_block.update({
            "health_metric": ns.get("metric"),
            "latest_value": ns.get("value"),
            "status": ns.get("status"),
            "target": ns.get("target"),
            "acceptable_floor": ns.get("acceptable_floor"),
        })
    pts, excluded = ns_points(md)
    chain = non_overlapping_chain(pts)
    ns_block.update({"metric": NS_METRIC, "points": len(pts), "excluded": excluded,
                     "chain": chain, "latest": pts[-1] if pts else None,
                     "first_point": pts[0]["date"] if pts else None})
    status = ns_block.get("status")
    if status != "verified":
        reason = ("North-Star ground truth is provisional" if status in (None, "provisional")
                  else f"North-Star health status is '{status}', not verified")
        ns_block["trend"] = ("UNKNOWN", {"reason": reason})
    elif not pts:
        ns_block["trend"] = ("INSUFFICIENT", {"reason": "no comparable NS rows in metrics_daily"})
    else:
        ns_block["trend"] = compare_halves(chain)
    out["north_star"] = ns_block

    # ── Q3 + Q4: component data consumption + sufficiency ──────────────────────
    comp: Dict[str, Any] = {}
    if health:
        comp["training_health"] = health.get("training_health", {})
        comp["deployment_health"] = health.get("deployment_health", {})
        comp["training_to_live_gap"] = health.get("training_to_live_gap")
        comp["scout_health"] = health.get("scout_health", {})
        comp["exit_quality"] = health.get("exit_quality", {})
        comp["red_flags"] = health.get("red_flags", [])
        comp["data_sources"] = health.get("data_sources", {})
    if lp and len(lp) >= 2:
        comp["bandit_updates_growth"] = {
            "first": _safe(lp[0].get("bandit_total_updates")),
            "last": _safe(lp[-1].get("bandit_total_updates")),
        }
    if md:
        last = md[-1]
        funnel = last.get("_backtest_top20_coverage_funnel.py", {}) or {}
        prec = last.get("_backtest_signal_precision.py", {}) or {}
        comp["data_sufficiency"] = {
            "top20_label_status": ns_block.get("status") or "unknown",
            "top20_winners": funnel.get("n_top20_winners"),
            "coverage_pct_raw": funnel.get("coverage_pct_raw"),
            "silent_miss_pct": funnel.get("silent_miss_pct"),
            "entries_per_day": prec.get("raw_entries_per_day"),
            "precision_pct": prec.get("precision_pct"),
            "metrics_daily_age_days": _age_days(last.get("ts")),
        }
    out["components"] = comp

    # ── Q3: pipeline effectiveness ────────────────────────────────────────────
    pipe: Dict[str, Any] = {
        "n_decisions": len(decisions),
        "by_stage": {},
        "attribution_meta": attr_meta or {},
    }
    for d in decisions:
        st = d.get("stage", "unknown")
        pipe["by_stage"][st] = pipe["by_stage"].get(st, 0) + 1
    pend = sum(1 for d in decisions if d.get("validation_verdict") == "pending_manual_validation")
    pipe["pending_manual_validation"] = pend
    out["pipeline"] = pipe

    # ── Q5: roadmap from the target specs; the legacy matrix is only flagged ──
    out["roadmap"] = {
        "target": parse_target_spec(_find_doc(repo, TARGET_SPEC_NAME)),
        "trader": parse_trader_roadmap(_find_doc(repo, TRADER_ROADMAP_NAME)),
        "legacy": parse_spec(spec_path),
    }
    return out


def _age_days(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return round((datetime.now(timezone.utc) - t).total_seconds() / 86400.0, 1)
    except (ValueError, TypeError):
        return None


# ── Rendering ─────────────────────────────────────────────────────────────────

def _f(v: Any, nd: int = 4) -> str:
    x = _safe(v)
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(v: Any) -> str:
    x = _safe(v)
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _fmt_verdict(name: str, tv: Any) -> List[str]:
    if not isinstance(tv, (list, tuple)) or len(tv) != 2:
        return [f"  {name:15s}: n/a"]
    verdict, info = tv
    info = info or {}
    if verdict == "DIAGNOSTIC":
        return [f"  {name:15s}: DIAGNOSTIC, no verdict (no n / variance logged) - "
                f"first {info['first_date']}={_f(info['first'])} -> "
                f"last {info['last_date']}={_f(info['last'])}, "
                f"last-first={info['last_minus_first']:+.4f} "
                f"({info['points']} points since {info['epoch_start']})"]
    e, l = info.get("early"), info.get("late")
    halves = ""
    if e and l:
        halves = (f"early days {e['covers'][0]}..{e['covers'][1]} {_f(e['value'])} "
                  f"(n={e['n']}, {e['windows']} win) -> late days "
                  f"{l['covers'][0]}..{l['covers'][1]} {_f(l['value'])} "
                  f"(n={l['n']}, {l['windows']} win)")
    if verdict in ("INSUFFICIENT", "UNKNOWN"):
        lines = [f"  {name:15s}: {verdict} - {info.get('reason', '')}"]
        if halves:
            lines.append(f"  {'':15s}  {halves}")
        return lines
    tag = " (рано судить)" if verdict == "INCONCLUSIVE" else ""
    lo, hi = info["ci95"]
    return [f"  {name:15s}: {verdict}{tag}",
            f"  {'':15s}  {halves}",
            f"  {'':15s}  delta={info['delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"({info['method']})"]


def render(a: Dict[str, Any]) -> str:
    L: List[str] = []
    P = L.append
    P("=" * 78)
    P("STRATEGIC LEARNING-PROGRESS REPORT")
    P(f"generated: {a['generated_at']}")
    P("=" * 78)

    # Q1
    ns = a.get("north_star", {})
    lt = ns.get("latest")
    P("\n[Q1] IS THE BOT IMPROVING TOWARD ITS TARGET (NORTH STAR)?")
    if lt:
        P(f"  metric        : {ns.get('metric')} (immutable later-EOD labels, "
          f"{lt.get('lead_definition')} lead, {lt.get('days_window')}-day window)")
        P(f"  latest        : {lt['date']} = {_f(lt['value'])}  n={lt['n']} winner-days  "
          f"full days {lt.get('days_full')}/{lt.get('days_window')}  "
          f"days_down_or_partial={lt.get('days_down_or_partial')}")
        P(f"  decomposition : coverage={_f(lt.get('coverage'), 3)} "
          f"capture={_f(lt.get('capture'), 3)} lead={_f(lt.get('lead'), 3)}  "
          f"(capture, lead = means over entered winners; NS = mean of per-winner product)")
        v3 = ("not computable" if lt.get("v3_value") is None else _f(lt.get("v3_value")))
        P(f"  v3 move-lead  : {lt.get('v3_metric') or 'NS_EarlyCapture_top20_v3'} = {v3}  "
          f"move_lead_n={lt.get('v3_n')}  winners without deadline={lt.get('v3_without_deadline')}")
    else:
        P(f"  metric        : {ns.get('metric')} - no comparable point")
    P(f"  health        : status {ns.get('status')}, value {_f(ns.get('latest_value'))}, "
      f"target {ns.get('target')}, floor {ns.get('acceptable_floor')}")
    exc = ns.get("excluded") or {}
    P(f"  comparable    : {ns.get('points', 0)} dated points since {ns.get('first_point')}; "
      f"left out {sum(exc.values())} rows")
    for reason, cnt in sorted(exc.items(), key=lambda kv: -kv[1]):
        P(f"      {cnt:4d}  {reason}")
    chain = ns.get("chain") or []
    if chain:
        P("  disjoint windows (date = value, n winner-days, full days):")
        for p in chain:
            P(f"      {p['date']} = {_f(p['value'])}  n={p['n']}  "
              f"full {p.get('days_full')}/{p.get('days_window')}")
    L.extend(_fmt_verdict("NS trend", ns.get("trend")))

    # Q2
    ml = a.get("ml_history", {})
    P("\n[Q2] LEARNING DIAGNOSTICS (activity, not objective progress; TH-02)")
    P(f"  span          : {ml.get('span')}")
    for k, ep in (ml.get("epochs") or {}).items():
        P(f"  {k + ' epoch':14s}: since {ep['start']} - {ep['why']}")
    L.extend(_fmt_verdict("recall@20", ml.get("recall_at_20")))
    rl = ml.get("recall_latest")
    if rl:
        P(f"  {'':15s}  latest {rl['date']}: recall={_f(rl['value'])} "
          f"n={rl['n'] if rl['n'] is not None else '?'} ENTER={_pct(rl['action_rate'])} "
          f"base={_pct(rl['base_rate'])} lift={_f(rl['lift'], 2)} "
          f"precision={_f(rl['precision'], 3)}  (7-day OOS holdout)")
    for p in ml.get("recall_chain") or []:
        P(f"  {'':15s}  window {p['span'][0]}..{p['span'][1]}: recall={_f(p['value'])} "
          f"n={p['n'] if p['n'] is not None else '?'} ENTER={_pct(p['action_rate'])} "
          f"base={_pct(p['base_rate'])} lift={_f(p['lift'], 2)}")
    L.extend(_fmt_verdict("UCB separation", ml.get("ucb_separation")))
    L.extend(_fmt_verdict("AUC top20", ml.get("auc_top20")))
    cnt = ml.get("counters") or {}
    if cnt:
        P("  counters      : " + "; ".join(
            f"{k} {_f(v[0], 0)} -> {_f(v[1], 0)}" for k, v in cnt.items())
          + "  (counts, not progress; n_signal is capped)")

    # Q3
    comp = a.get("components", {})
    pipe = a.get("pipeline", {})
    P("\n[Q3] ARE ALL COMPONENTS USING DATA TO APPROACH THE TARGET?")
    th = comp.get("training_health", {})
    dh = comp.get("deployment_health", {})
    sh = comp.get("scout_health", {})
    P(f"  ML/bandit diag: recall={th.get('recall_at_20')} auc={th.get('auc')} "
      f"bandit_scope={th.get('evaluation_scope')} model_scope={th.get('model_evaluation_scope')} "
      f"updates={th.get('bandit_total_updates')} (loop fed: "
      f"{'YES' if (comp.get('bandit_updates_growth') or {}).get('last') else 'UNKNOWN'})")
    P(f"  deployment    : {json.dumps(dh, default=str)[:140]}")
    P(f"  scout         : {json.dumps(sh, default=str)[:140]}")
    P(f"  train-live gap: {comp.get('training_to_live_gap')}")
    P(f"  pipeline      : {pipe.get('n_decisions')} decisions {pipe.get('by_stage')}")
    P(f"  attribution   : {json.dumps(pipe.get('attribution_meta', {}), default=str)[:160]}")
    if pipe.get("pending_manual_validation"):
        P(f"  (!) pipeline blocker: {pipe['pending_manual_validation']} decisions stuck "
          f"at pending_manual_validation")

    # Q4
    ds = comp.get("data_sufficiency", {})
    P("\n[Q4] IS THERE ENOUGH DATA TO KEEP LEARNING?")
    P(f"  top20 labels  : {ds.get('top20_label_status', 'unknown')}")
    P(f"  top20 winners : {ds.get('top20_winners')}")
    P(f"  coverage %    : {ds.get('coverage_pct_raw')}")
    P(f"  silent miss % : {ds.get('silent_miss_pct')}")
    P(f"  entries/day   : {ds.get('entries_per_day')}")
    P(f"  precision %   : {ds.get('precision_pct')}")
    P(f"  metrics age   : {ds.get('metrics_daily_age_days')} days old "
      f"({'STALE' if (ds.get('metrics_daily_age_days') or 0) > 3 else 'fresh'})")
    rf = comp.get("red_flags", [])
    P(f"  red flags     : {len(rf)}")
    for f in rf:
        if isinstance(f, dict):
            P(f"      [{f.get('severity','?'):8s}] {f.get('id','?')} "
              f"= {f.get('value')} (thr {f.get('threshold')})")

    # Q5
    rm = a.get("roadmap", {})
    P("\n[Q5] ROADMAP STAGE — AND DOES IT NEED CORRECTING?")
    t = rm.get("target") or {}
    if t.get("available"):
        P(f"  signal bot    : target spec {TARGET_SPEC_NAME}")
        P(f"      status    : {t.get('status')}  (revised {t.get('revised')}, last commit "
          f"{t.get('last_commit') or 'unknown'}, {_days_since(t.get('last_commit') or t.get('revised'))} d ago)")
        P("      rollout   : " + " | ".join(f"{pid} {name}" for pid, name in t.get("phases", [])))
        P("      reached   : not computed - the spec records no per-phase status (unverified)")
    else:
        P(f"  signal bot    : target spec {TARGET_SPEC_NAME} NOT FOUND - Q5 unanswerable for the bot")
    tr = rm.get("trader") or {}
    if tr.get("available"):
        P(f"  trader        : {TRADER_ROADMAP_NAME} §7, last commit "
          f"{tr.get('last_commit') or 'unknown'} ({_days_since(tr.get('last_commit'))} d ago), "
          f"commits to trader/ since then: {tr.get('trader_commits_since', 'unknown')}")
        P("      (verdict = first bold phrase of each row's status cell)")
        for row in tr.get("rows", []):
            P(f"      #{row['num']:3s} {row['spec']} [{row['stage']}]: {row['verdict']}")
    else:
        P(f"  trader        : {TRADER_ROADMAP_NAME} NOT FOUND")
    lg = rm.get("legacy") or {}
    if lg.get("available"):
        status = re.split(r" — | - ", lg.get("status") or "", maxsplit=1)[0]
        P(f"  legacy matrix : {LEGACY_SPEC_NAME}: {status} - as-built record, STALE: "
          f"not the plan, not read for Q5/Q6; RM-4/RM-5/RM-18 are retired (CLAUDE.md §0)")

    # Q6
    P("\n[Q6] CONCRETE NEXT STEPS")
    for step in _next_steps(a):
        P(f"  - {step}")

    P("\n" + "=" * 78)
    return "\n".join(L)


def _next_steps(a: Dict[str, Any]) -> List[str]:
    steps: List[str] = []
    pipe = a.get("pipeline", {})
    rm = a.get("roadmap", {})
    ns = a.get("north_star", {})
    comp = a.get("components", {})

    if pipe.get("pending_manual_validation"):
        steps.append(
            f"Unblock pipeline: {pipe['pending_manual_validation']} decisions have no "
            f"validator verdict — audit validator coverage and the evidence each one needs")
    nst = ns.get("trend")
    if isinstance(nst, (list, tuple)) and len(nst) == 2:
        verdict, info = nst
        if verdict == "UNKNOWN":
            steps.append(
                "North Star ground truth is not verified — restore immutable later-EOD "
                "labels and recompute before judging progress")
        elif verdict == "INSUFFICIENT":
            steps.append(
                f"North Star: {info.get('reason')} — no trend claim; report the latest "
                f"value with its n and wait for more comparable windows")
        elif verdict == "INCONCLUSIVE":
            lo, hi = info["ci95"]
            steps.append(
                f"North Star change between disjoint windows is inside noise "
                f"(95% CI [{lo:+.3f}, {hi:+.3f}]) — say 'рано судить', not a trend")
        elif verdict == "DEGRADING":
            steps.append(
                f"North Star DEGRADING beyond noise (delta {info['delta']:+.4f}) — find "
                f"where winners are lost first (incident_analyst)")
    ds = comp.get("data_sufficiency", {})
    age = ds.get("metrics_daily_age_days")
    if age is not None and age > 3:
        steps.append(
            f"metrics_daily.jsonl is {age}d stale — canonical NS metric not refreshing; "
            f"check report_metrics_daily.py scheduling FIRST (blocks Q1 honesty)")
    sm = ds.get("silent_miss_pct")
    if sm is not None and sm > 10:
        steps.append(
            f"silent_miss {sm:.0f}% of top-20 winners never produced any event — "
            f"investigate watchlist coverage / signal generation, not just gates")
    t = rm.get("target") or {}
    if t.get("available"):
        steps.append(
            f"Q5 cannot name the bot's stage: {TARGET_SPEC_NAME} §13 records no per-phase "
            f"status — record it there so the analyzer can report it")
    if not steps:
        steps.append("No blocking issues detected")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Strategic learning-progress analyzer")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    repo = _find_repo_root()
    spec = _find_spec(repo)
    a = analyze(repo, spec)

    if args.json:
        print(json.dumps(a, indent=2, default=str))
    else:
        print(f"[*] repo root : {repo}")
        print(f"[*] specs     : {(_find_doc(repo, TARGET_SPEC_NAME) or 'target NOT FOUND')}")
        print(render(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
