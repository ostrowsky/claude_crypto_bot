"""Strategic learning-progress analyzer for the crypto bot.

Single token-efficient entry point that answers six recurring questions
without the operator (or Claude) having to read large jsonl files:

  Q1. Is the bot improving toward its target (North Star) metric?
  Q2. Progress or degradation? (trend over the longest available history)
  Q3. Are all analytical components (ML / bandit / scout / pipeline)
      actually consuming data to move toward the target?
  Q4. Is there enough data to keep learning?
  Q5. What roadmap stage are we at, and does it need correcting?
  Q6. What are the concrete next steps?

It aggregates:
  - .runtime/learning_progress.jsonl     (ML/bandit daily history, ~longest)
  - .runtime/metrics_daily.jsonl         (canonical business metrics inc. NS)
  - .runtime/pipeline/health/health-*.json (pre-aggregated daily health)
  - .runtime/pipeline/decisions/decisions.jsonl (pipeline activity)
  - .runtime/pipeline/attribution/*.json (pipeline hit-rate / effect)
  - docs/specs/features/auto-improvement-loop-spec.md (roadmap + matrix)

Output is a compact structured text report. No file writes, no network.

Usage:
    pyembed\\python.exe files\\analyze_learning_progress.py
    pyembed\\python.exe files\\analyze_learning_progress.py --json   # machine form
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
    # Also walk up from cwd
    cur = Path.cwd()
    for _ in range(5):
        candidates.append(cur)
        cur = cur.parent
    for c in candidates:
        if (c / ".runtime" / "learning_progress.jsonl").exists():
            return c
    # Fall back to the first candidate even if data is absent
    return candidates[0]


def _find_spec(repo_root: Path) -> Optional[Path]:
    """Spec may live in a worktree even when data lives in the main repo."""
    direct = repo_root / "docs" / "specs" / "features" / "auto-improvement-loop-spec.md"
    if direct.exists():
        return direct
    wt = repo_root / ".claude" / "worktrees"
    if wt.exists():
        hits = sorted(wt.glob("*/docs/specs/features/auto-improvement-loop-spec.md"))
        if hits:
            # newest by mtime
            return max(hits, key=lambda p: p.stat().st_mtime)
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
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _slope(values: List[float]) -> float:
    """Least-squares slope per step. 0 if <2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _trend_verdict(values: List[float], *, eps: float, higher_is_better: bool = True) -> Tuple[str, Dict[str, float]]:
    """Classify a series as IMPROVING / FLAT / DEGRADING.

    Uses both the regression slope and a first-third vs last-third mean
    delta so a noisy-but-trending series is still caught.
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 3:
        return "INSUFFICIENT", {"n": len(clean)}
    k = max(1, len(clean) // 3)
    early = sum(clean[:k]) / k
    late = sum(clean[-k:]) / k
    delta = late - early
    slope = _slope(clean)
    info = {
        "n": float(len(clean)),
        "first": round(clean[0], 4),
        "last": round(clean[-1], 4),
        "early_mean": round(early, 4),
        "late_mean": round(late, 4),
        "delta": round(delta, 4),
        "slope_per_step": round(slope, 6),
    }
    signed = delta if higher_is_better else -delta
    if signed > eps:
        return "IMPROVING", info
    if signed < -eps:
        return "DEGRADING", info
    return "FLAT", info


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


# ── Spec parsing (roadmap + component matrix) ─────────────────────────────────

def parse_spec(spec_path: Optional[Path]) -> Dict[str, Any]:
    if spec_path is None or not spec_path.exists():
        return {"available": False}
    text = spec_path.read_text(encoding="utf-8", errors="replace")

    # Component matrix status tally across all "| ID | ... | Status |" rows
    status_counts = {"done": 0, "partial": 0, "missing": 0, "deferred": 0}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        joined = " ".join(cells)
        # Only count rows that look like matrix entries (have an ID like L1-a, RM-3, B-f)
        if not re.match(r"^[A-Z]{1,3}-?\w", cells[0]):
            continue
        if "✅" in joined:
            status_counts["done"] += 1
        elif "🟡" in joined:
            status_counts["partial"] += 1
        elif "❌" in joined:
            status_counts["missing"] += 1
        elif "⏸" in joined:
            status_counts["deferred"] += 1

    # Roadmap items: RM-N rows. Struck-through (~~RM-N~~) or "✅ DONE" => done.
    # IMPORTANT: only scan the "## 5. Roadmap" section. The §4 North Star
    # progress table also references RM-N in prose and would otherwise be
    # misparsed as roadmap rows.
    lines = text.splitlines()
    roadmap_start = roadmap_end = None
    for idx, ln in enumerate(lines):
        if roadmap_start is None and re.match(r"^##\s+5\.\s+Roadmap", ln):
            roadmap_start = idx
            continue
        if roadmap_start is not None and re.match(r"^##\s+(5b|5c|6)\b", ln):
            roadmap_end = idx
            break
    roadmap_lines = (
        lines[roadmap_start:roadmap_end] if roadmap_start is not None else []
    )

    rm_done: List[str] = []
    rm_pending: List[Tuple[str, str]] = []
    for line in roadmap_lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        id_cell = cells[0]
        m = re.search(r"\b(RM-\d+)\b", id_cell)
        if not m:
            continue
        rm = m.group(1)
        is_done = ("~~" in id_cell) or "✅" in line or "DONE" in line.upper()
        if is_done:
            if rm not in rm_done:
                rm_done.append(rm)
        else:
            desc = re.sub(r"\s+", " ", cells[1]) if len(cells) > 1 else ""
            desc = re.sub(r"[*~`]", "", desc)
            desc = desc.encode("ascii", "replace").decode("ascii")[:90]
            if rm not in [r for r, _ in rm_pending] and rm not in rm_done:
                rm_pending.append((rm, desc))

    # North Star progress table: rows with a date in col 1
    ns_rows: List[List[str]] = []
    for line in text.splitlines():
        if line.startswith("| 20") and "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5:
                ns_rows.append(cells)

    # "Last updated" header
    m = re.search(r"\*\*Last updated:\*\*\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[^\n]*)", text)
    last_updated = m.group(1).strip() if m else "unknown"

    return {
        "available": True,
        "path": str(spec_path),
        "last_updated": last_updated,
        "matrix": status_counts,
        "roadmap_done": rm_done,
        "roadmap_pending": rm_pending,
        "north_star_rows": ns_rows[-6:],
    }


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(repo: Path, spec_path: Optional[Path]) -> Dict[str, Any]:
    lp = load_learning_progress(repo)
    md = load_metrics_daily(repo)
    health = load_latest_health(repo)
    decisions = load_decisions(repo)
    attr_meta = load_attribution_meta(repo)
    spec = parse_spec(spec_path)

    out: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ── Q2: long-term ML/bandit trends ────────────────────────────────────────
    if lp:
        span = f"{lp[0].get('ts','?')[:10]} -> {lp[-1].get('ts','?')[:10]} ({len(lp)} snapshots)"
        recall = [_safe(r.get("bandit_recall_top20")) for r in lp]
        ucb = [_safe(r.get("bandit_ucb_separation")) for r in lp]
        auc = [_safe(r.get("model_auc_top20")) for r in lp]
        n_signal = [_safe(r.get("bandit_n_signal")) for r in lp]
        updates = [_safe(r.get("bandit_total_updates")) for r in lp]
        out["ml_history"] = {
            "span": span,
            "recall_at_20": _trend_verdict([v for v in recall if v is not None], eps=0.01),
            "ucb_separation": _trend_verdict([v for v in ucb if v is not None], eps=0.01),
            "auc_top20": _trend_verdict([v for v in auc if v is not None], eps=0.02),
            "n_signal": _trend_verdict([v for v in n_signal if v is not None], eps=1.0),
            "total_updates_first": updates[0] if updates and updates[0] else None,
            "total_updates_last": updates[-1] if updates and updates[-1] else None,
        }
    else:
        out["ml_history"] = {"span": "NO DATA"}

    # ── Q1: North Star trajectory ─────────────────────────────────────────────
    ns_series: List[Tuple[str, float]] = []
    for row in md:
        ec = row.get("_compute_early_capture.py", {})
        v = _safe(ec.get("early_capture"))
        if v is not None:
            ns_series.append((row.get("ts", "?")[:10], v))
    ns_block: Dict[str, Any] = {}
    if health and isinstance(health.get("north_star"), dict):
        ns = health["north_star"]
        ns_block.update({
            "metric": ns.get("metric"),
            "latest_value": ns.get("value"),
            "baseline_7d": ns.get("baseline_7d"),
            "status": ns.get("status"),
        })
    if ns_series:
        ns_block["history"] = [{"date": d, "early_capture": round(v, 4)} for d, v in ns_series]
        ns_block["trend"] = _trend_verdict([v for _, v in ns_series], eps=0.01)
    else:
        ns_block["trend"] = ("INSUFFICIENT", {"note": "metrics_daily has too few NS points"})
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
    # Dataset growth from learning_progress (proves ML loop is fed)
    if lp and len(lp) >= 2:
        comp["bandit_updates_growth"] = {
            "first": _safe(lp[0].get("bandit_total_updates")),
            "last": _safe(lp[-1].get("bandit_total_updates")),
            "n_signal_first": _safe(lp[0].get("bandit_n_signal")),
            "n_signal_last": _safe(lp[-1].get("bandit_n_signal")),
        }
    # Coverage / data sufficiency from canonical metrics
    if md:
        last = md[-1]
        funnel = last.get("_backtest_top20_coverage_funnel.py", {})
        prec = last.get("_backtest_signal_precision.py", {})
        comp["data_sufficiency"] = {
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
    # Detect the structural blocker pattern (pending_manual_validation)
    pend = sum(1 for d in decisions if d.get("validation_verdict") == "pending_manual_validation")
    pipe["pending_manual_validation"] = pend
    out["pipeline"] = pipe

    # ── Q5 + Q6: roadmap state ────────────────────────────────────────────────
    out["roadmap"] = spec

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

def _fmt_trend(name: str, tv: Any) -> str:
    if not isinstance(tv, (list, tuple)) or len(tv) != 2:
        return f"  {name:22s}: n/a"
    verdict, info = tv
    if verdict == "INSUFFICIENT":
        return f"  {name:22s}: INSUFFICIENT ({info})"
    return (f"  {name:22s}: {verdict:10s} "
            f"first={info.get('first')} last={info.get('last')} "
            f"delta={info.get('delta'):+}")


def render(a: Dict[str, Any]) -> str:
    L: List[str] = []
    P = L.append
    P("=" * 78)
    P("STRATEGIC LEARNING-PROGRESS REPORT")
    P(f"generated: {a['generated_at']}")
    P("=" * 78)

    # Q1
    ns = a.get("north_star", {})
    P("\n[Q1] IS THE BOT IMPROVING TOWARD ITS TARGET (NORTH STAR)?")
    P(f"  metric        : {ns.get('metric', 'watchlist_top_early_capture_pct')}")
    P(f"  latest value  : {ns.get('latest_value')}")
    P(f"  baseline_7d   : {ns.get('baseline_7d')}")
    P(f"  status        : {ns.get('status')}")
    P(_fmt_trend("NS early_capture", ns.get("trend")))
    if ns.get("history"):
        hist = ", ".join(f"{h['date']}={h['early_capture']}" for h in ns["history"])
        P(f"  history       : {hist}")

    # Q2
    ml = a.get("ml_history", {})
    P("\n[Q2] PROGRESS OR DEGRADATION? (longest available ML/bandit history)")
    P(f"  span          : {ml.get('span')}")
    P(_fmt_trend("recall@20", ml.get("recall_at_20")))
    P(_fmt_trend("UCB separation", ml.get("ucb_separation")))
    P(_fmt_trend("AUC top20", ml.get("auc_top20")))
    P(_fmt_trend("bandit n_signal", ml.get("n_signal")))
    P(f"  bandit updates: {ml.get('total_updates_first')} -> {ml.get('total_updates_last')}")

    # Q3
    comp = a.get("components", {})
    pipe = a.get("pipeline", {})
    P("\n[Q3] ARE ALL COMPONENTS USING DATA TO APPROACH THE TARGET?")
    th = comp.get("training_health", {})
    dh = comp.get("deployment_health", {})
    sh = comp.get("scout_health", {})
    P(f"  ML/bandit     : recall={th.get('recall_at_20')} auc={th.get('auc')} "
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
    if rm.get("available"):
        mx = rm.get("matrix", {})
        total = sum(mx.values()) or 1
        P(f"  spec updated  : {rm.get('last_updated')}")
        P(f"  component mtx : OK={mx.get('done',0)} partial={mx.get('partial',0)} "
          f"missing={mx.get('missing',0)} deferred={mx.get('deferred',0)} "
          f"({100*mx.get('done',0)//total}% done)")
        P(f"  roadmap done  : {', '.join(rm.get('roadmap_done', [])) or 'none'}")
        pend = rm.get("roadmap_pending", [])
        P(f"  roadmap todo  : {len(pend)} items")
        for rid, desc in pend[:6]:
            P(f"      - {rid}: {desc}")
    else:
        P("  spec NOT FOUND — cannot assess roadmap position")

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
            f"Unblock pipeline: {pipe['pending_manual_validation']} decisions stuck — "
            f"RM-1/RM-2 (blocked-event logging) must reach production bot to enable "
            f"honest Pareto sweeps")
    pend = rm.get("roadmap_pending", []) if rm.get("available") else []
    if pend:
        nxt = ", ".join(f"{r}" for r, _ in pend[:3])
        steps.append(f"Next roadmap items in priority order: {nxt}")
    nst = ns.get("trend")
    if isinstance(nst, (list, tuple)) and nst[0] in ("INSUFFICIENT", "DEGRADING"):
        steps.append(
            "North Star has too few/declining measurements — ensure metrics_daily "
            "(EOD job) runs daily; current NS history is the weakest link for Q1/Q2")
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
    if not steps:
        steps.append("No blocking issues detected — continue current roadmap sprint")
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
        print(f"[*] spec      : {spec if spec else 'NOT FOUND'}")
        print(render(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
