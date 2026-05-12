"""L1 — Bot Health Report.

Сливает 5 источников в единый JSON+markdown с явным training-to-live gap
и traffic-light классификацией по north-star метрике.

Usage:
    pyembed\\python.exe files\\bot_health_report.py
    pyembed\\python.exe files\\bot_health_report.py --date 2026-05-11
    pyembed\\python.exe files\\bot_health_report.py --run-evaluator   # also run _run_signal_evaluator.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import pipeline_lib as PL

# ---------------------------------------------------------------------------
# Collectors — each returns a small dict ready to drop into the report
# ---------------------------------------------------------------------------


def collect_training_health(today: date, n_days_trend: int = 7) -> dict:
    """Read learning_progress.jsonl, take today + last N days for trend."""
    records = list(PL.iter_jsonl(PL.LEARNING_PROGRESS))
    if not records:
        return {"available": False}

    # Most recent record overall
    latest = records[-1]
    cutoff = today - timedelta(days=n_days_trend)
    trend = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if ts.date() >= cutoff:
            trend.append({
                "date": ts.date().isoformat(),
                "recall_at_20":     r.get("bandit_recall_top20"),
                "ucb_separation":   r.get("bandit_ucb_separation"),
                "auc":              r.get("model_auc_top20"),
                "bandit_n_signal":  r.get("bandit_n_signal"),
            })

    return {
        "available": True,
        "latest_ts": latest.get("ts"),
        "recall_at_20":    latest.get("bandit_recall_top20"),
        "ucb_separation":  latest.get("bandit_ucb_separation"),
        "auc":             latest.get("model_auc_top20"),
        "bandit_total_updates": latest.get("bandit_total_updates"),
        "bandit_n_signal":      latest.get("bandit_n_signal"),
        "n_top20_in_watchlist": latest.get("n_top20_in_watchlist"),
        "trend": trend,
    }


def collect_critic(today: date) -> dict:
    """Take latest critic snapshot for today (final preferred over midday)."""
    for phase in ("final", "midday"):
        p = PL.REPORTS / f"top_gainer_critic_{today.isoformat()}_{phase}.json"
        if p.exists():
            data = PL.read_json(p)
            if data:
                data["_phase_used"] = phase
                data["_source_file"] = str(p)
                return {"available": True, "data": data}
    return {"available": False}


def collect_critic_baseline(today: date, n_days: int = 7) -> dict:
    """7-day rolling baseline from top_gainer_critic_history.jsonl (final phase only)."""
    records = list(PL.iter_jsonl(PL.CRITIC_HISTORY))
    if not records:
        return {"available": False}

    cutoff = today - timedelta(days=n_days)
    capt, early, fp = [], [], []
    for r in records:
        try:
            d = date.fromisoformat(r["target_day_local"])
        except (KeyError, ValueError):
            continue
        if d < cutoff or d >= today:
            continue
        if r.get("phase") != "final":
            continue
        s = r.get("summary", {})
        # critic history stores values as percentages (33.33), normalize to ratios (0.3333)
        c_pct = s.get("watchlist_top_capture_rate_pct")
        e_pct = s.get("watchlist_top_early_capture_rate_pct")
        capt.append(c_pct / 100.0 if c_pct is not None else None)
        early.append(e_pct / 100.0 if e_pct is not None else None)
        fps = (s.get("bot_false_positive_buys") or 0) / (s.get("bot_unique_buys") or 1)
        fp.append(fps)

    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "available": True,
        "window_days": n_days,
        "n_days_present": sum(1 for x in capt if x is not None),
        "avg_watchlist_top_bought_pct":          _avg(capt),
        "avg_watchlist_top_early_capture_pct":   _avg(early),
        "avg_false_positive_rate":               _avg(fp),
    }


def collect_per_mode_signals() -> dict:
    """Read evaluation_output/per_mode/<mode>/report.json (latest run)."""
    if not PL.PER_MODE_DIR.exists():
        return {"available": False}
    modes = {}
    for mode_dir in sorted(PL.PER_MODE_DIR.iterdir()):
        if not mode_dir.is_dir():
            continue
        rpt = mode_dir / "report.json"
        data = PL.read_json(rpt)
        if not data:
            continue
        s = data.get("summary", {})
        modes[mode_dir.name] = {
            "miss_rate":               s.get("miss_rate"),
            "false_positive_rate":     s.get("false_positive_rate"),
            "median_buy_lateness_pct": s.get("median_buy_lateness_pct_of_move"),
            "median_capture_ratio":    s.get("median_capture_ratio"),
            "total_realized_pnl_pct":  s.get("total_realized_pnl_pct"),
            "alpha_vs_bh_pct":         s.get("alpha_vs_buy_and_hold_pct"),
            "win_rate":                s.get("win_rate"),
            "profit_factor":           s.get("profit_factor"),
            "_window_start": data.get("config", {}).get("window_start"),
            "_window_end":   data.get("config", {}).get("window_end"),
        }
    return {"available": bool(modes), "modes": modes}


def collect_metrics_daily_latest() -> dict:
    """Last entry of .runtime/metrics_daily.jsonl — north-star + backtest metrics."""
    records = list(PL.iter_jsonl(PL.METRICS_DAILY))
    if not records:
        return {"available": False}
    latest = records[-1]
    extract = {}
    for key, sub in latest.items():
        if not isinstance(sub, dict):
            continue
        if key == "ts":
            continue
        for metric_name, value in sub.items():
            if metric_name == "metric" and isinstance(value, str):
                extract[value] = sub
                break
    return {"available": True, "ts": latest.get("ts"), "metrics": extract}


def collect_scout_gates() -> dict:
    """Run analyze_blocked_gates.py and parse its table."""
    try:
        result = subprocess.run(
            [str(PL.PYEMBED), str(PL.FILES_DIR / "analyze_blocked_gates.py")],
            cwd=PL.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"available": False, "error": str(e)}

    if result.returncode != 0:
        return {"available": False, "error": result.stderr[:500]}

    # Parse table: action reason_code n avg_r5% win5% sharpe*sqrtN miss_vs_take
    gates = []
    take_baseline = None
    in_over_blocking = False
    over_blockers = []
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("Over-blocking candidates"):
            in_over_blocking = True
            continue
        if in_over_blocking:
            # `  entry_score                  n= 2055  miss=+0.092%  win%=48.6  Sh*sqN=+4.10`
            m = re.match(r"\s+(\S+)\s+n=\s*(\d+)\s+miss=([+\-\d.]+)%\s+win%=([\d.]+)\s+Sh\*sqN=([+\-\d.]+)", line)
            if m:
                over_blockers.append({
                    "gate":     m.group(1),
                    "n":        int(m.group(2)),
                    "miss_pct": float(m.group(3)),
                    "win_pct":  float(m.group(4)),
                    "sharpe":   float(m.group(5)),
                })
            continue
        # Main table: action reason_code n avg_r5 win5 sharpe miss_vs_take
        parts = line.split()
        if len(parts) >= 7 and parts[2].isdigit():
            try:
                row = {
                    "action":       parts[0],
                    "gate":         parts[1],
                    "n":            int(parts[2]),
                    "avg_r5_pct":   float(parts[3]),
                    "win_pct":      float(parts[4]),
                    "sharpe":       float(parts[5]),
                    "miss_vs_take": float(parts[6]),
                }
                gates.append(row)
                if row["gate"] == "take" and row["action"] == "take":
                    take_baseline = row["avg_r5_pct"]
            except ValueError:
                continue

    working_correctly = [g["gate"] for g in gates
                         if g["action"] == "blocked" and g["avg_r5_pct"] < -0.10 and g["n"] >= 50]

    return {
        "available": True,
        "take_baseline_r5_pct": take_baseline,
        "gates_count": len([g for g in gates if g["action"] == "blocked"]),
        "over_blocking": over_blockers,
        "working_correctly": working_correctly,
        "all_gates": gates,
    }


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def compute_training_to_live_gap(training: dict, deploy: dict) -> dict:
    tr = training.get("recall_at_20")
    lv = deploy.get("watchlist_top_bought_pct")
    if tr is None or lv is None:
        return {"available": False}
    gap = round(tr - lv, 4)
    severity = "critical" if gap > 0.40 else "red" if gap > 0.20 else "yellow" if gap > 0.10 else "green"
    interp = (
        f"Training показывает recall@20={tr:.0%}, "
        f"live captures {lv:.0%} top-gainers из watchlist — gap={gap:+.0%}. "
    )
    if gap > 0.40:
        interp += "Огромный разрыв: модель ловит сигналы, но downstream (filters/scoring/watchlist matching) их теряет."
    elif gap > 0.20:
        interp += "Существенный разрыв — фильтры съедают треть+ сигналов модели."
    elif gap > 0.10:
        interp += "Умеренный разрыв — приемлемо, но есть запас."
    else:
        interp += "Training и live согласованы."
    return {"available": True, "value": gap, "severity": severity, "interpretation": interp}


def compute_north_star(deploy: dict, baseline: dict) -> dict:
    """North-star = early_capture_rate. Regression = today vs 7-day avg."""
    today_val = deploy.get("watchlist_top_early_capture_pct")
    base_val = baseline.get("avg_watchlist_top_early_capture_pct") if baseline.get("available") else None
    regression = None
    if today_val is not None and base_val is not None:
        regression = round(today_val - base_val, 4)
    status = PL.classify(today_val, "watchlist_top_early_capture_pct")
    return {
        "metric": "watchlist_top_early_capture_pct",
        "value": today_val,
        "baseline_7d": base_val,
        "regression_vs_7d_avg": regression,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Red-flag detection
# ---------------------------------------------------------------------------


def detect_red_flags(deploy: dict, per_mode: dict, gap: dict, scout: dict, critic_raw: dict) -> list[dict]:
    flags = []

    # RF1 — Early capture rate
    ec = deploy.get("watchlist_top_early_capture_pct")
    if ec is not None and ec < PL.THRESHOLDS["watchlist_top_early_capture_pct"]["red"]:
        # Find concrete missed cases
        missed = []
        if critic_raw.get("available"):
            for item in critic_raw["data"].get("watchlist_top_gainers", []):
                if item.get("status") in ("blocked_rule", "no_signal"):
                    missed.append({
                        "symbol": item["symbol"],
                        "day_change_pct": item.get("day_change_pct"),
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                    })
        flags.append({
            "id": "RF_early_capture",
            "metric": "watchlist_top_early_capture_pct",
            "value": ec,
            "threshold": PL.THRESHOLDS["watchlist_top_early_capture_pct"]["red"],
            "severity": "critical" if ec < 0.10 else "red",
            "evidence": {"missed_top_gainers": missed[:5]},
            "root_cause_hypothesis": "Сочетание over-blocking фильтров (blocked_rule) и model misses (no_signal). Разделять по причине прежде, чем чинить.",
        })

    # RF2 — False positives
    fpr = deploy.get("false_positive_rate")
    if fpr is not None and fpr > PL.THRESHOLDS["false_positive_rate"]["red"]:
        # Group false positives by mode if per_mode data available
        mode_fpr = {m: v.get("false_positive_rate") for m, v in (per_mode.get("modes") or {}).items()
                    if v.get("false_positive_rate") is not None}
        worst_modes = sorted(mode_fpr.items(), key=lambda kv: kv[1], reverse=True)[:3]
        flags.append({
            "id": "RF_false_positive",
            "metric": "false_positive_rate",
            "value": fpr,
            "threshold": PL.THRESHOLDS["false_positive_rate"]["red"],
            "severity": "critical" if fpr > 0.80 else "red",
            "evidence": {
                "false_positive_symbols": (critic_raw.get("data") or {}).get("bot_false_positive_symbols", [])[:10],
                "worst_modes": [{"mode": m, "fpr": v} for m, v in worst_modes],
            },
            "root_cause_hypothesis": "Если FP сконцентрированы в одном mode — ужесточить proba threshold именно там.",
        })

    # RF3 — Training-to-live gap
    if gap.get("available") and gap.get("severity") in ("red", "critical"):
        flags.append({
            "id": "RF_training_live_gap",
            "metric": "training_to_live_gap",
            "value": gap["value"],
            "threshold": 0.20,
            "severity": gap["severity"],
            "evidence": {"interpretation": gap["interpretation"]},
            "root_cause_hypothesis": "Огромный gap = downstream filter problem, не model. Смотреть analyze_blocked_gates over-blocking.",
        })

    # RF4 — Per-mode losing modes
    if per_mode.get("available"):
        for m, v in per_mode["modes"].items():
            pnl = v.get("total_realized_pnl_pct")
            if pnl is not None and pnl < -3.0:
                flags.append({
                    "id": f"RF_losing_mode_{m}",
                    "metric": "total_realized_pnl_pct",
                    "value": pnl,
                    "threshold": -3.0,
                    "severity": "critical" if pnl < -5.0 else "red",
                    "evidence": {
                        "mode": m,
                        "fpr": v.get("false_positive_rate"),
                        "median_lateness_pct": v.get("median_buy_lateness_pct"),
                        "alpha_vs_bh": v.get("alpha_vs_bh_pct"),
                    },
                    "root_cause_hypothesis": f"Mode {m} убыточен — рассмотреть отключение или сужение proba range.",
                })

    # RF5 — Over-blocking gates (skip if in do_not_touch)
    dnt_gates = {g["name"] for g in PL.load_do_not_touch().get("gates", [])}
    for ob in (scout.get("over_blocking") or []):
        if ob["gate"] in dnt_gates:
            continue  # protected
        if ob["sharpe"] >= 2.0 and ob["miss_pct"] >= 0.10:
            flags.append({
                "id": f"RF_overblock_{ob['gate']}",
                "metric": f"gate_{ob['gate']}_miss_pct",
                "value": ob["miss_pct"],
                "threshold": 0.10,
                "severity": "red" if ob["sharpe"] < 3.0 else "critical",
                "evidence": ob,
                "root_cause_hypothesis": f"Gate '{ob['gate']}' блокирует prof events лучше take_baseline на {ob['miss_pct']:.2f}pp при Sharpe x sqrt(n) = {ob['sharpe']:.2f}.",
            })

    return flags


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------


def build_report(today: date) -> dict:
    training = collect_training_health(today)
    critic_raw = collect_critic(today)
    baseline = collect_critic_baseline(today)
    per_mode = collect_per_mode_signals()
    metrics_daily = collect_metrics_daily_latest()
    scout = collect_scout_gates()

    # Deployment health from critic
    deploy = {"available": False}
    if critic_raw.get("available"):
        s = critic_raw["data"].get("summary", {})
        bought = s.get("watchlist_top_bought", 0)
        total = s.get("watchlist_top_count") or 1
        early = s.get("watchlist_top_early_captured", 0)
        fp = s.get("bot_false_positive_buys", 0)
        buys = s.get("bot_unique_buys") or 1
        deploy = {
            "available": True,
            "phase": critic_raw["data"].get("_phase_used"),
            "watchlist_top_bought_pct":        round(bought / total, 4),
            "watchlist_top_early_capture_pct": round(early / total, 4),
            "false_positive_rate":             round(fp / buys, 4),
            "bot_unique_buys":                 buys,
            "watchlist_top_bought":            bought,
            "watchlist_top_early":             early,
            "watchlist_top_total":             total,
            "missed_count":                    s.get("watchlist_top_missed"),
            "false_positive_buys":             fp,
        }

    gap = compute_training_to_live_gap(training, deploy) if training.get("available") and deploy.get("available") else {"available": False}
    north_star = compute_north_star(deploy, baseline)
    red_flags = detect_red_flags(deploy, per_mode, gap, scout, critic_raw)
    dnt = PL.load_do_not_touch()

    report = {
        "report_id": f"health-{today.isoformat()}",
        "schema_version": 1,
        "generated_at": PL.utc_now_iso(),
        "target_date": today.isoformat(),
        "window_days": 1,
        "north_star": north_star,
        "training_health": {k: v for k, v in training.items() if k != "trend"} if training.get("available") else {"available": False},
        "training_health_trend": training.get("trend") if training.get("available") else [],
        "deployment_health": deploy,
        "training_to_live_gap": gap,
        "exit_quality": per_mode,
        "scout_health": {
            "take_baseline_r5_pct": scout.get("take_baseline_r5_pct"),
            "over_blocking": scout.get("over_blocking", []),
            "working_correctly": scout.get("working_correctly", []),
        } if scout.get("available") else {"available": False},
        "metrics_daily_latest": metrics_daily,
        "red_flags": red_flags,
        "do_not_touch": dnt,
        "data_sources": {
            "training":         str(PL.LEARNING_PROGRESS),
            "critic":           critic_raw["data"]["_source_file"] if critic_raw.get("available") else None,
            "critic_baseline":  str(PL.CRITIC_HISTORY),
            "per_mode_eval":    str(PL.PER_MODE_DIR),
            "metrics_daily":    str(PL.METRICS_DAILY),
            "scout":            "files/analyze_blocked_gates.py (subprocess)",
        },
    }
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(r: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# Bot Health Report — {r['target_date']}")
    add("")
    add(f"_Generated: {r['generated_at']}_")
    add("")

    ns = r["north_star"]
    add(f"## {PL.status_emoji(ns['status'])} North-Star: early_capture_rate")
    add("")
    add(f"- Value today: **{ns['value']:.1%}**" if ns["value"] is not None else "- Value today: ❓")
    add(f"- 7-day baseline: {ns['baseline_7d']:.1%}" if ns["baseline_7d"] is not None else "- 7-day baseline: ❓")
    if ns["regression_vs_7d_avg"] is not None:
        sign = "+" if ns["regression_vs_7d_avg"] >= 0 else ""
        add(f"- Regression vs baseline: **{sign}{ns['regression_vs_7d_avg']:.1%}**")
    add(f"- Status: **{ns['status'].upper()}**")
    add("")

    # Training-to-live gap (the most important section)
    gap = r["training_to_live_gap"]
    if gap.get("available"):
        add(f"## {PL.status_emoji(gap['severity'])} Training-to-Live Gap")
        add("")
        add(f"- Gap: **{gap['value']:+.1%}**")
        add(f"- {gap['interpretation']}")
        add("")

    # Deployment health
    dh = r["deployment_health"]
    if dh.get("available"):
        add("## Deployment Health (live bot)")
        add("")
        add("| Метрика | Значение | Норма | Статус |")
        add("|---------|----------|-------|--------|")
        add(f"| watchlist_top_bought | {dh['watchlist_top_bought']}/{dh['watchlist_top_total']} ({dh['watchlist_top_bought_pct']:.1%}) | ≥50% | {PL.status_emoji(PL.classify(dh['watchlist_top_bought_pct'], 'watchlist_top_bought_pct'))} |")
        add(f"| early_captures | {dh['watchlist_top_early']}/{dh['watchlist_top_total']} ({dh['watchlist_top_early_capture_pct']:.1%}) | ≥25% | {PL.status_emoji(PL.classify(dh['watchlist_top_early_capture_pct'], 'watchlist_top_early_capture_pct'))} |")
        add(f"| false_positive_rate | {dh['false_positive_buys']}/{dh['bot_unique_buys']} ({dh['false_positive_rate']:.1%}) | <50% | {PL.status_emoji(PL.classify(dh['false_positive_rate'], 'false_positive_rate'))} |")
        add("")

    # Training health
    th = r["training_health"]
    if th.get("available"):
        add("## Training Health (модель)")
        add("")
        add("| Метрика | Значение | Норма | Статус |")
        add("|---------|----------|-------|--------|")
        add(f"| recall@20 | {th['recall_at_20']:.1%} | ≥80% | {PL.status_emoji(PL.classify(th['recall_at_20'], 'recall_at_20'))} |")
        add(f"| UCB separation | {th['ucb_separation']:+.4f} | ≥+0.10 | {PL.status_emoji(PL.classify(th['ucb_separation'], 'ucb_separation'))} |")
        add(f"| AUC top20 | {th['auc']:.3f} | ≥0.90 | {PL.status_emoji(PL.classify(th['auc'], 'model_auc_top20'))} |")
        add(f"| bandit updates total | {th['bandit_total_updates']:,} | — | — |")
        add("")
        if r["training_health_trend"]:
            add("**Trend (последние 7 дней):**")
            add("")
            add("| Дата | Recall | UCB Sep | AUC |")
            add("|------|--------|---------|-----|")
            for d in r["training_health_trend"]:
                rec = f"{d['recall_at_20']:.0%}" if d['recall_at_20'] is not None else "—"
                sep = f"{d['ucb_separation']:+.3f}" if d['ucb_separation'] is not None else "—"
                auc = f"{d['auc']:.3f}" if d['auc'] is not None else "—"
                add(f"| {d['date']} | {rec} | {sep} | {auc} |")
            add("")

    # Exit quality
    eq = r["exit_quality"]
    if eq.get("available"):
        add("## Exit Quality (per mode, last 24h)")
        add("")
        add("| Mode | miss_rate | FPR | median_late_pct | capture_ratio | realized_pnl% | alpha_vs_bh% | win_rate | PF |")
        add("|------|-----------|-----|-----------------|---------------|---------------|--------------|----------|-----|")
        def _fmt(x, fmt="{:.1%}"):
            return fmt.format(x) if x is not None else "—"
        for m, v in eq["modes"].items():
            row = [
                m,
                _fmt(v["miss_rate"]),
                _fmt(v["false_positive_rate"]),
                _fmt(v["median_buy_lateness_pct"], "{:.1f}%"),
                _fmt(v["median_capture_ratio"]),
                _fmt(v["total_realized_pnl_pct"], "{:+.2f}%"),
                _fmt(v["alpha_vs_bh_pct"], "{:+.2f}%"),
                _fmt(v["win_rate"]),
                _fmt(v["profit_factor"], "{:.2f}"),
            ]
            add("| " + " | ".join(row) + " |")
        add("")

    # Scout health
    sh = r["scout_health"]
    if sh.get("available", True) and sh.get("take_baseline_r5_pct") is not None:
        add("## Scout Health (blocked gates)")
        add("")
        add(f"Take baseline: avg_r5 = **{sh['take_baseline_r5_pct']:.3f}%**")
        add("")
        if sh.get("over_blocking"):
            add("**Over-blocking candidates** (excluding do_not_touch):")
            add("")
            add("| Gate | n | miss vs take | win% | Sharpe×√n |")
            add("|------|---|-------------|------|-----------|")
            dnt = {g["name"] for g in r["do_not_touch"].get("gates", [])}
            for ob in sh["over_blocking"]:
                if ob["gate"] in dnt:
                    continue
                add(f"| {ob['gate']} | {ob['n']} | {ob['miss_pct']:+.3f}% | {ob['win_pct']:.1f} | {ob['sharpe']:+.2f} |")
            add("")
        if sh.get("working_correctly"):
            add(f"**Working correctly (do not touch):** {', '.join(sh['working_correctly'])}")
            add("")

    # Red flags
    if r["red_flags"]:
        add("## 🚨 Red Flags (требуют действия)")
        add("")
        for rf in r["red_flags"]:
            add(f"### {PL.status_emoji(rf['severity'])} {rf['id']}: {rf['metric']}")
            add("")
            add(f"- Value: **{rf['value']:.4f}** (threshold {rf['threshold']})")
            add(f"- Hypothesis: {rf['root_cause_hypothesis']}")
            ev = rf.get("evidence", {})
            if ev:
                add(f"- Evidence: `{json.dumps(ev, ensure_ascii=False)[:300]}`")
            add("")
    else:
        add("## ✅ No red flags")
        add("")

    add("---")
    add("")
    add("## Data sources")
    add("")
    for k, v in r["data_sources"].items():
        add(f"- **{k}**: `{v}`")
    add("")
    return "\n".join(lines)


def render_telegram(r: dict) -> str:
    """Short Telegram summary — single screen."""
    ns = r["north_star"]
    gap = r["training_to_live_gap"]
    dh = r["deployment_health"]
    th = r["training_health"]
    rf_count = len(r["red_flags"])

    out = []
    out.append(f"🩺 <b>Bot Health Report</b> — {r['target_date']}")
    out.append("")

    if ns["value"] is not None:
        reg = ""
        if ns["regression_vs_7d_avg"] is not None:
            sign = "+" if ns["regression_vs_7d_avg"] >= 0 else ""
            reg = f" ({sign}{ns['regression_vs_7d_avg']:.1%} vs 7d)"
        out.append(f"{PL.status_emoji(ns['status'])} <b>North-star early_capture</b>: {ns['value']:.1%}{reg}")
    if gap.get("available"):
        out.append(f"{PL.status_emoji(gap['severity'])} <b>Training-to-live gap</b>: {gap['value']:+.1%}")

    if dh.get("available"):
        out.append("")
        out.append("<b>Deployment:</b>")
        out.append(f"  • bought: {dh['watchlist_top_bought']}/{dh['watchlist_top_total']} ({dh['watchlist_top_bought_pct']:.0%})")
        out.append(f"  • early: {dh['watchlist_top_early']}/{dh['watchlist_top_total']} ({dh['watchlist_top_early_capture_pct']:.0%})")
        out.append(f"  • FPR: {dh['false_positive_buys']}/{dh['bot_unique_buys']} ({dh['false_positive_rate']:.0%})")

    if th.get("available"):
        out.append("")
        out.append("<b>Training:</b>")
        out.append(f"  • recall@20: {th['recall_at_20']:.0%} | UCB: {th['ucb_separation']:+.3f} | AUC: {th['auc']:.3f}")

    out.append("")
    if rf_count:
        out.append(f"🚨 <b>{rf_count} red flag(s)</b> — см. полный отчёт:")
        for rf in r["red_flags"][:3]:
            out.append(f"  • {rf['id']} ({rf['severity']})")
    else:
        out.append("✅ No red flags")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--run-evaluator", action="store_true",
                    help="run _run_signal_evaluator.py before report (slow, ~3min)")
    ap.add_argument("--print-telegram", action="store_true")
    ap.add_argument("--print-markdown", action="store_true")
    args = ap.parse_args()

    today = date.fromisoformat(args.date) if args.date else datetime.now(timezone.utc).date()

    if args.run_evaluator:
        print(f"[bot_health] running _run_signal_evaluator.py --window-days 1 --per-mode (slow)...", file=sys.stderr)
        subprocess.run(
            [str(PL.PYEMBED), str(PL.FILES_DIR / "_run_signal_evaluator.py"),
             "--window-days", "1", "--per-mode"],
            cwd=PL.REPO_ROOT, check=False,
        )

    report = build_report(today)

    PL.HEALTH.mkdir(parents=True, exist_ok=True)
    json_path = PL.HEALTH / f"health-{today.isoformat()}.json"
    md_path   = PL.HEALTH / f"health-{today.isoformat()}.md"
    tg_path   = PL.HEALTH / f"health-{today.isoformat()}.tg.txt"

    PL.write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    tg_path.write_text(render_telegram(report), encoding="utf-8")

    print(f"[bot_health] wrote {json_path}", file=sys.stderr)
    print(f"[bot_health] wrote {md_path}", file=sys.stderr)
    print(f"[bot_health] wrote {tg_path}", file=sys.stderr)

    if args.print_telegram:
        print(tg_path.read_text(encoding="utf-8"))
    elif args.print_markdown:
        print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
