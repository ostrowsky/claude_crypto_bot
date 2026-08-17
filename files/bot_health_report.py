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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
            ratio_context_complete = all(
                r.get(key) is not None
                for key in ("bandit_action_rate", "bandit_top20_base_rate",
                            "bandit_recall_lift", "bandit_precision")
            )
            trend.append({
                "date": ts.date().isoformat(),
                # Legacy progress rows contained a post-fit recall without the
                # base/action rate. Suppress it rather than laundering it into
                # evidence through this report.
                "recall_at_20":     r.get("bandit_recall_top20") if ratio_context_complete else None,
                "ucb_separation":   r.get("bandit_ucb_separation"),
                "auc":              r.get("model_auc_top20"),
                "bandit_n_signal":  r.get("bandit_n_signal"),
                "evaluation_scope": r.get("bandit_evaluation_scope"),
                "action_rate":      r.get("bandit_action_rate"),
                "base_rate":        r.get("bandit_top20_base_rate"),
                "lift":             r.get("bandit_recall_lift"),
                "ratio_context_complete": ratio_context_complete,
            })

    latest_ratio_context_complete = all(
        latest.get(key) is not None
        for key in ("bandit_action_rate", "bandit_top20_base_rate",
                    "bandit_recall_lift", "bandit_precision")
    )
    return {
        "available": True,
        "latest_ts": latest.get("ts"),
        "recall_at_20":    latest.get("bandit_recall_top20") if latest_ratio_context_complete else None,
        "legacy_ratio_suppressed": not latest_ratio_context_complete
                                   and latest.get("bandit_recall_top20") is not None,
        "ratio_context_complete": latest_ratio_context_complete,
        "ucb_separation":  latest.get("bandit_ucb_separation"),
        "auc":             latest.get("model_auc_top20"),
        "bandit_total_updates": latest.get("bandit_total_updates"),
        "bandit_n_signal":      latest.get("bandit_n_signal"),
        "n_top20_in_watchlist": latest.get("n_top20_in_watchlist"),
        "evaluation_scope": latest.get("bandit_evaluation_scope"),
        "action_rate": latest.get("bandit_action_rate"),
        "base_rate": latest.get("bandit_top20_base_rate"),
        "lift": latest.get("bandit_recall_lift"),
        "precision": latest.get("bandit_precision"),
        "model_evaluation_scope": latest.get("model_evaluation_scope"),
        "model_label_timing": latest.get("model_label_timing"),
        "model_label_encoding_features": latest.get("model_label_encoding_features"),
        "trend": trend,
    }


def collect_critic(today: date, max_final_age_days: int = 3) -> dict:
    """Take the best critic available for an operational morning report.

    Today's final is preferred, then today's midday.  Before either exists,
    fall back to the latest *completed* final critic rather than silently
    disabling all deployment checks.  The age metadata lets red-flag logic
    distinguish the expected previous-day final from stale evidence.
    """
    candidates = [(today, "final"), (today, "midday")]
    candidates.extend((today - timedelta(days=age), "final")
                      for age in range(1, max_final_age_days + 1))
    for critic_day, phase in candidates:
        p = PL.REPORTS / f"top_gainer_critic_{critic_day.isoformat()}_{phase}.json"
        if p.exists():
            data = PL.read_json(p)
            if data:
                data["_phase_used"] = phase
                data["_source_file"] = str(p)
                data["_critic_target_date"] = str(
                    data.get("target_day_local") or critic_day.isoformat())
                data["_fallback_days"] = (today - critic_day).days
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


def collect_mode_curtail() -> dict:
    """Current impulse_speed regime-curtail state (auto-revive switch)."""
    try:
        import impulse_speed_curtail as ic
        if not bool(getattr(__import__("config"),
                            "IMPULSE_SPEED_REGIME_CURTAIL_ENABLED", False)):
            return {"available": True, "enabled": False}
        rec = {}
        if ic.STATE_FILE.exists():
            rec = json.loads(ic.STATE_FILE.read_text(encoding="utf-8"))
        return {
            "available": True,
            "enabled": True,
            "curtailed": bool(rec.get("curtailed", False)),
            "trailing_mean_pnl": rec.get("trailing_mean_pnl"),
            "window_days": rec.get("window_days"),
            "n_trades": rec.get("n_trades"),
            "computed_at": rec.get("computed_at"),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


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
    scope = training.get("evaluation_scope")
    if scope != "out_of_sample_time_holdout":
        return {
            "available": False,
            "reason": "training_metric_not_out_of_sample",
            "evaluation_scope": scope or "unknown",
        }
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



def _portfolio_alpha_entry() -> dict:
    """TH-11 canonical profitability, computed rather than declared unknown.

    Stays `unknown` when it genuinely cannot be computed — a window with no
    closed trades is not 0% alpha. It is reported as a DIAGNOSTIC: the bot is an
    alert system with no position sizing, so this answers "is the stream worth
    acting on", not "what should be maximised".
    """
    try:
        import portfolio_alpha
        res = portfolio_alpha.compute(30)
    except Exception as exc:                       # never break the report
        return {"value": None, "target": 0.0, "unit": "pct", "status": "unknown",
                "reason": f"portfolio_alpha failed: {type(exc).__name__}",
                "source": "portfolio_alpha.compute"}
    if not res.get("available"):
        return {"value": None, "target": 0.0, "unit": "pct", "status": "unknown",
                "reason": res.get("reason", "not computable"),
                "source": "portfolio_alpha.compute"}
    return {
        "value": res["alpha_vs_buy_and_hold_pct"], "target": 0.0, "unit": "pct",
        "status": "measured",
        "role": "diagnostic",
        "n": res["n_trades"], "window": res["window"],
        "bot_return_pct": res["bot_return_pct"],
        "buy_and_hold_pct": res["buy_and_hold_pct"],
        "win_rate_pct": res["win_rate_pct"],
        "source": "portfolio_alpha.compute (MAX_OPEN equal slots, closed trades)",
    }


def build_canonical_scorecard(metrics_daily: dict) -> dict:
    """One current value per canonical business question.

    Units are explicit because the underlying scripts mix ratios, percentages
    and hours. Unknown canonical evidence remains unknown; a nearby proxy is
    never substituted.
    """
    md = (metrics_daily or {}).get("metrics") or {}
    ns = md.get("NS_EarlyCapture_top20") or {}
    precision = md.get("D1_D2_precision_msgrate") or {}
    tts = md.get("E1_time_to_signal") or {}
    ex1 = (md.get("EX1_realized_potential") or {}).get("top20") or {}
    fr = md.get("Q1_Q3_fast_reversal") or {}
    whipsaw = md.get("Q2_whipsaw_rate") or {}
    return {
        "north_star": {
            "value": ns.get("early_capture"), "target": 0.40,
            "acceptable_floor": 0.25, "unit": "ratio",
            "status": "provisional",
            "provenance": "same_snapshot_rolling_24h_label; not immutable later EOD truth",
            "n": ns.get("n"), "days_window": ns.get("days_window"),
            "days_full": ns.get("days_full"),
            "definition": "mean(coverage * realized_capture * time_lead) on watchlist∩global-top20 winner-days",
            "source": "NS_EarlyCapture_top20",
        },
        "portfolio_alpha": _portfolio_alpha_entry(),
        "signal_precision": {
            "value": precision.get("precision_pct"), "target": 35.0,
            "unit": "pct", "n": precision.get("n_unique_entries"),
            "hits": precision.get("n_top20_entries"),
            "status": "provisional",
            "provenance": "top20 membership uses same-snapshot rolling-24h labels",
            "source": "D1_D2_precision_msgrate",
        },
        "message_rate": {
            "value": precision.get("unique_entries_per_day"), "target_max": 10.0,
            "unit": "per_day", "days": precision.get("n_days"),
            "source": "D1_D2_precision_msgrate",
        },
        "time_to_signal": {
            "value": tts.get("median_h"), "target_max": 0.5,
            "unit": "hours", "n": tts.get("n"),
            "status": "provisional",
            "provenance": "winner set uses same-snapshot rolling-24h labels",
            "source": "E1_time_to_signal",
        },
        "realized_potential": {
            "value": None, "diagnostic_proxy_value": ex1.get("median"),
            "target": 0.50,
            "unit": "ratio", "n": ex1.get("n"),
            "status": "unknown",
            "reason": "daily aggregator runs deprecated proxy-mode, not canonical --use-zigzag mode",
            "source": "EX1_realized_potential.top20",
        },
        "fast_reversal": {
            "value": fr.get("fr_v1_overall_pct"), "target_max": 8.0,
            "unit": "pct", "n": fr.get("n_total_pairs"),
            "source": "Q1_Q3_fast_reversal",
        },
        "whipsaw": {
            "value": whipsaw.get("overall_pct"), "target_max": 5.0,
            "unit": "pct", "n": whipsaw.get("n_total"),
            "source": "Q2_whipsaw_rate",
        },
    }


def derive_next_steps(scorecard: dict, training: dict, dnt: dict,
                      today: date) -> list[dict]:
    """Evidence-ranked work; never emits an unvalidated production change."""
    steps: list[dict] = []
    ns = scorecard.get("north_star") or {}
    if ns.get("status") != "verified":
        steps.append({
            "priority": "P0", "id": "restore_eod_ground_truth",
            "action": "Создать неизменяемые later-EOD top-20 labels и пересчитать всю историю North Star",
            "evidence": ns.get("provenance") or "North-Star label provenance is unverified",
            "gate": "Текущий North Star считать предварительным, а не доказательством прогресса",
        })
    if ns.get("days_full") is not None and ns.get("days_window") is not None \
            and ns["days_full"] < ns["days_window"]:
        steps.append({
            "priority": "P0", "id": "restore_measurement_coverage",
            "action": "Восстановить полные рабочие дни и наполнить сопоставимое 14-дневное окно",
            "evidence": f"days_full={ns['days_full']}/{ns['days_window']}",
            "gate": "Не оценивать тренд, пока denominators окон несопоставимы",
        })
    if training.get("evaluation_scope") != "out_of_sample_time_holdout":
        steps.append({
            "priority": "P0", "id": "repair_training_evaluation",
            "action": "Построить later-EOD labels и temporal holdout по целым дням; recall/AUC оставить diagnostic-only",
            "evidence": f"bandit_scope={training.get('evaluation_scope') or 'unknown'}; "
                        f"model_label_timing={training.get('model_label_timing') or 'unknown'}",
            "gate": "Не считать training-to-live gap до честного holdout",
        })
    verified = dnt.get("last_verified")
    budget = int(dnt.get("verify_every_days") or 0)
    if verified and budget:
        try:
            age = (today - date.fromisoformat(str(verified))).days
            if age > budget:
                steps.append({
                    "priority": "P0", "id": "refresh_gate_evidence",
                    "action": "Повторить targeted gate replays на максимальном периоде текущей policy",
                    "evidence": f"do_not_touch age={age}d > {budget}d budget",
                    "gate": "Не ослаблять и не объявлять gate доказанным по stale evidence",
                })
        except ValueError:
            pass
    alpha = scorecard.get("portfolio_alpha") or {}
    if alpha.get("value") is None:
        steps.append({
            "priority": "P0", "id": "restore_portfolio_alpha",
            "action": "Сформировать агрегированный weekly portfolio alpha vs buy-and-hold",
            "evidence": "canonical profitability metric is unknown",
            "gate": "Не называть per-mode P&L прибыльностью бота",
        })
    capture = scorecard.get("realized_potential") or {}
    if capture.get("value") is None:
        steps.append({
            "priority": "P0", "id": "restore_canonical_ex1",
            "action": "Рассчитать и сохранить EX1 в ZigZag-mode с provenance и покрытием текущей policy",
            "evidence": capture.get("reason") or "canonical EX1 is unknown",
            "gate": "Не называть legacy proxy-mode EX1 реализованным потенциалом",
        })
    elif capture["value"] < capture["target"]:
        steps.append({
            "priority": "P1", "id": "exit_monetization_replay",
            "action": "Проверить tail-hold, partial-exit и re-entry на максимальном периоде",
            "evidence": f"EX1 median={capture['value']:.3f} < target={capture['target']:.2f}; n={capture.get('n')}",
            "gate": "Не менять production SELL без положительного multi-objective backtest",
        })
    precision = scorecard.get("signal_precision") or {}
    msg = scorecard.get("message_rate") or {}
    if ((isinstance(precision.get("value"), (int, float)) and precision["value"] < precision["target"])
            or (isinstance(msg.get("value"), (int, float)) and msg["value"] > msg["target_max"])):
        steps.append({
            "priority": "P1", "id": "honest_alert_budget_ranker",
            "action": "Проверить time-held-out ranking при фиксированном alert budget до изменения BUY gates",
            "evidence": f"precision={precision.get('value')}% (target {precision.get('target')}%), "
                        f"messages={msg.get('value')}/d (max {msg.get('target_max')})",
            "gate": "Улучшать frontier precision/recall, не покупать recall спамом по большинству символов",
        })
    return steps


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


def detect_red_flags(deploy: dict, per_mode: dict, gap: dict, scout: dict,
                     critic_raw: dict, metrics_daily: dict | None = None,
                     training: dict | None = None,
                     do_not_touch: dict | None = None,
                     today: date | None = None) -> list[dict]:
    flags = []
    training = training or {}
    do_not_touch = do_not_touch or {}

    # RF0 — Evidence integrity.  A missing critic used to yield an empty list
    # and therefore the user-facing lie "No red flags".  The expected morning
    # state is yesterday's final (fallback_days=1); midday is explicitly
    # partial and older finals are stale.
    if not critic_raw.get("available"):
        flags.append({
            "id": "RF_critic_unavailable",
            "metric": "critic_available",
            "value": 0.0,
            "threshold": 1.0,
            "severity": "critical",
            "evidence": {},
            "root_cause_hypothesis": "Нет critic evidence: deployment-health и live North Star проверить нельзя.",
        })
    else:
        critic_data = critic_raw.get("data") or {}
        phase = critic_data.get("_phase_used")
        age = int(critic_data.get("_fallback_days") or 0)
        if phase != "final":
            flags.append({
                "id": "RF_critic_partial",
                "metric": "critic_final_available",
                "value": 0.0,
                "threshold": 1.0,
                "severity": "red",
                "evidence": {"phase": phase,
                             "target_date": critic_data.get("_critic_target_date")},
                "root_cause_hypothesis": "Доступен только partial/midday critic; итоговые deployment-метрики ещё не подтверждены.",
            })
        elif age > 1:
            flags.append({
                "id": "RF_critic_stale",
                "metric": "critic_age_days",
                "value": float(age),
                "threshold": 1.0,
                "severity": "critical" if age > 2 else "red",
                "evidence": {"target_date": critic_data.get("_critic_target_date")},
                "root_cause_hypothesis": "Последний final critic старше последнего завершённого торгового дня.",
            })

    if deploy.get("available") and not deploy.get("watchlist_top_total"):
        flags.append({
            "id": "RF_top_mover_denominator_unknown",
            "metric": "watchlist_top_total",
            "value": 0.0,
            "threshold": 1.0,
            "severity": "critical",
            "evidence": {"watchlist_top_total": deploy.get("watchlist_top_total")},
            "root_cause_hypothesis": "Critic denominator отсутствует/нулевой; ratio не вычисляется.",
        })

    # More than one partial/down day cannot be explained solely by today's
    # unfinished UTC day and is an operational-coverage alert.
    md = (metrics_daily or {}).get("metrics") or {}
    ns_md = md.get("NS_EarlyCapture_top20") or {}
    down = ns_md.get("days_down_or_partial")
    if isinstance(down, (int, float)) and down > 1:
        unexpected = float(down - 1)
        flags.append({
            "id": "RF_uptime_gap",
            "metric": "unexpected_down_or_partial_days",
            "value": unexpected,
            "threshold": 0.0,
            "severity": "critical" if unexpected >= 3 else "red",
            "evidence": {"days_window": ns_md.get("days_window"),
                         "days_full": ns_md.get("days_full"),
                         "days_down_or_partial": down},
            "root_cause_hypothesis": "В окне есть неполные/нерабочие дни сверх текущего незавершённого дня.",
        })

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
    elif training.get("available") and gap.get("reason") == "training_metric_not_out_of_sample":
        flags.append({
            "id": "RF_training_evidence_invalid",
            "metric": "training_evaluation_scope",
            "value": 0.0,
            "threshold": 1.0,
            "severity": "critical",
            "evidence": {
                "bandit_scope": training.get("evaluation_scope") or "unknown",
                "model_scope": training.get("model_evaluation_scope") or "unknown",
                "model_label_timing": training.get("model_label_timing") or "unknown",
            },
            "root_cause_hypothesis": (
                "Training recall/AUC не являются out-of-sample доказательством; "
                "training-to-live gap намеренно не вычисляется."
            ),
        })

    verified = do_not_touch.get("last_verified")
    budget = int(do_not_touch.get("verify_every_days") or 0)
    if verified and budget and today is not None:
        try:
            age = (today - date.fromisoformat(str(verified))).days
        except ValueError:
            age = None
        if age is not None and age > budget:
            flags.append({
                "id": "RF_gate_evidence_stale",
                "metric": "do_not_touch_evidence_age_days",
                "value": float(age),
                "threshold": float(budget),
                "severity": "critical",
                "evidence": {"last_verified": verified, "verify_every_days": budget},
                "root_cause_hypothesis": (
                    "Gate evidence старше допустимого бюджета; защиты остаются "
                    "fail-closed, но их эффект нельзя называть актуально доказанным."
                ),
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
        total = s.get("watchlist_top_count")
        early = s.get("watchlist_top_early_captured", 0)
        fp = s.get("bot_false_positive_buys", 0)
        buys = s.get("bot_unique_buys")
        deploy = {
            "available": True,
            "phase": critic_raw["data"].get("_phase_used"),
            "critic_target_date": critic_raw["data"].get("_critic_target_date"),
            "critic_fallback_days": critic_raw["data"].get("_fallback_days"),
            "watchlist_top_bought_pct":        round(bought / total, 4) if total else None,
            "watchlist_top_early_capture_pct": round(early / total, 4) if total else None,
            "false_positive_rate":             round(fp / buys, 4) if buys else None,
            "bot_unique_buys":                 buys,
            "watchlist_top_bought":            bought,
            "watchlist_top_early":             early,
            "watchlist_top_total":             total,
            "missed_count":                    s.get("watchlist_top_missed"),
            "false_positive_buys":             fp,
        }

    gap = compute_training_to_live_gap(training, deploy) if training.get("available") and deploy.get("available") else {"available": False}
    deployment_diagnostic = compute_north_star(deploy, baseline)
    dnt = PL.load_do_not_touch()
    scorecard = build_canonical_scorecard(metrics_daily)
    next_steps = derive_next_steps(scorecard, training, dnt, today)
    red_flags = detect_red_flags(
        deploy, per_mode, gap, scout, critic_raw, metrics_daily,
        training=training, do_not_touch=dnt, today=today,
    )
    if scorecard["north_star"].get("status") != "verified":
        red_flags.append({
            "id": "RF_north_star_ground_truth_provisional",
            "metric": "north_star_label_provenance",
            "value": 0.0,
            "threshold": 1.0,
            "severity": "critical",
            "evidence": {"provenance": scorecard["north_star"].get("provenance")},
            "root_cause_hypothesis": (
                "top_gainer_dataset label_top20 создаётся из того же rolling-24h "
                "snapshot; это не immutable later-EOD outcome."
            ),
        })

    report = {
        "report_id": f"health-{today.isoformat()}",
        "schema_version": 2,
        "generated_at": PL.utc_now_iso(),
        "target_date": today.isoformat(),
        "window_days": 1,
        "north_star": {
            "metric": "EarlyCapture@top20",
            "value": scorecard["north_star"].get("value"),
            "target": scorecard["north_star"].get("target"),
            "acceptable_floor": scorecard["north_star"].get("acceptable_floor"),
            "status": scorecard["north_star"].get("status"),
            "baseline_7d": None,
            "regression_vs_7d_avg": None,
            "n": scorecard["north_star"].get("n"),
            "days_window": scorecard["north_star"].get("days_window"),
            "days_full": scorecard["north_star"].get("days_full"),
            "reason": scorecard["north_star"].get("provenance"),
        },
        "canonical_scorecard": scorecard,
        "deployment_critic_diagnostic": deployment_diagnostic,
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
        "mode_curtail": collect_mode_curtail(),
        "red_flags": red_flags,
        "next_steps": next_steps,
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

    add("## Canonical Objective Scorecard")
    add("")
    add("| Question | Value | Target | Evidence |")
    add("|---|---:|---:|---|")
    for key, item in (r.get("canonical_scorecard") or {}).items():
        value = item.get("value")
        shown = "unknown" if value is None else str(value)
        target = item.get("target")
        if target is None:
            target = f"≤ {item.get('target_max')}"
        add(f"| {key} | {shown} | {target} | {item.get('source')} |")
    add("")

    # Training-to-live gap (the most important section)
    gap = r["training_to_live_gap"]
    if gap.get("available"):
        add(f"## {PL.status_emoji(gap['severity'])} Training-to-Live Gap")
        add("")
        add(f"- Gap: **{gap['value']:+.1%}**")
        add(f"- {gap['interpretation']}")
        add("")
    elif gap.get("reason"):
        add("## ❌ Training-to-Live Gap unavailable")
        add("")
        add(f"- Reason: `{gap['reason']}`")
        add(f"- Evaluation scope: `{gap.get('evaluation_scope', 'unknown')}`")
        add("")

    # Deployment health
    dh = r["deployment_health"]
    if dh.get("available"):
        def _pct_or_unknown(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.1%}"

        add("## Deployment Health (live bot)")
        add("")
        add("| Метрика | Значение | Норма | Статус |")
        add("|---------|----------|-------|--------|")
        add(f"| watchlist_top_bought | {dh['watchlist_top_bought']}/{dh['watchlist_top_total']} ({_pct_or_unknown(dh['watchlist_top_bought_pct'])}) | ≥50% | {PL.status_emoji(PL.classify(dh['watchlist_top_bought_pct'], 'watchlist_top_bought_pct'))} |")
        add(f"| early_captures | {dh['watchlist_top_early']}/{dh['watchlist_top_total']} ({_pct_or_unknown(dh['watchlist_top_early_capture_pct'])}) | ≥25% | {PL.status_emoji(PL.classify(dh['watchlist_top_early_capture_pct'], 'watchlist_top_early_capture_pct'))} |")
        add(f"| false_positive_rate | {dh['false_positive_buys']}/{dh['bot_unique_buys']} ({_pct_or_unknown(dh['false_positive_rate'])}) | <50% | {PL.status_emoji(PL.classify(dh['false_positive_rate'], 'false_positive_rate'))} |")
        add("")

    # Training health
    th = r["training_health"]
    if th.get("available"):
        add("## Training Evidence (diagnostic unless scope is OOS)")
        add("")
        add(f"- Bandit scope: `{th.get('evaluation_scope') or 'unknown'}`")
        add(f"- Model scope: `{th.get('model_evaluation_scope') or 'unknown'}`")
        add(f"- Label timing: `{th.get('model_label_timing') or 'unknown'}`")
        if th.get("legacy_ratio_suppressed"):
            add("- Legacy recall: **suppressed** (missing base rate, action rate, lift, precision)")
        elif th.get("recall_at_20") is not None:
            add(f"- Bandit diagnostic: recall={th['recall_at_20']:.1%}; "
                f"action={th['action_rate']:.1%}; base={th['base_rate']:.1%}; "
                f"lift={th['lift']:.2f}×; precision={th['precision']:.1%}")
        if th.get("auc") is not None:
            add(f"- Model diagnostic AUC: {th['auc']:.3f}")
        if th.get("bandit_total_updates") is not None:
            add(f"- Bandit updates total: {th['bandit_total_updates']:,} (activity, not quality)")
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

    if r.get("next_steps"):
        add("## Evidence-ranked next steps")
        add("")
        for step in r["next_steps"]:
            add(f"- **{step['priority']} {step['id']}** — {step['action']}; "
                f"evidence: {step['evidence']}; gate: {step['gate']}")
        add("")

    add("---")
    add("")
    add("## Data sources")
    add("")
    for k, v in r["data_sources"].items():
        add(f"- **{k}**: `{v}`")
    add("")
    return "\n".join(lines)


def _ns_history() -> list[tuple[str, float]]:
    """Comparable (date, early_capture) series, oldest→newest.

    Metrics before the uptime-adjusted 14-day schema used a different
    denominator.  Prefer schema-marked records and de-duplicate repeated daily
    pipeline runs.  A bounded legacy fallback keeps the report useful on a
    fresh installation that has not accumulated two current-schema rows yet.
    """
    comparable: dict[str, float] = {}
    legacy: dict[str, float] = {}
    for row in PL.iter_jsonl(PL.METRICS_DAILY):
        m = row.get("metrics") or row
        ns = m.get("NS_EarlyCapture_top20") or m.get("_compute_early_capture.py") or {}
        ec = ns.get("early_capture")
        if ec is None:
            continue
        try:
            day = str(row.get("ts", ""))[:10]
            if not day:
                continue
            value = float(ec)
            legacy[day] = value
            if ns.get("days_window") == 14 and ns.get("days_full") is not None:
                comparable[day] = value
        except (TypeError, ValueError):
            continue
    chosen = comparable if len(comparable) >= 2 else legacy
    return sorted(chosen.items())[-14:]


def _ns_history_with_meta() -> list[tuple[str, float, int | None]]:
    """(date, early_capture, days_full) — same series as _ns_history, but keeps
    how many working days each point was computed over, so the verdict can
    refuse to compare a 5-day window with a 10-day one."""
    comparable: dict[str, tuple[float, int | None]] = {}
    legacy: dict[str, tuple[float, int | None]] = {}
    for row in PL.iter_jsonl(PL.METRICS_DAILY):
        m = row.get("metrics") or row
        ns = m.get("NS_EarlyCapture_top20") or m.get("_compute_early_capture.py") or {}
        ec = ns.get("early_capture")
        if ec is None:
            continue
        day = str(row.get("ts", ""))[:10]
        if not day:
            continue
        try:
            value = float(ec)
        except (TypeError, ValueError):
            continue
        df = ns.get("days_full")
        legacy[day] = (value, df if isinstance(df, int) else None)
        if ns.get("days_window") == 14 and df is not None:
            comparable[day] = (value, df)
    chosen = comparable if len(comparable) >= 2 else legacy
    return [(d, v, f) for d, (v, f) in sorted(chosen.items())[-14:]]


def _per100(frac: float) -> int:
    """0.106 -> 11  (North-Star points per 100)."""
    return max(0, round(frac * 100))


def _progress_verdict(*, ground_truth_verified: bool = False) -> tuple[str, str, str]:
    """Line-1 answer in PLAIN words: developing / flat / worse.
    Judged only on the North Star over the longest history (§0).

    Endpoints must cover a comparable number of WORKING days. After the 8-day
    outage the 14-day window refilled gradually (days_full 5 -> 10, n 13 -> 26)
    and the average simply regressed toward its true level; comparing those
    endpoints produced "СТАЛО ХУЖЕ ~11 -> ~7" out of pure window growth. A
    headline verdict built on incomparable samples is worse than none.
    """
    if not ground_truth_verified:
        return ("❔", "МЕТРИКА ПРЕДВАРИТЕЛЬНАЯ",
                "тренд не оценивается до пересчёта на immutable later-EOD labels")
    h = _ns_history_with_meta()
    if len(h) < 2:
        return ("❔", "ПОКА НЕ ЯСНО", "мало истории, чтобы судить")
    (d0, first_v, f0), (d1, last_v, f1) = h[0], h[-1]
    if f0 is not None and f1 is not None and abs(f0 - f1) > 2:
        return ("❔", "РАНО СУДИТЬ",
                f"окно ещё наполняется после простоя: {d0} считалось по {f0} "
                f"рабочим дням, {d1} — по {f1}. Сравнивать их нельзя")
    delta_pp = (last_v - first_v) * 100
    trend = (f"за сопоставимый период {h[0][0]}–{h[-1][0]}: "
             f"North Star ~{_per100(first_v)} → ~{_per100(last_v)} из 100")
    if delta_pp > 1.0:
        return ("📈", "РАЗВИВАЕТСЯ (медленно)", trend)
    if delta_pp < -1.0:
        return ("📉", "СТАЛО ХУЖЕ", trend)
    return ("➖", "СТОИТ НА МЕСТЕ", trend)


def _superseded_hyps() -> set[str]:
    """hypothesis_ids that were deferred/rolled_back (terminal — sticky)."""
    s: set[str] = set()
    for d in PL.iter_jsonl(PL.DECISIONS_LOG):
        if d.get("stage") in ("deferred", "rolled_back"):
            if d.get("hypothesis_id"):
                s.add(d["hypothesis_id"])
            tgt = d.get("defers") or d.get("rolling_back")
            if tgt:
                s.add(tgt)
    return s


_RULE_PLAIN = {
    "disable_mode_impulse_speed":      "отключение режима «быстрый импульс»",
    "entry_score_floor_relax":         "порог входа (мягче)",
    "relax_gate_late_impulse_rotation":"фильтр «поздняя ротация»",
    "relax_gate_ranker_hard_veto":     "фильтр оценщика",
    "tighten_proba_impulse_speed":     "строже к «быстрому импульсу»",
    "widen_watchlist_match_tolerance": "шире сопоставление монет",
    "reduce_impulse_speed_lateness_window": "окно опоздания «быстрый импульс»",
    "shorten_late_impulse_rotation_cooldown": "пауза «поздняя ротация»",
}


def _rule_plain(rule: str, hid: str) -> str:
    if rule in _RULE_PLAIN:
        return _RULE_PLAIN[rule]
    return (rule or hid or "изменение").replace("_", " ")


def _past_decisions_resume() -> list[str]:
    """Per-decision plain-language outcome: helped / worsened / too
    early / rolled back / held. Dedup by hypothesis (terminal wins).
    Honest: if a change hasn't matured 14d we say 'рано судить'."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    superseded = _superseded_hyps()

    state: dict[str, dict] = {}
    for d in PL.iter_jsonl(PL.DECISIONS_LOG):
        hid = d.get("hypothesis_id")
        st = d.get("stage")
        if not hid or st not in ("approved", "rolled_back", "deferred", "rejected"):
            continue
        try:
            age = (now - datetime.fromisoformat(
                str(d.get("ts", "")).replace("Z", "+00:00"))).days
        except (ValueError, TypeError):
            age = None
        state[hid] = {"stage": st, "rule": d.get("rule"), "age": age,
                      "decision_id": d.get("decision_id")}

    # Per-decision attribution: the pipeline already computes a verdict PER
    # decision_id (attribution results[]). Use each decision's OWN verdict — not
    # one global hit_rate for all — so "helped/harmed" is honest per decision.
    results_by_id: dict[str, dict] = {}
    adir = PL.PIPELINE / "attribution"
    if adir.exists():
        files = sorted(adir.glob("attribution-*.json"))
        if files:
            try:
                _att = json.loads(files[-1].read_text(encoding="utf-8"))
                for r in (_att.get("results") or []):
                    did = r.get("decision_id")
                    if did:
                        results_by_id[did] = r
            except (OSError, json.JSONDecodeError):
                pass

    _metric_plain = {
        "watchlist_top_early_capture_pct": "ранний захват top-20",
        "avg_r5_entries": "качество входов",
        "entries_per_day": "число входов",
        "realised_pnl_pct": "P&L",
    }

    lines: list[str] = []
    for hid, s in state.items():
        name = _rule_plain(s["rule"], hid)
        st, age = s["stage"], s["age"]
        if hid in superseded and st != "rolled_back":
            lines.append(f"  • {name} — ⏸ придержали до проверки")
            continue
        if st == "rejected":
            continue
        if st == "rolled_back":
            lines.append(f"  • {name} — ↩️ откатили")
            continue
        # Shadow decisions log a would-action only (no decision impact), so no
        # helped/harmed verdict applies — label as data-collection.
        if "shadow" in str(s["rule"] or "").lower() or "shadow" in hid.lower():
            lines.append(f"  • {name} — 🔍 shadow (сбор данных, без влияния на решения)")
            continue
        if age is not None and age < 14:
            lines.append(f"  • {name} — применили {age} дн назад · "
                         f"⏳ рано судить (ещё ~{14 - age} дн)")
            continue
        res = results_by_id.get(s.get("decision_id") or "")
        status = _attribution_status(res)
        if status == "no_baseline":
            lines.append(f"  • {name} — ⚠️ baseline отсутствует; эффект не измеряется")
        elif status == "insufficient_data":
            lines.append(f"  • {name} — ⚠️ baseline есть, но целевые метрики не записывались")
        elif status == "pending":
            lines.append(f"  • {name} — применили · ⏳ ещё считаем")
        elif status == "helped":
            lines.append(f"  • {name} — ✅ помогло")
        elif status == "harmed":
            misses = res.get("expected_misses") or []
            det = f" (просело: {_metric_plain.get(misses[0], misses[0])})" if misses else ""
            lines.append(f"  • {name} — ❌ не помогло / навредило{det}")
        else:
            lines.append(f"  • {name} — ⚠️ эффект смешанный")

    return lines[-4:] if lines else ["  • пока ни одно решение не применяли"]


def _attribution_status(result: dict | None) -> str:
    """Normalize attribution into a truthful user-facing state.

    The attribution engine can emit outer verdict ``miss`` when every expected
    metric is actually ``insufficient_data``.  That is absence of measurement,
    not evidence of harm.
    """
    if not result:
        return "pending"
    verdict = result.get("verdict")
    if verdict == "no_baseline":
        return "no_baseline"
    if verdict in (None, "insufficient_data", "needs_data"):
        return "insufficient_data"
    rationale = [str(x).lower() for x in (result.get("rationale") or [])]
    if rationale and all("insufficient_data" in x for x in rationale):
        return "insufficient_data"
    if verdict in ("hit", "improvement", "win", "accept"):
        return "helped"
    if verdict in ("regression", "miss", "worse"):
        return "harmed"
    return "mixed"


def _action_needed_count() -> int:
    """How many hypotheses are ACTUALLY actionable for the operator —
    i.e. validated with a real verdict (accept/needs_review), not held,
    not still waiting for a validator. Avoids over-promising "N decisions
    for you" when those N have no data to decide on."""
    superseded = _superseded_hyps()
    n = 0
    for p in PL.HYPOTHESES.glob("h-*.json"):
        h = PL.read_json(p) or {}
        if h.get("status") != "pending_validation":
            continue
        if h.get("hypothesis_id") in superseded:
            continue
        v = ((h.get("validation_report") or {}).get("result") or {}).get("verdict")
        if v in ("accept", "needs_review"):
            n += 1
    return n


def render_telegram(r: dict) -> str:
    """Plain-language daily summary: in a few lines the operator sees if
    the bot is developing, where it loses, what past decisions did, and
    whether action is needed — zero jargon."""
    md = (r.get("metrics_daily_latest") or {}).get("metrics") or {}
    ns_md = md.get("NS_EarlyCapture_top20") or {}
    funnel = md.get("C1_C2_coverage_funnel") or {}
    rf = r.get("red_flags") or []

    score_ns = ((r.get("canonical_scorecard") or {}).get("north_star") or {})
    p_emoji, p_head, p_trend = _progress_verdict(
        ground_truth_verified=score_ns.get("status") == "verified",
    )
    n_act = _action_needed_count()
    if n_act:
        act = f"👉 нужно твоё решение ({n_act})"
    elif rf or not (r.get("deployment_health") or {}).get("available"):
        act = "👉 требуется проверка данных/метрик"
    else:
        act = "👉 от тебя ничего не требуется"

    out = [f"🩺 <b>Бот</b> — {r['target_date']}", ""]
    out.append(f"{p_emoji} <b>{p_head}</b>   ·   {act}")
    out.append("")

    dh = r.get("deployment_health") or {}
    critic_day = dh.get("critic_target_date")
    if dh.get("available") and critic_day:
        phase = dh.get("phase") or "unknown"
        suffix = " (partial)" if phase != "final" else ""
        out.append(f"🧾 critic: {critic_day} · {phase}{suffix}")
        out.append("")

    ec = ns_md.get("early_capture")
    if ec is not None:
        out.append(f"<b>Главное:</b> предварительный North Star раннего захвата = {ec:.1%} "
                   f"({p_trend}). Цель — 40%, минимально приемлемо 25%.")
        out.append("Это составной score, а не доля пойманных монет: "
                   "coverage × реализованная часть движения × своевременность.")
        out.append("⚠️ Ground truth пока provisional: label построен на rolling-24h snapshot, "
                   "а не на неизменяемом later-EOD top-20.")
        cov = funnel.get("coverage_pct_raw")
        sm = funnel.get("silent_miss_pct")
        capm = ns_md.get("decomp_capture_mean")
        if cov is not None and capm is not None:
            caught = round(cov / 10.0)
            fifth = max(1, round(1 / capm)) if capm > 0 else None
            out.append(f"<b>Где теряем:</b> из 10 событий watchlist∩global-top20 "
                       f"~{caught} имели вход, из доступного движения реализовано лишь "
                       f"{'1/' + str(fifth) if fifth else '0%'} их роста.")
        coverage = ns_md.get("decomp_coverage")
        lead = ns_md.get("decomp_time_lead_mean")
        components = {
            "coverage": coverage,
            "capture": capm,
            "lead": lead,
        }
        components = {k: float(v) for k, v in components.items()
                      if isinstance(v, (int, float))}
        if components:
            bottleneck = min(components, key=components.get)
            value = components[bottleneck]
            if bottleneck == "capture":
                out.append(f"<b>Главный тормоз:</b> монетизация после входа — "
                           f"бот реализует лишь ~{round(value * 100)}% доступного движения top-mover событий.")
            elif bottleneck == "coverage":
                out.append(f"<b>Главный тормоз:</b> покрытие — бот входит лишь "
                           f"примерно в {round(value * 100)}% событий watchlist∩global-top20.")
            else:
                out.append(f"<b>Главный тормоз:</b> запаздывание — после входа "
                           f"остаётся около {round(value * 100)}% времени движения.")
        if sm is not None and sm > 0:
            every = max(2, round(100 / sm))
            out.append(f"<b>Совсем не видит:</b> примерно каждое {every}-е "
                       f"top-mover событие в области наблюдения.")
        # Uptime context: the numbers above count only days the bot actually ran.
        # Without this line an outage reads as a performance collapse (2026-07-23:
        # 8 days down -> report said "~2 из 100" while live days were at a record).
        _dwin = ns_md.get("days_window")
        _dfull = ns_md.get("days_full")
        if _dwin and _dfull is not None and _dfull < _dwin:
            out.append(f"⚠️ <b>Покрытие работы: {_dfull} полных дней из {_dwin}</b> — "
                       f"{_dwin - _dfull} дня неполные или без данных; "
                       f"метрики считают только полные дни.")
    else:
        out.append("<b>Главное:</b> результат за сегодня ещё считается.")
    out.append("")

    score = r.get("canonical_scorecard") or {}
    if score:
        alpha = score.get("portfolio_alpha") or {}
        precision = score.get("signal_precision") or {}
        msg = score.get("message_rate") or {}
        tts = score.get("time_to_signal") or {}
        ex1 = score.get("realized_potential") or {}
        fr = score.get("fast_reversal") or {}
        wh = score.get("whipsaw") or {}

        def _v(item: dict, suffix: str = "", digits: int = 1) -> str:
            value = item.get("value")
            return "неизвестно" if not isinstance(value, (int, float)) else f"{value:.{digits}f}{suffix}"

        out.append("🎯 <b>Канонический scorecard</b>")
        out.append(f"  прибыль портфеля vs buy-and-hold: {_v(alpha, '%')} (цель &gt; 0%)")
        out.append(f"  precision сигналов: {_v(precision, '%')} / 35%; "
                   f"сообщений: {_v(msg, '/д')} / ≤10/д")
        out.append(f"  время до сигнала: {_v(tts, 'ч', 2)} / ≤0.5ч; "
                   f"реализованный потенциал: {_v(ex1, '', 3)} / 0.50")
        out.append(f"  fast reversal: {_v(fr, '%')} / ≤8%; whipsaw: {_v(wh, '%')} / ≤5%")
        out.append("")

    out.append("📋 <b>Прошлые решения</b> (помогли или нет):")
    out.extend(_past_decisions_resume())
    out.append("")

    mc = r.get("mode_curtail") or {}
    if mc.get("available") and mc.get("enabled"):
        tp = mc.get("trailing_mean_pnl")
        tp_s = f"{tp:+.2f}%/сделку" if isinstance(tp, (int, float)) else "n/a"
        wd = mc.get("window_days") or 14
        if mc.get("curtailed"):
            out.append(f"⏸️ <b>impulse_speed на паузе</b> (режим плох: "
                       f"{wd}-дн pnl {tp_s}) — авто-возврат при плюсе")
        else:
            out.append(f"▶️ impulse_speed активен ({wd}-дн pnl {tp_s})")
        out.append("")

    if rf:
        crit = sum(1 for x in rf if x.get("severity") == "critical")
        tail = f" ({crit} серьёзн.)" if crit else ""
        out.append(f"🚨 {len(rf)} сигнал(ов) тревоги{tail} — детали в полном отчёте")
    elif not dh.get("available"):
        out.append("⚠️ статус тревог неизвестен: deployment evidence недоступен")
    else:
        out.append("✅ Тревог нет")

    steps = r.get("next_steps") or []
    if steps:
        out.append("")
        out.append("🧭 <b>Следующие доказуемые шаги</b>")
        for idx, step in enumerate(steps[:6], 1):
            out.append(f"  {idx}. [{step.get('priority', '?')}] {step.get('action')} — "
                       f"{step.get('evidence')}")

    learn = _render_learning_block(r)
    if learn:
        out.append("")
        out.append(learn)

    agent = _render_agent_block()
    if agent:
        out.append("")
        out.append(agent)

    return "\n".join(out)


def _render_learning_block(r: dict) -> str | None:
    """Render training evidence without turning diagnostics into achievements."""
    th = r.get("training_health") or {}
    if not th.get("available"):
        return None
    rec, auc = th.get("recall_at_20"), th.get("auc")
    action_rate, base_rate = th.get("action_rate"), th.get("base_rate")
    lift, precision = th.get("lift"), th.get("precision")
    scope = th.get("evaluation_scope") or "unknown"
    model_scope = th.get("model_evaluation_scope") or "unknown"
    label_timing = th.get("model_label_timing") or "unknown"

    lines = ["🧠 <b>Обучение — статус доказательств</b>"]
    if th.get("legacy_ratio_suppressed"):
        lines.append("  legacy recall скрыт: в записи нет base rate, action rate и lift")
    elif isinstance(rec, (int, float)):
        lines.append(
            f"  bandit diagnostic: recall={rec:.1%}, ENTER={action_rate:.1%}, "
            f"base={base_rate:.1%}, lift={lift:.2f}×, precision={precision:.1%}; "
            f"scope={scope}"
        )
    if isinstance(auc, (int, float)):
        lines.append(
            f"  model diagnostic: AUC={auc:.3f}; scope={model_scope}; "
            f"label={label_timing}"
        )
    if th.get("model_label_encoding_features"):
        lines.append("  ⚠️ признаки могут кодировать текущий leaderboard: "
                     + ", ".join(map(str, th["model_label_encoding_features"])))

    gap = r.get("training_to_live_gap") or {}
    if gap.get("available") and isinstance(gap.get("value"), (int, float)):
        lines.append(f"  training↔live gap: {gap['value']:.1%} (валидный temporal holdout)")
    elif gap.get("reason") == "training_metric_not_out_of_sample":
        lines.append("  ❌ training↔live gap неизвестен: нет out-of-sample temporal holdout")
    return "\n".join(lines)


def _render_agent_block() -> str | None:
    """Is the LLM agent actually being used for hypotheses, and to what effect?

    Without this the operator cannot tell whether "no ideas" means the agent
    reviewed everything and found nothing, or was never called at all — the
    latter was true for 11 days in August (the weekly task refused to start on
    battery)."""
    calls_path = PL.PIPELINE / "claude_calls.jsonl"
    calls = [c for c in PL.iter_jsonl(calls_path) if isinstance(c, dict)]
    if not calls:
        return None
    from collections import Counter
    purposes = Counter(str(c.get("purpose") or "?") for c in calls)
    last_ts = str(calls[-1].get("ts") or "")[:10]
    gen = purposes.get("weekly_generation", 0)
    adv = purposes.get("approval_advice", 0)
    crit = purposes.get("blind_critique", 0)
    return ("🤖 <b>Агент (LLM) в контуре гипотез</b>\n"
            f"  последний вызов {last_ts} · всего {len(calls)}: "
            f"генерация {gen}, оценка одобрения {adv}, слепая критика {crit}")


def _render_telegram_legacy(r: dict) -> str:
    """Short Telegram summary — single screen."""
    ns = r["north_star"]
    gap = r["training_to_live_gap"]
    dh = r["deployment_health"]
    th = r["training_health"]
    rf_count = len(r["red_flags"])

    out = []
    out.append(f"🩺 <b>Bot Health Report</b> — {r['target_date']}")
    out.append("")

    # ── North Star — ALWAYS first, ALWAYS present ────────────────────────────
    # §0 honest-measurement: the project's purpose metric must never be
    # silently dropped. The L1 critic value (ns["value"]) is null until the
    # EOD critic runs; fall back to the canonical EOD metric from
    # metrics_daily (NS_EarlyCapture_top20) so the headline is the actual
    # target, not a saturated training proxy.
    md = (r.get("metrics_daily_latest") or {}).get("metrics") or {}
    ns_md = md.get("NS_EarlyCapture_top20") or {}
    funnel = md.get("C1_C2_coverage_funnel") or {}
    if ns["value"] is not None:
        reg = ""
        if ns["regression_vs_7d_avg"] is not None:
            sign = "+" if ns["regression_vs_7d_avg"] >= 0 else ""
            reg = f" ({sign}{ns['regression_vs_7d_avg']:.1%} vs 7d)"
        out.append(f"{PL.status_emoji(ns['status'])} <b>North-star early_capture</b>: {ns['value']:.1%}{reg}")
    elif ns_md.get("early_capture") is not None:
        ec = ns_md["early_capture"]
        cov = funnel.get("coverage_pct_raw")
        sm = funnel.get("silent_miss_pct")
        line = f"🎯 <b>North-star early_capture</b>: {ec:.1%} <i>(EOD canon)</i>"
        if cov is not None and sm is not None:
            line += f" | coverage {cov:.0f}% | silent-miss {sm:.0f}%"
        out.append(line)
    else:
        out.append("⚠️ <b>North-star</b>: ещё не измерен сегодня (критик не отработал)")
    if gap.get("available"):
        out.append(f"{PL.status_emoji(gap['severity'])} <b>Training-to-live gap</b>: {gap['value']:+.1%}")

    if dh.get("available"):
        def _legacy_pct(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.0%}"

        out.append("")
        out.append("<b>Deployment:</b>")
        out.append(f"  • bought: {dh['watchlist_top_bought']}/{dh['watchlist_top_total']} ({_legacy_pct(dh['watchlist_top_bought_pct'])})")
        out.append(f"  • early: {dh['watchlist_top_early']}/{dh['watchlist_top_total']} ({_legacy_pct(dh['watchlist_top_early_capture_pct'])})")
        out.append(f"  • FPR: {dh['false_positive_buys']}/{dh['bot_unique_buys']} ({_legacy_pct(dh['false_positive_rate'])})")

    # Training — one honest line. recall@20 is omitted on purpose: it has
    # been pinned at 100% (saturated plateau) since ~Apr 10 and carries no
    # signal — leading with it implies progress that isn't there. UCB
    # separation is the only training number that still moves.
    if th.get("available"):
        out.append("")
        ucb = th.get("ucb_separation")
        auc = th.get("auc")
        out.append(f"<i>Training diagnostics (не цель): UCB {ucb if ucb is not None else 'unknown'} | "
                   f"AUC {auc if auc is not None else 'unknown'} | recall suppressed/OOS required</i>")

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
