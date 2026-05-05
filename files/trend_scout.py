"""
trend_scout.py — Автоматический мониторинг расхождений между трендами рынка и сигналами бота.

Архитектура (5 фаз, запускается каждые 4 часа из rl_headless_worker):

  Phase 1 — Trend Scanner:
    Читает последние записи critic_dataset.jsonl,
    оценивает качество зарождающегося тренда для каждой монеты (trend_score 0-100),
    маркирует статус: entered | blocked | no_signal.

  Phase 2 — Diagnosis:
    Для высокоскоринговых монет со статусом blocked/no_signal
    парсит bot_events.jsonl и critic_dataset.decision,
    находит КОНКРЕТНЫЙ параметр config.py который заблокировал вход.

  Phase 3 — Proposal:
    Агрегирует диагнозы по параметру.
    Если ≥ MIN_MISSES разных монет заблокированы одним параметром → предлагает изменение.

  Phase 4 — Backtest:
    Использует blocked-записи critic_dataset с ret_5/ret_10 метками.
    Оценивает: если бы порог был другим — сколько было бы входов и каков их ret_5?

  Phase 5 — Apply:
    low risk + backtest approved → авто-применение + Telegram отчёт.
    medium/high → только Telegram с предложением.

Запуск:
    import trend_scout
    report = await trend_scout.run_pipeline()
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import config
import trend_scout_rules
from trend_scout_rules import BlockRule, find_rules, extract_values, compute_proposed_value

log = logging.getLogger("trend_scout")

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"
CRITIC_FILE = ROOT / "critic_dataset.jsonl"
CONFIG_FILE = ROOT / "config.py"
CHANGELOG_FILE = ROOT / "trend_scout_changelog.jsonl"
REPORT_FILE = RUNTIME_DIR / "trend_scout_report.json"

# ── Настройки ──────────────────────────────────────────────────────────────────
TREND_SCORE_MIN       = 62      # минимальный trend_score для "интересной" монеты
LOOKBACK_BARS         = 8       # сколько последних записей брать на монету для оценки тренда
LOOKBACK_EVENTS_HOURS = 6       # сколько часов bot_events смотреть назад
LOOKBACK_BACKTEST_DAYS = 14     # сколько дней брать для бэктеста
MIN_MISSES_FOR_PROPOSAL = 3     # минимум монет с одной причиной блока → предложение
MIN_BACKTEST_SAMPLES  = 5       # минимум samples для валидного бэктеста
BACKTEST_WIN_RATE_MIN = 0.40    # минимальный winrate для "approve"
BACKTEST_RET5_MIN     = 0.0     # минимальный средний ret_5% для "approve"


# ─────────────────────────────────────────────────────────────────────────────
# Датаклассы
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendCandidate:
    sym: str
    tf: str
    trend_score: float            # 0-100: сила зарождающегося тренда
    bot_status: str               # "entered" | "blocked" | "candidate" | "no_signal"
    features: dict = field(default_factory=dict)   # последний снимок фич
    last_bar_ts: int = 0          # ms timestamp последнего бара
    last_price: float = 0.0
    signal_type: str = ""         # тип сигнала если есть
    block_reasons: list[str] = field(default_factory=list)  # из bot_events


@dataclass
class MissReport:
    sym: str
    tf: str
    trend_score: float
    config_param: str              # "IMPULSE_SPEED_1H_RSI_MAX_BULL"
    current_value: float           # 76.0
    actual_value: float            # 79.0 (RSI монеты при блокировке)
    threshold_value: float         # 76.0 (использованный порог)
    rule: BlockRule
    block_reason_text: str
    ts: str = ""


@dataclass
class ConfigProposal:
    config_param: str
    current_value: float
    proposed_value: float
    affected_syms: list[str]
    miss_reports: list[MissReport]
    rule: BlockRule
    rationale: str = ""


@dataclass
class ValidationResult:
    proposal: ConfigProposal
    new_entries_count: int
    new_entries_avg_ret5: float
    new_entries_win_rate: float
    new_entries_avg_ret10: float
    verdict: str                   # "approve" | "reject" | "inconclusive"
    reason: str


@dataclass
class ScoutReport:
    ts: str
    candidates_total: int
    candidates_trending: int       # trend_score >= TREND_SCORE_MIN
    entered: int
    blocked_trending: int
    proposals: list[ConfigProposal]
    validated: list[ValidationResult]
    applied: list[dict]            # {"param": ..., "old": ..., "new": ...}
    telegram_text: str = ""
    has_findings: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Trend Scanner
# ─────────────────────────────────────────────────────────────────────────────

def _score_trend(rows: list[dict]) -> float:
    """
    Оценивает силу зарождающегося устойчивого тренда по последним N записям.

    Ключевое: ищет ТРАЕКТОРИЮ (slope ускоряется, ADX растёт), а не мгновенный сигнал.
    Это отличает трендовый вход от пикового входа.

    Returns: float 0-100
    """
    if not rows:
        return 0.0

    # Берём последнюю запись для абсолютных значений
    last = rows[-1]
    f = last.get("f", {})

    score = 0.0

    # ── 1. EMA slope (25 pts) ─────────────────────────────────────────────
    # Slope > 0 = EMA20 растёт. Slope ускоряется = тренд набирает силу.
    slope = f.get("slope", 0.0)
    if slope >= 0.8:
        score += 25
    elif slope >= 0.4:
        score += 18
    elif slope >= 0.2:
        score += 12
    elif slope >= 0.1:
        score += 6
    elif slope <= 0:
        score += 0  # нет тренда

    # Бонус за ускорение slope (trajectory)
    if len(rows) >= 3:
        slopes = [r.get("f", {}).get("slope", 0.0) for r in rows[-3:]]
        if slopes[2] > slopes[1] > slopes[0] > 0:
            score += 8  # slope ускоряется 3 бара подряд

    # ── 2. ADX (20 pts) ───────────────────────────────────────────────────
    adx = f.get("adx", 0.0)
    if adx >= 30:
        score += 20
    elif adx >= 22:
        score += 15
    elif adx >= 18:
        score += 10
    elif adx >= 14:
        score += 5

    # ── 3. RSI sweet spot (20 pts) ────────────────────────────────────────
    # 55-68: ранний тренд, не перегрет — идеально
    # < 50: нет импульса; > 75: перегрет
    rsi = f.get("rsi", 50.0)
    if 55 <= rsi <= 68:
        score += 20
    elif 50 <= rsi <= 73:
        score += 13
    elif 45 <= rsi <= 77:
        score += 6
    else:
        score += 0

    # ── 4. Volume (15 pts) ────────────────────────────────────────────────
    vol_x = f.get("vol_x", 1.0)
    if vol_x >= 1.8:
        score += 15
    elif vol_x >= 1.3:
        score += 10
    elif vol_x >= 1.0:
        score += 6
    elif vol_x >= 0.7:
        score += 2

    # ── 5. EMA structure (15 pts) ─────────────────────────────────────────
    # Price нед алеко от EMA20 (не перегрет), EMA20 выше EMA50
    ema20_edge = f.get("close_vs_ema20", 0.0)   # % выше EMA20
    ema_fan    = f.get("ema20_vs_ema50", 0.0)   # % EMA20 выше EMA50

    if 0 < ema20_edge < 1.5:
        score += 10   # цена близко к EMA20 — ранний вход
    elif 0 < ema20_edge < 3.0:
        score += 6
    elif ema20_edge <= 0:
        score += 0    # ниже EMA20

    if ema_fan > 0.3:
        score += 5
    elif ema_fan > 0:
        score += 2

    # ── 6. MACD подтверждение (5 pts) ─────────────────────────────────────
    macd = f.get("macd_hist_norm", 0.0)
    if macd > 0.001:
        score += 5
    elif macd > 0:
        score += 2

    # Нормализуем: теоретический максимум ~108, приводим к 100
    return min(round(score / 1.08, 1), 100.0)


def scan_trend_candidates(
    lookback_hours: float = 2.0,
    tf_filter: str = "",           # "" = все, "15m", "1h"
) -> list[TrendCandidate]:
    """
    Phase 1: читает последние записи critic_dataset, возвращает TrendCandidate список.

    Для каждой (sym, tf) берёт последние LOOKBACK_BARS записей,
    вычисляет trend_score и определяет bot_status.
    """
    if not CRITIC_FILE.exists():
        log.warning("critic_dataset.jsonl not found")
        return []

    cutoff_ms = int((time.time() - lookback_hours * 3600) * 1000)
    cutoff_ms = min(cutoff_ms, int((time.time() - 0.5 * 3600) * 1000))  # минимум 30 мин

    # Читаем и группируем по (sym, tf)
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    with CRITIC_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            tf = r.get("tf", "")
            if tf_filter and tf != tf_filter:
                continue
            bar_ts = r.get("bar_ts", 0)
            if bar_ts < cutoff_ms:
                continue
            key = (r.get("sym", ""), tf)
            by_key[key].append(r)

    candidates: list[TrendCandidate] = []
    for (sym, tf), rows in by_key.items():
        if not sym:
            continue
        # Сортируем по bar_ts, берём последние LOOKBACK_BARS
        rows_sorted = sorted(rows, key=lambda r: r.get("bar_ts", 0))
        rows_recent = rows_sorted[-LOOKBACK_BARS:]

        trend_score = _score_trend(rows_recent)
        last = rows_recent[-1]
        f = last.get("f", {})
        decision = last.get("decision", {})
        action = decision.get("action", "no_signal")

        # Маппинг action → bot_status
        if action == "take":
            bot_status = "entered"
        elif action == "candidate":
            bot_status = "candidate"
        elif action == "blocked":
            bot_status = "blocked"
        else:
            bot_status = "no_signal"

        # Приблизительная цена через ema20_edge
        # (нет прямой цены в f, но можно оценить)
        last_price = 0.0

        candidates.append(TrendCandidate(
            sym=sym,
            tf=tf,
            trend_score=trend_score,
            bot_status=bot_status,
            features=f,
            last_bar_ts=last.get("bar_ts", 0),
            signal_type=last.get("signal_type", ""),
        ))

    return sorted(candidates, key=lambda c: c.trend_score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def _load_recent_bot_events(hours: float) -> list[dict]:
    """Загружает события из bot_events.jsonl за последние N часов."""
    if not BOT_EVENTS_FILE.exists():
        return []
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    events = []
    with BOT_EVENTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("ts", "") >= cutoff_ts:
                    events.append(e)
            except Exception:
                continue
    return events


def _load_recent_critic_blocked(hours: float) -> list[dict]:
    """Загружает заблокированные записи critic_dataset за последние N часов."""
    if not CRITIC_FILE.exists():
        return []
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = []
    with CRITIC_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                d = r.get("decision", {})
                if d.get("action") == "blocked" and r.get("ts_signal", "") >= cutoff_ts:
                    rows.append(r)
            except Exception:
                continue
    return rows


def diagnose_misses(
    candidates: list[TrendCandidate],
    lookback_hours: float = LOOKBACK_EVENTS_HOURS,
) -> list[MissReport]:
    """
    Phase 2: для высокоскоринговых монет со статусом blocked/candidate
    ищет причину блокировки и маппирует на config-параметр.
    """
    # Фильтруем: только trending + не entered
    missed = [
        c for c in candidates
        if c.trend_score >= TREND_SCORE_MIN and c.bot_status in ("blocked", "candidate", "no_signal")
    ]
    if not missed:
        return []

    # Строим индекс sym→blocked events из bot_events.jsonl
    recent_events = _load_recent_bot_events(lookback_hours)
    events_by_sym: dict[str, list[dict]] = defaultdict(list)
    for e in recent_events:
        if e.get("event") == "blocked":
            events_by_sym[e.get("sym", "")].append(e)

    # Также из critic_dataset blocked records
    critic_blocked = _load_recent_critic_blocked(lookback_hours)
    critic_by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in critic_blocked:
        critic_by_sym[r.get("sym", "")].append(r)

    reports: list[MissReport] = []

    for cand in missed:
        sym = cand.sym
        tf = cand.tf

        # Собираем причины из bot_events
        sym_events = [
            e for e in events_by_sym.get(sym, [])
            if e.get("tf", tf) == tf
        ]

        # Собираем причины из critic_dataset
        sym_critic = [
            r for r in critic_by_sym.get(sym, [])
            if r.get("tf", tf) == tf
        ]

        # Пробуем найти matching rule
        for source, items in [("bot_events", sym_events), ("critic", sym_critic)]:
            for item in items:
                if source == "bot_events":
                    reason_text = item.get("reason", "")
                    reason_code = item.get("signal_type", "")
                    item_tf = item.get("tf", tf)
                else:
                    d = item.get("decision", {})
                    reason_text = d.get("reason", "")
                    reason_code = d.get("reason_code", "")
                    item_tf = item.get("tf", tf)

                rules = find_rules(reason_code, reason_text, item_tf)
                for rule in rules:
                    vals = extract_values(rule, reason_text)
                    if vals is None:
                        continue
                    actual_val, threshold_val = vals
                    current_cfg = _get_config_value(rule.config_param)
                    if current_cfg is None:
                        continue

                    reports.append(MissReport(
                        sym=sym,
                        tf=item_tf,
                        trend_score=cand.trend_score,
                        config_param=rule.config_param,
                        current_value=current_cfg,
                        actual_value=actual_val,
                        threshold_value=threshold_val,
                        rule=rule,
                        block_reason_text=reason_text,
                        ts=item.get("ts", ""),
                    ))

    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Proposal
# ─────────────────────────────────────────────────────────────────────────────

def propose_changes(
    misses: list[MissReport],
    min_misses: int = MIN_MISSES_FOR_PROPOSAL,
) -> list[ConfigProposal]:
    """
    Phase 3: агрегирует MissReport по config_param.
    Если ≥ min_misses разных монет заблокированы одним параметром → предлагает изменение.
    """
    # Группируем по config_param
    by_param: dict[str, list[MissReport]] = defaultdict(list)
    for m in misses:
        by_param[m.config_param].append(m)

    proposals = []
    for param, param_misses in by_param.items():
        # Уникальные монеты
        unique_syms = list({m.sym for m in param_misses})
        if len(unique_syms) < min_misses:
            continue

        rule = param_misses[0].rule
        current_value = param_misses[0].current_value
        actual_values = [m.actual_value for m in param_misses]
        proposed_value = compute_proposed_value(rule, current_value, actual_values)

        # Не предлагаем если изменение слишком маленькое
        delta = abs(proposed_value - current_value)
        if delta < 0.01 and not rule.is_integer:
            continue

        rationale = (
            f"{len(unique_syms)} trending монет заблокированы {param} "
            f"(trend_score avg={sum(m.trend_score for m in param_misses)/len(param_misses):.0f}). "
            f"Фактические значения: {[round(v,1) for v in sorted(set(actual_values))]}. "
            f"Предлагается: {current_value} → {proposed_value} "
            f"({rule.direction}, риск={rule.risk})."
        )

        proposals.append(ConfigProposal(
            config_param=param,
            current_value=current_value,
            proposed_value=proposed_value,
            affected_syms=unique_syms,
            miss_reports=param_misses,
            rule=rule,
            rationale=rationale,
        ))

    # Сортируем: сначала low risk, затем по количеству затронутых монет
    risk_order = {"low": 0, "medium": 1, "high": 2}
    proposals.sort(key=lambda p: (risk_order.get(p.rule.risk, 9), -len(p.affected_syms)))
    return proposals


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Backtest Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_proposal(proposal: ConfigProposal) -> ValidationResult:
    """
    Phase 4: бэктест на исторических blocked-записях critic_dataset.

    Логика:
      - Берём все blocked-записи за последние LOOKBACK_BACKTEST_DAYS дней
      - Фильтруем те, которые были заблокированы из-за proposal.config_param
      - Проверяем: если бы порог был proposed_value, запись прошла бы?
      - Смотрим на ret_5/ret_10 таких записей
    """
    if not CRITIC_FILE.exists():
        return ValidationResult(
            proposal=proposal,
            new_entries_count=0,
            new_entries_avg_ret5=0.0,
            new_entries_win_rate=0.0,
            new_entries_avg_ret10=0.0,
            verdict="inconclusive",
            reason="critic_dataset not found",
        )

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_BACKTEST_DAYS)).isoformat()
    rule = proposal.rule

    new_entries: list[dict] = []

    with CRITIC_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            if r.get("ts_signal", "") < cutoff_ts:
                continue

            d = r.get("decision", {})
            if d.get("action") != "blocked":
                continue

            reason_code = d.get("reason_code", "")
            reason_text = d.get("reason", "")
            tf = r.get("tf", "")

            # Проверяем соответствие правилу
            matched_rules = find_rules(reason_code, reason_text, tf)
            if not any(mr.config_param == proposal.config_param for mr in matched_rules):
                continue

            # Проверяем: прошёл бы этот record с новым порогом?
            vals = extract_values(rule, reason_text)
            if vals is None:
                continue
            actual_val, _ = vals

            # Симуляция: pass если actual_val теперь в пределах нового порога
            if rule.direction == "increase":
                would_pass = actual_val <= proposal.proposed_value
            else:
                would_pass = actual_val >= proposal.proposed_value

            if not would_pass:
                continue

            # Этот record прошёл бы → смотрим на результат
            labels = r.get("labels", {})
            ret5 = labels.get("ret_5")
            ret10 = labels.get("ret_10")
            if ret5 is None:
                continue

            new_entries.append({"ret5": ret5, "ret10": ret10 or 0.0})

    n = len(new_entries)
    if n < MIN_BACKTEST_SAMPLES:
        return ValidationResult(
            proposal=proposal,
            new_entries_count=n,
            new_entries_avg_ret5=0.0,
            new_entries_win_rate=0.0,
            new_entries_avg_ret10=0.0,
            verdict="inconclusive",
            reason=f"Слишком мало samples: {n} < {MIN_BACKTEST_SAMPLES}",
        )

    avg_ret5 = sum(e["ret5"] for e in new_entries) / n
    avg_ret10 = sum(e["ret10"] for e in new_entries) / n
    win_rate = sum(1 for e in new_entries if e["ret5"] > 0) / n

    # Вердикт
    if avg_ret5 >= BACKTEST_RET5_MIN and win_rate >= BACKTEST_WIN_RATE_MIN:
        verdict = "approve"
        reason = (
            f"{n} новых входов: avg_ret5={avg_ret5:+.2f}%, "
            f"win={win_rate*100:.0f}%, avg_ret10={avg_ret10:+.2f}%"
        )
    elif avg_ret5 < -0.5 or win_rate < 0.30:
        verdict = "reject"
        reason = (
            f"Убыточный бэктест: avg_ret5={avg_ret5:+.2f}%, win={win_rate*100:.0f}%"
        )
    else:
        verdict = "inconclusive"
        reason = (
            f"Слабый сигнал: avg_ret5={avg_ret5:+.2f}%, win={win_rate*100:.0f}%"
        )

    return ValidationResult(
        proposal=proposal,
        new_entries_count=n,
        new_entries_avg_ret5=avg_ret5,
        new_entries_win_rate=win_rate,
        new_entries_avg_ret10=avg_ret10,
        verdict=verdict,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Apply + Report
# ─────────────────────────────────────────────────────────────────────────────

def _get_config_value(param: str) -> float | None:
    """Читает текущее значение числового параметра из config.py."""
    return getattr(config, param, None)


def _apply_config_change(param: str, new_value: float, is_integer: bool = False) -> bool:
    """
    Записывает новое значение параметра в config.py через точечную замену строки.
    Безопасно: меняет только конкретную строку, остальное не трогает.
    """
    if not CONFIG_FILE.exists():
        log.error("config.py not found at %s", CONFIG_FILE)
        return False

    content = CONFIG_FILE.read_text(encoding="utf-8")

    # Ищем строку с этим параметром
    # Паттерн: `PARAM_NAME: type = value  # comment`
    if is_integer:
        pattern = rf"^({re.escape(param)}\s*:\s*int\s*=\s*)(\d+)(.*?)$"
        new_val_str = str(int(new_value))
    else:
        pattern = rf"^({re.escape(param)}\s*:\s*float\s*=\s*)(\d+\.?\d*)(.*?)$"
        new_val_str = str(new_value)

    def replacer(m):
        old_val = m.group(2)
        comment = m.group(3)
        # Добавляем пометку в комментарий если ещё нет
        ts = datetime.now(timezone.utc).strftime("%d.%m.%Y")
        tag = f"  # scout:{ts} was {old_val}"
        # Убираем старый scout-тег если есть
        clean_comment = re.sub(r"\s*#\s*scout:.*$", "", comment)
        return m.group(1) + new_val_str + clean_comment + tag

    new_content, n_subs = re.subn(pattern, replacer, content, flags=re.MULTILINE)
    if n_subs == 0:
        log.warning("Could not find param %s in config.py", param)
        return False

    CONFIG_FILE.write_text(new_content, encoding="utf-8")
    log.info("config.py updated: %s = %s", param, new_val_str)
    return True


def _log_changelog(entry: dict) -> None:
    """Добавляет запись в trend_scout_changelog.jsonl."""
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with CHANGELOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def apply_approved_changes(
    validated: list[ValidationResult],
    auto_apply_risk: str = "low",
) -> list[dict]:
    """
    Phase 5: применяет изменения с verdict='approve' и risk <= auto_apply_risk.
    Возвращает список применённых изменений.
    """
    risk_order = {"low": 0, "medium": 1, "high": 2}
    auto_threshold = risk_order.get(auto_apply_risk, 0)

    applied = []
    for vr in validated:
        if vr.verdict != "approve":
            continue

        p = vr.proposal
        rule = p.rule
        if risk_order.get(rule.risk, 9) > auto_threshold:
            continue

        old_value = p.current_value
        new_value = p.proposed_value

        success = _apply_config_change(p.config_param, new_value, is_integer=rule.is_integer)
        if success:
            entry = {
                "param": p.config_param,
                "old_value": old_value,
                "new_value": new_value,
                "affected_syms": p.affected_syms,
                "backtest_n": vr.new_entries_count,
                "backtest_ret5": round(vr.new_entries_avg_ret5, 3),
                "backtest_win": round(vr.new_entries_win_rate, 3),
                "rationale": p.rationale,
            }
            _log_changelog(entry)
            applied.append(entry)
            log.info(
                "Auto-applied: %s %s → %s (backtest: n=%d ret5=%.2f%% win=%.0f%%)",
                p.config_param, old_value, new_value,
                vr.new_entries_count, vr.new_entries_avg_ret5, vr.new_entries_win_rate * 100,
            )

    return applied


# ─────────────────────────────────────────────────────────────────────────────
# Report Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_telegram_text(report: ScoutReport) -> str:
    lines = ["🔍 <b>Trend Scout Report</b>"]

    lines.append(
        f"\n📈 Монет с трендом: <b>{report.candidates_trending}</b> из {report.candidates_total} "
        f"(порог score≥{TREND_SCORE_MIN})"
    )
    lines.append(
        f"✅ Бот вошёл: <b>{report.entered}</b>  ❌ Пропустил: <b>{report.blocked_trending}</b>"
    )

    if report.applied:
        lines.append("\n🔧 <b>Авто-применено:</b>")
        for a in report.applied:
            lines.append(
                f"  • {a['param']}: {a['old_value']} → {a['new_value']} "
                f"(n={a['backtest_n']}, ret5={a['backtest_ret5']:+.2f}%, "
                f"win={a['backtest_win']*100:.0f}%)"
            )

    pending = [
        vr for vr in report.validated
        if vr.verdict == "approve" and vr.proposal.rule.risk in ("medium", "high")
    ]
    if pending:
        lines.append("\n⚠️ <b>Требует подтверждения:</b>")
        for vr in pending:
            p = vr.proposal
            lines.append(
                f"  • {p.config_param}: {p.current_value} → {p.proposed_value} "
                f"(риск={p.rule.risk}, монет={len(p.affected_syms)})"
            )
            lines.append(
                f"    Бэктест: n={vr.new_entries_count}, "
                f"ret5={vr.new_entries_avg_ret5:+.2f}%, "
                f"win={vr.new_entries_win_rate*100:.0f}%"
            )

    rejected = [vr for vr in report.validated if vr.verdict == "reject"]
    if rejected:
        lines.append("\n🚫 <b>Отклонено бэктестом:</b>")
        for vr in rejected:
            lines.append(f"  • {vr.proposal.config_param}: {vr.reason}")

    inconclusive_proposals = [
        p for p in report.proposals
        if not any(vr.proposal.config_param == p.config_param for vr in report.validated)
    ]
    if inconclusive_proposals:
        lines.append("\n📊 <b>Мало данных для бэктеста:</b>")
        for p in inconclusive_proposals[:3]:
            lines.append(
                f"  • {p.config_param} ({len(p.affected_syms)} монет: "
                f"{', '.join(p.affected_syms[:3])})"
            )

    if not report.applied and not pending and not report.proposals:
        lines.append("\n✅ Расхождений не обнаружено — всё в норме.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_skill_missed_trends(missed_file: Path,
                             lookback_hours: float = 168.0) -> list[TrendCandidate]:
    """Hybrid architecture (2026-05-05): consume skill-evaluator's missed
    trends list as scout's input source instead of heuristic scan.

    File format (written by _weekly_signal_eval_with_tg.py):
      {
        "missed_trends": [
          {"symbol": "ICPUSDT", "tf": "15m", "start_ts": "...",
           "gain_pct": 6.8, "trend_score": 77.2, ...}
        ]
      }

    Returns synthetic TrendCandidate list (status='blocked') so existing
    diagnose_misses() can reuse its bot_events/critic_dataset matching.

    Spec: docs/specs/features/signal-evaluator-integration-spec.md
          (Hybrid architecture section)
    """
    if not missed_file.exists():
        log.warning("Skill missed-trends file not found: %s", missed_file)
        return []
    try:
        payload = json.loads(missed_file.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Skill missed-trends parse failed: %s", e)
        return []

    out: list[TrendCandidate] = []
    for t in payload.get("missed_trends", []):
        sym = (t.get("symbol") or "").upper()
        tf = t.get("tf", "15m")
        if not sym:
            continue
        out.append(TrendCandidate(
            sym=sym,
            tf=tf,
            trend_score=float(t.get("trend_score") or 70.0),
            bot_status="blocked",  # by skill definition: missed = not entered
            features={"gain_pct": t.get("gain_pct"),
                      "duration_bars": t.get("duration_bars"),
                      "skill_start_ts": t.get("start_ts"),
                      "source": "skill-evaluator"},
            last_bar_ts=0,
            last_price=0.0,
            signal_type="",
            block_reasons=[],
        ))
    log.info("Loaded %d missed trends from skill (%s)", len(out), missed_file.name)
    return out


async def run_pipeline(
    auto_apply_risk: str = "low",
    lookback_hours: float = 4.0,
    dry_run: bool = False,
    *,
    source: str = "heuristic",
    missed_file: Path | None = None,
) -> ScoutReport:
    """
    Запускает полный пайплайн Trend Scout (Phase 1–5).

    Args:
        auto_apply_risk: максимальный риск для авто-применения ("low" | "medium")
        lookback_hours: период сканирования
        dry_run: если True — не писать в config.py
        source: "heuristic" (default, scan_trend_candidates) or "skill"
                (read evaluation_output/skill_missed_trends.json).
        missed_file: explicit path for source="skill". Defaults to
                     <repo>/evaluation_output/skill_missed_trends.json.

    Returns:
        ScoutReport с полными результатами
    """
    import asyncio
    ts = datetime.now(timezone.utc).isoformat()
    log.info("Trend Scout pipeline started (source=%s, lookback=%.1fh, "
             "auto_apply=%s, dry_run=%s)",
             source, lookback_hours, auto_apply_risk, dry_run)

    # Phase 1: choose data source
    if source == "skill":
        # Hybrid: skill-fed input (week-long lookback for missed-trends file)
        if missed_file is None:
            missed_file = WORKSPACE_ROOT / "evaluation_output" / "skill_missed_trends.json"
        candidates = load_skill_missed_trends(missed_file)
        # Skill mode uses 7-day window for upstream context
        lookback_hours = max(lookback_hours, 24.0 * 7)
    else:
        try:
            candidates = await asyncio.to_thread(
                scan_trend_candidates, lookback_hours
            )
        except Exception as e:
            log.exception("Phase 1 failed: %s", e)
            candidates = []

    trending = [c for c in candidates if c.trend_score >= TREND_SCORE_MIN]
    entered = [c for c in trending if c.bot_status == "entered"]
    blocked_trending = [c for c in trending if c.bot_status in ("blocked", "candidate", "no_signal")]

    log.info("Phase 1: %d candidates, %d trending, %d entered, %d missed",
             len(candidates), len(trending), len(entered), len(blocked_trending))

    # Phase 2
    try:
        misses = await asyncio.to_thread(diagnose_misses, candidates, lookback_hours)
    except Exception as e:
        log.exception("Phase 2 failed: %s", e)
        misses = []

    log.info("Phase 2: %d miss reports", len(misses))

    # Phase 3
    try:
        proposals = await asyncio.to_thread(propose_changes, misses)
    except Exception as e:
        log.exception("Phase 3 failed: %s", e)
        proposals = []

    log.info("Phase 3: %d proposals", len(proposals))

    # Phase 4
    validated: list[ValidationResult] = []
    for proposal in proposals:
        try:
            vr = await asyncio.to_thread(validate_proposal, proposal)
            validated.append(vr)
            log.info(
                "Phase 4: %s → %s → %s (%s)",
                proposal.config_param, proposal.proposed_value, vr.verdict, vr.reason
            )
        except Exception as e:
            log.exception("Phase 4 validation failed for %s: %s", proposal.config_param, e)

    # Phase 5
    applied: list[dict] = []
    if not dry_run:
        try:
            applied = await asyncio.to_thread(
                apply_approved_changes, validated, auto_apply_risk
            )
        except Exception as e:
            log.exception("Phase 5 failed: %s", e)
    else:
        log.info("Phase 5: dry_run — пропускаем применение изменений")

    # Собираем отчёт
    report = ScoutReport(
        ts=ts,
        candidates_total=len(candidates),
        candidates_trending=len(trending),
        entered=len(entered),
        blocked_trending=len(blocked_trending),
        proposals=proposals,
        validated=validated,
        applied=applied,
        has_findings=bool(proposals or applied),
    )
    report.telegram_text = _build_telegram_text(report)

    # Сохраняем отчёт
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "ts": ts,
            "candidates_total": report.candidates_total,
            "candidates_trending": report.candidates_trending,
            "entered": report.entered,
            "blocked_trending": report.blocked_trending,
            "proposals": [
                {
                    "config_param": p.config_param,
                    "current_value": p.current_value,
                    "proposed_value": p.proposed_value,
                    "affected_syms": p.affected_syms,
                    "risk": p.rule.risk,
                    "rationale": p.rationale,
                }
                for p in proposals
            ],
            "validated": [
                {
                    "config_param": vr.proposal.config_param,
                    "verdict": vr.verdict,
                    "new_entries_count": vr.new_entries_count,
                    "new_entries_avg_ret5": round(vr.new_entries_avg_ret5, 3),
                    "new_entries_win_rate": round(vr.new_entries_win_rate, 3),
                    "reason": vr.reason,
                }
                for vr in validated
            ],
            "applied": applied,
        }
        REPORT_FILE.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Could not save report file: %s", e)

    log.info(
        "Trend Scout done: proposals=%d, applied=%d, approve=%d, reject=%d",
        len(proposals),
        len(applied),
        sum(1 for vr in validated if vr.verdict == "approve"),
        sum(1 for vr in validated if vr.verdict == "reject"),
    )
    return report


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", choices=["heuristic", "skill"], default="heuristic",
                   help="heuristic = scan_trend_candidates (default); "
                        "skill = read evaluation_output/skill_missed_trends.json "
                        "(hybrid architecture, 2026-05-05).")
    p.add_argument("--missed-file", type=Path, default=None,
                   help="explicit path for --source skill")
    p.add_argument("--auto-apply-risk", choices=["low", "medium", "high"],
                   default="low")
    p.add_argument("--lookback-hours", type=float, default=4.0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    report = asyncio.run(run_pipeline(
        dry_run=args.dry_run,
        source=args.source,
        missed_file=args.missed_file,
        auto_apply_risk=args.auto_apply_risk,
        lookback_hours=args.lookback_hours,
    ))
    print(report.telegram_text)
