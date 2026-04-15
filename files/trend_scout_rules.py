"""
trend_scout_rules.py — Правила маппинга причин блокировки на параметры config.py

Каждое правило описывает:
  - reason_code: код из decision.reason_code в critic_dataset (или regex для bot_events)
  - config_param: имя параметра в config.py
  - direction: "increase" или "decrease" — в какую сторону сдвигать
  - extract_pattern: regex для извлечения фактического/порогового значения из reason
  - safe_range: (min, max) допустимых значений
  - max_change_pct: максимальный % изменения за один раз (безопасность)
  - risk: "low" (авто-применение) | "medium" (запрос подтверждения) | "high" (только отчёт)
  - is_integer: True если параметр целочисленный
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlockRule:
    reason_code: str           # совпадает с decision.reason_code ИЛИ regex для reason-строки
    config_param: str          # имя параметра в config.py
    direction: str             # "increase" | "decrease"
    extract_pattern: str       # regex, group(1)=actual, group(2)=threshold
    safe_range: tuple          # (min_allowed, max_allowed)
    max_change_pct: float      # максимум % сдвига за один шаг
    risk: str                  # "low" | "medium" | "high"
    is_integer: bool = False   # True для int-параметров
    tf_filter: Optional[str] = None  # "15m" | "1h" | None (любой)
    reason_pattern: str = ""   # дополнительный regex для фильтрации строки reason


# ── Реестр правил ──────────────────────────────────────────────────────────────

BLOCK_RULES: list[BlockRule] = [

    # ── 1. Entry score слишком низкий (15m) ───────────────────────────────────
    BlockRule(
        reason_code="entry_score",
        config_param="ENTRY_SCORE_MIN_15M",
        direction="decrease",
        extract_pattern=r"entry score (\d+\.?\d*) < floor (\d+\.?\d*)",
        safe_range=(38.0, 52.0),
        max_change_pct=0.10,
        risk="medium",
        tf_filter="15m",
    ),

    # ── 2. Entry score слишком низкий (1h) ────────────────────────────────────
    BlockRule(
        reason_code="entry_score",
        config_param="ENTRY_SCORE_MIN_1H",
        direction="decrease",
        extract_pattern=r"entry score (\d+\.?\d*) < floor (\d+\.?\d*)",
        safe_range=(44.0, 62.0),
        max_change_pct=0.08,
        risk="medium",
        tf_filter="1h",
    ),

    # ── 3. RSI слишком высокий для 1h impulse_speed (bull day) ────────────────
    BlockRule(
        reason_code="impulse_guard",
        config_param="IMPULSE_SPEED_1H_RSI_MAX_BULL",
        direction="increase",
        extract_pattern=r"RSI (\d+\.?\d*) > (\d+\.?\d*)",
        safe_range=(76.0, 85.0),
        max_change_pct=0.08,
        risk="low",
        tf_filter="1h",
        reason_pattern=r"1h impulse_speed guard",
    ),

    # ── 4. RSI слишком высокий для 1h impulse_speed (обычный день) ───────────
    BlockRule(
        reason_code="impulse_guard",
        config_param="IMPULSE_SPEED_1H_RSI_MAX",
        direction="increase",
        extract_pattern=r"RSI (\d+\.?\d*) > (\d+\.?\d*)",
        safe_range=(68.0, 80.0),
        max_change_pct=0.08,
        risk="low",
        tf_filter="1h",
        reason_pattern=r"1h impulse_speed guard",
    ),

    # ── 5. Cluster cap: слишком много impulse_speed одновременно ──────────────
    BlockRule(
        reason_code="open_cluster_cap",
        config_param="OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX",
        direction="increase",
        extract_pattern=r"already (\d+)/(\d+)",
        safe_range=(2, 8),
        max_change_pct=0.50,
        risk="low",
        is_integer=True,
        reason_pattern=r"15m_impulse",
    ),

    # ── 6. Clone guard: слишком много похожих сетапов ─────────────────────────
    BlockRule(
        reason_code="clone_signal_guard",
        config_param="CLONE_SIGNAL_GUARD_MAX_SIMILAR",
        direction="increase",
        extract_pattern=r"already (\d+)/(\d+)",
        safe_range=(3, 10),
        max_change_pct=0.50,
        risk="low",
        is_integer=True,
    ),

    # ── 7. ML proba вне зоны (нижняя граница) ────────────────────────────────
    BlockRule(
        reason_code="ml_proba",
        config_param="ML_GENERAL_HARD_BLOCK_MIN",
        direction="decrease",
        extract_pattern=r"ml_proba (\d+\.?\d*) outside.*\[(\d+\.?\d*)",
        safe_range=(0.20, 0.42),
        max_change_pct=0.10,
        risk="medium",
    ),

    # ── 8. ADX слишком низкий для 1h impulse_speed ───────────────────────────
    BlockRule(
        reason_code="impulse_guard",
        config_param="IMPULSE_SPEED_1H_ADX_MIN",
        direction="decrease",
        extract_pattern=r"ADX (\d+\.?\d*) < (\d+\.?\d*)",
        safe_range=(14.0, 28.0),
        max_change_pct=0.12,
        risk="low",
        tf_filter="1h",
    ),
]


def find_rules(reason_code: str, reason_text: str, tf: str) -> list[BlockRule]:
    """
    Возвращает список подходящих правил для данного события блокировки.

    Args:
        reason_code: код из decision.reason_code
        reason_text: полный текст причины блокировки
        tf: таймфрейм ("15m" | "1h")
    """
    matched = []
    for rule in BLOCK_RULES:
        # Фильтр по reason_code (нечёткое совпадение)
        if rule.reason_code not in reason_code and reason_code not in rule.reason_code:
            continue
        # Фильтр по tf
        if rule.tf_filter and rule.tf_filter != tf:
            continue
        # Дополнительный паттерн в тексте
        if rule.reason_pattern and not re.search(rule.reason_pattern, reason_text, re.IGNORECASE):
            continue
        # Проверяем что extract_pattern реально матчит
        if not re.search(rule.extract_pattern, reason_text):
            continue
        matched.append(rule)
    return matched


def extract_values(rule: BlockRule, reason_text: str) -> tuple[float, float] | None:
    """
    Извлекает (actual_value, threshold_value) из строки reason.
    Возвращает None если паттерн не совпал.
    """
    m = re.search(rule.extract_pattern, reason_text)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except (IndexError, ValueError):
        return None


def compute_proposed_value(
    rule: BlockRule,
    current_config_value: float,
    actual_values: list[float],   # список фактических значений заблокированных монет
) -> float:
    """
    Вычисляет предлагаемое новое значение параметра.

    Логика: сдвинуть порог так чтобы покрыть max(actual_values) + небольшой запас,
    но не выйти за safe_range и не сдвинуть больше чем max_change_pct.
    """
    if not actual_values:
        return current_config_value

    if rule.direction == "increase":
        # Нужно поднять порог выше максимального фактического значения
        target = max(actual_values) * 1.05  # +5% запас
        proposed = min(target, current_config_value * (1 + rule.max_change_pct))
        proposed = min(proposed, rule.safe_range[1])
        proposed = max(proposed, current_config_value)  # не уменьшать
    else:
        # Нужно снизить порог ниже минимального фактического значения
        target = min(actual_values) * 0.95  # -5% запас
        proposed = max(target, current_config_value * (1 - rule.max_change_pct))
        proposed = max(proposed, rule.safe_range[0])
        proposed = min(proposed, current_config_value)  # не увеличивать

    if rule.is_integer:
        if rule.direction == "increase":
            proposed = max(int(current_config_value) + 1, int(proposed))
        else:
            proposed = min(int(current_config_value) - 1, int(proposed))
        proposed = float(int(proposed))

    return round(proposed, 2)
