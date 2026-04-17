"""
rotation.py — ML-Gated Portfolio Rotation.

Когда `_check_portfolio_limits` блокирует вход (портфель полон) и
существующая score-based ротация (`_find_replaceable_position`) не нашла
замену, этот модуль даёт «вторую попытку»: если у кандидата высокий
ml_proba — выбираем самую слабую по EV позицию и помечаем её на выход
через trail_stop (естественный ATR-exit на следующем поллинге).

Back-test: files/backtest_portfolio_rotation_grid.py
    Best: ml_proba >= 0.62  →  n=211, avg_r5=+0.241%, win=49.8%,
          Sharpe=+2.75, sumPnL5=+50.8% (vs naive MAX+1: +22.8%).

Config:
    ROTATION_ENABLED:             bool  — глобальный флаг
    ROTATION_ML_PROBA_MIN:        float — минимальный ml_proba кандидата
    ROTATION_WEAK_EV_MAX:         float — позиция «слабая» если ranker_ev < этого
    ROTATION_WEAK_BARS_MIN:       int   — минимум баров в позиции до eviction
    ROTATION_PROFIT_PROTECT_PCT:  float — не вытеснять позиции с PnL > этого %
    ROTATION_MAX_PER_POLL:        int   — максимум eviction-ов за один poll
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RotationDecision:
    """Результат проверки — можно ли вытеснить слабую ногу ради кандидата."""
    allowed: bool
    weak_sym: str = ""
    weak_ev: float = 0.0
    weak_bars: int = 0
    weak_pnl: float = 0.0
    reason: str = ""


def _get_cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _position_pnl_pct(pos: Any, last_price: Optional[float]) -> float:
    """Текущий PnL позиции в %, 0.0 если цена неизвестна."""
    if last_price is None or last_price <= 0:
        return 0.0
    try:
        return float(pos.pnl_pct(float(last_price)))
    except Exception:
        entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
        if entry <= 0:
            return 0.0
        return (float(last_price) / entry - 1.0) * 100.0


def find_weakest_leg(
    positions: Dict[str, Any],
    config: Any,
    last_prices: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Ищет позицию с минимальным ranker_ev среди удовлетворяющих критериям:
      - ranker_ev < ROTATION_WEAK_EV_MAX
      - bars_elapsed >= ROTATION_WEAK_BARS_MIN
      - current_pnl_pct <= ROTATION_PROFIT_PROTECT_PCT

    Возвращает symbol самой слабой ноги или None если подходящих нет.
    """
    if not positions:
        return None

    weak_ev_max = float(_get_cfg(config, "ROTATION_WEAK_EV_MAX", -0.40))
    bars_min = int(_get_cfg(config, "ROTATION_WEAK_BARS_MIN", 3))
    profit_protect = float(_get_cfg(config, "ROTATION_PROFIT_PROTECT_PCT", 0.5))
    last_prices = last_prices or {}

    candidates: list[tuple[float, str]] = []  # (ev, sym)
    for sym, pos in positions.items():
        ev = float(getattr(pos, "ranker_ev", 0.0) or 0.0)
        if ev >= weak_ev_max:
            continue
        bars = int(getattr(pos, "bars_elapsed", 0) or 0)
        if bars < bars_min:
            continue
        pnl = _position_pnl_pct(pos, last_prices.get(sym))
        if pnl >= profit_protect:
            continue
        candidates.append((ev, sym))

    if not candidates:
        return None

    candidates.sort()  # меньший EV — слабее
    return candidates[0][1]


def should_rotate(
    candidate_ml_proba: Optional[float],
    positions: Dict[str, Any],
    config: Any,
    last_prices: Optional[Dict[str, float]] = None,
) -> RotationDecision:
    """
    Решает, разрешать ли ML-gated ротацию.
    Вызывается когда существующая score-based ротация не сработала.
    """
    if not bool(_get_cfg(config, "ROTATION_ENABLED", False)):
        return RotationDecision(allowed=False, reason="disabled")

    mlp_min = float(_get_cfg(config, "ROTATION_ML_PROBA_MIN", 0.62))
    if candidate_ml_proba is None:
        return RotationDecision(allowed=False, reason="ml_proba unknown")
    try:
        mp = float(candidate_ml_proba)
    except (TypeError, ValueError):
        return RotationDecision(allowed=False, reason="ml_proba invalid")
    if mp < mlp_min:
        return RotationDecision(
            allowed=False,
            reason=f"ml_proba {mp:.3f} < {mlp_min:.2f}",
        )

    weak_sym = find_weakest_leg(positions, config, last_prices)
    if not weak_sym:
        return RotationDecision(allowed=False, reason="no weak leg (all strong/profitable/fresh)")

    weak_pos = positions[weak_sym]
    last_prices = last_prices or {}
    return RotationDecision(
        allowed=True,
        weak_sym=weak_sym,
        weak_ev=float(getattr(weak_pos, "ranker_ev", 0.0) or 0.0),
        weak_bars=int(getattr(weak_pos, "bars_elapsed", 0) or 0),
        weak_pnl=_position_pnl_pct(weak_pos, last_prices.get(weak_sym)),
        reason=(
            f"evict {weak_sym} (EV={float(getattr(weak_pos, 'ranker_ev', 0.0)):+.2f}, "
            f"bars={int(getattr(weak_pos, 'bars_elapsed', 0))}) "
            f"for mlp={mp:.3f}"
        ),
    )


def evict_position(pos: Any, last_price: Optional[float]) -> bool:
    """
    Помечает позицию на выход, поднимая trail_stop выше текущей цены.
    На следующем поллинге сработает стандартный ATR-trail выход,
    который использует весь штатный пайплайн (SELL-уведомление,
    critic_dataset, cooldown и т.п.).

    Возвращает True если отметка поставлена, False если цена неизвестна.
    """
    if last_price is None or last_price <= 0:
        return False
    try:
        new_stop = float(last_price) * 1.001
        current = float(getattr(pos, "trail_stop", 0.0) or 0.0)
        # Trail только повышается, поэтому сравниваем и ставим максимум
        pos.trail_stop = max(new_stop, current)
        return True
    except Exception:
        return False


def format_rotation_message(
    sym: str,
    mode: str,
    tf: str,
    candidate_ml_proba: float,
    decision: RotationDecision,
) -> str:
    """Строка для Telegram-уведомления."""
    return (
        f"ML-ROTATION\n\n"
        f"evict {decision.weak_sym} (EV={decision.weak_ev:+.2f}, "
        f"{decision.weak_bars}b, PnL={decision.weak_pnl:+.2f}%)\n"
        f"for {sym} [{tf}] {mode} (mlp={candidate_ml_proba:.3f})\n"
        f"backtest: +0.24% avg_r5, +0.71% avg_r10, Sharpe=+2.75"
    )
