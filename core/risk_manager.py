"""
Risk Manager — управление рисками с kill switches.

Вдохновлено:
- ent0n29/polybot: configurable limits, kill switches, exposure caps
- Polymarket-Trading-Bot: risk management модуль
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.strategy_engine import Signal
from core.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Лимиты риска."""
    max_position_size: float = 100.0        # макс. $ на позицию
    max_daily_loss: float = 50.0            # макс. дневной убыток
    max_open_positions: int = 10            # макс. открытых позиций
    max_exposure_pct: float = 0.50          # макс. 50% баланса в позициях
    kill_switch_loss_pct: float = 0.15      # kill switch при -15%
    max_single_market_exposure: float = 0.20  # макс. 20% в одном рынке
    min_balance_reserve: float = 100.0      # мин. резерв на балансе
    max_trades_per_hour: int = 20           # макс. сделок в час
    cooldown_after_loss_streak: int = 3     # пауза после N убыточных подряд
    cooldown_minutes: int = 30             # длительность паузы


class RiskCheckResult:
    """Результат проверки рисков."""
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self):
        return self.passed


class RiskManager:
    """
    Менеджер рисков. Проверяет каждый сигнал перед исполнением.
    Kill switch автоматически останавливает торговлю при критических потерях.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._trade_timestamps: List[float] = []
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._daily_loss = 0.0
        self._daily_reset_date = datetime.now().date()

    @property
    def is_trading_allowed(self) -> bool:
        return not self._kill_switch_active

    def check_signal(self, signal: Signal, trader: PaperTrader) -> RiskCheckResult:
        """Проверить сигнал на соответствие лимитам."""

        # Kill switch
        if self._kill_switch_active:
            return RiskCheckResult(False, f"🛑 Kill switch: {self._kill_switch_reason}")

        # Cooldown после серии убытков
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now()).seconds // 60
            return RiskCheckResult(False, f"⏸ Cooldown ({remaining} мин.)")

        # Сброс дневного убытка
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._daily_loss = 0.0
            self._daily_reset_date = today

        # Дневной убыток
        if self._daily_loss >= self.limits.max_daily_loss:
            return RiskCheckResult(False, f"🚫 Дневной лимит убытков: ${self._daily_loss:.2f}")

        # Количество позиций
        if len(trader.positions) >= self.limits.max_open_positions:
            return RiskCheckResult(False, f"🚫 Макс. позиций: {self.limits.max_open_positions}")

        # Размер позиции
        position_value = trader.balance * signal.suggested_size
        if position_value > self.limits.max_position_size:
            return RiskCheckResult(
                False, f"🚫 Размер ${position_value:.0f} > лимит ${self.limits.max_position_size:.0f}"
            )

        # Общая экспозиция
        total_exposure = sum(p.cost for p in trader.positions.values())
        exposure_pct = (total_exposure + position_value) / trader.total_equity if trader.total_equity > 0 else 1
        if exposure_pct > self.limits.max_exposure_pct:
            return RiskCheckResult(
                False, f"🚫 Экспозиция {exposure_pct:.0%} > лимит {self.limits.max_exposure_pct:.0%}"
            )

        # Экспозиция на один рынок
        market_exposure = sum(
            p.cost for p in trader.positions.values()
            if p.market_id == signal.market_id
        )
        market_pct = (market_exposure + position_value) / trader.total_equity if trader.total_equity > 0 else 1
        if market_pct > self.limits.max_single_market_exposure:
            return RiskCheckResult(
                False, f"🚫 Рынок exposure {market_pct:.0%} > лимит {self.limits.max_single_market_exposure:.0%}"
            )

        # Резерв баланса
        if trader.balance - position_value < self.limits.min_balance_reserve:
            return RiskCheckResult(False, f"🚫 Резерв баланса < ${self.limits.min_balance_reserve:.0f}")

        # Частота сделок
        now = datetime.now().timestamp()
        hour_ago = now - 3600
        self._trade_timestamps = [t for t in self._trade_timestamps if t > hour_ago]
        if len(self._trade_timestamps) >= self.limits.max_trades_per_hour:
            return RiskCheckResult(False, f"🚫 Лимит сделок: {self.limits.max_trades_per_hour}/час")

        # Kill switch: общий убыток
        loss_pct = (trader.initial_balance - trader.total_equity) / trader.initial_balance
        if loss_pct >= self.limits.kill_switch_loss_pct:
            self._activate_kill_switch(
                f"Убыток {loss_pct:.1%} превысил порог {self.limits.kill_switch_loss_pct:.1%}"
            )
            return RiskCheckResult(False, f"🛑 Kill switch активирован!")

        return RiskCheckResult(True)

    def record_trade(self, pnl: float):
        """Записать результат сделки для отслеживания серий."""
        self._trade_timestamps.append(datetime.now().timestamp())

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._consecutive_losses += 1

            if self._consecutive_losses >= self.limits.cooldown_after_loss_streak:
                self._cooldown_until = datetime.now() + timedelta(
                    minutes=self.limits.cooldown_minutes
                )
                logger.warning(
                    f"Cooldown activated: {self._consecutive_losses} consecutive losses. "
                    f"Paused for {self.limits.cooldown_minutes} min."
                )
                self._consecutive_losses = 0
        else:
            self._consecutive_losses = 0

    def _activate_kill_switch(self, reason: str):
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def reset_kill_switch(self):
        """Ручной сброс kill switch (только по команде пользователя)."""
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._consecutive_losses = 0
        self._cooldown_until = None
        logger.info("Kill switch reset manually")

    def status(self) -> str:
        """Статус риск-менеджера для Telegram."""
        if self._kill_switch_active:
            return f"🛑 *KILL SWITCH ACTIVE*\nПричина: {self._kill_switch_reason}\nИспользуйте /risk\\_reset для сброса"

        cooldown_str = ""
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now()).seconds // 60
            cooldown_str = f"\n⏸ Cooldown: {remaining} мин."

        return (
            f"🛡 *Risk Manager*\n"
            f"Статус: {'✅ Активен' if self.is_trading_allowed else '🛑 Остановлен'}\n"
            f"Дневной убыток: `${self._daily_loss:.2f}` / `${self.limits.max_daily_loss:.2f}`\n"
            f"Убытков подряд: {self._consecutive_losses} / {self.limits.cooldown_after_loss_streak}\n"
            f"Макс. позиция: `${self.limits.max_position_size:.0f}`\n"
            f"Макс. позиций: {self.limits.max_open_positions}\n"
            f"Kill switch: `{self.limits.kill_switch_loss_pct:.0%}`"
            f"{cooldown_str}"
        )
