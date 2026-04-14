"""
Paper Trader — симулятор торговли без реальных денег.

Вдохновлено:
- agent-next/polymarket-paper-trader: MCP-ready симулятор
- ent0n29/polybot: executor-service с paper trading simulation
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.strategy_engine import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Открытая paper-позиция."""
    id: str
    market_id: str
    market_question: str
    side: str         # "yes" | "no"
    entry_price: float
    shares: float
    cost: float
    current_price: float = 0.0
    opened_at: str = ""
    strategy: str = ""

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost == 0:
            return 0
        return self.unrealized_pnl / self.cost * 100

    @property
    def current_value(self) -> float:
        return self.current_price * self.shares


@dataclass
class PaperTrade:
    """Завершённая paper-сделка."""
    id: str
    market_id: str
    market_question: str
    side: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    strategy: str
    opened_at: str
    closed_at: str
    reason: str = ""


class PaperTrader:
    """
    Симулятор торговли с отслеживанием портфеля и P&L.
    Не требует API ключей или кошелька.
    """

    def __init__(self, initial_balance: float = 10000.0, commission_pct: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_pct = commission_pct

        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.daily_pnl: List[float] = []

        self._peak_balance = initial_balance
        self._total_pnl = 0.0
        self._started_at = datetime.now().isoformat()

    @property
    def total_equity(self) -> float:
        pos_value = sum(p.current_value for p in self.positions.values())
        return self.balance + pos_value

    @property
    def total_pnl(self) -> float:
        return self._total_pnl + sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance == 0:
            return 0
        return (self.total_equity - self.initial_balance) / self.initial_balance * 100

    @property
    def max_drawdown(self) -> float:
        if self._peak_balance == 0:
            return 0
        return (self._peak_balance - self.total_equity) / self._peak_balance * 100

    @property
    def win_rate(self) -> float:
        if not self.trade_history:
            return 0
        wins = sum(1 for t in self.trade_history if t.pnl > 0)
        return wins / len(self.trade_history)

    def execute_signal(
        self,
        signal: Signal,
        market_question: str = "",
    ) -> Optional[PaperPosition]:
        """Исполнить сигнал в paper режиме."""
        side = "yes" if signal.signal_type in (SignalType.BUY_YES,) else "no"
        price = signal.suggested_price

        # Размер позиции
        position_value = self.balance * signal.suggested_size
        if position_value > self.balance:
            logger.warning("Insufficient paper balance")
            return None

        commission = position_value * self.commission_pct
        total_cost = position_value + commission
        if total_cost > self.balance:
            total_cost = self.balance
            position_value = total_cost / (1 + self.commission_pct)
            commission = total_cost - position_value

        shares = position_value / price if price > 0 else 0

        pos = PaperPosition(
            id=str(uuid.uuid4())[:8],
            market_id=signal.market_id,
            market_question=market_question,
            side=side,
            entry_price=price,
            shares=shares,
            cost=position_value,
            current_price=price,
            opened_at=datetime.now().isoformat(),
            strategy=signal.strategy_name,
        )

        self.positions[pos.id] = pos
        self.balance -= total_cost

        logger.info(
            f"Paper BUY {side.upper()} | {shares:.2f} shares @ {price:.3f} "
            f"| Cost: ${position_value:.2f} + ${commission:.2f} fee"
        )
        return pos

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[PaperTrade]:
        """Закрыть paper-позицию."""
        pos = self.positions.get(position_id)
        if not pos:
            return None

        proceeds = pos.shares * exit_price
        commission = proceeds * self.commission_pct
        net_proceeds = proceeds - commission
        pnl = net_proceeds - pos.cost
        pnl_pct = (pnl / pos.cost * 100) if pos.cost > 0 else 0

        trade = PaperTrade(
            id=pos.id,
            market_id=pos.market_id,
            market_question=pos.market_question,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=pos.strategy,
            opened_at=pos.opened_at,
            closed_at=datetime.now().isoformat(),
            reason=reason,
        )

        self.trade_history.append(trade)
        self._total_pnl += pnl
        self.balance += net_proceeds
        del self.positions[position_id]

        # Обновляем peak
        if self.total_equity > self._peak_balance:
            self._peak_balance = self.total_equity

        logger.info(f"Paper SELL {pos.side.upper()} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        return trade

    def update_prices(self, market_prices: Dict[str, float]):
        """Обновить текущие цены открытых позиций."""
        for pos in self.positions.values():
            if pos.market_id in market_prices:
                pos.current_price = market_prices[pos.market_id]

    def portfolio_summary(self) -> str:
        """Сводка портфеля для Telegram."""
        equity = self.total_equity
        pnl = self.total_pnl
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        lines = [
            f"💼 *Paper Portfolio*",
            f"",
            f"💰 Баланс: `${self.balance:.2f}`",
            f"📊 Equity: `${equity:.2f}`",
            f"{pnl_emoji} P&L: `${pnl:+.2f}` ({self.total_return_pct:+.1f}%)",
            f"📉 Max DD: `{self.max_drawdown:.1f}%`",
            f"🎯 Win Rate: `{self.win_rate:.0%}`",
            f"📋 Сделок: {len(self.trade_history)}",
        ]

        if self.positions:
            lines.append(f"\n*Открытые позиции ({len(self.positions)}):*")
            for pos in self.positions.values():
                emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                q = pos.market_question[:40] + "..." if len(pos.market_question) > 40 else pos.market_question
                lines.append(
                    f"  {emoji} {pos.side.upper()} `{q}`\n"
                    f"    Entry: {pos.entry_price:.3f} → Now: {pos.current_price:.3f} "
                    f"({pos.unrealized_pnl_pct:+.1f}%)"
                )

        return "\n".join(lines)

    def trade_history_summary(self, last_n: int = 10) -> str:
        """История последних сделок."""
        if not self.trade_history:
            return "📋 История сделок пуста."

        recent = self.trade_history[-last_n:]
        lines = [f"📋 *Последние {len(recent)} сделок:*\n"]

        for t in reversed(recent):
            emoji = "✅" if t.pnl > 0 else "❌"
            q = t.market_question[:35] + "..." if len(t.market_question) > 35 else t.market_question
            lines.append(
                f"{emoji} {t.side.upper()} | `${t.pnl:+.2f}` ({t.pnl_pct:+.1f}%)\n"
                f"   _{q}_\n"
                f"   {t.entry_price:.3f} → {t.exit_price:.3f} | {t.strategy}"
            )

        return "\n".join(lines)
