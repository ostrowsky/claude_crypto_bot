"""
Бэктестер для prediction markets.

Вдохновлено:
- evan-kolberg/prediction-market-backtesting: движок для бинарных контрактов
- txbabaxyz/polyrec: fade impulse backtest
- Vibe-Trading: 7 бэктест-движков
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.strategy_engine import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Одна сделка в бэктесте."""
    entry_time: str
    exit_time: Optional[str]
    market_id: str
    side: str           # "yes" | "no"
    entry_price: float
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    resolved: bool = False  # контракт завершился?
    outcome: Optional[bool] = None  # True = YES победил


@dataclass
class BacktestResult:
    """Результат бэктеста."""
    strategy_name: str
    period: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        if self.initial_balance == 0:
            return 0
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    def summary(self) -> str:
        return (
            f"📊 *Бэктест: {self.strategy_name}*\n"
            f"Период: {self.period}\n\n"
            f"💰 Баланс: `${self.initial_balance:.0f}` → `${self.final_balance:.0f}` "
            f"({self.return_pct:+.1f}%)\n"
            f"📈 P&L: `${self.total_pnl:+.2f}`\n"
            f"📉 Max Drawdown: `{self.max_drawdown:.1f}%`\n\n"
            f"🎯 Сделок: {self.total_trades}\n"
            f"✅ Win rate: `{self.win_rate:.1%}`\n"
            f"📐 Profit Factor: `{self.profit_factor:.2f}`\n"
            f"📊 Sharpe: `{self.sharpe_ratio:.2f}`\n"
            f"💵 Avg P&L: `${self.avg_trade_pnl:+.2f}`\n"
            f"🏆 Best: `${self.best_trade:+.2f}` | "
            f"💀 Worst: `${self.worst_trade:+.2f}`"
        )


@dataclass
class PriceBar:
    """Одна точка исторических данных."""
    timestamp: str
    yes_price: float
    no_price: float
    volume: float = 0


class BacktestEngine:
    """
    Движок бэктестинга для prediction markets.
    Учитывает специфику бинарных контрактов:
    - фиксированный payout ($1 при выигрыше)
    - expiration / резолюция
    - комиссии
    """

    def __init__(
        self,
        initial_balance: float = 10000,
        commission_pct: float = 0.02,  # 2% комиссия
        slippage_pct: float = 0.005,   # 0.5% проскальзывание
    ):
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    async def run(
        self,
        strategy: BaseStrategy,
        price_data: List[PriceBar],
        market_id: str = "backtest",
        market_question: str = "Backtest market",
        outcome: Optional[bool] = None,
    ) -> BacktestResult:
        """Запустить бэктест стратегии на исторических данных."""

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: List[BacktestTrade] = []
        open_position: Optional[BacktestTrade] = None
        daily_pnls: List[float] = []

        from core.data_pipeline import MarketSnapshot

        for i, bar in enumerate(price_data):
            # Создаём снимок рынка из исторических данных
            snapshot = MarketSnapshot(
                market_id=market_id,
                question=market_question,
                yes_price=bar.yes_price,
                no_price=bar.no_price,
                spread=abs(bar.yes_price - bar.no_price),
                volume_24h=bar.volume,
                liquidity=bar.volume * 10,
            )

            signal = await strategy.evaluate(snapshot)

            if signal and signal.is_actionable and open_position is None:
                # Открываем позицию
                side = "yes" if signal.signal_type in (SignalType.BUY_YES,) else "no"
                entry_price = bar.yes_price if side == "yes" else bar.no_price

                # Применяем проскальзывание
                entry_price *= (1 + self.slippage_pct)
                size = min(balance * strategy.config.max_position_pct, balance * 0.2)
                shares = size / entry_price if entry_price > 0 else 0

                open_position = BacktestTrade(
                    entry_time=bar.timestamp,
                    exit_time=None,
                    market_id=market_id,
                    side=side,
                    entry_price=entry_price,
                    size=shares,
                )
                balance -= size * (1 + self.commission_pct)

            elif signal and open_position:
                # Если есть противоположный сигнал — закрываем
                current_side = open_position.side
                signal_is_opposite = (
                    (current_side == "yes" and signal.signal_type == SignalType.BUY_NO) or
                    (current_side == "no" and signal.signal_type == SignalType.BUY_YES)
                )

                if signal_is_opposite:
                    exit_price = bar.yes_price if current_side == "yes" else bar.no_price
                    exit_price *= (1 - self.slippage_pct)
                    proceeds = open_position.size * exit_price * (1 - self.commission_pct)
                    cost = open_position.size * open_position.entry_price
                    pnl = proceeds - cost

                    open_position.exit_time = bar.timestamp
                    open_position.exit_price = exit_price
                    open_position.pnl = pnl
                    trades.append(open_position)
                    daily_pnls.append(pnl)

                    balance += proceeds
                    open_position = None

            # Обновляем drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        # Если позиция осталась открытой — закрываем по последней цене
        if open_position and price_data:
            last = price_data[-1]
            exit_price = last.yes_price if open_position.side == "yes" else last.no_price

            # Если известен outcome — рассчитываем по нему
            if outcome is not None:
                if (open_position.side == "yes" and outcome) or \
                   (open_position.side == "no" and not outcome):
                    exit_price = 1.0  # выигрыш
                else:
                    exit_price = 0.0  # проигрыш

            proceeds = open_position.size * exit_price * (1 - self.commission_pct)
            cost = open_position.size * open_position.entry_price
            open_position.exit_price = exit_price
            open_position.pnl = proceeds - cost
            open_position.exit_time = last.timestamp
            open_position.resolved = outcome is not None
            open_position.outcome = outcome
            trades.append(open_position)
            daily_pnls.append(open_position.pnl)
            balance += proceeds

        # Рассчитываем метрики
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(winning) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_pnl = total_pnl / len(trades) if trades else 0
        best = max((t.pnl for t in trades), default=0)
        worst = min((t.pnl for t in trades), default=0)

        # Sharpe ratio (упрощённый)
        if len(daily_pnls) > 1:
            mean_r = sum(daily_pnls) / len(daily_pnls)
            var_r = sum((r - mean_r) ** 2 for r in daily_pnls) / (len(daily_pnls) - 1)
            std_r = var_r ** 0.5
            sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0
        else:
            sharpe = 0

        period = ""
        if price_data:
            period = f"{price_data[0].timestamp} → {price_data[-1].timestamp}"

        return BacktestResult(
            strategy_name=strategy.name,
            period=period,
            initial_balance=self.initial_balance,
            final_balance=balance,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            best_trade=best,
            worst_trade=worst,
            trades=trades,
        )
