"""
Strategy Engine — подключаемый фреймворк стратегий.

Вдохновлено:
- ent0n29/polybot: pluggable strategy framework
- dylanpersonguy/Polymarket-Trading-Bot: 7 стратегий
- Vibe-Trading: skills система с 68 навыками
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.data_pipeline import MarketSnapshot, OrderBook

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    HOLD = "hold"


class Confidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Signal:
    """Торговый сигнал от стратегии."""
    strategy_name: str
    market_id: str
    signal_type: SignalType
    confidence: Confidence
    suggested_price: float
    suggested_size: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def is_actionable(self) -> bool:
        return self.signal_type != SignalType.HOLD

    @property
    def confidence_emoji(self) -> str:
        return {
            Confidence.LOW: "🟡",
            Confidence.MEDIUM: "🟠",
            Confidence.HIGH: "🟢",
            Confidence.VERY_HIGH: "💚",
        }[self.confidence]


@dataclass
class StrategyConfig:
    """Конфигурация стратегии."""
    enabled: bool = True
    max_position_pct: float = 0.05     # макс. % баланса на позицию
    min_confidence: Confidence = Confidence.MEDIUM
    cooldown_seconds: int = 300         # минимум между сигналами
    params: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Базовый класс стратегии.
    Все стратегии наследуют от него и реализуют evaluate().
    """

    name: str = "base"
    description: str = ""
    version: str = "1.0"

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self._last_signal_time: Dict[str, float] = {}

    @abstractmethod
    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        """Оценить рынок и вернуть сигнал (или None)."""
        ...

    def _check_cooldown(self, market_id: str) -> bool:
        """Проверить cooldown для рынка."""
        last = self._last_signal_time.get(market_id, 0)
        now = datetime.now().timestamp()
        if now - last < self.config.cooldown_seconds:
            return False
        return True

    def _record_signal(self, market_id: str):
        self._last_signal_time[market_id] = datetime.now().timestamp()

    def _make_signal(
        self,
        market_id: str,
        signal_type: SignalType,
        confidence: Confidence,
        price: float,
        size: float,
        reason: str,
        **meta,
    ) -> Signal:
        self._record_signal(market_id)
        return Signal(
            strategy_name=self.name,
            market_id=market_id,
            signal_type=signal_type,
            confidence=confidence,
            suggested_price=price,
            suggested_size=size,
            reason=reason,
            metadata=meta,
        )

    def info(self) -> str:
        status = "✅" if self.config.enabled else "❌"
        return f"{status} *{self.name}* v{self.version}\n   {self.description}"


class StrategyEngine:
    """
    Движок стратегий — запускает все зарегистрированные стратегии
    и собирает сигналы.
    """

    def __init__(self):
        self._strategies: List[BaseStrategy] = []

    def register(self, strategy: BaseStrategy):
        self._strategies.append(strategy)
        logger.info(f"Strategy registered: {strategy.name} v{strategy.version}")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        for s in self._strategies:
            if s.name == name:
                return s
        return None

    def list_strategies(self) -> List[BaseStrategy]:
        return list(self._strategies)

    def enabled_strategies(self) -> List[BaseStrategy]:
        return [s for s in self._strategies if s.config.enabled]

    async def scan_market(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Signal]:
        """Прогнать все стратегии по одному рынку."""
        signals = []
        for strategy in self.enabled_strategies():
            try:
                if not strategy._check_cooldown(market.market_id):
                    continue
                signal = await strategy.evaluate(market, orderbook, context)
                if signal and signal.is_actionable:
                    if signal.confidence.value >= strategy.config.min_confidence.value:
                        signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} error: {e}")
        return signals

    async def scan_markets(
        self,
        markets: List[MarketSnapshot],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Signal]:
        """Параллельное сканирование множества рынков (паттерн из Polymarket-Trading-Bot)."""
        all_signals = []
        for market in markets:
            signals = await self.scan_market(market, context=context)
            all_signals.extend(signals)
        # Сортировка по confidence
        all_signals.sort(
            key=lambda s: list(Confidence).index(s.confidence),
            reverse=True,
        )
        return all_signals

    def format_signal(self, signal: Signal) -> str:
        """Форматировать сигнал для Telegram."""
        direction = {
            SignalType.BUY_YES: "🟢 BUY YES",
            SignalType.BUY_NO: "🔴 BUY NO",
            SignalType.SELL_YES: "📤 SELL YES",
            SignalType.SELL_NO: "📤 SELL NO",
            SignalType.HOLD: "⏸ HOLD",
        }[signal.signal_type]

        return (
            f"{signal.confidence_emoji} *{signal.strategy_name}*\n"
            f"  {direction}\n"
            f"  Цена: `{signal.suggested_price:.3f}`\n"
            f"  Размер: `${signal.suggested_size:.2f}`\n"
            f"  Причина: _{signal.reason}_"
        )
