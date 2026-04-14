"""
Торговые стратегии для prediction markets.

Реализованы 5 стратегий, вдохновлённых:
- dylanpersonguy/Polymarket-Trading-Bot (арбитраж, momentum, market making, AI forecast)
- txbabaxyz/polyrec (fade impulse)
"""
import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional

from core.data_pipeline import MarketSnapshot, OrderBook
from core.strategy_engine import (
    BaseStrategy,
    Confidence,
    Signal,
    SignalType,
    StrategyConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Momentum Strategy
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """
    Стратегия следования за трендом вероятности.
    Если цена YES растёт N периодов подряд → покупаем.
    Если падает → продаём / покупаем NO.
    """

    name = "momentum"
    description = "Следование за трендом вероятности контракта"
    version = "2.0"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self._price_history: Dict[str, deque] = {}
        self._window = self.config.params.get("window", 5)
        self._threshold = self.config.params.get("threshold", 0.03)  # 3% движение

    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        mid = market.market_id
        if mid not in self._price_history:
            self._price_history[mid] = deque(maxlen=self._window + 1)

        self._price_history[mid].append(market.yes_price)
        history = self._price_history[mid]

        if len(history) < self._window:
            return None

        prices = list(history)
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(1, len(prices)) if prices[i - 1] > 0]

        if not returns:
            return None

        avg_return = sum(returns) / len(returns)
        consecutive_up = all(r > 0 for r in returns[-3:]) if len(returns) >= 3 else False
        consecutive_down = all(r < 0 for r in returns[-3:]) if len(returns) >= 3 else False

        # Восходящий моментум
        if consecutive_up and avg_return > self._threshold:
            confidence = Confidence.HIGH if avg_return > self._threshold * 2 else Confidence.MEDIUM
            return self._make_signal(
                mid, SignalType.BUY_YES, confidence,
                price=market.yes_price,
                size=self.config.max_position_pct,
                reason=f"Восх. моментум: {avg_return:+.2%} за {self._window} периодов",
                avg_return=avg_return,
            )

        # Нисходящий моментум
        if consecutive_down and avg_return < -self._threshold:
            confidence = Confidence.HIGH if avg_return < -self._threshold * 2 else Confidence.MEDIUM
            return self._make_signal(
                mid, SignalType.BUY_NO, confidence,
                price=market.no_price,
                size=self.config.max_position_pct,
                reason=f"Нисх. моментум: {avg_return:+.2%} за {self._window} периодов",
                avg_return=avg_return,
            )

        return None


# ---------------------------------------------------------------------------
# 2. Arbitrage Strategy
# ---------------------------------------------------------------------------

class ArbitrageStrategy(BaseStrategy):
    """
    Арбитражная стратегия: ищет расхождения где YES + NO != 1.0
    или расхождения между связанными рынками.
    """

    name = "arbitrage"
    description = "Поиск арбитражных возможностей (YES + NO ≠ 1.0)"
    version = "2.0"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self._min_edge = self.config.params.get("min_edge", 0.02)  # мин. 2% edge

    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        total = market.yes_price + market.no_price
        edge = abs(total - 1.0)

        if edge < self._min_edge:
            return None

        # Если сумма > 1.0 → overpriced, можно продать обе стороны
        # Если сумма < 1.0 → underpriced, можно купить обе стороны
        if total > 1.0 + self._min_edge:
            # Продаём переоценённую сторону
            if market.yes_price > market.no_price:
                return self._make_signal(
                    market.market_id, SignalType.BUY_NO,
                    Confidence.HIGH if edge > 0.05 else Confidence.MEDIUM,
                    price=market.no_price,
                    size=self.config.max_position_pct,
                    reason=f"Арбитраж: YES+NO={total:.3f} (edge {edge:.1%})",
                    edge=edge, total=total,
                )
            else:
                return self._make_signal(
                    market.market_id, SignalType.BUY_YES,
                    Confidence.HIGH if edge > 0.05 else Confidence.MEDIUM,
                    price=market.yes_price,
                    size=self.config.max_position_pct,
                    reason=f"Арбитраж: YES+NO={total:.3f} (edge {edge:.1%})",
                    edge=edge, total=total,
                )

        elif total < 1.0 - self._min_edge:
            # Покупаем недооценённую сторону
            cheaper = SignalType.BUY_YES if market.yes_price < market.no_price else SignalType.BUY_NO
            price = min(market.yes_price, market.no_price)
            return self._make_signal(
                market.market_id, cheaper,
                Confidence.HIGH if edge > 0.05 else Confidence.MEDIUM,
                price=price,
                size=self.config.max_position_pct,
                reason=f"Арбитраж: YES+NO={total:.3f}, underprice edge {edge:.1%}",
                edge=edge, total=total,
            )

        return None


# ---------------------------------------------------------------------------
# 3. Market Making Strategy
# ---------------------------------------------------------------------------

class MarketMakingStrategy(BaseStrategy):
    """
    Market Making: зарабатываем на спреде, предоставляя ликвидность.
    Размещаем ордера по обе стороны ордербука.
    """

    name = "market_making"
    description = "Предоставление ликвидности, заработок на спреде"
    version = "2.0"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self._min_spread = self.config.params.get("min_spread", 0.04)  # мин. спред 4%
        self._min_liquidity = self.config.params.get("min_liquidity", 1000)

    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        if not orderbook or orderbook.spread is None:
            return None

        spread = orderbook.spread

        if spread < self._min_spread:
            return None

        if market.liquidity < self._min_liquidity:
            return None

        # Определяем справедливую цену
        fair = (orderbook.best_bid + orderbook.best_ask) / 2 if orderbook.best_bid and orderbook.best_ask else market.yes_price

        # Ставим бид ниже fair price
        bid_price = fair - spread * 0.3
        confidence = Confidence.HIGH if spread > 0.08 else Confidence.MEDIUM

        return self._make_signal(
            market.market_id, SignalType.BUY_YES, confidence,
            price=bid_price,
            size=self.config.max_position_pct * 0.5,
            reason=f"MM: спред {spread:.1%}, bid@{bid_price:.3f}",
            spread=spread, fair_price=fair,
            bid_depth=orderbook.bid_depth,
            ask_depth=orderbook.ask_depth,
        )


# ---------------------------------------------------------------------------
# 4. Fade Impulse Strategy (из polyrec)
# ---------------------------------------------------------------------------

class FadeImpulseStrategy(BaseStrategy):
    """
    Fade Impulse: торгуем против резких импульсных движений.
    Когда цена резко двигается, ставим на возврат к среднему.
    Паттерн из txbabaxyz/polyrec fade_impulse_backtest.py
    """

    name = "fade_impulse"
    description = "Противодействие импульсным движениям (mean reversion)"
    version = "2.0"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self._price_history: Dict[str, deque] = {}
        self._lookback = self.config.params.get("lookback", 10)
        self._impulse_threshold = self.config.params.get("impulse_threshold", 0.08)  # 8%
        self._mean_window = self.config.params.get("mean_window", 20)

    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        mid = market.market_id
        if mid not in self._price_history:
            self._price_history[mid] = deque(maxlen=self._mean_window + 5)

        self._price_history[mid].append(market.yes_price)
        history = list(self._price_history[mid])

        if len(history) < self._lookback:
            return None

        # Рассчитываем скользящее среднее
        sma = sum(history[-self._mean_window:]) / min(len(history), self._mean_window)

        # Рассчитываем импульс (текущая цена vs N периодов назад)
        prev_price = history[-self._lookback]
        if prev_price == 0:
            return None

        impulse = (market.yes_price - prev_price) / prev_price

        # Если импульс превышает порог — торгуем против
        if abs(impulse) < self._impulse_threshold:
            return None

        deviation = (market.yes_price - sma) / sma if sma > 0 else 0

        if impulse > self._impulse_threshold and deviation > 0.05:
            # Резкий рост → fade: покупаем NO
            conf = Confidence.HIGH if abs(impulse) > self._impulse_threshold * 2 else Confidence.MEDIUM
            return self._make_signal(
                mid, SignalType.BUY_NO, conf,
                price=market.no_price,
                size=self.config.max_position_pct,
                reason=f"Fade UP impulse: {impulse:+.1%}, deviation от SMA: {deviation:+.1%}",
                impulse=impulse, sma=sma, deviation=deviation,
            )

        elif impulse < -self._impulse_threshold and deviation < -0.05:
            # Резкое падение → fade: покупаем YES
            conf = Confidence.HIGH if abs(impulse) > self._impulse_threshold * 2 else Confidence.MEDIUM
            return self._make_signal(
                mid, SignalType.BUY_YES, conf,
                price=market.yes_price,
                size=self.config.max_position_pct,
                reason=f"Fade DOWN impulse: {impulse:+.1%}, deviation от SMA: {deviation:+.1%}",
                impulse=impulse, sma=sma, deviation=deviation,
            )

        return None


# ---------------------------------------------------------------------------
# 5. AI Forecast Strategy
# ---------------------------------------------------------------------------

class AIForecastStrategy(BaseStrategy):
    """
    AI Forecast: использует LLM для оценки вероятности события.
    Сравнивает LLM-оценку с рыночной ценой для поиска edge.
    """

    name = "ai_forecast"
    description = "LLM-прогнозирование вероятностей событий"
    version = "2.0"

    def __init__(self, config: Optional[StrategyConfig] = None, ai_analyzer=None):
        super().__init__(config)
        self.ai_analyzer = ai_analyzer
        self._min_edge = self.config.params.get("min_edge", 0.05)  # 5% расхождение

    async def evaluate(
        self,
        market: MarketSnapshot,
        orderbook: Optional[OrderBook] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Signal]:
        if not self.ai_analyzer:
            return None

        try:
            ai_prob = await self.ai_analyzer.estimate_probability(
                question=market.question,
                current_price=market.yes_price,
                context=context,
            )

            if ai_prob is None:
                return None

            edge = ai_prob - market.yes_price

            if abs(edge) < self._min_edge:
                return None

            if edge > 0:
                # AI считает вероятность выше рынка → BUY YES
                conf = Confidence.VERY_HIGH if edge > 0.15 else (
                    Confidence.HIGH if edge > 0.10 else Confidence.MEDIUM
                )
                return self._make_signal(
                    market.market_id, SignalType.BUY_YES, conf,
                    price=market.yes_price,
                    size=self.config.max_position_pct,
                    reason=f"AI: P={ai_prob:.0%} vs Market={market.yes_price:.0%} (edge {edge:+.1%})",
                    ai_probability=ai_prob, market_price=market.yes_price, edge=edge,
                )
            else:
                # AI считает вероятность ниже рынка → BUY NO
                conf = Confidence.VERY_HIGH if abs(edge) > 0.15 else (
                    Confidence.HIGH if abs(edge) > 0.10 else Confidence.MEDIUM
                )
                return self._make_signal(
                    market.market_id, SignalType.BUY_NO, conf,
                    price=market.no_price,
                    size=self.config.max_position_pct,
                    reason=f"AI: P={ai_prob:.0%} vs Market={market.yes_price:.0%} (edge {edge:+.1%})",
                    ai_probability=ai_prob, market_price=market.yes_price, edge=edge,
                )

        except Exception as e:
            logger.error(f"AI forecast error: {e}")
            return None


# ---------------------------------------------------------------------------
# Utility: register all strategies
# ---------------------------------------------------------------------------

def register_all_strategies(engine, ai_analyzer=None):
    """Зарегистрировать все стратегии в движке."""
    engine.register(MomentumStrategy(StrategyConfig(
        params={"window": 5, "threshold": 0.03}
    )))
    engine.register(ArbitrageStrategy(StrategyConfig(
        params={"min_edge": 0.02}
    )))
    engine.register(MarketMakingStrategy(StrategyConfig(
        params={"min_spread": 0.04, "min_liquidity": 1000}
    )))
    engine.register(FadeImpulseStrategy(StrategyConfig(
        params={"lookback": 10, "impulse_threshold": 0.08}
    )))
    engine.register(AIForecastStrategy(
        StrategyConfig(params={"min_edge": 0.05}),
        ai_analyzer=ai_analyzer,
    ))
