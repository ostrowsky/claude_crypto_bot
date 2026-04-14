"""
Data Pipeline — мульти-источниковая агрегация данных с auto-fallback.

Вдохновлено:
- polyrec: WebSocket агрегация Chainlink + Binance + Polymarket
- Vibe-Trading: DataLoader Protocol + Registry + auto-fallback chains
"""
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """Снимок рынка prediction market."""
    market_id: str
    question: str
    yes_price: float
    no_price: float
    spread: float
    volume_24h: float
    liquidity: float
    end_date: Optional[str] = None
    category: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def mid_price(self) -> float:
        return (self.yes_price + self.no_price) / 2

    @property
    def implied_probability(self) -> float:
        return self.yes_price


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    market_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def bid_depth(self) -> float:
        return sum(l.size for l in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(l.size for l in self.asks)


@dataclass
class PriceTick:
    symbol: str
    price: float
    volume: float = 0.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DataLoader Protocol — вдохновлено Vibe-Trading Registry
# ---------------------------------------------------------------------------

class DataLoader(ABC):
    """Абстрактный загрузчик данных с приоритетом и fallback."""

    name: str = "base"
    priority: int = 0  # чем ниже — тем выше приоритет

    @abstractmethod
    async def fetch_markets(self, limit: int = 50, **filters) -> List[MarketSnapshot]:
        ...

    @abstractmethod
    async def fetch_orderbook(self, market_id: str) -> Optional[OrderBook]:
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        ...


class DataLoaderRegistry:
    """
    Реестр загрузчиков данных с auto-fallback.
    Паттерн из Vibe-Trading: если основной источник недоступен,
    автоматически переключаемся на следующий по приоритету.
    """

    def __init__(self):
        self._loaders: List[DataLoader] = []
        self._active_loader: Optional[DataLoader] = None
        self._fallback_count: int = 0

    def register(self, loader: DataLoader):
        self._loaders.append(loader)
        self._loaders.sort(key=lambda x: x.priority)
        logger.info(f"Registered data loader: {loader.name} (priority={loader.priority})")

    async def get_active_loader(self) -> Optional[DataLoader]:
        """Получить активный загрузчик или fallback."""
        if self._active_loader:
            if await self._active_loader.is_available():
                return self._active_loader

        for loader in self._loaders:
            try:
                if await loader.is_available():
                    if self._active_loader and self._active_loader.name != loader.name:
                        self._fallback_count += 1
                        logger.warning(
                            f"Fallback #{self._fallback_count}: "
                            f"{self._active_loader.name} → {loader.name}"
                        )
                    self._active_loader = loader
                    return loader
            except Exception as e:
                logger.error(f"Loader {loader.name} check failed: {e}")

        logger.error("No data loaders available!")
        return None

    async def fetch_markets(self, limit: int = 50, **filters) -> List[MarketSnapshot]:
        loader = await self.get_active_loader()
        if not loader:
            return []
        try:
            return await loader.fetch_markets(limit, **filters)
        except Exception as e:
            logger.error(f"{loader.name} fetch_markets failed: {e}")
            self._active_loader = None
            return await self.fetch_markets(limit, **filters)

    async def fetch_orderbook(self, market_id: str) -> Optional[OrderBook]:
        loader = await self.get_active_loader()
        if not loader:
            return None
        try:
            return await loader.fetch_orderbook(market_id)
        except Exception as e:
            logger.error(f"{loader.name} fetch_orderbook failed: {e}")
            self._active_loader = None
            return await self.fetch_orderbook(market_id)


# ---------------------------------------------------------------------------
# Polymarket REST DataLoader
# ---------------------------------------------------------------------------

class PolymarketRESTLoader(DataLoader):
    """Загрузчик данных через Polymarket CLOB REST API."""

    name = "polymarket_rest"
    priority = 0

    def __init__(self, api_url: str = "https://clob.polymarket.com"):
        self.api_url = api_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def is_available(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.api_url}/time") as r:
                return r.status == 200
        except Exception:
            return False

    async def fetch_markets(self, limit: int = 50, **filters) -> List[MarketSnapshot]:
        session = await self._get_session()
        params = {"limit": limit, "active": "true", "closed": "false"}
        if "category" in filters:
            params["tag"] = filters["category"]

        try:
            # Gamma API для списка рынков
            async with session.get(
                "https://gamma-api.polymarket.com/markets",
                params=params,
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()

            markets = []
            for item in data:
                try:
                    tokens = item.get("clobTokenIds", "")
                    if isinstance(tokens, str):
                        tokens = json.loads(tokens) if tokens.startswith("[") else []

                    yes_price = float(item.get("outcomePrices", "[0.5,0.5]").strip("[]").split(",")[0])
                    no_price = 1.0 - yes_price

                    markets.append(MarketSnapshot(
                        market_id=item.get("conditionId", item.get("id", "")),
                        question=item.get("question", ""),
                        yes_price=yes_price,
                        no_price=no_price,
                        spread=abs(yes_price - no_price),
                        volume_24h=float(item.get("volume24hr", 0)),
                        liquidity=float(item.get("liquidityClob", 0)),
                        end_date=item.get("endDate"),
                        category=item.get("groupItemTitle", ""),
                    ))
                except (ValueError, KeyError, IndexError):
                    continue

            return markets
        except Exception as e:
            logger.error(f"Polymarket fetch_markets error: {e}")
            return []

    async def fetch_orderbook(self, market_id: str) -> Optional[OrderBook]:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.api_url}/book",
                params={"token_id": market_id},
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()

            return OrderBook(
                market_id=market_id,
                bids=[OrderBookLevel(float(b["price"]), float(b["size"]))
                      for b in data.get("bids", [])],
                asks=[OrderBookLevel(float(a["price"]), float(a["size"]))
                      for a in data.get("asks", [])],
            )
        except Exception as e:
            logger.error(f"Orderbook fetch error: {e}")
            return None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Price Feed (Binance WebSocket — паттерн из polyrec)
# ---------------------------------------------------------------------------

class PriceFeed:
    """
    Ценовой фид через внешние источники (Binance и др.).
    Паттерн из polyrec: агрегация нескольких WebSocket стримов.
    """

    def __init__(self):
        self._prices: Dict[str, PriceTick] = {}
        self._callbacks: List[Callable] = []
        self._running = False

    def get_price(self, symbol: str) -> Optional[PriceTick]:
        return self._prices.get(symbol.upper())

    def on_price(self, callback: Callable):
        self._callbacks.append(callback)

    async def start_binance_feed(self, symbols: List[str]):
        """Подписка на Binance 1s klines (паттерн из polyrec dash.py)."""
        streams = "/".join(f"{s.lower()}@kline_1s" for s in symbols)
        url = f"wss://stream.binance.com:9443/ws/{streams}"

        self._running = True
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info(f"Binance WS connected: {symbols}")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                k = data.get("k", {})
                                tick = PriceTick(
                                    symbol=k.get("s", ""),
                                    price=float(k.get("c", 0)),
                                    volume=float(k.get("v", 0)),
                                    source="binance",
                                )
                                self._prices[tick.symbol] = tick
                                for cb in self._callbacks:
                                    try:
                                        await cb(tick)
                                    except Exception:
                                        pass
            except Exception as e:
                logger.error(f"Binance WS error: {e}")
                if self._running:
                    await asyncio.sleep(5)

    def stop(self):
        self._running = False
