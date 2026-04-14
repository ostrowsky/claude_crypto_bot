"""
Reverse Engineering — анализ стратегий других трейдеров.

Вдохновлено:
- ent0n29/polybot: user trade analysis, pattern recognition, replication scoring
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TraderTrade:
    """Сделка трейдера (из Polymarket API)."""
    market_id: str
    market_question: str
    side: str
    price: float
    size: float
    timestamp: str
    outcome: Optional[str] = None


@dataclass
class TraderProfile:
    """Профиль трейдера."""
    address: str
    total_trades: int = 0
    total_volume: float = 0.0
    win_rate: float = 0.0
    avg_position_size: float = 0.0
    preferred_side: str = ""
    top_categories: List[str] = field(default_factory=list)
    avg_entry_price: float = 0.0
    timing_pattern: str = ""
    trades: List[TraderTrade] = field(default_factory=list)


class ReverseEngineer:
    """
    Анализирует историю сделок трейдера на Polymarket
    и выявляет паттерны для копирования.
    """

    POLY_API = "https://gamma-api.polymarket.com"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def analyze_trader(self, address: str, limit: int = 100) -> Optional[TraderProfile]:
        """Проанализировать историю сделок адреса."""
        try:
            session = await self._get_session()

            # Получаем историю позиций
            async with session.get(
                f"{self.POLY_API}/positions",
                params={"user": address, "limit": limit, "sortBy": "value", "sortOrder": "desc"},
            ) as r:
                if r.status != 200:
                    return None
                positions = await r.json()

            if not positions:
                return None

            trades = []
            sides = []
            prices = []
            sizes = []
            categories = []

            for pos in positions:
                side = pos.get("outcome", "YES")
                price = float(pos.get("avgPrice", 0.5))
                size = float(pos.get("size", 0))

                trades.append(TraderTrade(
                    market_id=pos.get("conditionId", ""),
                    market_question=pos.get("title", pos.get("question", "")),
                    side=side.lower(),
                    price=price,
                    size=size,
                    timestamp=pos.get("createdAt", ""),
                ))
                sides.append(side.lower())
                prices.append(price)
                sizes.append(size)

                cat = pos.get("groupItemTitle", pos.get("category", ""))
                if cat:
                    categories.append(cat)

            # Анализ паттернов
            side_counts = Counter(sides)
            preferred = side_counts.most_common(1)[0][0] if side_counts else "yes"

            cat_counts = Counter(categories)
            top_cats = [c for c, _ in cat_counts.most_common(5)]

            # Паттерн входа по цене
            avg_price = sum(prices) / len(prices) if prices else 0.5
            low_entries = sum(1 for p in prices if p < 0.3)
            high_entries = sum(1 for p in prices if p > 0.7)
            mid_entries = len(prices) - low_entries - high_entries

            if low_entries > len(prices) * 0.5:
                timing = "Contrarian — входит при низких вероятностях (<30¢)"
            elif high_entries > len(prices) * 0.5:
                timing = "Conviction — входит при высоких вероятностях (>70¢)"
            else:
                timing = "Balanced — торгует по всему спектру цен"

            return TraderProfile(
                address=address,
                total_trades=len(trades),
                total_volume=sum(sizes),
                avg_position_size=sum(sizes) / len(sizes) if sizes else 0,
                preferred_side=preferred,
                top_categories=top_cats,
                avg_entry_price=avg_price,
                timing_pattern=timing,
                trades=trades,
            )

        except Exception as e:
            logger.error(f"Reverse engineer error: {e}")
            return None

    def format_profile(self, profile: TraderProfile) -> str:
        """Форматировать профиль для Telegram."""
        addr_short = f"{profile.address[:6]}...{profile.address[-4:]}"

        cats_str = ", ".join(profile.top_categories[:3]) if profile.top_categories else "N/A"

        return (
            f"🕵️ *Анализ трейдера* `{addr_short}`\n\n"
            f"📊 Сделок: {profile.total_trades}\n"
            f"💰 Объём: `${profile.total_volume:,.0f}`\n"
            f"📏 Ср. позиция: `${profile.avg_position_size:,.0f}`\n"
            f"🎯 Ср. вход: `{profile.avg_entry_price:.2f}`\n"
            f"↕️ Предпочтение: {profile.preferred_side.upper()}\n\n"
            f"🧠 *Паттерн:* {profile.timing_pattern}\n"
            f"🏷 *Категории:* {cats_str}"
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
