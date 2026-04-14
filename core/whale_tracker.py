"""
Whale Tracker — отслеживание крупных игроков на Polymarket.

Вдохновлено:
- dylanpersonguy/Polymarket-Trading-Bot: whale tracker + copy-trade simulator
- ent0n29/polybot: user trade analysis
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class WhaleActivity:
    """Активность кита."""
    address: str
    market_id: str
    market_question: str
    side: str
    size: float
    price: float
    timestamp: str
    is_new_position: bool = False


@dataclass
class TrackedWhale:
    """Отслеживаемый кит."""
    address: str
    label: str              # пользовательская метка
    total_volume: float = 0.0
    win_rate: float = 0.0
    recent_activity: List[WhaleActivity] = field(default_factory=list)
    added_at: str = ""


class WhaleTracker:
    """
    Отслеживание крупных позиций и активности «китов».
    Позволяет добавлять адреса для мониторинга и получать
    оповещения о их сделках.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"

    def __init__(self, min_position_size: float = 5000.0):
        self.min_position_size = min_position_size
        self._tracked: Dict[str, TrackedWhale] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._known_positions: Dict[str, Set[str]] = {}  # address -> set of market_ids

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)
            )
        return self._session

    def add_whale(self, address: str, label: str = ""):
        """Добавить адрес для отслеживания."""
        if address not in self._tracked:
            self._tracked[address] = TrackedWhale(
                address=address,
                label=label or f"Whale_{len(self._tracked)+1}",
                added_at=datetime.now().isoformat(),
            )
            self._known_positions[address] = set()
            logger.info(f"Tracking whale: {label} ({address[:10]}...)")

    def remove_whale(self, address: str):
        """Убрать адрес из отслеживания."""
        self._tracked.pop(address, None)
        self._known_positions.pop(address, None)

    def list_whales(self) -> List[TrackedWhale]:
        return list(self._tracked.values())

    async def check_activity(self) -> List[WhaleActivity]:
        """Проверить новую активность всех отслеживаемых китов."""
        all_activity = []

        for address, whale in self._tracked.items():
            try:
                new_acts = await self._fetch_positions(address)
                if new_acts:
                    whale.recent_activity = new_acts[-10:]
                    all_activity.extend(new_acts)
            except Exception as e:
                logger.error(f"Whale check error for {address[:10]}: {e}")

        return all_activity

    async def _fetch_positions(self, address: str) -> List[WhaleActivity]:
        """Получить позиции адреса и определить новые."""
        session = await self._get_session()
        activities = []

        try:
            async with session.get(
                f"{self.GAMMA_API}/positions",
                params={
                    "user": address,
                    "limit": 20,
                    "sortBy": "value",
                    "sortOrder": "desc",
                },
            ) as r:
                if r.status != 200:
                    return []
                positions = await r.json()

            known = self._known_positions.get(address, set())

            for pos in positions:
                market_id = pos.get("conditionId", "")
                size = float(pos.get("currentValue", pos.get("size", 0)))

                if size < self.min_position_size:
                    continue

                is_new = market_id not in known
                known.add(market_id)

                activities.append(WhaleActivity(
                    address=address,
                    market_id=market_id,
                    market_question=pos.get("title", pos.get("question", "unknown")),
                    side=pos.get("outcome", "YES").lower(),
                    size=size,
                    price=float(pos.get("avgPrice", 0.5)),
                    timestamp=pos.get("createdAt", datetime.now().isoformat()),
                    is_new_position=is_new,
                ))

            self._known_positions[address] = known

        except Exception as e:
            logger.error(f"Fetch positions error: {e}")

        return activities

    async def find_top_whales(self, market_id: str, limit: int = 5) -> List[Dict]:
        """Найти крупнейших игроков на конкретном рынке."""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.GAMMA_API}/positions",
                params={
                    "market": market_id,
                    "limit": limit,
                    "sortBy": "value",
                    "sortOrder": "desc",
                },
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()

            whales = []
            for pos in data:
                size = float(pos.get("currentValue", pos.get("size", 0)))
                if size >= self.min_position_size:
                    whales.append({
                        "address": pos.get("proxyWallet", pos.get("user", "")),
                        "side": pos.get("outcome", "YES"),
                        "size": size,
                        "avg_price": float(pos.get("avgPrice", 0)),
                    })

            return whales

        except Exception as e:
            logger.error(f"Find whales error: {e}")
            return []

    def format_activity(self, activity: WhaleActivity) -> str:
        """Форматировать активность для Telegram."""
        whale = self._tracked.get(activity.address)
        label = whale.label if whale else f"{activity.address[:8]}..."
        new_tag = "🆕 " if activity.is_new_position else ""

        q = activity.market_question[:45]
        if len(activity.market_question) > 45:
            q += "..."

        return (
            f"🐋 *{label}*\n"
            f"{new_tag}{activity.side.upper()} | `${activity.size:,.0f}` @ {activity.price:.3f}\n"
            f"_{q}_"
        )

    def format_whale_list(self) -> str:
        """Список отслеживаемых китов."""
        if not self._tracked:
            return "🐋 Нет отслеживаемых китов.\n\nДобавьте: `/whale_add <адрес> <метка>`"

        lines = [f"🐋 *Отслеживаемые киты ({len(self._tracked)}):*\n"]
        for whale in self._tracked.values():
            addr = f"{whale.address[:6]}...{whale.address[-4:]}"
            recent = len(whale.recent_activity)
            lines.append(f"  • *{whale.label}* `{addr}` — {recent} позиций")

        return "\n".join(lines)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
