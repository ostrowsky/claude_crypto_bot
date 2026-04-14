"""
Макроэкономические данные — интеграция с FRED API.

Вдохновлено: mortada/fredapi
Предоставляет CPI, ставки ФРС, GDP и другие макро-серии
для контекста торговых решений на prediction markets.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MacroIndicator:
    series_id: str
    name: str
    value: float
    date: str
    units: str = ""
    change_pct: float = 0.0  # изменение vs предыдущий период


# Ключевые макро-серии FRED, релевантные для prediction markets
MACRO_SERIES = {
    "CPIAUCSL":    {"name": "CPI (Inflation)",        "units": "Index"},
    "FEDFUNDS":    {"name": "Fed Funds Rate",          "units": "%"},
    "UNRATE":      {"name": "Unemployment Rate",       "units": "%"},
    "GDP":         {"name": "GDP",                     "units": "Billions $"},
    "T10Y2Y":      {"name": "10Y-2Y Yield Spread",    "units": "%"},
    "VIXCLS":      {"name": "VIX Volatility Index",   "units": "Index"},
    "DGS10":       {"name": "10-Year Treasury Yield",  "units": "%"},
    "DCOILWTICO":  {"name": "WTI Crude Oil Price",    "units": "$/barrel"},
    "DEXUSEU":     {"name": "USD/EUR Exchange Rate",  "units": "USD per EUR"},
    "UMCSENT":     {"name": "Consumer Sentiment",      "units": "Index"},
}


class MacroDataService:
    """
    Сервис макроэкономических данных через FRED API.
    Паттерн из fredapi: Fred(api_key) → get_series('SP500').
    """

    FRED_BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Dict[str, MacroIndicator] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def get_series_latest(self, series_id: str) -> Optional[MacroIndicator]:
        """Получить последнее значение макро-серии."""
        if not self.api_key:
            logger.warning("FRED API key not set — macro data unavailable")
            return None

        # Проверяем кеш (обновляем раз в час)
        cached = self._cache.get(series_id)
        if cached:
            from time import time
            age_min = (time() - float(cached.date.split("|")[-1] if "|" in cached.date else "0")) / 60
            if age_min < 60:
                return cached

        try:
            session = await self._get_session()
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,  # текущее + предыдущее для расчёта изменения
            }
            async with session.get(
                f"{self.FRED_BASE}/series/observations", params=params
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()

            observations = data.get("observations", [])
            if not observations:
                return None

            current = observations[0]
            value = float(current["value"]) if current["value"] != "." else 0.0

            change_pct = 0.0
            if len(observations) > 1:
                prev_val = float(observations[1]["value"]) if observations[1]["value"] != "." else 0
                if prev_val != 0:
                    change_pct = ((value - prev_val) / prev_val) * 100

            meta = MACRO_SERIES.get(series_id, {"name": series_id, "units": ""})
            indicator = MacroIndicator(
                series_id=series_id,
                name=meta["name"],
                value=value,
                date=current["date"],
                units=meta["units"],
                change_pct=round(change_pct, 2),
            )
            self._cache[series_id] = indicator
            return indicator

        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return None

    async def get_macro_dashboard(self) -> List[MacroIndicator]:
        """Получить дашборд ключевых макро-индикаторов."""
        results = []
        for series_id in MACRO_SERIES:
            indicator = await self.get_series_latest(series_id)
            if indicator:
                results.append(indicator)
        return results

    async def get_fed_context(self) -> str:
        """
        Краткий текстовый контекст макро-среды для AI-анализатора.
        Используется как input для LLM при принятии решений.
        """
        indicators = await self.get_macro_dashboard()
        if not indicators:
            return "Макро-данные недоступны (FRED API key не задан)."

        lines = ["📊 Макро-контекст:"]
        for ind in indicators:
            arrow = "📈" if ind.change_pct > 0 else "📉" if ind.change_pct < 0 else "➡️"
            lines.append(
                f"  {arrow} {ind.name}: {ind.value:.2f} {ind.units} "
                f"({ind.change_pct:+.1f}%) [{ind.date}]"
            )
        return "\n".join(lines)

    def format_indicator(self, ind: MacroIndicator) -> str:
        arrow = "🟢" if ind.change_pct > 0 else "🔴" if ind.change_pct < 0 else "⚪"
        return (
            f"{arrow} *{ind.name}*\n"
            f"   Значение: `{ind.value:.2f}` {ind.units}\n"
            f"   Изменение: `{ind.change_pct:+.2f}%`\n"
            f"   Дата: {ind.date}"
        )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
