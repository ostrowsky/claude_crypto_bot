"""
AI Analyzer — LLM-интеграция для анализа рынков.

Вдохновлено:
- Vibe-Trading: multi-LLM providers, 68 skills
- Goose: расширяемый AI-агент
- Polymarket-Trading-Bot: AI forecast стратегия
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """
    LLM-анализатор для prediction markets.
    Поддерживает OpenAI, Anthropic, DeepSeek.
    """

    PROVIDERS = {
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "header": "Authorization",
            "prefix": "Bearer ",
        },
        "anthropic": {
            "url": "https://api.anthropic.com/v1/messages",
            "header": "x-api-key",
            "prefix": "",
        },
        "deepseek": {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "header": "Authorization",
            "prefix": "Bearer ",
        },
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Вызов LLM API."""
        if not self.api_key:
            return None

        provider_config = self.PROVIDERS.get(self.provider)
        if not provider_config:
            return None

        session = await self._get_session()

        if self.provider == "anthropic":
            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            }
            headers = {
                provider_config["header"]: f"{provider_config['prefix']}{self.api_key}",
                "content-type": "application/json",
            }

        try:
            async with session.post(
                provider_config["url"],
                json=payload,
                headers=headers,
            ) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.error(f"LLM API error {r.status}: {text[:200]}")
                    return None

                data = await r.json()

                if self.provider == "anthropic":
                    return data["content"][0]["text"]
                else:
                    return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return None

    async def estimate_probability(
        self,
        question: str,
        current_price: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        Оценить вероятность события через LLM.
        Возвращает число от 0 до 1.
        """
        system = (
            "You are an expert prediction market analyst. "
            "Your task is to estimate the probability of an event occurring. "
            "Respond ONLY with a JSON object: {\"probability\": 0.XX, \"reasoning\": \"...\"}"
        )

        context_str = ""
        if context:
            macro = context.get("macro_context", "")
            if macro:
                context_str = f"\n\nMacroeconomic context:\n{macro}"

        user = (
            f"Event: {question}\n"
            f"Current market price (implied probability): {current_price:.2%}\n"
            f"{context_str}\n\n"
            f"Estimate the TRUE probability of this event. "
            f"Consider all available information and biases. "
            f"Respond with JSON only."
        )

        response = await self._call_llm(system, user)
        if not response:
            return None

        try:
            # Парсим JSON из ответа
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                prob = float(result.get("probability", 0.5))
                return max(0.01, min(0.99, prob))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse AI probability: {e}")

        return None

    async def analyze_market(
        self,
        question: str,
        yes_price: float,
        volume: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Полный анализ рынка через LLM."""
        system = (
            "You are an expert prediction market analyst. "
            "Provide a concise but insightful analysis in Russian. "
            "Focus on: key factors, probability assessment, risk/reward, and recommendation."
        )

        context_str = ""
        if context:
            macro = context.get("macro_context", "")
            if macro:
                context_str = f"\n\nМакро-контекст:\n{macro}"

        user = (
            f"Рынок: {question}\n"
            f"Цена YES: {yes_price:.2%}\n"
            f"Объём 24ч: ${volume:,.0f}\n"
            f"{context_str}\n\n"
            f"Дай краткий анализ (3-5 предложений): ключевые факторы, "
            f"твоя оценка вероятности vs рынок, рекомендация."
        )

        return await self._call_llm(system, user)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
