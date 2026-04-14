"""
Конфигурация бота. Загружает переменные из .env файла.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Главная конфигурация бота."""

    # Telegram
    telegram_token: str = ""

    # Polymarket
    polymarket_api_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polymarket_api_key: Optional[str] = None
    polymarket_secret: Optional[str] = None
    polymarket_passphrase: Optional[str] = None

    # Макро-данные (FRED)
    fred_api_key: Optional[str] = None

    # AI / LLM
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"  # openai | anthropic | deepseek
    llm_model: str = "gpt-4o-mini"

    # Binance (для ценовых фидов)
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"

    # Risk Management
    max_position_size: float = 100.0       # макс. размер позиции в $
    max_daily_loss: float = 50.0           # макс. дневной убыток
    max_open_positions: int = 10           # макс. открытых позиций
    kill_switch_loss_pct: float = 0.15     # kill switch при -15% от баланса
    default_position_pct: float = 0.05     # 5% баланса на позицию

    # Paper Trading
    paper_trading: bool = True             # по умолчанию paper mode
    paper_balance: float = 10000.0         # стартовый paper баланс

    # Data
    data_fallback_enabled: bool = True
    csv_log_dir: str = "./logs"

    # Допустимые пользователи (Telegram IDs)
    allowed_users: list = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Config":
        """Загрузка конфигурации из переменных окружения."""
        cfg = cls()
        cfg.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        cfg.polymarket_api_key = os.getenv("POLYMARKET_API_KEY")
        cfg.polymarket_secret = os.getenv("POLYMARKET_SECRET")
        cfg.polymarket_passphrase = os.getenv("POLYMARKET_PASSPHRASE")
        cfg.fred_api_key = os.getenv("FRED_API_KEY")
        cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
        cfg.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        cfg.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        cfg.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        cfg.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        cfg.paper_balance = float(os.getenv("PAPER_BALANCE", "10000"))
        cfg.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "100"))
        cfg.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "50"))
        cfg.kill_switch_loss_pct = float(os.getenv("KILL_SWITCH_LOSS_PCT", "0.15"))

        users = os.getenv("ALLOWED_USERS", "")
        if users:
            cfg.allowed_users = [int(x.strip()) for x in users.split(",") if x.strip()]

        return cfg
