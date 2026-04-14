"""
PolyBot v2.0 «Deep Signal» — Telegram бот для Polymarket.

Entry point. Запуск:
    python main.py

Или с Docker:
    docker build -t polybot .
    docker run --env-file .env polybot
"""
import asyncio
import logging
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import Config
from version import VERSION, BUILD_DATE, CODENAME
from bot.handlers import router, setup_bot_commands, init_app


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("polybot")


def print_banner():
    banner = f"""
╔══════════════════════════════════════════╗
║          🤖 PolyBot v{VERSION}              ║
║          «{CODENAME}»                     ║
║          Build: {BUILD_DATE}             ║
╠══════════════════════════════════════════╣
║  Модули:                                 ║
║   📡 Data Pipeline (multi-source)        ║
║   📊 Macro Data (FRED API)               ║
║   ⚙️  5 Strategies (momentum, arb, ...)  ║
║   🧪 Backtester (pred. markets)          ║
║   📝 Paper Trader                        ║
║   🛡  Risk Manager (kill switch)         ║
║   🕵️  Reverse Engineering               ║
║   🧠 AI Analyzer (multi-LLM)            ║
╚══════════════════════════════════════════╝
"""
    print(banner)


async def main():
    print_banner()

    # Загрузка .env
    env_file = Path(".env")
    if env_file.exists():
        logger.info("Loading .env file...")
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                import os
                os.environ.setdefault(key.strip(), value.strip())

    config = Config.from_env()

    if not config.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not set! Add it to .env file.")
        sys.exit(1)

    # Инициализация контекста
    init_app(config)

    mode = "PAPER" if config.paper_trading else "LIVE"
    logger.info(f"Mode: {mode} | Balance: ${config.paper_balance:,.0f}")
    logger.info(f"LLM: {config.llm_provider} / {config.llm_model}")
    logger.info(f"FRED API: {'configured' if config.fred_api_key else 'not set'}")

    # Создание бота
    bot = Bot(
        token=config.telegram_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )

    dp = Dispatcher()
    dp.include_router(router)

    # Установка команд меню
    await setup_bot_commands(bot)

    logger.info(f"PolyBot v{VERSION} starting...")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
