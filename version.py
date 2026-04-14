"""
Версия и метаданные сборки.
Обновляется при каждом релизе.
"""
from datetime import datetime

VERSION = "2.0.0"
BUILD_DATE = datetime.now().strftime("%Y-%m-%d %H:%M")
CODENAME = "Deep Signal"

def get_version_string() -> str:
    return f"v{VERSION} ({CODENAME})"

def get_build_info() -> str:
    return (
        f"🤖 PolyBot v{VERSION} «{CODENAME}»\n"
        f"📅 Сборка: {BUILD_DATE}\n"
        f"🐍 Python | aiogram 3.x\n"
        f"📦 Модули: data pipeline, strategies, backtester,\n"
        f"   paper trader, risk mgmt, AI analyzer"
    )
