"""
Telegram Bot Handlers — все команды и меню бота.

Версия и дата билда отображаются в меню.
"""
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from version import VERSION, BUILD_DATE, CODENAME, get_build_info
from config import Config
from core.data_pipeline import DataLoaderRegistry, PolymarketRESTLoader
from core.macro_data import MacroDataService
from core.strategy_engine import StrategyEngine
from core.paper_trader import PaperTrader
from core.risk_manager import RiskManager, RiskLimits
from core.reverse_engineer import ReverseEngineer
from core.ai_analyzer import AIAnalyzer
from core.backtester import BacktestEngine
from core.whale_tracker import WhaleTracker
from utils.csv_logger import CSVLogger
from strategies.all_strategies import register_all_strategies

logger = logging.getLogger(__name__)
router = Router()


# ---------------------------------------------------------------------------
# App Context — хранит все компоненты бота
# ---------------------------------------------------------------------------

class AppContext:
    """Singleton контекст приложения."""

    def __init__(self, config: Config):
        self.config = config

        # Data Layer
        self.data_registry = DataLoaderRegistry()
        self.data_registry.register(PolymarketRESTLoader(config.polymarket_api_url))

        # Macro Data
        self.macro = MacroDataService(api_key=config.fred_api_key)

        # AI Analyzer
        self.ai = AIAnalyzer(
            provider=config.llm_provider,
            api_key=config.openai_api_key or config.anthropic_api_key,
            model=config.llm_model,
        )

        # Strategy Engine
        self.strategy_engine = StrategyEngine()
        register_all_strategies(self.strategy_engine, ai_analyzer=self.ai)

        # Paper Trader
        self.paper_trader = PaperTrader(
            initial_balance=config.paper_balance,
            commission_pct=0.02,
        )

        # Risk Manager
        self.risk_manager = RiskManager(RiskLimits(
            max_position_size=config.max_position_size,
            max_daily_loss=config.max_daily_loss,
            max_open_positions=config.max_open_positions,
            kill_switch_loss_pct=config.kill_switch_loss_pct,
        ))

        # Backtester
        self.backtester = BacktestEngine()

        # Reverse Engineer
        self.reverse_eng = ReverseEngineer()

        # Whale Tracker
        self.whale_tracker = WhaleTracker(min_position_size=5000)

        # CSV Logger
        self.csv_logger = CSVLogger(log_dir=config.csv_log_dir)


# Global app context
app: Optional[AppContext] = None


def init_app(config: Config):
    global app
    app = AppContext(config)


# ---------------------------------------------------------------------------
# Меню / Клавиатура
# ---------------------------------------------------------------------------

def main_menu_keyboard() -> InlineKeyboardMarkup:
    """Главное меню бота с версией."""
    buttons = [
        [
            InlineKeyboardButton(text="📊 Рынки", callback_data="markets"),
            InlineKeyboardButton(text="🔍 Сканер", callback_data="scanner"),
        ],
        [
            InlineKeyboardButton(text="💼 Портфель", callback_data="portfolio"),
            InlineKeyboardButton(text="📋 История", callback_data="history"),
        ],
        [
            InlineKeyboardButton(text="🧠 AI Анализ", callback_data="ai_menu"),
            InlineKeyboardButton(text="📈 Макро", callback_data="macro"),
        ],
        [
            InlineKeyboardButton(text="⚙️ Стратегии", callback_data="strategies"),
            InlineKeyboardButton(text="🛡 Риски", callback_data="risk_status"),
        ],
        [
            InlineKeyboardButton(text="🕵️ Трейдер-анализ", callback_data="reverse_eng"),
            InlineKeyboardButton(text="📊 Бэктест", callback_data="backtest_menu"),
        ],
        [
            InlineKeyboardButton(text="🐋 Киты", callback_data="whales"),
            InlineKeyboardButton(text="📂 Логи", callback_data="logs"),
        ],
        [
            InlineKeyboardButton(
                text=f"ℹ️ v{VERSION} | {BUILD_DATE}",
                callback_data="about",
            ),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def back_button() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад в меню", callback_data="main_menu")]
    ])


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message):
    mode = "📝 Paper Trading" if app.config.paper_trading else "💰 LIVE Trading"
    await message.answer(
        f"🤖 *PolyBot v{VERSION}* «{CODENAME}»\n"
        f"📅 Сборка: {BUILD_DATE}\n\n"
        f"Режим: {mode}\n"
        f"Баланс: `${app.paper_trader.balance:,.2f}`\n\n"
        f"Выберите раздел:",
        reply_markup=main_menu_keyboard(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# /menu
# ---------------------------------------------------------------------------

@router.message(Command("menu"))
async def cmd_menu(message: Message):
    await message.answer(
        f"🤖 *PolyBot* v{VERSION}\n📅 {BUILD_DATE}\n\nГлавное меню:",
        reply_markup=main_menu_keyboard(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# /version
# ---------------------------------------------------------------------------

@router.message(Command("version"))
async def cmd_version(message: Message):
    await message.answer(
        get_build_info(),
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Main Menu
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "main_menu")
async def cb_main_menu(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🤖 *PolyBot* v{VERSION} | {BUILD_DATE}\n\nГлавное меню:",
        reply_markup=main_menu_keyboard(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: About / Version
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "about")
async def cb_about(callback: CallbackQuery):
    strategies = app.strategy_engine.list_strategies()
    enabled = len(app.strategy_engine.enabled_strategies())

    text = (
        f"{get_build_info()}\n\n"
        f"*Компоненты:*\n"
        f"  📡 Data Pipeline — multi-source + auto-fallback\n"
        f"  📊 Macro Data — FRED API (CPI, ставки ФРС)\n"
        f"  ⚙️ Стратегий: {len(strategies)} (активных: {enabled})\n"
        f"  🧪 Бэктестер — prediction market engine\n"
        f"  📝 Paper Trader — симулятор торговли\n"
        f"  🛡 Risk Manager — kill switch + лимиты\n"
        f"  🕵️ Reverse Engineering — анализ трейдеров\n"
        f"  🧠 AI Analyzer — LLM ({app.config.llm_provider})\n\n"
        f"*Источники вдохновения:*\n"
        f"  polybot, polyrec, Vibe-Trading, fredapi,\n"
        f"  Polymarket-Trading-Bot, lightweight-charts,\n"
        f"  prediction-market-backtesting, RTK, Goose"
    )
    await callback.message.edit_text(
        text,
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Markets
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "markets")
async def cb_markets(callback: CallbackQuery):
    await callback.answer("Загрузка рынков...")

    markets = await app.data_registry.fetch_markets(limit=10)

    if not markets:
        await callback.message.edit_text(
            "❌ Не удалось загрузить рынки. Проверьте подключение.",
            reply_markup=back_button(),
        )
        return

    lines = ["📊 *Топ рынки Polymarket:*\n"]
    for i, m in enumerate(markets[:10], 1):
        q = m.question[:50] + "..." if len(m.question) > 50 else m.question
        vol = f"${m.volume_24h:,.0f}" if m.volume_24h else "N/A"
        lines.append(
            f"*{i}.* {q}\n"
            f"   YES: `{m.yes_price:.0%}` | Vol: {vol}\n"
        )

    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="markets"),
                InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
            ]
        ]),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Scanner
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "scanner")
async def cb_scanner(callback: CallbackQuery):
    await callback.answer("Сканирование рынков...")

    markets = await app.data_registry.fetch_markets(limit=20)
    if not markets:
        await callback.message.edit_text(
            "❌ Нет данных для сканирования.",
            reply_markup=back_button(),
        )
        return

    # Получаем макро-контекст для стратегий
    macro_ctx = await app.macro.get_fed_context()
    context = {"macro_context": macro_ctx}

    signals = await app.strategy_engine.scan_markets(markets, context=context)

    if not signals:
        await callback.message.edit_text(
            "🔍 *Сканер*\n\nСигналов не найдено. Все стратегии проверены.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔄 Повторить", callback_data="scanner"),
                    InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
                ]
            ]),
            parse_mode="Markdown",
        )
        return

    lines = [f"🔍 *Сканер: {len(signals)} сигналов*\n"]
    for sig in signals[:5]:
        lines.append(app.strategy_engine.format_signal(sig))
        lines.append("")

    # Кнопки для исполнения
    buttons = []
    for i, sig in enumerate(signals[:3]):
        buttons.append([InlineKeyboardButton(
            text=f"✅ Исполнить #{i+1} ({sig.strategy_name})",
            callback_data=f"exec_signal:{i}",
        )])
    buttons.append([
        InlineKeyboardButton(text="🔄 Повторить", callback_data="scanner"),
        InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
    ])

    # Сохраняем сигналы для исполнения
    if not hasattr(app, '_pending_signals'):
        app._pending_signals = {}
    app._pending_signals[callback.from_user.id] = signals

    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Execute Signal
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("exec_signal:"))
async def cb_exec_signal(callback: CallbackQuery):
    idx = int(callback.data.split(":")[1])
    signals = getattr(app, '_pending_signals', {}).get(callback.from_user.id, [])

    if idx >= len(signals):
        await callback.answer("Сигнал устарел. Запустите сканер заново.")
        return

    signal = signals[idx]

    # Проверка рисков
    risk_check = app.risk_manager.check_signal(signal, app.paper_trader)
    if not risk_check:
        await callback.answer(f"Риск: {risk_check.reason}", show_alert=True)
        return

    # Исполняем
    position = app.paper_trader.execute_signal(signal, market_question=signal.metadata.get("question", ""))
    if position:
        app.risk_manager.record_trade(0)  # будет обновлён при закрытии
        await callback.message.edit_text(
            f"✅ *Позиция открыта!*\n\n"
            f"ID: `{position.id}`\n"
            f"Сторона: {position.side.upper()}\n"
            f"Цена: `{position.entry_price:.3f}`\n"
            f"Акций: `{position.shares:.2f}`\n"
            f"Стоимость: `${position.cost:.2f}`\n"
            f"Стратегия: {signal.strategy_name}",
            reply_markup=back_button(),
            parse_mode="Markdown",
        )
    else:
        await callback.answer("Не удалось открыть позицию", show_alert=True)


# ---------------------------------------------------------------------------
# Callback: Portfolio
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "portfolio")
async def cb_portfolio(callback: CallbackQuery):
    text = app.paper_trader.portfolio_summary()

    buttons = [[
        InlineKeyboardButton(text="🔄 Обновить", callback_data="portfolio"),
        InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
    ]]

    # Добавляем кнопки закрытия позиций
    for pos_id, pos in app.paper_trader.positions.items():
        q = pos.market_question[:25] + "..." if len(pos.market_question) > 25 else pos.market_question
        buttons.insert(0, [InlineKeyboardButton(
            text=f"❌ Закрыть: {q}",
            callback_data=f"close_pos:{pos_id}",
        )])

    await callback.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Close Position
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("close_pos:"))
async def cb_close_position(callback: CallbackQuery):
    pos_id = callback.data.split(":")[1]
    pos = app.paper_trader.positions.get(pos_id)
    if not pos:
        await callback.answer("Позиция не найдена")
        return

    trade = app.paper_trader.close_position(pos_id, pos.current_price, reason="manual")
    if trade:
        app.risk_manager.record_trade(trade.pnl)
        emoji = "✅" if trade.pnl > 0 else "❌"
        await callback.answer(f"{emoji} P&L: ${trade.pnl:+.2f}", show_alert=True)

    # Обновляем портфель
    await cb_portfolio(callback)


# ---------------------------------------------------------------------------
# Callback: History
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "history")
async def cb_history(callback: CallbackQuery):
    text = app.paper_trader.trade_history_summary(last_n=10)
    await callback.message.edit_text(
        text,
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Strategies
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "strategies")
async def cb_strategies(callback: CallbackQuery):
    strategies = app.strategy_engine.list_strategies()
    lines = [f"⚙️ *Стратегии* ({len(strategies)}):\n"]

    buttons = []
    for s in strategies:
        lines.append(s.info())
        lines.append("")
        status = "off" if s.config.enabled else "on"
        emoji = "❌" if s.config.enabled else "✅"
        buttons.append([InlineKeyboardButton(
            text=f"{emoji} {s.name}",
            callback_data=f"toggle_strategy:{s.name}:{status}",
        )])

    buttons.append([InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu")])

    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="Markdown",
    )


@router.callback_query(F.data.startswith("toggle_strategy:"))
async def cb_toggle_strategy(callback: CallbackQuery):
    _, name, action = callback.data.split(":")
    strategy = app.strategy_engine.get_strategy(name)
    if strategy:
        strategy.config.enabled = (action == "on")
        status = "включена" if strategy.config.enabled else "отключена"
        await callback.answer(f"{name}: {status}")
    await cb_strategies(callback)


# ---------------------------------------------------------------------------
# Callback: Risk Status
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "risk_status")
async def cb_risk_status(callback: CallbackQuery):
    text = app.risk_manager.status()
    buttons = []

    if not app.risk_manager.is_trading_allowed:
        buttons.append([InlineKeyboardButton(
            text="🔓 Сбросить Kill Switch",
            callback_data="risk_reset",
        )])

    buttons.append([InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu")])

    await callback.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="Markdown",
    )


@router.callback_query(F.data == "risk_reset")
async def cb_risk_reset(callback: CallbackQuery):
    app.risk_manager.reset_kill_switch()
    await callback.answer("✅ Kill switch сброшен")
    await cb_risk_status(callback)


# ---------------------------------------------------------------------------
# Callback: Macro Dashboard
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "macro")
async def cb_macro(callback: CallbackQuery):
    await callback.answer("Загрузка макро-данных...")

    if not app.config.fred_api_key:
        await callback.message.edit_text(
            "📈 *Макро-данные*\n\n"
            "⚠️ FRED API key не задан.\n"
            "Установите `FRED_API_KEY` в `.env` файле.\n"
            "Получить бесплатно: https://fred.stlouisfed.org/docs/api/api\\_key.html",
            reply_markup=back_button(),
            parse_mode="Markdown",
        )
        return

    indicators = await app.macro.get_macro_dashboard()
    if not indicators:
        await callback.message.edit_text(
            "❌ Не удалось загрузить макро-данные.",
            reply_markup=back_button(),
        )
        return

    lines = ["📈 *Макро-дашборд:*\n"]
    for ind in indicators:
        lines.append(app.macro.format_indicator(ind))
        lines.append("")

    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="macro"),
                InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
            ]
        ]),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: AI Menu
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "ai_menu")
async def cb_ai_menu(callback: CallbackQuery):
    has_key = bool(app.config.openai_api_key or app.config.anthropic_api_key)
    status = f"✅ {app.config.llm_provider} / {app.config.llm_model}" if has_key else "⚠️ API key не задан"

    await callback.message.edit_text(
        f"🧠 *AI Анализ*\n\n"
        f"Провайдер: {status}\n\n"
        f"Отправьте мне вопрос с рынка Polymarket,\n"
        f"и я оценю вероятность через AI.\n\n"
        f"Пример:\n"
        f"`/ai Will Bitcoin reach $100k by end of 2025?`",
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


@router.message(Command("ai"))
async def cmd_ai_analyze(message: Message):
    question = message.text.replace("/ai", "").strip()
    if not question:
        await message.answer("Укажите вопрос: `/ai <вопрос>`", parse_mode="Markdown")
        return

    await message.answer("🧠 Анализирую...")

    macro_ctx = await app.macro.get_fed_context()
    analysis = await app.ai.analyze_market(
        question=question,
        yes_price=0.5,
        volume=0,
        context={"macro_context": macro_ctx},
    )

    if analysis:
        await message.answer(
            f"🧠 *AI Анализ:*\n\n{analysis}",
            reply_markup=back_button(),
            parse_mode="Markdown",
        )
    else:
        await message.answer(
            "❌ Не удалось получить анализ. Проверьте API ключ.",
            reply_markup=back_button(),
        )


# ---------------------------------------------------------------------------
# Callback: Reverse Engineer
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "reverse_eng")
async def cb_reverse_eng(callback: CallbackQuery):
    await callback.message.edit_text(
        "🕵️ *Анализ трейдера*\n\n"
        "Отправьте Polymarket адрес кошелька трейдера:\n"
        "`/trader 0x1234...abcd`\n\n"
        "Бот проанализирует стратегию: паттерны входа,\n"
        "предпочтения по сторонам, тайминг, категории.",
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


@router.message(Command("trader"))
async def cmd_trader(message: Message):
    address = message.text.replace("/trader", "").strip()
    if not address or len(address) < 10:
        await message.answer("Укажите адрес: `/trader 0x...`", parse_mode="Markdown")
        return

    await message.answer("🕵️ Анализирую трейдера...")

    profile = await app.reverse_eng.analyze_trader(address)
    if profile:
        text = app.reverse_eng.format_profile(profile)
        await message.answer(text, reply_markup=back_button(), parse_mode="Markdown")
    else:
        await message.answer(
            "❌ Не удалось получить данные трейдера.",
            reply_markup=back_button(),
        )


# ---------------------------------------------------------------------------
# Callback: Backtest Menu
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "backtest_menu")
async def cb_backtest_menu(callback: CallbackQuery):
    strategies = [s.name for s in app.strategy_engine.list_strategies()]
    strat_list = ", ".join(strategies)

    await callback.message.edit_text(
        f"📊 *Бэктестинг*\n\n"
        f"Запустите бэктест стратегии:\n"
        f"`/backtest <strategy_name>`\n\n"
        f"Доступные стратегии:\n"
        f"`{strat_list}`\n\n"
        f"Пример: `/backtest momentum`",
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


@router.message(Command("backtest"))
async def cmd_backtest(message: Message):
    strategy_name = message.text.replace("/backtest", "").strip()
    if not strategy_name:
        await message.answer("Укажите стратегию: `/backtest <name>`", parse_mode="Markdown")
        return

    strategy = app.strategy_engine.get_strategy(strategy_name)
    if not strategy:
        names = [s.name for s in app.strategy_engine.list_strategies()]
        await message.answer(
            f"❌ Стратегия `{strategy_name}` не найдена.\n"
            f"Доступные: {', '.join(names)}",
            parse_mode="Markdown",
        )
        return

    await message.answer(f"📊 Запуск бэктеста `{strategy_name}`...")

    # Генерируем тестовые данные (в реальности — из API / CSV)
    import random
    from core.backtester import PriceBar

    price = 0.5
    bars = []
    for i in range(100):
        price += random.gauss(0, 0.02)
        price = max(0.05, min(0.95, price))
        bars.append(PriceBar(
            timestamp=f"2025-01-{(i//4)+1:02d} {(i%4)*6:02d}:00",
            yes_price=price,
            no_price=1.0 - price,
            volume=random.uniform(1000, 50000),
        ))

    result = await app.backtester.run(strategy, bars)
    await message.answer(
        result.summary(),
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Callback: Whales
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "whales")
async def cb_whales(callback: CallbackQuery):
    text = app.whale_tracker.format_whale_list()

    buttons = []
    if app.whale_tracker.list_whales():
        buttons.append([InlineKeyboardButton(
            text="🔄 Проверить активность", callback_data="whale_check",
        )])

    buttons.append([InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu")])

    await callback.message.edit_text(
        text + "\n\nДобавить: `/whale_add <адрес> <метка>`\nУдалить: `/whale_rm <адрес>`",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="Markdown",
    )


@router.callback_query(F.data == "whale_check")
async def cb_whale_check(callback: CallbackQuery):
    await callback.answer("Проверяю активность китов...")

    activities = await app.whale_tracker.check_activity()
    new_acts = [a for a in activities if a.is_new_position]

    if not activities:
        await callback.message.edit_text(
            "🐋 Нет активности отслеживаемых китов.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔄 Повторить", callback_data="whale_check"),
                    InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
                ]
            ]),
        )
        return

    lines = [f"🐋 *Активность китов* ({len(activities)} позиций, {len(new_acts)} новых)\n"]
    for act in activities[:8]:
        lines.append(app.whale_tracker.format_activity(act))
        lines.append("")

    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="whale_check"),
                InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
            ]
        ]),
        parse_mode="Markdown",
    )


@router.message(Command("whale_add"))
async def cmd_whale_add(message: Message):
    parts = message.text.replace("/whale_add", "").strip().split(maxsplit=1)
    if not parts:
        await message.answer("Формат: `/whale_add <адрес> [метка]`", parse_mode="Markdown")
        return

    address = parts[0]
    label = parts[1] if len(parts) > 1 else ""
    app.whale_tracker.add_whale(address, label)

    addr_short = f"{address[:6]}...{address[-4:]}"
    await message.answer(
        f"✅ Кит добавлен: *{label or addr_short}*\n`{addr_short}`",
        reply_markup=back_button(),
        parse_mode="Markdown",
    )


@router.message(Command("whale_rm"))
async def cmd_whale_rm(message: Message):
    address = message.text.replace("/whale_rm", "").strip()
    if not address:
        await message.answer("Формат: `/whale_rm <адрес>`", parse_mode="Markdown")
        return
    app.whale_tracker.remove_whale(address)
    await message.answer("✅ Кит удалён.", reply_markup=back_button())


# ---------------------------------------------------------------------------
# Callback: Logs
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "logs")
async def cb_logs(callback: CallbackQuery):
    text = app.csv_logger.get_log_stats()
    await callback.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="logs"),
                InlineKeyboardButton(text="◀️ Меню", callback_data="main_menu"),
            ]
        ]),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Setup bot commands (видны в Telegram меню)
# ---------------------------------------------------------------------------

async def setup_bot_commands(bot: Bot):
    """Установить команды бота, видимые в Telegram."""
    commands = [
        BotCommand(command="start", description=f"Запуск бота v{VERSION}"),
        BotCommand(command="menu", description="Главное меню"),
        BotCommand(command="version", description=f"Версия {VERSION} ({BUILD_DATE})"),
        BotCommand(command="ai", description="AI анализ рынка"),
        BotCommand(command="trader", description="Анализ трейдера"),
        BotCommand(command="backtest", description="Бэктест стратегии"),
        BotCommand(command="whale_add", description="Добавить кита для отслеживания"),
        BotCommand(command="whale_rm", description="Удалить кита"),
    ]
    await bot.set_my_commands(commands)
