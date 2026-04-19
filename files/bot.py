from __future__ import annotations

"""
Crypto Trend Bot — Telegram interface.

Меню:
  🔍 Анализ всего списка   → форвард-тест + активные сигналы по всем 100 монетам
  ▶️ Старт мониторинга     → запускает фоновый мониторинг
  ⏹ Стоп мониторинга      → останавливает мониторинг
  📊 Активные сигналы      → открытые позиции + прогнозы
  📋 Список монет          → просмотр / добавление / удаление
  ⚙️ Настройки             → текущие параметры стратегии
"""

import asyncio
import time
from pathlib import Path
import logging
from types import SimpleNamespace

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from monitor import MonitorState, monitoring_loop, load_positions, save_positions
from strategy import market_scan, check_entry_conditions, check_setup_conditions, analyze_coin, fetch_klines, get_entry_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

state = MonitorState()
# Фикс: восстанавливаем позиции после рестарта
state.positions = load_positions()
state.auto_reanalyze_task = None
state.early_scanner_task = None
state.data_collector_task = None

UI_CONNECT_TIMEOUT = 4.0
UI_WRITE_TIMEOUT = 6.0
UI_READ_TIMEOUT = 8.0
UI_POOL_TIMEOUT = 4.0
UI_SEND_DEADLINE = 5.0
UI_REQUEST_POOL_SIZE = 32
UPDATES_REQUEST_POOL_SIZE = 4
UPDATES_CONNECT_TIMEOUT = 5.0
UPDATES_WRITE_TIMEOUT = 10.0
UPDATES_READ_TIMEOUT = 10.0
UPDATES_POOL_TIMEOUT = 2.0


BTN_MENU = "Menu"
BTN_SCAN_START = "Анализ + старт"
BTN_SCAN_ONLY = "Только анализ"
BTN_STOP_MONITOR = "Стоп мониторинга"
BTN_POSITIONS = "Позиции"
BTN_WATCHLIST = "Список монет"
BTN_ADD_COIN = "Добавить монету"
BTN_DEL_COIN = "Удалить монету"
BTN_SETTINGS = "Настройки"
BTN_HIDE_MENU = "▾"
BTN_SHOW_MENU = "Показать меню"

REPLY_ACTIONS = {
    BTN_MENU.casefold(): "show_menu",
    BTN_SCAN_START.casefold(): "scan_and_start",
    BTN_SCAN_ONLY.casefold(): "market_scan",
    BTN_STOP_MONITOR.casefold(): "stop_monitor",
    BTN_POSITIONS.casefold(): "positions",
    BTN_WATCHLIST.casefold(): "watchlist",
    BTN_ADD_COIN.casefold(): "add_coin",
    BTN_DEL_COIN.casefold(): "del_coin",
    BTN_SETTINGS.casefold(): "settings",
    BTN_HIDE_MENU.casefold(): "hide_menu",
    BTN_SHOW_MENU.casefold(): "show_menu",
    "скрыть меню": "hide_menu",
}


# ── Keyboards ─────────────────────────────────────────────────────────────────

def kb_main() -> InlineKeyboardMarkup:
    # Единая кнопка с тремя состояниями:
    #   ▶️ Анализ + Мониторинг  — ни анализа, ни мониторинга нет
    #   🔄 Повторный анализ     — мониторинг уже работает (перезапустить анализ)
    #   ⏹ Стоп мониторинга     — остановить всё
    pos  = len(state.positions)
    wl   = len(config.load_watchlist())
    hot  = len(state.hot_coins)
    conf = len([r for r in state.hot_coins if r.today_confirmed])

    if state.running:
        main_btn = InlineKeyboardButton(
            f"⏹ Стоп мониторинга  [{conf} подтв. | {pos} поз.]",
            callback_data="stop_monitor",
        )
        rescan_btn = [InlineKeyboardButton(
            f"🔄 Повторный анализ  [{hot} монет]",
            callback_data="market_scan",
        )]
    else:
        main_btn = InlineKeyboardButton(
            f"▶️ Анализ + Мониторинг  ({wl} монет)",
            callback_data="scan_and_start",
        )
        rescan_btn = [InlineKeyboardButton(
            "🔍 Только анализ",
            callback_data="market_scan",
        )]

    signals_lbl = f"📊 Позиции  [{pos}/{getattr(config,'MAX_OPEN_POSITIONS',6)}]" if pos else "📊 Позиции"
    list_lbl    = f"📋 Список монет  [{wl}]"

    return InlineKeyboardMarkup([
        [main_btn],
        rescan_btn,
        [InlineKeyboardButton(signals_lbl, callback_data="positions")],
        [InlineKeyboardButton(list_lbl,    callback_data="watchlist")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
    ])


def kb_watchlist() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить монету", callback_data="add_coin")],
        [InlineKeyboardButton("➖ Удалить монету",  callback_data="del_coin")],
        [InlineKeyboardButton("🔙 Назад",           callback_data="back_main")],
    ])


def _track_background_task(task: asyncio.Task, name: str) -> asyncio.Task:
    def _done_callback(done: asyncio.Task) -> None:
        try:
            if done.cancelled():
                log.warning("%s task cancelled", name)
                return
            exc = done.exception()
            if exc is not None:
                log.error("%s task failed: %s", name, exc)
            else:
                log.info("%s task finished", name)
        except Exception as cb_err:
            log.error("%s task callback failed: %s", name, cb_err)

    task.add_done_callback(_done_callback)
    return task


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_main")]])


def kb_show_menu_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(BTN_SHOW_MENU, callback_data="show_menu")]]
    )


def kb_menu_root_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Состояние 📊", callback_data="menu_state"),
                InlineKeyboardButton("Управление 🕹️", callback_data="menu_control"),
            ],
            [
                InlineKeyboardButton("Уведомления ⚙️", callback_data="settings"),
            ],
            [
                InlineKeyboardButton("Скрыть меню", callback_data="menu_close_inline"),
            ],
        ]
    )


def kb_menu_state_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Позиции", callback_data="positions"),
                InlineKeyboardButton("Список монет", callback_data="watchlist"),
            ],
            [
                InlineKeyboardButton("↩ Назад", callback_data="show_menu"),
            ],
        ]
    )


def kb_menu_control_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Анализ + старт", callback_data="scan_and_start"),
            ],
            [
                InlineKeyboardButton("Только анализ", callback_data="market_scan"),
                InlineKeyboardButton("Стоп мониторинга", callback_data="stop_monitor"),
            ],
            [
                InlineKeyboardButton("↩ Назад", callback_data="show_menu"),
            ],
        ]
    )


def kb_menu_reply() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_MENU), KeyboardButton(BTN_HIDE_MENU)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Выберите действие",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_truncate(text: str, max_len: int = 4000) -> str:
    """Обрезает текст до max_len символов по границе строки, не посередине тега."""
    if len(text) <= max_len:
        return text
    # Обрезаем по последнему переносу строки в пределах лимита
    cut = text[:max_len].rfind("\n")
    if cut < max_len // 2:
        cut = max_len  # нет переносов — режем жёстко
    return text[:cut] + "\n…"


def _main_menu_text() -> str:
    wl = config.load_watchlist()
    status = "▶️ запущен" if state.running else "⏹ остановлен"
    try:
        import build_info as _bi
        _build_str = _bi.get_build_info_str()
    except Exception:
        _build_str = "—"
    return (
        f"👋 *Crypto Trend Bot*\n"
        f"🔖 `{_build_str}`\n\n"
        f"Мониторинг: {status}\n"
        f"Монет в списке: *{len(wl)}*\n"
        f"Монет «в игре» сегодня: *{len(state.hot_coins)}*\n"
        f"Открытых сигналов: *{len(state.positions)}*"
    )


async def _await_with_deadline(awaitable, timeout: float = UI_SEND_DEADLINE):
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def _send(chat_id: int, text: str, app: Application) -> None:
    await _await_with_deadline(
        app.bot.send_message(
            chat_id=chat_id,
            text=_safe_truncate(text),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_reply(),
            connect_timeout=UI_CONNECT_TIMEOUT,
            write_timeout=UI_WRITE_TIMEOUT,
            read_timeout=UI_READ_TIMEOUT,
            pool_timeout=UI_POOL_TIMEOUT,
        )
    )


def _is_retryable_send_error(exc: Exception) -> bool:
    if isinstance(exc, (TimedOut, NetworkError, TimeoutError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "timed out",
            "timeout",
            "server disconnected",
            "remoteprotocolerror",
            "networkerror",
        )
    )


async def _send_with_retry(chat_id: int, text: str, app: Application, retries: int = 2) -> None:
    for attempt in range(retries + 1):
        try:
            await _send(chat_id, text, app)
            return
        except Exception as e:
            if not _is_retryable_send_error(e) or attempt >= retries:
                raise
            delay = min(4.0, 1.0 * (2 ** attempt))
            log.warning(
                "broadcast send retry %s/%s for %s after %.1fs: %s",
                attempt + 1,
                retries,
                chat_id,
                delay,
                e,
            )
            await asyncio.sleep(delay)


def _with_fast_ui_timeouts(kwargs: dict) -> dict:
    merged = dict(kwargs)
    merged.setdefault("connect_timeout", UI_CONNECT_TIMEOUT)
    merged.setdefault("write_timeout", UI_WRITE_TIMEOUT)
    merged.setdefault("read_timeout", UI_READ_TIMEOUT)
    merged.setdefault("pool_timeout", UI_POOL_TIMEOUT)
    return merged


async def _reply_text_with_retry(msg, text: str, retries: int = 1, **kwargs):
    send_kwargs = _with_fast_ui_timeouts(kwargs)
    for attempt in range(retries + 1):
        try:
            return await _await_with_deadline(
                msg.reply_text(text, **send_kwargs)
            )
        except Exception as e:
            if not _is_retryable_send_error(e) or attempt >= retries:
                raise
            delay = min(4.0, 1.0 * (2 ** attempt))
            log.warning(
                "ui send retry %s/%s after %.1fs: %s",
                attempt + 1,
                retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)


async def _answer_callback_fast(query, text: str | None = None) -> None:
    send_kwargs = _with_fast_ui_timeouts({})
    send_kwargs["cache_time"] = 1
    if text:
        send_kwargs["text"] = text
    try:
        await asyncio.wait_for(query.answer(**send_kwargs), timeout=3.0)
    except Exception as e:
        log.warning("callback answer failed: %s", e)


async def _edit_callback_text_with_retry(query, text: str, retries: int = 1, **kwargs):
    send_kwargs = _with_fast_ui_timeouts(kwargs)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await _await_with_deadline(
                query.edit_message_text(text, **send_kwargs)
            )
        except Exception as e:
            last_exc = e
            if not _is_retryable_send_error(e) or attempt >= retries:
                break
            delay = min(3.0, 0.75 * (2 ** attempt))
            log.warning(
                "ui edit retry %s/%s after %.1fs: %s",
                attempt + 1,
                retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    msg = getattr(query, "message", None)
    if msg is not None:
        try:
            return await _reply_text_with_retry(msg, text, retries=0, **send_kwargs)
        except Exception as e:
            log.warning("ui edit fallback failed: %s", e)
            if last_exc is None:
                last_exc = e
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("callback edit failed without exception")


async def _send_hide_menu_followup(msg) -> None:
    try:
        await _reply_text_with_retry(
            msg,
            "Меню можно вернуть в любой момент.",
            reply_markup=kb_show_menu_inline(),
        )
    except Exception as e:
        log.warning("hide menu follow-up failed: %s", e)


def _menu_panel_text(prefix: str | None = None) -> str:
    text = "Выберите раздел ниже."
    if prefix:
        return f"{prefix}\n\n{text}"
    return text


async def _show_menu_panel_message(msg, force_refresh: bool = False, prefix: str | None = None) -> None:
    text = _menu_panel_text(prefix)
    if force_refresh:
        try:
            await _reply_text_with_retry(
                msg,
                "Меню активно.",
                reply_markup=kb_menu_reply(),
            )
        except Exception as e:
            log.warning("menu keyboard refresh failed: %s", e)
    await _reply_text_with_retry(
        msg,
        text,
        reply_markup=kb_menu_root_inline(),
    )


async def _show_main_menu_message(msg, force_refresh: bool = False, prefix: str | None = None) -> None:
    text = _main_menu_text()
    if prefix:
        text = f"{prefix}\n\n{text}"
    if force_refresh:
        try:
            await _reply_text_with_retry(
                msg,
                "Меню активно.",
                reply_markup=kb_menu_reply(),
            )
        except Exception as e:
            log.warning("menu keyboard refresh failed: %s", e)
    await _reply_text_with_retry(
        msg,
        text + "\n\nВыберите раздел ниже.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_menu_root_inline(),
    )


class _ReplyActionQuery:
    def __init__(self, message, action: str) -> None:
        self.message = message
        self.data = action

    async def answer(self) -> None:
        return None

    async def edit_message_text(self, text: str, **kwargs) -> None:
        kwargs.setdefault("reply_markup", kb_menu_reply())
        await _reply_text_with_retry(self.message, text, **kwargs)


def _sorted_position_items(positions=None, hot_coins=None):
    from monitor import _signal_priority

    positions = state.positions if positions is None else positions
    hot_coins = state.hot_coins if hot_coins is None else hot_coins
    hot_by_sym = {r.symbol: r for r in hot_coins}

    def _sort_key(item):
        sym, pos = item
        coin_report = hot_by_sym.get(sym)
        forecast_return = float(getattr(coin_report, "forecast_return_pct", 0.0))
        if abs(forecast_return) < 1e-12:
            forecast_return = float(getattr(pos, "forecast_return_pct", 0.0))
        today_change = float(getattr(coin_report, "today_change_pct", 0.0))
        if abs(today_change) < 1e-12:
            today_change = float(getattr(pos, "today_change_pct", 0.0))
        signal_mode = getattr(coin_report, "signal_mode", getattr(pos, "signal_mode", ""))
        best_accuracy = float(getattr(coin_report, "best_accuracy", 0.0))
        signal_now = bool(getattr(coin_report, "signal_now", False))
        today_confirmed = bool(getattr(coin_report, "today_confirmed", False))
        ranker_final_score = float(getattr(pos, "ranker_final_score", 0.0))
        entry_quality = float(getattr(pos, "entry_adx", 0.0)) * max(
            0.0, float(getattr(pos, "entry_vol_x", 0.0))
        )
        return (
            not signal_now,
            not today_confirmed,
            -ranker_final_score,
            -_signal_priority(signal_mode),
            -forecast_return,
            int(getattr(pos, "bars_elapsed", 0)),
            -best_accuracy,
            -entry_quality,
            -today_change,
            sym,
        )

    return sorted(positions.items(), key=_sort_key)


def _position_ranker_line(pos) -> str:
    final_score = float(getattr(pos, "ranker_final_score", 0.0))
    ranker_ev = float(getattr(pos, "ranker_ev", 0.0))
    top_prob = float(getattr(pos, "ranker_top_gainer_prob", 0.0))
    capture_pred = float(getattr(pos, "ranker_capture_ratio_pred", 0.0))
    quality_proba = float(getattr(pos, "ranker_quality_proba", 0.0))
    entry_ml_proba = float(getattr(pos, "entry_ml_proba", 0.0))
    candidate_score = float(getattr(pos, "candidate_score_at_entry", 0.0))
    if (
        abs(final_score) < 1e-12
        and abs(ranker_ev) < 1e-12
        and abs(top_prob) < 1e-12
        and abs(capture_pred) < 1e-12
        and abs(quality_proba) < 1e-12
        and abs(entry_ml_proba) < 1e-12
        and abs(candidate_score) < 1e-12
    ):
        return ""
    parts = []
    if abs(candidate_score) >= 1e-12:
        parts.append(f"Score {candidate_score:.1f}")
    if abs(final_score) >= 1e-12:
        parts.append(f"Rank {final_score:+.2f}")
    if abs(ranker_ev) >= 1e-12:
        parts.append(f"EV {ranker_ev:+.2f}")
    if abs(quality_proba) >= 1e-12:
        parts.append(f"Q {quality_proba:.2f}")
    elif abs(entry_ml_proba) >= 1e-12:
        parts.append(f"ML {entry_ml_proba:.2f}")
    if abs(top_prob) >= 1e-12:
        parts.append(f"TG {top_prob:.2f}")
    if abs(capture_pred) >= 1e-12:
        parts.append(f"CAP {capture_pred:.2f}")
    return "  <code>" + " | ".join(parts) + "</code>\n"


def _positions_message_html() -> str:
    if not state.positions:
        max_pos = getattr(config, "MAX_OPEN_POSITIONS", 6)
        return f"📊 Активных позиций нет.  <i>(лимит: {max_pos})</i>"

    import html as _html
    from monitor import _get_coin_group

    max_len = 4000
    max_pos = getattr(config, "MAX_OPEN_POSITIONS", 6)
    n_open = len(state.positions)
    port_bar = "█" * n_open + "░" * (max_pos - n_open)
    lines = [
        f"📊 <b>Открытых позиций: {n_open}/{max_pos}</b>\n",
        f"<b>Портфель:</b> {n_open}/{max_pos}  <code>{port_bar}</code>\n",
    ]

    hot_by_sym = {r.symbol: r for r in state.hot_coins}

    shown = 0
    for sym, pos in _sorted_position_items():
        scan_icon = " 🔍" if any(r.symbol == sym and r.from_scan for r in state.hot_coins) else ""
        grp = _get_coin_group(sym)
        grp_str = f"  <i>[{_html.escape(grp)}]</i>" if grp else ""

        ev_line = ""
        coin_report = hot_by_sym.get(sym)
        if coin_report and coin_report.today_accuracy:
            ev_parts = []
            for h, fa in sorted(coin_report.today_accuracy.items()):
                if fa.total > 0 and fa.expected_return is not None:
                    ev_icon = "▲" if fa.expected_return > 0 else "▼"
                    ev_parts.append(f"T+{h}:{fa.pct:.0f}%{ev_icon}{fa.expected_return:+.2f}%")
                elif fa.total > 0:
                    ev_parts.append(f"T+{h}:{fa.pct:.0f}%")
            if ev_parts:
                ev_line = f"  📊 {' '.join(ev_parts)}\n"

        ranker_line = _position_ranker_line(pos)

        block = (
            f"<b>{_html.escape(sym)}</b>{scan_icon}{grp_str}  "
            f"<code>[{_html.escape(pos.tf)}]</code>\n"
            f"  💰 Вход: <code>{pos.entry_price:.6g}</code>  "
            f"⏱ {pos.bars_elapsed}б\n"
            + ev_line +
            ranker_line +
            f"  🎯 {pos.prediction_summary()}\n"
        )
        current_len = sum(len(line) for line in lines)
        if current_len + len(block) > max_len:
            remaining = len(state.positions) - shown
            lines.append(f"\n<i>...и ещё {remaining} позиций</i>")
            break
        lines.append(block)
        shown += 1

    return "\n".join(lines)


async def _show_positions_message(msg) -> None:
    await _reply_text_with_retry(
        msg,
        _positions_message_html(),
        parse_mode=ParseMode.HTML,
        reply_markup=kb_menu_reply(),
    )


async def _dispatch_reply_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str) -> None:
    pseudo_update = SimpleNamespace(
        callback_query=_ReplyActionQuery(update.effective_message, action)
    )
    await btn(pseudo_update, ctx)


def _early_signal_line(r) -> str:
    """Компактная строка для секции ⚡ — сигнал активен но не подтверждён."""
    acc_parts = "  ".join(
        str(r.today_accuracy[h])
        for h in config.FORWARD_BARS
        if h in r.today_accuracy and r.today_accuracy[h].total > 0
    ) or "нет оценок"
    return (
        f"⚡ *{r.symbol}* `[{r.tf}]`  "
        f"сигналов сегодня: {r.today_signals}  {acc_parts}\n"
        f"   Цена: `{r.current_price:.6g}`  "
        f"slope:`{r.current_slope:+.2f}%`  "
        f"RSI:`{r.current_rsi:.1f}`  "
        f"ADX:`{r.current_adx:.1f}`  "
        f"vol×:`{r.current_vol_x:.2f}`"
    )


def _format_analysis_result(in_play, skipped, title: str, scan: bool = False) -> str:
    lines = [title, ""]

    if in_play:
        active  = [r for r in in_play if r.signal_now]
        waiting = [r for r in in_play if not r.signal_now]

        header = f"✅ *Подтверждено сегодня — {len(in_play)} монет:*"
        if active:
            header += f"\n🟢 *Сигнал активен прямо сейчас: {len(active)}*"
        lines.append(header)

        for r in active:
            lines.append(r.summary())

        if waiting:
            lines.append(f"\n⏸ *Стратегия подтверждена, ждут сигнала: {len(waiting)}*")
            for r in waiting:
                lines.append(r.summary())
    else:
        lines.append(
            "❌ Ни одна монета не прошла порог точности.\n"
            f"Порог точности: {config.MIN_ACCURACY}%  |  T+3 ≥ {config.TODAY_T3_MIN}%  |  T+10 ≥ {config.TODAY_T10_MIN}%\n"
            f"Мин. сигналов сегодня: {config.TODAY_MIN_SIGNALS}"
        )

    # ── ⚡ BUY активен, но форвард-тест не подтверждён ─────────────────────────
    early = sorted(
        [r for r in skipped if r.signal_now],
        key=lambda r: r.today_signals, reverse=True,
    )
    if early:
        lines.append(
            f"\n⚡ *BUY активен, не подтверждены ({len(early)}):*\n"
            f"_Форвард-тест не накопил данных — повышенный риск_"
        )
        for r in early[:6]:
            lines.append(_early_signal_line(r))
        if len(early) > 6:
            lines.append(f"_...и ещё {len(early) - 6} с активным BUY_")

    # ── 🟡 SETUP — тренд зарождается, до BUY не хватает 1 фильтра ───────────
    setup_list = sorted(
        [r for r in skipped if r.setup_now and not r.signal_now],
        key=lambda r: r.today_signals, reverse=True,
    )
    if setup_list:
        lines.append(f"\n🟡 *SETUP — структура складывается ({len(setup_list)}):*")
        for r in setup_list[:6]:
            lines.append(
                f"🟡 *{r.symbol}* `[{r.tf}]`  "
                f"`{r.current_price:.6g}`  "
                f"slope:`{r.current_slope:+.2f}%`  "
                f"RSI:`{r.current_rsi:.1f}`  "
                f"vol×:`{r.current_vol_x:.2f}`\n"
                + "   _" + (r.setup_reason
                    .replace("[", "\\[").replace("]", "\\]")
                    .replace("_", "\\_").replace("(", "\\(").replace(")", "\\)")
                ) + "_"
            )
        if len(setup_list) > 6:
            lines.append(f"_...и ещё {len(setup_list) - 6} в SETUP_")

    # ── Пропущенные без сигнала ───────────────────────────────────────────────
    cold = [r for r in skipped if not r.signal_now and not r.setup_now]
    if cold:
        lines.append(f"\n⏭ *Пропущено ({len(cold)}):*")
        for r in cold[:3]:
            lines.append(r.summary())
        if len(cold) > 3:
            lines.append(f"_...и ещё {len(cold) - 3} монет не прошли порог_")

    return "\n".join(lines)


# ── Handlers ──────────────────────────────────────────────────────────────────

# Хранилище chat_id для авто-уведомлений и реанализа
_known_chat_ids: set[int] = set()

try:
    _chat_ids_file = Path(".chat_ids")
    if _chat_ids_file.exists():
        import json as _json
        _known_chat_ids = set(_json.loads(_chat_ids_file.read_text()))
except Exception:
    pass


def _save_chat_id(chat_id: int) -> None:
    _known_chat_ids.add(chat_id)
    try:
        import json as _json
        Path(".chat_ids").write_text(_json.dumps(list(_known_chat_ids)))
    except Exception:
        pass


def _refresh_positions_state() -> None:
    """
    Sync in-memory positions with the persisted positions file before rendering UI.
    This keeps chat views consistent even if positions were closed by the monitor
    loop shortly before the user requested a status screen.
    """
    try:
        live_positions = dict(state.positions)
        loaded_positions = load_positions()
        hot_by_sym = {r.symbol: r for r in state.hot_coins}

        # Keep fresh in-memory metrics if the same position is still open.
        for sym, loaded_pos in loaded_positions.items():
            live_pos = live_positions.get(sym)
            if live_pos is not None:
                for attr in (
                    "bars_elapsed",
                    "predictions",
                    "trail_stop",
                    "macd_warn_streak",
                    "macd_warned",
                    "last_macd_bar_i",
                    "forecast_return_pct",
                    "today_change_pct",
                ):
                    if hasattr(live_pos, attr):
                        setattr(loaded_pos, attr, getattr(live_pos, attr))

            report = hot_by_sym.get(sym)
            if report is not None:
                forecast_return = float(getattr(report, "forecast_return_pct", getattr(loaded_pos, "forecast_return_pct", 0.0)))
                today_change = float(getattr(report, "today_change_pct", getattr(loaded_pos, "today_change_pct", 0.0)))
                if abs(forecast_return) > 1e-12:
                    loaded_pos.forecast_return_pct = forecast_return
                if abs(today_change) > 1e-12:
                    loaded_pos.today_change_pct = today_change

        state.positions = loaded_positions
    except Exception as e:
        log.warning("refresh positions failed: %s", e)


def _make_broadcast_send(app: "Application"):
    """
    Возвращает функцию send которая рассылает сообщение всем известным chat_id.
    Используется для monitoring_loop запущенного из авто-реанализа.
    """
    async def _broadcast(text: str) -> None:
        for cid in list(_known_chat_ids):
            try:
                await _send_with_retry(cid, text, app)
            except Exception as e:
                log.warning("broadcast send failed for %s: %s", cid, e)
        # Сохраняем chat_ids на диск при каждой отправке
        try:
            import json as _json
            Path(".chat_ids").write_text(_json.dumps(list(_known_chat_ids)))
        except Exception:
            pass
    return _broadcast



def _update_hot_coins(state, in_play, skipped) -> None:
    """
    Обновляет state.hot_coins:
    - confirmed монеты (in_play) — мониторятся всегда
    - НЕ-confirmed монеты с signal_now=True — мониторятся с пометкой риска
    
    Это главный фикс: раньше монеты типа TRXUSDT (signal_now=True, confirmed=False)
    полностью игнорировались мониторингом. Теперь они включаются.
    """
    # Confirmed всегда в списке
    hot = list(in_play)
    
    # Добавляем не-confirmed с активным сигналом прямо сейчас
    already = {r.symbol for r in hot}
    for r in skipped:
        if r.signal_now and r.symbol not in already:
            hot.append(r)
    
    state.hot_coins = hot
    _ensure_positions_monitored(state)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return
    log.info("command /start from %s", msg.chat_id)
    chat_id = msg.chat_id
    _save_chat_id(chat_id)
    await _show_main_menu_message(msg, force_refresh=True)
    return
    _refresh_positions_state()
    wl     = config.load_watchlist()
    status = "▶️ запущен" if state.running else "⏹ остановлен"
    text   = (
        f"👋 *Crypto Trend Bot*\n\n"
        f"Мониторинг: {status}\n"
        f"Монет в списке: *{len(wl)}*\n"
        f"Монет «в игре» сегодня: *{len(state.hot_coins)}*\n"
        f"Открытых сигналов: *{len(state.positions)}*\n\n"
        f"Выберите действие кнопками снизу."
    )
    await msg.reply_text(
        text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_menu_reply(),
    )


async def cmd_show_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is not None:
        log.info("command /menu from %s", msg.chat_id)
    if msg is None:
        return
    await _show_menu_panel_message(msg, force_refresh=True, prefix="Меню снова показано.")


async def cmd_hide_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return
    log.info("hide menu requested by %s", msg.chat_id)
    await _reply_text_with_retry(
        msg,
        "Меню скрыто. Нажмите кнопку ниже или отправьте /menu, чтобы вернуть его.",
        reply_markup=ReplyKeyboardRemove(),
    )
    asyncio.create_task(_send_hide_menu_followup(msg))
    return
    await msg.reply_text(
        "Меню скрыто. Нажмите кнопку ниже или отправьте /menu, чтобы вернуть его.",
        reply_markup=ReplyKeyboardRemove(),
    )
    await msg.reply_text(
        "Меню можно вернуть в любой момент.",
        reply_markup=kb_show_menu_inline(),
    )


async def btn(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query   = update.callback_query
    # query.answer() может упасть если запрос пришёл с задержкой >2 мин
    # (бот был выключен, накопились старые нажатия). Проглатываем — action
    # всё равно выполняем, просто спиннер у кнопки не уберётся.
    await _answer_callback_fast(query)
    action  = query.data
    chat_id = query.message.chat_id
    log.info("callback from %s: %s", chat_id, action)

    # ── 🌅 Анализ на день ─────────────────────────────────────────────────────
    if action == "market_scan":
        n = len(config.load_watchlist())
        rescan_note = " (мониторинг продолжается)" if state.running else ""
        await query.edit_message_text(
            f"🔍 Анализирую список — *{n}* монет{rescan_note}...\n"
            f"Форвард-тест на данных сегодняшнего дня.\n"
            f"Займёт ~1–2 минуты.",
            parse_mode=ParseMode.MARKDOWN,
        )
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        mon_line = f"▶️ Мониторинг запущен | Монет: {len(state.hot_coins)}"
        from datetime import datetime, timezone
        now_str    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        conf_count = len([r for r in in_play if r.today_confirmed])
        mon_note   = f"\n▶️ Мониторинг продолжается | Подтверждено: *{conf_count}*" if state.running else ""
        text = _format_analysis_result(
            in_play, skipped,
            f"🔍 *Анализ завершён* — {n} монет  _{now_str}_{mon_note}",
        )
        await query.edit_message_text(
            _safe_truncate(text), parse_mode=ParseMode.MARKDOWN,
        )

    # ── ▶️ Анализ + Мониторинг (единая кнопка) ───────────────────────────────
    elif action == "scan_and_start":
        if state.running:
            # query.answer() уже был вызван выше — дублировать нельзя
            await query.edit_message_text(
                "▶️ Мониторинг уже запущен.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        n = len(config.load_watchlist())
        await query.edit_message_text(
            f"🔍 Анализирую список — *{n}* монет...\n"
            f"Форвард-тест на данных сегодняшнего дня.\n"
            f"Займёт ~1–2 минуты, затем мониторинг запустится автоматически.",
            parse_mode=ParseMode.MARKDOWN,
        )
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        # Автоматически стартуем мониторинг после анализа
        state.running = True

        async def send_msg_sas(text: str) -> None:
            # Отправляем в чат где нажата кнопка + всем остальным подписчикам
            sent_to = set()
            for cid in list(_known_chat_ids):
                try:
                    await _send(cid, text, ctx.application)
                    sent_to.add(cid)
                except Exception:
                    pass
            # Если chat_id почему-то не в _known_chat_ids — шлём отдельно
            if chat_id not in sent_to:
                try:
                    await _send(chat_id, text, ctx.application)
                except Exception:
                    pass

        state.task = asyncio.create_task(monitoring_loop(state, send_msg_sas))

        from datetime import datetime, timezone
        now_str    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        conf_count = len([r for r in in_play if r.today_confirmed])
        text = _format_analysis_result(
            in_play, skipped,
            f"🔍 *Анализ завершён* — {n} монет  _{now_str}_\n"
            f"▶️ *Мониторинг запущен автоматически* | Подтверждено: *{conf_count}*",
        )
        await query.edit_message_text(
            _safe_truncate(text), parse_mode=ParseMode.MARKDOWN,
        )

    # ── ⏹ Стоп мониторинга ───────────────────────────────────────────────────
    elif action == "stop_monitor":
        _refresh_positions_state()
        state.running = False
        if state.task:
            state.task.cancel()
            state.task = None
        await query.edit_message_text(
            f"⏹ *Мониторинг остановлен.*\n"
            f"Открытых сигналов: {len(state.positions)}",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── 📊 Активные сигналы ───────────────────────────────────────────────────
    elif action == "positions":
        await _reply_text_with_retry(
            query.message,
            _positions_message_html(),
            parse_mode=ParseMode.HTML,
            reply_markup=kb_menu_state_inline(),
        )
        return
        _refresh_positions_state()
        if not state.positions:
            max_pos = getattr(config, "MAX_OPEN_POSITIONS", 6)
            txt = f"📊 Активных позиций нет.  <i>(лимит: {max_pos})</i>"
        else:
            import html as _html
            from monitor import _get_coin_group
            MAX_LEN  = 4000
            max_pos  = getattr(config, "MAX_OPEN_POSITIONS", 6)
            max_grp  = getattr(config, "MAX_POSITIONS_PER_GROUP", 2)
            n_open   = len(state.positions)

            # Портфельный статус
            port_pct  = int(n_open / max_pos * 100)
            port_bar  = "█" * (n_open) + "░" * (max_pos - n_open)
            port_line = f"<b>Портфель:</b> {n_open}/{max_pos}  <code>{port_bar}</code>\n"

            lines = [f"📊 <b>Открытых позиций: {n_open}/{max_pos}</b>\n", port_line]
            shown = 0
            for sym, pos in state.positions.items():
                scan_icon = " 🔍" if any(
                    r.symbol == sym and r.from_scan for r in state.hot_coins
                ) else ""
                grp = _get_coin_group(sym)
                grp_str = f"  <i>[{_html.escape(grp)}]</i>" if grp else ""

                # EV из hot_coins если доступно
                ev_line = ""
                coin_report = next(
                    (r for r in state.hot_coins if r.symbol == sym), None
                )
                if coin_report and coin_report.today_accuracy:
                    ev_parts = []
                    for h, fa in sorted(coin_report.today_accuracy.items()):
                        if fa.total > 0 and fa.expected_return is not None:
                            ev_icon = "▲" if fa.expected_return > 0 else "▼"
                            ev_parts.append(f"T+{h}:{fa.pct:.0f}%{ev_icon}{fa.expected_return:+.2f}%")
                        elif fa.total > 0:
                            ev_parts.append(f"T+{h}:{fa.pct:.0f}%")
                    if ev_parts:
                        ev_line = f"  📊 {' '.join(ev_parts)}\n"

                block = (
                    f"<b>{_html.escape(sym)}</b>{scan_icon}{grp_str}  "
                    f"<code>[{_html.escape(pos.tf)}]</code>\n"
                    f"  💰 Вход: <code>{pos.entry_price:.6g}</code>  "
                    f"⏱ {pos.bars_elapsed}б\n"
                    + ev_line +
                    f"  🎯 {pos.prediction_summary()}\n"
                )
                current_len = sum(len(l) for l in lines)
                if current_len + len(block) > MAX_LEN:
                    remaining = len(state.positions) - shown
                    lines.append(f"\n<i>...и ещё {remaining} позиций</i>")
                    break
                lines.append(block)
                shown += 1
            txt = "\n".join(lines)
        await query.edit_message_text(
            txt, parse_mode=ParseMode.HTML,
        )

    # ── 📋 Список монет ───────────────────────────────────────────────────────
    elif action == "watchlist":
        wl   = config.load_watchlist()
        rows = [wl[i:i+5] for i in range(0, len(wl), 5)]
        grid = "\n".join("  ".join(f"`{s}`" for s in row) for row in rows)
        txt  = f"📋 *Ваш список монет* ({len(wl)} шт.):\n\n{grid}"
        await _reply_text_with_retry(
            query.message,
            _safe_truncate(txt), parse_mode=ParseMode.MARKDOWN, reply_markup=kb_watchlist(),
        )

    elif action == "menu_state":
        _refresh_positions_state()
        wl = config.load_watchlist()
        await _reply_text_with_retry(
            query.message,
            (
                f"📊 *Состояние*\n\n"
                f"Мониторинг: {'▶️ запущен' if state.running else '⏹ остановлен'}\n"
                f"Монет в списке: *{len(wl)}*\n"
                f"Монет «в игре»: *{len(state.hot_coins)}*\n"
                f"Открытых сигналов: *{len(state.positions)}*"
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_state_inline(),
        )

    elif action == "menu_control":
        await _reply_text_with_retry(
            query.message,
            (
                "🕹️ *Управление*\n\n"
                "Выберите действие для анализа и мониторинга."
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_control_inline(),
        )

    elif action == "add_coin":
        ctx.user_data["awaiting"] = "add_coin"
        await _reply_text_with_retry(
            query.message,
            "Введите тикер монеты, например `SOLUSDT`:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_reply(),
        )

    elif action == "del_coin":
        ctx.user_data["awaiting"] = "del_coin"
        wl = config.load_watchlist()
        await _reply_text_with_retry(
            query.message,
            "Введите тикер для удаления.\n\nСейчас в списке:\n"
            + "  ".join(f"`{s}`" for s in wl[:20])
            + (" ..." if len(wl) > 20 else ""),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_reply(),
        )

    # ── ⚙️ Настройки ──────────────────────────────────────────────────────────
    elif action == "settings":
        tf_str = ", ".join(config.TIMEFRAMES)
        fb_str = ", ".join(f"T+{h}" for h in config.FORWARD_BARS)
        txt = (
            f"⚙️ *Текущие настройки*\n\n"
            f"*Таймфреймы:* `{tf_str}`\n"
            f"*Горизонты прогноза:* `{fb_str}`\n"
            f"*Порог точности:* `{config.MIN_ACCURACY}%`\n"
            f"*Дневное подтверждение:*\n"
            f"  Мин. сигналов сегодня: `{config.TODAY_MIN_SIGNALS}`\n"
            f"  T+3 ≥ `{config.TODAY_T3_MIN}%`  |  T+10 ≥ `{config.TODAY_T10_MIN}%`\n"
            f"  Лучший горизонт ≥ `{config.MIN_ACCURACY}%`\n\n"
            f"*Условия входа:*\n"
            f"  Цена > EMA20 > EMA50\n"
            f"  EMA20 наклон ≥ `{config.EMA_SLOPE_MIN}%` за {config.SLOPE_LOOKBACK} баров\n"
            f"  ADX ≥ `{config.ADX_MIN}` и ADX > SMA(ADX,{config.ADX_SMA_PERIOD})\n"
            f"  Объём ≥ `{config.VOL_MULT}×` среднего\n"
            f"  RSI: `{config.RSI_BUY_LO}` – `{config.RSI_BUY_HI}`\n"
            f"  MACD гистограмма > 0\n"
            f"  Рост от дна дня < `{config.DAILY_RANGE_MAX}%`\n\n"
            f"*Условия выхода:*\n"
            f"  ATR-трейлинг (×`{config.ATR_TRAIL_K}` ATR / сильный тренд ×`{getattr(config,'ATR_TRAIL_K_STRONG', config.ATR_TRAIL_K + 0.5):.1f}`)\n"
            f"  2 закрытия подряд ниже EMA20 — ранний разворот _(новое)_\n"
            f"  Одиночное закрытие ниже EMA20 — страховка\n"
            f"  RSI > `{config.RSI_OVERBOUGHT}`\n"
            f"  EMA20 slope < 0\n"
            f"  ADX < `{config.ADX_DROP_RATIO*100:.0f}%` × ADX[{config.ADX_GROW_BARS} бара назад]\n"
            f"  Лимит: `{config.MAX_HOLD_BARS}` баров\n\n"
            f"*ADX-bypass:* при ADX ≥ `{getattr(config,'ADX_SMA_BYPASS',35.0):.0f}` фильтр ADX>SMA пропускается\n"
            f"⚠️ MACD-предупреждение: падает `{config.MACDWARN_BARS}` бара подряд\n"
            f"🔄 Авто-реанализ: каждые `{config.AUTO_REANALYZE_SEC//3600}ч`\n\n"
            f"⏱ Интервал опроса: `{config.POLL_SEC}с`\n\n"
            f"_Для изменения — отредактируйте config.py_"
        )
        await _reply_text_with_retry(
            query.message,
            txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_back(),
        )

    elif action == "back_main":
        _refresh_positions_state()
        wl = config.load_watchlist()
        await _reply_text_with_retry(
            query.message,
            f"Монет в списке: *{len(wl)}*  |  В игре: *{len(state.hot_coins)}*  |  "
            f"Сигналов: *{len(state.positions)}*\n\n"
            f"Используйте кнопки снизу для следующего действия.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_menu_root_inline(),
        )


# ── Text input (add / remove coin) ────────────────────────────────────────────

    elif action == "show_menu":
        try:
            await _reply_text_with_retry(
                query.message,
                "Меню снова показано.",
                reply_markup=kb_menu_reply(),
            )
        except Exception as e:
            log.warning("show menu keyboard restore failed: %s", e)
        await _reply_text_with_retry(
            query.message,
            _menu_panel_text(),
            reply_markup=kb_menu_root_inline(),
        )
        return

    elif action == "menu_close_inline":
        await _reply_text_with_retry(
            query.message,
            "Меню скрыто. Нажмите `Menu` или `/menu`, чтобы открыть его снова.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_show_menu_inline(),
        )
        return


async def cmd_why(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /why SYMBOL — показывает почему нет сигнала по монете прямо сейчас."""
    parts = (update.message.text or "").strip().split()
    if len(parts) < 2:
        await update.message.reply_text(
            "Использование: `/why SYMBOL`\nПример: `/why TONUSDT`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    symbol = parts[1].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    await update.message.reply_text(
        f"🔍 Проверяю *{symbol}*...", parse_mode=ParseMode.MARKDOWN
    )

    import aiohttp as _aiohttp
    import numpy as np
    from indicators import compute_features
    from strategy import (
        check_retest_conditions, check_breakout_conditions,
        check_impulse_conditions, check_alignment_conditions,
    )

    results = []
    async with _aiohttp.ClientSession() as session:
        for tf in config.TIMEFRAMES:
            # Используем LIVE_LIMIT — те же данные что и у мониторинга
            data = await fetch_klines(session, symbol, tf, limit=config.LIVE_LIMIT)
            if data is None:
                results.append(f"`{tf}`: ❌ нет данных с Binance")
                continue

            c    = data["c"].astype(float)
            feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])
            i    = len(c) - 2

            ef    = float(feat["ema_fast"][i])
            es    = float(feat["ema_slow"][i])
            slp   = float(feat["slope"][i])
            rsi   = float(feat["rsi"][i])
            adx   = float(feat["adx"][i])
            adx_s = float(feat["adx_sma"][i]) if np.isfinite(feat["adx_sma"][i]) else 0.0
            vx    = float(feat["vol_x"][i])
            dr    = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
            price = float(c[i])

            buy_ok,  buy_r  = check_entry_conditions(feat, i, c)
            ret_ok,  ret_r  = check_retest_conditions(feat, i)
            brk_ok,  brk_r  = check_breakout_conditions(feat, i)
            imp_ok,  imp_r  = check_impulse_conditions(feat, i)
            aln_ok,  aln_r  = check_alignment_conditions(feat, i)
            stp_ok,  stp_r, _ = check_setup_conditions(feat, i, c)

            any_sig = buy_ok or ret_ok or brk_ok or imp_ok or aln_ok

            def _fmt(ok, reason, label):
                icon = "🟢" if ok else "🔴"
                return f"{icon} {label}" + ("" if ok else f": {reason[:80]}")

            lines = [
                f"*{tf}*  `{price:.6g}`  EMA20:`{ef:.4g}`  EMA50:`{es:.4g}`",
                f"  slope:`{slp:+.2f}%`  ADX:`{adx:.1f}`(sma:{adx_s:.1f})  RSI:`{rsi:.1f}`  vol×:`{vx:.2f}`  dr:`{dr:.1f}%`",
                f"  {'✅ СИГНАЛ ЕСТЬ' if any_sig else '⛔ сигнала нет'}",
                f"  {_fmt(buy_ok, buy_r, 'BUY')}",
                f"  {_fmt(ret_ok, ret_r, 'RETEST')}",
                f"  {_fmt(brk_ok, brk_r, 'BREAKOUT')}",
                f"  {_fmt(imp_ok, imp_r, 'IMPULSE')}",
                f"  {_fmt(aln_ok, aln_r, 'ALIGNMENT')}",
            ]
            if not any_sig:
                lines.append(f"  ⬜ SETUP: {'есть' if stp_ok else stp_r[:80]}")
            results.append("\n".join(lines))

    text = f"🔍 *{symbol}* (данные мониторинга, {config.LIVE_LIMIT} баров)\n\n" + "\n\n".join(results)
    await update.message.reply_text(_safe_truncate(text), parse_mode=ParseMode.MARKDOWN)


async def cmd_test(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /test — диагностика состояния бота и мониторинга."""
    from datetime import datetime, timezone
    chat_id = update.message.chat_id

    task_alive = False
    task_status = "⏹ остановлен"
    if state.task is not None:
        if state.task.done():
            exc = state.task.exception() if not state.task.cancelled() else None
            task_status = f"💀 зомби (упал: {exc})" if exc else "💀 зомби (отменён)"
            # Реанимируем
            if state.running:
                state.task = asyncio.create_task(
                    monitoring_loop(state, _make_broadcast_send(ctx.application))
                )
                task_status += " → 🔄 перезапущен"
        else:
            task_alive = True
            task_status = "▶️ работает"
    elif state.running:
        task_status = "⚠️ running=True но task=None"

    import time as _time
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ms  = int(_time.time() * 1000)

    # Позиции
    pos_lines = ""
    if state.positions:
        lines = []
        for sym, pos in state.positions.items():
            bars_in = (now_ms - pos.entry_ts) // (
                15 * 60 * 1000 if pos.tf == "15m" else 60 * 60 * 1000
            )
            lines.append(f"  {sym} [{pos.tf}] {pos.signal_mode} @{pos.entry_price:.6g} | {bars_in} баров")
        pos_lines = "\n<b>💼 Открытые позиции:</b>\n" + "\n".join(lines)
    else:
        pos_lines = "\n💼 Открытых позиций: 0"

    # Кулдауны
    cd_lines = ""
    if state.cooldowns:
        active_cd = {s: u for s, u in state.cooldowns.items() if u > now_ms}
        if active_cd:
            lines = []
            for sym, until_ms in sorted(active_cd.items(), key=lambda x: x[1]):
                left_min = max(0, (until_ms - now_ms) // 60000)
                left_h   = left_min // 60
                left_m   = left_min % 60
                time_str = f"{left_h}ч {left_m}м" if left_h else f"{left_m}м"
                lines.append(f"  {sym}: ещё {time_str}")
            cd_lines = f"\n\n<b>⏳ Кулдаун ({len(active_cd)} монет):</b>\n" + "\n".join(lines)
        else:
            cd_lines = "\n\n⏳ Кулдаунов нет"
    else:
        cd_lines = "\n\n⏳ Кулдаунов нет"

    # HTML — не зависит от [] () * _ в exception-тексте и chat_id
    text = (
        f"✅ <b>Тестовое сообщение</b>\n\n"
        f"🕐 Время UTC: <code>{now_str}</code>\n"
        f"📡 Мониторинг: {task_status}\n"
        f"👀 Монет в слежке: {len(state.hot_coins)}"
        f"{pos_lines}"
        f"{cd_lines}\n\n"
        f"📬 Известных чатов: <code>{list(_known_chat_ids)}</code>\n"
        f"🔑 Этот chat_id: <code>{chat_id}</code>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    incoming = (update.message.text or "").strip()
    ingress_lag_ms = None
    try:
        if update.message and update.message.date:
            msg_dt = update.message.date
            if msg_dt.tzinfo is None:
                msg_dt = msg_dt.replace(tzinfo=timezone.utc)
            ingress_lag_ms = max(
                0.0,
                (datetime.now(timezone.utc) - msg_dt).total_seconds() * 1000.0,
            )
    except Exception:
        ingress_lag_ms = None

    if ingress_lag_ms is None:
        log.info("text received from %s: %r", update.effective_chat.id, incoming)
    else:
        log.info(
            "text received from %s: %r (ingress %.0fms)",
            update.effective_chat.id,
            incoming,
            ingress_lag_ms,
        )
    lowered = incoming.casefold()
    action = REPLY_ACTIONS.get(lowered)
    if action:
        started_at = time.perf_counter()
        ctx.user_data.pop("awaiting", None)
        log.info("reply action received from %s: %s", update.effective_chat.id, action)
        try:
            if action == "hide_menu":
                await cmd_hide_menu(update, ctx)
            elif action == "positions":
                await _show_positions_message(update.message)
            elif action == "show_menu":
                await _show_menu_panel_message(update.message, force_refresh=False)
            else:
                await _dispatch_reply_action(update, ctx, action)
        except Exception as e:
            log.exception("reply action failed: %s", action)
            await update.message.reply_text(
                f"⚠️ Команда не выполнилась: {e}",
                reply_markup=kb_menu_reply(),
            )
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            log.info(
                "reply action completed for %s: %s in %.0fms",
                update.effective_chat.id,
                action,
                elapsed_ms,
            )
        return
    if incoming.lower() in {"menu", "меню"}:
        ctx.user_data.pop("awaiting", None)
        await _show_menu_panel_message(update.message, force_refresh=False)
        return

    awaiting = ctx.user_data.get("awaiting")
    if not awaiting:
        await cmd_start(update, ctx)
        return

    ticker = update.message.text.strip().upper()
    wl     = config.load_watchlist()

    if awaiting == "add_coin":
        if ticker in wl:
            await update.message.reply_text(
                f"⚠️ `{ticker}` уже есть в списке.", parse_mode=ParseMode.MARKDOWN,
            )
        else:
            wl.append(ticker)
            config.save_watchlist(wl)
            await update.message.reply_text(
                f"✅ `{ticker}` добавлен. Всего: {len(wl)}",
                parse_mode=ParseMode.MARKDOWN, reply_markup=kb_menu_reply(),
            )

    elif awaiting == "del_coin":
        if ticker not in wl:
            await update.message.reply_text(
                f"⚠️ `{ticker}` не найден.", parse_mode=ParseMode.MARKDOWN,
            )
        else:
            wl.remove(ticker)
            config.save_watchlist(wl)
            state.hot_coins = [r for r in state.hot_coins if r.symbol != ticker]
            await update.message.reply_text(
                f"✅ `{ticker}` удалён. Осталось: {len(wl)}",
                parse_mode=ParseMode.MARKDOWN, reply_markup=kb_menu_reply(),
            )

    ctx.user_data.pop("awaiting", None)


# ── Entry point ───────────────────────────────────────────────────────────────

def _ensure_positions_monitored(state) -> None:
    """
    Гарантирует что монеты с открытыми позициями всегда в hot_coins.
    Фикс 15.03.2026: реанализ больше не закрывает открытые позиции.
    Позиция выходит только по техническим условиям (ATR/EMA20/RSI/hold).
    """
    from strategy import CoinReport
    hot_syms = {r.symbol for r in state.hot_coins}
    for sym, pos in list(state.positions.items()):
        if sym not in hot_syms:
            from strategy import CoinReport
            dummy = CoinReport(
                symbol=sym, tf=pos.tf,
                today_signals=[], today_confirmed=True,
                signal_now=False, today_accuracy={},
                best_accuracy=0.0,
                note="удерживаем позицию (exit guard)",
                in_play=True,
            )
            state.hot_coins.append(dummy)
            log.info("_ensure_positions_monitored: keeping %s [%s] in hot_coins", sym, pos.tf)




def _ensure_positions_monitored(state) -> None:
    """
    Guarantees that restored open positions stay inside hot_coins so exit logic
    keeps polling them immediately after restart.
    """
    from strategy import CoinReport

    hot_syms = {r.symbol for r in state.hot_coins}
    for sym, pos in list(state.positions.items()):
        if sym in hot_syms:
            continue

        state.hot_coins.append(
            CoinReport(
                symbol=sym,
                tf=pos.tf,
                today_signals=0,
                today_accuracy={},
                today_confirmed=True,
                best_horizon=0,
                best_accuracy=0.0,
                in_play=True,
                note="holding restored position (exit guard)",
                signal_now=False,
            )
        )
        hot_syms.add(sym)
        log.info("_ensure_positions_monitored: keeping %s [%s] in hot_coins", sym, pos.tf)


async def _auto_reanalyze(app: Application) -> None:
    """Фоновая задача: пересчитывает список монет каждые AUTO_REANALYZE_SEC секунд."""
    if config.AUTO_REANALYZE_SEC <= 0:
        return
    first_delay_sec = max(
        0,
        int(getattr(config, "AUTO_REANALYZE_FIRST_DELAY_SEC", config.AUTO_REANALYZE_SEC)),
    )
    next_delay_sec = first_delay_sec if first_delay_sec > 0 else int(config.AUTO_REANALYZE_SEC)
    first_cycle = True
    while True:
        waited_sec = next_delay_sec
        await asyncio.sleep(waited_sec)
        if first_cycle:
            first_cycle = False
            next_delay_sec = int(config.AUTO_REANALYZE_SEC)
        try:
            from strategy import market_scan, check_entry_conditions, check_setup_conditions, analyze_coin, fetch_klines, get_entry_mode
            from datetime import datetime, timezone
            log.info("Auto-reanalyze started after %ss", waited_sec)
            in_play, skipped = await market_scan()
            _update_hot_coins(state, in_play, skipped)
            # Фикс 15.03.2026: монеты с позицией остаются в мониторинге
            _ensure_positions_monitored(state)
            now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
            n = len(config.load_watchlist())

            # ── Авто-старт мониторинга если он не работает (после перезапуска бота) ──
            _task_dead = state.task is not None and state.task.done()
            if _task_dead:
                exc = state.task.exception() if not state.task.cancelled() else None
                log.warning("Auto-reanalyze: zombie task detected (exc=%s) — restarting", exc)
                state.task = None
                state.running = False

            if in_play and (not state.running or _task_dead):
                state.running = True
                state.task = asyncio.create_task(
                    monitoring_loop(state, _make_broadcast_send(app))
                )
                mon_note = f"\n▶️ *Мониторинг запущен автоматически*"
            elif not in_play and state.running and not state.positions:
                # Нет подтверждённых монет и нет открытых позиций — останавливаем
                state.running = False
                if state.task:
                    state.task.cancel()
                    state.task = None
                mon_note = "\n⏹ Мониторинг приостановлен (нет подтверждённых)"
            else:
                mon_note = ""

            if getattr(config, "SEND_SERVICE_NOTIFICATIONS", False):
                msg = (
                    f"🔄 *Авто-реанализ* _{now_str}_\n"
                    f"Пересчитано {n} монет.\n"
                    f"Подтверждено: *{len(in_play)}*"
                    f"{mon_note}"
                )
                for cid in list(_known_chat_ids):
                    try:
                        await _send(cid, msg, app)
                    except Exception:
                        pass
        except Exception as e:
            log.error("Auto-reanalyze error: %s", e)


async def _run_post_init_scan(app: Application, notify_service: bool, watchlist_size: int) -> None:
    try:
        from strategy import market_scan

        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        if state.running:
            mon_line = f"в–¶пёЏ РњРѕРЅРёС‚РѕСЂРёРЅРі Р·Р°РїСѓС‰РµРЅ | РњРѕРЅРµС‚: {len(state.hot_coins)}"
        else:
            state.task = asyncio.create_task(monitoring_loop(state, _make_broadcast_send(app)))
            state.running = True
            mon_line = f"в–¶пёЏ РњРѕРЅРёС‚РѕСЂРёРЅРі Р·Р°РїСѓС‰РµРЅ | РњРѕРЅРµС‚: {len(state.hot_coins)}"

        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        if notify_service:
            for cid in list(_known_chat_ids):
                try:
                    await app.bot.send_message(
                        chat_id=cid,
                        text=f"🔍 Анализ завершён — {watchlist_size} монет  {now_str}\n{mon_line}",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_menu_reply(),
                    )
                except Exception as e:
                    log.warning("post_init result notify failed for %s: %s", cid, e)
    except Exception as e:
        log.error("post_init авто-анализ упал: %s", e)
        if notify_service:
            for cid in list(_known_chat_ids):
                try:
                    await app.bot.send_message(
                        chat_id=cid,
                        text="⚠️ Авто-анализ не удался. Нажмите *🔍 Анализ* вручную.",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_menu_reply(),
                    )
                except Exception:
                    pass


async def _post_init(app: Application) -> None:
    """При старте: уведомляет пользователей и автоматически запускает анализ+мониторинг."""
    state.auto_reanalyze_task = _track_background_task(
        asyncio.create_task(_auto_reanalyze(app)),
        "auto_reanalyze",
    )
    log.info("Auto-reanalyze task scheduled")
    if getattr(config, "ENABLE_EARLY_SCANNER_ALERTS", False):
        try:
            import impulse_scanner as _is
            state.early_scanner_task = _track_background_task(
                asyncio.create_task(_is.run_forever(app)),
                "impulse_scanner",
            )
            log.info("ImpulseScanner task started")
        except Exception as _is_err:
            log.error("ImpulseScanner failed to start: %s", _is_err)

    # ML: запускаем фоновый сборщик данных (каждые 15 минут обновляет ml_dataset.jsonl)
    # Важно: data_collector импортируется здесь чтобы не создавать циклических импортов
    if getattr(config, "BOT_ENABLE_DATA_COLLECTOR", False):
        try:
            import data_collector as _dc
            state.data_collector_task = _track_background_task(
                asyncio.create_task(_dc.run_forever(app)),
                "data_collector",
            )
            log.info("DataCollector task started")
        except Exception as _dc_err:
            log.error("DataCollector failed to start: %s", _dc_err)
    else:
        log.info("DataCollector disabled in Telegram bot process")

    # Phase 2: top gainer daily label collector (runs at 00:05 UTC daily)
    if getattr(config, "TOP_GAINER_DATASET_ENABLED", False):
        async def _top_gainer_label_collector() -> None:
            """Collect daily top gainer labels at 00:05 UTC."""
            import aiohttp
            from datetime import datetime, timezone
            while True:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next 00:05 UTC
                target_hour = getattr(config, "TOP_GAINER_LABEL_COLLECTION_HOUR_UTC", 0)
                target = now.replace(hour=target_hour, minute=5, second=0, microsecond=0)
                if now >= target:
                    from datetime import timedelta
                    target += timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                log.info("TopGainer label collector: next run in %.0f seconds (%s UTC)", wait_seconds, target)
                await asyncio.sleep(wait_seconds)
                try:
                    from backfill_top_gainer_dataset import collect_daily
                    async with aiohttp.ClientSession() as sess:
                        count = await collect_daily(sess)
                        log.info("TopGainer labels collected: %d records", count)
                except Exception as _tg_err:
                    log.error("TopGainer label collection failed: %s", _tg_err)

        _track_background_task(
            asyncio.create_task(_top_gainer_label_collector()),
            "top_gainer_label_collector",
        )
        log.info("TopGainer daily label collector scheduled")

    # Daily top-gainer performance report (runs at 22:00 Belgrade time)
    async def _rl_daily_report_task() -> None:
        """Send top-gainer performance report daily at 22:00 Europe/Belgrade."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        import json as _json
        from pathlib import Path as _Path
        _belgrade = ZoneInfo("Europe/Belgrade")
        _send = _make_broadcast_send(app)
        while True:
            now_bel = datetime.now(_belgrade)
            target = now_bel.replace(hour=22, minute=0, second=0, microsecond=0)
            if now_bel >= target:
                target += timedelta(days=1)
            wait_sec = (target - now_bel).total_seconds()
            log.info("RL daily report: next run in %.0f s (%s Belgrade)", wait_sec, target)
            await asyncio.sleep(wait_sec)
            try:
                from report_rl_daily import build_top_gainer_daily_report, render_top_gainer_text
                from datetime import date
                # Report on yesterday (EOD labels collected at 00:05 UTC)
                yesterday = date.today() - timedelta(days=1)
                report = build_top_gainer_daily_report(yesterday)
                text = render_top_gainer_text(report)
                await _send(text)
                log.info("Top-gainer daily report sent for %s", yesterday)
            except Exception as _rr_err:
                log.error("RL daily report failed: %s", _rr_err)

    _track_background_task(
        asyncio.create_task(_rl_daily_report_task()),
        "rl_daily_report",
    )
    log.info("RL daily report task scheduled")

    if state.positions and not state.running:
        try:
            _ensure_positions_monitored(state)
            state.task = asyncio.create_task(monitoring_loop(state, _make_broadcast_send(app)))
            state.running = True
            log.info(
                "Started monitoring restored positions immediately: %s position(s), %s hot coin(s)",
                len(state.positions),
                len(state.hot_coins),
            )
        except Exception as _restore_err:
            log.error("Immediate monitoring for restored positions failed: %s", _restore_err)

    if not _known_chat_ids:
        notify_service = False
    else:
        notify_service = bool(getattr(config, "SEND_SERVICE_NOTIFICATIONS", False))
    wl = config.load_watchlist()
    if notify_service:
        for cid in list(_known_chat_ids):
            try:
                await app.bot.send_message(
                    chat_id=cid,
                    text=(
                        f"🤖 *Бот перезапущен*\n\n"
                        f"Монет в списке: *{len(wl)}*\n"
                        f"🔍 Запускаю анализ автоматически..."
                    ),
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb_menu_reply(),
                )
            except Exception as e:
                log.warning("post_init notify failed for %s: %s", cid, e)

    if state.positions:
        log.info(
            "Skipping immediate startup market_scan because restored positions are already being monitored; "
            "auto-reanalyze first delay=%ss",
            int(getattr(config, "AUTO_REANALYZE_FIRST_DELAY_SEC", config.AUTO_REANALYZE_SEC)),
        )
    else:
        asyncio.create_task(_run_post_init_scan(app, notify_service, len(wl)))
    return

    try:
        from strategy import market_scan
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        if state.running:
            mon_line = f"▶️ Мониторинг запущен | Монет: {len(state.hot_coins)}"
        else:
            state.task    = asyncio.create_task(monitoring_loop(state, _make_broadcast_send(app)))
            state.running = True
            mon_line = f"▶️ Мониторинг запущен | Монет: {len(state.hot_coins)}"

        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        if notify_service:
            for cid in list(_known_chat_ids):
                try:
                    await app.bot.send_message(
                        chat_id=cid,
                        text=f"🔍 Анализ завершён — {len(wl)} монет  {now_str}\n{mon_line}",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_menu_reply(),
                    )
                except Exception as e:
                    log.warning("post_init result notify failed for %s: %s", cid, e)
    except Exception as e:
        log.error("post_init авто-анализ упал: %s", e)
        if notify_service:
            for cid in list(_known_chat_ids):
                try:
                    await app.bot.send_message(
                        chat_id=cid,
                        text="⚠️ Авто-анализ не удался. Нажмите *🔍 Анализ* вручную.",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_menu_reply(),
                    )
                except Exception:
                    pass


def main() -> None:
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "Токен не задан.\nСоздайте файл .env:\n"
            "TELEGRAM_BOT_TOKEN=ваш_токен"
        )
    ui_request = HTTPXRequest(
        connection_pool_size=UI_REQUEST_POOL_SIZE,
        connect_timeout=UI_CONNECT_TIMEOUT,
        write_timeout=UI_WRITE_TIMEOUT,
        read_timeout=UI_READ_TIMEOUT,
        pool_timeout=UI_POOL_TIMEOUT,
    )
    updates_request = HTTPXRequest(
        connection_pool_size=UPDATES_REQUEST_POOL_SIZE,
        connect_timeout=UPDATES_CONNECT_TIMEOUT,
        write_timeout=UPDATES_WRITE_TIMEOUT,
        read_timeout=UPDATES_READ_TIMEOUT,
        pool_timeout=UPDATES_POOL_TIMEOUT,
    )
    app = (
        Application.builder()
        .token(token)
        .request(ui_request)
        .get_updates_request(updates_request)
        .concurrent_updates(True)
        .post_init(_post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start, block=False))
    app.add_handler(CommandHandler("menu",  cmd_show_menu, block=False))
    app.add_handler(CommandHandler("hide",  cmd_hide_menu, block=False))
    app.add_handler(CommandHandler("why",   cmd_why))
    app.add_handler(CommandHandler("test",  cmd_test))
    app.add_handler(CallbackQueryHandler(btn, block=False))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler, block=False))

    # 19.04.2026: wire CMA-ES optimizer output (rl_params.json) into live config.
    # Was completely orphaned — 60 generations of optimization were not reaching
    # the trading loop. apply_optimized_params() reads rl_params.json and
    # patches matching attributes on `config` module via setattr.
    # OPT-IN via RL_PARAMS_APPLY_ON_STARTUP — default False, because applying
    # ~22 param overrides at once to a profitable bot (+146% / 7d) is risky;
    # CMA-ES reward != live PnL. Flip the flag after canary-validating offline.
    if bool(getattr(config, "RL_PARAMS_APPLY_ON_STARTUP", False)):
        try:
            from rl_agent import rl_agent as _rl_agent
            if _rl_agent.apply_optimized_params():
                log.info("RL: optimized params applied to config")
            else:
                log.info("RL: no optimized params to apply (rl_params.json missing/empty)")
        except Exception as exc:
            log.warning("RL: failed to apply optimized params: %s", exc)
    else:
        log.info("RL: rl_params.json apply DISABLED (set RL_PARAMS_APPLY_ON_STARTUP=True to enable)")

    log.info("Bot started.")
    log.info(
        "Starting polling with timeout=%s poll_interval=%s drop_pending_updates=%s",
        5,
        0.2,
        False,
    )
    app.run_polling(
        drop_pending_updates=False,
        poll_interval=0.2,
        timeout=5,
        bootstrap_retries=3,
    )


if __name__ == "__main__":
    main()
