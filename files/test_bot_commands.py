"""The Telegram Menu button: its command list, and /positions.

Until 2026-09-01 the bot never called setMyCommands, so the blue Menu button was
empty and every command had to be typed from memory. `/positions` did not exist
at all — the portfolio was reachable only through an inline button.

Two ways this can rot silently, and one test each:

* a command is added to BOT_COMMANDS but no handler is registered, so the menu
  advertises something that does nothing;
* a handler is registered but left out of BOT_COMMANDS, so it stays invisible.

Both are drift between two lists in the same file, which is exactly the kind of
thing nobody notices until a user taps it.

Spec: none — UI wiring, no behaviour change to the trading path.
"""
from __future__ import annotations

import asyncio
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SRC = (HERE / "bot.py").read_text(encoding="utf-8")

EXPECTED = [
    ("start", "Открыть главное меню"),
    ("menu", "Открыть меню"),
    ("positions", "Показать открытые позиции"),
    ("why", "Объяснить последнее решение"),
    ("test", "Проверить состояние бота"),
]


def _declared_commands():
    """Parse BOT_COMMANDS out of the source without importing bot.py.

    bot.py starts background tasks and wants a Telegram token at import time, so
    the test reads the declaration instead of executing the module.
    """
    block = SRC.split("BOT_COMMANDS: tuple = (")[1].split(")\n", 1)[0]
    return re.findall(r'\("([a-z_]+)",\s*"([^"]+)"\)', block)


def _registered_commands():
    return re.findall(r'CommandHandler\("([a-z_]+)"', SRC)


class TestTheMenuListIsPublished(unittest.TestCase):
    def test_set_my_commands_is_actually_called(self):
        # The list existing is not the same as Telegram being told about it.
        self.assertIn("await app.bot.set_my_commands(", SRC)

    def test_it_is_called_on_startup(self):
        post_init = SRC.split("async def _post_init(")[1][:600]
        self.assertIn("_register_bot_commands(app)", post_init)

    def test_a_failure_cannot_block_startup(self):
        # An empty menu is cosmetic; a bot that will not start is not.
        body = SRC.split("async def _register_bot_commands(")[1].split("\nasync def ")[0]
        self.assertIn("except Exception", body)
        self.assertIn("log.warning", body)


class TestTheListMatchesTheHandlers(unittest.TestCase):
    """Drift between these two lists is the failure this file exists for."""

    def test_the_five_commands_are_declared_in_order(self):
        self.assertEqual(_declared_commands(), EXPECTED)

    def test_every_advertised_command_has_a_handler(self):
        registered = set(_registered_commands())
        for name, _ in _declared_commands():
            self.assertIn(name, registered,
                          "/%s is in the Menu button but has no handler" % name)

    def test_every_description_is_non_empty(self):
        for name, desc in _declared_commands():
            self.assertTrue(desc.strip(), "/%s has an empty description" % name)

    def test_hide_is_deliberately_not_advertised(self):
        # /hide exists as a handler but is an escape hatch, not something to put
        # in front of a user who opened the menu to see what the bot can do.
        self.assertIn("hide", _registered_commands())
        self.assertNotIn("hide", [c for c, _ in _declared_commands()])


class TestPositionsCommand(unittest.TestCase):
    def test_the_handler_exists_and_is_registered(self):
        self.assertIn("async def cmd_positions(", SRC)
        self.assertIn('CommandHandler("positions", cmd_positions', SRC)

    def test_it_reuses_the_same_renderer_as_the_button(self):
        # Two renderers would eventually disagree about what is in the portfolio.
        body = SRC.split("async def cmd_positions(")[1].split("\nasync def ")[0]
        self.assertIn("_positions_message_html()", body)

    def test_it_is_non_blocking_like_the_other_ui_commands(self):
        # The UI path has a send deadline and the loop already reports lag.
        self.assertIn('CommandHandler("positions", cmd_positions, block=False)', SRC)

    def test_it_tolerates_an_update_without_a_message(self):
        body = SRC.split("async def cmd_positions(")[1].split("\nasync def ")[0]
        self.assertIn("if msg is None:", body)


class TestPositionsRendererHandlesEmptyPortfolio(unittest.TestCase):
    """The renderer is shared, so an empty portfolio must not raise — that is
    the state the bot was in for most of 2026-08-20."""

    def test_empty_portfolio_returns_a_message_not_an_exception(self):
        body = SRC.split("def _positions_message_html(")[1].split("\ndef ")[0]
        self.assertIn("if not state.positions:", body)
        self.assertIn("Активных позиций нет", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
