# Telegram command menu, and /positions

- **Slug:** `telegram-command-menu`
- **Status:** DEPLOYED 2026-09-01 (verified live: `setMyCommands` → 200)
- **Truth-harness invariants:** TH-10 (verified live, not assumed), TH-12
  (evidence with the change), TH-13 (the live check is quoted, not described)
- **Flag:** none — UI wiring
- **Rollback:** revert this commit. Nothing in the trading path is touched.

## What was missing

Telegram fills the blue **Menu** button from `setMyCommands`. The bot never
called it, so the button was empty: every command had to be typed from memory by
someone who already knew it existed.

`/positions` did not exist at all. The portfolio was reachable only by opening the
main panel and tapping an inline button — two steps and a panel refresh to answer
"what am I holding", which is the question asked most often during a rally.

## What it does now

Published on startup from one list, `BOT_COMMANDS` in `bot.py`:

| command | description |
|---|---|
| `/start` | Открыть главное меню |
| `/menu` | Открыть меню |
| `/positions` | Показать открытые позиции |
| `/why` | Объяснить последнее решение |
| `/test` | Проверить состояние бота |

`/hide` remains a registered handler and is **deliberately not advertised**: it is
an escape hatch, not something to offer someone who opened the menu to find out
what the bot can do. A test pins that asymmetry so it reads as a decision rather
than an oversight.

## Design points that are not obvious

**One renderer, two entry points.** `cmd_positions` calls
`_positions_message_html()` — the same function the inline button uses. A second
renderer would eventually disagree with the first about what the portfolio
contains, and the disagreement would surface during exactly the moment it matters.

**Non-blocking, like the other UI commands.** Registered with `block=False`. The
render path has a send deadline and the event loop already reports 2–4s lag under
dataset label writes; a blocking handler there stalls the whole UI.

**Publication cannot break startup.** `_register_bot_commands` catches and logs.
An empty Menu button is a cosmetic loss; a bot that refuses to start is not.

## No backtest, and why

Nothing in the trading path changes: no gate, threshold, model, entry, exit or
sizing. The only new decision the bot makes is which five strings to send to
Telegram once at startup. The claim that needed evidence — that the menu is
actually published — is verified directly below rather than argued.

## Verification

`test_bot_commands.py`, 12 tests. The two that carry the weight guard **drift
between two lists in the same file**, which is what nobody notices until a user
taps a command that does nothing:

- every command in `BOT_COMMANDS` has a registered `CommandHandler`;
- the declared list matches the intended five exactly, in order.

The tests parse `BOT_COMMANDS` out of the source instead of importing `bot.py`,
which wants a Telegram token and starts background tasks at import time.

Live check after restart, quoted rather than summarised:

```
HTTP Request: POST https://api.telegram.org/bot.../setMyCommands "HTTP/1.1 200 OK"
bot commands published to the Menu button: /start, /menu, /positions, /why, /test
```

## Adjacent finding, not fixed here

`httpx` logs every Telegram request URL at INFO, and the bot token is part of that
URL — so `bot_stderr.log` contains the token in plaintext on every poll, roughly
every five seconds. The file is gitignored and local, so this is not a leak to
anywhere, but it is a credential sitting in a rotating log. Fixing it means
raising the `httpx` logger level or redacting the URL, and it is left out of this
change deliberately so a UI commit does not quietly alter logging. Per CLAUDE.md
§13 the decision to rotate is the operator's.
