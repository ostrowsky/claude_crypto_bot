"""Configuration of the demo TRADER -- a separate application next to the signal bot.

The trader never modifies the signal bot (files/, its config, its runtime, its process).
It READS three things of the bot, read-only:
  BOT_EVENTS   files/bot_events.jsonl         candidates and the bot's own entries/exits (E1 replay)
  HISTORY      history/<sym>_15m*.csv         klines for the replay grid and history checks
  ENV_FILE     files/.env                     BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET only
Everything it writes lives under trader/runtime/.

Signed requests go to Binance DEMO hosts only: exchange_client.ALLOWED_HOSTS is a code
constant, not a setting here, and there is no setting that allows mainnet.
"""
from __future__ import annotations

from pathlib import Path

TRADER_DIR = Path(__file__).resolve().parent
REPO = TRADER_DIR.parent
RUNTIME = TRADER_DIR / "runtime"

# read-only inputs from the signal bot
BOT_FILES = REPO / "files"
BOT_EVENTS = BOT_FILES / "bot_events.jsonl"
HISTORY = REPO / "history"
ENV_FILE = BOT_FILES / ".env"
POSITIONING = BOT_FILES / "positioning_history.jsonl"   # E8, read-only (rewritten by the bot's recorder)

# ── phase A (demo-connection-a-1010-spec.md) ──────────────────────────────────
TRADING_ENABLED: bool = True             # C2 ON 2026-10-20 (operator); rollback = False + restart executor
TRADING_MODE: str = "demo"               # "paper" | "demo"; mainnet is not a value (C1 paper until 2026-10-20)
TRADING_WEIGHT_BUDGET_SHARE: float = 0.5
TRADING_EQUITY_SNAPSHOT_SEC: int = 300
MICROSTRUCTURE_RECORDER_ENABLED: bool = True
MICRO_RECORD_INTERVAL_SEC: int = 300
MICRO_WEIGHT_BUDGET_SHARE: float = 0.10

# ── phase C executor (demo-execution-c-1013-spec.md) ── technical check with small sums.
# The executor mirrors the E7 book x4_floor8; a real order = paper size x TRADING_SCALE.
# Rollback: TRADING_ENABLED = False (stops stay on the exchange) + file FLATTEN in runtime/executor/control/.
TRADING_SCALE: float = 0.02                 # ~20 USDT per position
TRADING_MIN_ORDER_USDT: float = 6.0         # below max(this, minNotional x 1.2) -> skip below_min_notional
TRADING_MAX_EXPOSURE_USDT: float = 250.0    # open positions at entry cost
TRADING_DAILY_LOSS_LIMIT_USDT: float = 25.0 # realized loss of the UTC day -> no new entries until 00:00 UTC
TRADING_KILL_SWITCH_USDT: float = 60.0      # account equity below its peak by this -> flatten + PAUSE
TRADING_CATASTROPHE_GAP_PCT: float = 3.0    # exchange stop this far below the policy stop
TRADING_TRAIL_UPDATE_MIN_PCT: float = 0.2
TRADING_MAX_UNPROTECTED_SEC: int = 60
TRADING_POLL_SEC: int = 15
TRADING_PROTECT_RETRIES: int = 3
TRADING_RECONCILE_SEC: int = 300
TRADING_LOOP_SEC: int = 5
TRADING_EPOCH: str = "c1013a01"             # 8 chars, part of every clientOrderId

# ── E7 shadow portfolios (e7-shadow-portfolio-1012-spec.md) ── paper only, no key, no order.
# Rollback: False (and remove the scheduled step).
SHADOW_PORTFOLIO_ENABLED: bool = True

# ── FC_PASSIVE shadow book on live funding (fc-passive-next-1022-spec.md §A) ── paper only, public data,
# no key, no order; runs inside run_shadow.cmd after the E7 step. Rollback: False (the step does nothing).
FC_SHADOW_ENABLED: bool = True

# ── Audit journals (trader-audit-log-1012-spec.md): every Binance request and every decision,
# append-only, trader/runtime/audit/{exchange,decisions}/YYYY-MM-DD.jsonl. 0 = keep everything.
AUDIT_EXCHANGE_ENABLED: bool = True
AUDIT_DECISIONS_ENABLED: bool = True
AUDIT_RETENTION_DAYS: int = 0

# ── E1 replay (execution-replay-e1-1010-spec.md) ──────────────────────────────
# The replay must reproduce the signal bot, so it reads the bot's own parameters
# (ATR_PERIOD, MAX_OPEN_POSITIONS, TRAIL_MIN_BUFFER_PCT_*) from files/config.py -- by
# import, read-only, through bot_readonly(). Nothing is written back.


def bot_readonly():
    """Make the signal bot's modules importable for READ-ONLY use (appended, never first)."""
    import sys
    p = str(BOT_FILES)
    if p not in sys.path:
        sys.path.append(p)
