"""Phase B2 -- entry candidates on the whole universe by the SIGNAL BOT's own rules (wide-universe-b-1014-spec.md §3).

The bot's code is imported read-only (TC.bot_readonly): indicators.compute_features and the strategy
checks check_entry_conditions / check_breakout_conditions / check_retest_conditions /
check_trend_surge_conditions / check_impulse_conditions / check_alignment_conditions /
check_ema_cross_conditions, and get_effective_entry_mode for the entry mode. Nothing is copied.
The mode is chosen in the order of monitor.py (~L5621-5694); trail_k comes from the bot's config.

Not reproduced (disclosed in the calibration report): the bot's ML gate, ranker, bandit, guards,
rotation and cooldowns; the catch-up entry on a recently "discovered" coin (state of the live
process); the impulse_speed -> trend downgrade while the mode is curtailed (state by day).
So the generator gives a SUPERSET of the bot's entries; selection is the trader policy's job.

Events have the bot's format, so the E1 replay and the executor read them unchanged:
{"event": "entry", "sym", "tf", "ts" (bar close, UTC ISO), "price" (= close of that bar), "mode", "trail_k", "source": "wide_gen"}.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trader_config as TC        # noqa: E402

BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}
SOURCE = "wide_gen"


def bot_rules():
    """The bot's own functions (read-only import) and its config module."""
    TC.bot_readonly()
    import config as BOT
    import indicators as IND
    import strategy as ST
    rules = {"features": IND.compute_features,
             "entry": lambda feat, i, c, tf: ST.check_entry_conditions(feat, i, c, tf=tf),
             "breakout": ST.check_breakout_conditions, "retest": ST.check_retest_conditions,
             "surge": ST.check_trend_surge_conditions, "impulse": ST.check_impulse_conditions,
             "alignment": lambda feat, i, tf: ST.check_alignment_conditions(feat, i, tf=tf),
             "ema_cross": ST.check_ema_cross_conditions,
             "entry_mode": lambda feat, i, c, tf: ST.get_effective_entry_mode(feat, i, c, tf=tf)}
    return rules, BOT


def warmup(cfg) -> int:
    """= strategy.analyze_coin: indicators are not used before this bar."""
    return max(cfg.EMA_SLOW + cfg.SLOPE_LOOKBACK + 2, cfg.ADX_PERIOD * 2 + cfg.ADX_GROW_BARS + 2, cfg.VOL_LOOKBACK + 2, 60)


def select_mode(flags, entry_mode, cfg):
    """-> (mode, trail_k) or None, in the order of monitor.py ~L5621-5694."""
    g = lambda k, d: getattr(cfg, k, d)  # noqa: E731
    if flags.get("breakout"):
        return "breakout", g("ATR_TRAIL_K_BREAKOUT", 1.5)
    if flags.get("retest"):
        return "retest", g("ATR_TRAIL_K_RETEST", 1.8)
    if flags.get("surge") and g("TREND_SURGE_PRECEDENCE_ENABLED", False):
        return "trend_surge", g("ATR_TRAIL_K_TREND_SURGE", g("ATR_TRAIL_K_STRONG", 2.5))
    if flags.get("entry"):
        mode = entry_mode or "trend"
        return mode, (g("ATR_TRAIL_K_STRONG", 2.5) if mode in ("strong_trend", "impulse_speed") else g("ATR_TRAIL_K", 2.0))
    if flags.get("surge"):
        return "impulse_speed", g("ATR_TRAIL_K_STRONG", 2.5)
    if flags.get("impulse"):
        return "impulse", g("ATR_TRAIL_K", 2.0)
    if flags.get("alignment"):
        return "alignment", g("ATR_TRAIL_K", 2.0)
    if flags.get("ema_cross") and g("EMA_CROSS_ENABLED", True):
        return "impulse_cross", g("ATR_TRAIL_K_CROSS", 2.5)
    return None


def _iso(ms) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def generate(sym, tf, arr, cfg=None, rules=None, now=None, start_ms=None, end_ms=None) -> list:
    """Candidates on closed bars i >= warmup of one symbol and timeframe. arr: {t (open ms), o, h, l, c, v}."""
    if rules is None or cfg is None:
        r_, c_ = bot_rules()
        rules, cfg = rules or r_, cfg or c_
    t, c = arr["t"], arr["c"]
    n = len(c)
    if n == 0:
        return []
    bar = BAR_MS[tf]
    now_ms = int(now.timestamp() * 1000) if now is not None else None
    feat = rules["features"](arr["o"], arr["h"], arr["l"], c, arr["v"])
    block = set(getattr(cfg, "ENTRY_BLOCK_HOURS", []) or [])
    cross_on = bool(getattr(cfg, "EMA_CROSS_ENABLED", True))
    cross_cd = int(getattr(cfg, "CROSS_COOLDOWN_BARS", 6))
    cross_until = -1
    out = []
    for i in range(warmup(cfg), n):
        close_ms = int(t[i]) + bar
        if now_ms is not None and close_ms > now_ms:
            break
        if start_ms is not None and close_ms < start_ms:
            continue
        if end_ms is not None and close_ms > end_ms:
            break
        if block and datetime.fromtimestamp(int(t[i]) / 1000, timezone.utc).hour in block:
            continue
        flags = {"entry": rules["entry"](feat, i, c, tf)[0], "breakout": rules["breakout"](feat, i)[0],
                 "retest": rules["retest"](feat, i)[0], "surge": rules["surge"](feat, i)[0],
                 "impulse": rules["impulse"](feat, i)[0], "alignment": rules["alignment"](feat, i, tf)[0], "ema_cross": False}
        if cross_on and i >= cross_until:
            flags["ema_cross"] = bool(rules["ema_cross"](feat, i)[0])
            if flags["ema_cross"]:
                cross_until = i + cross_cd
        if not any(flags.values()):
            continue
        em = rules["entry_mode"](feat, i, c, tf)[0] if flags["entry"] else None
        sel = select_mode(flags, em, cfg)
        if sel is None:
            continue
        out.append({"event": "entry", "sym": sym, "tf": tf, "ts": _iso(close_ms), "price": float(c[i]), "mode": sel[0],
                    "trail_k": float(sel[1]), "source": SOURCE, "flags": [k for k, v in flags.items() if v]})
    return out
