"""Is this symbol still trading, or does it only look like it?

`/api/v3/ticker/24hr` keeps returning a row for delisted pairs, with a non-zero
quote volume. The snapshot builds features from that ticker, so a dead pair
arrives with a complete, plausible feature row — `EOSUSDT` carried
`tg_return_since_open = 6.79` while its last candle was May 2025, and the early
ranking put it first.

The test is the immutable label store, not the ticker: the store is built from
klines, so a symbol that has not printed a candle cannot have a recent label.
No extra network call, and nothing a phantom ticker can fake.

Stdlib plus the label store. Nothing here decides anything; it filters.

Spec: docs/specs/features/phantom-symbol-filter-spec.md
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from label_store import LabelStore  # noqa: E402

DEFAULT_MAX_AGE_DAYS = 14

_NEWEST: dict | None = None


def _newest_labels(store: LabelStore | None = None) -> dict:
    """symbol -> newest COMPLETE label day. Cached for the real store only."""
    global _NEWEST
    if store is not None:
        newest: dict = {}
        for r in store.records():
            if not r.get("complete"):
                continue
            day = r["utc_day"]
            if day > newest.get(r["symbol"], ""):
                newest[r["symbol"]] = day
        return newest
    if _NEWEST is None:
        _NEWEST = _newest_labels(LabelStore())
    return _NEWEST


def liveness(symbol: str, *, store: LabelStore | None = None,
             max_age_days: int = DEFAULT_MAX_AGE_DAYS,
             today: str | None = None) -> str:
    """`"live"`, `"stale"` or `"unknown"`.

    The three are different facts and are kept apart on purpose: `stale` is a
    delisted or renamed pair still answering on the ticker endpoint, `unknown`
    is a symbol the store has never seen — a fresh listing, say. Both fail
    `is_live`, but reporting them as one number would hide which problem is
    growing.
    """
    newest = _newest_labels(store).get(symbol)
    if not newest:
        return "unknown"
    cutoff = ((datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if today else datetime.now(timezone.utc))
              - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    return "live" if newest >= cutoff else "stale"


def is_live(symbol: str, *, store: LabelStore | None = None,
            max_age_days: int = DEFAULT_MAX_AGE_DAYS,
            today: str | None = None) -> bool:
    return liveness(symbol, store=store, max_age_days=max_age_days,
                    today=today) == "live"


MAX_DROP_FRACTION = 0.25


def filter_live(symbols, *, store: LabelStore | None = None,
                max_age_days: int = DEFAULT_MAX_AGE_DAYS,
                enabled: bool = True,
                today: str | None = None,
                max_drop_fraction: float = MAX_DROP_FRACTION) -> tuple[list, dict]:
    """`(kept, dropped)` where `dropped` splits `stale` from `unknown`.

    What was dropped travels with the result rather than being logged and
    forgotten: a filter that silently shrinks its input is how a coverage number
    ends up meaning something other than it says.
    """
    dropped: dict[str, list] = {"stale": [], "unknown": []}
    if not enabled:
        return list(symbols), dropped
    symbols = list(symbols)
    kept = []
    for s in symbols:
        state = liveness(s, store=store, max_age_days=max_age_days, today=today)
        if state == "live":
            kept.append(s)
        else:
            dropped[state].append(s)

    # Fail OPEN, not closed. The store is rebuilt daily; if that build lags or
    # breaks, every symbol goes stale at once and this filter would silently
    # empty the universe — a data-pipeline outage rendered as "nothing is
    # trading today". Losing a quarter of the watchlist overnight is a fault in
    # the store, not a market event, so the filter stands down and says so.
    if symbols and (len(symbols) - len(kept)) / len(symbols) > max_drop_fraction:
        dropped["stood_down"] = True
        dropped["would_have_dropped"] = (len(symbols) - len(kept))
        return symbols, dropped
    return kept, dropped


def row_is_phantom(record: dict, *, store: LabelStore | None = None,
                   max_age_days: int | None = None) -> bool:
    """Was this dataset row written for an instrument that had already stopped
    trading on the day it describes?

    Judged AS OF the row's own day, never with today's knowledge: a symbol that
    was live in March and delisted in June produced good rows in March, and
    deleting those retroactively would be a different (and wrong) claim.

    4 973 of 125 008 rows (3.98%) across 13 symbols are phantoms, and the share
    is growing — 1 563 in July alone — because a delisted pair keeps answering
    the ticker endpoint that feeds the snapshot.
    """
    symbol = record.get("symbol")
    if not symbol:
        return False
    ts = record.get("ts") or 0
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return False
    if ts > 1e11:
        ts /= 1000.0
    day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    return liveness(symbol, store=store,
                    max_age_days=max_age_days or DEFAULT_MAX_AGE_DAYS,
                    today=day) == "stale"


def drop_phantom_rows(records, *, store: LabelStore | None = None) -> tuple[list, int]:
    """`(kept, dropped)`. The file is NOT rewritten.

    Rewriting a 147 MB journal in place is the operation that froze the event
    loop for 130-300s in August and then failed with PermissionError while the
    bot held the file open. Filtering on read costs one predicate per row and
    leaves the journal intact.
    """
    if not enabled():
        return list(records), 0
    kept = [r for r in records if not row_is_phantom(r, store=store)]
    return kept, len(records) - len(kept)


def _config_flag(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:                       # a filter must not need config
        return default


def enabled() -> bool:
    return bool(_config_flag("PHANTOM_SYMBOL_FILTER_ENABLED", True))


def max_age_days() -> int:
    return int(_config_flag("PHANTOM_MAX_LABEL_AGE_DAYS", DEFAULT_MAX_AGE_DAYS))


if __name__ == "__main__":
    import io
    import json
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    wl = json.loads((HERE / "watchlist.json").read_text(encoding="utf-8"))
    kept, dropped = filter_live(wl)
    print(f"watchlist {len(wl)}  live {len(kept)}  "
          f"stale {len(dropped['stale'])}  unknown {len(dropped['unknown'])}")
    for state in ("stale", "unknown"):
        for s in dropped[state]:
            newest = _newest_labels().get(s, "never")
            print(f"  {state:<8}{s:<14}newest label {newest}")
