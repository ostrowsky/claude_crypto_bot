from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp


BINANCE_API = "https://api.binance.com"
WATCHLIST_FILE = Path("watchlist.json")
BOT_EVENTS_FILE = Path("bot_events.jsonl")
POSITIONS_FILE = Path("positions.json")


@dataclass
class SymbolDayStats:
    symbol: str
    last_price: float
    day_open: float
    day_change_pct: float
    high_price: float
    low_price: float
    price_change_24h_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top movers from watchlist for the current day and comparison with bot signals."
    )
    parser.add_argument("--top", type=int, default=15, help="How many top gainers to show")
    parser.add_argument(
        "--timezone",
        default="Europe/Budapest",
        help="IANA timezone used to calculate 'today'",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date in YYYY-MM-DD format in the local timezone. Defaults to today.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of a text report")
    return parser.parse_args()


def load_watchlist() -> list[str]:
    return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))


def load_portfolio_symbols() -> list[str]:
    if not POSITIONS_FILE.exists():
        return []
    try:
        payload = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, dict):
        return list(payload.keys())
    return []


def parse_local_day(day_str: str | None, tz: ZoneInfo) -> tuple[datetime, datetime]:
    if day_str:
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
    else:
        day = datetime.now(tz).date()
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = datetime.now(tz) if day == datetime.now(tz).date() else datetime.combine(day, time.max, tzinfo=tz)
    return start_local, end_local


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict[str, Any] | None = None) -> Any:
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_tickers(session: aiohttp.ClientSession, symbols: list[str]) -> dict[str, dict[str, Any]]:
    data = await fetch_json(session, f"{BINANCE_API}/api/v3/ticker/24hr", {"type": "FULL"})
    ticker_map = {item["symbol"]: item for item in data}
    return {sym: ticker_map[sym] for sym in symbols if sym in ticker_map}


async def fetch_day_open(session: aiohttp.ClientSession, symbol: str, start_ms: int) -> tuple[str, float | None]:
    try:
        data = await fetch_json(
            session,
            f"{BINANCE_API}/api/v3/klines",
            {"symbol": symbol, "interval": "1h", "startTime": start_ms, "limit": 1},
        )
    except aiohttp.ClientResponseError:
        return symbol, None
    if not data:
        return symbol, None
    return symbol, float(data[0][1])


async def fetch_day_stats(symbols: list[str], start_local: datetime) -> list[SymbolDayStats]:
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tickers = await fetch_tickers(session, symbols)
        opens = await asyncio.gather(*(fetch_day_open(session, sym, start_ms) for sym in symbols))
    day_opens = dict(opens)
    stats: list[SymbolDayStats] = []
    for sym in symbols:
        ticker = tickers.get(sym)
        day_open = day_opens.get(sym)
        if not ticker or not day_open or day_open <= 0:
            continue
        last_price = float(ticker["lastPrice"])
        stats.append(
            SymbolDayStats(
                symbol=sym,
                last_price=last_price,
                day_open=day_open,
                day_change_pct=(last_price / day_open - 1.0) * 100.0,
                high_price=float(ticker["highPrice"]),
                low_price=float(ticker["lowPrice"]),
                price_change_24h_pct=float(ticker["priceChangePercent"]),
            )
        )
    stats.sort(key=lambda x: x.day_change_pct, reverse=True)
    return stats


def load_today_events(start_local: datetime, end_local: datetime, tz: ZoneInfo) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "entries": [],
            "exits": [],
            "blocked": [],
            "cooldowns": [],
            "forwards": [],
        }
    )
    with BOT_EVENTS_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_raw = obj.get("ts")
            sym = obj.get("sym")
            if not ts_raw or not sym:
                continue
            try:
                ts_utc = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            ts_local = ts_utc.astimezone(tz)
            if not (start_local <= ts_local <= end_local):
                continue
            event = obj.get("event", "")
            obj["_ts_local"] = ts_local.strftime("%H:%M")
            if event == "cooldown_start":
                rows[sym]["cooldowns"].append(obj)
            elif event == "forward":
                rows[sym]["forwards"].append(obj)
            elif event == "entry":
                rows[sym]["entries"].append(obj)
            elif event == "exit":
                rows[sym]["exits"].append(obj)
            elif event == "blocked":
                rows[sym]["blocked"].append(obj)
    return rows


def summarize_symbol(stats: SymbolDayStats, events: dict[str, Any]) -> dict[str, Any]:
    entries = events.get("entries", [])
    exits = events.get("exits", [])
    blocked = events.get("blocked", [])
    blocked_reasons = Counter(item.get("reason", "") for item in blocked)
    top_reason = blocked_reasons.most_common(1)[0][0] if blocked_reasons else None
    latest_exit = exits[-1] if exits else None
    first_entry = entries[0] if entries else None
    latest_entry = entries[-1] if entries else None
    return {
        "symbol": stats.symbol,
        "last_price": stats.last_price,
        "day_open": stats.day_open,
        "day_change_pct": stats.day_change_pct,
        "price_change_24h_pct": stats.price_change_24h_pct,
        "entries_count": len(entries),
        "exits_count": len(exits),
        "blocked_count": len(blocked),
        "first_entry_time": first_entry.get("_ts_local") if first_entry else None,
        "first_entry_mode": first_entry.get("mode") if first_entry else None,
        "last_entry_time": latest_entry.get("_ts_local") if latest_entry else None,
        "latest_exit_time": latest_exit.get("_ts_local") if latest_exit else None,
        "latest_exit_pnl_pct": latest_exit.get("pnl_pct") if latest_exit else None,
        "top_block_reason": top_reason,
        "status": (
            "bought"
            if entries
            else "blocked"
            if blocked
            else "no_signal"
        ),
    }


def build_portfolio_audit(summaries: list[dict[str, Any]], portfolio_symbols: list[str]) -> dict[str, Any]:
    top_symbols = [item["symbol"] for item in summaries]
    top_set = set(top_symbols)
    portfolio_set = set(portfolio_symbols)
    captured = [sym for sym in top_symbols if sym in portfolio_set]
    missed = [sym for sym in top_symbols if sym not in portfolio_set]
    extra = [sym for sym in portfolio_symbols if sym not in top_set]
    capture_rate = (len(captured) / len(top_symbols) * 100.0) if top_symbols else 0.0
    return {
        "portfolio_symbols": portfolio_symbols,
        "top_symbols": top_symbols,
        "captured_symbols": captured,
        "missed_symbols": missed,
        "extra_symbols": extra,
        "captured_count": len(captured),
        "missed_count": len(missed),
        "extra_count": len(extra),
        "capture_rate_pct": capture_rate,
    }


def format_report(
    summaries: list[dict[str, Any]],
    portfolio_audit: dict[str, Any],
    start_local: datetime,
    end_local: datetime,
    watchlist_size: int,
    top_n: int,
) -> str:
    lines: list[str] = []
    lines.append(f"Top movers from watchlist for {start_local.strftime('%Y-%m-%d')} ({start_local.tzinfo})")
    lines.append(f"Window: {start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')}")
    lines.append(f"Watchlist size: {watchlist_size}")
    lines.append("")
    lines.append(f"Top {top_n} gainers and bot reaction:")
    for i, item in enumerate(summaries, start=1):
        line = (
            f"{i}. {item['symbol']}  today {item['day_change_pct']:+.2f}%"
            f"  price {item['last_price']:.8g}"
            f"  24h {item['price_change_24h_pct']:+.2f}%"
            f"  status={item['status']}"
        )
        lines.append(line)
        if item["entries_count"]:
            lines.append(
                f"   BUY: {item['entries_count']}  first={item['first_entry_time']} mode={item['first_entry_mode']}"
            )
        if item["exits_count"]:
            pnl = item["latest_exit_pnl_pct"]
            pnl_text = f"{float(pnl):+.2f}%" if pnl is not None else "n/a"
            lines.append(f"   SELL: {item['exits_count']}  last={item['latest_exit_time']} pnl={pnl_text}")
        if item["blocked_count"] and item["top_block_reason"]:
            lines.append(f"   BLOCK: {item['blocked_count']}  main={item['top_block_reason']}")
    missed = [x for x in summaries if x["status"] != "bought"]
    if missed:
        lines.append("")
        lines.append("Missed / not bought movers:")
        for item in missed:
            why = item["top_block_reason"] or "no fresh signal_now"
            lines.append(f"- {item['symbol']} {item['day_change_pct']:+.2f}% -> {item['status']} ({why})")
    lines.append("")
    lines.append("Portfolio vs top movers:")
    lines.append(
        f"- capture rate: {portfolio_audit['captured_count']}/{len(summaries)}"
        f" ({portfolio_audit['capture_rate_pct']:.1f}%)"
    )
    lines.append(f"- captured: {', '.join(portfolio_audit['captured_symbols']) or '-'}")
    lines.append(f"- missed: {', '.join(portfolio_audit['missed_symbols']) or '-'}")
    lines.append(f"- extra in portfolio: {', '.join(portfolio_audit['extra_symbols']) or '-'}")
    return "\n".join(lines)


async def main() -> int:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    start_local, end_local = parse_local_day(args.date, tz)
    watchlist = load_watchlist()
    portfolio_symbols = load_portfolio_symbols()
    stats = await fetch_day_stats(watchlist, start_local)
    top_stats = stats[: args.top]
    event_rows = load_today_events(start_local, end_local, tz)
    summaries = []
    for item in top_stats:
        summary = summarize_symbol(item, event_rows.get(item.symbol, {}))
        summary["in_portfolio"] = item.symbol in portfolio_symbols
        summaries.append(summary)
    portfolio_audit = build_portfolio_audit(summaries, portfolio_symbols)
    if args.json:
        print(
            json.dumps(
                {
                    "date": start_local.strftime("%Y-%m-%d"),
                    "timezone": str(tz),
                    "watchlist_size": len(watchlist),
                    "portfolio_audit": portfolio_audit,
                    "top": summaries,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report(summaries, portfolio_audit, start_local, end_local, len(watchlist), args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
