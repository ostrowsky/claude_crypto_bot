from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import aiohttp

import critic_dataset


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"
HISTORY_FILE = REPORT_DIR / "top_gainer_critic_history.jsonl"

BINANCE_API = "https://api.binance.com"
WATCHLIST_FILE = ROOT / "watchlist.json"
BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"

DEFAULT_TOP_N = 15
DEFAULT_MIN_QUOTE_VOLUME = 1_000_000.0
DEFAULT_TZ = "Europe/Budapest"

_STABLE_OR_NON_TARGET = {
    "USDCUSDT",
    "BUSDUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "DAIUSDT",
    "USDPUSDT",
    "USDSUSDT",
    "EURUSDT",
    "GBPUSDT",
    "TRYUSDT",
    "BRLUSDT",
}


@dataclass
class DayPerformance:
    symbol: str
    day_open: float
    day_close: float
    day_high: float
    day_low: float
    day_change_pct: float
    quote_volume_24h: float
    in_watchlist: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Critic report: compare bot BUY decisions with top gainers of the day or midday window."
    )
    parser.add_argument(
        "--phase",
        choices=("midday", "final"),
        required=True,
        help="midday = 00:00-12:00 local window, final = 00:00-24:00 completed local day",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Local date YYYY-MM-DD. For final reports this is the audited local day.",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-quote-volume", type=float, default=DEFAULT_MIN_QUOTE_VOLUME)
    parser.add_argument("--timezone", default=DEFAULT_TZ)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _load_watchlist() -> set[str]:
    try:
        payload = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {str(x) for x in payload if isinstance(x, str)}


def _local_window(target_day: date, phase: str, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, time.min, tzinfo=tz)
    if phase == "midday":
        end_local = datetime.combine(target_day, time(hour=12), tzinfo=tz)
    else:
        end_local = datetime.combine(target_day, time.max, tzinfo=tz)
    return start_local, end_local


def _as_utc_ms(dt_local: datetime) -> int:
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)


def _is_target_symbol(symbol: str) -> bool:
    if not symbol.endswith("USDT"):
        return False
    if symbol in _STABLE_OR_NON_TARGET:
        return False
    for suffix in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"):
        if symbol.endswith(suffix):
            return False
    return True


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=30)
    async with session.get(url, params=params, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_exchange_symbols(session: aiohttp.ClientSession) -> list[str]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/exchangeInfo")
    out: list[str] = []
    for item in payload.get("symbols", []):
        sym = str(item.get("symbol", ""))
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if not bool(item.get("isSpotTradingAllowed", True)):
            continue
        if _is_target_symbol(sym):
            out.append(sym)
    return out


async def _fetch_all_tickers(session: aiohttp.ClientSession) -> dict[str, dict[str, Any]]:
    payload = await _fetch_json(session, f"{BINANCE_API}/api/v3/ticker/24hr", {"type": "FULL"})
    return {
        str(item.get("symbol", "")): item
        for item in payload
        if isinstance(item, dict)
    }


async def _fetch_symbol_window(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    in_watchlist: bool,
    quote_volume_24h: float,
) -> DayPerformance | None:
    payload = await _fetch_json(
        session,
        f"{BINANCE_API}/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "1h",
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": 32,
        },
    )
    if not payload:
        return None
    opens = [float(row[1]) for row in payload]
    highs = [float(row[2]) for row in payload]
    lows = [float(row[3]) for row in payload]
    closes = [float(row[4]) for row in payload]
    day_open = float(opens[0])
    day_close = float(closes[-1])
    if day_open <= 0:
        return None
    return DayPerformance(
        symbol=symbol,
        day_open=day_open,
        day_close=day_close,
        day_high=max(highs),
        day_low=min(lows),
        day_change_pct=(day_close / day_open - 1.0) * 100.0,
        quote_volume_24h=float(quote_volume_24h),
        in_watchlist=bool(in_watchlist),
    )


async def fetch_day_performance(
    *,
    target_day: date,
    phase: str,
    timezone_name: str,
    min_quote_volume: float,
) -> list[DayPerformance]:
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _local_window(target_day, phase, tz)
    start_ms = _as_utc_ms(start_local)
    end_ms = _as_utc_ms(end_local)
    watchlist = _load_watchlist()

    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        exchange_symbols = await _fetch_exchange_symbols(session)
        ticker_map = await _fetch_all_tickers(session)

        sem = asyncio.Semaphore(20)

        async def _one(symbol: str) -> DayPerformance | None:
            ticker = ticker_map.get(symbol)
            if not ticker:
                return None
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            if quote_volume < min_quote_volume:
                return None
            async with sem:
                try:
                    return await _fetch_symbol_window(
                        session,
                        symbol,
                        start_ms,
                        end_ms,
                        in_watchlist=symbol in watchlist,
                        quote_volume_24h=quote_volume,
                    )
                except Exception:
                    return None

        results = await asyncio.gather(*(_one(sym) for sym in exchange_symbols))

    perf = [item for item in results if item is not None]
    perf.sort(key=lambda x: x.day_change_pct, reverse=True)
    return perf


def _load_day_events(start_local: datetime, end_local: datetime, tz: ZoneInfo) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in _iter_jsonl(BOT_EVENTS_FILE):
        sym = str(rec.get("sym", ""))
        if not sym:
            continue
        ts = _parse_utc_ts(rec.get("ts"))
        if not ts:
            continue
        ts_local = ts.astimezone(tz)
        if not (start_local <= ts_local <= end_local):
            continue
        bucket = rows.setdefault(
            sym,
            {"entries": [], "exits": [], "blocked": [], "forwards": []},
        )
        rec = dict(rec)
        rec["_ts_local"] = ts_local.strftime("%H:%M")
        event = str(rec.get("event", ""))
        if event == "entry":
            bucket["entries"].append(rec)
        elif event == "exit":
            bucket["exits"].append(rec)
        elif event == "blocked":
            bucket["blocked"].append(rec)
        elif event == "forward":
            bucket["forwards"].append(rec)
    return rows


def _capture_ratio(day_open: float, day_close: float, entry_price: float) -> float | None:
    move = day_close - day_open
    if move <= 0:
        return None
    ratio = (day_close - entry_price) / move
    return max(0.0, min(1.5, ratio))


def _status_for_symbol(in_watchlist: bool, events: dict[str, list[dict[str, Any]]]) -> tuple[str, str | None]:
    if not in_watchlist:
        return "not_in_watchlist", None
    entries = events.get("entries", [])
    blocked = events.get("blocked", [])
    if entries:
        return "bought", None
    if blocked:
        reasons = Counter(str(x.get("reason", "")) for x in blocked)
        top_reason = reasons.most_common(1)[0][0] if reasons else None
        if top_reason and "портфель полон" in top_reason:
            return "blocked_portfolio", top_reason
        return "blocked_rule", top_reason
    return "no_signal", None


def summarize_top_gainer(
    perf: DayPerformance,
    events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    entries = sorted(events.get("entries", []), key=lambda x: x.get("ts", ""))
    exits = sorted(events.get("exits", []), key=lambda x: x.get("ts", ""))
    first_entry = entries[0] if entries else None
    last_exit = exits[-1] if exits else None
    status, reason = _status_for_symbol(perf.in_watchlist, events)

    first_entry_price = float(first_entry.get("price", 0.0)) if first_entry else None
    capture_ratio = (
        _capture_ratio(perf.day_open, perf.day_close, first_entry_price)
        if first_entry_price
        else None
    )
    opportunity_from_entry = (
        (perf.day_close / first_entry_price - 1.0) * 100.0
        if first_entry_price and first_entry_price > 0
        else None
    )

    summary = {
        "symbol": perf.symbol,
        "day_change_pct": round(perf.day_change_pct, 3),
        "day_open": perf.day_open,
        "day_close": perf.day_close,
        "quote_volume_24h": round(perf.quote_volume_24h, 2),
        "in_watchlist": perf.in_watchlist,
        "status": status,
        "reason": reason,
        "entries_count": len(entries),
        "blocked_count": len(events.get("blocked", [])),
        "first_entry_time": first_entry.get("_ts_local") if first_entry else None,
        "first_entry_mode": first_entry.get("mode") if first_entry else None,
        "first_entry_price": first_entry_price,
        "capture_ratio": None if capture_ratio is None else round(capture_ratio, 4),
        "opportunity_from_entry_pct": None if opportunity_from_entry is None else round(opportunity_from_entry, 3),
        "latest_exit_time": last_exit.get("_ts_local") if last_exit else None,
        "latest_exit_pnl_pct": last_exit.get("pnl_pct") if last_exit else None,
        "latest_exit_reason": last_exit.get("reason") if last_exit else None,
    }
    return summary


def build_report(
    *,
    target_day: date,
    phase: str,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = DEFAULT_TOP_N,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
    day_performance: Optional[list[DayPerformance]] = None,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _local_window(target_day, phase, tz)
    if day_performance is None:
        day_performance = asyncio.run(
            fetch_day_performance(
                target_day=target_day,
                phase=phase,
                timezone_name=timezone_name,
                min_quote_volume=min_quote_volume,
            )
        )

    all_top = day_performance[:top_n]
    watchlist_top = [x for x in day_performance if x.in_watchlist][:top_n]
    event_rows = _load_day_events(start_local, end_local, tz)

    all_top_summary = [summarize_top_gainer(item, event_rows.get(item.symbol, {})) for item in all_top]
    watchlist_top_summary = [summarize_top_gainer(item, event_rows.get(item.symbol, {})) for item in watchlist_top]

    watchlist_top_set = {item["symbol"] for item in watchlist_top_summary}
    bought_symbols = {
        str(rec.get("sym"))
        for rec in _iter_jsonl(BOT_EVENTS_FILE)
        if rec.get("event") == "entry"
        and (ts := _parse_utc_ts(rec.get("ts")))
        and start_local <= ts.astimezone(tz) <= end_local
    }
    false_positive_symbols = sorted(sym for sym in bought_symbols if sym not in watchlist_top_set)

    bought_top = [x for x in watchlist_top_summary if x["status"] == "bought"]
    missed_top = [x for x in watchlist_top_summary if x["status"] != "bought"]
    early_captured = [
        x for x in bought_top
        if (x.get("capture_ratio") or 0.0) >= 0.35
    ]
    coverage = [x for x in all_top_summary if x["in_watchlist"]]

    report = {
        "target_day_local": target_day.isoformat(),
        "phase": phase,
        "window_local": {
            "start": start_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "end": end_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
        "settings": {
            "timezone": timezone_name,
            "top_n": top_n,
            "min_quote_volume_24h": min_quote_volume,
            "early_capture_ratio_min": 0.35,
        },
        "summary": {
            "exchange_top_count": len(all_top_summary),
            "exchange_top_in_watchlist": len(coverage),
            "watchlist_top_count": len(watchlist_top_summary),
            "watchlist_top_bought": len(bought_top),
            "watchlist_top_early_captured": len(early_captured),
            "watchlist_top_missed": len(missed_top),
            "watchlist_top_capture_rate_pct": round((len(bought_top) / len(watchlist_top_summary) * 100.0), 2) if watchlist_top_summary else 0.0,
            "watchlist_top_early_capture_rate_pct": round((len(early_captured) / len(watchlist_top_summary) * 100.0), 2) if watchlist_top_summary else 0.0,
            "bot_unique_buys": len(bought_symbols),
            "bot_false_positive_buys": len(false_positive_symbols),
        },
        "exchange_top_gainers": all_top_summary,
        "watchlist_top_gainers": watchlist_top_summary,
        "bot_false_positive_symbols": false_positive_symbols[:top_n],
    }
    return report


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Top-gainer critic for {report['target_day_local']} ({report['phase']})",
        f"Window: {report['window_local']['start']} -> {report['window_local']['end']}",
        "",
        "Summary:",
        f"  exchange top in watchlist: {summary['exchange_top_in_watchlist']}/{summary['exchange_top_count']}",
        f"  watchlist top bought: {summary['watchlist_top_bought']}/{summary['watchlist_top_count']} ({summary['watchlist_top_capture_rate_pct']:.1f}%)",
        f"  early captures: {summary['watchlist_top_early_captured']}/{summary['watchlist_top_count']} ({summary['watchlist_top_early_capture_rate_pct']:.1f}%)",
        f"  bot false-positive buys: {summary['bot_false_positive_buys']}/{summary['bot_unique_buys']}",
        "",
        "Watchlist top gainers and bot reaction:",
    ]
    for idx, item in enumerate(report["watchlist_top_gainers"], start=1):
        lines.append(
            f"{idx}. {item['symbol']} {item['day_change_pct']:+.2f}% status={item['status']}"
        )
        if item["first_entry_time"]:
            capture = "n/a" if item["capture_ratio"] is None else f"{float(item['capture_ratio']) * 100:.1f}%"
            lines.append(
                f"   BUY {item['first_entry_time']} mode={item['first_entry_mode']} entry={item['first_entry_price']} capture={capture}"
            )
        if item["latest_exit_pnl_pct"] is not None:
            lines.append(
                f"   EXIT {item['latest_exit_time']} pnl={float(item['latest_exit_pnl_pct']):+.2f}%"
            )
        if item["reason"]:
            lines.append(f"   WHY {item['reason']}")

    if report["bot_false_positive_symbols"]:
        lines.append("")
        lines.append("Bot bought, but symbol did not finish in watchlist top gainers:")
        lines.append("  " + ", ".join(report["bot_false_positive_symbols"]))
    return "\n".join(lines)


def save_report(report: dict[str, Any]) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"top_gainer_critic_{report['target_day_local']}_{report['phase']}"
    json_path = REPORT_DIR / f"{base}.json"
    txt_path = REPORT_DIR / f"{base}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(report), encoding="utf-8")
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        history_row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_day_local": report["target_day_local"],
            "phase": report["phase"],
            "summary": report["summary"],
            "files": {"json": str(json_path), "txt": str(txt_path)},
        }
        fh.write(json.dumps(history_row, ensure_ascii=False) + "\n")
    return {"json": str(json_path), "txt": str(txt_path)}


def run_report(
    *,
    target_day: date,
    phase: str,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = DEFAULT_TOP_N,
    min_quote_volume: float = DEFAULT_MIN_QUOTE_VOLUME,
) -> dict[str, Any]:
    report = build_report(
        target_day=target_day,
        phase=phase,
        timezone_name=timezone_name,
        top_n=top_n,
        min_quote_volume=min_quote_volume,
    )
    report["teacher_annotation"] = critic_dataset.annotate_top_gainer_teacher(report)
    files = save_report(report)
    report["files"] = files
    return report


def main() -> int:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    if args.date:
        target_day = date.fromisoformat(args.date)
    else:
        today = datetime.now(tz).date()
        target_day = today if args.phase == "midday" else date.fromordinal(today.toordinal() - 1)
    report = run_report(
        target_day=target_day,
        phase=args.phase,
        timezone_name=args.timezone,
        top_n=args.top,
        min_quote_volume=args.min_quote_volume,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"JSON report saved to: {report['files']['json']}")
        print(f"Text report saved to: {report['files']['txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
