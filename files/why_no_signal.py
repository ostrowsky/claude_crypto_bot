"""Answer "почему нет сигнала по X?" from the logs, in one command.

    pyembed\\python.exe files\\why_no_signal.py POLUSDT --days 7
    pyembed\\python.exe files\\why_no_signal.py C98USDT BTCUSDT --hours 24

This question came up four times in one session — POL, C98, BTC, ATOM — and each
answer took a manual dig through three datasets. POL turned out to have gone 15
days without a single scan, which no report showed.

Silence is itself an answer, but it is an ambiguous one and the report says so.
Events are only written when something happens: a setup forming and being
rejected produces a `blocked` row, a quiet coin produces nothing at all. So an
empty result cannot separate "never scanned" from "scanned, never set up" — the
logs do not carry a per-scan record. What the report CAN do is place the silence
against the rest of the bot: if other symbols were producing events the whole
time, the bot was alive and the coin was quiet or unscanned; if nothing at all
was written, the bot was down and no gate is to blame.

Reads `bot_events.jsonl` backwards from EOF so the cost is proportional to the
window, not to the 98 MB file. Read-only.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from block_reasons import normalize_block_reason  # noqa: E402

EVENTS = HERE / "bot_events.jsonl"
WATCHLIST = HERE / "watchlist.json"


def iter_jsonl_reverse(path: Path, *, chunk: int = 1 << 20) -> Iterator[str]:
    """Yield raw lines from EOF backwards, so a recent window stays cheap."""
    if not path.exists():
        return
    with path.open("rb") as fh:
        pos = fh.seek(0, io.SEEK_END)
        carry = b""
        while pos > 0:
            size = min(chunk, pos)
            pos -= size
            fh.seek(pos)
            carry = fh.read(size) + carry
            parts = carry.split(b"\n")
            carry = parts.pop(0)
            for raw in reversed(parts):
                if raw.strip():
                    yield raw.decode("utf-8", "replace")
        if carry.strip():
            yield carry.decode("utf-8", "replace")


def parse_ts(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def collect(symbol: str, since: datetime) -> tuple[list[dict], dict[str, Any]]:
    """Events for one symbol newer than `since`, plus context about the bot.

    The context matters as much as the events: an empty result means something
    different depending on whether the bot was writing anything at all. Both are
    gathered in the same reverse pass.

    Stops on the first *symbol-matching* row older than the window. Rows for
    other symbols cannot end the scan — the file interleaves symbols, so an old
    row of another coin says nothing about ours — so a hard scan cap bounds the
    walk when the symbol is absent entirely.
    """
    needle = f'"{symbol}"'
    out: list[dict] = []
    scanned = 0
    other_in_window = 0
    newest_any: datetime | None = None
    oldest_seen: datetime | None = None

    for raw in iter_jsonl_reverse(EVENTS):
        scanned += 1
        ts = None
        mine = needle in raw
        if mine or scanned % 5 == 0:
            # every row for our symbol, plus a 20% sample of the rest — enough
            # to establish that the bot was alive without parsing 98 MB
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(rec.get("ts"))
            if ts is None:
                continue
            newest_any = newest_any or ts
            oldest_seen = ts
            if rec.get("sym") == symbol:
                if ts < since:
                    break
                rec["_ts"] = ts
                out.append(rec)
                continue
            if ts >= since:
                other_in_window += 1
        if oldest_seen is not None and oldest_seen < since and not out:
            # walked past the window without meeting the symbol once
            break

    out.reverse()
    return out, {
        "scanned": scanned,
        "other_events_in_window": other_in_window,
        "newest_any": newest_any,
    }


def report(symbol: str, hours: float, top: int) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    events, ctx = collect(symbol, since)

    try:
        in_watchlist = symbol in set(json.load(io.open(WATCHLIST, encoding="utf-8")))
    except Exception:
        in_watchlist = None

    print("=" * 74)
    print(f"{symbol} · последние {hours:g} ч · просмотрено строк: {ctx['scanned']}")
    print("=" * 74)
    if in_watchlist is False:
        print("НЕ В ВОТЧЛИСТЕ — бот его не торгует, это ожидаемо, а не сбой.")
        print("(watchlist.json неизменяем — см. CLAUDE.md §14)")
        return {"symbol": symbol, "verdict": "not_in_watchlist"}

    if not events:
        alive = ctx["other_events_in_window"]
        print("НИ ОДНОГО СОБЫТИЯ ПО ЭТОЙ МОНЕТЕ ЗА ОКНО.")
        print()
        if alive:
            print(f"Бот в это время работал: ~{alive} событий по другим монетам "
                  f"(выборка 1 из 5 строк).")
            print("Логи НЕ различают два случая — они пишутся только когда что-то")
            print("произошло, а тихая монета не пишет ничего:")
            print("  a) монету опрашивали, но сетап ни разу не сложился;")
            print("  b) монету не опрашивали вовсе.")
            print("Что проверить, чтобы различить:")
            print("  - 'full watchlist: N coins from scan + M re-added' в bot_stderr.log")
            print("  - MONITOR_FULL_WATCHLIST=True и MAX_POLL_PER_CYCLE в config.py")
            verdict = "silent_bot_alive"
        else:
            newest = ctx["newest_any"]
            print("И по другим монетам событий в окне тоже нет — бот молчал целиком.")
            if newest:
                print(f"Последнее событие вообще: {newest:%Y-%m-%d %H:%M} UTC.")
            print("Это отказ процесса, а не решение фильтра. Гейты ни при чём.")
            verdict = "bot_silent"
        return {"symbol": symbol, "verdict": verdict, "n_events": 0,
                "other_events_in_window": alive}

    by_type = Counter(str(e.get("event", "?")) for e in events)
    first, last = events[0]["_ts"], events[-1]["_ts"]
    print(f"событий: {len(events)}  ·  с {first:%m-%d %H:%M} по {last:%m-%d %H:%M} UTC")
    print("  " + "  ".join(f"{k}={v}" for k, v in by_type.most_common()))

    entries = [e for e in events if e.get("event") == "entry"]
    exits = [e for e in events if e.get("event") == "exit"]
    if entries:
        print(f"\nВХОДЫ ({len(entries)}):")
        for e in entries[-5:]:
            print(f"  {e['_ts']:%m-%d %H:%M}  {e.get('mode','?'):<14} "
                  f"{e.get('tf','?'):<4} price={e.get('price')}")
    if exits:
        print(f"\nВЫХОДЫ ({len(exits)}):")
        for e in exits[-5:]:
            pnl = e.get("pnl_pct")
            pnl_s = f"{pnl:+.2f}%" if isinstance(pnl, (int, float)) else "?"
            print(f"  {e['_ts']:%m-%d %H:%M}  {e.get('mode','?'):<14} pnl={pnl_s:<9} "
                  f"bars={e.get('bars_held')} reason={str(e.get('reason'))[:40]}")

    blocked = [e for e in events if e.get("event") == "blocked"]
    verdict = "entered" if entries else ("blocked" if blocked else "observed_no_setup")
    if blocked:
        codes: Counter[str] = Counter()
        example: dict[str, str] = {}
        last_seen: dict[str, datetime] = {}
        for e in blocked:
            code = normalize_block_reason(e.get("reason", ""), e.get("signal_type", ""))
            codes[code] += 1
            example.setdefault(code, str(e.get("reason", ""))[:78])
            last_seen[code] = e["_ts"]
        total = sum(codes.values())
        print(f"\nПОЧЕМУ НЕ ВОШЛИ — блокировки ({total}):")
        print(f"  {'причина':<26}{'раз':>6}{'доля':>8}  последняя")
        for code, n in codes.most_common(top):
            print(f"  {code:<26}{n:>6}{100*n/total:>7.0f}%  {last_seen[code]:%m-%d %H:%M}")
            print(f"      пример: {example[code]}")
        dominant, dn = codes.most_common(1)[0]
        print(f"\nГЛАВНАЯ ПРИЧИНА: {dominant} ({100*dn/total:.0f}% блокировок)")
        if entries:
            print("Но входы в окне ЕСТЬ — значит гейт отсекал не всё подряд.")
        verdict = f"blocked:{dominant}"
    elif not entries:
        print("\nБлокировок нет и входов нет: сетап ни разу не сложился.")
        print("Гейты тут ни при чём — стратегия не увидела условий входа.")

    return {"symbol": symbol, "verdict": verdict, "n_events": len(events),
            "n_blocked": len(blocked), "n_entries": len(entries)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="почему нет сигнала по символу")
    ap.add_argument("symbols", nargs="+", help="например POLUSDT C98USDT")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--top", type=int, default=6, help="сколько причин показать")
    ap.add_argument("--json", action="store_true", help="только машинный итог")
    args = ap.parse_args(argv)

    hours = args.days * 24 if args.days else args.hours
    results = []
    for sym in args.symbols:
        results.append(report(sym.upper(), hours, args.top))
        print()
    if args.json:
        print(json.dumps(results, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
