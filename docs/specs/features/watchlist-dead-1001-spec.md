# Dead pairs removed from the watchlist; daily liveness check (2026-10-01)

- **Slug:** `watchlist-dead-1001`
- **Status:** SHIPPED 2026-10-01 (operator: «проверь BAKEUSDT и убери мёртвые монеты»)
- **Truth harness:** TH-05 (stale data is no data), TH-13 (status checked on the exchange)
- **Rollback:** restore `files/watchlist.pre_dead_cleanup_1001.json` over
  `files/watchlist.json` + restart. The liveness check has no behaviour effect.
- **Maximum period / shadow:** not applicable — removing pairs that do not trade
  changes no decision on a pair that does

## Finding

The poll heartbeat showed BAKEUSDT evaluated on a bar of 2025-09-17. Checked every
watchlist pair on Binance (`exchangeInfo` status + newest 15m candle):

| pair | status | last 15m candle | dead for |
|---|---|---|---|
| SNTUSDT | BREAK | 2025-04-16 | 533 d |
| MKRUSDT | BREAK | 2025-09-15 | 381 d |
| BAKEUSDT | BREAK | 2025-09-17 | 379 d |
| LRCUSDT | BREAK | 2026-04-01 | 183 d |
| MDTUSDT | BREAK | 2026-04-23 | 161 d |
| OXTUSDT | BREAK | 2026-04-23 | 161 d |
| TRUUSDT | BREAK | 2026-04-28 | 156 d |
| TONUSDT | BREAK | 2026-06-30 | 93 d |
| PYRUSDT | BREAK | 2026-08-17 | 45 d |

Since 2026-08-01 only PYRUSDT produced events (8 entries / 8 exits, last
08-14, while it still traded); the other eight produced none but were still
polled — slots of the 45-per-cycle rotation spent on year-old bars.

## Change

- `files/watchlist.json`: 102 → **93** pairs (backup
  `files/watchlist.pre_dead_cleanup_1001.json`). Removing delisted pairs is not an
  exception to "DO NOT EXPAND" (CLAUDE.md §14), same as 2026-08-19.
- `config.DEFAULT_WATCHLIST` (fallback when the file is missing): the nine plus the
  three removed from the file on 08-19 (RNDR, EOS, ACA) — 103 → 91; otherwise a
  missing file would bring twelve dead pairs back.
- `bot.py`: the `/why` usage example named TONUSDT → SOLUSDT.
- `files/watchlist_liveness.py`, daily step `watchlist liveness` (before notify):
  status != TRADING or newest candle > 24 h → dead; written to
  `.runtime/watchlist_liveness.json`; the morning report shows a P1 step
  "Убрать из watchlist мёртвые пары". It never edits the watchlist (the operator's
  list), and an exchange outage is "unknown", never "dead". First run after the
  cleanup: 93 pairs, 0 dead.

## Tests

`files/test_watchlist_dead_1001.py` (9).
