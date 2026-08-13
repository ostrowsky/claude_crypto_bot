# Event store — the journal stays, the queries move to SQLite

- **Slug:** `event-store`
- **Status:** shipped 2026-08-13 (read path); write path migration open
- **Roadmap:** [`gpt-bot-transfer`](../../roadmaps/gpt-bot-transfer-roadmap.md) item G1
- **Truth-harness invariants:** TH-05 (a metric must know what it does not know),
  TH-12 (traceable to evidence)
- **Rollback:** delete `.runtime/event_store.sqlite3`; nothing reads it yet that
  cannot fall back to the JSONL path

## Problem

Two costs, one root cause: every consumer treats a 98 MB append-only log as a
random-access database.

**Reads re-parse everything.** `analyze_blocked_gates.py`,
`_backtest_top20_coverage_funnel.py`, `_compute_early_capture.py` and every
ad-hoc investigation walk `bot_events.jsonl` from byte zero. 245 967 rows, ~7–12
seconds per pass, repeated per script per run.

**Writes rewrite the whole file, and that is now failing.** CLAUDE.md §7 already
names it — "Real fix still open: stop rewriting whole files" — but on 2026-08-13
it stopped being a performance note. With the stale backfill lock cleared, the
backfill fetched all 11 908 labels and died on the final step:

```
PermissionError: [WinError 5]
  critic_dataset.jsonl.backfill.tmp -> critic_dataset.jsonl
```

On Windows a file the live bot holds open cannot be replaced. The debris made
the history visible: **nineteen orphaned `.tmp` files, 1.09 GB**, owned by
long-dead PIDs, the oldest 40 days — one per earlier failed rewrite. The stale
lock was a symptom; this was the crash that produced it.

## Design

`files/event_store.py`. The JSONL remains the source of truth and is never
rewritten; SQLite is a derived mirror brought forward from the last byte offset
it consumed.

| Decision | Why |
|---|---|
| PK `(source_file, byte_offset)` | a replayed sync cannot double-count a row |
| offset > current size ⇒ reset + drop rows | a shrunk source was rotated or truncated; resuming would splice two different files into one table |
| stop at a line with no trailing newline | a writer mid-append leaves a partial line; storing half a record then skipping its remainder is silent corruption |
| `SCHEMA_VERSION` bump rebuilds | the store is derived, so a rebuild is cheap and a half-applied migration is not |
| lock file with `{pid, ts}`, stale-aware | same lesson as the backfill lock: a killed writer must not block the next one forever |
| malformed lines counted, not fatal | one bad row must not stop a sync of 245 967 |

Indexed columns are extracted for querying (`day`, `event`, `sym`, `tf`, `mode`,
`reason_code`, `pnl_pct`); the full record stays in `payload` so nothing is lost
in translation.

```
pyembed\python.exe files\event_store.py sync
pyembed\python.exe files\event_store.py stats
```

## Verification

**Roadmap gate — identical aggregates, both paths.**
`files/_verify_event_store_parity.py` computes per-event and per-block-reason
counts over the same 14-day window from the JSONL and from SQLite:

```
JSONL  : 44649 событий за 6.98с
SQLite : 44649 событий за 3.85с
совпадает по типам событий: 11 ключей
совпадает по причинам блокировок: 12 ключей
```

**The gate earned its keep on the first run**, reporting 7 517 blocks as
`trend_1h_chop` from one path and `unclassified` from the other. Two real
defects, not noise:

- `bot_events.jsonl` writes `trend_chop` (15 626 rows) as a short `reason_code`
  for the same gate whose free text reads `trend/1h chop:`. The taxonomy knew
  only the second spelling, so every one of those rows was unclassified wherever
  the short code was used — including the harm table shipped hours earlier.
- The parity script compared the store's `reason_code` against the journal's
  `reason`, i.e. two different field choices rather than the sync itself. Both
  sides now mirror the same precedence.

**Cost.** Full initial sync of 245 967 rows: 40.9s. Re-sync with nothing
appended: **0.02s**. That ratio is the whole point — an analysis re-run pays for
the bytes added since it last looked, not for the file.

**Tests.** `python -m unittest test_event_store` — 10 tests on real files:
resume reads only appended bytes, truncation resets rather than splices, a
partial trailing line waits for its newline, a replayed offset cannot
double-count, malformed lines are counted, a schema bump rebuilds without loss.

## Not done yet

The write path. Nothing yet writes labels through the store, so
`backfill_critic_labels` still rewrites `critic_dataset.jsonl` whole and still
cannot run while the bot holds the file. Until that migrates, label backfills
require a stopped bot. That is the next slice of G1, and it is what actually
retires the `.tmp` graveyard.
