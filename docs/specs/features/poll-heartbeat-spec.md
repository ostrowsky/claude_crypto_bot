# Poll heartbeat — what the live poll saw, per coin per closed bar

- **Slug:** `poll-heartbeat`
- **Status:** SHIPPED 2026-09-25, flag ON; takes effect at the next bot restart
- **Origin:** algorithm audit 2026-09-25 (`algorithm-audit-0925-spec.md`), P0
  hypothesis П-2
- **Truth-harness invariants:** TH-05 (absence of data must be distinguishable
  from a negative), TH-07 (behind a flag, stated rollback), TH-12, TH-13
- **Code:** `files/poll_heartbeat.py`, hooks in `files/monitor.py::_poll_coin`,
  `files/config.py` (`POLL_HEARTBEAT_ENABLED`, `POLL_HEARTBEAT_KEEP_DAYS`)
- **Tests:** `files/test_poll_heartbeat.py` (10)
- **Rollback:** `POLL_HEARTBEAT_ENABLED = False` and restart. Nothing reads the
  files, so switching it off changes no decision; the directory can be deleted.

## Why

On 17.5% of the 166 immutable top-20 winner-days of the audit an entry rule fired
on the stored klines inside the birth window (UTC open .. first +2.5% crossing),
yet the bot logged no candidate, no block and no entry. `_poll_coin` returns
silently in four places before a candidate exists — klines missing, history too
short, open position, cooldown (logged only on its first bar) — and when no rule
fires it returns without a trace unless the bar is a near-miss. So the logs
cannot tell apart:

- the coin was not polled on that bar (rotation, missing from the monitored set);
- it was polled but the live bar differed from the stored one;
- it was polled and in cooldown, or already held;
- it was polled and the rule did not fire live.

Each has a different fix. Guessing between them is exactly the "absence of data
read as evidence" failure TH-05 forbids, so the first step is to measure.

## What is written

`.runtime/poll_heartbeat/<UTC day>.jsonl`, one line per (symbol, timeframe,
closed bar) — the FIRST poll of that bar; later polls of the same bar are
dropped (same closed bar, same inputs).

| field | meaning |
|---|---|
| `stage` | `no_data`, `short_history`, `position_open`, `cooldown`, `evaluated` |
| `bar_ts`, `bar_utc` | the last closed bar the decision used; for `no_data` / `short_history` the wall-clock bucket |
| `rules` | `entry`, `breakout`, `retest`, `surge`, `impulse`, `alignment`, `ema_cross` → bool |
| `fired` | the rules that passed |
| `reasons` | the failing reason of every rule that did not pass, ≤ 80 chars (`ema_cross` says `cross cooldown` when its own cooldown skipped it) |
| `price`, `hour_blocked`, `bars_left`, `n_bars` | context where applicable |

What happens to a candidate after `evaluated` is already in `bot_events.jsonl`
(blocked / entry); the heartbeat covers only the part that had no log.

Volume: ~100 coins × 96 bars/day on 15m ≈ 10k lines, ~3 MB a day; files older
than `POLL_HEARTBEAT_KEEP_DAYS` (45) are pruned once a day.

## Safety

Logging only. Every call is wrapped; an exception inside the heartbeat is logged
at debug level and never reaches the poll loop. The rule reasons were already
computed and discarded — the change keeps them instead of `_`. A test asserts
that `monitor.py` only calls `poll_heartbeat.record` / `enabled`, so no decision
can start depending on it silently, and that every silent `return` before rule
evaluation records first.

## Backtest and shadow decision

No maximum-period backtest applies: the change alters no decision, so there is
nothing to replay — the same signals, blocks and exits come out with the flag on
or off (the only code change on the decision path is naming the rule reasons
that were already computed). It is itself a shadow instrument: it observes and
writes, and no gate, score or exit reads it. The evidence that it is inert is
`test_poll_heartbeat.py` (logging-only and fail-safe tests) plus `test_bot.py`,
whose failure set is identical to origin/main except one order-dependent test
(`test_T34_fill_labels_updates_record`) that passes in isolation on both.

## How it will be used

After ≥ 7 days of data, rerun the winner funnel of the audit
(`_audit_winner_funnel.py`) with the heartbeat joined in and split the 17.5%
bucket into the four causes above. Until then the bucket stays unexplained —
"рано судить", not a guess.
