# Operational diagnostics — `why_no_signal` + a restart that fails loudly

- **Slug:** `operational-diagnostics`
- **Status:** shipped 2026-08-13
- **Roadmap:** [`gpt-bot-transfer`](../../roadmaps/gpt-bot-transfer-roadmap.md) items G3 and G9
- **Rollback:** delete the new files; nothing existing changed behaviour

## Problem

Two recurring costs, both diagnostic rather than algorithmic.

**"Где сигналы по X?"** was asked four times in one session — POL, C98, BTC,
ATOM. Each answer took a manual dig through `bot_events.jsonl`,
`critic_dataset.jsonl` and the stderr log. POL had gone 15 days without a scan
and no report showed it.

**Restarts reported success over a dead bot.** `restart_bot.bat` re-launches
itself with `cmd /k "%~f0" --run` and then `exit`s, so the caller sees exit code
0 immediately while the real work happens in a detached window. It prints
`Bot: exit code !errorlevel!` and never acts on it. On 2026-08-13 a restart
left `bot.py` not running and it was noticed only by listing processes by hand;
the bot had previously stayed dead 8 days (07-23) and 7 hours (08-04).

The other bot solved both: `files/why_no_signal_report.py` and
`restart_full_stack.bat` with `-FailIfNotRunning` status checks.

## Behaviour

### `files/why_no_signal.py`

```
pyembed\python.exe files\why_no_signal.py POLUSDT --days 3
pyembed\python.exe files\why_no_signal.py C98USDT BTCUSDT --hours 24 --json
```

Reads `bot_events.jsonl` backwards from EOF in 1 MB chunks, so cost tracks the
window, not the 98 MB file (POL over 3 days: 7 804 lines scanned).

Verdicts, and what each one licenses:

| verdict | meaning | where to look next |
|---|---|---|
| `not_in_watchlist` | expected, not a fault | watchlist is immutable (CLAUDE.md §14) |
| `blocked:<code>` | a gate rejected it; the dominant code and its share are printed | that gate's thresholds |
| `entered` | it did fire | entry/exit list with pnl |
| `observed_no_setup` | events exist, none are blocks | strategy conditions never met |
| `silent_bot_alive` | no events for this symbol, plenty for others | monitoring — but see below |
| `bot_silent` | no events at all | the process, not the gates |

**The ambiguity is stated, not hidden.** Events are written only when something
happens, so an empty result cannot separate "never scanned" from "scanned, never
set up" — no per-scan record exists. The report says which of the two it cannot
distinguish and names the check that does (`full watchlist: N coins from scan`
in the stderr log). Claiming a monitoring bug from silence alone would violate
§0a rule 10.

### `files/block_reasons.py`

`normalize_block_reason()` maps the free-text `reason` onto a stable code.
Blocked rows carry a human sentence in two languages and several spellings of
the same gate: `MTF: 1м MACD` uses a Cyrillic `м` while `MTF: 1m retest` uses
Latin `m`, `<=` and `≤` both occur, and 449 rows read `????????: портфель
полон` where a cp1251 write mangled the prefix — 310 distinct templates across
213 621 blocked rows.

Unmatched text returns `unclassified` rather than a nearest guess: an unmatched
reason is evidence that a gate changed its wording, and an "other" bucket is how
a taxonomy rots unnoticed. This module is a bridge; the destination is a
structured `reason_code` at the decision site (roadmap E5).

### `files/run_test_suite.py`

The other bot gates its restart on a green suite. Ours cannot: discovery over
`test_*.py` reports **757 tests with 40 failures**, all pre-existing. So the gate
compares against a baseline in `.runtime/test_baseline.json` and fails only on
tests that regressed — the same shape as the pre-commit harness, which blocks
staged files and warns about legacy debt.

```
pyembed\python.exe files\run_test_suite.py           # regression gate
pyembed\python.exe files\run_test_suite.py --update  # re-record the baseline
pyembed\python.exe files\run_test_suite.py --strict  # require green
```

It runs with `files/` as the working directory. Discovering from the repo root
made relative data paths fail and turned 40 real failures into 148 phantom ones —
the gate would have recorded its own misconfiguration as project debt.

### `restart_full_stack.bat`

One process, no `cmd /k` self-relaunch, non-zero exit on any failure:

1. test regression gate (`--skip-tests` for an emergency restart);
2. stop RL worker and bot;
3. read the token from the generated runner file (never printed, §13);
4. start RL worker, then bot, checking each launcher's exit code;
5. settle;
6. `bot_status.ps1 -FailIfNotRunning` and `rl_worker_status.ps1 -FailIfNotRunning`.

`bot_status.ps1` and `rl_worker_status.ps1` gained `-FailIfNotRunning` (they
always exited 0 before) and now redact the bot token from the stderr tail they
print — every `httpx` line contains it in the URL (§13).

`restart_bot.bat` is left untouched as the interactive path.

## Verification

- `python -m unittest test_why_no_signal` — 8 tests: the taxonomy pinned against
  strings sampled from the live file, including the Cyrillic and mangled
  variants, and one test per verdict.
- Roadmap gate for G3: run on POL and C98 and reproduce the manual answer.
  POL/3d → 44 events, 100% `trend_quality` (`RSI 73.2 > 72.0`), matching the
  hand analysis. C98/3d → one entry on 08-11 closed at −2.59% on the ATR trail,
  then 69% `trend_quality`. ATOM/6h → 60 blocks, 58% `trend_1h_chop`.
- Roadmap gate for G9: the gate was proven to catch regressions by adding a
  deliberately failing test — exit 1, test named — then removing it. A live run
  restarted both workers and reported `RESTART OK` with new PIDs.

## Fixed along the way

`test_correlation_guard.py` replaced `sys.modules["config"]` with a stub at
import time and never restored it, so every module importing config later in a
discovery run got the stub. Under `unittest discover -p "test_*.py"` that turned
3 errors into **221**, all `module 'config' has no attribute ...`. The stub is
now visible only while `correlation_guard` binds its reference. Its own two
`TestPruneCandidates` failures pre-date this change and are in the baseline.
