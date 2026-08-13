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

## Backfill lock — a dead run must not block labels forever

**Truth-harness invariants: TH-05** (a metric must know what it does not know —
absence of data is not evidence of anything) and **TH-12** (every change
traceable to evidence).

Looking for our instance of the other bot's stale-input defect found a worse
one. `.runtime/backfill_critic.lock` was an empty file **1389 hours (58 days)
old**. The lock was `O_CREAT|O_EXCL` with no owner and no timestamp, so the run
that died in mid-June left it behind permanently and every backfill since
returned at INFO level with `Backfill already running (lock file exists),
skipping`. 11 894 rows stayed unlabelled and no report mentioned it — the
learning loop was quietly missing an input for two months.

Behaviour now:

- the lock records `{pid, ts}`;
- a live holder is respected and the run skips, as before;
- a dead or expired holder (`_LOCK_TTL_SEC = 3h`) is taken over, logged at
  **WARNING** — an input that stopped filling is not routine news;
- `_lock_owner_alive` requires the pid to be a *python* process, because pids
  recycle and "some process has this id" is not evidence the backfill lives;
- the liveness check fails safe: if it errors, the owner is assumed alive, so a
  working backfill is never displaced.

Rollback: revert the file; the lock degrades to the previous behaviour.

Verification: `python -m unittest test_backfill_lock` — 6 tests covering a free
lock, a live owner, a dead owner, an expired lock, the exact zero-byte artefact
found in production, and an unavailable liveness check. Suite after the change:
763 tests, 40 failing, unchanged from baseline.

This removes one cause of a silent input stall. Detecting the next one is the
freshness manifest below.

## Artifact freshness SLO — how a stall gets noticed

**Truth-harness invariant: TH-05.** A learning input that stopped arriving is
neither a crash nor a filter decision; it is missing evidence, and every number
downstream inherits the gap without saying so. Two cases prompted this: our
backfill lock (58 days) and, in the sibling bot, `critic_dataset.jsonl` frozen
on 2026-08-04 while every other dataset kept updating.

`files/artifact_freshness.py` declares a maximum age for each artifact the loop
depends on:

| artifact | limit | observed 2026-08-13 |
|---|---|---|
| `bot_events.jsonl` | 2h | 0.0h |
| `critic_dataset.jsonl` · `ml_dataset.jsonl` | 6h | 0.0h · 0.2h |
| `ml_candidate_ranker.json` | 6h | 0.3h |
| `top_gainer_dataset.jsonl` | 12h | 3.2h |
| `bandit_entry_state.json` · `top_gainer_model.json` | 36h | 4.8h · 15.1h |
| `learning_progress.jsonl` · `metrics_daily.jsonl` · `pipeline/health` | 36h | 15.1h · 5.8h · 4.8h |
| `fast_reversal_catboost.cbm` | 36h, flag-gated | disabled |

Every limit is roughly 2–3× the observed cadence, so normal jitter never fires.
Each carries a stated reason in code: a threshold without one rots into a number
nobody dares change.

An artifact behind a config flag reports `disabled`, not `stale`, when the flag
is off. `fast_reversal_catboost.cbm` is 46 days old on purpose, and a checker
that cries wolf about it is one people stop reading — after which the next real
stall is invisible again. The flag lookup fails **open**: if config cannot be
read the artifact is treated as expected, so a config error never hides a stall.

Wired into `truth_harness.audit_artifact_freshness` as `TH05_ARTIFACT_FRESHNESS`
(stale → error, missing → warning), so it runs with every full audit rather than
waiting to be remembered. Also standalone:

```
pyembed\python.exe files\artifact_freshness.py         # table, exit 1 if stale
pyembed\python.exe files\artifact_freshness.py --json
```

Verification: `python -m unittest test_artifact_freshness` — 11 tests using real
backdated files rather than a mocked clock, including the roadmap gate (alarm
fires on a deliberately stale copy, scaled to the 58-day outage), the boundary
case, a disabled flag, an unreadable config, and directory artifacts taking
their newest file.

## Daily artifact names its denominator and ranks gates by harm

**Truth-harness invariants: TH-01** (a ratio without its base is not evidence)
and **TH-04** (comparable windows).

Two changes to the daily metrics, both report-side; no trading behaviour moves.

**The denominator is written into the artifact.** `NS_EarlyCapture_top20` and
`C1_C2_coverage_funnel` now carry
`denominator = "top20_within_watchlist_from_top_gainer_dataset"`, and the North
Star also carries `label_provenance = "rolling_24h_same_snapshot"`. The recall
denominator changed once already, in April, and PROJECT_CONTEXT records the two
methodologies as "несопоставимы напрямую" — a rate whose denominator lives only
in a script comment can be redefined without anyone noticing.

**Gates are ranked by harm, not by volume.** The funnel groups blocked top-20
winners through `normalize_block_reason` and reports how many winners each gate
cost:

```
Harm per gate — top-20 winners lost, by normalised block reason:
  trend_quality        2 winners   7.7% of all top-20
  ml_proba_zone        2 winners   7.7% of all top-20
```

"How often did this gate fire" is a volume statistic and ranks the noisiest
gate first; "how many winners did it cost" is the only figure that ranks gates
against the North Star.

Two defects surfaced while wiring it:

- The funnel grouped by the raw reason string. `bot_events.jsonl` carries free
  text (310 templates) while `critic_dataset.jsonl` carries `decision.reason_code`
  (22 short values), so running the codes through the free-text regexes returned
  `unclassified` for **every** blocked winner — the harm table named no gate at
  all. `block_reasons` now passes known codes through and maps the legacy
  spellings (`ml_zone` → `ml_proba_zone`, `cooldown` → `symbol_cooldown`,
  `portfolio` → `portfolio_full`). `entry_score_soft_pass` is mapped explicitly
  so it cannot be folded into `entry_score`: it is the opposite of a block.
- The emitted buckets did not reconcile — `entered + blocked_only + no_event`
  left 3 of 26 winners unaccounted for, which reads as an arithmetic error to
  anyone adding them up. `candidate_only`, `other_event` and `buckets_sum` are
  now emitted too, and 16 + 4 + 0 + 3 + 3 = 26.

Verification: `_backtest_top20_coverage_funnel.py` and `_compute_early_capture.py`
re-run on the live 14-day window; 4 new taxonomy tests in `test_why_no_signal.py`
pin the pass-through and the legacy spellings.

## Fixed along the way

`test_correlation_guard.py` replaced `sys.modules["config"]` with a stub at
import time and never restored it, so every module importing config later in a
discovery run got the stub. Under `unittest discover -p "test_*.py"` that turned
3 errors into **221**, all `module 'config' has no attribute ...`. The stub is
now visible only while `correlation_guard` binds its reference. Its own two
`TestPruneCandidates` failures pre-date this change and are in the baseline.
