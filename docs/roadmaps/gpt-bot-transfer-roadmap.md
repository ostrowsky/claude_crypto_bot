# Transfer roadmap — what `gpt_crypto_bot` already solved

- **Slug:** `gpt-bot-transfer`
- **Status:** in progress
- **Created:** 2026-08-13
- **Source:** read-only architecture review of `D:\Projects\gpt_crypto_bot`
  (79 261 LOC / 273 py files / **127 test files** / 147 specs / 188 commits in 90 days)
- **Owner:** Vasiliy Ostrovsky + Claude

> Living document. Every item that ships updates its status row here, plus the
> component row in
> [`auto-improvement-loop-spec.md`](../specs/features/auto-improvement-loop-spec.md)
> when it touches a loop component.

## Why this exists

Both bots chase the same objective and hit the same wall. The other bot's daily
critic for 2026-08-13 says it plainly: of the exchange top-20, five were in its
watchlist, it bought **5/5 (capture 100%)** and was early on **1 (20%)**. It
buys the winners late — our problem exactly — but it *measures* that as two
separate numbers every day, and it has built the tooling to attack it.

What transfers is not their trading policy (different watchlist, they size
positions, we alert). What transfers is **infrastructure, discipline, and their
refuted hypotheses**, which we get for free.

None of the numbers quoted from their repo have been reproduced on our data.
They are their claims, and each item below carries its own validation gate
before anything changes our behaviour.

## Priorities

Ordered by value ÷ cost, not by ambition. P0 items pay for themselves the first
time they are used; P3 is a research programme.

| ID | Item | Priority | Cost | Status |
|----|------|----------|------|--------|
| G3 | `why_no_signal` — one command answers "где сигналы по X?" | **P0** | hours | shipped 2026-08-13 |
| G9 | Restart stack: tests first, then verify every worker is up | **P0** | hours | shipped 2026-08-13 |
| G7 | Daily artifact names its denominator + blocker harm | **P1** | 1 day | planned |
| G1 | SQLite event store with byte-offset incremental sync | **P1** | 2–3 days | planned |
| G4 | Forward-cohort promotion gate before production | **P1** | 1–2 days | planned |
| E3 | Freshness SLO per learning artifact | **P1** | hours | part shipped 2026-08-13 (stale-lock half) |
| G2 | Canonical continuous OHLCV store + coverage gate | **P2** | 2 days | planned |
| G6 | Decomposed bandit reward components (logged, not enforced) | **P2** | 1 day | planned |
| G10 | Inherit their refuted hypotheses; re-test H5 their way | **P2** | 1 day | planned |
| E5 | Structured `reason_code` at the decision site | **P2** | 2 days | planned |
| E4 | Extract entry gates into pure, unit-testable predicates | **P3** | weeks | planned |
| G5 | Offline decision environment with an oracle upper bound | **P3** | weeks | planned |

---

## P0

### G3 — `why_no_signal`

**Their version:** `files/why_no_signal_report.py` (270 lines) reads
`critic_dataset.jsonl` / `bot_events.jsonl` backwards from EOF in 1 MB chunks
and reconstructs, for one symbol and window, every decision and block.

**Our pain, measured:** in a single session the operator asked this four times —
POL, C98, BTC, ATOM — and each answer took a manual dig through three datasets.
POL had gone 15 days without a scan and nobody could see it.

**Gate:** run it on POL and C98 and reproduce, mechanically, the root causes
found by hand. If it disagrees with the manual answer, the tool is wrong.
Read-only; no flag needed.

### G9 — restart stack that fails loudly

**Their version:** `restart_full_stack.bat` runs `[0/8] Running tests` and
aborts on failure, prints the build commit and date, restarts four workers, then
calls `<worker>_status.ps1 -FailIfNotRunning` for each.

**Our pain, measured:** today `restart_bot.bat` returned success and **did not
start `bot.py`**; it was noticed only by listing processes by hand. On 07-23 the
bot stayed dead 8 days, on 08-04 for 7 hours.

**Gate:** kill the bot, run the script, and confirm a non-zero exit code when a
worker fails to come up. Rollback: the old scripts remain untouched.

---

## P1

### G7 — the daily artifact must name its own denominator

Their critic JSON carries `watchlist_top_denominator =
"exchange_top_filtered_to_watchlist"` beside the rate, publishes
`capture_rate` and `early_capture_rate` separately, and adds
`blocked_winner_count`, `blocked_reason_harm`, `why_no_signal_examples`.

We have already been burned by exactly this: PROJECT_CONTEXT records that the
recall denominator changed in April and "методологии несопоставимы напрямую".
A name inside the artifact makes that failure impossible.

**Gate:** recompute one historical day both ways and show the two denominators
produce different numbers — that difference is the whole point.

### G1 — SQLite event store, synced by byte offset

Their `research_event_cohort_store.py` keeps JSONL as an append-only journal and
syncs into SQLite from the last byte offset: `source_state(source_file,
byte_offset, source_size)`, primary key `(source_file, byte_offset)`, a
single-writer lock, `SCHEMA_VERSION`, and truncation detection that resets the
offset.

CLAUDE.md §7 already names this as our open defect: "Real fix still open: stop
rewriting whole files". Streaming rewrites and reverse `get_record` (17.6s →
0.11s) treated the symptom.

**Gate:** identical aggregates from the SQLite path and the JSONL path on the
same window, plus wall-clock before/after for `analyze_blocked_gates.py`. A
mismatch means the migration lost rows.

### G4 — forward cohort before production

Their `forward-shadow-promotion-gates.md`: after a replay approves a hypothesis,
it must earn an **independent forward cohort** — rows strictly after the model's
`created_at_utc`, a maturity rule (both T+5 and T+10 present), a minimum of 30
mature candidates and 5 local days, and only then numeric thresholds. Every
decision is stamped `production_eligible=false`.

We ship on one backtest. The cost is documented: the 8% trail widen backtested
"+net over 35 days" and lost 54.9% cumulative on impulse_speed in 5 live days.

**Gate (cheap, historical):** replay the rule against our last three rollouts —
trail 8%, curtail hard-block, soft gate — and count how many it would have
stopped.

### E3 — freshness SLO per artifact

In their repo `critic_dataset.jsonl` was last written 2026-08-04 while every
other dataset updated the same day I looked (08-13): a learning input silently
stopped nine days earlier, on a bot that has freshness verdicts. Our mirror of
this: scheduled tasks silently skipped for 11 days on battery power.

**Gate:** declare the expected write interval per artifact, then verify the
alarm fires on a deliberately stale copy.

**Part shipped 2026-08-13 — the stale-lock instance.** Looking for our version
of this defect found a worse one. `.runtime/backfill_critic.lock` was an empty
file **1389 hours (58 days) old**: the lock was `O_CREAT|O_EXCL` with no owner
and no timestamp, so the run that died in mid-June left it forever and every
backfill since returned at INFO level with `Backfill already running`. 11 894
rows sat unlabelled the whole time and no report said a word.

The lock now records `{pid, ts}`, respects a live holder, and takes over a dead
or expired one with a WARNING — an input that stopped filling is not routine
news. `_lock_owner_alive` requires the PID to be a *python* process, since PIDs
recycle, and fails safe: if the check itself errors it assumes alive, so a
working backfill is never displaced. Six tests in `test_backfill_lock.py`,
including the exact zero-byte artefact found in production.

Still owed for E3 proper: a declared write interval per artifact and an alarm
when one lapses. This fix removes one cause; it does not detect the next one.

---

## P2

### G2 — canonical OHLCV store

`files/v2/history_store.py` keeps per-symbol/timeframe slices with metadata
(`rows`, `start_ts_ms`, `end_ts_ms`, `source`, `updated_at_utc`), upsert keyed on
`open_ts_ms`, and a coverage gate ("60d passed on 95/105 symbols").

Our mirror: the ceiling experiment needed 200 days of klines fetched ad hoc, and
the previous cache had 11 gappy symbols that put 271 rows in train against 3100
in holdout.

### G6 — decomposed reward

Their reward names every component: `early_capture`, `trend_hold`,
`realized_pnl`, `mfe_retention`, `false_buy_penalty`, `late_entry_penalty`,
`giveback_penalty`, `churn_penalty`, `blocked_winner_penalty` — "so later
reports can show *why* an agent won or lost reward instead of hiding the answer
inside one scalar". Ours is ±1.0 / −0.8 / +0.10 / −0.12.

`late_entry_penalty` is the interesting one: 62% of our impulse_speed entries
are late and the reward has no idea.

### G10 — inherit their refuted hypotheses

Committed with numbers, so we never pay to learn them again:

| Their hypothesis | Result |
|---|---|
| exhaustion-aware exits (4 variants) | all lost to `base_sell_0_70`; static threshold exits rejected |
| early RSI-WEAK exit (hold/grace/confirmation/veto/partial-tail) | 831 exits, every profile failed validation+holdout |
| impulse-expansion protected tail (8 profiles) | avg **−0.0829pp**, median −0.1772pp, **82% harmed** |
| temporal exit `mature_decay_late_rise` | OOS +122.90, 9/9 grid cells positive, **but −100.29 in the latest window** |

Our H5 (trailing-only after break-even) and the EX1 trail widen belong to the
same family of static threshold exits. **Re-test H5 their way** — causal replay
on the bot's own exits — expecting it not to survive.

### E5 — structured reason codes

Their `blocking.normalize_blocked_reason` classifies by substring (`"portfolio"
in text or "портфель" in text`, `"cluster"` and `"cap"`, `"accuracy <"`). Rename
a log line and the taxonomy silently reclassifies — and critic reports, casebook
and hypothesis priority all sit on that taxonomy. Our version of the defect: 740
un-wired block sites emit `<none>`, so Pareto sweeps under-count.

---

## P3

### E4 — testable decision path

`monitor.py` is 7 550 lines with `_poll_coin` spanning lines 4987–7422; a gate
cannot be tested without running the loop, which is likely why their validation
happens through dozens of heavy replay scripts. Ours is 6 556 lines. We already
have the right pattern in `trend_scout_rules.BlockRule` — it is just applied
partially.

### G5 — offline decision environment with an oracle

`v2/offline_env.py` + `policy_baselines.py` anchor every policy against
`always_flat` (0), `lifecycle_oracle` (**+5867**, explicitly labelled an
optimistic upper bound, not deployable) and `belief_policy_v1` (**−1864**).
Their policy-gap audit then localised the defect — 1529 noise entries against
373 emerging — and the residual-gap decomposition moved the priority to exit
monetisation.

We have no upper bound at all. Today's day-start ceiling (37% of rockets at 10
alerts/day, lift 3.48×) is the first such number in this project and it is a
single point. The value of an oracle is that it says where *not* to invest.

---

## Deliberately not transferred

**Their v2 research programme as a whole.** 147 specs, 20 modules under
`files/v2/`, and by their own numbers the belief policy is still at −1864
against an oracle of +5867, with the best threshold bridge at −554. `v2` is
imported by no production module. The rigour is exemplary; the missing piece is
a promotion path. Copying the programme without one would reproduce our current
failure — twelve unused backtests and no pipeline decision since 2026-06-17 —
at a larger scale.

We take their tools and their refuted results, not their open research bill.
