# Status panel: live flags and the day's leaders

- **Slug:** `status-panel-truthful`
- **Status:** DEPLOYED 2026-08-21 (live restart 00:28 local)
- **Truth-harness invariants:** TH-10 (results committed with their numbers),
  TH-12 (evidence travels with the change), TH-13 (never assert without proof —
  this change exists so the panel stops making an unverifiable claim)
- **Flags:** `UI_LEADERS_COUNT = 8`, `UI_LEADERS_REFRESH_SEC = 60.0`
- **Rollback:** set `UI_LEADERS_COUNT = 0` to drop the leaders block, or revert
  this commit. Display-only; no gate, threshold, model or trading path is
  touched, so nothing about what the bot *does* changes either way.

## What the panel could not say

**What is running.** The version line is git HEAD of the working tree. On
2026-08-20 it read `a041af9 · 08-19` for hours while the process ran a config
edited one minute before startup (config.py mtime 23:38:17, process start
23:39:28). This session's commits were pushed through a separate worktree, so
local HEAD never moved and the line described a commit unrelated to the running
behaviour. The operator reasonably concluded the bot was running old code; it was
not, and proving that took a process-start-vs-mtime comparison that the panel
should have made unnecessary.

**What the day is doing.** The panel reported `Открытых сигналов: 0` on a day when
three watchlist coins sat in Binance's daily top-20 (XRP, ORDI, ENA, +19–21%).
That is accurate and useless: an empty portfolio with no indication of what the
market did or why nothing was taken. Answering "it's in the top-20, why isn't it
in the portfolio" required excavating `bot_events.jsonl`.

## What it says now

```
🔖 vr195-a041af9 · cfg 08-21 00:05
⚙️ ml floor 0.10/0.10 - segments off - bandit on - rotation on

Мониторинг: ▶️ запущен
Монет в списке: 102
Открытых сигналов: 0

Лидеры дня по движению:
▫️ XTZ  +29.6% (закр +2.6%)
▫️ CELO +28.0% (закр +4.0%)
▫️ OXT  +26.5% (закр −4.1%)
▫️ ENA  +23.9% (закр +23.1%)
✅ ORDI +21.4% (закр +12.9%) — в портфеле
▫️ XRP  +19.8% (закр +12.3%) — отклонён: ml_zone
```

The build string is kept — it still identifies the source tree — but it is no
longer the only claim on the line. `cfg` is the mtime of the config this process
loaded, and the flags are read from the loaded module, so both describe the
running process rather than the repository.

## Ranked by MOVE, not by close

`high / open − 1` over the rolling 24h window, the same definition used
throughout the trend work and in durable memory: the target is the day's largest
**move**, and a coin that ran and gave it back was still a run the bot should
have caught. The first three live rows make the case on their own — XTZ ran
+29.6% and closed +2.6%; OXT ran +26.5% and closed **negative**. A close-ranked
panel would have hidden all three behind ENA.

Non-watchlist symbols are excluded. The watchlist is immutable, so a coin the
operator cannot trade is noise on this panel rather than a finding.

## Why there is no backtest here

The truth harness asks for a maximum-period backtest on behaviour changes, and
this one has none because it changes no behaviour: no gate, threshold, model,
entry, exit or sizing path is touched. The only new decision the bot makes is
which eight rows to print. The claim that needs evidence — that the panel now
tells the truth about the running process — is verified directly instead, by
rendering it against the live config and comparing to the values the process
loaded (§ Verification).

## Shadow / canary decision

**Not applicable, and not skipped silently.** A shadow period compares a proposed
decision against the live one; there is no decision here to compare. The risk
this change carries is not a wrong trade but a slow menu, and that is addressed
structurally rather than by observation: see below.

## The render path must never block

The menu has a hard send deadline (`UI_SEND_DEADLINE`) and the event loop already
emits lag warnings of 2–4s under label writes. So:

- A daemon thread (`ui_leaders.start`) refreshes every `UI_LEADERS_REFRESH_SEC`.
- `ui_leaders.get_cached()` returns the last completed snapshot and never waits.
- Before the first refresh lands, the panel shows "лидеры дня ещё не загружены".
- A ticker outage leaves the previous rows plus a staleness note; the refresher
  swallows its own exceptions so a Binance blip cannot kill the UI thread.
- The refresher is deliberately NOT folded into the 5-second snapshot keeper,
  which exists precisely to keep disk and network off the hot path.

Rejection reasons come from one in-memory dict written in `botlog.log_blocked` —
the single choke point every gate passes through. Hooking anywhere else would
cover some gates and silently miss others, and no disk read happens on any path.

## Verification

`test_ui_leaders.py` — 17 tests. The ones that matter:

- ranking is by move and not by close, proven on a fixture where the
  lower-closing coin must rank first;
- non-watchlist symbols excluded; a zero open price skipped rather than divided by;
- held coins carry no reason, rejected coins carry the gate, reasons older than
  an hour are dropped as describing a market that no longer exists;
- the recorder is hooked into `log_blocked` specifically;
- the menu reads `get_cached()` and never calls `compute_leaders`, so no test can
  pass while the render path fetches;
- an empty cache renders a message rather than raising, and the flags line
  degrades to "flags unavailable" instead of taking the whole panel down.

Live check after restart: `daily-leaders refresher started (every 60s)` in the
log, and the rendered panel above reproduced against the loaded config.

## Known gap

`_LAST_BLOCK` is in-memory, so after a restart the leaders show no rejection
reasons until each coin is evaluated again — a minute or two in practice. Making
it survive restarts means persisting it, and a disk read on this path is exactly
what the design forbids; the gap is left open deliberately rather than traded for
that risk.
