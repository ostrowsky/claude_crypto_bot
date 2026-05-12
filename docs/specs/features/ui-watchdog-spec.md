# UI watchdog — guarantee menu responsiveness under heavy load

- **Slug:** `ui-watchdog`
- **Status:** shipped 2026-05-06 v2.14.0
- **Created:** 2026-05-06
- **Owner:** core
- **Related:** ICPUSDT case 2026-05-06 (user `/start` not consumed because
  monitoring_loop saturated event loop).

---

## 1. Problem

Bot.py runs Telegram UI polling AND `monitoring_loop` (heavy analysis,
data_collector, impulse_scanner) in the **same asyncio event loop**.
When monitor's hot path holds CPU without yielding, polling task can't
fetch updates from Telegram, and:

- `/start`, `/menu`, callback buttons don't respond
- Bot's process is alive (sends SELL signals from background tasks)
  but UI is dead
- Pending updates pile up at Telegram side
- Manual restart required to recover

Real case 2026-05-06: `/start` from chat_id 179184487 sat in queue
unconsumed, `pending_update_count=1` from `getWebhookInfo`,
bot PID 14788 alive ~14h.

## 2. Success metric

**Primary:** stuck-polling MTTR (mean-time-to-recovery) drops from
"requires manual restart" to **≤ 90+30×3 = 180 sec** (3 watchdog warnings
+ self-exit + wrapper relaunch).

**Acceptance:** within 7 d:
- 0 manual restarts of bot for "UI not responding"
- ≥ 1 watchdog auto-restart logged (proves detection works)
- Menu response latency on `/start` ≤ 2 sec under normal load

## 3. Scope

### In scope
- New global `_last_update_processed_at` timestamp in bot.py
- `TypeHandler(Update, _watchdog_ping)` at group=-1 (highest priority)
  pings timestamp on EVERY incoming update
- Background `_ui_watchdog` task: every 30 s checks idle time +
  Telegram pending-update-count
- Force `os._exit(2)` after N warnings → wrapper restarts process
- Config flags for thresholds

### Out of scope
- Splitting bot.py UI from monitor.py into separate processes (bigger
  refactor — Phase 2 of architecture cleanup)
- Webhook mode (requires public URL + extra infrastructure)
- Yield points in monitoring_loop (covered by separate spec if needed)

## 4. Behaviour / design

```python
# files/bot.py globals
_last_update_processed_at: float = time.time()

# Highest-priority handler — pings on every update
async def _watchdog_ping(update, ctx):
    global _last_update_processed_at
    _last_update_processed_at = time.time()
app.add_handler(TypeHandler(Update, _watchdog_ping), group=-1)

# Background task scheduled in _post_init
async def _ui_watchdog():
    warn_count = 0
    while True:
        await asyncio.sleep(30)
        idle = time.time() - _last_update_processed_at
        if idle < WARN_THRESHOLD_SEC: warn_count = 0; continue
        # Idle for >= threshold — check if Telegram has pending
        pending = await getWebhookInfo().pending_update_count
        if pending > 0:
            warn_count += 1
            log.warning("idle=%.0fs pending=%d warn=%d/%d", ...)
            if warn_count >= FORCE_EXIT_AFTER_WARNS:
                os._exit(2)  # wrapper relaunches
        else:
            warn_count = max(0, warn_count - 1)  # quiet period, decay
```

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `UI_WATCHDOG_WARN_THRESHOLD_SEC` | 90.0 | Min idle before checking pending |
| `UI_WATCHDOG_FORCE_EXIT_AFTER_WARNS` | 3 | Warnings → force-exit |

**Rollback:** set `UI_WATCHDOG_FORCE_EXIT_AFTER_WARNS = 999` to disable
auto-exit (warnings still logged for diagnosis).

Time to first auto-restart on stuck polling:
`90s + 2 × 30s = 150s` (worst case 180s).

## 6. Risks

- **False-positive force-exit during legitimate quiet periods.**
  Mitigation: requires BOTH idle ≥ 90s AND pending updates > 0.
  Quiet bot at night = no users typing → no pending updates → no exit.
- **Force-exit during open trade.** Bot wrapper restarts; positions
  persist in `positions.json`; on restart `monitoring_loop` resumes
  with same trail stops. No trade lost.
- **Restart loop** if root cause not fixed and warns immediately
  re-trigger after restart. Mitigation: log + manual investigation
  if 3+ auto-restarts within 1 h.

## 7. Verification

- [x] Spec written.
- [x] Code in bot.py: global timestamp, TypeHandler ping, watchdog task.
- [x] Config flags in config.py.
- [ ] Bot restart applies the watchdog.
- [ ] Smoke: artificially block event loop with `time.sleep(120)` in some
      handler; observe watchdog logs warning + force-exit.
- [ ] 7-d live: count auto-restarts in logs.

## 8. Follow-ups

- **UI/monitor process split** (large refactor): run bot.py for UI ONLY,
  monitor.py as separate `python files/monitor.py` daemon. Communicate
  via positions.json + .runtime/. Eliminates root cause.
- **Yield points in monitoring_loop** hot path (`await asyncio.sleep(0)`
  every N symbols).
- **Watchdog metric** in `report_metrics_daily.py`: count auto-restarts
  per week as a health KPI.
