# Always-watch pump detector

- **Slug:** `always-watch-pump-detector`
- **Status:** shipped 2026-05-06 v2.15.0
- **Created:** 2026-05-06
- **Owner:** core
- **Related:** STRKUSDT 2026-05-06 silent miss case (pump started during
  bot's quiet-coin exclusion window).

---

## 1. Problem

Bot scans only `hot_coins` subset (30-40 of 105 watchlist) between full
re-scans (`AUTO_REANALYZE_SEC = 7200`, 2 h). Coin that goes from quiet
→ pump in this window is **NOT in scan-set** → bot doesn't see it until
next full re-scan.

Real case STRKUSDT 2026-05-06:
- Last STRK event: May 4 04:26 (then quiet for 2 days)
- Dropped from `hot_coins` after subsequent re-scans (no signal_now)
- May 6 ~07:00 UTC: pump start (+15% by 11:30)
- Bot's last re-scan: May 6 05:58 (BEFORE pump)
- Bot's next re-scan would have been May 6 ~07:58 — but bot got stuck
  (separate UI watchdog issue, fixed in v2.14.0)
- **By the time bot recovers, ~70-90% of move already done**

Even without UI lock-up, 2-h gap is too long for fast crypto pumps.

## 2. Success metric

**Primary:** time-to-first-signal on a coin entering top-20 EOD drops
by ≥ 60 % vs current (current: 4.94 h median per E1 metric).

**Secondary:**
- 0 silent-miss "STRK-class" cases per week (where coin pumps >5 % within
  1 h and bot has 0 events on it during the move)
- Number of pump-injections per day logged + visible

## 3. Scope

### In scope
- Background task in bot.py (next to `_auto_reanalyze`, `_ui_watchdog`)
- Polls Binance `/api/v3/ticker/24hr` once per `PUMP_DETECTOR_INTERVAL_SEC`
  (default 300 s = 5 min). Single API call returns all symbols.
- Tracks **rate-of-change** of `priceChangePercent` per watchlist coin
  in memory: last N snapshots ring buffer.
- If a watchlist coin's price jumped > `PUMP_TRIGGER_PCT` in last
  ≤ `PUMP_LOOKBACK_MIN` minutes → inject into `state.hot_coins` if not
  already there.
- Logs `PUMP DETECTED %s: dx=+%.2f%% in %.0fmin` for audit.

### Out of scope
- Auto-entry on pump (skill writes recommendation; bot's existing
  pipeline decides).
- 1m polling per coin (too aggressive, rate-limit risk).
- Detector for downward moves (we only signal BUY).

## 4. Behaviour / design

### Algorithm

```python
SNAPSHOT_RING = {}  # sym -> deque[(ts, priceChangePct)] last 6 entries (~30min)

async def pump_detector_loop():
    interval = config.PUMP_DETECTOR_INTERVAL_SEC  # 300
    trigger = config.PUMP_TRIGGER_PCT             # 2.0
    lookback_min = config.PUMP_LOOKBACK_MIN       # 15
    while True:
        await asyncio.sleep(interval)
        try:
            snap = await fetch_24hr_ticker()    # 1 API call
            now = time.time()
            for entry in snap:
                sym = entry["symbol"]
                if sym not in WATCHLIST: continue
                cur = float(entry["priceChangePercent"])
                ring = SNAPSHOT_RING.setdefault(sym, deque(maxlen=6))
                ring.append((now, cur))
                # Compute delta within lookback window
                cutoff = now - lookback_min * 60
                old = next((p for ts, p in ring if ts >= cutoff), cur)
                dx = cur - old
                if dx >= trigger and sym not in {r.symbol for r in state.hot_coins}:
                    inject_pump_candidate(sym, dx)
        except Exception as e:
            log.warning("pump_detector loop error: %s", e)


def inject_pump_candidate(sym, dx):
    log.info("PUMP DETECTED %s: dx=+%.2f%% in %d min — injecting", sym, dx, lookback_min)
    # Build dummy CoinReport with signal_now=True so monitoring_loop scans it
    state.hot_coins.append(_build_pump_coin_report(sym))
    # Persist event for offline audit
    botlog.log_pump_detection(sym, dx)
```

### Why ticker/24hr not 5m klines

- 1 API call covers all 105 coins (vs 105 calls if doing per-coin klines)
- Binance returns `priceChangePercent` (24h) — relative, comparable
- Δ between snapshots = recent acceleration, perfect pump signature

## 5. Config flags & rollback

| Flag | Default | Effect |
|------|---------|--------|
| `PUMP_DETECTOR_ENABLED` | True | Master switch |
| `PUMP_DETECTOR_INTERVAL_SEC` | 300 (5 min) | How often to poll ticker |
| `PUMP_TRIGGER_PCT` | 2.0 | Min Δ in priceChangePct over lookback |
| `PUMP_LOOKBACK_MIN` | 15 | Window for delta calculation |
| `AUTO_REANALYZE_SEC` | **1800** (was 7200) | Full re-scan more often |

**Rollback:** `PUMP_DETECTOR_ENABLED=False`. Reverts to legacy 30 min
re-scan only.

## 6. Risks

- **False-positive injections** (coin moves 2% on noise, not real pump)
  → bot scans extra coin, no harm done; just wasted API calls.
- **Duplicate injections** if pump persists across snapshots
  → check `sym not in hot_coins` before injecting.
- **Memory growth in SNAPSHOT_RING** if running for months
  → ring is bounded (maxlen=6 per coin × 105 coins = 630 entries max).
- **API rate limit** (Binance: 1200 req/min per IP)
  → 1 ticker/24hr per 5 min = 12/hour. Negligible.

## 7. Verification

- [x] Spec written.
- [ ] Code in bot.py: `pump_detector_loop` task, registered in `_post_init`.
- [ ] Config flags in config.py.
- [ ] Smoke test: artificially set ring buffer to trigger pump on
      a known coin → log line emitted.
- [ ] 7-d live: count `PUMP DETECTED` log lines vs how many of those
      coins later appear in entry events.

## 8. Follow-ups

- Tune `PUMP_TRIGGER_PCT` after 7 d data: too sensitive vs too lazy.
- Cross-correlate with skill weekly digest: how many late entries
  from pre-pump-detector era would have been on-time post-detector.
- Phase 2: add `DUMP_DETECTOR` for symmetric protection (avoid entering
  coins in active dump).
