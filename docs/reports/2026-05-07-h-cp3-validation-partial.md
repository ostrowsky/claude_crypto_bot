# H-CP3 partial validation · ZigZag-gated WEAK suppression

**Status:** PARTIAL — directionally positive but klines coverage limits
statistical power.

**Window:** 108 d events; klines cache 30 d → only recent 23 events
coverable.

## Result

```
Total slow-trend WEAK exits on profitable trades: 331
No klines available (>30 d ago):                  308 (93%)
Coverable (last ~30 d):                            23

Of 23 coverable:
  WOULD suppress (ZigZag still active):  11
  Keep exit (no ZigZag uptrend):         12

Of 11 would-suppress (with EOD data):
  Avg delta if held:                  +1.45 pp
  Positive delta share:               8/11 (73%)
  Aggregate gain:                     +16.0 pp
```

## Top wins

| Date | Symbol | Mode | pnl_exit | EOD | Delta |
|------|--------|------|---------:|----:|------:|
| 2026-06-11 07:06 | ZROUSDT | alignment/15m | +2.38 % | +10.02 % | +7.64 pp |
| 2026-06-07 08:53 | TIAUSDT | alignment/15m | +0.73 % | +5.75 % | +5.02 pp |
| 2026-06-11 01:00 | DOGSUSDT | alignment/15m | +0.48 % | +4.33 % | +3.85 pp |
| 2026-06-13 12:01 | ILVUSDT | alignment/1h | +2.11 % | +4.89 % | +2.78 pp |
| 2026-06-07 08:55 | ATOMUSDT | trend/15m | +1.72 % | +4.51 % | +2.78 pp |

## Worst losses

| Date | Symbol | Mode | pnl_exit | EOD | Delta |
|------|--------|------|---------:|----:|------:|
| 2026-06-09 12:15 | INJUSDT | alignment/15m | +1.55 % | −3.85 % | −5.40 pp |
| 2026-05-31 00:31 | BATUSDT | trend/15m | +0.92 % | −2.47 % | −3.38 pp |

## Per-mode breakdown

| Mode/TF | WEAK exits | ZZ active | Avg delta |
|---------|-----------:|----------:|----------:|
| alignment/15m | 124 | 6 | **+1.96 pp** |
| alignment/1h | 26 | 2 | +1.79 pp |
| trend/15m | 81 | 3 | +0.20 pp |
| trend/1h | 41 | 0 | n/a |
| retest/15m | 37 | 0 | n/a |
| strong_trend/15m | 9 | 0 | n/a |

`alignment` modes are the main beneficiaries (matches design intuition —
alignment is the slow-trend catcher).

## Verdict

**PARTIAL PASS** — directional support but small sample.

**Recommendation:**
1. Ship with `H_CP3_ZIGZAG_GATED_WEAK_ENABLED=False` (default), then
   `H_CP3_SHADOW=True` for 7 d.
2. Add peer-breadth component later (requires per-symbol cluster map
   AND peer klines coverage — not feasible until klines backfill
   expanded to 90+ d).
3. Re-run backtest after 60 d more klines accumulated.

Net est: smaller than H-CP1. Apply mainly to `alignment` modes for now.

## Limitations

- **Klines coverage 30 d** drops effective backtest period from 108 d to
  ~25 d. Cannot validate older events.
- **No peer-breadth component tested** — would need pre-computed
  cluster map per (date, sym). Defer.
- **EOD proxy overstates continuation** (same caveat as H-CP1).
- **Worst case INJUSDT −5.40 pp** is a single bad case but real risk.
