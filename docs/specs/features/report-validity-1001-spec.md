# Morning report: no invalid statements (2026-10-01)

- **Slug:** `report-validity-1001`
- **Status:** SHIPPED 2026-10-01 (operator: «исправь отчёт, чтобы невалидная информация в нём не содержалась»)
- **Truth harness:** TH-02 (a number under the wrong name), TH-03 (leaky labels
  replaced by immutable ones), TH-04 (versioned, never redefined; no trend across
  a label change), TH-05, TH-13 (every line checked against its source)
- **Rollback:** revert the commit; the legacy values still travel inside every
  v2 payload (`legacy_*`), so nothing is lost
- **Maximum period / shadow:** not applicable — reporting only, no trading
  decision reads these fields

## Line-by-line audit of the 2026-10-01 report

| line | verdict | cause | fix |
|---|---|---|---|
| «МЕТРИКА ПРЕДВАРИТЕЛЬНАЯ», «тренд не оценивается до пересчёта на immutable labels» | **false** | scorecard wrote status `measured`, every consumer checked `verified` | `ns_ground_truth_verified()` — one definition (immutable provenance, not degraded) → status `verified` |
| «предварительный North Star» | **false** | same | word only when not verified |
| «⚠️ Ground truth пока provisional: rolling-24h» | **false** | provenance was `immutable_later_eod_klines` since 2026-08-17 | line only when not verified |
| red flag `RF_north_star_ground_truth_provisional` (critical) | **false** | same mismatch | disappears with the fix |
| step P0 «Создать неизменяемые later-EOD labels» | **false** | same | disappears with the fix |
| «из 10 событий watchlist∩global-top20 ~6 имели вход» | **mislabelled** | from `C1_C2_coverage_funnel` on `label_top20` (rolling-24h, rank INSIDE the watchlist): 57% of 116 | funnel v2 on immutable labels: **60 of 75 = 80%**, equal to the North Star coverage (0.80); a legacy funnel is never printed under that name |
| «Совсем не видит: каждое 23-е» | **mislabelled** | legacy funnel | v2: silent misses 0 of 75 → line absent |
| «precision сигналов 15.4%» | **leaky label** | `label_top20` | v2 immutable: **13.6%** (62 of 457 entries, 14 labelled days); legacy 15.2% kept as `legacy_precision_pct` |
| «время до сигнала 5.71 ч» | **wrong reference** | measured from the first daily SNAPSHOT with +2% — snapshots ~6 h apart | v2: first entry minus the first hourly close ≥ +2.5% (the North Star deadline): **+3.51 h**, 14% entered before (n=59) |
| «прибыль портфеля vs buy-and-hold −44.6%» | true by its definition, unexplained | equal-slot model of an alert bot without sizing | printed with its parts: bot −7.7% vs +37.3% holding the watchlist, window, "равные слоты (у бота нет размера позиций)" |
| «alert during cooldown — ❌ не помогло (просело: realert_rate)» | **false** | no expected metric was measured (all `insufficient_data`); the verdict came only from a portfolio guard: maxdd +0.1003 vs a 0.1000 limit | status `guard_only`: «целевые метрики не измерены; нарушено ограничение maxdd_abs_growth: +0.1003 при пределе 0.1000»; a miss counts only if measured |
| «bandit diagnostic …» | misleading | the bandit is OFF in the live path (`BANDIT_ENABLED = False`, ac33744) | labelled «бандит выключен в живом пути — на сигналы не влияет» |
| «training↔live gap −97% … согласованы» | **false** | the off bandit's recall (3%) minus live buys (100%) | not computed while the bandit is off: «не применимо» |
| step P0 «Восстановить полные рабочие дни» | not an action | past uptime cannot be restored; the window refills | removed (the uptime line stays; the verdict refuses incomparable windows) |
| step P0 «Рассчитать EX1 в ZigZag-mode» | not doable as written | recomputation cannot raise coverage: trades do not overlap ZigZag(4%) uptrends | P2 «Выбрать эталон потенциала для EX1» |
| step order P0, P2, P1; `13.566739606126914%` | presentation | — | sorted by priority, one decimal |

## The trend verdict

With the status fixed the headline judges the trend again — on rows whose North
Star was computed on immutable labels only (`_ns_history*` filter): a row on the
rolling-24h label is never an endpoint (TH-04). 2026-10-01: «СТОИТ НА МЕСТЕ —
2026-08-25..10-01: ~11 → ~11 из 100».

## Metrics v2

`D1_D2_precision_msgrate_v2`, `E1_time_to_signal_v2`, `C1_C2_coverage_funnel_v2`:
winners = immutable later-EOD top-20 of the global universe, then the watchlist
(`immutable_labels.winners_by_day(rank_before_filter=True)`) — the North Star's
denominator; days without labels are no data (TH-05). Each payload carries
`label_provenance` and its legacy value (`legacy_*`, `legacy_label_provenance =
rolling_24h_same_snapshot`). The report loader resolves `_v2` under the base name,
so the truth harness and every consumer keep working.

## Tests

`files/test_report_validity_1001.py` (11); `files/test_bot_health_report_integrity.py`
updated to the corrected semantics (+3 tests: no trend across a label change, no
gap for a bandit that is off, no legacy funnel under the global name) — 24 pass.
