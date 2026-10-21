# Demo trader — a separate application next to the signal bot

Goal: trade a Binance **demo** account by itself (buy, exchange-side TP/SL, sell) and maximise
**daily revenue** (`docs/specs/features/daily-revenue-metric-1010-spec.md`). Roadmap:
`docs/specs/features/autonomous-trading-roadmap-1010-spec.md`.

## Boundary with the signal bot (operator decision 2026-10-10)

- The trader **never modifies** the signal bot: no edits in `files/`, no flags in
  `files/config.py`, no writes to `.runtime/`, no interaction with the bot process.
- It **reads** only: `files/bot_events.jsonl` (candidates, the bot's entries/exits),
  `history/<sym>_15m*.csv` (klines), `files/.env` (`BINANCE_DEMO_API_KEY`,
  `BINANCE_DEMO_API_SECRET` only). The E1 replay imports bot modules (`exit_validator`,
  `config`) **read-only** via `trader_config.bot_readonly()` so that it reproduces the bot.
- It **writes** only `trader/runtime/` (gitignored).
- `test_demo_connection_a.TestSeparateApp` fails if a bot file imports a trader module, if
  `files/config.py` carries trader flags, or if a trader module writes to `.runtime`.

## Safety

- Signed requests go to demo hosts only: `exchange_client.ALLOWED_HOSTS` is a code constant.
  No setting enables mainnet. Real money = a new spec and an operator decision.
- No order is placed while `TRADING_ENABLED = False` (`trader_config.py`); phase A only reads
  and runs `order/test`.
- Keys are never printed or logged (`exchange_client.redact`).

## Layout

| file | role | spec |
|---|---|---|
| `trader_config.py` | paths, flags, read-only access to the bot | — |
| `exchange_client.py` | signed demo REST client, filters, host guard | phase A |
| `trading_universe.py` | demo USDT universe with exclusion reasons → `runtime/trading/universe.json` | A |
| `microstructure_recorder.py` | A0: spread, depth cost, demo-vs-live divergence → `runtime/microstructure/` | A |
| `equity_snapshots.py` | account equity in USDT → `runtime/trading/equity_snapshots.jsonl` | A / metric |
| `_smoke_demo_connection.py` | gate-A smoke test (read-only + order/test) | A |
| `fill_model.py` | fees, spread, slippage — shared by replay and paper | E1 / C |
| `trading_policy.py` | the one decision function (admission, sizing, exits, risk) | E1 / C |
| `daily_revenue.py` | DailyRevenue v1 aggregation (CI, compound, drawdown) | metric |
| `execution_replay.py` | E1 equity replay with V1–V3 self-checks | E1 |
| `_backtest_daily_revenue_baseline.py` | the baseline (−0.83%/day) | roadmap §0 |
| `run_recorder.cmd` | scheduled every 5 min (`CryptoBot_MicrostructureRecorder`) | A |
| `shadow_portfolio.py`, `shadow_readout.py` | E7 paper portfolios P0 vs P4 on live data, readout | E7 |
| `run_shadow.cmd` | scheduled every 5 min (`CryptoBot_TraderShadow`) | E7 |
| `trade_ledger.py` | D: demo-account fills from myTrades → `runtime/trading/ledger.jsonl` (idempotent, BNB fees in USDT, decision_id) | D |
| `daily_revenue.py compute` | D: day R from equity marks, P−L reconciliation, stablecoin drift apart, benchmarks → `runtime/metrics/daily_revenue.jsonl` | D |
| `trader_report.py` | D: operator report → `runtime/reports/YYYY-MM-DD.txt` | D |
| `run_daily.cmd` | scheduled 02:20 local (`CryptoBot_TraderDaily`); the report is posted to the Claude chat at 09:00 by Claude task `crypto-trader-daily-report` | D |
| `executor.py` | C: исполнитель книги X4 на демо (`TRADING_SCALE` 0.02), аварийный стоп, сверка, файлы управления `runtime/executor/control/` (PAUSE, FLATTEN, RESET_NOTED); `run` / `status` | C |
| `reconcile.py` | C: сверка счёта с позициями исполнителя | C |
| `_smoke_demo_orders.py` | C0: пробный прогон ордеров на демо (запускает оператор, `--go`) | C |
| `run_executor.cmd` | задача `CryptoBot_TraderExecutor` каждые 5 мин; `TRADING_ENABLED=False` → бумажный режим C1 | C |
| `history_store.py`, `run_backfill.cmd` | B1: свечи 15m всех спотовых пар с 2023 (вкл. делистнутые) → `runtime/history/` | B |
| `candidate_gen.py` | B2: входы по правилам бота (импорт только на чтение) на всём универсуме | B |
| `gen_calibration.py` | B2: сверка генератора с реальными входами бота (порог полноты 0.80) | B |
| `lazy_grid.py` | B3: экономная сетка баров (memmap по парам), паритет с плотной | B |
| `experiment_b3.py`, `run_b3_train.cmd` | B3: W93 / WALL / WNEW в деньгах; train 2023–2025, test 2026 один раз | B |
| `xsmom.py` | XSMOM: дневные бары, панель, универсум (100 самых ликвидных), ротация по моментуму, контроли (случайный выбор с тем же оборотом, весь рынок), сверка учёта с E1 | XSMOM |
| `experiment_xsmom.py` | XSMOM: `panel` / `train` (2023-03..2025-12) / `test` (2026, один раз, только с `--operator-approved`) | XSMOM |
| `experiment_s1.py` | S1: разворот (K = 10 худших за L дней, 16 кандидатов, возраст пары ≥ 60 дней); `train` (2023-03..2025-12) / `test` (2026, один раз на семейство вместе с XSMOM, только с `--operator-approved`) | S1 |
| `funding_store.py` | FC-0: фандинг (с `startTime`) и свечи 15m бессрочных фьючерсов живого рынка → `runtime/funding/`, `runtime/perp_history/`; сопоставление perp ↔ spot | FC |
| `carry.py`, `experiment_fc.py` | FC: кэрри «спот в лонг + perp в шорт» (плечо 1, изолированная маржа, маржинальный стоп, ликвидация, фандинг); `panel` / `train` / `test` (2026, один раз, только с `--operator-approved`) | FC |
| `demo_futures.py` | F0: оценка готовности демо-счёта к фьючерсам (GO / NO_GO, предупреждения) | FC |
| `audit.py`, `audit_report.py` | audit journals: every Binance request and every decision (`runtime/audit/{exchange,decisions}/`), summary | audit |
| `run_tests.py` | runs all `test_*.py` (embedded Python does not put cwd on sys.path) | — |

Run everything from `trader/`: `..\pyembed\python.exe <script>.py`.
