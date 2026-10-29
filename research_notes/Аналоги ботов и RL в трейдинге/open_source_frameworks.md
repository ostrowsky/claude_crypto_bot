# Open-source crypto trading bots and algorithmic trading frameworks: feature inventory for a GAP analysis (as of 2026-10-29)

Method note: the repository maturity figures (stars, last push, archived flag, licence, latest GitHub release) were pulled from the GitHub REST API (`gh api repos/<owner>/<repo>`) on 2026-10-29. Feature claims come from official docs and READMEs unless marked otherwise. "Documented" means the project's own docs describe the mechanism. "Marketing" means a landing-page claim with no mechanism described. 19 tool calls were used, so some projects are covered less deeply (see Gaps).

## 1. Maturity and maintenance status (who is alive in 2025-2026)

### Takeaway
Three projects are clearly alive and large: Freqtrade (56k stars, monthly releases), NautilusTrader (31k) and Hummingbot (21k, v2.17 in Sep 2026). Jesse, OctoBot, vectorbt, Passivbot, CCXT and Lean are also actively maintained. Zenbot, Gekko and Catalyst are archived. Blankly, Krypto-trading-bot and Backtrader are dormant: no push since 2024.

### Cited Findings
GitHub API snapshot, 2026-10-29 (stars / last push / archived / licence / latest GitHub release):
- **Freqtrade**: 56,126 / 2026-10-29 / no / GPL-3.0 / `2026.9` (2026-09-29). Calendar-versioned monthly releases. — [GitHub](https://github.com/freqtrade/freqtrade)
- **freqtrade-strategies** (community strategy repo): 5,603 / 2026-09-08 / GPL-3.0. — [GitHub](https://github.com/freqtrade/freqtrade-strategies)
- **Hummingbot**: 20,783 / 2026-10-20 / no / Apache-2.0 / `v2.17.0` (2026-09-22). — [GitHub](https://github.com/hummingbot/hummingbot)
- **Hummingbot Condor** (API/executor front end): 246 / 2026-10-28 / MIT. **Hummingbot quants-lab** (research): 301 / 2025-11-04. — [Condor](https://github.com/hummingbot/condor), [quants-lab](https://github.com/hummingbot/quants-lab)
- **NautilusTrader**: 31,024 / 2026-10-29 / no / LGPL-3.0 / `v1.231.0` (2026-08-02). — [GitHub](https://github.com/nautechsystems/nautilus_trader)
- **QuantConnect Lean**: 22,291 / 2026-10-29 / no / Apache-2.0. Its latest GitHub "release" is from 2017 because Lean does not publish through GitHub releases, so pushes are the activity signal. — [GitHub](https://github.com/QuantConnect/Lean)
- **Jesse**: 8,734 / 2026-10-26 / no / MIT / no GitHub releases (distributed via PyPI). — [GitHub](https://github.com/jesse-ai/jesse), [PyPI](https://www.pypi.org/project/jesse/)
- **OctoBot**: 6,872 / 2026-10-27 / no / GPL-3.0 / `3.0.0-beta3` (2026-10-08). — [GitHub](https://github.com/Drakkar-Software/OctoBot)
- **Superalgos**: 5,741 / last push 2026-10-27 / no / Apache-2.0 / latest release `v.1.6.1` (2024-11-02). A third-party tracker reports superalgos.org offline since 2026-09-29 (not confirmed against the project). No 2025-2026 development news found. — [GitHub](https://github.com/Superalgos/Superalgos), [Blockspot](https://blockspot.io/de/coin/superalgos-sa/)
- **vectorbt** (open-source community edition): 9,584 / 2026-10-28 / licence NOASSERTION (custom) / `v1.2.0` (2026-10-19). — [GitHub](https://github.com/polakowo/vectorbt)
- **Passivbot**: 2,148 / 2026-10-29 / Unlicense / `v8.1.0` (2026-08-10). — [GitHub](https://github.com/enarjord/passivbot)
- **CCXT** (library, not a bot): 44,617 / 2026-10-29 / MIT / `v4.5.90` (2026-10-23). — [GitHub](https://github.com/ccxt/ccxt)
- **Backtrader**: 23,613 / last push 2024-08-19 / not archived / GPL-3.0 / no releases. Effectively dormant. — [GitHub](https://github.com/mementum/backtrader)
- **Zenbot**: 8,255 / last push 2022-02-14 / **archived** / MIT / last release 4.1.4 (2018). — [GitHub](https://github.com/DeviaVir/zenbot)
- **Gekko**: 10,180 / last push 2020-02-16 / **archived** / MIT / last release v0.6.8 (2018). — [GitHub](https://github.com/askmike/gekko)
- **Catalyst** (Enigma/scrtlabs, Zipline-based): 2,559 / last push 2022-11-26 / **archived** / Apache-2.0. — [GitHub](https://github.com/scrtlabs/catalyst)
- **Blankly**: 2,483 / last push 2024-12-30 / LGPL-3.0 / last release `v1.18.0-beta` (2022-06-15). Dormant. — [GitHub](https://github.com/blankly-finance/blankly)
- **Krypto-trading-bot** (C++ market maker): 3,728 / last push 2024-12-15 / release `0.7.x` (2024-09-17). Dormant since end of 2024. — [GitHub](https://github.com/ctubio/Krypto-trading-bot)
- Other notable active projects (2024-2026):
  - **FinRL** (DRL for finance, incl. crypto): 16,859 / 2026-10-25 / MIT / v0.3.8 (2026-03-20). — [GitHub](https://github.com/AI4Finance-Foundation/FinRL)
  - **Microsoft Qlib** (ML quant research platform, equities-centric): 49,808 / 2026-10-22 / MIT / v0.9.8 (2026-10-08). — [GitHub](https://github.com/microsoft/qlib)
  - **backtesting.py**: 9,157 / 2026-08-05 / AGPL-3.0. — [GitHub](https://github.com/kernc/backtesting.py)
  - **PyBroker** (ML-oriented backtester with walk-forward): 3,666 / 2026-10-29 / v2.0.2 (2026-10-20). — [GitHub](https://github.com/edtechre/pybroker)
  - **Haehnchen/crypto-trading-bot** (Node.js, CCXT-style multi-exchange): 3,543 / 2026-08-02 / MIT. — [GitHub](https://github.com/Haehnchen/crypto-trading-bot)
  - **barter-rs** (Rust event-driven trading engine): 2,360 / 2026-08-24 / MIT. — [GitHub](https://github.com/barter-rs/barter-rs)
  - **OpenTrader** (grid/DCA bot with UI): 2,913 / last push 2025-06-29 / v1.0.0-beta.29 (2025-03). Slowing. — [GitHub](https://github.com/Open-Trader/opentrader)
  - **StockSharp** (C# platform, crypto connectors): 11,021 / 2026-10-25. — [GitHub](https://github.com/StockSharp/StockSharp)
  - **TensorTrade** (RL trading environments): 7,254 / 2026-02-19 / Apache-2.0 / v1.0.4 (2026-02-06). — [GitHub](https://github.com/tensortrade-org/tensortrade)
  - **FinRL-Meta** (market environments for FinRL): 1,955 / 2026-10-12 / MIT / last release v0.3.6 (2022). — [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta)
  - **bbgo** (Go crypto trading framework, grid/MM strategies): 1,675 / 2026-10-29 / AGPL-3.0 / bbgo-0.4.6 (2026-04-14). — [GitHub](https://github.com/c9s/bbgo)
  - **OpenBB** (data/research platform, not a bot): 74,855 / 2026-10-24. The API returned the owner as `openbq-org`, probably a redirect; verify. — [GitHub](https://github.com/OpenBB-finance/OpenBB)
  - **WolfBot** (TypeScript, multi-exchange): 788 / last push 2023-02-17. Dormant. — [GitHub](https://github.com/Ekliptor/WolfBot)
- Gekko successor repos (`Gekko-Plus/gekko-plus`, `3rd-Eden/gekko-x`) returned 404 on GitHub. Those names no longer exist. — GitHub API 404, 2026-10-29.

### Inferences
- The live open-source landscape has consolidated around Freqtrade (indicator/ML bots), Hummingbot (market making and executors), NautilusTrader (institutional-grade event-driven engine) and Lean (multi-asset). The 2017-2019 Node.js generation (Gekko, Zenbot) and the Zipline-based Catalyst are dead.
- Superalgos shows recent pushes but no release in nearly two years and a reported site outage. Treat it as "at risk / low activity", not as maintained.

### Gaps
- Commit counts, contributor counts and issue velocity were not pulled, so "maintenance" is judged only from last push and last release.
- vectorbt PRO is a private, paid repository. Its activity cannot be measured from public GitHub.

## 2. Strategy families supported out of the box

### Takeaway
No single framework covers every family. Freqtrade is the strongest for indicator, ML and RL strategies on candles. Hummingbot owns market making, cross-exchange MM, arbitrage, grid, DCA and TWAP as executors. Passivbot is a specialised martingale/grid market maker for perpetuals. OctoBot covers grid, DCA, TradingView signals, LLM-driven and index baskets. NautilusTrader, Lean, Jesse, Backtrader and vectorbt are frameworks: they ship engines, not trading families.

### Cited Findings
- **Freqtrade**: user-written indicator strategies (`populate_indicators` / `populate_entry_trend` / `populate_exit_trend`), spot plus futures with shorting and `leverage()` callbacks, and FreqAI for supervised ML and RL. — [Lookahead analysis docs](https://www.freqtrade.io/en/stable/lookahead-analysis/), [FreqAI RL docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)
  - Pair selection is a strategy layer of its own: VolumePairList, PercentChangePairList, MarketCapPairList (CoinGecko rank), RemotePairList (URL/JSON), ProducerPairList (consumer mode), and filters including AgeFilter, DelistFilter, SpreadFilter, VolatilityFilter, RangeStabilityFilter and PerformanceFilter. Several of these are "not available in backtesting" (VolumePairList, PercentChangePairList, DelistFilter, PerformanceFilter, FullTradesFilter). — [Freqtrade plugins](https://www.freqtrade.io/en/stable/plugins/)
- **FreqAI RL** (stable-baselines3 + gym): `model_type` picks an SB3 model, for example PPO with `MlpPolicy`. Three environment types:
  - Base3ActionRLEnv: hold / long / short. Long-only is possible with `can_short=False`.
  - Base4ActionEnvironment: enter long, enter short, hold, exit.
  - Base5ActionEnvironment: as Base4, with separate exit-long and exit-short.
  - The reward is user-defined by overriding `calculate_reward()`. The built-in reward is "a benchmark" and "_not_ for live production."
  - `continual_learning=true` continues from the previous model, otherwise each retrain starts from scratch. `train_cycles` sets the number of passes. Tensorboard logging is built in.
  - Caveats in the docs: the RL environment does not include `custom_exit`, `custom_stoploss` or leverage controls. "Improperly trained RL agents may find 'cheats' and 'tricks' to maximize reward without actually winning any trades." State info (current profit, position, trade duration) is used in dry/live but is **not available in backtesting**, which is a documented train/backtest/live mismatch.
  - Source: [FreqAI RL docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)
- **Hummingbot**: V2 framework (recommended) = Scripts + Controllers (multiple per bot via `v2_with_controllers.py`) + Executors (self-managing, finite workflows that can be launched via the Hummingbot API without a running script). The V1 framework is legacy but "still supported". New development focuses on V2. An LP Executor and an `lp_rebalancer` controller exist for DEX LP. — [Hummingbot strategies](https://hummingbot.org/strategies/), [V2 strategies](https://hummingbot.org/strategies/v2-strategies/)
  - PositionExecutor uses a triple barrier (stop loss, take profit, time limit) plus trailing stop. On spot it places TP/SL orders after the fill, and on perpetuals it manages the position. — [PositionExecutor](https://hummingbot.org/v2-strategies/executors/positionexecutor)
  - XEMMExecutor does cross-exchange market making. — [XEMM executor](https://hummingbot.org/strategies/v2-strategies/executors/xemm-executor/)
  - grid_executor ("Grid trading with multiple levels") and dca_executor ("multiple entry points") are listed in the Hummingbot API reference. — [Condor API reference](https://condor.hummingbot.org/api-reference/executors/create-executor.md)
  - Avellaneda-Stoikov market making (V1): orders around a reservation price computed from inventory, volatility and session time, plus an optimal spread. The current version replaced `inventory_risk_aversion` with `risk_factor` and removed several parameters. Hanging orders keep the opposite side alive after a fill, with a default 10 s refill delay. — [Avellaneda docs](https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/), [advanced MM features](https://docs.hummingbot.org/academy-content/how-to-use-advanced-market-making-features/)
- **Passivbot**: perpetual-futures "contrarian market maker".
  - The default `trailing_martingale` strategy enters small and adds as price moves against the position. `ema_anchor` is also available. `trailing_grid_v7` is deprecated.
  - Entries and closes can trail.
  - "Forager" picks markets by relative volume, quote volume, EMA readiness and 1m log-range volatility.
  - "Unstuck" realises small losses on stuck positions, capped against peak balance.
  - It explicitly "does not predict prices or follow trends."
  - Source: [Passivbot README](https://github.com/enarjord/passivbot)
- **OctoBot**: grid, DCA, an AI trading mode (OpenAI models or local Ollama), TradingView alert execution, crypto baskets/indexes, and social indicators (Google Trends, Reddit) as well as technical ones. Market making lives in a separate project or hosted platform. Copy trading is not mentioned in the README. — [OctoBot README](https://github.com/Drakkar-Software/OctoBot)
- **Jesse**: user-coded strategies (multi-symbol, multi-timeframe, futures, leverage, short) and an end-to-end ML pipeline. It collects labelled data from backtests, trains scikit-learn classifiers or regressors, and deploys their predictions in strategies. RL workflows are "coming soon" (not available). — [Jesse docs](https://docs.jesse.trade/)
- **NautilusTrader**: a framework with strategy and exec-algorithm abstractions, no packaged trading families. It runs one code path for backtest and live. — [Nautilus overview](https://nautilustrader.io/docs/latest/concepts/overview)

### Inferences
- **Copy-trading** is not a first-class feature in any project reviewed. OctoBot's TradingView/signal mode is the nearest equivalent.
- **Triangular arbitrage**: no current documentation was retrieved. Hummingbot's V1 had arbitrage strategies, and V2 has an arbitrage executor by name (see search results), but its details were not verified.
- **Early-momentum/top-gainer detection** (this repo's North Star) is not a family in any of these projects. The closest are Freqtrade's PercentChangePairList and Passivbot's Forager, which rank by volume, volatility and change but do not predict end-of-day leaders.

### Gaps
- Hummingbot GridExecutor, DCAExecutor, ArbitrageExecutor and TWAPExecutor configuration pages were not retrieved (404 on the first URL tried).
- The full Hummingbot V1 list (pure MM, cross-exchange MM, AMM arb, perpetual MM, spot-perp arb) was not verified on a current page, and deprecation dates were not found.
- OctoBot 3.0's specific changes are not described in the README beyond "node mode" and a beta interface.

## 3. Backtesting, optimisation and overfitting protection

### Takeaway
Freqtrade has the most explicit bias tooling: lookahead-analysis, recursive-analysis and Optuna NSGA-III hyperopt with 12 loss functions. It has **no documented walk-forward** and **no slippage model**. Jesse has Optuna with train/test periods, Monte Carlo and bootstrap rule-significance tests. vectorbt PRO documents walk-forward and purged/combinatorial CV. NautilusTrader and Lean have event-driven fidelity, but their crypto slippage defaults are null in Lean's case.

### Cited Findings
- **Freqtrade lookahead-analysis**: chained backtests that alter indicator values and entry/exit timing, rather than reading the code.
  - Method: a baseline backtest on all pairs, then a per-signal verification backtest on a single-pair list, then a column-by-column dataframe comparison.
  - Output: `has_bias`, `biased_entry_signals`, `biased_exit_signals`, `biased_indicators`. Default 20 signals checked.
  - It forces cache off, protections off, a fixed 10k stake and market orders.
  - Caveats: it only checks signals that actually fire, so false negatives are possible. Pairlist-dependent logic, limit orders with custom prices, and FreqAI targets can be false positives.
  - It runs from freqUI as a background task.
  - Source: [Lookahead analysis](https://www.freqtrade.io/en/stable/lookahead-analysis/)
- **Freqtrade recursive-analysis**: recomputes indicators with different startup-candle counts (199/499/999/1999) against a full-range benchmark and reports the % variance on the last row. This catches EMA-type warm-up dependence. It only covers `populate_indicators` and `@informative`, and does not backtest. — [Recursive analysis](https://www.freqtrade.io/en/stable/recursive-analysis/)
- **Freqtrade hyperopt**: uses `optuna` ("currently NSGAIIISampler").
  - Loss functions: ShortTradeDur, OnlyProfit, Sharpe, SharpeDaily, Sortino, SortinoDaily, Calmar, MaxDrawDown, MaxDrawDownRelative, MaxDrawDownPerPair, ProfitDrawDown, MultiMetric.
  - Warnings: precision finer than 3 decimals "will usually result in overfitted results", and gains plateau after 500-1000 epochs.
  - **No walk-forward is described.**
  - Source: [Hyperopt docs](https://www.freqtrade.io/en/stable/hyperopt/)
- **Freqtrade backtest assumptions** (documented):
  - Fills happen at the requested price with **no slippage** if within the candle's high/low.
  - Stoploss exits at the stop price even if the low was lower, with a loss 2×fees over the stop.
  - The low is assumed to occur before the high. Order within a candle: exit-signal, stoploss, ROI, trailing.
  - Stoploss exits can appear more often than in dry/live.
  - `--timeframe-detail` simulates intra-candle moves on a lower timeframe.
  - Protections run in backtest only with `--enable-protections`.
  - Source: [Backtesting docs](https://www.freqtrade.io/en/stable/backtesting/), [plugins](https://www.freqtrade.io/en/stable/plugins/)
- Independent commentary (informal, not peer-reviewed): Freqtrade's candle-only backtester has to assume the intra-candle path, which can inflate results on longer timeframes ([Backtesting traps](https://brookmiles.github.io/freqtrade-stuff/2021/04/12/backtesting-traps/)). A 2026 review names hyperopt as the main overfitting risk and cites GitHub issues where hyperopt and backtest diverge on identical settings ([backtestscore.com](https://www.backtestscore.com/tools/freqtrade)).
- **Jesse**:
  - Optuna parameter search, parallelised with Ray, with separate training and testing periods.
  - Monte Carlo (trade-order shuffling and candle-based simulations).
  - Bootstrap "rule significance testing".
  - Multi-timeframe/multi-symbol backtests "without look-ahead bias" (claim). Rust-powered indicators. Partial fills.
  - Source: [Jesse docs](https://docs.jesse.trade/)
- **vectorbt PRO** (proprietary successor to open-source vectorbt): walk-forward optimisation, purged and combinatorial cross-validation, robustness checks, split analytics, and portfolio simulation that reconstructs orders, trades, positions and drawdowns. — [vectorbt.pro](https://vectorbt.pro/), [getting started](https://vectorbt.pro/getting-started), [community README](https://cdn.jsdelivr.net/gh/polakowo/vectorbt@master/README.md)
- **Passivbot**: a shared Rust orchestrator plans orders for both live trading and backtesting. The optimizer runs thousands of backtests with an evolutionary algorithm. — [Passivbot README](https://github.com/enarjord/passivbot)
- **QuantConnect Lean crypto models**: Bybit uses `NullSlippageModel` (no slippage modelled). Binance and Coinbase orders "do not experience slippage in backtests and QuantConnect Paper Trading". Live market orders larger than the top of book fill against the actual book. — [Lean Bybit models](https://www.quantconnect.com/docs/v2//writing-algorithms/reality-modeling/brokerages/supported-models/bybit), [Binance brokerage](https://www.quantconnect.com/docs/v2/our-platform/live-trading/brokerages/binance), [Coinbase brokerage](https://quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/coinbase)
- **OctoBot**: backtesting is available only in "manual" OctoBot instances. No optimizer is described. — [OctoBot README](https://github.com/Drakkar-Software/OctoBot)
- Academic context: the Probability of Backtest Overfitting (PBO, Bailey et al., via CSCV) is the standard overfitting metric ([arXiv 1910.05555](https://arxiv.org/pdf/1910.05555)). A crypto DRL paper argues that a single validation split is insufficient and proposes overfitting tests for DRL agents ([arXiv 2209.05559](https://arxiv.org/pdf/2209.05559)).

### Inferences
- Of the frameworks reviewed, **none ships PBO/CSCV as a default report**. vectorbt PRO's combinatorial CV is the closest. This repo already computes PBO for its trader hypotheses, which is ahead of the open-source defaults.
- Freqtrade's bias tools are about **code-level leakage** (shift(-n), non-rolling stats, warm-up). They do not cover **label leakage** of the kind this repo found in `tg_return_since_open`, and they do not cover selection bias across many strategies.
- Default fill models in Freqtrade and Lean are optimistic (no slippage). Any backtest edge taken from them should be haircut. NautilusTrader's order-book-delta data allows more realistic fills, but its fill-model specifics were not retrieved.

### Gaps
- NautilusTrader fill, slippage and latency models: the "Behavioral Models" docs page was not fetched.
- Hummingbot backtesting capability (quants-lab, the V2 backtesting engine) was not verified.
- Whether Jesse's Monte Carlo includes slippage perturbation was not verified.

## 4. Risk management

### Takeaway
Freqtrade has the richest documented protection layer: StoplossGuard, MaxDrawdown, LowProfitPairs and CooldownPeriod, plus spread, volatility and age filters. Hummingbot works per executor with a triple barrier. Passivbot manages exposure and martingale risk with "unstuck" and an equity hard stop. OctoBot and Jesse describe risk only generically.

### Cited Findings
- **Freqtrade protections**:
  - StoplossGuard: halts after N stoploss exits in a window, globally or per pair/side.
  - MaxDrawdown: `ratios` or `equity` calculation mode.
  - LowProfitPairs: locks pairs whose recent profit is below a target.
  - CooldownPeriod: locks a pair after an exit.
  - Common settings: `stop_duration(_candles)`, `lookback_period(_candles)`, `trade_limit`, and `unlock_at` (daily HH:MM).
  - PrecisionFilter removes pairs where rounding makes a proper stoploss impossible.
  - Source: [Freqtrade plugins](https://www.freqtrade.io/en/stable/plugins/)
- Freqtrade has a `leverage()` callback and `custom_stake_amount()` for position sizing. — [Lookahead analysis docs](https://www.freqtrade.io/en/stable/lookahead-analysis/) (mentioned as false-positive sources)
- **Hummingbot PositionExecutor**: stop loss, take profit, time limit and trailing stop per position. TP/SL are placed as orders on spot and managed on perpetuals. — [PositionExecutor](https://hummingbot.org/v2-strategies/executors/positionexecutor)
- **Hummingbot Avellaneda**: inventory target with `risk_factor` skew. Near zero means symmetric quotes and more inventory risk. — [Avellaneda docs](https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/)
- **Passivbot**:
  - Exposure and risk limits are configured (details not on the README).
  - "Unstuck" losses are capped so the balance stays within a set % of its peak. The README links an Equity Hard Stop Loss doc.
  - Warnings: "Used at one's own risk". Don't run v8 live on an unreviewed v7 config. Don't run competing bots on one account.
  - Source: [Passivbot README](https://github.com/enarjord/passivbot)
- **NautilusTrader**: reduce-only, post-only, OCO/OUO/OTO contingencies, and trailing stop market/limit. Cash, margin and betting accounts. Live state reconciliation. — [Nautilus overview](https://nautilustrader.io/docs/latest/concepts/overview)
- **Jesse**: "built-in helper functions" for risk management, plus leverage and short support. No mechanism is described on the landing docs page. — [Jesse docs](https://docs.jesse.trade/)

### Inferences
- **Exchange-side stops**: Freqtrade has a `stoploss_on_exchange` option (not retrieved in this pass). Hummingbot places TP/SL orders on spot. Nautilus supports native stop and trailing order types where the venue allows them. Freqtrade's lookahead tooling forces market orders, so exchange-stop behaviour is not part of bias checks.
- No framework reviewed documents **liquidation-price modelling in backtests** for perpetuals in what was fetched. This matters for martingale bots such as Passivbot, and for this repo's N6 finding that tails liquidate shorts.

### Gaps
- Freqtrade `stoploss_on_exchange`, liquidation handling, and `trading_mode: futures` margin modes (isolated/cross): not fetched in this pass.
- Passivbot's wallet-exposure-limit parameters: not on the README.

## 5. Execution quality and data

### Takeaway
NautilusTrader is the only reviewed framework with a full professional order model (L2/L3-style book deltas, nanosecond timestamps, a rich set of order types and TIFs, funding-rate and mark-price data). Hummingbot is order-book-native for market making. Freqtrade and Jesse are candle-driven. Lean models crypto fills against the book only live.

### Cited Findings
- **NautilusTrader**:
  - Order types: market, limit, stop market/limit, market-to-limit, MIT, LIT, trailing stop market/limit.
  - TIF: IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE.
  - Options: post-only, reduce-only, iceberg, OCO/OUO/OTO.
  - Data: OrderBookDelta/Deltas/Depth (10 levels), quotes, trades, bars, mark price, index price, **funding rate updates**, option Greeks, instrument status. Bars can be aggregated by time, tick, volume, value, Renko and imbalance/run.
  - Rust core, tokio networking, Arrow/Parquet storage, UNIX-nanosecond timestamps.
  - Source: [Nautilus overview](https://nautilustrader.io/docs/latest/concepts/overview)
- **Freqtrade**: SpreadFilter (`max_spread_ratio` default 0.005) uses top-of-book. Backtests assume no slippage and fills within the candle. — [plugins](https://www.freqtrade.io/en/stable/plugins/), [backtesting](https://www.freqtrade.io/en/stable/backtesting/)
- **Lean**: backtests fill market orders immediately and completely. Live fills that exceed the top of book follow the order book. — [Coinbase brokerage](https://quantconnect.com/docs/v2/cloud-platform/live-trading/brokerages/coinbase)
- **Exchange coverage**:
  - Lean lists 6 crypto brokerages for local live trading: Binance, Bybit, Kraken, Coinbase, Bitfinex, dYdX ([Lean brokerages search summary](https://www.quantconnect.com/docs/v2/our-platform/live-trading/brokerages/binance)).
  - Passivbot: Bybit, OKX, Bitget, Bitunix, GateIO, Binance, Kucoin, Hyperliquid, WEEX, Lighter ([README](https://github.com/enarjord/passivbot)).
  - OctoBot: 15+ via CCXT, incl. Binance, Coinbase, Bybit, Hyperliquid, MEXC, KuCoin ([README](https://github.com/Drakkar-Software/OctoBot)).
  - Jesse: mentions DEX support ([docs](https://docs.jesse.trade/)).
  - Nautilus: adapter-based CEX and DEX, plus FX, equities, futures, options and betting ([overview](https://nautilustrader.io/docs/latest/concepts/overview)).
- **Alternative data**: OctoBot offers Google Trends and Reddit social indicators ([README](https://github.com/Drakkar-Software/OctoBot)). Freqtrade's MarketCapPairList uses CoinGecko ranks ([plugins](https://www.freqtrade.io/en/stable/plugins/)). Nautilus handles funding and custom data types ([overview](https://nautilustrader.io/docs/latest/concepts/overview)).

### Inferences
- **Open interest, long/short ratios and on-chain data** are not native in any reviewed framework's docs as retrieved. They must be added as custom data (Nautilus custom types, Freqtrade informative/dataprovider). This repo's `positioning_recorder.py` is therefore not a solved problem elsewhere.
- Latency modelling is documented only for Nautilus (Behavioral Models page, not fetched) and is absent from candle-based bots.

### Gaps
- Freqtrade's exchange list and its futures-supported exchanges, and Hummingbot's connector count (CEX/DEX/Gateway): not fetched.
- Whether Freqtrade exposes order-book data to strategies (`dp.orderbook`) and funding-rate candles: not verified in this pass.

## 6. Operations: paper/live parity, UI, monitoring, deployment

### Takeaway
Freqtrade (dry-run, Telegram, FreqUI, REST API), Hummingbot (CLI, API, Condor), OctoBot (web UI, Telegram, mobile app, cloud) and Jesse (dashboard, paper and live via a licensed plugin) are the operator-ready projects. Nautilus claims one code path for backtest and live, but warns that live adds venue and reconciliation behaviour that simulation may not reproduce.

### Cited Findings
- Nautilus: the "same strategy and execution-algorithm code" runs in backtest and live. It warns that live adds venue, transport, timing, persistence and reconciliation effects. — [Nautilus overview](https://nautilustrader.io/docs/latest/concepts/overview)
- FreqAI RL: state features (profit, position, duration) are used in dry/live but **not available in backtesting**. This is a documented parity break. — [FreqAI RL](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)
- Freqtrade: stoploss exits "can appear more often than in dry-run or live", per the backtesting assumptions. Lookahead and recursive analysis run from freqUI or the REST API. — [Backtesting](https://www.freqtrade.io/en/stable/backtesting/), [Recursive analysis](https://www.freqtrade.io/en/stable/recursive-analysis/)
- Passivbot: a shared Rust orchestrator serves both live trading and backtesting, which the README credits for consistency. — [README](https://github.com/enarjord/passivbot)
- Jesse:
  - Live and paper trading run through an official plugin that needs a license key generated on jesse.trade ([Jesse live trade docs](https://docs.jesse.trade/docs/livetrade)).
  - Pricing is only reported by third-party aggregators, which conflict: $899 / $999 "lifetime" tiers vs $899-$1,599 "monthly" ([toolradar](https://toolradar.com/tools/jesse/pricing), [saasbrowser](https://saasbrowser.com/en/saas/142599/jesse)). Unverified.
  - Jesse MCP lets AI assistants (Claude, Codex, Cursor) drive a local project ([Jesse docs](https://docs.jesse.trade/)).
- OctoBot:
  - GPL-3.0 open source can be self-hosted (PC, server, Raspberry Pi, Docker). octobot.cloud is the hosted option.
  - Its newest release defaults to "node mode", which backs a unified dashboard and mobile app, with self-custody of keys and end-to-end encryption. This is a beta.
  - Telegram control is available.
  - Source: [OctoBot README](https://github.com/Drakkar-Software/OctoBot)
- Hummingbot: executors can be created via the Hummingbot API without a running script. Condor is the API front end. — [Hummingbot strategies](https://hummingbot.org/strategies/), [Condor](https://github.com/hummingbot/condor)

### Inferences
- Paper/live parity is a recurring weakness. Two of the most-used ML/RL paths (FreqAI RL and Lean crypto) have documented backtest-vs-live differences. This matches §0 rule 3 of this repo (the training-vs-deployment gap must stay visible).

### Gaps
- Freqtrade Telegram commands, FreqUI and the producer/consumer mode were not fetched directly in this pass. They are widely documented, but no URL was retrieved.
- Hummingbot's paper-trade connector behaviour was not verified.

## 7. Published evidence on performance of default strategies

### Takeaway
No peer-reviewed or independent benchmark of these bots' **default** strategies was found. Evidence is anecdotal, tutorial-grade or documentation warnings. The projects themselves say their defaults are not production strategies: FreqAI's RL reward is "_not_ for live production", and Passivbot is "Used at one's own risk".

### Cited Findings
- An extended web search found no peer-reviewed study measuring backtest overfitting or live P&L across open-source crypto bot strategies. Hits were tutorials (DEV.to Freqtrade lessons), general overfitting literature and informal reviews. — [Search hits incl. DEV.to](https://dev.to/henry_lin_3ac6363747f45b4/lesson-6-freqtrade-strategy-performance-analysis-3b9o), [backtestscore.com](https://www.backtestscore.com/tools/freqtrade)
- FreqAI RL's default reward is described as a benchmark, not for live production, and the docs warn that agents can learn "cheats". — [FreqAI RL](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)
- Crypto DRL research (FinRL authors) treats backtest overfitting as the central practical problem and proposes hypothesis tests to reject overfitted agents. — [arXiv 2209.05559](https://arxiv.org/pdf/2209.05559)

### Inferences
- For the GAP analysis, "default strategies are profitable" is an **unsupported claim** for every project reviewed. The only defensible comparison is on tooling: bias checks, CV, fill realism and protections. This repo's own measured baseline (bot signals traded as-is −0.83%/day after fees; most trader hypotheses rejected; FC passive carry ~0.9%/yr on its 2026 test) is the kind of evidence these projects do not publish.

### Gaps
- Hummingbot liquidity-mining/miner historical returns and any third-party audits of Passivbot or Hummingbot market-making P&L were not found in this pass.
- No independent live-trading track records for Freqtrade community strategies (e.g. NostalgiaForInfinity) were retrieved.
