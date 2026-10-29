# Crypto data and signal services, and the out-of-sample evidence for their data types (short horizon, minutes to 1 day), 2025-2026

Scope note: about 20 search and fetch calls. I verified vendor facts for Binance endpoints, Coinglass pricing, Glassnode point-in-time data and Amberdata tiers. I did NOT verify vendor pages for CryptoQuant, Santiment, LunarCrush (beyond its Galaxy Score definition), Token Metrics, Nansen, Arkham, Kaiko, Laevitas/Greeks.live, TradingView, Hyblock, TensorCharts/Bookmap or Velo; those are listed under Gaps. The academic evidence is uneven: much of it is preprints, theses or single-author SSRN papers, and I label each one.

## Q1. For each data type, what is the evidence of predictive power (horizon, sample, OOS or not, after fees)?

### Takeaway
No data type in this list has strong, peer-reviewed, out-of-sample, after-cost evidence of predicting directional returns at horizons from minutes to 1 day. The best documented effects fall into three groups:
- **Microstructure.** Order-flow and order-book imbalance predict returns at horizons of seconds. The effect is small (R² about 0.1-1%), unstable and fragile, and an AR(1) benchmark matches it.
- **Intraday stablecoin exchange inflows.** USDT net inflows to exchanges predict BTC and ETH returns at 1-6 h (one preprint, in-sample and OOS regressions, no cost analysis).
- **Attention.** Attention proxies such as Google searches predict returns at 1-6 weeks, outside the 1-day horizon (Liu and Tsyvinski, RFS 2021).

Funding rates are themselves predictable, but forecasting them does not survive fees. Positioning data (open interest, long/short ratios, liquidations) is mainly descriptive or an early-warning signal for volatility and crashes, with weak and event-dependent OOS results. Options skew and whale alerts have essentially no rigorous evidence for direction.

### Cited Findings

**Funding rates / basis (derivatives positioning)**
- Inan (SSRN, Oct 2025), "Predictability of Funding Rates": an out-of-sample forecast of BTC perpetual funding rates on Binance and Bybit with double-autoregressive models beats a no-change benchmark. Directional accuracy is 48.70-60.16% on Binance and 49.48-62.5% on Bybit, depending on the estimation method. This is predictability of the funding rate, not of the return. — [SSRN 5576424](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- A secondary write-up of the same paper has a section titled "Why Trading Fees Reverse the Backtest Results", so the trading backtest appears not to survive costs. I could not see the table figures. — [Securities.io](https://www.securities.io/bitcoin-funding-rates-forecasting-trading-costs/)
- Crypto carry (short perpetual, long spot) had an annualized Sharpe of 6.45 over 2020-2025. The Sharpe falls to 4.06 from 2024 and turns negative in 2025. This is a carry harvest, not a directional signal. — [arXiv 2510.14435](https://arxiv.org/pdf/2510.14435) (via search summary)
- Schmeling, Schrimpf and Todorov, "Crypto carry" (BIS WP 1087, 2023, rev. Oct 2025): the futures-spot carry can exceed 40% p.a. The earlier abstract says up to 60%, so the versions differ. The carry is driven by trend-chasing and attention from small investors seeking leverage, plus scarce arbitrage capital. Margin spikes and liquidations in drawdowns make the arbitrage risky and are linked to crashes. I did not see their return-predictability tests. — [BIS WP 1087](https://www.bis.org/publ/work1087.htm); [CEPR DP20719](https://cepr.org/publications/dp20719)

**Open interest, long/short ratios, taker ratio, liquidations**
- arXiv 2607.27070 (2026 preprint) studies seven BTCUSDT-perp liquidation cascades (May 2022 to Oct 2025). It uses 1-min klines and 5-min Binance metrics (OI, top-trader long/short ratio, global long/short ratio, taker buy/sell ratio) from data.binance.vision. The test is early warning by critical slowing down, not return prediction.
  - Rising lag-1 autocorrelation of OI is in-sample positive in 23 of 39 configurations for Oct 2025, but OOS positive in only 3 of 39 for Aug 2024.
  - Per-event shares: May 2022 .38, Nov 2022 .54, Aug 2024 .08, Dec 2024 .26, Feb 2025 .10, Apr 2025 .21, Oct 2025 .59.
  - Price shows the signature in 5 of 7 events and is silent in the two exogenous tariff shocks.
  - The authors call funding "far too coarse" to serve as an early-warning variable.
  - Source: [arXiv 2607.27070](https://arxiv.org/html/2607.27070)
- Oct 10-11 2025 cascade: about $19B of OI was erased within 36 h (Ali, SSRN). In a separate minute-level study (Lim, SSRN), BTC fell 12.6% in 10 minutes on Binance, with futures leading. Both papers describe the event; neither tests prediction. — [SSRN 5611392](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5611392); [SSRN 6579278](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6579278)
- I found no peer-reviewed paper that tests OI changes or liquidation volume as an OOS, after-cost predictor of returns at horizons from minutes to 1 day.

**Order book / order-flow imbalance (heatmap and depth data)**
- Schmalz (SSRN) tests Cont-Kukanov-Stoikov order-flow imbalance (OFI) on Binance.US BTC/USDT at horizons of 1-10 s:
  - The out-of-sample sign flipped as the sample grew: positive at about 7 days, null or negative at about 9 days, positive again at about 17 days.
  - R² is about 0.1-1%.
  - An AR(1) on trailing returns matches or beats OFI at most horizons.
  - Signed trade-flow imbalance is never significant.
  - Source: [SSRN 7227998](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7227998)
- arXiv 2602.00776: top-of-book and trade-flow features have similar predictive shapes across crypto assets. The OFI effect is mostly monotone, diminishes at the extremes, and wider spreads weaken it. — [arXiv 2602.00776](https://arxiv.org/html/2602.00776v1)
- Anastasopoulos and Gradojevic, "Order Flow and Cryptocurrency Returns" (EFMA 2025 conference paper; I did not read the details). — [EFMA PDF](https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf)
- A Charles University thesis (2025, unreviewed) on Binance futures BTC/ETH/LTC finds that bid-ask asymmetry measures predict immediate returns. — [CUNI thesis](https://dspace.cuni.cz/bitstream/handle/20.500.11956/200516/120505902.pdf?sequence=1)

**Exchange netflows / stablecoin flows (on-chain)**
- Chi, Chu and Hao (arXiv 2411.06327, v2 Sep 2025). Sample 2017-2023, horizons of 1-6 h intraday, with in-sample and out-of-sample regressions. The abstract gives no coefficients and no cost treatment.
  - USDT net inflows to exchanges positively predict BTC and ETH returns at several intervals.
  - ETH net inflows negatively predict ETH returns and volatility at all intraday intervals.
  - BTC net inflows generally lack predictive power for BTC returns, except at 4 h.
  - Source: [arXiv 2411.06327](https://arxiv.org/abs/2411.06327)
- Griffin and Shams (Tether, Mar 2017 to Mar 2018, hourly): hours after negative returns with large Tether flows were under 1% of hours (87 h) but were associated with about 50% of BTC's compounded return. They read this as price support or manipulation by one large Bitfinex player. It is a historical, one-regime result. — [Quantpedia summary](https://quantpedia.com/a-very-influential-paper-about-tether-bitcoin-relationship-manipulation/)
- Pre-registered student project (EPAT, not peer-reviewed): whale features for next-day BTC direction give OOS ROC-AUC of 0.49-0.53 over 15 combinations, and the edge vanishes at about 0.02% per-trade cost. — [QuantInsti blog](https://blog.quantinsti.com/does-whale-activity-predict-bitcoin-a-pre-registered-machine-learning-test-epat-project/)
- Herremans and Low (arXiv 2211.08281, 2018-2021): whale-transaction tweets plus CryptoQuant data forecast extreme volatility spikes, not direction. Daily net exchange flow correlates with daily BTC volatility (Pearson 0.47). — [arXiv 2211.08281](https://arxiv.org/pdf/2211.08281)

**Whale alerts**
- Saggu (Finance Research Letters 49, 2022): BTC's price response to Tether minting is significantly stronger when Whale Alert announces the mint on Twitter, with a short-term price increase, especially in bullish sentiment. — [Whale Alert research list](https://whale-alert.io/academic-research.html) (vendor-curated list)
- Magner and Sanhueza (FRL 85, 2025), "Moby Dick effect": a whale signal built from Whale Alert Telegram wallet-to-exchange transfers shows significant contagion, mainly after 6 and 24 h. — [Whale Alert research list](https://whale-alert.io/academic-research.html)
- Neither paper reports a tradable, after-cost OOS strategy, as far as the summaries show.

**Social sentiment / attention**
- Liu and Tsyvinski (RFS 2021; NBER w24877): time-series momentum and investor-attention proxies, such as Google searches, strongly forecast crypto returns. The horizon is weekly: 1-2 weeks for BTC, about 1 week for XRP, 1-6 weeks for ETH, as reported by secondary coverage. Secondary reports also give a +2.3% price change within two weeks per 1 SD of "Bitcoin" search volume; I did not check this against the paper. The evidence is in-sample time-series predictability. — [NBER w24877](https://www.nber.org/system/files/working_papers/w24877/w24877.pdf); [CoinDesk](https://coindesk.com/markets/2018/08/10/google-search-can-predict-bitcoin-price-increases-study-finds)
- Kraaijeveld and De Smedt (JIFMIM 2020), "The predictive power of public Twitter sentiment": Twitter sentiment predicts returns for some of nine cryptocurrencies, and about 14% of the tweets came from bots. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S104244312030072X)
- A Düsseldorf dissertation (Kürzinger) finds intraday sentiment windows of about 18-108 minutes most promising for BTC. — [Dissertation PDF](https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-72975/Dissertation_Kuerzinger.pdf)
- BERT-based sentiment from news, Reddit and Tweets (arXiv 2204.05781). Ortu et al. (2022), cited in that literature, find that GitHub and Reddit sentiment improve hourly and daily direction prediction for BTC and ETH. — [arXiv 2204.05781](https://arxiv.org/pdf/2204.05781)
- No study I found nets out transaction costs for sentiment signals at horizons of a day or less.
- LunarCrush Galaxy Score is a composite that includes price trend and the correlation of social metrics with price and volume. Because price is an input, testing it against future returns is partly circular. I found no independent evaluation of the score. — [TradingView Galaxy Score](https://in.tradingview.com/support/solutions/43000739595-galaxy-score/)

**Options (skew / implied volatility)**
- I found no peer-reviewed test of Deribit skew predicting BTC returns at horizons from minutes to 1 day.
- A hobby backtest (GitHub, about 4.5 years): the sign of the 25-delta risk reversal predicted the next move 58.9% of the time on 56 monthly samples (p ≈ 0.18, not significant). The correlation between risk-reversal size and the next move was near zero. — [GitHub deribit-rr](https://github.com/JERONIMODIFRANCO/deribit-rr/tree/main)
- arXiv 2510.21297 models jump risk premia in BTC options with Hawkes processes. It concerns option-selling premia, not spot direction. — [arXiv 2510.21297](https://arxiv.org/pdf/2510.21297)

### Inferences
- A short-horizon directional edge from these data types is either very short-lived (microstructure, seconds) or regime-specific (Tether 2017-18). A retail bot polling every 1-15 minutes and paying taker fees of about 0.04-0.1% per side cannot capture an effect of 0.1-1% R² at seconds.
- Positioning data looks more useful for risk control and volatility forecasting (cascade precursors, volatility spikes) than for entry direction. That matches this repo's own pre-registered results:
  - Positioning added +0.022 AUC [−0.027, +0.064] for rockets (R-11, refuted).
  - Leader-continuation tests were too early to judge.
  - Announcements follow the move.
  - The only accepted trader hypothesis is passive funding carry, about 0.9%/yr on its 2026 test.
  - Source: [CLAUDE.md](D:/Projects/claude_crypto_bot/CLAUDE.md)
- The most promising untested lever at 1-6 h in the literature is USDT exchange netflow (Chi, Chu and Hao). It needs entity-labeled exchange wallets, which brings in the look-ahead problem from revisions (Q4).

### Gaps
- I did not read the full texts of Inan, Chi/Chu/Hao, Schmeling et al. or Liu-Tsyvinski. Coefficients, OOS R² and after-cost Sharpe are not extracted.
- No rigorous OOS, after-cost study was found for: OI change as a return predictor, liquidation volume as a reversal or continuation signal, liquidation-heatmap "magnet" claims, long/short-ratio contrarian signals, or options skew/IV at horizons of a day or less.
- I did not search the cross-sectional crypto factor literature (e.g. Liu, Tsyvinski and Wu, JF 2022) or funding as a cross-sectional predictor.

## Q2. What is free with history (Binance public endpoints, data.binance.vision) and what is paid? Service inventory

### Takeaway
Binance REST "futures data" endpoints keep only the latest 30 days (1 month for OI history), at 5m to 1d periods. To build longer history, either record them yourself or use the data.binance.vision `metrics` archive: 5-minute OI, top-trader and global long/short ratios, and taker ratio. Per-event liquidation archives (`liquidationSnapshot`) are discontinued, and the live `forceOrder` stream throttles to one liquidation per symbol per second. Long liquidation history and modeled heatmaps therefore need a paid vendor or self-recording.

### Cited Findings
- `GET /futures/data/openInterestHist`: `symbol`, `period`, `limit` (max 500, default 30), `startTime` and `endTime`. "Only the data of the latest 1 month is available." Weight 0, IP limit 1000 requests per 5 min. Periods with no trades are omitted and not backfilled. — [Binance docs, Open Interest Statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- `GET /futures/data/globalLongShortAccountRatio` and `GET /futures/data/basis`: latest 30 days only. Periods are 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, plus one more value the page hides (presumably 1d; not verified). — [same Binance docs page](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- The top-trader position ratio is the net long and net short share of total open positions among the top 20% of users by margin balance. — [Binance docs, Top Trader Long/Short Ratio](https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/Top-Trader-Long-Short-Ratio)
- `!forceOrder@arr` and `<symbol>@forceOrder`: "For each symbol, only the latest one liquidation order within 1000ms will be pushed as the snapshot." Earlier liquidations in the same second are not delivered, so the stream undercounts during cascades. — [Binance docs, Liquidation streams](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/All-Market-Liquidation-Order-Streams)
- "Binance's per-event liquidationSnapshot dumps have been discontinued and are no longer served for any date." Intraday liquidation microstructure needs a commercial source. — [arXiv 2607.27070](https://arxiv.org/html/2607.27070)
- data.binance.vision `metrics` files carry `sum_open_interest`, `count_toptrader_long_short_ratio`, `sum_taker_long_short_vol_ratio` and related columns. The sources disagree on granularity: a GitHub issue says daily files with OI history only daily, while arXiv 2607.27070 used 5-minute metrics rows (about 17,600 per two-month window). — [binance-public-data issue #211](https://github.com/binance/binance-public-data/issues/211); [arXiv 2607.27070](https://arxiv.org/html/2607.27070)
- Coinglass API plans:
  - Hobbyist: $29/mo, 80+ endpoints, 30 requests/min, updates within 1 min, personal use only.
  - Startup: $79/mo.
  - Standard: $299/mo, the first tier allowing commercial use.
  - Hobbyist lookback is 6 days at 1-min and 180 days at 1-h; daily data is all-time on every plan.
  - Heatmap snapshots at specific timestamps are an Enterprise customization.
  - Coinglass liquidation heatmaps are modeled estimates, not exchange-confirmed figures.
  - Sources: [Coinglass pricing](https://coinglass.com/pricing); [Coinglass Enterprise docs](https://docs.coinglass.com/reference/enterprise-customization); [DEV review](https://dev.to/great-time-flies/coinglass-api-review-2026-is-it-worth-it-for-crypto-quant-traders-2bcf)
- Glassnode point-in-time (PiT) metrics need the Professional plan (Studio and API), and are also offered via Snowflake. — [Glassnode research](https://research.glassnode.com/why-use-point-in-time-data/)
- Amberdata API tiers: Trial 15 calls/s and 20k calls/day; On-Demand 20 calls/s and 250k calls/day; Enterprise up to 60 calls/s. Institutional pricing is mostly by quote. The source is an AI-generated comparison site, so confidence is low. — [rfp.wiki Amberdata](https://www.rfp.wiki/crypto/digital-assets-nfts/crypto-data-analytics-market-risk/amberdata/santiment)
- Coin Metrics publishes predicted funding rates as a market-data product. — [Coin Metrics docs](https://gitbook-docs.coinmetrics.io/market-data/market-data-overview/market-funding-rates-predicted)
- Laevitas competes with Deribit, Kaiko and CCData (CB Insights). Kaiko sells market data, analytics and indices for risk, surveillance and strategy. — [CB Insights](https://www.cbinsights.com/company/laevitas/alternatives-competitors)

### Inferences
Free and self-recordable data covers most of what this bot needs:
- klines, aggTrades and bookTicker archives on data.binance.vision;
- the `metrics` archive for OI and long/short ratios;
- live OI, ratios and the forceOrder stream recorded locally (as `positioning_recorder.py` already does).

Paid services add three things:
- multi-exchange aggregation;
- deep liquidation history and modeled heatmaps;
- entity-labeled on-chain flows (CryptoQuant, Glassnode, Nansen, Arkham).

Only the last cannot be reproduced from Binance data.

The repo's own positioning store holds 60 days, 671 snapshots and 87 coins ([CLAUDE.md](D:/Projects/claude_crypto_bot/CLAUDE.md)). The data.binance.vision `metrics` archive could extend this history backward, if it really is 5-minute; check this directly.

### Gaps
- I did not verify the `metrics` archive's start date, its granularity per symbol, or whether `takerlongshortRatio` and `topLongShortAccountRatio` have the same 30-day REST limit (very likely, but not fetched).
- I did not verify the status of the REST `allForceOrders` endpoint. It is believed to have been retired; unverified.
- Vendor data catalogs and prices were not fetched for CryptoQuant, Santiment, LunarCrush, Token Metrics, Nansen, Arkham, Kaiko, Laevitas/Greeks.live, Hyblock, TensorCharts/Bookmap and Velo.

## Q3. What alerting and automation do these services offer that a self-built bot lacks?

### Takeaway
Little verified. Coinglass alerts appear to be an app (Pro) feature rather than part of the API, and API products are mostly raw data. What vendors add is aggregation (many exchanges), labeling (entity and wallet clustering) and modeled products (liquidation heatmaps). None of these comes with OOS evidence of returns.

### Cited Findings
- No evidence that Coinglass API plans include alerts. A third-party overview lists real-time alerts and API access as separate Pro features. The only alert-type API route found is a Hyperliquid whale-alert endpoint, in a third-party API specification. — [Coinglass API spec (glama)](https://glama.ai/mcp/servers/@forgequant/coinglass-mcp/blob/3ff015e428d99f013e3e7fe363b98d82adeb00a3/docs/02-api-specification.md); [NexusFi Coinglass](https://nexusfi.com/d/data-providers/coinglass/)
- Coinglass heatmap snapshots (full distribution at chosen timestamps, multi-exchange) are a custom enterprise product. — [Coinglass Enterprise docs](https://docs.coinglass.com/reference/enterprise-customization)
- Whale Alert distributes large-transfer announcements via Twitter and Telegram. Academic work uses these feeds as event signals (Saggu 2022; Magner and Sanhueza 2025). — [Whale Alert research](https://whale-alert.io/academic-research.html)
- LunarCrush aggregates data from X, Reddit, YouTube and TikTok into scores. The vendor itself frames its metrics as indicators, "not predictive guarantees". — [CryptoSlate LunarCrush](https://cryptoslate.com/companies/lunarcrush/); [LunarCrush blog](https://lunarcrush.com/blog/what-is-social-intelligence)

### Inferences
A self-built bot already has what these services sell as automation: screeners, threshold alerts, Telegram delivery. What it cannot cheaply build is cross-exchange liquidation and OI aggregation, entity-labeled on-chain flows, and social-volume collection at scale. Those are the only reasons to pay. Each should first be tested as a SHADOW feature on the bot's own candidates (TH rule 6).

### Gaps
- I did not verify the alert and automation features of TradingView (alerts, screeners, Pine webhooks), Hyblock, Velo, Nansen Smart Alerts, Arkham alerts, Santiment alerts or Token Metrics. They need direct vendor-page checks.

## Q4. Known pitfalls: survivorship, look-ahead in on-chain metrics, sentiment data quality

### Takeaway
On-chain exchange-flow metrics are revised after the fact as clustering and labeling improve, so backtests on standard history are look-ahead-biased. Glassnode's own PiT comparison shows a strategy getting "notably worse" on PiT data. Sentiment data is polluted by bots (about 14% of tweets in one study). Composite social scores embed price. Binance's liquidation stream and REST positioning endpoints silently truncate history, and delisted pairs disappear.

### Cited Findings
- Glassnode on revisions and PiT data:
  - Standard on-chain values for a past date can differ from what was first published, because of clustering improvements, late-reported exchange data and refined labeling. Metrics that depend on entity identification, such as exchange flows, are the worst affected.
  - Glassnode ran the same BTC exchange-balance strategy on standard and on PiT data (same logic, dates and fees); on PiT data the result was "notably worse". This is a single, vendor-run example.
  - PiT metrics are append-only and carry a `computed_at` timestamp.
  - Sources: [Glassnode, "Your Backtest Is Lying"](https://insights.glassnode.com/why-use-point-in-time-data); [MEXC republication](https://www.mexc.com/news/922242)
- About 14% of tweets in a nine-coin Twitter sentiment study came from bot accounts. — [Kraaijeveld and De Smedt 2020](https://www.sciencedirect.com/science/article/abs/pii/S104244312030072X)
- The Galaxy Score includes price trend and price/volume correlation, so circularity is a risk when it is tested against returns. — [TradingView](https://in.tradingview.com/support/solutions/43000739595-galaxy-score/)
- The OFI sign flipped OOS with sample length (7, then 9, then 17 days): small samples give unstable conclusions. — [SSRN 7227998](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7227998)
- Early-warning results differ by event (OI lag-1 AR at .08-.59 across seven cascades), and the in-sample signal did not hold OOS (23 of 39 vs 3 of 39). — [arXiv 2607.27070](https://arxiv.org/html/2607.27070)
- Binance forceOrder pushes only one liquidation per symbol per second, and the liquidationSnapshot archive is discontinued. Liquidation counts from free sources are therefore biased low exactly during cascades. — [Binance docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/All-Market-Liquidation-Order-Streams); [arXiv 2607.27070](https://arxiv.org/html/2607.27070)
- Binance OI history omits periods with no trades and does not backfill them. — [Binance docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- Griffin and Shams' Tether result is specific to 2017-18 and to one large player; critics say the conclusions are stronger than the data allows. — [Quantpedia](https://quantpedia.com/a-very-influential-paper-about-tether-bitcoin-relationship-manipulation/)
- Carry Sharpe decays (6.45 full sample, then 4.06 from 2024, then negative in 2025). Published edges decay. — [arXiv 2510.14435](https://arxiv.org/pdf/2510.14435)
- Survivorship: delisted pairs cannot be recovered from exchange klines. The repo has its own example: the Binance 24h ticker keeps answering for delisted pairs with non-zero volume, which created phantom rows. — [CLAUDE.md](D:/Projects/claude_crypto_bot/CLAUDE.md)

### Inferences
- Any on-chain test should use PiT data, or the bot's own forward recording. Vendor history from a standard (non-PiT) API is a look-ahead trap.
- Self-recorded Binance positioning is free of revisions but short. Its main risk is statistical power, not bias.
- Social-sentiment vendors would need a bot-filtering audit and a forward SHADOW test. Backtests on vendor history, which may be backfilled, are not trustworthy.

### Gaps
- I found no independent audit of revision rates for CryptoQuant, Nansen or Arkham labels, and none of Santiment or LunarCrush bot filtering.
- I found no systematic survivorship study for crypto short-horizon signal papers.
