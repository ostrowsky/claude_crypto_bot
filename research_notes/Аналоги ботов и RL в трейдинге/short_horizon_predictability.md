# Short-horizon crypto price-direction predictability (seconds to 1 h) and supervised learning vs RL (as of 2026-10)

Scope note: the research covered 2014–2026 literature, with priority on out-of-sample, time-split, after-cost evidence. Several key 2026 sources are recent arXiv/SSRN preprints that have not been peer reviewed. They are marked as such. Where the full text was read (extracted from the PDF), the notes say "full text read". Otherwise the claim rests on the abstract or a search snippet.

## 1. Empirical efficiency of crypto at intraday horizons (autocorrelation, variance ratios, time-varying efficiency, intraday reversal/momentum)

### Takeaway
Intraday crypto returns are not a pure random walk. There is a small, persistent, statistically robust directional effect, mainly short-horizon REVERSAL at 5 m–1 h, and it is far stronger in crypto than in US equities. The best and most recent matched study (2026) finds that it is worth about 1.3 bp per trade at best on 15 m bars, against a cheapest round-trip cost of 5 bp. It is detectable but not tradable at public fee schedules. Efficiency has been improving over time, and it is lower at higher sampling frequencies.

### Cited Findings
- Kitron & Wengrowicz (arXiv 2608.21888, Aug 2026, preprint; full text read). Data: 15-minute candles, 2025-01-01..2026-02-11, the 183 highest-volume Binance USDT spot pairs vs 187 liquid US stocks/ETFs. Multi-year 15 m histories go back to 2021-01. Coinbase/OKX/Bybit refetches were used as controls, and a frozen post-sample holdout covers 2026-02..2026-08. — [arXiv abs](https://arxiv.org/abs/2608.21888)
  - Directional reversal at 15 m is significant in 90% of the 183 crypto pairs vs 2.7% of the 187 stocks. Crypto mean out-of-sample AUC is 0.531, with 98% of pairs above 0.5. Stocks average 0.499. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
  - Lag-1 return autocorrelation is about 0 on the major coins. The effect lives in the SIGN of returns, not their size: simply "betting against the previous candle's sign" reaches a crypto class-mean AUC of 0.527. Unthresholded accuracy on BTC is 52.3%. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
  - Horizon profile: the signal is strongest at the highest frequency and decays monotonically to no skill by about 4 hours. At 1 h, 58% of the 169 eligible crypto pairs remain significant after FDR, vs 0% of stocks. It is a standing effect: every focal coin-year since 2021 is significant. BTC's per-year AUC drifts down mildly, which is consistent with slow erosion. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
  - Holdout: the class-mean AUC gap is +0.031 as designed, and +0.011 [+0.008, +0.014] under the conservative scoring. Crypto mean AUC is 0.531 primary vs 0.522 on the 2026-02..08 holdout. The attenuation is 77% decay on a fixed universe and 23% universe change. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
  - Economics: "the gross edge peaks near 1.3 bp per trade against a 5 bp cheapest round-trip cost". The paper's spot cost bands are maker 5 bp and taker 10–20 bp round trip. Not one of the 183 pairs clears even the 5 bp maker band at any confidence threshold. The median pair earns 0.46 bp per trade. Going to 5-minute bars makes it worse (0.15 bp), "because higher frequency shrinks each opportunity faster than it shrinks costs". Thresholding cannot rescue it, because fewer trades shrink the opportunity count faster than the edge grows. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
  - Mechanism: reversal concentrates after moves pushed by aggressive taker flow (Binance kline taker-buy share). AUC is 0.540 after flow-driven bars vs 0.519 after flow-opposed ones. Book depth consumed does not matter. The authors read this as compensated liquidity provision (untested). Twelve taker-imbalance lags alone give OOS AUC 0.521, below the sign-only model. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
- Sensoy (Finance Research Letters), permutation entropy on BTCUSD/BTCEUR intraday data: both markets have become more informationally efficient, pricing efficiency has been improving at the intraday level, and "the higher the return frequency, the lower the informational efficiency". — [Sensoy, FRL (PDF)](https://yoksis.bilkent.edu.tr/pdf/files/14366.pdf)
- Variance-ratio tests on 15–120-minute Bitcoin data find informational inefficiency at higher frequencies. — [Informational inefficiency of Bitcoin: high-frequency data, Research in International Business and Finance](https://www.sciencedirect.com/science/article/abs/pii/S0275531918303532)
- Another high-frequency study reports Bitcoin is efficient at 1 minute but inefficient at 5 minutes (snippet only). — [The inefficiency of Bitcoin revisited: high-frequency analysis, FRL](https://www.sciencedirect.com/science/article/abs/pii/S1544612318302320)
- Aslan & Sensoy: efficiency of the largest cryptocurrencies shows a U-shape across sampling frequencies (snippet). — [Intraday efficiency-frequency nexus in the cryptocurrency markets](https://www.researchgate.net/publication/334941629_Intraday_efficiency-frequency_nexus_in_the_cryptocurrency_markets)
- A tick-level Bitcoin study finds no return predictability at or above 1 day, predictability at frequencies up to 6 h, and time-varying predictability (snippet). — [Bitcoin at High Frequency](https://www.researchgate.net/publication/331172774_Bitcoin_at_High_Frequency)
- A daily-data Hurst-exponent study disagrees with the improving-efficiency narrative: it finds Bitcoin does not become more efficient over time (snippet, via the same search). — [Ledger 2021](https://ledger.pitt.edu/ojs/ledger/article/download/213/212/)
- The Quarter-Hour Effect (arXiv 2607.09426, 2026 preprint, snippet only): quarter-hour opening returns in crypto futures are predictable out of sample across six assets, with a cross-sectional average OOS R² of 3.37%. Opening order imbalance predicts returns over 4–12 hours, with weaker effects at finer clock frequencies. — [arXiv 2607.09426](https://arxiv.org/html/2607.09426v2)

### Inferences
- The strongest robust intraday effect in crypto is negative sign autocorrelation (reversal) at 5 m–1 h. It looks like the reward for supplying liquidity, competed down to roughly the cost of supplying it. A taker strategy, which is what a retail bot using market orders is, sits on the paying side of that equilibrium.
- Short-horizon "momentum" (buying after an up bar) is the wrong sign for the dominant effect. This matches the project's own refutations: crossing entries, BTC-leads-alts, and XSMOM/S1 at longer horizons.
- "Statistically significant" and "economically exploitable" diverge by roughly a factor of 4 (1.3 bp vs 5 bp) even for the best-case maker. For a taker at 0.1% per side on spot, the gap is about 15x.

### Gaps
- Urquhart (2016, Economics Letters, "The inefficiency of Bitcoin") and Bariviera's (2017) Hurst/DFA work were not fetched in this pass. They use daily data, finding Bitcoin inefficient over 2010–2016 but moving toward efficiency in later subsamples, so they bear little on intraday horizons. Their exact numbers are not verified here.
- No large, matched study of 1 h momentum (as opposed to reversal) in altcoins with costs was found beyond the 2026 reversal paper.

## 2. Order-flow / microstructure predictability (OFI, LOB features, DeepLOB, horizon decay, does the move exceed spread + fees)

### Takeaway
Order-flow and order-book state carry real information, but over seconds, not 15 min–1 h. Seconds-ahead mid-price direction can reach about 60–70% accuracy on liquid books. That predictability decays to near noise within tens to hundreds of events. The predicted moves are usually smaller than the spread plus taker fees. The microstructure literature, including DeepLOB's own authors, repeatedly says that forecast accuracy is not tradable profit.

### Cited Findings
- Cont, Kukanov & Stoikov (2014, J. Financial Econometrics) "The Price Impact of Order Book Events": over short intervals, price changes are mainly driven by order flow imbalance (OFI), in an approximately linear relation whose slope is inversely proportional to market depth. This is a CONTEMPORANEOUS relation (OFI in the same interval explains the price change), not a forecast of the next interval. — [arXiv 1011.6402](https://arxiv.org/abs/1011.6402)
- Schmalz (SSRN 7227998, 2026, snippet only): on BTC/USDT (Binance.US), OFI had a small positive out-of-sample fit over about 7 days, which turned null or negative as the sample grew to about 9 days. A simple AR(1) trailing-return model matched or beat OFI at every horizon except 10 s. — [SSRN 7227998](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7227998)
- Bieganowski (arXiv 2602.00776, 2026 preprint; HTML read). Data: Binance Futures books and trades at 1 s resolution, 2022-01..2025-10-12, for BTC, LTC, ETC, ENJ and ROSE. Target: 3-second mid log return, modelled with CatBoost under walk-forward CV with a purge gap. Spread, depth and OFI/trade-imbalance features show similar importance across assets. — [arXiv 2602.00776](https://arxiv.org/html/2602.00776v1)
  - Taker backtest results (cross the spread, latency NOT modelled): annualised return BTC 0.13 vs buy-and-hold 0.83, and LTC 0.07 vs 0.53, so the liquid majors underperform. Thinner coins show significant outperformance at 5%: ETC 5.78, ENJ 4.06, ROSE 7.00, the last mostly from the 10-Oct-2025 crash event. — [arXiv 2602.00776](https://arxiv.org/html/2602.00776v1)
  - Maker simulation results: no asset is statistically significant. — [arXiv 2602.00776](https://arxiv.org/html/2602.00776v1)
  - The paper reports no OOS R² or accuracy numbers.
- Jha et al. (arXiv 2010.01241, 2020; full text read): a temporal CNN on the Coinbase BTC-USD book (50 levels, 100 ms, 12–20 June 2019, about 8 days) reaches 71% walk-forward accuracy at a 2-second horizon. Per-direction accuracy is about 61% down / 66% up. Accuracy falls as the horizon lengthens, and no trading after fees is evaluated. — [arXiv 2010.01241](https://arxiv.org/abs/2010.01241)
- DeepLOB (Zhang, Zohren & Roberts 2019) was evaluated on FI-2010 and LSE equities, not crypto. Its trading simulation assumes mid-price execution and reports gross profits before fees, "mainly meant as a method of comparison between models". The authors note that inconsistent signals cause redundant trades and higher transaction costs. — [DeepLOB, ar5iv 1808.03668](https://ar5iv.labs.arxiv.org/html/1808.03668)
- Briola, Bartolucci & Aste, "Deep limit order book forecasting: a microstructural guide" (Quantitative Finance 2025; LOBSTER NASDAQ stocks, 15 deep models). — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
  - Horizon decay: for small-tick stocks, MCC is 0.11 at 10 events (H10), 0.04 at H50 and 0.01 at H100 ("random forecasts"). For medium-tick stocks it is 0.13, 0.085 and 0.036. Large-tick stocks stay at 0.26–0.36. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
  - Conclusion: "high forecasting power does not necessarily correspond to actionable trading signals". The authors propose instead the probability of executing a correct transaction, which FALLS as confidence thresholds rise even while MCC rises. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
  - Relevance to crypto: crypto majors on Binance behave like SMALL-tick instruments (price far above the tick). That is the regime where the predictability decays fastest.
- arXiv 2607.09230 ("When Does Order Flow Matter?", 2026, abstract only). Data: Binance BTCUSDT/ETHUSDT futures, 2023–2026, top-20 L2 with rolling monthly OOS folds. Order flow adds value only ON TOP of the L2 state, "not as a replacement". For BTC it appears only in isolated 5-minute passes, and no cost or P&L analysis is given. — [arXiv 2607.09230](https://arxiv.org/abs/2607.09230)
- A personal, non-peer-reviewed GitHub project on self-collected live Binance L2 data reports that the raw book (L1 queue imbalance) beats an OFI-only model at short horizons. — [jaefit/deep-ofi](https://github.com/jaefit/deep-ofi)
- A 5-minute trade-by-trade deep-learning study notes the strategy trades in every interval, and in quiet periods "the price movements ... are so small that it could not cover the transaction cost". — [arXiv 2010.07404](https://arxiv.org/pdf/2010.07404)

### Inferences
- The informative horizon of OFI/LOB features (seconds to about 1 minute) is one to three orders of magnitude shorter than the 15 min–1 h horizon in question. Book features computed at decision time carry little information about the sign of a 15 m or 1 h return, beyond the reversal effect in §1.
- Exploiting seconds-scale LOB predictability requires maker execution, colocation-grade latency and VIP or rebate fee tiers. That is market-making infrastructure, not a 5-minute scheduled bot. On a retail taker schedule the predicted move (well below one spread) does not cover costs.

### Gaps
- No peer-reviewed crypto DeepLOB replication with realistic Binance fees and latency was found. The 2020 Coinbase study tests accuracy only.
- The exact out-of-sample R² values for OFI on Binance at 1 s / 10 s / 1 m horizons were not found in readable form. The Schmalz SSRN page was blocked (HTTP 403).

## 3. ML on OHLCV (and similar features) at 15 m / 1 h: realistic OOS directional accuracy and pitfalls

### Takeaway
With proper time splits, out-of-sample directional accuracy for BTC at 1–60 minutes is about 51–56%. It rises with the horizon, to about 0.537 at 15 min and 0.557 at 60 min in the most-cited rigorous study. Long-short strategies built on these forecasts that look profitable before costs turn negative after a 30 bp round-trip cost. Headline accuracies of 80–99% in the literature come from daily data, class imbalance, random splits or look-ahead.

### Cited Findings
- Jaquart, Dann & Weinhardt (2021, J. Finance and Data Science 7:45–66; full text read). Data: minutely BTC data, 2019-03-11..2019-12-10, with a chronological split of about 5 months train, 1 month validation and 3 months test. Targets are balanced by the training-set median return. Features are technical, blockchain, sentiment and asset-based. — [KIT repository](https://publikationen.bibliothek.kit.edu/1000150665)
  - Test accuracy by model and horizon (Table 3): — [KIT](https://publikationen.bibliothek.kit.edu/1000150665)

    | model | 1 min | 5 min | 15 min | 60 min |
    |---|---|---|---|---|
    | GRU | 0.518 | 0.525 | 0.536 | 0.557 |
    | LSTM | 0.519 | 0.525 | 0.532 | 0.560 |
    | GBC | 0.511 | 0.529 | 0.537 | 0.557 |
    | ensemble | 0.515 | 0.526 | 0.538 | 0.558 |
    | logistic regression | 0.511 | 0.518 | 0.520 | 0.539 |

  - All accuracies are highly significant against 50% (the test set has about 95k observations at 1 min). Predictability increases with the horizon.
  - Trading: a long-short strategy on the 1% probability quantiles "generates monthly returns of up to 39% before transaction costs, [but] leads to negative returns after taking transaction costs into account due to the particularly short holding periods". The round-trip cost assumed is 30 bp. Buy-and-hold over the test period returned −29.6%. — [KIT](https://publikationen.bibliothek.kit.edu/1000150665)
- A 2026 review of peer-reviewed BTC prediction evidence cites Arain & Snudden (2025): out-of-sample directional accuracy in the low-to-mid 50s, peaking around 57% for the best predictor. It adds that statistically significant edges often fail to generate economic value after costs (snippet). — [arXiv 2606.00071](https://arxiv.org/html/2606.00071v1)
- An MDPI 2026 multi-timeframe BTC study (snippet) reports a look-ahead bias in multi-timeframe alignment that inflated performance by about 0.20 ROC-AUC before correction. Its 2025 out-of-sample gross return of about 36% over 185 trades becomes negligible after round-trip fees. — [MDPI Forecasting 8(3):40](https://www.mdpi.com/2571-9394/8/3/40)
- A ScienceDirect 2025 study reports XGBoost at 59.4% accuracy on 5-minute direction. The snippet does not show whether the split was temporal, so treat it with caution. — [Eng. Appl. of AI, 2025](https://www.sciencedirect.com/science/article/abs/pii/S0952197625010875)
- Kaggle G-Research Crypto Forecasting (2021–22) had 14 coins and was scored by a weighted Pearson correlation of forecasts with short-term returns. Final scoring ran on 3 months of live data, and the top 3 all used LightGBM. The host called it a "low signal, high noise world", and the winners credited feature engineering over model choice. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
  - A #28 (silver) participant's slides show a single-model private score falling from 0.0202 to 0.0159. — [participant slides](https://docs.google.com/presentation/d/110XGGTDAQlnfa5psJPEi4r1EyB15_fEH/htmlpresent)
  - A third-party Kaggle profile tracker lists 0.03 at rank 2 (unverified whether this is the private leaderboard). — [clist.by](https://clist.by/account/nathanrm/resource/kaggle.com/)
- Lopez de Prado's methods, as implemented by Hudson & Thames, warn that labels and models must go through purged and embargoed splits, because overlapping labels leak information. A practitioner reports that meta-labelling on an already decent classifier gave no gain and was worse, suspecting leakage from training both models on the same data. — [fynance docs](https://fynance.readthedocs.io/en/latest/features.labels.html); [QuantConnect forum](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)

### Inferences
- The realistic out-of-sample ceiling for 15 m–1 h direction from price-derived features is about 52–56% unconditional accuracy. That corresponds to AUC of roughly 0.52–0.58, and it is consistent with the 2026 reversal paper's AUC of about 0.53.
- The G-Research correlations, a few hundredths at best, are a realistic ceiling for a well-resourced crowd on liquid coins. A correlation of about 0.03 between forecast and realised 15 m return implies an expected per-trade edge that is a small fraction of the return's standard deviation.
- Typical failure modes: random or shuffled splits (rows of the same day leak), multi-timeframe look-ahead, class imbalance read as accuracy, and backtests without fees or with very short test windows.

### Gaps
- The official G-Research private-leaderboard winning score was not confirmed. Neither was the exact target definition: it is believed to be a 15-minute-ahead residualised log return, but this is not verified in the sources fetched.
- No large-scale, many-coin, multi-year benchmark of 15 m / 1 h direction with costs was found, other than the 2026 reversal paper, which is effectively one.

## 4. Break-even math: required hit rate / edge given fees and holding period; turnover and cost drag

### Takeaway
At Binance retail fees, a round trip costs about 20 bp on spot as taker, or about 4–10 bp on USDⓈ-M futures (maker/taker), before spread and slippage. Measured gross edges at 15 m are about 0.5–1.3 bp per trade at best. The hit rate needed to break even exceeds what is achievable unless the average absolute move per trade is many times larger than the cost. That pushes toward longer holds and fewer trades, and the published intraday evidence does not show that this rescues profitability.

### Cited Findings
- Binance USDⓈ-M futures regular-user rates are 0.02% maker and 0.05% taker per side, with a 10% BNB discount and lower rates at VIP tiers. — [Binance Futures fee FAQ](https://www.binance.com/support/faq/360033544231); [Finder](https://www.finder.com/cryptocurrency/trading/binance-futures-fees)
- Binance spot regular-user rate is about 0.1% maker and taker per side (secondary source; verify on the official fee table). — [Tradersunion](https://tradersunion.com/brokers/crypto/view/binance/fees/)
- The 2026 reversal paper's spot cost bands are 5 bp cheapest round trip (maker) and 10–20 bp taker. The quoted spread on major Binance pairs is about one tick, so the explicit FEE, not the spread, is what binds. — [arXiv 2608.21888](https://arxiv.org/abs/2608.21888)
- Jaquart et al. used 30 bp round trip, and every short-horizon strategy turned negative after costs "due to the particularly short holding periods". — [KIT](https://publikationen.bibliothek.kit.edu/1000150665)

### Inferences (own derivation; no external source needed for the algebra)
- Symmetric bet with average absolute move m per holding period and round-trip cost c: expected net return = (2p − 1)·m − c. Break-even hit rate is p* = 0.5 + c/(2m).
- Illustrative only; the m values are ASSUMPTIONS, not measured here:

  | case | round-trip cost c | assumed m | break-even p* |
  |---|---|---|---|
  | spot taker, 15 m | 20 bp | 25 bp | 0.90 |
  | spot taker, 1 h | 20 bp | 50 bp | 0.70 |
  | futures taker | 10 bp | 25 bp (15 m) / 50 bp (1 h) | 0.70 / 0.60 |
  | futures maker | 4 bp | 25 bp (15 m) / 50 bp (1 h) | 0.58 / 0.54 |

  - Every case except futures maker at 1 h is above the about 52–56% achievable OOS (§3). That remaining case still ignores adverse selection: maker fills are disproportionately the losing trades.
  - Direction accuracy is also not the same as edge, because hits and misses have different sizes. The edge actually measured, 1.3 bp at best, is the binding number.
- Turnover drag: a strategy holding 15 m and trading 24/7 can make up to 96 round trips a day. At 20 bp each that is 19.2% of notional per day in fees, so even 10% duty cycle is about 1.9%/day. This is the arithmetic behind the trader app's baseline: −0.83%/day on the bot's signals after 0.20% fees (CLAUDE.md, internal).
- Longer horizons raise m faster than they raise predictability, which favours 1 h over 15 m. But §1 shows the directional effect decays to zero by about 4 h, and Jaquart's 60-minute strategies were still negative after 30 bp.

### Gaps
- No source found gives the measured distribution of |15 m| and |1 h| returns for Binance alts over 2025–26. A local computation from the project's `history/*.csv` would replace the assumed m with measured values.

## 5. Supervised vs RL: theory, and studies comparing them on the same data with costs

### Takeaway
Predicting the direction of the next 15 m / 1 h is a supervised problem: the labels are known with a short delay, and a single trade's action does not change the market. RL adds value only where actions change the agent's own future state or costs, such as position and inventory, turnover penalties, execution or path-dependent exits. Even there, its advantage is mostly in folding costs into the objective, which supervised learning plus an explicit decision rule (cost-aware threshold, meta-labelling) can also do. RL brings higher variance, sample inefficiency and a large backtest-overfitting risk. No rigorous out-of-sample, after-cost evidence was found that RL beats supervised learning plus a simple decision rule on intraday crypto.

### Cited Findings
- Moody & Saffell (2001, IEEE Trans. Neural Networks) "Learning to trade via direct reinforcement": recurrent RL that optimises a differential Sharpe ratio net of transaction costs "eliminates the need to build forecasting models" and produced better strategies than Q-learning or forecast-based trading in their simulations. These were an intraday FX trader and a monthly S&P/T-bill allocator, i.e. 1990s data (abstract/snippet). — [UT Austin reading list entry](https://cs.utexas.edu/~shivaram/readings/b2hd-MoodySaffell2001.html); [Moody talk summary 2001](https://mailman.cs.uchicago.edu/mailman/public/colloquium/2001-February/000039.html)
- Moody & Saffell (NIPS 1998): RL is "a more elegant and effective means for training trading systems when transaction costs are included, than do more standard supervised approaches". The core argument is that the supervised loss (forecast error) is not the trading objective once costs make the optimal action depend on the current position. — [NIPS 1998](https://papers.nips.cc/paper/1998/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html)
- Counter-evidence (Expert Systems with Applications, 2022, "Outperforming algorithmic trading reinforcement learning systems: a supervised approach to the cryptocurrency market"): a supervised ResNet-LSTM actor beat recurrent RL, DQN and A2C on BTC, LTC, ETH, XMR, NXT and DASH. It was evaluated with and without costs. Costs were handled by limiting trade frequency and using cumulative future returns as labels. The same paper notes that Deng et al. (2017) found the opposite (RL better under costs), so the evidence is contested (abstract/snippet). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417422006339)
- Gort et al., "Deep RL for cryptocurrency trading: practical approach to address backtest overfitting" (arXiv 2209.05559): prior DRL crypto papers may report inflated profits. They propose a hypothesis test for the probability of overfitting and reject overfitted agents. The less-overfitted agents beat more-overfitted ones and the benchmarks on 10 coins, but over a test window of May–June 2022, i.e. a few weeks (snippet). — [arXiv 2209.05559](https://arxiv.org/abs/2209.05559)
- FinRL's crypto PPO example used a 10-day test window: PPO about 103% cumulative vs equal-weight about 99% and buy-and-hold about 93%. — [FinRL arXiv 2111.09395](https://arxiv.org/pdf/2111.09395)
- An informal Q-learning experiment on 15 m BTC/USDT (blog, not peer reviewed): out of sample, 9 of 10 agents stayed flat and the one that traded lost 5.7%. The author argues "no trade" is a rational outcome once costs and noise enter. — [QuantJourney substack](https://quantjourney.substack.com/p/reinforcement-learning-for-trading)
- A Stanford CS224R student project (supervised LSTM feeding a PPO agent on small-cap coins) gives no head-to-head comparison against pure supervised learning. — [CS224R report](https://cs224r.stanford.edu/projects/pdfs/CS224R_Report12.pdf)
- Meta-labelling (Lopez de Prado, AFML): a primary model picks the side and a secondary classifier estimates P(primary signal succeeds), then filters and sizes. It is the supervised route to "when to act and how much", the decision layer that RL proponents claim. — [Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy/)

### Inferences
- Conceptual split:
  - Supervised learning estimates E[r_{t+h} | x_t] or P(up | x_t). A decision rule then trades only when the predicted edge exceeds cost: enter if p̂·m − (1−p̂)·m > c, which is the §4 formula.
  - RL learns the policy directly. It earns its keep when (a) the action changes the state (holding a position changes whether the next trade costs a fee), (b) the reward is delayed or path-dependent (trailing exits, holding time), or (c) execution affects price.
  - For a one-step directional bet with no inventory interaction, the Bayes-optimal policy is just thresholding the supervised forecast against cost. RL then solves the same problem with a noisier signal: the reward is a realised P&L with variance much greater than its mean, so sample efficiency is far worse.
- Where the signal-to-noise ratio is as low as in §1–3 (AUC about 0.53, edge about 1 bp), RL's extra variance and its tendency to overfit backtests make it strictly harder to tell an edge from noise. If there is no supervised edge after costs, RL cannot create one: it can only learn to trade less, which is the "stay flat" outcome.
- Where RL is legitimately the right framing: exit and holding policy (the trail, max-hold) and execution (maker vs taker, order slicing). These are sequential decisions with path-dependent reward, and the project's exit experiments live there. Even then, an offline contextual bandit or supervised exit model on replayed trades is the lower-variance first step, as the project already does with `exit_validator`.

### Gaps
- No peer-reviewed study was found (2014–2026) that compares RL with a supervised forecast plus a cost-aware decision rule on the SAME intraday crypto data, with a TIME split, realistic Binance fees and a multi-month out-of-sample window, and in which RL wins. The 2022 ESWA paper (supervised wins) and Gort et al. are the closest, and their full methodology and cost levels were not verified in full text here.
- Deng et al. (2017, "Deep direct reinforcement learning for financial signal representation and trading", IEEE TNNLS) was not fetched. Its claim that RL beats supervised under costs is cited second-hand via the ESWA paper, on Chinese futures rather than crypto.
- Funding rate / open interest at intraday horizons: no peer-reviewed out-of-sample study was found showing that funding or OI predicts 15 m–1 h direction. Funding is an 8-hour, time-weighted-average quantity sitting at 0.01% most of the time (e.g. 78% of Q3-2025 on BitMEX BTC, per a non-academic source), so it is a slow carry variable, not a direction signal. — [Coincub](https://coincub.com/blog/perpetual-futures-funding-rates-78-of-a-quarter-at-0-01/); [Coin Metrics funding docs](https://docs.coinmetrics.io/market-data/market-data-overview/funding-rates/funding-rates). The project's own tests point the same way: R-11 positioning refuted, only oi_1h/oi_4h weakly consistent (CLAUDE.md, internal).
- Cross-asset lead-lag (BTC leading alts at minutes): no external rigorous after-cost study was found in this pass. The project's internal test (2026-10-03) found the moves happen in the same 15 m bar (corr 0.80–0.86, next-bar lead about 0), which is consistent with the absence of tradable lead-lag at bar frequency.
