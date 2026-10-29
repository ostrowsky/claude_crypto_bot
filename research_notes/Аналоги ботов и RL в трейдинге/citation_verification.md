# Citation verification: crypto trading bots and RL report (as of 2026-10-29)

Verdict scale: CONFIRMED / PARTLY (what differs) / NOT FOUND / CONTRADICTED.
Summary: 9 CONFIRMED, 5 PARTLY, 1 NOT FOUND (the numbers in claim 3). No arXiv ID resolved to a different paper.

## 1. Kitron & Wengrowicz, arXiv 2608.21888

### Takeaway
CONFIRMED. The ID resolves to the named paper, and every number in the claim is in the full text.

### Cited Findings
- Title "Short-horizon mean reversion in cryptocurrency markets: a matched cross-market measurement". Nadav A. Kitron and Jonathan M. Wengrowicz, submitted 22 Aug 2026 — [arXiv abs](https://arxiv.org/abs/2608.21888)
- Abstract: "90% of 183 Binance pairs carry significant directional reversal against 2.7% of 187 US stocks and ETFs". Also: "The gross edge peaks near 1.3 bp per trade against a 5 bp round-trip cost" — [arXiv abs](https://arxiv.org/abs/2608.21888)
- Full text — [arXiv HTML](https://arxiv.org/html/2608.21888):
  - Data: 15-minute candles, 2025-01-01..2026-02-11; holdout 2026-02-12..2026-08-08.
  - Mean crypto OOS AUC 0.531 in the primary window and 0.522 on the six-month holdout (stocks 0.501).
  - "not one of the 183 crypto pairs clears even the 5 bp maker band at any threshold"; the taker band is 10–20 bp.
  - The signal decays and is "gone by four hours".

### Inferences
- "≥5 bp round trip" is the cheapest (maker) band. Taker costs are 2–4× higher.

### Gaps
- None.

## 2. Jaquart, Dann & Weinhardt, JFDS 2021

### Takeaway
CONFIRMED. The accuracies are the ensemble row "E (All)" of Table 3, rounded correctly. The cost is 30 bp per round trip.

### Cited Findings
- Paper details: J. Finance and Data Science vol. 7, pp. 45–66 (2021), DOI 10.1016/j.jfds.2021.03.001 — [KITopen record](https://publikationen.bibliothek.kit.edu/1000150665)
- Table 3, "E (All)" row: 0.514626 (1 min), 0.526092 (5), 0.537863 (15), 0.557579 (60). The best single models are a little higher: LSTM 0.5193 at 1 min, GBC 0.5293 at 5 min, LSTM 0.5601 at 60 min — [PDF](https://publikationen.bibliothek.kit.edu/1000150665/149388567)
- Costs, from the same PDF:
  - "we assume round-trip transaction costs of 30 basis points".
  - After costs "trading performance becomes negative for all methods".
  - Abstract: "monthly returns of up to 39% before transaction costs".

### Inferences
- The claim's 0.515/0.526/0.538/0.558 are the ensemble values, not the best model's values. That is fine if it is labelled "ensemble".

### Gaps
- None.

## 3. Schmalz, SSRN 7227998

### Takeaway
PARTLY. The paper exists and matches on the main points: Binance.US, OFI, sign flips as the sample grows, and an AR benchmark that matches OFI. The R² of ~0.1–1% could not be checked.

### Cited Findings
- From the search index, since the SSRN pages returned HTTP 403 to WebFetch:
  - Title "Order Flow Imbalance and Short-Horizon BTC/USDT Returns: A Signal That Kept Needing More Scrutiny", Michael Schmalz. SSRN lists it as dated 29 Jul 2026, posted 4 Aug 2026.
  - It uses Binance.US order-book data. Results were positive on ~7 days, null or negative on ~9 days, and positive again on ~17 days of data.
  - "a simple autoregressive benchmark matches or beats" OFI at most horizons, and trade-flow imbalance is never significant.
  - Source: [SSRN delivery link](https://papers.ssrn.com/sol3/Delivery.cfm/7227998.pdf?abstractid=7227998&mirid=1)

### Inferences
- The claim's "AR(1) matches" fits "matches or beats". Whether the benchmark is exactly AR(1) is unverified.

### Gaps
- R² ~0.1–1%: NOT FOUND. The primary page is blocked (403), and the search snippets give no R² values.

## 4. arXiv 2602.00776

### Takeaway
PARTLY. The paper and the result exist, but "lose to buy-and-hold" is right only in a relative sense. BTC and LTC taker returns are positive. The fee level is not stated.

### Cited Findings
- Title "Explainable Patterns in Cryptocurrency Microstructure", Bieganowski & Ślepaczuk, 31 Jan 2026. Data: Binance Futures perpetual order books at 1-second frequency, assets BTC, LTC, ETC, ENJ and ROSE — [arXiv abs](https://arxiv.org/abs/2602.00776)
- Table 1 annualized return, taker strategy vs buy-and-hold — [arXiv HTML](https://arxiv.org/html/2602.00776):

  | Asset | Taker | Buy-and-hold |
  |---|---|---|
  | BTC | 0.13 | 0.83 |
  | LTC | 0.07 | 0.53 |
  | ENJ | 4.06 | −0.67 |
  | ETC | 5.78 | −0.10 |
  | ROSE | 7.00 | −0.71 |

- The backtest crosses the spread: "Buys are executed at the best ask and sells at the best bid". Fees are only "Optional proportional costs can be introduced to reflect taker fees", with no rate given. Latency is not modelled ("an upper bound") — [arXiv HTML](https://arxiv.org/html/2602.00776)

### Inferences
- Correct wording: "BTC/LTC taker strategies stay positive but trail buy-and-hold". "With fees" cannot be confirmed.

### Gaps
- Whether Table 1 includes fees is unclear in the paper.

## 5. arXiv 2504.02281 (FinRL Contests)

### Takeaway
PARTLY. The claim mixes two tables. 0.66% vs 0.74% is the paper's own ensemble experiment (Table 7). The Sharpe values 0.0037 and −0.0008 are contest winners (Table 8), whose returns are 0.23% and −0.05%.

### Cited Findings
- Title "FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents", Wang, Holzer, … Liu Yanglet. v1 3 Apr 2025, v4 15 Jul 2025 — [arXiv abs](https://arxiv.org/abs/2504.02281)
- Table 7, the paper's ensemble experiment — [arXiv HTML](https://arxiv.org/html/2504.02281):
  - Ensemble-1/2/3: return 0.66%, Sharpe 0.28.
  - Individual DQN: 0.34%, Sharpe 0.15.
  - BTC price: 0.74%, Sharpe 0.20.
- Table 8, contest winners — [arXiv HTML](https://arxiv.org/html/2504.02281):
  - Fermion (2024): 0.23%, Sharpe 0.0037, max drawdown −0.28%.
  - Mt.Everest (2025): −0.05%, Sharpe −0.0008, max drawdown −1.23%.
  - BTC baseline: −7.35%, Sharpe −0.0022. The test window is longer, re-partitioned.

### Inferences
- In Table 8 the 2024 winner beat a falling BTC (−7.35%). Only the 2025 winner was negative.
- Pairing 0.66% with Sharpe 0.0037 is wrong. Report Table 7 and Table 8 separately.

### Gaps
- None.

## 6. Gort et al. 2022, arXiv 2209.05559

### Takeaway
CONFIRMED.

### Cited Findings
- Title "Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting". Test set: 10 cryptocurrencies, 05/01/2022–06/27/2022, "the crypto market crashed twice" — [arXiv abs](https://arxiv.org/abs/2209.05559)
- Table 3 cumulative returns — [arXiv PDF](https://arxiv.org/pdf/2209.05559):
  - S&P BDM Index −50.78%; equal-weight −47.78%.
  - PPO (CPCV, the accepted agent) −34.96%, probability of overfitting 8.0%.
  - Other agents: PPO WF −49.39%, PPO KCV −55.54%, TD3 −59.08%, SAC −59.48%.

### Inferences
- The "best" agent still lost about 35% in absolute terms. It only lost less than the index.

### Gaps
- None.

## 7. ESWA 2022: supervised model beats RRL, DQN and A2C

### Takeaway
Paper identified, PARTLY verified. The paper is Felizardo et al., "Outperforming algorithmic trading reinforcement learning systems: A supervised approach to the cryptocurrency market", Expert Systems with Applications vol. 202 (2022), article 117259, DOI 10.1016/j.eswa.2022.117259. The abstract and highlights confirm the setup. The numbers for the runs with and without costs were not read.

### Cited Findings
- Identification and setup, from search-indexed abstract and highlights:
  - RSLSTM-A (ResNet-LSTM actor) against recurrent RL, DQN and A2C.
  - Six coins: Bitcoin, Litecoin, Ethereum, Monero, Nxt and Dash.
  - "better overall performance"; the highlights mention a robustness evaluation with and without transaction costs.
  - Source: [datalearner index](https://www.datalearner.com/academic/journal-papers/0957-4174/volumes-and-issues/406/paper-detail/18673); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417422006339) (403 to fetch)

### Inferences
- "Wins both with and without costs" is consistent with the highlights, but was not checked against the result tables.

### Gaps
- ScienceDirect returned 403. A fetch of the PoliTo repository was denied by the tool permission classifier, and the Medium summary returned 403. No result tables were read.

## 8. Chi, Chu & Hao, arXiv 2411.06327

### Takeaway
CONFIRMED, with a nuance. The abstract says "at multiple intervals" within the 1–6 h frequencies tested, not at every interval.

### Cited Findings
- Title "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets", Yeguang Chi, Qionghua (Ruihua) Chu, Wenyan Hao. v1 10 Nov 2024, v2 1 Sep 2025. Data 2017–2023 at 1–6 hour frequencies. Abstract: "USDT net inflows into the exchanges, positively predicts BTC and ETH returns at multiple intervals" — [arXiv abs](https://arxiv.org/abs/2411.06327)

### Inferences
- None.

### Gaps
- Which of the 1–6 h intervals are significant (full text not read).

## 9. arXiv 2510.14435

### Takeaway
CONFIRMED. All three numbers are in the paper. Note the paper is a broad asset-class survey, not a carry paper.

### Cited Findings
- Title "Cryptocurrency as an Investable Asset Class: Coming of Age", Borri, Liu, Tsyvinski, Wu. v1 16 Oct 2025, v4 21 Mar 2026 — [arXiv abs](https://arxiv.org/abs/2510.14435)
- Stylized Fact 6, carry (short perpetual, long spot) — [arXiv HTML](https://arxiv.org/html/2510.14435):
  - "the annualized Sharpe ratio of the cryptocurrency carry is 6.45", sample Aug 1 2020–May 31 2025.
  - "Beginning in 2024, the Sharpe ratio falls to 4.06, and it turns negative in 2025".
  - Profits are "mostly driven by the funding rate".

### Inferences
- "2025" here means Jan–May 2025 only, because the sample ends 31 May 2025.

### Gaps
- None.

## 10. arXiv 2607.27070

### Takeaway
CONFIRMED. The numbers are in the full text, not the abstract. "Out-of-sample" here means a different event (Aug 2024), not a later time split.

### Cited Findings
- Title "Where does the criticality live? Early-warning signals are event-heterogeneous across seven crypto-perpetual liquidation cascades", Ramon Marc Garcia Seuma, 29 Jul 2026. The abstract does not mention OI — [arXiv abs](https://arxiv.org/abs/2607.27070)
- Table 2, lag-1 autocorrelation of open interest — [arXiv HTML](https://arxiv.org/html/2607.27070):
  - Positive and significant in 23/39 configurations in-sample (Oct 2025 event), median τ +0.053.
  - Only 3/39 out-of-sample (Aug 2024 event), median τ −0.033.
  - Text: "open interest (23/39)" vs "(3/39, 4/39, 1/39)".
- What the paper treats as robust is "a compression of taker order-flow variance" — [arXiv abs](https://arxiv.org/abs/2607.27070)

### Inferences
- The signals are early-warning statistics for cascades, not trading signals.

### Gaps
- None.

## 11. ACM CHI 2024 "Stranger Danger?"

### Takeaway
CONFIRMED, with a nuance. The "about a week then negative" finding is about Bybit portfolios after promotion to the Gold leader level, not leaderboard promotion in general.

### Cited Findings
- Title "Stranger Danger? Investor Behavior and Incentives on Cryptocurrency Copy-Trading Platforms", Kawai, Soska, Routledge, Zetlin-Jones, Christin. CHI '24, DOI 10.1145/3613904.3642715. Data: TraderWagon and Bybit, Oct 2022–Aug 2023 — [CMU PDF](https://www.andrew.cmu.edu/user/nicolasc/publications/Kawai-CHI24.pdf)
- Quote: "For about one week, copiers make a profit, but their PnL plummets and even goes negative". The bottom 25% of Gold portfolios lose more than USD 15,000 within ten days of promotion — [CMU PDF](https://www.andrew.cmu.edu/user/nicolasc/publications/Kawai-CHI24.pdf)

### Inferences
- None.

### Gaps
- None.

## 12. arXiv 2506.11921

### Takeaway
CONFIRMED, with a nuance. The paper says "traditional" grid trading under "simple assumptions", not "static".

### Cited Findings
- Title "Dynamic Grid Trading Strategy: From Zero Expectation to Market Outperformance", Chen, Chen, Jang, 13 Jun 2025. The expected return of traditional grid trading is "essentially zero"; the paper's DGT is backtested on BTC and ETH minute data 2021-01..2024-07 — [arXiv abs](https://arxiv.org/abs/2506.11921)

### Inferences
- None.

### Gaps
- None.

## 13. CFTC customer advisory, 2024-01-25

### Takeaway
CONFIRMED.

### Cited Findings
- Advisory title "Customer Advisory: AI Won't Turn Trading Bots into Money Machines" — [CFTC advisory](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/AITradingBots.html)
- Release 8854-24, dated January 25, 2024, announces the advisory — [CFTC press release](https://www.cftc.gov/PressRoom/PressReleases/8854-24)

### Inferences
- None.

### Gaps
- None.

## 14. Freqtrade FreqAI RL docs and QuantConnect Lean slippage

### Takeaway
CONFIRMED for both tools.

### Cited Findings
- Freqtrade:
  - On the default reward: "This is a benchmark, it is _not_ for live production".
  - Current profit, position and trade duration feed the agent in dry/live runs, "(this functionality is not available in backtesting)".
  - Source: [Freqtrade docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/)
- QuantConnect: "The Binance brokerage models use the NullSlippageModel", which means zero slippage — [QuantConnect Binance model](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/brokerages/supported-models/binance); [slippage models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/supported-models)

### Inferences
- Only the Binance model page was read. Applying this to all crypto brokerages is an extrapolation.

### Gaps
- Other crypto brokerage models (Coinbase, Bybit, …) were not checked.

## 15. Binance fees

### Takeaway
Spot: CONFIRMED. USDⓈ-M futures: PARTLY.

### Cited Findings
- Spot, Regular/VIP 0: 0.100% maker / 0.100% taker (0.075% with the BNB discount) — [Binance fee schedule](https://www.binance.com/en/fee/schedule)
- USDⓈ-M futures:
  - The fee table page is rendered by JavaScript and returned no table to the fetch.
  - The Binance FAQ "Binance Futures Fee Structure & Fee Calculations" (updated 2026-05-01) gives "a Regular User's maker fee is 0.02% and a Regular User's taker fee is 0.05%". It calls these "hypothetical fee rates" in a worked example and points to the fee table.
  - Source: [Binance FAQ](https://www.binance.com/en/support/faq/360033544231); [futures fee table](https://www.binance.com/en/fee/futureFee) (no data in fetch)

### Inferences
- 0.02% / 0.05% matches Binance's own example and third-party listings. It is not confirmed from the live table.

### Gaps
- The live USDⓈ-M VIP 0 table was not read.
