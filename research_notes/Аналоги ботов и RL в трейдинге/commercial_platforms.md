# Commercial crypto trading bot platforms (2025-2026): feature inventory and evidence on outcomes

Research date: 2026-10-29. About 18 web searches and fetches. Most pricing and feature data comes from third-party
review and affiliate sites, which often conflict with each other. Every price below is **unverified against the
vendor's official pricing page**, unless it is marked otherwise. Several sources are competitor blogs
(Bitsgap, Gainium, WunderTrading, 3Commas), and they are marked where they are used.

## Which bot families dominate, and which features are table stakes?

### Takeaway
Grid, DCA/martingale, TradingView-webhook signal bots, rebalancing and copy trading make up the common core of
almost every platform, exchange-native or third-party. Exchanges add funding-rate arbitrage and TWAP/VP execution
algorithms. Third-party platforms compete on non-custodial access to many exchanges, a scripting language
(HaasScript, Gunbot), or price (Pionex has free bots and charges trading fees instead).

### Cited Findings

**Feature matrix (rows are platforms; each cell gives its source)**

| Platform | Bot families | Custody / model | Pricing (unverified) | AI claims | Status |
|---|---|---|---|---|---|
| **Binance (exchange-native)** | Spot Grid, Futures Grid, Spot DCA, Futures DCA, Rebalancing Bot, Futures Funding Rate Arbitrage Bot, TWAP and VP algo tools — [Binance Trading Bots Terms](https://www.binance.com/support/faq/d5a7e374026f4f19a9c1aa0ae226c3ca). Futures Grid allows leverage, up to 125x per a Binance Square post (not official docs), and shorts — [Binance Square](https://www.binance.com/ar/square/post/316727487549874) | Custodial (exchange) | Bots free; standard trading fees. The bot guide says its profit calculation **excludes fees** — [Binance Square #gridbot](https://www.binance.com/square/hashtag/GridBot) | Bot marketplace / "Best PnL" ranking — [DEV Community](https://dev.to/aduenasdev/i-backtested-the-most-popular-binance-grid-bots-on-4-years-of-real-data-1gnb) | Active |
| **Bybit (exchange-native)** | Spot Grid and Advanced DCA (2022 launch; minimum $3 grid, $1 DCA); futures grid with copy trading of Master Traders' grid settings (Dec 2023); bot leaderboard — [InvestingCube](https://www.investingcube.com/cryptocurrency/bybit-launches-upgraded-spot-grid-trading-and-dca-trading-bots/), [FinanceFeeds](https://financefeeds.com/bybit-revamps-grid-trading-with-more-bots-and-features/), [AAP/Cision](https://www.aap.com.au/aapreleases/cision20231215ae94883/) | Custodial | Not found | The grid bot "uses AI to establish parameters" (press release) — [Guardian.ng](https://guardian.ng/?p=1861811) | Active |
| **OKX (exchange-native)** | Spot grid, futures grid, spot and futures DCA (martingale), smart arbitrage, recurring buy, signal bot, infinity grid, smart portfolio. Futures DCA scales up after losses and can hold long and short at once — [third-party OKX help mirror](https://okx-api.aperture.finance/help/what-are-okxs-crypto-trading-bots-and-how-do-i-utilize-it) | Custodial | Not found | Not found | Active |
| **Pionex** | 16 bots per most 2026 reviews (12 per Gainium): grid, DCA, arbitrage, plus the PionexGPT assistant — [agent-finder](https://agent-finder.co/reviews/pionex), [Gainium](https://gainium.io/review/pionex) | **Custodial** (funds sit on Pionex) — [cointech2vip](https://www.cointech2vip.com/pages/bitsgap-vs-pionex) | Bots free; spot 0.05% maker and taker; futures 0.02% / 0.05% — [airdropalert](https://airdropalert.com/blogs/pionex-review/) | PionexGPT (strategy-setup assistant) | Active |
| **3Commas** | DCA, Grid, Signal (TradingView alerts), SmartTrade, Options bot (on the free tier per one source) — [Capterra](https://www.capterra.com/p/10030799/3Commas/), [GetApp](https://www.getapp.com/all-software/a/3commas) | Non-custodial (exchange API keys) | Capterra: Pro ~$40/mo, Expert ~$110/mo (annual billing), 7-day trial; other sources give $37-59/mo or a free Starter plan — [Capterra](https://www.capterra.com/p/10030799/3Commas/), [Gainium](https://gainium.io/review/3commas) (competitor) | Conversational AI that builds a bot and backtest from plain English — [Capterra](https://www.capterra.com/p/10030799/3Commas/); "QuantPilot" agentic AI platform reportedly launched 2026-04-02 — [stork.ai](https://www.stork.ai/en/3commas) (single source, unverified) | Active; 2022 API-key breach (see the security section) |
| **Bitsgap** | Grid, DCA, futures bots; multi-exchange; non-custodial | Non-custodial — [cointech2vip](https://www.cointech2vip.com/pages/bitsgap-vs-pionex) | Basic $23-29/mo, Advanced $69, Pro $119-149/mo, 7-day trial — figures conflict across [CaptainAltcoin](https://captainaltcoin.com/pionex-vs-bitsgap/), [Wallet Reviewer](https://walletreviewer.com/bitsgap-vs-pionex-comparison/), [Ventureburn](https://ventureburn.com/best-crypto-trading-bot/) | Not established | Active |
| **Cryptohopper** | Copy bots (unlimited on the free plan), strategy designer and marketplace on paid tiers — [Koinly](https://koinly.io/blog/best-crypto-trading-bots/) | Non-custodial | Free plan; paid $24-107.50/mo; other 2025 sources start at $19 or $29 — [Koinly](https://koinly.io/blog/best-crypto-trading-bots/) | "AI" strategy marketing (details not verified) | Active |
| **HaasOnline** | HaasScript scripting language, 1,500+ indicators per one source — [MEXC news roundup](https://www.mexc.com/news/1113070) | Non-custodial / self-hosted options | Inconsistent: custom quote vs ~$9-149/mo vs BTC-denominated quarterly — roundups above | n/a | Active (pricing unclear) |
| **Gunbot** | Scriptable, self-hosted bot; GunBotShop marketplace | Self-hosted | One-time licence; ranges conflict ($59-249 vs $199-499 vs a £99 licence) — [Koinly](https://koinly.io/blog/best-crypto-trading-bots/), [Tangem](https://tangem.com/en/blog/post/best-ai-crypto-trading-bots/) | n/a | Active |
| **Kryll** | Drag-and-drop strategy builder, marketplace, free backtesting | Non-custodial | Pay-per-use in KRL token, ~$1-50/mo — roundups above | n/a | Status not confirmed in this search |
| **Altrady** | Signal, Grid and DCA bots, TradingView webhooks, backtesting, portfolio; **MCP server** so LLM assistants (e.g. Claude) can scan markets and execute and manage trades — [Cornix (competitor)](https://cornix.io/altrady-alternatives/) | Non-custodial | Not found | MCP / LLM integration | Active |
| **WunderTrading** | TradingView signal bots (custom webhooks, spot and futures, multi-API), Multi-Pair Grid Bot that auto-selects pairs — [Finestel](https://finestel.com/blog/wundertrading-review/), [WunderTrading journal (vendor)](https://wundertrading.com/journal/en/reviews/article/best-mudrex-alternatives) | Non-custodial | Not found | "Profit-optimized pair selection" (marketing) | Active |
| **Mudrex** | "Coin Sets" (curated thematic portfolios), automated trading, backtesting, strategy marketplace — [SoftwareSuggest](https://www.softwaresuggest.com/mudrex), [SourceForge](https://sourceforge.net/software/product/Mudrex/) | Custodial (India-focused; not verified here) | Not found | Generic "AI-type tools" | Active (pivoted toward investing) |
| **Shrimpy** | Was: rebalancing, social copy trading, DCA | — | — | — | **Retail platform discontinued.** Shutdown announced mid-2023 (final transactions 2023-07-07); acquired by Bitcoin IRA 2023-11-15 — [Brave New Coin](https://bravenewcoin.com/insights/crypto-trading-bots-in-trouble), [CoinCodeCap](https://coincodecap.com/shrimpy-crypto-trading-bot-review); developer API shut down — [Shrimpy help](https://help.shrimpy.io/hc/en-us/articles/4412537448219-Shrimpy-Developer-API-Shutdown). One review still calls it operational — [InvestingBrokers](https://investingbrokers.com/shrimpy-review/) (conflict) |
| **Coinrule, TradeSanta** | Rule-based "if-this-then-that" (Coinrule); grid and DCA (TradeSanta) | Non-custodial | Not found | — | Status not confirmed (no search hits) |

- The Binance arbitrage bot is a funding-rate carry: it holds opposite positions in perpetual futures and spot to collect funding fees while hedging price risk — [Binance blog](https://www.binance.com/en/blog/futures/binance-trading-bots-characteristics-and-use-cases-4987930405804222569)
- TWAP and VP are order-execution algorithms for splitting large orders to reduce market impact, not alpha strategies — [Binance Square](https://www.binance.com/ar/square/post/316727487549874); [Binance terms](https://www.binance.com/support/faq/d5a7e374026f4f19a9c1aa0ae226c3ca)
- Cost trade-off: per Bitsgap's own (competitor) blog, a bot trading $50k/month on Pionex pays ~$25 in fees at 0.05%, so five such bots (~$125/mo) can cost more than a flat subscription — [Bitsgap blog](https://bitsgap.com/blog/best-pionex-alternatives-in-2026-keep-the-bots-choose-your-exchange)

### Inferences
- **Table stakes in 2025-2026:** spot grid, spot DCA (with safety orders / martingale scaling), TradingView-webhook
  signal bots, a "smart trade" (entry with TP/SL ladders and trailing), rebalancing, and copy trading or a bot
  marketplace with leaderboards. A platform without these is not competitive.
- **Differentiators:** futures grid and DCA with leverage, funding-rate arbitrage, scripting (HaasScript, Gunbot),
  multi-exchange non-custodial access, and since 2025-26 LLM or agent front-ends (3Commas conversational AI,
  PionexGPT, Altrady MCP).
- **No commercial platform found sells an entry-signal model judged by a time-split holdout or a published
  out-of-sample record.** Products are execution templates (grid, DCA, trail) whose outcome depends on the
  parameters and the market regime the user picks.
- For this project: the demo trader in `trader/` already covers what TWAP/VP-type execution and the funding-carry
  bot do (FC_PASSIVE is the Binance "funding rate arbitrage bot" idea). The signal bot's lane, early detection of
  day leaders, has no commercial equivalent that publishes honest evidence.

### Gaps
- Official current pricing pages were not fetched for any vendor. All prices come from third-party roundups, and
  they conflict.
- The current status of Kryll, Coinrule, TradeSanta and HaasOnline was not confirmed by any 2025-26 source in
  this search. Mudrex custody was not verified.
- No Bybit 2025-26 bot lineup was found (the sources are 2022-23 press releases). No official OKX doc was found
  (only a third-party mirror). OKX and Bybit copy-trading feature details were not retrieved.

## What do "AI" features actually do on these platforms?

### Takeaway
On the evidence found, "AI" on these platforms means four things: parameter auto-fill (grid range and step from
recent volatility), natural-language bot builders or LLM assistants, marketplaces and leaderboards of other
users' bots, and lately LLM agent connectors (MCP). None of the evidence found shows a learned predictive model
with published out-of-sample performance.

### Cited Findings
- Bybit's grid bot "uses AI to establish parameters" to optimise ROI. That is a press-release claim and the
  method is not disclosed — [Guardian.ng (Bybit PR)](https://guardian.ng/?p=1861811)
- 3Commas: conversational AI that builds a strategy from plain English and generates a setup and backtest (e.g.
  "create a DCA bot for BTC") — [Capterra](https://www.capterra.com/p/10030799/3Commas/); "QuantPilot" agentic AI
  platform reportedly launched 2026-04-02 — [stork.ai](https://www.stork.ai/en/3commas) (single source)
- Pionex ships a "PionexGPT" assistant alongside its bots — [agent-finder](https://agent-finder.co/reviews/pionex)
- Altrady runs an MCP server so LLM assistants can scan markets, analyse charts and execute and manage trades on
  connected exchanges — [Cornix](https://cornix.io/altrady-alternatives/)
- WunderTrading's Multi-Pair Grid Bot "scans the market, selects the best profit-optimized trading pairs" —
  [Finestel](https://finestel.com/blog/wundertrading-review/) (marketing description)
- Wider 2025 agent launches: OpenAI ChatGPT Agent (2025-07-17), used by traders for research, charting and
  sentiment — [Cointelegraph via TradingView](https://www.tradingview.com/news/cointelegraph:b1f54b5f0094b:0-how-to-use-chatgpt-agent-for-crypto-trading-in-2025/);
  Coinbase's tool letting an AI agent operate a trading account through LLM natural-language commands —
  [Yahoo Finance](https://finance.yahoo.com/markets/crypto/articles/let-ai-agent-trade-crypto-091200660.html);
  Crypto.com's LLM-integrated MCP — [Crypto.com](https://crypto.com/us/company-news/cryptocom-debuts-first-ai-llm-integrated-mcp);
  the open-source TradingAgents multi-LLM framework — [GitHub](https://github.com/tauricresearch/tradingagents)
- A Stevens Institute project testing grid bots on Bybit found that ML integration did **not** outperform
  traditional methods — [Stevens FSC](https://fsc.stevens.edu/cryptocurrency-market-making-improving-grid-trading-strategies-in-bitcoin/)
- CFTC: "AI technology can't predict the future or sudden market changes" — [CFTC Release 8854-24](https://www.cftc.gov/PressRoom/PressReleases/8854-24)

### Inferences
- "AI" is mostly a UX layer: it lowers the cost of configuring a template. It does not change the template's
  expectancy. In truth-harness terms (TH-11), none of these AI features is judged by a business outcome.
- The 2025-26 trend is LLM agents and MCP connectors that act on accounts. Their value is convenience. Their risk
  is autonomous order placement with no validated edge.

### Gaps
- No vendor documentation was found that explains the Bybit or Pionex "AI" parameter method, or any
  out-of-sample record for the 3Commas AI or PionexGPT recommendations.
- Sentiment-based bot features were not verified on any named platform.

## Risk-management features: position sizing, drawdown stops, trailing, safety orders, portfolio limits

### Takeaway
Per-bot controls are universal: SL/TP, trailing TP and trailing up for grids, DCA safety-order count and step
scaling, leverage caps. Portfolio-level controls are rare, and leaderboards hide tail risk: Binance shows only a
7-day drawdown next to "Best PnL" rankings. Martingale scaling is offered as a standard bot type by major
exchanges.

### Cited Findings
- OKX Futures DCA (Martingale) "increases your investment after each losing trade" and can hold long and short at
  once — [OKX help mirror](https://okx-api.aperture.finance/help/what-are-okxs-crypto-trading-bots-and-how-do-i-utilize-it)
- Binance Futures Grid allows leverage (up to 125x per a Binance Square post) and shorts —
  [Binance Square](https://www.binance.com/ar/square/post/316727487549874)
- The Binance bot marketplace ranks by "Best PnL" and shows only a 7-day drawdown. The independent backtester
  calls this survivorship-biased: users see only the top performers, so the real risk is hidden —
  [DEV Community](https://dev.to/aduenasdev/i-backtested-the-most-popular-binance-grid-bots-on-4-years-of-real-data-1gnb)
- Copy-trading ROI and PnL rankings mislead: a high PnL with a low ROI only means a large capital base, and the
  top ROI figures are snapshots on small samples — [CriptoNoticias](https://criptonoticias.com/finanzas/estrategias-mas-rentables-trading-bots-ia-binance-okx-bybit-okcoin)
- 3Commas caps bots per plan (Expert: 5,000 grid, 1,000 DCA, 1,000 signal bots, 5,000 SmartTrades, 15 accounts) —
  [Capterra](https://www.capterra.com/p/10030799/3Commas/). These are usage caps, not risk limits.

### Inferences
- The gap that matters: no platform found offers portfolio-level risk controls (cross-bot exposure limits,
  account-level drawdown kill switch, correlation caps) as a documented, default feature. This project's
  `correlation_guard`, catastrophe stop and the trader's PAUSE/FLATTEN controls go beyond what was documented.
- Martingale or DCA with leverage plus rankings by recent PnL is a structure that selects for hidden tail risk.

### Gaps
- Official docs on Binance/3Commas trailing TP, trailing up/down and account-level stop features were not
  fetched; feature presence is from general knowledge and is unverified here.

## Backtesting and paper trading: offered, and how honest?

### Takeaway
Backtesting is widespread on third-party platforms (Kryll free, Mudrex, Altrady, 3Commas AI-generated backtests,
plan-gated). But published bot "profit" figures on exchanges often exclude fees, and marketplace PnL is
survivorship-biased. No evidence was found that any platform models slippage or reports out-of-sample results.

### Cited Findings
- A Binance-hosted grid guide states that fees are **not included** in its profit calculation —
  [Binance Square #gridbot](https://www.binance.com/square/hashtag/GridBot)
- 3Commas backtesting is gated by plan — [Gainium review (competitor)](https://gainium.io/review/3commas); its AI
  builder generates a backtest — [Capterra](https://www.capterra.com/p/10030799/3Commas/)
- Kryll offers free backtesting and strategy building — [roundup](https://www.mexc.com/news/1113070); Altrady and
  Mudrex list backtesting — [Cornix](https://cornix.io/altrady-alternatives/), [SoftwareSuggest](https://www.softwaresuggest.com/mudrex)
- Promotional Binance Square backtests: an XRP grid "+27.74% vs 0.24% hold", with 98.2% of capital sitting idle
  in cash by the end; a BTC grid 7.35% vs a 14.15% buy-and-hold return — [Binance Square #gridbot](https://www.binance.com/square/hashtag/GridBot)

### Inferences
- Platform backtests are in-sample tools for choosing parameters. With the user picking the window and the
  parameters, they invite overfitting (TH-03). Fees and slippage are the first things to check, and they are
  documented as excluded in at least one Binance guide.

### Gaps
- No vendor doc was retrieved that states the fee and slippage model of its backtester (3Commas, Bitsgap,
  Cryptohopper, Pionex). This needs a direct doc fetch.

## Independent evidence on real user outcomes, security incidents and regulatory warnings

### Takeaway
No exchange publishes the share of bot users who are profitable. The independent evidence that exists is
negative or mixed. Grid lost to buy-and-hold in every regime of a 4-year backtest. Copy-trading followers'
profits collapse within about a week of a leader being promoted (ACM CHI 2024). Traders who gain followers
perform ~10% worse. An industry 90-day sample has 48.5% of copiers profitable. The 3Commas API-key leak (2022)
is the defining security incident of the category.

### Cited Findings
**Performance evidence**
- No official Binance figure on the share of profitable grid users was found. "Binance data" claims in blogs
  (e.g. "grids beat buy-and-hold by 18.7% in 2023-24") cannot be traced to a primary source —
  [search summary; theledgermind](https://theledgermind.com/grid-trading-bot-setup/) (unverified)
- Independent 4-year hourly backtest of popular Binance grids, net of fees (rate not stated, slippage not
  modelled), on BTC:

  | Regime | Grid | Buy-and-hold |
  |---|---|---|
  | Crash 2022 | −35% (max drawdown −39%) | −65% |
  | Bull 2023-24 | +7% | +127% |
  | Sideways H2 2024 | +8% | +49% |

  Crash drawdowns ran 30-58%; memecoin grids ranged from +25% to −56%; no variant beat buy-and-hold —
  [DEV Community](https://dev.to/aduenasdev/i-backtested-the-most-popular-binance-grid-bots-on-4-years-of-real-data-1gnb)
  (single author, not peer-reviewed)
- Copy trading, crypto-specific: an ACM CHI 2024 paper, "Stranger Danger? Investor Behavior and Incentives on
  Cryptocurrency Copy-Trading Platforms", studies TraderWagon (Binance-affiliated) and Bybit. Copiers profit for
  about one week after leaders are promoted, then their PnL "plummets and even goes negative" —
  [ACM DL](https://dl.acm.org/doi/full/10.1145/3613904.3642715) (full text returned 403; finding taken from the
  search summary)
- Field study with an anonymous social trading platform (UF / UMD / UW, Qiu et al., reported 2025-03): traders
  who gain followers trade more often and with more risk, and perform **~10% worse** in the following weeks.
  Losing followers also makes them more aggressive and worse — [UF News](https://news.ufl.edu/2025/03/crypto-social-traders/)
  (journal and sample size not given)
- Lab experiment: Apesteguia, Oechssler & Weidenholzer, "Copy Trading" (Management Science, 2020). Information on
  others' success significantly raises risk-taking, and copy trading reduces ex-ante welfare —
  [ResearchGate](https://www.researchgate.net/publication/326080864_Copy_Trading)
- Counterpoint: a review argues that copying can offset overconfidence by letting weaker traders imitate
  competent ones — [Drake eScholarShare](https://escholarshare.drake.edu/server/api/core/bitstreams/4698b9bc-e014-4f4e-a767-ed003a1d58ac/content)
- Industry 90-day analysis (~100k copy outcomes on Binance, Bybit and MEXC): 48.48% of copiers profitable; 97.04%
  of leaders green on their own PnL, but only 43.61% produced positive follower PnL —
  [YieldFund](https://www.yieldfund.com/news/is-copy-trading-profitable-a-90-day-multi-exchange-study) (company blog,
  thin methodology)

**Security incidents**
- 3Commas, Oct-Dec 2022: compromised API keys were used for unauthorized trades on FTX (DMG token). One trader
  lost ~$1.6M, and the victims' group leader estimated total losses at ≥$20M. 3Commas first blamed phishing, then
  in late December confirmed that the leaked data was authentic. The hacker claimed an employee sold it; the
  internal investigation found no employee involvement. Reports put the number of keys at 10k or ~100k. The FBI
  reportedly joined on 2022-12-29. Exchanges (Binance, KuCoin, Coinbase) revoked 3Commas keys, and HAPI criticised
  them as slow to notify users —
  [ForkLog/HAPI](https://forklog.com/en/api-key-leaks-and-exchange-inaction-a-hapi-analysis-of-the-3commas-incident/),
  [ForkLog FBI](https://forklog.com/en/fbi-probes-3commas-api-key-leak-reports-say/),
  [BeInCrypto](https://beincrypto.com/ftx-users-lose-millions-to-api-exploit),
  [Bitdefender](https://www.bitdefender.com/en-us/blog/hotforsecurity/the-fbi-is-reportedly-probing-3commas-after-api-keys-leaked-on-twitter)
- No final FBI or charging outcome was found.

**Regulatory**
- CFTC customer advisory "AI Won't Turn Trading Bots into Money Machines" (2024-01-25). Claims of high or
  guaranteed returns from bots, signal services and arbitrage algorithms are red flags. It cites Mirror Trading
  International (23,000 defrauded, little real trading) —
  [CFTC 8854-24](https://www.cftc.gov/PressRoom/PressReleases/8854-24); a law-firm note points out that MTI may not
  have used AI at all — [Sidley](https://datamatters.sidley.com/2024/02/07/u-s-cftc-seeks-public-input-on-use-of-artificial-intelligence-in-commodity-markets-and-simultaneously-warns-of-ai-scams/)
- FCA finfluencer crackdown from October 2024 (20 interviewed under caution; 50 warning alerts and a joint
  multi-regulator effort by June 2025) — [Mondaq](https://www.mondaq.com/uk/financial-services/1540624/finreg-update-fca-cracks-down-on-finfluencers),
  [The Paypers](https://thepaypers.com/digital-identity-security-online-fraud/fca-joins-global-regulators-to-fight-against-finfluencers--1274002)
- ESMA finfluencer factsheet (January 2026, amplified by CONSOB): EU investment-recommendation and advertising
  rules apply to crypto content, and disclaimers do not remove liability — [Cointelegraph](https://cointelegraph.com/news/italy-consob-esma-crypto-finfluencers-advertising-rules)
- NZ FMA (2026-04-22) flags finfluencers urging followers to copy trades — [FMA](https://www.fma.govt.nz/news/all-releases/media-releases/tackling-unlawful-influencers/)

### Inferences
- Every user-outcome signal found points the same way: leaderboard and copy selection picks past luck and risk,
  followers arrive after the run, and leader behaviour degrades under follower incentives. This is the same
  selection-on-the-answer failure as TH-02 and TH-06 in this repo.
- An API-key custody model (non-custodial platforms) shifts risk from fund custody to key custody.
  Trade-only keys without withdrawal rights limited but did not prevent the 3Commas losses, because the
  attackers traded victims' funds into illiquid pairs.

### Gaps
- No exchange (Binance, Bybit, OKX, Pionex) publishes aggregate user-profitability statistics for bots. None was
  found.
- Full text of the ACM CHI 2024 paper was blocked (403), so its sample size and exact numbers are not captured.
  The Qiu et al. journal and sample size are not stated in the UF article.
- No SEC action specific to bot platforms was found; no ESMA or FCA statement specific to copy trading in
  2024-25 was found.
- Other security incidents beyond 3Commas (e.g. other bot platforms) were not searched.

## When do grid and DCA strategies make or lose money? (regime dependence)

### Takeaway
Grid trading profits from oscillation inside a range and loses relative to holding in trends. In a rally it
sells inventory too early. In a crash price leaves the range and the grid holds the inventory all the way down.
Theory shows near-zero expectation for a static grid, and fees eat small grid steps. DCA/martingale shares the
crash risk and amplifies it with scaling.

### Cited Findings
- Chen, Chen & Jang, "Dynamic Grid Trading Strategy: From Zero Expectation to Market Outperformance"
  (arXiv:2506.11921, June 2025): a traditional grid has roughly **zero expected return** under simple
  assumptions. A dynamic re-centering variant on minute BTC and ETH data (2021-01..2024-07) reports positive IRR
  up to 60-70%, beating static grid and buy-and-hold. The authors attribute much of that to the strong price rise
  in the period. ETH did better than BTC because of higher volatility. Small grid steps lose to fees —
  [arXiv 2506.11921](https://arxiv.org/abs/2506.11921) (in-sample backtest)
- Grid underperforms spot in a one-way rise and has few opportunities in a one-way fall; it is best when the
  price moves up and down repeatedly — [arXiv 2506.11921 / related review](https://arxiv.org/pdf/2506.11921)
- The 4-year backtest by regime (table above): the grid beat hold only in the 2022 crash, by losing less (−35% vs
  −65%), and lagged badly in bull and sideways windows — [DEV Community](https://dev.to/aduenasdev/i-backtested-the-most-popular-binance-grid-bots-on-4-years-of-real-data-1gnb)
- Bitcoin grid feasibility backtest on daily data from July 2019 — [EUDL (2022 conference paper)](https://eudl.eu/pdf/10.4108/eai.18-11-2022.2327164)
- Grid plus hedging beat holding BTC in a Stevens project, while ML did not help —
  [Stevens FSC](https://fsc.stevens.edu/cryptocurrency-market-making-improving-grid-trading-strategies-in-bitcoin/)
- Martingale DCA mechanics, scaling after losses — [OKX help mirror](https://okx-api.aperture.finance/help/what-are-okxs-crypto-trading-bots-and-how-do-i-utilize-it)

### Inferences
- A grid is a short-volatility / short-trend position: it collects small range gains and gives back to trends.
  For this project's goal (catch day leaders and rockets, hold the move), a grid is structurally the opposite
  bet. It sells into exactly the trend days the signal bot targets.
- The one "bot" family with positive independent evidence that fits the trader app is funding-rate carry.
  This repo already accepted FC_PASSIVE (about 0.9%/yr on 2026), which matches what Binance packages as its
  arbitrage bot.

### Gaps
- No quantitative peer-reviewed study of DCA/martingale bot outcomes on crypto was found.
- Out-of-sample (time-split) validation of the dynamic grid paper's result was not found.
