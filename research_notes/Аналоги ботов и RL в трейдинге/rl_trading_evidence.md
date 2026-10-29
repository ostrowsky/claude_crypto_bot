# Reinforcement learning for crypto and financial trading: formulations, frameworks, and what the out-of-sample evidence after costs shows (2006-2026)

Research date: 2026-10-29. Budget-limited pass (about 15 tool calls). Every claim below carries the source it was read from in this session. Items recalled from background knowledge but not re-read are listed under Gaps and marked "unverified this session". The evidence grade used throughout: **A** = peer-reviewed with real data, temporal OOS and costs; **B** = real data and costs but a short or single OOS window, or a preprint; **C** = simulator-only, no costs, or self-reported without baselines.

## Formulations: algorithms, state, action, reward, transaction-cost modelling

### Takeaway
The standard setup is a gym-style MDP:
- **State:** a price, indicator or LOB-feature window, plus account balance and holdings.
- **Action:** discrete {buy, hold, sell} or {long, flat, short} for DQN-family agents; a continuous weight vector for PPO, SAC, DDPG or TD3 in portfolio tasks.
- **Reward:** the change in portfolio value, occasionally Sharpe-type.

Costs are usually a flat proportional fee per action. Execution is assumed certain, with no slippage, queue position or impact.

### Cited Findings
- **FinRL contests, the canonical setup (2024-2025 BTC task):**
  - State = balance, price, holdings and features. With 1 asset and 8 indicators the state has 11 dimensions; the 8 indicators are compressed by an RNN from 101 alphas computed on second-level LOB data.
  - Discrete actions, justified because "high-frequency crypto moves are modeled as discrete events". Agents are DQN, Double DQN and Dueling DQN.
  - Reward = change in total asset value.
  - Cost = "0.1% for each action", covering commission plus slippage.
  - Source: [FinRL Contests benchmark paper, arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Freqtrade FreqAI RL action spaces** are Base3 (hold/long/short), Base4 (enter long, enter short, neutral, exit) and Base5 (separate exit-long and exit-short). They are built on stable-baselines3 with a gym `Env`, and the example config uses PPO with `MlpPolicy` — [FreqAI RL docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/).
- **FreqAI's default reward is a hand-shaped demo, not PnL:**
  - −2 for an invalid action;
  - an entry bonus of 25, scaled up when RSI < 40;
  - −1 for staying flat, plus a holding penalty that grows with duration;
  - on exit, PnL × a duration factor, with a ×2 multiplier on winning exits.
  - The docs page does not mention fees at all — [FreqAI RL docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/).
- **The 167-paper survey on cost and execution assumptions:**
  - RL models "often simplify transaction costs as fixed and assume certainty of execution". Only a few studies incorporate "execution and slippage costs".
  - Minimal market impact is called realistic "only for small investors".
  - "Adding more features does not necessarily improve performance."
  - Q-learning and SARSA applied to continuous actions "can lead to suboptimal solutions or even failure to converge".
  - Source: [Pippas, Turkay & Ludvig 2024, arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Reward choice matters:** Lucarelli et al. found Sharpe-ratio rewards outperformed profit-based rewards in DQN settings, as reported in the same survey — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Algorithm frequency in that survey (Table 1):** DQN 37 entries, Q-learning 36, Recurrent RL (RRL, the Moody-Saffell direct-RL family) 31 — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Cost modelling across trading-system papers:** a survey that tabulates transaction-cost treatment per study notes "only five studies did not use any TC, while three authors did not specify adopted values" — [RL applied to trading systems survey, arXiv 2212.06064](https://arxiv.org/pdf/2212.06064).
- **Stock and forex systematic review:** "all reviewed articles had some unrealistic assumptions such as no transaction costs, no liquidity issues and no bid or ask spread issues" — reported via search of [MDPI Data 6(11):119](https://mdpi.com/2306-5729/6/11/119) and related reviews (exact attribution between the two reviews is not confirmed; see Gaps).

### Inferences
- A flat 0.1% per action with certain fills is the modal cost model. For a high-frequency BTC agent that trades on most bars, that model excludes the two costs that dominate at that horizon: spread crossing and adverse selection.
- Hand-shaped rewards, like FreqAI's entry bonus and its ×2 multiplier on winners, are a direct reward-hacking surface. The agent is paid for activity and for the sign of the outcome, not for risk-adjusted PnL.

### Gaps
- **Moody & Saffell** ("Learning to trade via direct reinforcement", IEEE Trans. Neural Networks 2001; the differential Sharpe ratio) was not re-read this session. The details below are unverified: the reward is an online, incremental Sharpe derivative, and their RRL is policy-gradient-like with no value function.
- I did not find a per-paper count of which DRL trading papers use drawdown-penalised rewards.

## Frameworks and their honest track record (FinRL, FinRL-Crypto, FinRL contests, FreqAI, SB3, ElegantRL, TensorTrade, Qlib)

### Takeaway
FinRL's own published contest results are the best public benchmark, and they are sobering:
- On the 2024 second-level BTC task, the ensemble of DQN variants did **not** beat BTC buy-and-hold on return (0.66% vs 0.74%).
- The 2024 and 2025 winners earned Sharpe ≈ 0 (0.0037 and −0.0008) over tiny test windows.
- FinRL-Crypto's "overfitting-rejected" PPO agent still lost 35% in a crash window. It only lost less than the index.

Grade: B, because test windows are short and single-regime.

### Cited Findings
- **FinRL contests 2023-2025:**
  - Tasks: stock trading, order execution, crypto trading, and LLM-engineered signals.
  - Teams: 46 / 21 / 85 across 2023-2025; about 200 participants from more than 100 institutions in 22 countries.
  - Source: [Benchmark paper summary (search)](https://www.bohrium.com/paper-details/finrl-contests-benchmarking-data-driven-financial-reinforcement-learning-agents/1152994362511065089-2000000); [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Crypto task scope and data:**
  - There was no crypto task in 2023.
  - 2024 task: "Crypto Trading with Ensemble Learning" on single-asset second-level BTC.
  - 2025 tasks: FinRL-AlphaSeek (factor mining plus ensemble) and FinRL-DeepSeek (LLM signals; the paper reports no results for it).
  - Data covers only **04/07/2021-04/19/2021**, about 12 days of BTC LOB in total.
  - Source: [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Out-of-sample window:** about 45 minutes (04/19 09:09:22-09:54:22). The text calls it "3030 minutes", which is internally inconsistent. The data was later re-partitioned "to allow for a longer testing period" — [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Table 7, 2024 ensemble experiment:**

  | Model | Cumulative return | Sharpe | Max drawdown |
  |---|---|---|---|
  | Ensembles 1-3 | 0.66% | 0.28 | −0.73% |
  | DQN | 0.34% | 0.15 | −0.93% |
  | Double / Dueling DQN | 0.48% | 0.21 | −0.98% |
  | BTC price (buy-and-hold) | 0.74% | 0.20 | −1.30% |

  The ensembles lose to buy-and-hold on return and win only on Sharpe and drawdown — [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **The authors' own admissions:**
  - Ensembles and individual agents were "near identical", because the "limited action space at each timestep caus[es] agents to output identical actions".
  - "Agent policies may be converging", "indicat[ing] a greater risk of significant losses".
  - Source: [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Table 8, contest winners** (BTC baseline −7.35% / Sharpe −0.0022 / max drawdown −17.02% over a different window):
  - 2024 winner Fermion: +0.23% return, Sharpe 0.0037, max drawdown −0.28%.
  - 2025 winner Mt.Everest: −0.05% return, Sharpe −0.0008, max drawdown −1.23%.
  - No significance testing is reported.
  - Source: [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **The FinRL team's ensemble paper** reports "reducing maximum drawdown by up to 4.17% and improving the Sharpe ratio by up to 0.21" — relative to individual agents, not to buy-and-hold — [Holzer et al., arXiv 2501.10709](https://pdf.arxiv.org/pdf/2501.10709).
- **FinRL-Crypto** (Gort, Liu, Sun, Gao, Chen, Wang; arXiv 2209.05559, v6 Jan 2023, a preprint):
  - It frames backtest-overfitting detection as a hypothesis test and rejects agents with a high estimated probability of overfitting.
  - Test window: 10 coins, 05/01/2022-06/27/2022, which contains two crashes.
  - Less-overfitted agents beat more-overfitted agents, equal-weight, and the S&P BDM index. The authors claim "confidence in possible deployment to a real market".
  - Source: [arXiv 2209.05559](https://arxiv.org/abs/2209.05559).
- **FinRL-Crypto absolute numbers:** agents were rejected at a 10% significance level. The best PPO agent returned **−34.96%**, against −50.78% for the S&P crypto BDM index and −47.78% for equal-weight — [arXiv 2504.02281 citing FinRL-Crypto](https://arxiv.org/html/2504.02281v4).
- **FreqAI RL** (stable-baselines3) ships with explicit disclaimers:
  - The default reward "is a benchmark, it is not for live production."
  - "Improperly trained RL agents may find 'cheats' and 'tricks' to maximize reward without actually winning any trades."
  - "The agent training environment is not identical to the real world": custom_exit, custom_stoploss and leverage controls are absent from training.
  - Source: [FreqAI RL docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/).

### Inferences
- **On the contests:** a 45-minute OOS window on one asset from a 12-day 2021 sample cannot distinguish skill from noise. Calling it "OOS performance" breaks TH-04 and TH-10 (comparable windows; no conclusion beyond the data).
- **On FinRL-Crypto:** "beat the index in a crash" is largely a statement about holding less exposure. A drawdown-limited or partly-in-cash rule would plausibly match it. No such simple risk-off baseline is reported in the sources read.
- **No published live track record:** no framework (FinRL, FreqAI, TensorTrade, ElegantRL, Qlib RL) has a published, independently audited live track record that I could find.

### Gaps
- **Not researched this session (budget):** TensorTrade, ElegantRL standalone results, and the Qlib RL module, whose documented use is order execution.
- **Not extracted:** Gort et al.'s exact fee assumption, bar frequency, number of hyperparameter trials, and whether CSCV/PBO is used by name. The PDF text layer could not be read here; the abstract page does not name PBO/CSCV. I believe the paper uses Bailey et al.'s CSCV-based PBO, but that is unverified this session.
- **Not read:** the FinRL-Meta paper.

## Offline RL, Decision Transformer, and contextual bandits

### Takeaway
Offline RL and Decision Transformer work in trading is recent (2024+), mostly preprints. It compares against other offline-RL baselines (CQL, IQL, behaviour cloning), not against buy-and-hold or simple rules after costs. Grade: C/B.

### Cited Findings
- **Yun (arXiv 2411.17900, Nov 2024):**
  - A GPT-2-initialised Decision Transformer fine-tuned with LoRA for offline RL in quantitative trading.
  - Motivation: offline RL "avoids live exploration".
  - Reports "competitive performance" vs CQL, IQL and BC, with higher rewards "in some trading scenarios".
  - Code: syyunn/finrl-dt, built on the FinRL environment.
  - Source: [arXiv 2411.17900](https://arxiv.org/html/2411.17900v1).
- **Sample inefficiency and limited history** restrict what RL agents can learn in finance — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).

### Inferences
- Offline RL inherits whatever behaviour policy generated the dataset. In trading, the logged policy is usually the backtest itself, so offline RL does not escape the backtest-overfitting problem. It removes exploration risk, not selection bias.
- **Contextual bandits for strategy selection:** the in-repo record (CLAUDE.md §4) is directly relevant evidence. This project's LinUCB entry bandit scored lift 0.65× on a temporal holdout when trained on a leaky label, and was switched OFF on 2026-09-07 after its rejects outperformed its entries. That is internal evidence, not literature.

### Gaps
- I found no peer-reviewed crypto-specific Decision Transformer paper with fees and temporal OOS.
- I found no rigorous literature evaluation of bandit-based strategy selection in crypto.

## Where RL demonstrably adds value: execution, market making, LOB trading, industry

### Takeaway
The strongest evidence for RL is in **execution**. There, the objective (implementation shortfall vs TWAP) is well defined, the benchmark is fixed, and the effect does not require predicting direction. Real-data improvements:
- 27-35% vs submit-and-leave on NASDAQ (Nevmyvaka et al. 2006);
- Double DQN beat TWAP on 7 of 9 stocks (Ning et al. 2021).

Market making evidence is mostly simulator-based (Spooner 2018 is a high-fidelity LOB simulator). Directional HFT RL (Briola 2021) reports "stable positive returns" with no visible cost model. Industry use (JPMorgan LOXM) is execution, documented only by 2017 press.

### Cited Findings
- **Nevmyvaka, Feng & Kearns, ICML 2006:**
  - The first large-scale empirical RL application to optimised execution.
  - Data: 1.5 years of millisecond NASDAQ limit-order data.
  - Source: [ICML 2006 / Kearns page](https://www.cis.upenn.edu/~mkearns/papers/rlexec); [mlanthology](https://mlanthology.org/icml/2006/nevmyvaka2006icml-reinforcement).
- **Nevmyvaka et al. results, per a secondary lecture deck:**
  - Tested on AMZN, NVDA and QCOM, with 2 order sizes, 2 horizons and a 6-month test set.
  - With private variables only: 27.16%-35.50% relative improvement over submit-and-leave.
  - The "50% or more" figure, likely with market variables, was not confirmed in the paper.
  - Source: [Waterloo CS885 slides](https://cs.uwaterloo.ca/%7Eppoupart/teaching/cs885-spring20/slides/cs885-reinforcement-learning-for-optimized-trade-execution.pdf).
- **Replication of Nevmyvaka et al.:** a later thesis attempting it achieved results "on par ... but only for a few cases" — [thesis PDF](https://www.smallake.kr/wp-content/uploads/2019/01/FULLTEXT01.pdf).
- **Ning et al. (2021), Double DQN for execution:** on real data, DDQN outperforms TWAP in **seven of nine** stocks — as cited in [arXiv 2411.06389](https://arxiv.org/pdf/2411.06389).
- **Lin & Beling (2021), DQN variant:** "outperforms TWAP, VWAP, AC and 2 DQN algorithms" in backtests on 14 US equities — as cited in search results around [arXiv 2411.06389](https://arxiv.org/pdf/2411.06389).
- **2024 agent-based simulator study** (one-sided t-tests on implementation shortfall):
  - With 10 noise agents, RL beats TWAP (t = 4.52) and Passive (t = 2.08).
  - With 24 momentum agents, RL beats TWAP (t = 2.26) and Random (t = 1.81).
  - This is simulation only.
  - Source: [arXiv 2411.06389](https://arxiv.org/pdf/2411.06389).
- **Why TWAP is a hard benchmark:** under Almgren-Chriss assumptions (linear impact, Brownian price, constant volatility) TWAP is optimal. Any RL gain must come from violations of those assumptions — [arXiv 2511.15262](https://arxiv.org/pdf/2511.15262).
- **Spooner, Fearnley, Savani & Koukorinis (AAMAS 2018):**
  - A temporal-difference market-making agent with tile-coding value functions and an inventory-penalising reward, trained in a high-fidelity LOB simulator.
  - It outperforms simple benchmarks and a recent online-learning approach.
  - Source: [arXiv 1804.04216](https://ar5iv.labs.arxiv.org/html/1804.04216).
- **Gašperov & Kostanjčar (IEEE Access 2021), "Market Making With Signals Through Deep RL":**
  - It addresses the fact that prior ML market making used "crudely discretized quotes or fail[ed] to incorporate additional predictive signals".
  - Survey by the same group: *Mathematics* 2021.
  - Source: [academia.edu copy](https://www.academia.edu/73246700/Market_Making_With_Signals_Through_Deep_Reinforcement_Learning); [CROSBI survey record](https://www.bib.irb.hr/1154869).
- **Robust market making:** Spooner & Savani were first to apply adversarial RL for a market maker robust to epistemic (model) risk. Gašperov & Kostanjčar perturb an adversarial agent for robustness — [arXiv 2508.16588](https://arxiv.org/pdf/2508.16588).
- **Crypto market making with DRL:** "Extending Deep RL Frameworks in Cryptocurrency Market Making" (arXiv 2004.06985) exists. It was not read in detail — [arXiv 2004.06985](https://arxiv.org/pdf/2004.06985).
- **Briola, Turiel, Marcaccioli, Cauderan & Aste (arXiv 2101.07107, v3 2023):**
  - PPO on Intel LOB data: 3 months of training (the last used for validation) and 1 test month.
  - Training samples were **selected for the largest price changes** to raise signal-to-noise.
  - Reports "stable positive returns".
  - The abstract mentions no transaction costs or spread, and gives no numeric benchmark.
  - Source: [arXiv 2101.07107](https://arxiv.org/abs/2101.07107).
- **JPMorgan LOXM (deep RL execution):**
  - Tested in Europe from Q1 2017, with a planned rollout to Asia and the US.
  - The bank claimed pricing "significantly better than its benchmark" and that it "far outperformed" manual and automated methods.
  - Trained on "billions of historic and simulated trades". Actions: how much, at what price, for how long. Constrained within the firm's electronic risk framework.
  - Sources are 2017 press and case studies only; current status is unconfirmed.
  - Source: [Markets Media](https://marketsmedia.com/j-p-morgan-ai-trading-global/); [bestpractice.ai case](https://bestpractice.ai/ai-use-cases/case-studies/financial-services/jpmorgan-s-new-ai-program-for-automatically-executing-equity-trades-in-real-time-out-performed-current-manual-and-automated-methods-in-trial).
- **Hambly, Xu & Yang survey application list:** optimal execution, portfolio optimisation, option pricing and hedging, market making, smart order routing, robo-advising — [arXiv 2112.04553](https://arxiv.org/abs/2112.04553).

### Inferences
- **The pattern across sources:** RL adds value where (a) a strong analytic baseline exists (TWAP, Almgren-Chriss, Avellaneda-Stoikov), (b) the objective is cost minimisation or spread capture rather than direction, and (c) the agent can exploit microstructure state (queue, imbalance) that the analytic model ignores.
- **Directional alpha** has no fixed benchmark to beat. It also fails on non-stationarity, the second weak spot in the literature.
- **For a retail crypto signal bot**, the transferable evidence is in exit and execution mechanics (order placement, TP/SL execution), not entry selection.
- **Briola et al.:** selecting training samples on large price moves, then evaluating on unfiltered test data without costs, is a recipe for optimistic HFT results. Grade C until the full paper's cost model is checked.

### Gaps
- **Avellaneda & Stoikov 2008** (Quantitative Finance, "High-frequency trading in a limit order book") was not re-read. Unverified this session: it is the standard MM baseline, with inventory-skewed reservation price and exponential fill intensities.
- **Ning et al. citation:** the primary paper (Ning, Lin & Jaimungal, "Double deep Q-learning for optimal execution", Applied Mathematical Finance 2021; arXiv 1812.06600) was not fetched.
- **Missing industry evidence:** I found no post-2017 public evidence of LOXM's status, and no other bank's audited RL-execution results.
- **Not verified:** whether the "Extending DRL frameworks in crypto market making" paper includes maker/taker fees and queue modelling.

## Out-of-sample evidence after realistic fees, replications, critical surveys

### Takeaway
Critical surveys converge: most DRL trading papers lack realistic costs, strong baselines, or proper temporal splits, and "no decent profitability level was obtained" even where improvements over baselines are statistically significant. Recent crypto evidence with fees leans toward RL **underperforming buy-and-hold**. Positive single-paper claims exist but are self-reported. Grade: mixed B/C. There is no A-grade positive result for directional crypto RL.

### Cited Findings
- **Millea, "Deep RL for Trading — A Critical Survey" (Data, 2021):** "a lack of consistency in the community can significantly impede research and the development of DRL agents for trading" — [MDPI Data 6(11):119](https://mdpi.com/2306-5729/6/11/119).
- **2021 review of DRL stock trading:** "despite all showing statistically significant improvements in performance compared to established baseline strategies, no decent profitability level was obtained". Few compared RL with other sophisticated ML/DL models — via search across [MDPI Data review](https://mdpi.com/2306-5729/6/11/119) and [preprints.org 202111.0044](https://www.preprints.org/manuscript/202111.0044/v1/download). The exact paper attribution of the quote is uncertain.
- **Pippas et al. 2024:**
  - Survey of 167 publications. Only 27.5% predate 2016.
  - RL "must be carefully designed and validated to avoid overfitting".
  - Financial data are "laden with noise, exhibit non-stationarity and stochasticity".
  - Disclosure: the author is employed by HSBC Asset Management, which partly funded the research.
  - Source: [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **2025 SSRN preprint, risk-aware DRL with transaction costs, OOS 2024 BTC:**
  - RL Sharpe 1.23 vs buy-and-hold 1.46.
  - Final NAV 1.916 vs 2.213.
  - The RL policy loses on both.
  - Source: [SSRN 5662930](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5662930).
- **IEEE, Dueling DQN crypto agent (BTC, ETH, LTC):** buy-and-hold beat the agent — [IEEE 10068485](https://ieeexplore.ieee.org/document/10068485/).
- **Counter-claim, Cogent Economics & Finance 2025:** a DQN selecting among technical strategies on BTC, 2022-mid-2025, beats buy-and-hold, and the authors say OOS "confirmed generalization". Self-reported, single paper — [T&F 10.1080/23322039.2025.2594873](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873).
- **FinRL's own contest benchmarks:** winners at Sharpe ≈ 0, and the ensemble below buy-and-hold on return (see the frameworks section) — [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).

### Inferences
- **Publication bias:** a positive crypto RL paper with fees exists for nearly every negative one. The negative ones are more often the ones that compare directly with buy-and-hold over a full year. Without a registry of trials, the positive set is subject to exactly the selection bias that the Deflated Sharpe Ratio (DSR) corrects.
- **Agreement with this repo's trader research:** directional sources (XSMOM, S1 reversal, TM timing, MD multi-day) were all rejected on train. The only accepted trader hypothesis is non-directional carry (FC_PASSIVE). That matches the literature pattern: non-directional and execution edges are real; directional ones rarely survive.

### Gaps
- I did not locate an independent replication study that re-ran multiple published DRL trading papers under a common cost model and temporal split. I believe some exist (e.g., FinRL benchmark re-evaluations), but none was confirmed this session.
- I did not obtain per-paper fractions in the 2024 survey of papers that use costs, temporal splits, or multiple seeds. Those sections of 2408.10932 were not reached.

## Backtest overfitting: PBO, CSCV, Deflated Sharpe Ratio, and RL hyperparameter search

### Takeaway
The probability of backtest overfitting (PBO) via combinatorially symmetric cross-validation (CSCV) (Bailey, Borwein, López de Prado & Zhu, J. Computational Finance 2017), and the Deflated Sharpe Ratio (Bailey & López de Prado, JPM 2014), are the accepted corrections. Each RL hyperparameter config, seed and reward variant is a trial. Gort et al. is the main example of applying an overfitting test to DRL crypto agents, and even their accepted agent lost 35% in absolute terms.

### Cited Findings
- **Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting":**
  - Published in J. Computational Finance (2017; DOI 10.21314/jcf.2016.322).
  - Hold-out validation is "unreliable and inaccurate" for investment backtests.
  - It proposes a PBO framework implemented as CSCV, which is "generic, model-free and nonparametric", with "reasonable estimates of PBO, with relatively small errors".
  - Source: [SSRN 2326253](https://papers.ssrn.com/abstract=2326253); [Risk.net JCF](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting); [eScholarship PDF](https://escholarship.org/content/qt4w1110bb/qt4w1110bb.pdf).
- **Bailey & López de Prado, "The Deflated Sharpe Ratio":**
  - Published in JPM 40(5):94-107, 2014, DOI 10.3905/jpm.2014.40.5.094.
  - It corrects for "selection bias under multiple testing and non-Normally distributed returns".
  - Inputs: the observed Sharpe ratio, number of trials, variance of trial Sharpe ratios, skewness, kurtosis, and track length.
  - Source: [SSRN 2460551](https://papers.ssrn.com/abstract=2460551); [quantstrat implementation](https://www.rdocumentation.org/packages/quantstrat/versions/0.16.7/topics/deflatedSharpe).
- **Gort et al. (FinRL-Crypto):**
  - Formulate overfitting detection as a hypothesis test and reject agents at a 10% significance level.
  - The accepted best PPO agent returned −34.96%, vs index −50.78% and equal-weight −47.78%.
  - Source: [arXiv 2209.05559](https://arxiv.org/abs/2209.05559); [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Briola et al. tuned hyperparameters with Sequential Model-Based Optimization.** This is a trial-generating search, and the abstract reports no multiple-testing correction — [arXiv 2101.07107](https://arxiv.org/abs/2101.07107).

### Inferences
- DRL multiplies the trial count well beyond classical strategy search: hyperparameters × seeds × reward shapes × state designs × train-window choices. Unless the DSR's N counts all of these, any reported RL Sharpe should be treated as heavily inflated.
- The PBO test filters relatively overfitted agents within a family. It does not establish absolute edge, as Gort et al.'s negative absolute return shows.

### Gaps
- Not obtained: Gort et al.'s number of trials, the CSCV split count (S), and how the PBO threshold was set.
- I found no study that quantifies the DSR haircut specifically for published DRL trading results.

## Failure modes: non-stationarity, reward hacking, leakage, seed variance, sample inefficiency, sim-to-real

### Takeaway
The documented failure modes are consistent across sources:
- **Non-stationarity:** noise and regime shift.
- **Reward hacking:** explicitly warned of by FreqAI.
- **Unrealistic environments:** fixed costs, certain fills, no impact.
- **Sample inefficiency** on limited history.
- **Policy collapse** in small discrete action spaces: FinRL ensembles produced "identical actions".
- **Training-data selection** that inflates signal: Briola selected training samples on large moves.

Seed variance is widely known from general deep-RL literature but rarely reported in trading papers; I did not find a trading-specific quantification.

### Cited Findings
- **Non-stationarity:** financial data "exhibit non-stationarity and stochasticity", which can lead to unpredictable results — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Reward hacking:** "Improperly trained RL agents may find 'cheats' and 'tricks' to maximize reward without actually winning any trades" — [FreqAI docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/).
- **Sim-to-real gap:**
  - FreqAI's training environment lacks custom_exit, custom_stoploss and leverage — [FreqAI docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/).
  - RL papers assume fixed costs and certain execution; minimal market impact is realistic only for small investors — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Policy collapse / lack of diversity:** ensemble and individual DQN agents were near identical, because a "limited action space at each timestep, causing agents to output identical actions"; "a greater risk of significant losses" — [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Evaluation-window fragility:** the FinRL contest test window was about 45 minutes and inconsistently described, then re-partitioned — [arXiv 2504.02281](https://arxiv.org/html/2504.02281v4).
- **Sample inefficiency with limited historical data** — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Selection of training samples on large price changes** (signal inflation) — [arXiv 2101.07107](https://arxiv.org/abs/2101.07107).
- **Feature overload:** "adding more features does not necessarily improve performance" — [arXiv 2408.10932](https://arxiv.org/html/2408.10932v3).
- **Fee mismatch in open-source crypto RL:** a public project trained without fees and evaluated with a 0.1% fee, claiming about 30% above buy-and-hold. It is not peer-reviewed and is an example of a train/eval cost mismatch — [GitHub NickKaparinos](https://github.com/NickKaparinos/Automated-Cryptocurrency-trading-using-Deep-RL).

### Inferences
- **Look-ahead leakage** in gym trading environments typically enters through:
  - features computed on the full series (normalisation, indicators with centred windows);
  - acting on the close of the bar that produced the observation;
  - random rather than temporal train/test splits.

  None of the sources read quantified this. This repo's own history of leakage (label_top20, top_gainer AUC 0.99 → 0.83 honest) is a direct analogue.
- Reward-shaped activity bonuses, like FreqAI's entry bonus of 25, push agents toward over-trading. Under real fees, over-trading is the most common way a backtest-positive agent turns negative.

### Gaps
- **Henderson et al. 2018** ("Deep RL that Matters", AAAI; arXiv 1709.06560) was not re-read. Unverified this session: it showed large seed-to-seed variance in standard DRL benchmarks, with different seed sets yielding statistically different learning curves.
- I found no trading paper that quantifies seed variance of OOS PnL across, for example, 10+ seeds with fees.
- I found no quantitative study of the sim-to-real gap (backtest vs live PnL) for any open-source RL trading framework.
