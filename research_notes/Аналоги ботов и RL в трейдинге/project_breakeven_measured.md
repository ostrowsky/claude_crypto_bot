# Break-even hit rate measured on the project's own Binance spot bars (2026-10-29)

Source: trader/runtime/history/*_15m.npz (Binance spot klines), last 365 days to 2026-10-29, pairs with >= ~312 days of bars
and still trading in late Oct 2026; script D:/ClaudeData/temp/claude/D--Projects-claude-crypto-bot/8de47132-2b41-4816-8336-b818e67c3e75/scratchpad/breakeven.py.
Replaces the assumed move sizes (25 bp at 15m, 50 bp at 1h) in short_horizon_predictability.md.

Method: m = mean |log return| over the holding period (close to close, non-overlapping, gaps dropped), taken per pair, then the
median across pairs. Expected net per trade with hit rate p: (2p - 1) m - c, so p* = 0.5 + c / (2m). This assumes symmetric
win/loss size (a direction call, no stop/target shaping), which is the setting of the operator's question.

Round-trip costs c (bp): spot taker incl. the project's 8 bp/side slippage model 36; spot taker fees only 20 (0.10% VIP0, confirmed);
USD-M futures taker 10 (0.05%); futures maker 4 (0.02%; maker fills carry adverse selection, not modelled).

| universe | horizon | m (bp) | p* spot 36 | p* spot 20 | p* fut taker 10 | p* fut maker 4 |
|---|---|---|---|---|---|---|
| BTC | 15m | 15.2 | >1 | >1 | 0.829 | 0.632 |
| BTC | 1h | 30.0 | >1 | 0.833 | 0.666 | 0.567 |
| BTC | 4h | 60.8 | 0.796 | 0.664 | 0.582 | 0.533 |
| ETH | 15m | 20.1 | >1 | 0.998 | 0.749 | 0.600 |
| ETH | 1h | 39.8 | 0.953 | 0.751 | 0.626 | 0.550 |
| bot watchlist (93) | 15m | 32.8 | >1 | 0.805 | 0.653 | 0.561 |
| bot watchlist (93) | 1h | 63.6 | 0.783 | 0.657 | 0.579 | 0.531 |
| bot watchlist (93) | 4h | 126.9 | 0.642 | 0.579 | 0.539 | 0.516 |
| all spot (430) | 15m | 36.2 | 0.997 | 0.776 | 0.638 | 0.555 |
| all spot (430) | 1h | 71.1 | 0.753 | 0.641 | 0.570 | 0.528 |
| all spot (430) | 4h | 140.0 | 0.629 | 0.571 | 0.536 | 0.514 |

Reading against the literature: achievable OOS directional accuracy is 0.515-0.558 (Jaquart et al. 2021, BTC 1-60 min, confirmed)
and AUC ~0.53 on 183 Binance pairs at 15m (Kitron & Wengrowicz 2026, confirmed). Every spot cell needs >= 0.64 at 1h and >= 0.78
at 15m, i.e. well above what any study reports. Only futures maker at 1h-4h (p* 0.51-0.53) sits inside the achievable range, and
that case ignores adverse selection of passive fills — exactly the regime where the published edge (~1.3 bp gross) does not clear
even 5 bp maker cost. Median |r| is ~30% below the mean (fat tails), so a symmetric-payoff assumption is optimistic for p*.
Caveat: m varies by regime; this is one 365-day window.

## Addendum 2026-10-29: 1d horizon (same script, same window)

| universe | horizon | m (bp) | p* spot 36 | p* spot 20 | p* fut taker 10 | p* fut maker 4 |
|---|---|---|---|---|---|---|
| BTC | 1d | 154.1 | 0.617 | 0.565 | 0.532 | 0.513 |
| ETH | 1d | 213.4 | 0.584 | 0.547 | 0.523 | 0.509 |
| bot watchlist (93) | 1d | 301.9 | 0.560 | 0.533 | 0.517 | 0.507 |
| all spot (430) | 1d | 335.1 | 0.554 | 0.530 | 0.515 | 0.506 |

Tested right after on train (direction-forecast-dir-1029-spec.md): a CatBoost direction model reaches AUC 0.529 at 1d, but
its top-decile longs rise only 48% of the time (needed 0.546) — REJECTED.
