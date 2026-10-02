# "BTC leads SOL/ETH with a bigger move" — REFUTED (2026-10-03)

- **Slug:** `btc-leads-alts-1003`
- **Status:** research, negative result — no behaviour change
- **Hypothesis (operator):** SOL and ETH start rising right after BTC with a move
  several times larger; watching BTC therefore predicts — "guaranteed" — the early
  start of their trend (and other coins'), and its end.
- **Truth harness:** TH-01 (base rate beside every hit rate), TH-03 (first/second
  half), TH-08 (negative result committed), TH-10
- **Rollback / shadow:** nothing shipped
- **Script:** `files/_backtest_btc_leads_alts.py`; maximum period of the 15m long
  store, 2025-06-27 .. 2026-10-02 (462 days), 93 watchlist coins

Prior work: `docs/reports/2026-05-07-cluster-lead-lag.md` (90 days) — BTC and alts
move contemporaneously (corr 0.82), a big BTC bar is followed by +0.07% on the
alt basket.

## Criteria (fixed before the run) and results

| claim | test | criterion | result | verdict |
|---|---|---|---|---|
| move "several times larger" | beta to BTC, 15m / 1h / 1d | ≥ 2 | SOL 1.33 / 1.31 / 1.37, ETH 1.24 / 1.25 / 1.32; watchlist median 1.29–1.38; **no coin ≥ 2** | refuted (≈1.3×) |
| "right after" BTC | alt_{t+1} ~ btc_t + alt_t | lead > 0 both halves and ≥ 0.1% per +1% BTC | same-bar corr SOL 0.80, ETH 0.86; next-bar lead 15m SOL −0.011%, ETH −0.026%; 1h +0.030% / +0.020%; watchlist median 15m +0.136% → +0.016% (1st → 2nd half) | refuted: they move in the SAME bar |
| trend START, "guaranteed" | BTC 1h ≥ +0.5% after 6 quiet h (414 events); ≥ +1.0% (90) | P(alt up 4h after) ≥ 80% both halves | SOL already +1.08% INSIDE the BTC hour (+1.97% at ≥ 1%); after: 4h +0.05%, P(up) 55% vs base 50%; ETH 53% vs 51%; watchlist 49% | refuted |
| trend END | BTC +2% in 12h, then 1h ≤ −0.5% (58 events) | P(alt down 4h) ≥ 80% | SOL 55% (halves 43% / 67%), ETH 57% vs base 49–50% | refuted |
| tradable | buy at the event's close, 0.10% fee | mean > 0 with CI | SOL 4h −0.05% [−0.23, +0.12], ETH 4h −0.07% [−0.21, +0.07]; ≥ 1%: SOL −0.14%, ETH −0.09%; win ≈ 50% | refuted |
| which coin | share of watchlist coins that are top-20 winners | higher on BTC event days | 0.035 on event days vs 0.035 otherwise; SOL and ETH were immutable top-20 winners on **1 day each** | no information |

## What is true

SOL and ETH (and the watchlist) move **together** with BTC, inside the same 15m bar,
about **1.3×** BTC's move. By the time a BTC hour has closed up, SOL has already
made its part of the move; what follows is indistinguishable from an ordinary hour.
BTC tells the market's direction, not which coin will lead or when its own trend
starts — consistent with the decoupling result (2026-05-07): rockets are coins that
move AWAY from BTC. Do not re-test BTC-led entries/exits on price data.

## Tests

`files/test_btc_leads_alts.py`.
