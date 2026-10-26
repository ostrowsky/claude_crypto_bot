"""Fees paid in BNB -- tests written BEFORE the code (operator 2026-10-22: check the BNB discount, fix the accounting
first, then buy BNB on the demo account).

Checked on the demo account 2026-10-22 (read only): standard fee 0.10% maker/taker, BNB discount enabled for the
account and the pairs (factor 0.75), BNB balance 0 -- so fees are still paid at the full rate. With BNB on the account
the exchange charges the fee in BNB, and two numbers would be wrong:
  1. the executor counts only USDT commissions: a BNB fee on the buy or the sell is invisible, so every trade's pnl
     reads better than it was (by ~0.15 pp a round trip);
  2. the daily revenue splits the stablecoin drift off the policy revenue, but not the price move of the BNB fee
     reserve, which would be counted as trading.
"""
import unittest
from datetime import datetime, timedelta, timezone

from test_executor import NOW, closed, make
from test_demo_ledger_d import D0, day_snaps

KEY_SYM = "ABCUSDT"


def bnb_fills(o, comm_bnb):
    o = dict(o)
    o["fills"] = [{"price": str(float(o["cummulativeQuoteQty"]) / float(o["executedQty"])), "qty": o["executedQty"],
                   "commission": str(comm_bnb), "commissionAsset": "BNB"}]
    return o


class TestExecutorBnbFees(unittest.TestCase):
    def setup(self, bnb_px=600.0):
        prices = {KEY_SYM: 2.0}
        if bnb_px is not None:
            prices["BNBUSDT"] = bnb_px
        E, paper, ex, w, exe = make(prices=prices)
        ex.fee = 0.0                                   # with BNB the coin and USDT amounts carry no fee
        return E, paper, ex, w, exe

    def test_buy_fee_in_bnb_is_part_of_the_cost(self):
        E, paper, ex, w, exe = self.setup()
        orig = ex.market_buy
        ex.market_buy = lambda sym, q, cid: bnb_fills(orig(sym, q, cid), 0.00002)     # 0.012 USDT at 600
        key = paper.enter(sym=KEY_SYM)
        exe.cycle([], NOW)
        p = exe.positions[key]
        quote = float([c for c in ex.calls if c[0] == "market_buy"][0][2])
        self.assertAlmostEqual(p["cost"], quote + 0.012, places=6)
        rec = [r for r in w.kind("real_entry") if r.get("action") == "enter"][-1]
        self.assertAlmostEqual(rec["fee_usdt"], 0.012, places=6)

    def test_sell_fee_in_bnb_reduces_the_proceeds(self):
        E, paper, ex, w, exe = self.setup()
        key = paper.enter(sym=KEY_SYM)
        exe.cycle([], NOW)
        cost = exe.positions[key]["cost"]
        orig = ex.market_sell
        ex.market_sell = lambda sym, q, cid: bnb_fills(orig(sym, q, cid), 0.00003)    # 0.018 USDT at 600
        paper.exit(key)
        exe.cycle([], NOW + timedelta(minutes=20))
        p = closed(exe, key)
        gross = p["sold_qty"] * 2.0
        self.assertAlmostEqual(p["net_proceeds"], gross - 0.018, places=6)
        self.assertAlmostEqual(p["pnl"], gross - 0.018 - cost, places=6)

    def test_bnb_price_unknown_estimates_the_discounted_rate(self):
        E, paper, ex, w, exe = self.setup(bnb_px=None)
        orig = ex.market_buy
        ex.market_buy = lambda sym, q, cid: bnb_fills(orig(sym, q, cid), 0.00002)
        key = paper.enter(sym=KEY_SYM)
        exe.cycle([], NOW)
        quote = float([c for c in ex.calls if c[0] == "market_buy"][0][2])
        rec = [r for r in w.kind("real_entry") if r.get("action") == "enter"][-1]
        self.assertAlmostEqual(rec["fee_usdt"], quote * 0.00075, places=6)
        self.assertTrue(rec["fee_estimated"])
        self.assertAlmostEqual(exe.positions[key]["cost"], quote * (1 + 0.00075), places=6)

    def test_quote_and_base_commissions_unchanged(self):
        import executor as E
        exe = make()[-1]
        o = {"fills": [{"commission": "0.02", "commissionAsset": "USDT"}, {"commission": "0.5", "commissionAsset": "ABC"}]}
        self.assertAlmostEqual(exe._fee_usdt(o, 20.0), 0.02)                # base-asset fee already cut the qty
        self.assertAlmostEqual(exe._fee_usdt({"fills": []}, 20.0), 0.02)     # no fills: taker estimate as before
        _ = E


class TestDailyRevenueBnbReserve(unittest.TestCase):
    def hold(self, bnb_px):
        return {"USDT": {"qty": 4980.0, "px": 1.0}, "USDC": {"qty": 5000.0, "px": 1.0},
                "BNB": {"qty": 0.02, "px": bnb_px}}

    def test_bnb_price_move_is_not_policy_revenue(self):
        import daily_revenue as DR
        snaps = day_snaps(D0, lambda t: self.hold(600.0 + 30.0 * (t - D0).total_seconds() / 86400))
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), snaps, fills=[])
        e0 = 4980.0 + 5000.0 + 0.02 * 600.0
        self.assertAlmostEqual(rec["fee_reserve_contrib_pct"], 0.02 * 30.0 / e0 * 100, places=6)
        self.assertLess(abs(rec["R_policy_pct"]), 1e-6)                    # nothing traded: the policy made 0
        self.assertAlmostEqual(rec["stable_contrib_pct"], 0.0, places=9)

    def test_no_bnb_no_reserve_term(self):
        import daily_revenue as DR
        h = {"USDT": {"qty": 5000.0, "px": 1.0}, "USDC": {"qty": 5000.0, "px": 1.0}}
        rec = DR.compute_day(D0.strftime("%Y-%m-%d"), day_snaps(D0, lambda t: h), fills=[])
        self.assertAlmostEqual(rec["fee_reserve_contrib_pct"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
