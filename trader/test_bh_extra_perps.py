"""Fetch of the 4 coin perps missing from the 2020-2022 store (hype-short-family-1105-spec.md §1). Written BEFORE the
code. Offline: list_all and fetch_series are replaced by fakes."""
import unittest


class TestExtra(unittest.TestCase):
    def test_list_and_fetch_named_perps_only(self):
        import archive_bh as A
        self.assertEqual(A.EXTRA_PERPS, ("BNXUSDT", "DOTECOUSDT", "LUNA2USDT", "TLMUSDT"))
        listed, fetched = [], []
        orig = (A.list_all, A.fetch_series)

        def fake_list(prefix, delimiter=True):
            listed.append(prefix)
            sym = prefix.split("/")[5]
            kind = "klines" if "/klines/" in prefix else "fundingRate"
            mid = "-15m-" if kind == "klines" else "-fundingRate-"
            return [f"{prefix}{sym}{mid}{m}.zip" for m in ("2019-12", "2021-03", "2023-01")]

        def fake_fetch(kind_prefix, sym, months, out_npz, parser, keys, ex):
            fetched.append((sym, tuple(months), out_npz.parent.name))
            return {"new": len(months)}
        try:
            A.list_all, A.fetch_series = fake_list, fake_fetch
            tot = A.fetch_extra(log=lambda s: None)
        finally:
            A.list_all, A.fetch_series = orig
        self.assertEqual(tot["perps"], 4)
        self.assertTrue(all(m == ("2021-03",) for _, m, _ in fetched))          # 2020-01..2022-12 only
        self.assertEqual({d for _, _, d in fetched}, {"perp", "funding"})
        self.assertTrue(all(p.startswith("data/futures/um/monthly/") for p in listed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
