"""CatBoost as a third candidate family beside logistic and MLP.

Added 2026-08-20 on the operator's instruction: the trainer should choose the
best of three rather than the best of two.

What the tests pin is not "CatBoost is good" -- on the first real selection it
came LAST (score 0.057 against mlp 0.323 and logistic 0.126). They pin that it
is wired correctly, survives the JSON round-trip the payload requires, is cached
on the hot path, and cannot take the nightly retrain down with it if the install
breaks.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ml_signal_model as M  # noqa: E402

SRC = (HERE / "ml_signal_model.py").read_text(encoding="utf-8")


class TestCatBoostIsACandidate(unittest.TestCase):
    def test_it_is_in_the_selection(self):
        self.assertIn('models["catboost"] = CatBoostModel(', SRC)

    def test_a_broken_install_cannot_kill_the_nightly_retrain(self):
        # The trainer runs unattended at 02:30. An ImportError there would mean
        # no model at all, which is strictly worse than two families.
        self.assertIn("catboost unavailable, selecting from logistic/mlp only", SRC)

    def test_the_scorer_understands_the_new_type(self):
        self.assertIn('if model["type"] == "catboost":', SRC)

    def test_loading_is_cached(self):
        # predict_proba_from_payload runs per candidate per poll; rebuilding the
        # booster each call would put deserialisation on the hot path.
        self.assertIn("_CATBOOST_CACHE", SRC)
        self.assertIn("def _load_catboost(", SRC)


class TestCatBoostRoundTrip(unittest.TestCase):
    """A model that cannot survive the payload is a model the bot cannot run."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.n_feat = len(M.safe_feature_names())
        X = rng.normal(size=(400, cls.n_feat))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        cls.X, cls.y = X, y
        try:
            cls.model = M.CatBoostModel(cls.n_feat, iterations=60).fit(X, y)
        except ImportError:
            cls.model = None

    def setUp(self):
        if self.model is None:
            self.skipTest("catboost not installed in this environment")

    def test_the_payload_is_json_serialisable(self):
        d = self.model.to_dict()
        self.assertEqual(d["type"], "catboost")
        json.dumps(d)   # must not raise

    def test_predictions_survive_the_round_trip_exactly(self):
        names = M.safe_feature_names()
        payload = {
            "model_name": "catboost",
            "feature_names": names,
            "scaler_mean": [0.0] * self.n_feat,
            "scaler_scale": [1.0] * self.n_feat,
            "threshold": 0.5,
            "positive_ret_threshold": 0.0,
            "model": self.model.to_dict(),
        }
        payload = json.loads(json.dumps(payload))
        direct = float(self.model.predict_proba(self.X[:1])[0])
        vals = {k: float(self.X[0][i]) for i, k in enumerate(names)}
        orig = M.build_feature_dict
        M.build_feature_dict = lambda rec: vals
        try:
            through = M.predict_proba_from_payload(payload, {})
        finally:
            M.build_feature_dict = orig
        self.assertAlmostEqual(direct, through, places=9)

    def test_it_can_express_what_a_linear_model_cannot(self):
        # The reason for adding it: the logistic and MLP families here are
        # hand-written on 60 standardised features, and the failure that started
        # this work was a model scoring an entire regime near zero -- a shape a
        # linear model has no way to condition on.
        lg = M.LogisticModel(self.n_feat).fit(self.X, self.y)
        auc = lambda p: M.roc_auc_score_np(self.y.astype(float), np.asarray(p, dtype=float))
        self.assertGreater(auc(self.model.predict_proba(self.X)),
                           auc(lg.predict_proba(self.X)))

    def test_predict_before_fit_does_not_explode(self):
        m = M.CatBoostModel(self.n_feat)
        out = m.predict_proba(np.zeros((3, self.n_feat)))
        self.assertEqual(len(out), 3)
        self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
