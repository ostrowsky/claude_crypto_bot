"""Tests for fast-reversal model training + inference (RM-3 Step 2).

Validates:
- AUC computation correctness
- Logistic regression converges on linearly separable data
- Model save/load round-trip
- Inference returns valid probabilities
- Inference returns None when no model is present
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fast_reversal_model
import train_fast_reversal as tfr
import config
import monitor


class AUCTests(unittest.TestCase):
    """Verify AUC computation via Mann-Whitney U."""

    def test_perfect_separator_auc_1(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        self.assertAlmostEqual(tfr.compute_auc(y_true, y_score), 1.0)

    def test_inverse_separator_auc_0(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        self.assertAlmostEqual(tfr.compute_auc(y_true, y_score), 0.0)

    def test_random_separator_auc_0_5(self):
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_score = [0.5] * 8
        self.assertAlmostEqual(tfr.compute_auc(y_true, y_score), 0.5)

    def test_all_positive_returns_0_5(self):
        y_true = [1, 1, 1]
        y_score = [0.5, 0.7, 0.3]
        self.assertAlmostEqual(tfr.compute_auc(y_true, y_score), 0.5)


class LogisticTrainTests(unittest.TestCase):
    """Verify logistic regression converges on simple data."""

    def test_linearly_separable_converges(self):
        # Feature 0 is the signal: < 0 → negative, > 0 → positive
        X = [
            [-1.0, 0.0], [-0.5, 0.0], [-0.8, 0.0], [-1.2, 0.0],
            [0.5, 0.0],  [1.0, 0.0],  [0.8, 0.0],  [1.2, 0.0],
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        weights, bias, means, stds = tfr.train_logistic(X, y, n_iter=500, lr=0.3)

        # Predictions should separate classes
        preds = [tfr.predict_proba(x, weights, bias, means, stds) for x in X]
        for pred, label in zip(preds, y):
            if label == 1:
                self.assertGreater(pred, 0.5, f"Expected high prob for positive sample, got {pred}")
            else:
                self.assertLess(pred, 0.5, f"Expected low prob for negative sample, got {pred}")

    def test_weight_picks_up_signal_feature(self):
        # Feature 0 carries signal, feature 1 is noise
        X = []
        y = []
        for i in range(50):
            X.append([-1.0 - 0.01 * i, 0.5])
            y.append(0)
            X.append([1.0 + 0.01 * i, 0.5])
            y.append(1)

        weights, bias, means, stds = tfr.train_logistic(X, y, n_iter=300)
        # Weight on feature 0 should be substantial; weight on feature 1 (constant) should be ~0
        self.assertGreater(abs(weights[0]), abs(weights[1]) * 5)


class InferenceTests(unittest.TestCase):
    """Verify inference module behaviour."""

    def setUp(self):
        # Each test gets a fresh temp dir + model file, and the module's cache is cleared
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_model = Path(self.temp_dir.name) / "model.json"
        self.patcher = mock.patch.object(fast_reversal_model, "MODEL_FILE", self.temp_model)
        self.patcher.start()
        # Clear module cache between tests
        fast_reversal_model._MODEL_CACHE = None
        fast_reversal_model._MODEL_LOADED_MTIME = 0.0

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_no_model_returns_none(self):
        self.assertFalse(fast_reversal_model.is_available())
        self.assertIsNone(fast_reversal_model.predict_proba({"vol_x": 1.0}))

    def test_loaded_model_returns_probability(self):
        # Save a minimal model
        model_data = {
            "model_type": "logistic_regression",
            "feature_names": ["vol_x", "rsi"],
            "weights": [0.5, -0.5],
            "bias": 0.0,
            "means": [1.0, 50.0],
            "stds": [0.5, 10.0],
            "trained_on": 100,
            "auc_train": 0.7,
            "auc_val": 0.65,
            "label_positive_rate": 0.3,
        }
        self.temp_model.write_text(json.dumps(model_data))

        self.assertTrue(fast_reversal_model.is_available())

        proba = fast_reversal_model.predict_proba({"vol_x": 1.5, "rsi": 60.0})
        self.assertIsNotNone(proba)
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)

    def test_missing_features_treated_as_zero(self):
        model_data = {
            "model_type": "logistic_regression",
            "feature_names": ["vol_x", "rsi", "missing_feature"],
            "weights": [0.5, -0.5, 1.0],
            "bias": 0.0,
            "means": [1.0, 50.0, 0.0],
            "stds": [0.5, 10.0, 1.0],
            "trained_on": 100,
            "auc_train": 0.7,
            "auc_val": 0.65,
            "label_positive_rate": 0.3,
        }
        self.temp_model.write_text(json.dumps(model_data))

        # missing_feature absent → treated as 0
        proba = fast_reversal_model.predict_proba({"vol_x": 1.0, "rsi": 50.0})
        self.assertIsNotNone(proba)

    def test_malformed_model_returns_none(self):
        self.temp_model.write_text("not valid json {")
        self.assertFalse(fast_reversal_model.is_available())
        self.assertIsNone(fast_reversal_model.predict_proba({}))

    def test_metadata_when_loaded(self):
        model_data = {
            "model_type": "logistic_regression",
            "feature_names": ["vol_x"],
            "weights": [0.5],
            "bias": 0.0,
            "means": [1.0],
            "stds": [0.5],
            "trained_on": 200,
            "auc_train": 0.72,
            "auc_val": 0.68,
            "label_positive_rate": 0.35,
        }
        self.temp_model.write_text(json.dumps(model_data))

        meta = fast_reversal_model.model_metadata()
        self.assertIsNotNone(meta)
        self.assertEqual(meta["trained_on"], 200)
        self.assertAlmostEqual(meta["auc_val"], 0.68)


if __name__ == "__main__":
    unittest.main(verbosity=2)
