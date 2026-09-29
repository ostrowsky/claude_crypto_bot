"""L3 rejects are remembered; the chop bull-day ADX hypothesis evidence (2026-09-29)."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline_lib as PL  # noqa: E402
import pipeline_validator as V  # noqa: E402


class TestRejectionIsRemembered(unittest.TestCase):
    def test_reject_writes_already_tried(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "already_tried.jsonl"
            with mock.patch.object(PL, "ALREADY_TRIED", p):
                V.record_rejection({"rule": "relax_x", "config_key": "X_MIN", "hypothesis_id": "h-1"},
                                   {"verdict": "reject", "reason": "0.70x"})
            rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((rec["rule"], rec["config_key"], rec["stage"]), ("relax_x", "X_MIN", "rejected"))

    def test_l3_calls_it_on_reject(self):
        src = (HERE / "pipeline_validator.py").read_text(encoding="utf-8")
        i = src.index('elif vr.get("verdict") == "reject":')
        self.assertIn("record_rejection(hyp, vr)", src[i:i + 200])

    def test_l2_filters_by_that_memory(self):
        src = (HERE / "pipeline_hypothesis.py").read_text(encoding="utf-8")
        self.assertIn('blocked.add((t.get("rule"), t.get("config_key")))', src)


class TestChopEvidence(unittest.TestCase):
    def test_script(self):
        src = (HERE / "_backtest_chop_bull_adx.py").read_text(encoding="utf-8")
        ast.parse(src)
        self.assertNotIn("D:/", src)
        self.assertIn("TO <= adx < FROM and slope >= SLOPE_B and vol >= VOL_B", src)

    def test_spec(self):
        spec = HERE.parent / "docs/specs/features/chop-bull-adx-spec.md"
        if not spec.exists():
            self.skipTest("spec not in this checkout")
        t = spec.read_text(encoding="utf-8")
        for tok in ("43", "11.0% → 11.0%", "+0.44%", "+1.66%", "−0.134", "ОТКЛОНЕНА"):
            self.assertIn(tok, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
