"""Guards for the 2026-09-25 algorithm audit (hypothesis package, no behaviour change).

Spec: docs/specs/features/algorithm-audit-0925-spec.md
"""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / "docs" / "specs" / "features" / "algorithm-audit-0925-spec.md"
SCRIPTS = ("_audit_winner_funnel.py", "_audit_birth_blockers.py", "_audit_exits.py", "_audit_misc.py")


class TestScriptsAreReproducible(unittest.TestCase):
    def test_they_parse_and_use_no_machine_paths(self):
        for s in SCRIPTS:
            src = (HERE / s).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertNotIn("D:/Projects", src, s)
            self.assertIn("Path(__file__).resolve().parent", src, s)

    def test_the_winner_population_is_the_immutable_one(self):
        src = (HERE / "_audit_winner_funnel.py").read_text(encoding="utf-8")
        self.assertIn("rank_before_filter=True", src)
        self.assertIn("LS.intraday_deadlines()", src)

    def test_exit_leader_rank_is_real_time_observable(self):
        # return since the UTC open at the exit bar -- nothing from later in the day
        src = (HERE / "_audit_exits.py").read_text(encoding="utf-8")
        self.assertIn("ret[x[0]][sym] = (x[4] / day_open - 1) * 100", src)


class TestThePackage(unittest.TestCase):
    def setUp(self):
        if not SPEC.exists():
            self.skipTest("spec not in this checkout")
        self.text = SPEC.read_text(encoding="utf-8")

    def test_every_hypothesis_has_a_priority_and_a_validation(self):
        section = self.text.split("## Пакет гипотез", 1)[1].split(chr(10) + "## ", 1)[0]
        rows = [l for l in section.splitlines()
                if l.startswith("| ") and not l.startswith("| ID") and not l.startswith("|---")]
        ids = [l.split("|")[1].strip().strip("*") for l in rows]
        self.assertGreaterEqual(len(ids), 20)
        for l in rows:
            cells = [c.strip() for c in l.split("|")]
            self.assertIn(cells[2], ("P0", "P1", "P2", "P3"), l[:40])
            self.assertTrue(cells[5], "validation missing: " + l[:40])

    def test_headline_numbers_travel_with_the_package(self):
        for tok in ("166", "42.2%", "9.0%", "+0.72%", "0.421", "+6.0", "372 из 465", "+0.46%"):
            self.assertIn(tok, self.text)

    def test_the_validation_principle_is_the_goal_metric(self):
        self.assertIn("EarlyCapture@move_lead", self.text)
        self.assertIn("неухудшения", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
