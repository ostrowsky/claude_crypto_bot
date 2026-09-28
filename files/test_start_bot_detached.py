"""start_bot_detached.ps1: start the bot outside the caller's process tree (2026-09-28)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PS1 = ROOT / "start_bot_detached.ps1"


class TestDetachedStart(unittest.TestCase):
    def setUp(self):
        self.src = PS1.read_text(encoding="utf-8")

    def test_ascii_only(self):
        # CLAUDE.md: Windows consoles are cp1251 -- scripts stay ASCII
        self.src.encode("ascii")

    def test_reads_the_token_itself_and_hands_it_to_start_bot_bg(self):
        self.assertIn(r".runtime\bot_bg_runner.cmd", self.src)
        self.assertIn(r"files\.env", self.src)
        self.assertIn('start_bot_bg.ps1") -Token $tok', self.src)

    def test_never_writes_the_token(self):
        for line in self.src.splitlines():
            if "Out-File" in line and "$tok" in line:
                # only the boolean "was a token found" may reach the log
                self.assertEqual(line.count("$tok"), line.count("[bool]$tok"), line)

    def test_no_hardcoded_root(self):
        self.assertIn("$root = $PSScriptRoot", self.src)

    def test_documented(self):
        for doc in ("CLAUDE.md", "PROJECT_CONTEXT.md"):
            self.assertIn("start_bot_detached.ps1", (ROOT / doc).read_text(encoding="utf-8"), doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
