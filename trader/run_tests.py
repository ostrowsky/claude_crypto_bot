"""Run the trader's tests: ../pyembed/python.exe run_tests.py (embedded Python does not put cwd on sys.path)."""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# tests must not write into the real audit journal; tests that check auditing pass their own writer
os.environ["TRADER_AUDIT_DISABLED"] = "1"
sys.path.insert(0, str(HERE))

if __name__ == "__main__":    # a process pool (wide_scan) re-imports __main__ in its children on Windows
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))
    sys.exit(0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1)
