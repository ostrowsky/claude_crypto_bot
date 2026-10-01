"""Morning report split into Telegram-safe messages (agent-tasks-0929-spec.md §9).

2026-10-01: the report grew past the limit, the old truncation cut through an
HTML tag and Telegram rejected the whole message (HTTP 400) -- no report that day.
"""
import re
import unittest
from datetime import date
from pathlib import Path

import pipeline_notify as N

ROOT = Path(__file__).resolve().parent.parent
DANGLING = re.compile(r"<(?!/?(b|i|u|s|code|pre|a)\b)")


def _balanced(s):
    for t in N._HTML_TAGS:
        if s.count("<%s>" % t) != s.count("</%s>" % t):
            return False
    return not DANGLING.search(s)


class TestSafeTruncate(unittest.TestCase):
    def test_never_cuts_inside_a_tag(self):
        line = "<b>Блок</b> текст " * 6
        msg = "\n".join(line for _ in range(80))
        for limit in range(300, 1200, 37):
            out = N.safe_truncate(msg, limit)
            self.assertLessEqual(len(out), limit + 20)
            self.assertTrue(_balanced(out), out[-80:])
            self.assertTrue(out.endswith("[truncated]"))

    def test_open_tag_is_closed(self):
        self.assertEqual(N._close_tags("a <b>bold"), "a <b>bold</b>")
        self.assertEqual(N._close_tags("a <b>x</b> <"), "a <b>x</b> ")
        self.assertEqual(N._close_tags("<i>a <b>b"), "<i>a <b>b</b></i>")

    def test_short_message_untouched(self):
        self.assertEqual(N.safe_truncate("<b>ok</b>", 100), "<b>ok</b>")


class TestSplit(unittest.TestCase):
    def test_packs_parts_within_the_limit(self):
        parts = ["<b>A</b>" + "x" * 900, "<b>B</b>" + "y" * 900, "<b>C</b>" + "z" * 900]
        chunks = N.split_for_telegram(parts, limit=2000)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 2000 for c in chunks))
        self.assertTrue(all(_balanced(c) for c in chunks))

    def test_oversized_part_is_truncated_safely(self):
        chunks = N.split_for_telegram(["<b>big</b>\n" + "<i>q</i> w\n" * 500], limit=1000)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 1020)
        self.assertTrue(_balanced(chunks[0]))

    def test_notify_sends_every_chunk_and_reports_the_failing_part(self):
        sent = []

        def post(url, payload, timeout):
            sent.append(payload["text"])
            return (400, '{"ok":false,"description":"bad"}') if len(sent) == 2 else (200, "{}")
        old = (N.get_telegram_token, N.load_chat_ids, N._message_parts, N.is_dedup_blocked, N.mark_dedup)
        N.get_telegram_token = lambda: "T"
        N.load_chat_ids = lambda path=None: [1]
        N._message_parts = lambda d, **k: ["<b>h</b>" + "x" * 3000, "<b>a</b>" + "y" * 3000]
        N.is_dedup_blocked = lambda d, state=None: False
        N.mark_dedup = lambda d, **k: None
        try:
            res = N.notify(date(2026, 10, 1), http_post=post)
        finally:
            (N.get_telegram_token, N.load_chat_ids, N._message_parts, N.is_dedup_blocked, N.mark_dedup) = old
        self.assertEqual(res["messages"], 2)
        self.assertEqual(len(sent), 2)
        self.assertEqual(res["sent"], [])
        self.assertEqual(res["errors"][0]["part"], 2)
        self.assertEqual(res["errors"][0]["status"], 400)

    def test_documented(self):
        spec = (ROOT / "docs/specs/features/agent-tasks-0929-spec.md").read_text(encoding="utf-8")
        self.assertIn("split_for_telegram", spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
