"""Unit tests for pipeline_notify.

Run:
    pyembed\\python.exe files\\test_pipeline_notify.py
    pyembed\\python.exe -m unittest files/test_pipeline_notify.py

Design rules:
  - NEVER touch the real Telegram API: all sends go through an injected mock
    http_post returning (status, body).
  - NEVER touch the real .chat_ids / dedup file: every test points module
    constants at tempdir paths via monkeypatch helpers.
  - NEVER depend on env vars in tests: token is set/cleared explicitly per
    test.

If you change pipeline_notify.py, add a test before merging. The whole point
of having tests is to detect when "harmless refactor" silently changes the
delivery contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, timezone
from pathlib import Path
from unittest import mock

# Make sure we import the module under test from this folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline_notify as N
import pipeline_lib as PL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def temp_tree():
    """Yield a temporary tree with chat_ids / runtime / pipeline subdirs.
    All module-level paths are patched to it for the duration of the test."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chat_ids   = root / ".chat_ids"
        runtime    = root / ".runtime"
        pipeline   = runtime / "pipeline"
        health     = pipeline / "health"
        attrib_dir = pipeline / "attribution"
        hypotheses = pipeline / "hypotheses"
        do_not_touch = pipeline / "do_not_touch.json"
        for d in (pipeline, health, attrib_dir, hypotheses):
            d.mkdir(parents=True, exist_ok=True)
        dedup = runtime / "tg_send_dedup.json"

        # Patch module-level constants. Multiple modules see PL.HEALTH etc.,
        # so we patch them too.
        with mock.patch.object(N, "CHAT_IDS_FILE", chat_ids), \
             mock.patch.object(N, "DEDUP_FILE", dedup), \
             mock.patch.object(N.PL, "HEALTH", health), \
             mock.patch.object(N.PL, "PIPELINE", pipeline), \
             mock.patch.object(N.PL, "RUNTIME", runtime), \
             mock.patch.object(N.PL, "HYPOTHESES", hypotheses), \
             mock.patch.object(N.PL, "DO_NOT_TOUCH", do_not_touch):
            yield {
                "root":       root,
                "chat_ids":   chat_ids,
                "dedup":      dedup,
                "health":     health,
                "attrib":     attrib_dir,
                "hypotheses": hypotheses,
                "do_not_touch": do_not_touch,
            }


def write_hyp(hyp_dir: Path, hyp: dict) -> Path:
    """Write a hypothesis JSON into the patched HYPOTHESES dir."""
    hid = hyp["hypothesis_id"]
    p = hyp_dir / f"{hid}.json"
    p.write_text(json.dumps(hyp), encoding="utf-8")
    return p


def make_hyp(
    *,
    hid: str = "h-test",
    rule: str = "test_rule",
    config_key: str = "TEST_KEY",
    status: str = "pending_validation",
    severity: str = "red",
    generator: str = "rule",
    days_red: int = 1,
    expected: dict | None = None,
    l3_verdict: str | None = None,
    shadow_verdict: dict | None = None,
) -> dict:
    out = {
        "hypothesis_id":   hid,
        "rule":            rule,
        "config_key":      config_key,
        "status":          status,
        "severity":        severity,
        "generator":       generator,
        "persistence":     {"days_red": days_red, "values": []},
        "expected_delta":  expected or {"x": "+0.1..+0.2"},
        "diff":            {"from": 1, "to": 2},
    }
    if l3_verdict is not None:
        out["validation_report"] = {"result": {"verdict": l3_verdict, "reason": "test"}}
    if shadow_verdict is not None:
        out["shadow_report"] = {"verdict": shadow_verdict}
    return out


def write(p: Path, text: str, encoding: str = "utf-8") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)


def fake_http_post_factory(*, status: int = 200, body: str = '{"ok":true}'):
    """Make a recording mock http_post."""
    calls: list[dict] = []

    def post(url: str, payload: dict, *, timeout: int) -> tuple[int, str]:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return status, body

    post.calls = calls   # type: ignore[attr-defined]
    return post


# ---------------------------------------------------------------------------
# load_chat_ids
# ---------------------------------------------------------------------------


class LoadChatIdsTests(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        with temp_tree() as t:
            self.assertEqual(N.load_chat_ids(), [])

    def test_plain_json_list(self):
        with temp_tree() as t:
            write(t["chat_ids"], "[111, 222, 333]")
            self.assertEqual(N.load_chat_ids(), [111, 222, 333])

    def test_handles_utf8_bom(self):
        # PowerShell's Out-File defaults to UTF-8 with BOM. The reader must cope.
        with temp_tree() as t:
            write(t["chat_ids"], "[555, 666]", encoding="utf-8-sig")
            self.assertEqual(N.load_chat_ids(), [555, 666])

    def test_dedups_and_sorts(self):
        with temp_tree() as t:
            write(t["chat_ids"], "[300, 100, 200, 100]")
            self.assertEqual(N.load_chat_ids(), [100, 200, 300])

    def test_malformed_json_returns_empty(self):
        with temp_tree() as t:
            write(t["chat_ids"], "{not even close to json")
            self.assertEqual(N.load_chat_ids(), [])

    def test_non_list_returns_empty(self):
        with temp_tree() as t:
            write(t["chat_ids"], '{"chat_id": 111}')
            self.assertEqual(N.load_chat_ids(), [])

    def test_non_int_items_filtered(self):
        with temp_tree() as t:
            write(t["chat_ids"], '[123, "abc", null, 456]')
            self.assertEqual(N.load_chat_ids(), [123, 456])


# ---------------------------------------------------------------------------
# get_telegram_token
# ---------------------------------------------------------------------------


class GetTokenTests(unittest.TestCase):

    def setUp(self):
        # Save & clear env to ensure deterministic results per test
        self._saved = os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["TELEGRAM_BOT_TOKEN"] = self._saved
        else:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def test_token_from_env(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "test:token"
        self.assertEqual(N.get_telegram_token(), "test:token")

    def test_token_missing_returns_none(self):
        with temp_tree() as t:   # also clears runner file location
            self.assertIsNone(N.get_telegram_token())

    def test_token_from_runner_fallback(self):
        with temp_tree() as t:
            write(t["root"] / ".runtime" / "bot_bg_runner.cmd",
                  "@echo off\nset TELEGRAM_BOT_TOKEN=fallback:tok\nstart bot\n")
            self.assertEqual(N.get_telegram_token(), "fallback:tok")

    def test_env_wins_over_file(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "from_env"
        with temp_tree() as t:
            write(t["root"] / ".runtime" / "bot_bg_runner.cmd",
                  "set TELEGRAM_BOT_TOKEN=from_file\n")
            self.assertEqual(N.get_telegram_token(), "from_env")


# ---------------------------------------------------------------------------
# read_health_tg + build_attribution_block + build_full_message
# ---------------------------------------------------------------------------


class HealthReadingTests(unittest.TestCase):

    def test_read_existing(self):
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            self.assertEqual(N.read_health_tg(date(2026, 5, 12)), "🩺 ok")

    def test_read_missing_returns_none(self):
        with temp_tree() as t:
            self.assertIsNone(N.read_health_tg(date(2026, 5, 12)))


class AttributionBlockTests(unittest.TestCase):

    def test_none_when_no_report(self):
        with temp_tree() as t:
            self.assertIsNone(N.build_attribution_block(None))

    def test_none_when_all_pending(self):
        rep = {"pipeline_meta": {"n_evaluated": 2, "by_verdict": {"needs_data": 2},
                                 "hit_rate": None}}
        self.assertIsNone(N.build_attribution_block(rep))

    def test_renders_hits_and_misses(self):
        rep = {"pipeline_meta": {
            "n_evaluated": 4,
            "by_verdict": {"hit": 2, "miss": 1, "needs_data": 1},
            "hit_rate": 0.67,
        }}
        block = N.build_attribution_block(rep)
        self.assertIsNotNone(block)
        self.assertIn("hit_rate", block)
        self.assertIn("0.67", block)
        self.assertIn("✅2", block)
        self.assertIn("❌1", block)
        self.assertIn("⏳1", block)

    def test_warns_on_regression(self):
        rep = {"pipeline_meta": {
            "n_evaluated": 2,
            "by_verdict": {"hit": 1, "regression": 1},
            "hit_rate": 0.5,
        }}
        block = N.build_attribution_block(rep)
        self.assertIn("regression", block.lower())


class BuildFullMessageTests(unittest.TestCase):

    def test_none_when_no_health(self):
        with temp_tree() as t:
            self.assertIsNone(N.build_full_message(date(2026, 5, 12)))

    def test_health_only(self):
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 north star ok")
            msg = N.build_full_message(date(2026, 5, 12))
            self.assertIsNotNone(msg)
            self.assertIn("🩺 north star ok", msg)
            self.assertNotIn("Pipeline Attribution", msg)

    def test_health_with_attribution(self):
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            report = {"pipeline_meta": {
                "n_evaluated": 1, "by_verdict": {"hit": 1},
                "hit_rate": 1.0}}
            write(t["attrib"] / "attribution-x.json", json.dumps(report))
            msg = N.build_full_message(date(2026, 5, 12))
            self.assertIn("🩺 ok", msg)
            self.assertIn("Pipeline Attribution", msg)

    def test_truncates_oversize(self):
        with temp_tree() as t:
            big = "x" * 10000
            write(t["health"] / "health-2026-05-12.tg.txt", big)
            msg = N.build_full_message(date(2026, 5, 12))
            self.assertLessEqual(len(msg), N.TG_MAX_CHARS)
            self.assertTrue(msg.endswith("[truncated]"))


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class DedupTests(unittest.TestCase):

    def test_empty_file_not_blocked(self):
        with temp_tree() as t:
            self.assertFalse(N.is_dedup_blocked(date(2026, 5, 12)))

    def test_mark_blocks_same_day(self):
        with temp_tree() as t:
            N.mark_dedup(date(2026, 5, 12), now_iso="2026-05-12T00:00:00Z")
            self.assertTrue(N.is_dedup_blocked(date(2026, 5, 12)))

    def test_different_day_not_blocked(self):
        with temp_tree() as t:
            N.mark_dedup(date(2026, 5, 11), now_iso="2026-05-11T00:00:00Z")
            self.assertFalse(N.is_dedup_blocked(date(2026, 5, 12)))

    def test_mark_preserves_existing(self):
        with temp_tree() as t:
            # Pre-existing unrelated key from another component
            t["dedup"].write_text(json.dumps({"train_session": "2026-05-10T00:00:00Z"}),
                                  encoding="utf-8")
            N.mark_dedup(date(2026, 5, 12), now_iso="2026-05-12T03:00:00Z")
            state = json.loads(t["dedup"].read_text(encoding="utf-8"))
            self.assertEqual(state["train_session"], "2026-05-10T00:00:00Z")
            self.assertEqual(state["pipeline_health_2026-05-12"],
                             "2026-05-12T03:00:00Z")


# ---------------------------------------------------------------------------
# send_to_chat — pure injection, no network
# ---------------------------------------------------------------------------


class SendToChatTests(unittest.TestCase):

    def test_success(self):
        post = fake_http_post_factory(status=200, body='{"ok":true}')
        r = N.send_to_chat("tok", 999, "hello", http_post=post)
        self.assertTrue(r["ok"])
        self.assertEqual(r["status"], 200)
        self.assertEqual(len(post.calls), 1)
        call = post.calls[0]
        self.assertIn("/bot tok/sendMessage".replace(" ", ""),
                      call["url"])  # url contains token
        self.assertEqual(call["payload"]["chat_id"], 999)
        self.assertEqual(call["payload"]["parse_mode"], "HTML")
        self.assertTrue(call["payload"]["disable_web_page_preview"])

    def test_http_error_marks_not_ok(self):
        post = fake_http_post_factory(status=429, body='{"ok":false,"description":"rate"}')
        r = N.send_to_chat("tok", 999, "hello", http_post=post)
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], 429)

    def test_network_error_marks_not_ok(self):
        post = fake_http_post_factory(status=0, body="network: ConnectionRefusedError")
        r = N.send_to_chat("tok", 999, "hello", http_post=post)
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], 0)


# ---------------------------------------------------------------------------
# notify — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------


class NotifyTests(unittest.TestCase):

    def setUp(self):
        self._saved_tok = os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def tearDown(self):
        if self._saved_tok is not None:
            os.environ["TELEGRAM_BOT_TOKEN"] = self._saved_tok
        else:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def test_skips_when_no_token(self):
        with temp_tree() as t:
            write(t["chat_ids"], "[1]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory()
            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["skipped"], "no_token")
            self.assertEqual(len(post.calls), 0)

    def test_skips_when_no_chat_ids(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory()
            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["skipped"], "no_chat_ids")
            self.assertEqual(len(post.calls), 0)

    def test_skips_when_no_health_report(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[1]")
            post = fake_http_post_factory()
            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["skipped"], "no_health_report")
            self.assertEqual(len(post.calls), 0)

    def test_sends_to_all_chats(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10, 20, 30]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory(status=200)
            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["sent"], [10, 20, 30])
            self.assertEqual(r["errors"], [])
            self.assertEqual(len(post.calls), 3)
            for cid, call in zip([10, 20, 30], post.calls):
                self.assertEqual(call["payload"]["chat_id"], cid)
                self.assertIn("🩺 ok", call["payload"]["text"])

    def test_partial_failure_recorded(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10, 20]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            statuses = iter([200, 500])

            def post(url, payload, *, timeout):
                return next(statuses), '{"ok":false}'

            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["sent"], [10])
            self.assertEqual(len(r["errors"]), 1)
            self.assertEqual(r["errors"][0]["chat_id"], 20)
            self.assertEqual(r["errors"][0]["status"], 500)

    def test_dedup_blocks_second_call_same_day(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory()
            r1 = N.notify(date(2026, 5, 12), http_post=post,
                          now_iso="2026-05-12T03:00:00Z")
            r2 = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r1["sent"], [10])
            self.assertEqual(r2["skipped"], "already_sent_today")
            self.assertEqual(len(post.calls), 1)   # second call must NOT POST

    def test_force_bypasses_dedup(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory()
            N.notify(date(2026, 5, 12), http_post=post,
                     now_iso="2026-05-12T03:00:00Z")
            r2 = N.notify(date(2026, 5, 12), http_post=post, force=True)
            self.assertEqual(r2["sent"], [10])
            self.assertEqual(len(post.calls), 2)

    def test_dry_run_does_not_post_or_dedup(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory()
            r = N.notify(date(2026, 5, 12), dry_run=True, http_post=post)
            self.assertEqual(r["skipped"], "dry_run")
            self.assertIn("message_preview", r)
            self.assertEqual(len(post.calls), 0)
            # And dedup must remain unchanged
            self.assertFalse(N.is_dedup_blocked(date(2026, 5, 12)))

    def test_dedup_only_marked_when_any_success(self):
        """Total failure must NOT mark dedup — we want a retry to be possible."""
        os.environ["TELEGRAM_BOT_TOKEN"] = "tok"
        with temp_tree() as t:
            write(t["chat_ids"], "[10, 20]")
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            post = fake_http_post_factory(status=500, body="server err")
            r = N.notify(date(2026, 5, 12), http_post=post)
            self.assertEqual(r["sent"], [])
            self.assertEqual(len(r["errors"]), 2)
            self.assertFalse(N.is_dedup_blocked(date(2026, 5, 12)))


# ---------------------------------------------------------------------------
# build_hypothesis_review_block + collect_review_candidates
# ---------------------------------------------------------------------------


class HypothesisReviewBlockTests(unittest.TestCase):

    def test_no_candidates_returns_none(self):
        with temp_tree() as t:
            self.assertIsNone(N.build_hypothesis_review_block())
            self.assertEqual(N.collect_review_candidates(), [])

    def test_pending_validation_with_no_l3_is_eligible(self):
        with temp_tree() as t:
            # No validation_report yet → still eligible (operator should see it)
            write_hyp(t["hypotheses"], make_hyp(hid="h-eligible"))
            block = N.build_hypothesis_review_block()
            self.assertIsNotNone(block)
            self.assertIn("h-eligible", block)
            # New format starts with the Russian header
            self.assertIn("Готовы к применению", block)

    def test_l3_reject_excluded(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-rejected",
                                                l3_verdict="reject"))
            self.assertIsNone(N.build_hypothesis_review_block())

    def test_status_approved_excluded(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-done", status="approved"))
            self.assertIsNone(N.build_hypothesis_review_block())

    def test_status_rolled_back_excluded(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-rb", status="rolled_back"))
            self.assertIsNone(N.build_hypothesis_review_block())

    def test_locked_config_key_excluded(self):
        with temp_tree() as t:
            t["do_not_touch"].write_text(
                json.dumps({"config_keys_locked": ["LOCKED_KEY"], "gates": []}),
                encoding="utf-8")
            write_hyp(t["hypotheses"], make_hyp(hid="h-locked",
                                                config_key="LOCKED_KEY"))
            self.assertIsNone(N.build_hypothesis_review_block())

    def test_sort_by_severity_then_persistence(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-yellow", severity="yellow", days_red=10))
            write_hyp(t["hypotheses"], make_hyp(hid="h-red",      severity="red",    days_red=2))
            write_hyp(t["hypotheses"], make_hyp(hid="h-critical", severity="critical", days_red=1))
            cands = N.collect_review_candidates()
            ids = [c["hypothesis_id"] for c in cands]
            self.assertEqual(ids[0], "h-critical")
            self.assertEqual(ids[1], "h-red")
            self.assertEqual(ids[2], "h-yellow")

    def test_top_n_caps_output(self):
        with temp_tree() as t:
            for i in range(5):
                write_hyp(t["hypotheses"], make_hyp(hid=f"h-{i}"))
            self.assertEqual(len(N.collect_review_candidates(top_n=2)), 2)

    def test_includes_measured_pnl_before_after(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-measured", rule="disable_mode_impulse_speed")
            hyp["shadow_report"] = {
                "window_days": 60,
                "verdict": {"DISABLE_MODE_IMPULSE_SPEED": {"verdict": "accept",
                                                          "n_events": 946}},
                "summary": {"by_feature": {"DISABLE_MODE_IMPULSE_SPEED": {
                    "n_events": 946,
                    "prod_median_pnl_pct":   -0.0016,
                    "shadow_median_pnl_pct": 0.0,
                    "delta_median_pnl_pct":  0.0016,
                }}},
            }
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertIn("Бэктест", block)
            self.assertIn("946", block)
            # Before/after numbers must appear
            self.assertIn("PnL/трейд", block)

    def test_volume_drop_for_disable_mode(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-vol", rule="disable_mode_impulse_speed")
            hyp["shadow_report"] = {
                "window_days": 60,
                "verdict": {"X": {"verdict": "accept", "n_events": 900}},
                "summary": {"by_feature": {"X": {
                    "n_events": 900,
                    "prod_median_pnl_pct": -0.01,
                    "shadow_median_pnl_pct": 0.0,
                    "delta_median_pnl_pct": 0.01,
                }}},
            }
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            # Volume reduction is rendered as "−N трейдов/день" + % context
            self.assertIn("Объём после apply", block)
            self.assertIn("трейдов/день", block)

    def test_volume_increase_for_widen(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-widen", rule="widen_watchlist_match_tolerance")
            hyp["shadow_report"] = {
                "window_days": 60,
                "verdict": {"X": {"verdict": "accept", "n_events": 2081}},
                "summary": {"by_feature": {"X": {
                    "n_events": 2081,
                    "prod_median_pnl_pct": 0.0,
                    "shadow_median_pnl_pct": 0.0,
                    "delta_median_pnl_pct": 0.0,
                }}},
            }
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertIn("2081", block)
            self.assertIn("/день", block)
            # The hand-curated caveat should be shown
            self.assertIn("upper-bound", block)

    def test_no_shadow_shows_no_backtest_note(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-nosim", rule="relax_gate_foo")
            hyp["rationale"] = "Gate foo over-blocks profitable events"
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertIn("не выполнен", block)
            self.assertIn("Gate foo", block)

    def test_expected_delta_shown_as_projection(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-proj")
            hyp["expected_delta"] = {"total_realized_pnl_pct": "+5..+9"}
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertIn("Цель", block)
            self.assertIn("+5..+9", block)

    def test_includes_apply_command_per_hypothesis(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-cmd-check"))
            block = N.build_hypothesis_review_block()
            # Must be runnable in PowerShell — needs the pyembed interpreter
            self.assertIn("pyembed", block)
            self.assertIn("pipeline_approve.py --hypothesis h-cmd-check", block)

    def test_claude_advisor_picked_up_from_log(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-advised"))
            adv = {
                "ts": "2026-05-11T16:00:00Z",
                "hypothesis_id": "h-advised",
                "recommendation": "approve",
                "confidence": "high",
                "justification": "looks good",
            }
            advisory_path = t["root"] / ".runtime" / "pipeline" / "approve_advisory.jsonl"
            advisory_path.write_text(json.dumps(adv) + "\n", encoding="utf-8")
            block = N.build_hypothesis_review_block()
            self.assertIn("APPROVE", block)
            self.assertIn("high", block)
            # Header summary counts approve as ✅
            self.assertIn("✅1", block)

    def test_latest_advisor_record_wins(self):
        with temp_tree() as t:
            write_hyp(t["hypotheses"], make_hyp(hid="h-multi"))
            advisory_path = t["root"] / ".runtime" / "pipeline" / "approve_advisory.jsonl"
            lines = [
                {"hypothesis_id": "h-multi", "recommendation": "reject",
                 "confidence": "low", "ts": "2026-05-10T00:00:00Z"},
                {"hypothesis_id": "h-multi", "recommendation": "approve",
                 "confidence": "medium", "ts": "2026-05-11T00:00:00Z"},
            ]
            advisory_path.write_text("\n".join(json.dumps(l) for l in lines),
                                     encoding="utf-8")
            block = N.build_hypothesis_review_block()
            self.assertIn("APPROVE", block)
            self.assertNotIn("REJECT", block)


class PipelineVerdictTests(unittest.TestCase):
    """The bottom-line ✅/⚠️/❌ verdict is the headline the operator reads
    first. The logic must stay deterministic and conservative — these tests
    pin it down."""

    def _hyp(self, rule: str = "disable_mode_x") -> dict:
        return {"rule": rule}

    def test_no_measurement_is_reject(self):
        e, h, _ = N._pipeline_verdict(self._hyp(), None, None)
        self.assertEqual(e, "❌")
        self.assertIn("Не апрув", h)   # "Не апрувить"

    def test_small_n_is_caution(self):
        m = {"pnl_delta": 0.5, "n_events": 10}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "⚠️")

    def test_clear_regression_is_reject(self):
        m = {"pnl_delta": -0.2, "n_events": 500}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "❌")

    def test_noise_is_caution(self):
        m = {"pnl_delta": 0.01, "n_events": 500}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "⚠️")
        self.assertIn("значимого эффекта", _)

    def test_upper_bound_caveat_caps_to_caution(self):
        # widen_watchlist has a hand-curated upper-bound caveat
        m = {"pnl_delta": 0.5, "n_events": 500}
        e, h, _ = N._pipeline_verdict(
            {"rule": "widen_watchlist_match_tolerance"}, m, None)
        self.assertEqual(e, "⚠️")
        self.assertIn("upper-bound", _)

    def test_strong_signal_is_approve(self):
        m = {"pnl_delta": 0.25, "n_events": 946,
             "recent": {"recent_days": 30, "n_events": 480, "pnl_delta": 0.22}}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "✅")
        self.assertIn("APPROVE", h)

    def test_recency_reversal_downgrades_to_caution(self):
        m = {"pnl_delta": 0.25, "n_events": 946,
             "recent": {"recent_days": 30, "n_events": 480, "pnl_delta": -0.15}}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "⚠️")
        self.assertIn("минус", _)

    def test_recency_weakening_says_with_caveat(self):
        m = {"pnl_delta": 0.30, "n_events": 946,
             "recent": {"recent_days": 30, "n_events": 480, "pnl_delta": 0.08}}
        e, h, _ = N._pipeline_verdict(self._hyp(), m, None)
        self.assertEqual(e, "⚠️")
        self.assertIn("оговор", h.lower() + _.lower())


class VolumeContextTests(unittest.TestCase):

    def test_returns_none_when_no_volume_data(self):
        # No trades_before/after → can't compute
        m = {"trades_before": None, "trades_after": None, "context": {}}
        self.assertIsNone(N._volume_context_str(m, days=60))

    def test_renders_pct_when_total_available(self):
        m = {"trades_before": 946, "trades_after": 0,
             "context": {"total_takes_in_window": 3000}}
        s = N._volume_context_str(m, days=60)
        self.assertIn("трейдов/день", s)
        self.assertIn("%", s)
        # 946 → 0 over 60d is −15.77/day; total is 50/day; so ~−31.5% → -32
        self.assertIn("32", s)

    def test_falls_back_when_no_total(self):
        m = {"trades_before": 100, "trades_after": 0, "context": {}}
        s = N._volume_context_str(m, days=60)
        self.assertIn("трейдов/день", s)
        self.assertNotIn("%", s)


class RecencyDriftLabelTests(unittest.TestCase):

    def test_none_when_both_below_noise(self):
        self.assertIsNone(N._recency_drift_label(0.001, 0.001))

    def test_strengthening(self):
        self.assertEqual(N._recency_drift_label(0.1, 0.2),
                         "эффект усиливается")

    def test_weakening(self):
        s = N._recency_drift_label(0.3, 0.1)
        self.assertIn("ослаб", s)

    def test_direction_reversal(self):
        s = N._recency_drift_label(0.2, -0.2)
        self.assertIn("направление", s)

    def test_stable(self):
        self.assertEqual(N._recency_drift_label(0.2, 0.2),
                         "эффект стабилен")


class HumanizeRuleTests(unittest.TestCase):
    """Make sure rule_id translations stay stable — operator depends on
    these strings being readable Russian, not technical IDs."""

    def test_disable_mode(self):
        h = {"rule": "disable_mode_impulse_speed", "diff": {"from": True, "to": False}}
        self.assertEqual(N.humanize_rule(h), "Отключить режим входа «impulse_speed»")

    def test_widen_watchlist(self):
        h = {"rule": "widen_watchlist_match_tolerance",
             "diff": {"from": 0.0, "to": 0.15}}
        self.assertIn("watchlist", N.humanize_rule(h))
        self.assertIn("0.0", N.humanize_rule(h))
        self.assertIn("0.15", N.humanize_rule(h))

    def test_relax_gate(self):
        h = {"rule": "relax_gate_ranker_hard_veto", "diff": {"from": "x", "to": "y"}}
        self.assertEqual(N.humanize_rule(h), "Ослабить фильтр «ranker_hard_veto» (+10%)")

    def test_tighten_proba(self):
        h = {"rule": "tighten_proba_alignment", "diff": {}}
        title = N.humanize_rule(h)
        self.assertIn("уверенности", title)
        self.assertIn("alignment", title)

    def test_unknown_rule_fallback_to_diff(self):
        h = {"rule": "wholly_new_rule", "diff": {"from": 1, "to": 2}}
        self.assertIn("1", N.humanize_rule(h))
        self.assertIn("2", N.humanize_rule(h))

    def test_unknown_rule_no_diff(self):
        h = {"rule": "exotic_rule", "diff": {}}
        self.assertEqual(N.humanize_rule(h), "exotic_rule")


class DeltaFormattingTests(unittest.TestCase):

    def test_emoji_positive_normal(self):
        self.assertEqual(N._delta_emoji(0.01), "✅")

    def test_emoji_negative_normal(self):
        self.assertEqual(N._delta_emoji(-0.01), "❌")

    def test_emoji_lower_is_better_positive_is_bad(self):
        self.assertEqual(N._delta_emoji(0.01, lower_is_better=True), "❌")

    def test_emoji_lower_is_better_negative_is_good(self):
        self.assertEqual(N._delta_emoji(-0.01, lower_is_better=True), "✅")

    def test_emoji_noise_is_neutral(self):
        self.assertEqual(N._delta_emoji(0.0001), "◐")

    def test_emoji_none_is_neutral(self):
        self.assertEqual(N._delta_emoji(None), "◐")


class RollbackBlockTests(unittest.TestCase):
    """L7 monitor writes rollback recommendations to a jsonl file. We must
    de-dup against the decisions log (so a decision already rolled back
    doesn't keep appearing in Telegram)."""

    def _write_log(self, dir_path: Path, records: list[dict]) -> None:
        log = dir_path / "rollback_recommendations.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_none_when_no_log(self):
        with temp_tree() as t:
            self.assertIsNone(N.build_rollback_block())

    def test_renders_single_recommendation(self):
        with temp_tree() as t:
            mon = t["root"] / ".runtime" / "pipeline" / "monitor"
            self._write_log(mon, [{
                "decision_id":   "d-2026-05-04T000000Z-abc",
                "hypothesis_id": "h-foo",
                "rule":          "disable_mode_impulse",
                "config_key":    "MODE_IMPULSE_ENABLED",
                "outcome":       {"verdict": "miss", "reason": "expected +5pp, actual -2pp"},
            }])
            with mock.patch.object(N, "ROLLBACK_LOG_PATH",
                                   mon / "rollback_recommendations.jsonl"):
                block = N.build_rollback_block()
            self.assertIsNotNone(block)
            self.assertIn("disable_mode_impulse", block)
            self.assertIn("d-2026-05-04T000000Z-abc", block)
            self.assertIn("--rollback", block)
            self.assertIn("expected +5pp", block)

    def test_excludes_already_rolled_back(self):
        with temp_tree() as t:
            mon = t["root"] / ".runtime" / "pipeline" / "monitor"
            self._write_log(mon, [
                {"decision_id": "d-stale", "rule": "foo", "outcome": {"verdict": "miss"}},
                {"decision_id": "d-fresh", "rule": "bar", "outcome": {"verdict": "miss"}},
            ])
            # Pretend d-stale was already rolled back
            decisions = t["root"] / ".runtime" / "pipeline" / "decisions"
            decisions.mkdir(parents=True, exist_ok=True)
            with (decisions / "decisions.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"decision_id": "d-rb-x",
                                    "stage": "rolled_back",
                                    "rolling_back": "d-stale"}) + "\n")
            with mock.patch.object(N, "ROLLBACK_LOG_PATH",
                                   mon / "rollback_recommendations.jsonl"), \
                 mock.patch.object(N.PL, "DECISIONS_LOG",
                                   decisions / "decisions.jsonl"):
                items = N.collect_rollback_recommendations()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["decision_id"], "d-fresh")

    def test_dedups_same_decision_in_log(self):
        """L7 appends a fresh recommendation every cycle until acted on —
        we must de-dup so Telegram shows each decision once."""
        with temp_tree() as t:
            mon = t["root"] / ".runtime" / "pipeline" / "monitor"
            self._write_log(mon, [
                {"decision_id": "d-A", "rule": "r", "outcome": {"verdict": "miss"}},
                {"decision_id": "d-A", "rule": "r", "outcome": {"verdict": "miss"}},
                {"decision_id": "d-A", "rule": "r", "outcome": {"verdict": "miss"}},
            ])
            with mock.patch.object(N, "ROLLBACK_LOG_PATH",
                                   mon / "rollback_recommendations.jsonl"):
                items = N.collect_rollback_recommendations()
            self.assertEqual(len(items), 1)

    def test_caps_to_max_items(self):
        with temp_tree() as t:
            mon = t["root"] / ".runtime" / "pipeline" / "monitor"
            self._write_log(mon, [
                {"decision_id": f"d-{i}", "rule": "r",
                 "outcome": {"verdict": "miss"}}
                for i in range(10)
            ])
            with mock.patch.object(N, "ROLLBACK_LOG_PATH",
                                   mon / "rollback_recommendations.jsonl"):
                items = N.collect_rollback_recommendations(max_items=2)
            self.assertEqual(len(items), 2)


class HypothesisReviewIncidentEvidenceTests(unittest.TestCase):
    """When L2 attaches `incident_evidence` to a hypothesis, the review
    block must render a one-line "📎 Подкреплено: ..." footnote so the
    operator sees why this hypothesis just got stronger."""

    def test_renders_incident_evidence_line(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-with-evidence", severity="critical")
            hyp["incident_evidence"] = {
                "mode": "impulse_speed",
                "premature": 2, "losers": 5, "missed": 0,
            }
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertIn("Подкреплено", block)
            self.assertIn("premature exit", block)
            self.assertIn("losing trade", block)

    def test_no_line_when_evidence_empty(self):
        with temp_tree() as t:
            hyp = make_hyp(hid="h-no-evidence")
            hyp["incident_evidence"] = {"mode": "x", "premature": 0,
                                        "losers": 0, "missed": 0}
            write_hyp(t["hypotheses"], hyp)
            block = N.build_hypothesis_review_block()
            self.assertNotIn("Подкреплено", block)


class IncidentsBlockTests(unittest.TestCase):
    """Folded-in Skill daily check incidents — premature exits, missed
    sustained trends, losing trades."""

    @staticmethod
    def _mode_report(mode_dir, *, premature=(), losers=(), misses=(),
                     generated_at=None):
        from datetime import datetime, timezone
        if generated_at is None:
            generated_at = datetime.now(timezone.utc).isoformat()
        verdicts = []
        for v in premature:
            verdicts.append({
                "symbol": v["symbol"],
                "sell_lateness_bars": v.get("sell_lateness_bars", -10),
                "capture_ratio": v.get("capture_ratio", 0.1),
                "captured_pnl_pct": v.get("captured_pnl_pct", 0.5),
            })
        for v in losers:
            verdicts.append({
                "symbol": v["symbol"],
                "captured_pnl_pct": v.get("captured_pnl_pct", -5.0),
                "sell_lateness_bars": v.get("sell_lateness_bars", 0),
                "capture_ratio": v.get("capture_ratio", 1.0),
            })
        report = {
            "generated_at": generated_at,
            "trade_verdicts": verdicts,
            "missed_opportunities": [
                {"symbol": m["symbol"],
                 "gain_pct": m.get("gain_pct", 10.0),
                 "true_start_ts": m.get("ts", "2026-05-12T00:00:00+00:00")}
                for m in misses
            ],
            "summary": {},
        }
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    @contextmanager
    def _per_mode(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "per_mode"
            yield d

    def test_none_when_no_reports(self):
        with self._per_mode() as d:
            self.assertIsNone(N.build_incidents_block(per_mode_dir=d))

    def test_none_when_no_incidents(self):
        with self._per_mode() as d:
            self._mode_report(d / "impulse_speed")   # empty trade_verdicts
            self.assertIsNone(N.build_incidents_block(per_mode_dir=d))

    def test_premature_exit_rendered(self):
        with self._per_mode() as d:
            self._mode_report(d / "impulse_speed", premature=[
                {"symbol": "GLMUSDT", "sell_lateness_bars": -32,
                 "capture_ratio": 0.02, "captured_pnl_pct": 0.5},
            ])
            block = N.build_incidents_block(per_mode_dir=d)
            self.assertIsNotNone(block)
            self.assertIn("Premature exits", block)
            self.assertIn("GLMUSDT", block)
            self.assertIn("impulse_speed", block)
            self.assertIn("32b before peak", block)

    def test_losing_trade_rendered(self):
        with self._per_mode() as d:
            self._mode_report(d / "impulse_speed", losers=[
                {"symbol": "LDOUSDT", "captured_pnl_pct": -9.23},
            ])
            block = N.build_incidents_block(per_mode_dir=d)
            self.assertIn("Losing trades", block)
            self.assertIn("LDOUSDT", block)
            self.assertIn("-9.23%", block)

    def test_missed_trend_rendered(self):
        with self._per_mode() as d:
            self._mode_report(d / "alignment", misses=[
                {"symbol": "AUDIOUSDT", "gain_pct": 11.3,
                 "ts": "2026-05-12T10:00:00+00:00"},
            ])
            block = N.build_incidents_block(per_mode_dir=d)
            self.assertIn("Missed sustained trends", block)
            self.assertIn("AUDIOUSDT", block)
            self.assertIn("+11.3%", block)

    def test_missed_trend_below_threshold_skipped(self):
        with self._per_mode() as d:
            self._mode_report(d / "alignment", misses=[
                {"symbol": "X", "gain_pct": 2.0,    # < 5% threshold
                 "ts": "2026-05-12T10:00:00+00:00"},
            ])
            self.assertIsNone(N.build_incidents_block(per_mode_dir=d))

    def test_dedup_misses_across_modes(self):
        """Same coin appearing in multiple per-mode reports is counted once."""
        with self._per_mode() as d:
            miss = {"symbol": "Z", "gain_pct": 8.0,
                    "ts": "2026-05-12T10:00:00+00:00"}
            self._mode_report(d / "trend", misses=[miss])
            self._mode_report(d / "impulse", misses=[miss])
            block = N.build_incidents_block(per_mode_dir=d)
            self.assertEqual(block.count("Z [trend]") + block.count("Z [impulse]"), 1)
            # And the header count reflects dedup: 1 miss, not 2
            self.assertIn("Missed sustained trends (1)", block)

    def test_stale_report_ignored(self):
        from datetime import datetime, timezone, timedelta
        too_old = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        with self._per_mode() as d:
            self._mode_report(d / "impulse_speed", losers=[
                {"symbol": "Y", "captured_pnl_pct": -5.0},
            ], generated_at=too_old)
            self.assertIsNone(N.build_incidents_block(per_mode_dir=d))

    def test_premature_requires_positive_pnl(self):
        """Premature semantic: sold early BUT was in profit. A losing
        position sold early is just a loser, classified there instead."""
        with self._per_mode() as d:
            self._mode_report(d / "impulse_speed", premature=[
                {"symbol": "X", "sell_lateness_bars": -10,
                 "capture_ratio": 0.1, "captured_pnl_pct": -1.0},
            ])
            block = N.build_incidents_block(per_mode_dir=d)
            # neg pnl + bars_early but pnl<=0 → not premature
            self.assertNotIn("Premature exits", block or "")

    def test_max_per_section_caps_output(self):
        with self._per_mode() as d:
            losers = [{"symbol": f"S{i}", "captured_pnl_pct": -3.0 - i}
                      for i in range(10)]
            self._mode_report(d / "impulse_speed", losers=losers)
            block = N.build_incidents_block(per_mode_dir=d, max_per_section=2)
            # 10 losers total in header, only 2 shown in detail
            self.assertIn("Losing trades (10)", block)
            self.assertEqual(block.count("[impulse_speed]"), 2)


class TelegramFlagDefaultsTests(unittest.TestCase):
    """Each of the four duplicating Telegram messages must default to off
    after the B+ rollout. Flipping any back to True without a follow-up
    discussion would re-introduce the noise."""

    def test_all_four_flags_default_false(self):
        import config
        flags = [
            "TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED",
            "RL_TRAIN_TELEGRAM_REPORTS_ENABLED",
            "DAILY_LEARNING_TELEGRAM_REPORTS_ENABLED",
            "SIGNAL_EVALUATOR_TELEGRAM_REPORTS_ENABLED",
        ]
        for f in flags:
            with self.subTest(flag=f):
                self.assertFalse(
                    getattr(config, f, True),
                    f"{f} should default to False after the unified-Telegram rollout."
                )


class FullMessageWithReviewTests(unittest.TestCase):

    def test_health_plus_review_block(self):
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            write_hyp(t["hypotheses"], make_hyp(hid="h-x", severity="critical"))
            msg = N.build_full_message(date(2026, 5, 12))
            self.assertIn("🩺 ok", msg)
            self.assertIn("Готовы к применению", msg)
            self.assertIn("h-x", msg)

    def test_no_review_block_when_nothing_ready(self):
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            msg = N.build_full_message(date(2026, 5, 12))
            self.assertIn("🩺 ok", msg)
            self.assertNotIn("Готовы к применению", msg)

    def test_explicit_review_block_override(self):
        """Tests can short-circuit review building by passing review_block=."""
        with temp_tree() as t:
            write(t["health"] / "health-2026-05-12.tg.txt", "🩺 ok")
            write_hyp(t["hypotheses"], make_hyp(hid="h-suppressed"))
            msg = N.build_full_message(date(2026, 5, 12), review_block=None)
            self.assertNotIn("h-suppressed", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
