"""Diagnose why bot doesn't react to /start.

Reads token from .runtime/bot_bg_runner.cmd (where it's already set as env
var for the running bot), then queries Telegram API:
  - getMe (verify token works)
  - getWebhookInfo (check webhook conflict)
  - getUpdates with offset=-3 (see last 3 updates — confirms /start arrived)
"""
from __future__ import annotations
import asyncio, io, re, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

def read_token() -> str:
    cmd = ROOT / ".runtime" / "bot_bg_runner.cmd"
    if not cmd.exists():
        return ""
    txt = cmd.read_text(encoding="utf-8")
    m = re.search(r"set\s+TELEGRAM_BOT_TOKEN\s*=\s*(\S+)", txt, re.IGNORECASE)
    return m.group(1).strip() if m else ""


async def main():
    import aiohttp
    token = read_token()
    if not token:
        print("[diag] no token found in .runtime/bot_bg_runner.cmd"); return
    masked = token.split(":")[0] + ":***"
    print(f"[diag] token id: {masked}")

    base = f"https://api.telegram.org/bot{token}"
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(f"{base}/getMe") as r:
            j = await r.json()
            print(f"\n[getMe] ok={j.get('ok')}")
            if j.get("ok"):
                u = j.get("result", {})
                print(f"  bot_id={u.get('id')}  username=@{u.get('username')}")
                print(f"  can_read_group_msgs={u.get('can_read_all_group_messages')}")
            else:
                print(f"  error: {j}")
                return

        async with s.get(f"{base}/getWebhookInfo") as r:
            j = await r.json()
            print(f"\n[getWebhookInfo] ok={j.get('ok')}")
            wh = j.get("result", {}) or {}
            url = wh.get("url") or ""
            print(f"  webhook_url:        {url!r}")
            print(f"  pending_update_cnt: {wh.get('pending_update_count')}")
            print(f"  last_error_date:    {wh.get('last_error_date')}")
            print(f"  last_error_msg:     {wh.get('last_error_message')!r}")
            if url:
                print("  ⚠️  WEBHOOK SET — getUpdates will conflict; needs deleteWebhook")

        # offset=-3 returns last 3 updates without consuming them
        async with s.get(f"{base}/getUpdates?limit=10&offset=-10") as r:
            j = await r.json()
            print(f"\n[getUpdates last-10] ok={j.get('ok')}  count={len(j.get('result',[]) or [])}")
            for u in (j.get("result", []) or [])[-10:]:
                msg = u.get("message") or u.get("edited_message") or {}
                chat = msg.get("chat", {}) or {}
                from_u = msg.get("from", {}) or {}
                txt = msg.get("text") or "(no text)"
                print(f"  upd_id={u.get('update_id')}  chat={chat.get('id')} "
                      f"from={from_u.get('id')} ({from_u.get('username','-')})  text={txt[:60]!r}")

if __name__ == "__main__":
    asyncio.run(main())
