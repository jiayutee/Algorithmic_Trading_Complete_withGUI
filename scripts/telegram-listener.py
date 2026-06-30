#!/usr/bin/env python3
"""
Telegram → Orchestrator bridge.
Polls for incoming messages from the owner, passes them to Claude orchestrator,
sends the response back. Only responds to ALLOWED_CHAT_ID for security.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
ENV_FILE = PROJECT_DIR / ".env"

def load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

load_env()

BOT_TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_CHAT   = int(os.environ.get("TELEGRAM_CHAT_ID", "51218456"))
OFFSET_FILE    = PROJECT_DIR / "logs" / ".telegram_offset"
LOG_FILE       = PROJECT_DIR / "logs" / "telegram-listener.log"
POLL_INTERVAL  = 3   # seconds between getUpdates calls
CLAUDE_TIMEOUT = 300 # max seconds to wait for Claude response

API = f"https://api.telegram.org/bot{BOT_TOKEN}"

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def send(chat_id: int, text: str):
    """Send a Telegram message, splitting if over 4096 chars."""
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            requests.post(f"{API}/sendMessage", json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
            }, timeout=10)
        except Exception as e:
            log(f"Send error: {e}")

def send_typing(chat_id: int):
    try:
        requests.post(f"{API}/sendChatAction",
                      json={"chat_id": chat_id, "action": "typing"}, timeout=5)
    except Exception:
        pass

def get_updates(offset: int) -> list:
    try:
        r = requests.get(f"{API}/getUpdates",
                         params={"offset": offset, "timeout": 20, "limit": 10},
                         timeout=25)
        return r.json().get("result", [])
    except Exception as e:
        log(f"getUpdates error: {e}")
        return []

def run_orchestrator(user_message: str) -> str:
    """Pass the user message to Claude orchestrator and return its response."""
    prompt = (
        f"Message from owner via Telegram: \"{user_message}\"\n\n"
        "Respond helpfully and concisely. If it's a status question, check the "
        "Sprint Board and Notion Daily Log. If it's an instruction, acknowledge "
        "it, act on it, and confirm what you did. Keep your Telegram reply under "
        "500 words — the user is reading on a phone."
    )
    try:
        claude_bin = os.environ.get("CLAUDE_BIN", "/Users/jiayutee/.local/bin/claude")
        result = subprocess.run(
            [claude_bin,
             "--append-system-prompt-file",
             str(PROJECT_DIR / ".github/agents/orchestrator.agent.md"),
             "--print",
             "--allowedTools", "Bash,Read,Edit,Write,Agent,WebSearch,WebFetch",
             "--max-turns", "30", "-p", prompt],
            capture_output=True, text=True,
            timeout=CLAUDE_TIMEOUT,
            cwd=str(PROJECT_DIR),
        )
        output = result.stdout.strip() or result.stderr.strip()
        # Extract last assistant text block if JSON output
        try:
            data = json.loads(output)
            msgs = data.get("messages", [])
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content if b.get("type") == "text"
                        )
                    return content.strip()
        except Exception:
            pass
        return output[:3000] if output else "✅ Done — no output returned."
    except subprocess.TimeoutExpired:
        return "⏱ Orchestrator timed out. Check logs for details."
    except Exception as e:
        return f"❌ Error running orchestrator: {e}"

def load_offset() -> int:
    try:
        return int(OFFSET_FILE.read_text().strip())
    except Exception:
        return 0

def save_offset(offset: int):
    OFFSET_FILE.write_text(str(offset))

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("Telegram listener started. Waiting for messages...")
    send(ALLOWED_CHAT, "🤖 Orchestrator is online. Send me a message to get started.")

    offset = load_offset()

    while True:
        updates = get_updates(offset)

        for update in updates:
            offset = update["update_id"] + 1
            save_offset(offset)

            msg = update.get("message") or update.get("edited_message")
            if not msg:
                continue

            chat_id   = msg["chat"]["id"]
            text      = msg.get("text", "").strip()
            username  = msg.get("from", {}).get("first_name", "unknown")

            # Security: only respond to the owner
            if chat_id != ALLOWED_CHAT:
                log(f"Ignored message from unauthorised chat_id={chat_id}")
                continue

            if not text:
                continue

            log(f"Message from {username}: {text[:100]}")
            send_typing(chat_id)
            send(chat_id, "🔄 _On it..._")

            response = run_orchestrator(text)
            send(chat_id, response)
            log(f"Response sent ({len(response)} chars)")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Listener stopped.")
