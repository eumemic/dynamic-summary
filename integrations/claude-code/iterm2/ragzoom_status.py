#!/usr/bin/env python3
"""iTerm2 status bar component for ragzoom memory status.
Reads per-PID status from /tmp/ragzoom-status/{pid}
"""

import iterm2
import os
import subprocess

STATUS_DIR = "/tmp/ragzoom-status"

async def main(connection):
    component = iterm2.StatusBarComponent(
        short_description="Ragzoom",
        detailed_description="Shows ragzoom memory status for Claude Code in this terminal",
        knobs=[],
        exemplar="memory: synced",
        update_cadence=2,
        identifier="com.ragzoom.status"
    )

    @iterm2.StatusBarRPC
    async def ragzoom_status_coro(knobs, session_id=iterm2.Reference("id")):
        try:
            # Get the session's TTY
            app = await iterm2.async_get_app(connection)
            session = app.get_session_by_id(session_id)
            if not session:
                return ""

            tty = await session.async_get_variable("tty")
            if not tty:
                return ""

            tty_name = tty.split("/")[-1]
            if not tty_name:
                return ""

            # Find Claude process on this TTY
            result = subprocess.run(
                ["ps", "-eo", "pid,tty,comm"],
                capture_output=True, text=True, timeout=2
            )

            claude_pid = None
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3 and "claude" in parts[2].lower():
                    pid, tty_col = parts[0], parts[1]
                    if tty_name == tty_col:
                        claude_pid = pid
                        break

            if not claude_pid:
                return ""

            # Read status from per-PID file
            status_file = os.path.join(STATUS_DIR, claude_pid)
            if os.path.exists(status_file):
                with open(status_file) as f:
                    return f.read().strip()

            return ""

        except Exception:
            return ""

    await component.async_register(connection, ragzoom_status_coro)

iterm2.run_forever(main)
