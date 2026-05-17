#!/usr/bin/env python3
"""PostToolUse hook — ast.parse the file that was just edited if it's
Python. Catches syntax errors instantly instead of letting them surface
30 seconds later when the server restarts.

Reads the hook event JSON from stdin, looks up the edited file path,
runs `ast.parse` on it. Silent on success. On failure, prints the
SyntaxError to stderr and exits 2 so the message is shown to Claude
(who then knows to fix the file before doing anything else).

Triggered by the matcher in .claude/settings.json — see PostToolUse
entry there.
"""
import json
import sys
import ast


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Hook fired with no/bad payload — not our problem.
        return 0

    tool_input = event.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""

    # MultiEdit and NotebookEdit can come through too; they also expose
    # file_path. Anything not Python is a no-op.
    if not file_path.endswith(".py"):
        return 0

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except (FileNotFoundError, PermissionError, OSError) as e:
        # File vanished or unreadable after the edit — surface but don't
        # block aggressively; the model can decide what to do.
        print(f"check-py-syntax: couldn't read {file_path}: {e}", file=sys.stderr)
        return 1

    try:
        ast.parse(source, filename=file_path)
    except SyntaxError as e:
        # exit 2 = block; the model sees the stderr message and knows
        # to fix the file before doing anything else.
        print(
            f"SYNTAX ERROR introduced in {file_path}\n"
            f"  Line {e.lineno}, col {e.offset}: {e.msg}\n"
            f"  Fix this before continuing — the file is currently unparseable.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
