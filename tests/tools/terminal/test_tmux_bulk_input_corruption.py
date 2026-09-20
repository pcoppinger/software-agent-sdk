"""Regression tests for bulk multi-line input corruption in TmuxTerminal.

A single large `send-keys -l` burst into a bash pane with readline active
duplicates/splices chunks of the input (readline redraw races the incoming
bytes). TmuxTerminal.send_keys must send long multi-line payloads
line-by-line with pacing. Reproduced raw-tmux: 3/3 corrupt at 80x24 and
5/5 corrupt at 256x200 with a 26-line heredoc; 5/5 clean line-by-line.
"""

import os
import platform
import tempfile
import time

import pytest


if platform.system() == "Windows":
    pytest.skip("tmux backend is not available on Windows", allow_module_level=True)

from unittest.mock import MagicMock

import libtmux

from openhands.tools.terminal.terminal.tmux_terminal import TmuxTerminal


LINES = 60


def heredoc_command(path: str) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    body = "".join(f"line-{i + 1:02d}-{letters[i % 26] * 72}\n" for i in range(LINES))
    return f"cat > {path} <<'EOF'\n{body}EOF"


def expected_content() -> bytes:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(
        f"line-{i + 1:02d}-{letters[i % 26] * 72}\n" for i in range(LINES)
    ).encode()


# ── unit: chunking dispatch ─────────────────────────────────────────


def test_multiline_text_is_sent_line_by_line() -> None:
    term = TmuxTerminal(work_dir="/tmp")
    term._initialized = True
    pane = MagicMock(spec=libtmux.Pane)
    term.pane = pane

    text = heredoc_command("/tmp/whatever")
    term.send_keys(text, enter=True)

    # "cat > …" + LINES body lines + "EOF", then a final Enter
    sent_lines = text.split("\n")
    calls = pane.send_keys.call_args_list
    assert len(calls) == len(sent_lines) + 1
    for i, call in enumerate(calls[:-1]):
        args, kwargs = call
        assert kwargs.get("literal") is True
        assert kwargs.get("enter") is False
        assert args[0] == sent_lines[i] + "\n"
    assert calls[-1].args[0] == "Enter"


def test_short_text_still_sent_in_one_burst() -> None:
    term = TmuxTerminal(work_dir="/tmp")
    term._initialized = True
    pane = MagicMock(spec=libtmux.Pane)
    term.pane = pane

    term.send_keys("echo one\necho two", enter=True)

    assert pane.send_keys.call_count == 2
    assert pane.send_keys.call_args_list[0].args[0] == "echo one\necho two"


# ── integration: byte-exact landing through a real pane ─────────────


@pytest.mark.parametrize("attempt", range(3))
def test_bulk_heredoc_lands_byte_exact(attempt: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = f"{tmpdir}/bulk.txt"
        term = TmuxTerminal(work_dir=tmpdir)
        term.initialize()
        try:
            term.send_keys(heredoc_command(out))
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    data = open(out, "rb").read()
                    if data == expected_content():
                        return
                except FileNotFoundError:
                    pass
                time.sleep(0.2)
            data = open(out, "rb").read() if os.path.exists(out) else b""
            lines = [ln for ln in data.split(b"\n") if ln]
            bad = [(i, len(ln)) for i, ln in enumerate(lines) if len(ln) != 80]
            pytest.fail(
                f"bulk heredoc corrupted in pane (attempt {attempt}): "
                f"lines={len(lines)} bad={bad}"
            )
        finally:
            term.close()
