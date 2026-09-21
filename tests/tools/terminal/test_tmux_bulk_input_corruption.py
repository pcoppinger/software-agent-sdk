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


# ── integration: readline disabled at pane creation ─────────────────


def _tab_heredoc_command(path: str) -> str:
    # Tab-indented Go-style body, 12 lines: under _MULTILINE_THRESHOLD so it
    # travels the burst path. With readline active, the TABs trigger filename
    # completion and splice a directory listing into the heredoc body.
    body = "".join(f"\tif x == {i} {{\n\t\treturn {i * 2}\n\t}}\n" for i in range(4))
    return f"cat > {path} <<'EOF'\n{body}EOF"


def _tab_expected_content() -> bytes:
    body = "".join(f"\tif x == {i} {{\n\t\treturn {i * 2}\n\t}}\n" for i in range(4))
    return body.encode()


def test_readline_disabled_in_real_pane() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        term = TmuxTerminal(work_dir=tmpdir)
        term.initialize()
        try:
            out = f"{tmpdir}/emacs.txt"
            term.send_keys(f"set -o | awk '/^emacs/ {{print $2}}' > {out}")
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    if open(out).read().strip() == "off":
                        return
                except FileNotFoundError:
                    pass
                time.sleep(0.2)
            pytest.fail(f"readline still active in pane: {open(out).read()!r}")
        finally:
            term.close()


@pytest.mark.parametrize("attempt", range(3))
def test_tab_indented_heredoc_lands_byte_exact(attempt: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = f"{tmpdir}/tabbed.go"
        term = TmuxTerminal(work_dir=tmpdir)
        term.initialize()
        try:
            term.send_keys(_tab_heredoc_command(out))
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    if open(out, "rb").read() == _tab_expected_content():
                        return
                except FileNotFoundError:
                    pass
                time.sleep(0.2)
            data = open(out, "rb").read() if os.path.exists(out) else b""
            pytest.fail(
                f"tab-indented heredoc corrupted in pane (attempt {attempt}): "
                f"got {data[:200]!r}"
            )
        finally:
            term.close()


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
