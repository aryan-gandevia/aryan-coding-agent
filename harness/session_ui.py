"""Small terminal UI for selecting a session to resume.

Uses raw-terminal input and ANSI escape sequences so the user can navigate with
arrow keys without taking over the whole screen. Falls back to a plain
numbered prompt if stdin is not a TTY.
"""

import os
import re
import sys


def _terminal_width() -> int:
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 80


def _format_option(
    session: tuple[int, str, str], selected: bool, width: int
) -> str:
    """Return one menu row with a marker; selected row is highlighted in bold."""
    session_id, name, _summary = session
    marker = ">" if selected else " "
    plain = f"{marker} [{session_id}] {name}"
    if len(plain) > width:
        plain = plain[: max(0, width - 3)] + "..."
    if selected:
        return f"\x1b[1m{plain}\x1b[0m"
    return plain


def _print_options(
    sessions: list[tuple[int, str, str]], selected: int, width: int
) -> None:
    """Print the option rows."""
    for i, session in enumerate(sessions):
        sys.stdout.write(_format_option(session, i == selected, width) + "\n")
    sys.stdout.flush()


def _read_raw_key(fd: int) -> bytes:
    """Read the next key or escape sequence from the terminal as raw bytes."""
    import select

    ch = os.read(fd, 1)
    if ch != b"\x1b":
        return ch

    # Arrow keys send ESC followed by '[' or 'O' and a final letter.
    if select.select([fd], [], [], 0.2)[0]:
        b1 = os.read(fd, 1)
        if b1 in (b"[", b"O") and select.select([fd], [], [], 0.2)[0]:
            b2 = os.read(fd, 1)
            return b"\x1b" + b1 + b2
        return b"\x1b" + b1
    return ch


def _inline_select(sessions: list[tuple[int, str, str]]) -> int | None:
    """Inline arrow-key selection that stays anchored in one place."""
    import select
    import termios
    import tty

    selected = 0
    width = _terminal_width()
    fd = sys.stdin.fileno()

    print("Select a session to resume (↑/↓, Enter to select, q to cancel):")
    _print_options(sessions, selected, width)

    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)

    def _redraw() -> None:
        # Move cursor back to the first option row, then overwrite each row in
        # place. A carriage return ensures every line starts at column 0.
        sys.stdout.write(f"\x1b[{len(sessions)}A")
        for i, session in enumerate(sessions):
            sys.stdout.write("\r\x1b[K")
            sys.stdout.write(_format_option(session, i == selected, width))
            sys.stdout.write("\n")
        sys.stdout.flush()

    try:
        while True:
            if not select.select([fd], [], [], None)[0]:
                continue
            key = _read_raw_key(fd)

            # Enter (carriage return in raw mode)
            if key in (b"\n", b"\r"):
                return sessions[selected][0]

            # Ctrl-C or 'q' cancels
            if key == b"\x03" or key == b"q":
                return None

            # Lone ESC cancels
            if key == b"\x1b":
                return None

            # Arrow up
            if key in (b"\x1b[A", b"\x1bOA"):
                selected = (selected - 1) % len(sessions)
                _redraw()
                continue

            # Arrow down
            if key in (b"\x1b[B", b"\x1bOB"):
                selected = (selected + 1) % len(sessions)
                _redraw()
                continue
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Move to a clean line after the menu.
        print()


def _fallback_select(sessions: list[tuple[int, str, str]]) -> int | None:
    """Plain numbered-list selection for non-TTY stdin."""
    print("Available sessions:")
    for i, (session_id, name, _summary) in enumerate(sessions, 1):
        print(f"  {i}. [{session_id}] {name}")
    while True:
        try:
            choice = input("Enter number (or q to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx][0]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def select_session_to_resume(
    sessions: list[tuple[int, str, str]],
) -> int | None:
    """Return the selected session ID, or None if the user cancels."""
    if not sessions:
        print("No saved sessions to resume.")
        return None

    if sys.stdin.isatty():
        return _inline_select(sessions)

    return _fallback_select(sessions)
