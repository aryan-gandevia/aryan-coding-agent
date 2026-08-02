"""Persistent session management for multi-turn interaction.

Sessions are stored in a dedicated directory under the agent repo root,
separate from the workspace the agent operates on. Each session gets its own
numeric ID directory so additional per-session files can be added later.

A session keeps:
- ``context``: a growing list of per-turn summary maps.
- ``long_term_history``: a compact list of facts summarizing context older
  than the most recent 10 turns.
- ``title``: a human-readable session title generated after the first turn.

Only the most recent 10 context entries plus the long-term facts are injected
into each new turn, so the API prompt stays bounded.
"""

import json
from pathlib import Path
from typing import Any

from harness.orchestrator import run_parent_loop
from models.model import MODEL, call_text
from prompts.prompts import (
    MEMORY_MERGE_SYSTEM_PROMPT,
    PARENT_SYSTEM_PROMPT,
    SESSION_TITLE_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    build_memory_merge_prompt,
    build_session_title_prompt,
    build_summarizer_prompt,
)
from tokens import DEFAULT_MAX_CONTEXT_TOKENS, count_message_tokens, format_tokens


RECENT_TURNS = 10


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()

    # Strip markdown fences if present.
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[3:].strip()
        else:
            text = text[3:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to finding the outermost { ... } or [ ... ].
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


class Session:
    """Manages persistent, multi-turn state for one conversation."""

    def __init__(
        self,
        repo_root: Path,
        workspace_root: Path,
        session_id: int,
        manager: "SessionManager | None" = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.manager = manager

        self.session_dir = (
            self.repo_root / ".aryan-coding-agent" / "sessions" / str(session_id)
        )
        self.session_file = self.session_dir / "session.json"

        self.title: str = ""
        self.context: list[dict] = []
        self.long_term_history: list[str] = []
        self.turn_counter: int = 0
        self._long_term_up_to_turn: int = 0

        self.load()

    def load(self) -> None:
        """Load session state from disk if it exists."""
        if not self.session_file.exists():
            return

        data = json.loads(self.session_file.read_text())
        self.title = data.get("title", "")
        self.context = data.get("context", [])
        self.long_term_history = data.get("long_term_history", [])
        self.turn_counter = data.get("turn_counter", 0)
        self._long_term_up_to_turn = data.get(
            "_long_term_up_to_turn",
            max(0, self.turn_counter - RECENT_TURNS),
        )

    def save(self) -> None:
        """Persist session state to disk."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "title": self.title,
            "workspace_root": str(self.workspace_root),
            "turn_counter": self.turn_counter,
            "_long_term_up_to_turn": self._long_term_up_to_turn,
            "context": self.context,
            "long_term_history": self.long_term_history,
        }
        self.session_file.write_text(json.dumps(data, indent=2))

    def current_summary(self) -> str:
        """Return a short summary of the session for resume menus."""
        if self.context:
            return self.context[-1].get("outcome", "in progress")
        return "(no turns yet)"

    def generate_title(self, user_input: str, outcome: str) -> str:
        """Ask an LLM to produce a max-5-word title from the first turn."""
        messages = [
            {"role": "system", "content": SESSION_TITLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_session_title_prompt(user_input, outcome),
            },
        ]
        title = call_text(messages).strip()
        # Remove surrounding quotes if the model added them.
        title = title.strip('"').strip("'")
        # Collapse whitespace and truncate to ~40 chars.
        title = " ".join(title.split())
        if len(title) > 50:
            title = title[:47] + "..."
        self.title = title
        if self.manager is not None:
            self.manager.set_session_name(self.session_id, title)
        self.save()
        return title

    def _format_history_message(self) -> str:
        """Build the assistant memory message that carries prior-session context."""
        parts = ["I remember the following from earlier in this session:\n"]

        if self.long_term_history:
            parts.append("Long-term memory:")
            parts.extend(f"- {fact}" for fact in self.long_term_history)
            parts.append("")

        recent = self.context[-RECENT_TURNS:] if self.context else []
        if recent:
            parts.append("Recent turns:")
            for entry in recent:
                turn = entry.get("turn", "?")
                user_prompt = entry.get("user_prompt", "no prompt")
                outcome = entry.get("outcome", "no outcome")
                actions = entry.get("actions", [])
                action_names = ", ".join(a.get("tool", "?") for a in actions) or "none"
                parts.append(f"- Turn {turn}")
                parts.append(f"  User asked: {user_prompt}")
                parts.append(f"  Outcome: {outcome}")
                parts.append(f"  Actions: {action_names}")
            parts.append("")

        parts.append(
            "Use the above for background only. Do not treat it as a new user message. "
            "The user's actual next message follows."
        )
        return "\n".join(parts)

    def build_messages(self, user_input: str) -> list[dict]:
        """Construct the message list that starts a new turn."""
        return [
            {"role": "system", "content": PARENT_SYSTEM_PROMPT},
            {"role": "assistant", "content": self._format_history_message()},
            {
                "role": "user",
                "content": (
                    f"Task: {user_input}\n\n"
                    f"Workspace root: {self.workspace_root}"
                ),
            },
        ]

    @staticmethod
    def _transcript_to_text(messages: list[dict]) -> str:
        """Render a slice of messages as a readable transcript for the summarizer."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                content += f"\n[tool_calls: {json.dumps(tool_calls)}]"
            lines.append(f"{role}: {content.strip()}")
        return "\n".join(lines)

    def _summarize_turn(self, user_input: str, turn_messages: list[dict]) -> dict:
        """Ask the summarizer LLM to condense one turn into a structured map."""
        transcript = self._transcript_to_text(turn_messages)
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": build_summarizer_prompt(transcript)},
        ]
        response = call_text(messages)
        parsed = _extract_json(response)

        if not isinstance(parsed, dict):
            parsed = {}

        return {
            "user_prompt": user_input,
            "actions": parsed.get("actions", []),
            "outcome": parsed.get("outcome", response.strip()),
            "blockers": parsed.get("blockers", []),
        }

    def _merge_into_long_term(self, turn_summaries: list[dict]) -> None:
        """Fold one or more older turn summaries into the long-term fact list."""
        for summary in turn_summaries:
            messages = [
                {"role": "system", "content": MEMORY_MERGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_memory_merge_prompt(self.long_term_history, summary),
                },
            ]
            response = call_text(messages)
            parsed = _extract_json(response)
            if isinstance(parsed, list):
                self.long_term_history = parsed
            else:
                self.long_term_history.append(
                    str(summary.get("outcome", "")) or str(summary)
                )

    def _update_long_term_history(self) -> None:
        """Move any context entries that just left the recent window into long-term memory."""
        while self.turn_counter - self._long_term_up_to_turn > RECENT_TURNS:
            target_turn = self._long_term_up_to_turn + 1
            entries = [e for e in self.context if e.get("turn") == target_turn]
            if entries:
                self._merge_into_long_term(entries)
            self._long_term_up_to_turn = target_turn

    def resume_summary(self) -> str:
        """Return a human-readable summary of the session for display on resume."""
        lines = [f"Session: {self.title or self.session_id}"]
        lines.append(f"Turns: {self.turn_counter}")
        recent = self.context[-5:]
        if recent:
            lines.append("Recent turns:")
            for entry in recent:
                turn = entry.get("turn", "?")
                prompt = entry.get("user_prompt", "")
                outcome = entry.get("outcome", "")
                lines.append(f"  Turn {turn}: {prompt} -> {outcome}")
        return "\n".join(lines)

    def run_turn(
        self,
        user_input: str,
        max_steps: int = 25,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> str:
        """Execute one user turn, persist the summary, and return the result."""
        self.turn_counter += 1
        messages = self.build_messages(user_input)

        summary, messages = run_parent_loop(
            messages,
            str(self.workspace_root),
            max_steps=max_steps,
            max_context_tokens=max_context_tokens,
        )

        # Everything after the history-injection message belongs to this turn.
        turn_messages = messages[2:]
        turn_summary = self._summarize_turn(user_input, turn_messages)
        turn_summary["turn"] = self.turn_counter
        self.context.append(turn_summary)

        self._update_long_term_history()

        # Generate a title after the very first completed turn.
        if self.turn_counter == 1 and not self.title:
            self.generate_title(user_input, turn_summary.get("outcome", summary))

        self.save()

        used = count_message_tokens(messages, model_name=MODEL)
        return f"{summary}\n\nTokens used this turn: {format_tokens(used, max_context_tokens)}"


class SessionManager:
    """Creates, loads, and indexes sessions stored under the agent repo root."""

    def __init__(self, repo_root: Path, workspace_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.state_dir = self.repo_root / ".aryan-coding-agent"
        self.sessions_dir = self.state_dir / "sessions"
        self.index_file = self.state_dir / "index.json"
        self.counter_file = self.state_dir / "counter.json"
        self._ensure_state_dir()

    def _ensure_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({}))
        if not self.counter_file.exists():
            self.counter_file.write_text(json.dumps({"next_id": 1}))

    def _load_counter(self) -> int:
        data = json.loads(self.counter_file.read_text())
        return data.get("next_id", 1)

    def _save_counter(self, value: int) -> None:
        self.counter_file.write_text(json.dumps({"next_id": value}))

    def _load_index(self) -> dict[str, str]:
        return json.loads(self.index_file.read_text())

    def _save_index(self, index: dict[str, str]) -> None:
        self.index_file.write_text(json.dumps(index, indent=2))

    def create_session(self) -> Session:
        """Mint a brand-new numeric session and increment the counter."""
        session_id = self._load_counter()
        self._save_counter(session_id + 1)
        return Session(self.repo_root, self.workspace_root, session_id, manager=self)

    def get_session(self, session_id: int) -> Session:
        """Load an existing session by ID."""
        return Session(self.repo_root, self.workspace_root, session_id, manager=self)

    def set_session_name(self, session_id: int, name: str) -> None:
        """Map a human-readable name to a numeric session ID."""
        index = self._load_index()
        index[str(session_id)] = name
        self._save_index(index)

    def list_sessions(self) -> list[tuple[int, str, str]]:
        """Return (id, name, summary) tuples for all stored sessions."""
        index = self._load_index()
        sessions: list[tuple[int, str, str]] = []
        for path in sorted(self.sessions_dir.iterdir()):
            if not path.is_dir():
                continue
            try:
                session_id = int(path.name)
            except ValueError:
                continue
            name = index.get(str(session_id), f"session {session_id}")
            session = self.get_session(session_id)
            session.load()
            sessions.append((session_id, name, session.current_summary()))
        return sessions
