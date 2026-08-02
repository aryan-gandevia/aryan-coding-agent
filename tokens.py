"""Token-counting helpers for context-window management.

Uses ``tiktoken`` with a conservative fallback encoder so that unknown model
names still produce a reasonable estimate.
"""

import tiktoken

DEFAULT_MAX_CONTEXT_TOKENS = 100_000
WARNING_THRESHOLD = 0.8


def get_encoder(model_name: str | None = None) -> tiktoken.Encoding:
    """Return a tiktoken encoder for ``model_name`` or a safe fallback."""
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except (KeyError, ValueError):
            pass
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[dict], model_name: str | None = None) -> int:
    """Approximate the number of tokens consumed by a list of chat messages.

    This is a practical estimate: it tokenizes every string field that is sent
    to the model (role, content, tool names/arguments, tool results) and adds a
    small per-message overhead.
    """
    encoder = get_encoder(model_name)
    total = 0

    for msg in messages:
        total += 3  # per-message overhead (role / delimiter tokens)

        for key, value in msg.items():
            if not value:
                continue

            if key in {"content", "role", "name"} and isinstance(value, str):
                total += len(encoder.encode(value))
            elif key == "tool_calls" and isinstance(value, list):
                for tool_call in value:
                    if not isinstance(tool_call, dict):
                        total += len(encoder.encode(str(tool_call)))
                        continue
                    fn = tool_call.get("function") or {}
                    for fn_key in ("name", "arguments"):
                        fn_value = fn.get(fn_key)
                        if isinstance(fn_value, str):
                            total += len(encoder.encode(fn_value))

    total += 3  # every reply is primed with assistant delimiter tokens
    return total


def format_tokens(used: int, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> str:
    """Return a human-readable ``used / max`` token string."""
    return f"{used:,} / {max_tokens:,}"


def usage_fraction(used: int, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> float:
    """Return the fraction of the context window currently used."""
    if max_tokens <= 0:
        return 0.0
    return used / max_tokens
