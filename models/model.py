"""OpenAI API client and model callers for Agent 7."""

import os
import re
import time
from pathlib import Path

from openai import OpenAI, RateLimitError


def _load_api_key() -> str:
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    key_file = Path(__file__).resolve().parent.parent / ".openai_key"
    raw = key_file.read_text().strip()
    return raw.removeprefix("OPENAI_API_KEY=").strip()


def _load_model_name() -> str:
    instruction_file = Path(__file__).resolve().parent.parent / "llm-api-instructions.md"
    text = instruction_file.read_text()
    match = re.search(r'"([^"]+)"', text)
    if not match:
        raise ValueError("Could not find a quoted model name in llm-api-instructions.md")
    return match.group(1)


MODEL = _load_model_name()
client = OpenAI(api_key=_load_api_key())


def _call_with_backoff(fn, *args, **kwargs):
    """Call ``fn`` with retries when OpenAI returns a rate-limit error."""
    max_retries = 6
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            wait = None
            if exc.response is not None:
                wait = exc.response.headers.get("retry-after")
            if wait is None and hasattr(exc, "retry_after") and exc.retry_after is not None:
                wait = exc.retry_after
            if wait is None:
                wait = 0.5 * (2 ** attempt)
            wait = max(float(wait), 0.05)
            print(f"[model] Rate limit hit (attempt {attempt + 1}/{max_retries}), sleeping {wait:.2f}s...")
            time.sleep(wait)
    raise RateLimitError("Exceeded retries for rate limit")


def call_model(messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
    """Call the configured OpenAI model with the given messages and tool definitions."""
    return _call_with_backoff(
        client.chat.completions.create,
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort="none",
    )


def call_text(messages: list[dict]) -> str:
    """Call the model without tool support for text-only agents."""
    response = _call_with_backoff(
        client.chat.completions.create,
        model=MODEL,
        messages=messages,
        reasoning_effort="none",
    )
    return response.choices[0].message.content or ""
