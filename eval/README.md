# Agent Evaluation Harness

This directory contains lightweight benchmarks to measure the coding agent's
quality on concrete tasks.

## Running

```bash
python3 eval/runner.py
```

The runner copies each benchmark workspace to `eval/results/<name>/workspace`,
runs the agent on the task prompt, then runs `pytest -q` to verify the result.

## Benchmarks

- **buggy_add**: a simple `add` function with an off-by-one bug.
- **greeter**: a greeting function that returns the wrong string.
- **number_utils**: a small utility module with multiple broken functions.

## Future tasks to add

- Feature addition (e.g., add `multiply` to an existing calculator)
- Multi-turn follow-up (resume a session and extend the previous work)
- Long-context memory (many turns, then ask about an earlier turn)
- Ambiguous request (agent should ask for clarification)
- Error recovery (bad edit, failing tests, agent should iterate)
