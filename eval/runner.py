"""Benchmark runner for the coding agent.

Each benchmark is a directory under ``eval/benchmarks/`` containing a workspace
(repo with code and tests) and a ``task.txt`` prompt. The runner copies the
workspace to a fresh location, runs the agent, then runs ``pytest`` to verify.
"""

import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.orchestrator import run


BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _parse_tokens(text: str) -> int | None:
    for pattern in (
        r"Tokens used this turn: ([\d,]+) /",
        r"Tokens used: ([\d,]+) /",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _count_parent_steps(text: str) -> int:
    return len(re.findall(r"\[Parent\] Step \d+", text))


def _run_pytest(workspace: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _load_tasks() -> list[dict]:
    """Discover benchmarks that include a task.txt file."""
    tasks = []
    for workspace_dir in sorted(BENCHMARKS_DIR.iterdir()):
        if not workspace_dir.is_dir():
            continue
        task_file = workspace_dir / "task.txt"
        prompt = (
            task_file.read_text().strip()
            if task_file.exists()
            else "Fix the code so that all tests pass."
        )
        tasks.append(
            {
                "name": workspace_dir.name,
                "prompt": prompt,
                "workspace": workspace_dir.name,
            }
        )
    return tasks


def run_benchmark(task: dict) -> dict:
    """Run a single benchmark task and return the result."""
    source = BENCHMARKS_DIR / task["workspace"]
    workspace = RESULTS_DIR / task["name"] / "workspace"

    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(source, workspace)

    print(f"\n=== Running benchmark: {task['name']} ===")
    start = time.time()

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        try:
            agent_result = run(task["prompt"], str(workspace))
        except Exception as exc:
            agent_result = f"ERROR: {exc}"

    raw_output = captured.getvalue()
    elapsed = time.time() - start

    pytest_result = _run_pytest(workspace)
    tests_passed = pytest_result["returncode"] == 0

    result = {
        "name": task["name"],
        "prompt": task["prompt"],
        "passed": tests_passed,
        "agent_result": agent_result,
        "raw_output": raw_output,
        "tokens": _parse_tokens(agent_result + raw_output),
        "parent_steps": _count_parent_steps(raw_output),
        "duration_seconds": round(elapsed, 2),
        "pytest": pytest_result,
    }

    status = "PASS" if tests_passed else "FAIL"
    print(f"Result: {status}")
    print(f"Tokens: {result['tokens']}")
    print(f"Parent steps: {result['parent_steps']}")
    print(f"Duration: {result['duration_seconds']}s")
    if not tests_passed:
        print("Pytest stdout:", pytest_result["stdout"])
        print("Pytest stderr:", pytest_result["stderr"])

    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks()
    if not tasks:
        print("No benchmarks found in", BENCHMARKS_DIR)
        return

    results = []
    for task in tasks:
        results.append(run_benchmark(task))

    summary_path = RESULTS_DIR / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {status} {r['name']}: "
            f"{r['tokens']} tokens, {r['parent_steps']} steps, "
            f"{r['duration_seconds']}s"
        )
    print(f"Detailed results written to {summary_path}")


if __name__ == "__main__":
    main()
