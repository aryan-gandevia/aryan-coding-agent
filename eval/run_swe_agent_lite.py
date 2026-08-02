"""Run the agent against the public swe-agent-lite benchmark.

The benchmark repo is cloned on demand into ``eval/.cache/swe-agent-lite``,
copied into a per-task workspace under ``eval/results/``, run, and scored with
pytest. The cache is kept across runs for speed; workspaces can be inspected or
removed afterwards.
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


EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache" / "swe-agent-lite"
RESULTS_DIR = EVAL_DIR / "results"
REPO_URL = "https://github.com/dhruvpatel1706/swe-agent-lite.git"


def _clone_benchmark() -> None:
    if CACHE_DIR.exists():
        return
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CACHE_DIR)],
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_task_yaml(path: Path) -> dict:
    """Parse just enough of the task.yaml to extract id/title/problem/difficulty."""
    text = path.read_text()
    data: dict[str, str | list[str]] = {}
    for line in text.splitlines():
        if ":" in line and not line.strip().startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"')
            data[key] = value

    # Multi-line `problem: |` block
    problem_lines: list[str] = []
    in_problem = False
    for line in text.splitlines():
        if line.startswith("problem:"):
            in_problem = True
            continue
        if in_problem:
            if line and not line.startswith("  "):
                break
            problem_lines.append(line)
    data["problem"] = "\n".join(problem_lines).strip()
    return data


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
    tasks_dir = CACHE_DIR / "tasks"
    tasks = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.exists():
            continue
        info = _parse_task_yaml(yaml_path)
        tasks.append(
            {
                "id": info.get("id", task_dir.name),
                "title": info.get("title", task_dir.name),
                "difficulty": info.get("difficulty", "unknown"),
                "problem": info.get("problem", "Fix the code so all tests pass."),
                "source": task_dir,
            }
        )
    return tasks


def run_task(task: dict) -> dict:
    task_result_dir = RESULTS_DIR / f"swe-lite-{task['id']}"
    workspace = task_result_dir / "workspace"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # Copy repo and tests into the workspace so pytest can discover them.
    shutil.copytree(task["source"] / "repo", workspace / "repo")
    shutil.copytree(task["source"] / "tests", workspace / "tests")

    prompt = (
        f"{task['problem']}\n\nThe code is in repo/solution.py and the tests "
        "are in tests/test_solution.py. Fix the code so all tests pass."
    )

    print(f"\n=== swe-agent-lite: {task['id']} ({task['difficulty']}) ===")
    print(task["title"])

    start = time.time()
    captured = io.StringIO()
    stderr_cap = io.StringIO()
    with (
        contextlib.redirect_stdout(captured),
        contextlib.redirect_stderr(stderr_cap),
    ):
        try:
            agent_result = run(prompt, str(workspace))
        except Exception as exc:
            agent_result = f"ERROR: {exc}"
        stderr_text = stderr_cap.getvalue()

    raw_output = captured.getvalue() + stderr_text
    elapsed = time.time() - start

    pytest_result = _run_pytest(workspace)
    tests_passed = pytest_result["returncode"] == 0

    result = {
        "id": task["id"],
        "title": task["title"],
        "difficulty": task["difficulty"],
        "prompt": prompt,
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

    # Clean up the per-task workspace so evals do not bloat the repo.
    shutil.rmtree(task_result_dir, ignore_errors=True)

    return result


def main() -> None:
    _clone_benchmark()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks()
    if not tasks:
        print("No tasks found in", CACHE_DIR / "tasks")
        return

    results = []
    for task in tasks:
        results.append(run_task(task))

    summary_path = RESULTS_DIR / f"swe_agent_lite_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {status} {r['id']} ({r['difficulty']}): "
            f"{r['tokens']} tokens, {r['parent_steps']} steps, "
            f"{r['duration_seconds']}s"
        )
    print(f"Detailed results written to {summary_path}")


if __name__ == "__main__":
    main()
