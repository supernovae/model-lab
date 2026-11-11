"""Native checkpoint generation scored by Inspect's optional Docker sandbox.

This is a raw-Python, one-sample-per-task adapter, not an agent benchmark. Tests
are withheld from the model prompt, but are visible to code in its container.
No generated code is executed on the host; missing dependencies fail closed.
"""

from __future__ import annotations

import importlib.metadata
import json
import secrets
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from sparselab.canonical import canonical_digest
from sparselab.config.models import StrictModel
from sparselab.documents import read_document
from sparselab.evaluation.generation_request import generate_result
from sparselab.evaluation.inference import load_run
from sparselab.lab_records import write_json_atomic
from sparselab.result_publication import sync_directory
from sparselab.training.attempt_budget import AttemptBudget
from sparselab.training.manifest import source_identity

INSPECT_VERSION = "0.3.287"
SHA_PATTERN = r"^[0-9a-f]{64}$"


class CodeWorkTask(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    family: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=100_000)
    tests: str = Field(min_length=1, max_length=100_000)
    tests_sha256: str = Field(pattern=SHA_PATTERN)

    @model_validator(mode="after")
    def verify_tests(self) -> CodeWorkTask:
        if canonical_digest(self.tests) != self.tests_sha256:
            raise ValueError("test source digest mismatch")
        return self


class CodeWorkDeclaration(StrictModel):
    version: Literal[1] = 1
    name: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=SHA_PATTERN)
    tokenizer_file_sha256: str = Field(pattern=SHA_PATTERN)
    image: str = Field(pattern=r"^[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}$")
    inspect_version: Literal["0.3.287"] = INSPECT_VERSION
    sequence_length: int = Field(ge=2, le=131072)
    max_new_tokens: int = Field(ge=1, le=8192)
    seed: int = Field(ge=0)
    timeout_seconds: int = Field(ge=1, le=120)
    tasks: tuple[CodeWorkTask, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def verify_tasks(self) -> CodeWorkDeclaration:
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task id")
        if self.max_new_tokens >= self.sequence_length:
            raise ValueError("completion must leave context for the prompt")
        return self


class ExecutorUnavailable(RuntimeError):
    """No verified local optional execution backend is available."""


def preflight_executor(declaration: CodeWorkDeclaration) -> dict[str, Any]:
    """Read local dependency/image identities only; never install, build or pull."""
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError as error:
        raise ExecutorUnavailable(
            f"optional inspect-ai=={INSPECT_VERSION} is absent"
        ) from error
    if version != declaration.inspect_version:
        raise ExecutorUnavailable("installed Inspect version differs from declaration")
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        image = subprocess.run(
            ["docker", "image", "inspect", declaration.image],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        info = json.loads(image.stdout)[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        raise ExecutorUnavailable(
            "Docker Compose or declared local image is unavailable; no pull attempted"
        ) from error
    if declaration.image not in info.get("RepoDigests", []):
        raise ExecutorUnavailable("local image has no matching repository digest")
    return {
        "inspect_version": version,
        "image": declaration.image,
        "image_id": info["Id"],
    }


async def execute_python_case(
    sandbox: Any, solution: str, tests: str, timeout: int
) -> dict[str, Any]:
    """Score inside a supplied Inspect SandboxEnvironment, never a host process.

    Timeout applies independently to syntax validation and test execution. A
    fixture sandbox proves adapter classification only, not process isolation.
    """
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "executor_error",
        "solution_sha256": canonical_digest(solution),
        "tests_sha256": canonical_digest(tests),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "output_truncated": False,
    }
    try:
        await sandbox.write_file("/tmp/solution.py", solution)
        await sandbox.write_file("/tmp/tests.py", tests)
        syntax = await sandbox.exec(
            [
                "python3",
                "-I",
                "-c",
                "import ast, sys\ntry:\n ast.parse(open('/tmp/solution.py').read())\nexcept SyntaxError as error:\n print(error, file=sys.stderr); sys.exit(65)",
            ],
            timeout=timeout,
            timeout_retry=False,
        )
        execution = syntax
        if syntax.returncode:
            result["status"] = (
                "syntax_error" if syntax.returncode == 65 else "executor_error"
            )
        else:
            marker = "SPARSELAB_TESTS_COMPLETED_" + secrets.token_hex(16)
            execution = await sandbox.exec(
                [
                    "python3",
                    "-I",
                    "-c",
                    (
                        "import sys, runpy; sys.path.insert(0, '/tmp'); "
                        "runpy.run_path('/tmp/tests.py', run_name='__main__'); "
                        f"print({marker!r})"
                    ),
                ],
                timeout=timeout,
                timeout_retry=False,
            )
            result["status"] = (
                "passed"
                if execution.returncode == 0 and marker in execution.stdout.splitlines()
                else "test_failed"
            )
        result.update(
            returncode=execution.returncode,
            stdout=execution.stdout[:8192],
            stderr=execution.stderr[:8192],
            output_truncated=len(execution.stdout) > 8192
            or len(execution.stderr) > 8192,
        )
    except TimeoutError:
        result["status"] = "timeout"
    except Exception as error:  # noqa: BLE001 -- preserve backend failure evidence
        result["reason"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - started
    return result


def score_with_inspect(
    declaration: CodeWorkDeclaration,
    solutions: dict[str, str],
    directory: Path,
    *,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Use public Inspect APIs; a no-op solver never invokes a model provider."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, accuracy, scorer
    from inspect_ai.solver import solver
    from inspect_ai.util import sandbox

    directory.mkdir(exist_ok=False)
    compose = directory / "compose.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "default": {
                        "image": declaration.image,
                        "x-local": True,
                        "pull_policy": "never",
                        "command": ["tail", "-f", "/dev/null"],
                        "init": True,
                        "network_mode": "none",
                        "read_only": True,
                        "tmpfs": ["/tmp:rw,nosuid,nodev,size=64m,mode=1777"],
                        "user": "65534:65534",
                        "cap_drop": ["ALL"],
                        "security_opt": ["no-new-privileges:true"],
                        "pids_limit": 64,
                        "mem_limit": "512m",
                        "cpus": 1.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    outcomes: dict[str, Any] = {}
    by_id = {task.id: task for task in declaration.tasks}

    @solver
    def native_completion():
        async def solve(state, generate):
            return state

        return solve

    @scorer(metrics=[accuracy()])
    def python_tests():
        async def score(state, target):
            task = by_id[str(state.sample_id)]
            outcome = await execute_python_case(
                sandbox(), solutions[task.id], task.tests, declaration.timeout_seconds
            )
            outcomes[task.id] = outcome
            if progress is not None:
                progress(task.id, outcome)
            return Score(
                value=1 if outcome["status"] == "passed" else 0, metadata=outcome
            )

        return score

    task = Task(
        dataset=[Sample(id=item.id, input=item.prompt) for item in declaration.tasks],
        solver=native_completion(),
        scorer=python_tests(),
        sandbox=("docker", str(compose.resolve())),
    )
    logs = inspect_eval(
        task,
        model="mockllm/model",
        sandbox_prebuilt=True,
        sandbox_cleanup=True,
        max_samples=1,
        epochs=1,
        retry_on_error=0,
        log_dir=str(directory / "logs"),
    )
    if not logs or any(log.status != "success" for log in logs):
        raise ExecutorUnavailable(
            "Inspect evaluation incomplete; inspect retained logs"
        )
    return outcomes


def run_code_work(
    run_id: str,
    runs_dir: Path,
    checkpoint: str | None,
    declaration: Path,
    output: Path,
    *,
    backend: str | None = None,
    authorization: Any = None,
) -> dict[str, Any]:
    """Generate once per declared task, preserve partial evidence, then sandbox.

    Output is exclusive; interrupted work cannot silently rerun the same path.
    Pass@1 includes every declared task, and is absent when execution is incomplete.
    """
    spec = CodeWorkDeclaration.model_validate(read_document(declaration))
    if output.resolve().is_relative_to((runs_dir / run_id).resolve()):
        raise ValueError("code-work output must be outside immutable run inputs")
    protocol = spec.model_dump(mode="json", exclude={"checkpoint_sha256"})
    result: dict[str, Any] = {
        "source": "code_work",
        "publication_id": secrets.token_hex(16),
        "evaluation_source_sha256": source_identity()["sha256"],
        "version": 1,
        "status": "incomplete",
        "protocol": protocol,
        "protocol_sha256": canonical_digest(protocol),
        "requested_checkpoint_sha256": spec.checkpoint_sha256,
        "identity": None,
        "executor": None,
        "tasks": {},
        "pass_at_1": None,
        "declared_tasks": len(spec.tasks),
        "evidence_complete": False,
    }
    publication_id = result["publication_id"]
    # The first immutable publication is also the exclusive invocation claim.
    # A later invocation must never overwrite an interrupted evaluation.
    write_json_atomic(output, result, exclusive=True)
    sync_directory(output.parent)

    def persist() -> None:
        write_json_atomic(
            output,
            result,
            exclusive=True,
            may_replace=lambda path: (
                read_document(path).get("publication_id") == publication_id
            ),
        )
        sync_directory(output.parent)

    try:
        result["executor"] = preflight_executor(spec)
        AttemptBudget.require_fixed_evaluation_allocation_from_environment(
            fixed_positions=0,
            total_positions=len(spec.tasks)
            * spec.sequence_length
            * spec.max_new_tokens,
            calls=len(spec.tasks),
            tokens=len(spec.tasks) * spec.max_new_tokens,
        )
        loaded = load_run(
            run_id,
            runs_dir,
            checkpoint,
            backend,
            authorization=authorization,
            expected_checkpoint_sha256=spec.checkpoint_sha256,
        )
        result["identity"] = loaded.identity
        if loaded.identity["tokenizer_sha256"] != spec.tokenizer_file_sha256:
            raise ValueError("code-work checkpoint tokenizer mismatch")
        if spec.sequence_length > loaded.config.model.max_seq_len:
            raise ValueError("code-work sequence length exceeds checkpoint context")
        solutions = {}
        for task in spec.tasks:
            result["tasks"][task.id] = {"status": "generating"}
            persist()
            generated = generate_result(
                loaded.model,
                loaded.tokenizer,
                task.prompt,
                spec.sequence_length,
                spec.max_new_tokens,
                loaded.device,
                temperature=0.0,
                seed=spec.seed,
                strict_context=True,
                engine=loaded.engine,
                accounting_label=f"code-work:{result['protocol_sha256']}:{task.id}",
            )
            solution = generated.text[len(task.prompt) :]
            solutions[task.id] = solution
            result["tasks"][task.id] = {
                "status": "generated",
                "solution": solution,
                "solution_sha256": canonical_digest(solution),
                "completion_tokens": generated.completion_tokens,
                "forward_input_positions": generated.forward_input_positions,
                "finish_reason": generated.finish_reason,
            }
            persist()

        def scored(task_id: str, outcome: dict[str, Any]) -> None:
            result["tasks"][task_id].update(outcome)
            persist()

        outcomes = score_with_inspect(
            spec,
            solutions,
            output.with_suffix(output.suffix + ".inspect"),
            progress=scored,
        )
        for task_id, outcome in outcomes.items():
            result["tasks"][task_id].update(outcome)
        if set(outcomes) != set(solutions) or any(
            outcome["status"] == "executor_error" for outcome in outcomes.values()
        ):
            raise ExecutorUnavailable("not all tasks have valid execution evidence")
        result.update(
            status="complete",
            evidence_complete=True,
            pass_at_1=sum(
                outcome["status"] == "passed" for outcome in outcomes.values()
            )
            / len(spec.tasks),
        )
    except ExecutorUnavailable as error:
        result.update(status="unavailable", reason=str(error))
    except Exception as error:  # noqa: BLE001 -- preserve backend failure evidence
        result["reason"] = f"{type(error).__name__}: {error}"
    finally:
        persist()
    return result
