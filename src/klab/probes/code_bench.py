"""``klab bench``: execution-scored code benchmarks (HumanEval, MBPP, task files).

A task is data: a pinned dataset file (Hugging Face repository and commit, or
a local file) with its SHA-256, a prompt template, a program template that
joins the prompt, the model's completion and the tests, stop sequences and a
completion budget. Each problem gets one greedy completion (pass@1, n=1); the
assembled program runs once in the :mod:`klab.probes.code_sandbox` sandbox and
passes when it exits 0 within its limits.

The sealed record (``LAB/benchmarks/<id>/bench.json``) keeps every problem's
outcome, the decoding and sandbox settings and a ``benchmark_group``: two
pass@1 values compare only when the task, its data and items, the decoding
and the sandbox all match. ``klab compare`` reads it like any other result;
reference models (``ref:NAME``) are text-level, so they share the group.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.lab_records import KINDS, comparison_group, read_lab_record, write_sealed
from klab.probes import code_sandbox
from klab.training.manifest import canonical_digest, canonical_json, sha256_file

BENCH_FORMAT = KINDS["bench"][0]
# Bump when how a program is assembled or judged changes.
SCORING_PROTOCOL = "klab-code-exec-v1"
DECODING = {"strategy": "greedy", "temperature": 0.0, "samples": 1}


@dataclass(frozen=True)
class CodeTask:
    """One execution-scored benchmark, declared as data."""

    name: str
    # Hugging Face dataset repository; None: ``data_file`` is a local path.
    repo_id: str | None
    revision: str | None
    data_file: str
    data_sha256: str
    id_field: str
    # ``str.format`` over a problem's fields (a list field ``x`` is also
    # offered joined by newlines as ``x_lines``); ``program`` adds
    # ``{prompt}`` (the rendered prompt) and ``{completion}``.
    prompt: str
    program: str
    stop: tuple[str, ...]
    max_new_tokens: int
    source: str

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), "stop": list(self.stop)}


# Stop sequences and prompt formats of bigcode-evaluation-harness; HumanEval's
# prompt is used exactly as published (Chen et al. 2021).
TASKS: Mapping[str, CodeTask] = {
    task.name: task
    for task in (
        CodeTask(
            name="humaneval",
            repo_id="openai/openai_humaneval",
            revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
            data_file="openai_humaneval/test-00000-of-00001.parquet",
            data_sha256="2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
            id_field="task_id",
            prompt="{prompt}",
            program="{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n",
            stop=("\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"),
            max_new_tokens=512,
            source="HumanEval (Chen et al. 2021), 164 problems, original tests",
        ),
        CodeTask(
            name="mbpp",
            repo_id="google-research-datasets/mbpp",
            revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
            data_file="full/test-00000-of-00001.parquet",
            data_sha256="566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77",
            id_field="task_id",
            prompt='"""\n{text}\n{test_list[0]}\n"""\n',
            program="{prompt}{completion}\n\n{test_setup_code}\n{test_list_lines}\n",
            stop=("\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n```"),
            max_new_tokens=512,
            source="MBPP (Austin et al. 2021) full test split, 500 problems, "
            "zero-shot docstring prompt of bigcode-evaluation-harness",
        ),
    )
}


def resolve_task(spec: str) -> CodeTask:
    """A named task, or a JSON/YAML task declaration (local data relative to it)."""
    if spec in TASKS:
        return TASKS[spec]
    path = Path(spec)
    if not path.is_file():
        raise ValueError(
            f"unknown code task {spec!r}: name one of {', '.join(TASKS)} or a "
            "task declaration file"
        )
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = {f.name for f in fields(CodeTask)}
    if not isinstance(document, dict) or set(document) != names:
        raise ValueError(f"a code task declares exactly: {', '.join(sorted(names))}")
    document["stop"] = tuple(document["stop"])
    if document["repo_id"] is None:
        local = Path(document["data_file"])
        document["data_file"] = str(
            (local if local.is_absolute() else path.parent / local).resolve()
        )
    task = CodeTask(**document)
    if task.repo_id is not None and (
        not isinstance(task.revision, str) or len(task.revision) != 40
    ):
        raise ValueError("a Hub task pins a 40-character commit revision")
    if task.max_new_tokens < 1 or not task.stop or not all(task.stop):
        raise ValueError("a code task needs a positive budget and non-empty stops")
    return task


def load_problems(task: CodeTask, cache_dir: Path | None = None) -> list[dict]:
    """The task's problems from its pinned file, refused on a digest mismatch."""
    if task.repo_id is None:
        path = Path(task.data_file)
    else:
        from huggingface_hub import hf_hub_download

        from klab.hf_auth import hub_access_errors, hub_auth_kwargs, raise_for_hub_auth

        auth = hub_auth_kwargs()
        try:
            path = Path(
                hf_hub_download(
                    task.repo_id,
                    task.data_file,
                    repo_type="dataset",
                    revision=task.revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    **auth,
                )
            )
        except hub_access_errors() as error:
            raise_for_hub_auth(error, credential_supplied=bool(auth))
    digest = sha256_file(path)
    if digest != task.data_sha256:
        raise ValueError(
            f"{task.name} data digest {digest} differs from the pinned "
            f"{task.data_sha256}"
        )
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    else:
        raise ValueError("code task data must be .parquet or .jsonl")
    ids = [row.get(task.id_field) for row in rows]
    if None in ids or len(set(map(str, ids))) != len(ids):
        raise ValueError(f"{task.name}: every problem needs a unique {task.id_field}")
    return rows


def _fields(problem: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(problem)
    for key, value in problem.items():
        if isinstance(value, list):
            out[f"{key}_lines"] = "\n".join(map(str, value))
    return out


def render_prompt(task: CodeTask, problem: Mapping[str, Any]) -> str:
    return task.prompt.format(**_fields(problem))


def render_program(
    task: CodeTask, problem: Mapping[str, Any], prompt: str, completion: str
) -> str:
    return task.program.format(
        **{**_fields(problem), "prompt": prompt, "completion": completion}
    )


def truncate_at_stop(text: str, stops: Sequence[str]) -> str:
    cut = min((text.find(s) for s in stops if s in text), default=len(text))
    return text[:cut]


# --- Completions --------------------------------------------------------------


def complete(loaded: Any, prompt: str, task: CodeTask) -> tuple[str, int]:
    """One greedy completion (stop sequences removed) and its token count."""
    if getattr(loaded, "reference", None) is not None:
        return _reference_completion(loaded, prompt, task)
    from klab.evaluation.generation import generate_with_token_ids

    text, ids = generate_with_token_ids(
        loaded.model,
        loaded.tokenizer,
        prompt,
        loaded.config.model.max_seq_len,
        task.max_new_tokens,
        loaded.device,
        temperature=0.0,
        stop_sequences=task.stop,
        engine=loaded.engine,
    )
    return truncate_at_stop(text[len(prompt) :], task.stop), len(ids)


def _reference_completion(loaded: Any, prompt: str, task: CodeTask) -> tuple[str, int]:
    """Greedy decoding of a pinned reference with its own KV cache."""
    import torch

    hf = loaded.model.hf
    tokenizer = loaded.tokenizer
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    generated: list[int] = []
    with torch.inference_mode():
        out = hf(input_ids=torch.tensor([ids], device=loaded.device), use_cache=True)
        for _ in range(task.max_new_tokens):
            token = int(out.logits[0, -1].argmax())
            if token == loaded.eot_token_id:
                break
            generated.append(token)
            # Stops are matched on the decoded tail only (cheap and exact).
            tail = tokenizer.decode(generated[-16:], skip_special_tokens=True)
            if any(s in tail for s in task.stop):
                break
            out = hf(
                input_ids=torch.tensor([[token]], device=loaded.device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return truncate_at_stop(text, task.stop), len(generated)


def load_bench_target(
    spec: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None,
    backend: str | None,
    authorization: Any,
) -> Any:
    from klab.reference_models import is_reference, load_reference

    if is_reference(spec):
        return load_reference(spec, device=backend or "cpu")
    from klab.probes.cli import load_target

    return load_target(
        spec,
        lab_dir=lab_dir,
        runs_dir=runs_dir,
        backend=backend,
        authorization=authorization,
    )


# --- Running a benchmark ------------------------------------------------------


def benchmark_identity(
    task: CodeTask,
    problems: Sequence[Mapping[str, Any]],
    limits: code_sandbox.SandboxLimits,
    mechanism: str,
) -> dict[str, Any]:
    """Everything that must match for two pass@1 values to compare."""
    items = [
        {
            "id": str(problem[task.id_field]),
            "prompt": render_prompt(task, problem),
            "program": render_program(task, problem, "{prompt}", "{completion}"),
        }
        for problem in problems
    ]
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "task": task.identity() | {"data_file": Path(task.data_file).name},
        "items": len(items),
        "items_sha256": canonical_digest(items),
        "decoding": {**DECODING, "max_new_tokens": task.max_new_tokens},
        "sandbox": {
            "protocol": code_sandbox.SANDBOX_PROTOCOL,
            "mechanism": mechanism,
            "python": ".".join(map(str, sys.version_info[:2])),
            "limits": asdict(limits),
        },
    }


def run_bench(
    target_spec: str,
    task_spec: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None = None,
    backend: str | None = None,
    authorization: Any = None,
    limit: int | None = None,
    limits: code_sandbox.SandboxLimits | None = None,
    allow_unsandboxed: bool = False,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Generate, execute and publish ``LAB/benchmarks/<id>/bench.json`` (sealed).

    The sandbox is checked before anything is loaded: without isolation (and
    without ``allow_unsandboxed``) nothing runs. Touch the printed CANCEL file
    or press Ctrl-C to stop; the partial record is published ``incomplete``.
    """
    from klab.lab_context import LabCancelled, LabContext, LabSignal
    from klab.probes.runner import arm_identity

    limits = limits or code_sandbox.SandboxLimits()
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    task = resolve_task(task_spec)
    mechanism = code_sandbox.require_sandbox(allow_unsandboxed=allow_unsandboxed)
    problems = load_problems(task, cache_dir)[:limit]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bench_id = f"bench-{stamp}-{secrets.token_hex(4)}"
    folder = lab_dir / KINDS["bench"][1] / bench_id
    folder.mkdir(parents=True, exist_ok=False)
    context = LabContext(folder / "CANCEL", workspace=lab_dir)
    say = progress or (lambda line: print(line, file=sys.stderr))
    say(f"bench {bench_id}: touch {context.cancel_path} (or Ctrl-C) to stop")
    benchmark = benchmark_identity(task, problems, limits, mechanism)
    record: dict[str, Any] = {
        "format": BENCH_FORMAT,
        "bench_id": bench_id,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {"target": target_spec, "task": task_spec, "limit": limit},
        "task": task.identity(),
        "decoding": benchmark["decoding"],
        "sandbox": code_sandbox.describe(mechanism, limits),
        "benchmark": benchmark,
        "target": None,
        "problems": [],
    }
    rows: list[dict[str, Any]] = record["problems"]
    completions: list[dict[str, str]] = []
    timing = {"generation_seconds": 0.0, "execution_seconds": 0.0, "tokens": 0}
    stop: dict[str, Any] = {"kind": None, "reason": None}
    context.install()
    try:
        try:
            loaded = load_bench_target(
                target_spec,
                lab_dir=lab_dir,
                runs_dir=runs_dir,
                backend=backend,
                authorization=authorization,
            )
            record["target"] = arm_identity(loaded)
            record["target"]["context_length"] = loaded.config.model.max_seq_len
            loaded.model.eval()
            for index, problem in enumerate(problems):
                context.checkpoint()
                prompt = render_prompt(task, problem)
                tick = time.monotonic()
                completion, tokens = complete(loaded, prompt, task)
                timing["generation_seconds"] += time.monotonic() - tick
                timing["tokens"] += tokens
                outcome = code_sandbox.run_program(
                    render_program(task, problem, prompt, completion),
                    limits,
                    mechanism=mechanism,
                )
                timing["execution_seconds"] += outcome["seconds"]
                problem_id = str(problem[task.id_field])
                completions.append({"id": problem_id, "completion": completion})
                rows.append(
                    {
                        "id": problem_id,
                        **outcome,
                        "completion_tokens": tokens,
                        "completion_sha256": hashlib.sha256(
                            completion.encode()
                        ).hexdigest(),
                    }
                )
                passed = sum(r["status"] == "passed" for r in rows)
                say(
                    f"  [{index + 1}/{len(problems)}] {problem_id}: "
                    f"{outcome['status']} (pass@1 so far {passed}/{len(rows)})"
                )
        except LabCancelled as error:
            stop.update(kind=error.kind, reason=error.reason)
        except (LabSignal, KeyboardInterrupt) as error:
            name = error.name if isinstance(error, LabSignal) else "keyboard_interrupt"
            stop.update(kind="interrupted", reason=f"interrupted: {name}")
        except code_sandbox.SandboxError as error:
            stop.update(kind="sandbox_error", reason=str(error))
        context.finalizing = True
        complete_run = stop["kind"] is None and len(rows) == len(problems)
        passed = sum(r["status"] == "passed" for r in rows)
        statuses = {
            s: sum(r["status"] == s for r in rows)
            for s in ("passed", "failed", "timeout", "error")
        }
        (folder / "completions.jsonl").write_bytes(
            b"".join(canonical_json(row) + b"\n" for row in completions)
        )
        record.update(
            status="complete" if complete_run else "incomplete",
            stop=stop,
            completions_sha256=sha256_file(folder / "completions.jsonl"),
            timing={k: round(v, 3) for k, v in timing.items()},
            code_pass1={
                "value": passed / len(rows) if complete_run else None,
                "passed": passed,
                "scored": len(rows),
                "total": len(problems),
                "statuses": statuses,
                # Only a complete benchmark has a group (partial is missing
                # evidence, never a low score).
                "benchmark_group": comparison_group(benchmark)
                if complete_run
                else None,
            },
        )
        path = folder / KINDS["bench"][2]
        sealed = write_sealed(path, record)
    finally:
        context.restore()
    if stop["kind"] == "interrupted":
        raise KeyboardInterrupt
    return sealed, path


def read_bench(path: Path) -> dict[str, Any]:
    return read_lab_record(path, "bench")[1]


def render(record: Mapping[str, Any]) -> str:
    """Plain-text report of one sealed benchmark record."""
    target = record.get("target") or {}
    task = record["task"]
    code = record.get("code_pass1") or {}
    sandbox = record["sandbox"]
    who = (target.get("reference") or {}).get("name") or (
        f"{target.get('run_id')}@{target.get('step')}"
    )
    value = code.get("value")
    statuses = " · ".join(f"{k} {v}" for k, v in (code.get("statuses") or {}).items())
    limits = sandbox["limits"]
    lines = [
        (
            f"CODE BENCH {record['bench_id']}  {task['name']} · {who} · "
            f"{record['status'].upper()}"
        ),
        f"  pass@1  {value:.4f}" if value is not None else "  pass@1  –",
        f"  problems  {code.get('scored')}/{code.get('total')} scored · {statuses}",
        (
            f"  data  {task['repo_id'] or 'local'}@{task['revision'] or '-'} "
            f"{task['data_file']} sha256 {task['data_sha256'][:12]}"
        ),
        (
            f"  decoding  greedy, n=1, max {record['decoding']['max_new_tokens']} "
            f"new tokens, stops {json.dumps(task['stop'])}"
        ),
        (
            f"  sandbox  {sandbox['mechanism']} (network {sandbox['network']}) · "
            f"timeout {limits['timeout_seconds']} s · "
            f"memory {limits['memory_bytes'] // 1024**2} MiB"
        ),
        f"  group  {(code.get('benchmark_group') or 'none (incomplete)')[:16]}",
    ]
    timing = record.get("timing") or {}
    if timing:
        lines.append(
            f"  time  generation {timing['generation_seconds']} s "
            f"({timing['tokens']} tokens) · execution {timing['execution_seconds']} s"
        )
    stop = record.get("stop") or {}
    if stop.get("kind"):
        lines.append(f"  stopped  {stop['kind']}: {stop['reason']}")
    return "\n".join(lines)
