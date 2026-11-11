"""Adapter fixtures only: these tests do not establish sandbox isolation."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from klab.canonical import canonical_digest
from klab.evaluation import code_work


@pytest.fixture
def declaration(tmp_path):
    tests = "from solution import add\nassert add(2, 3) == 5\n"
    value = {
        "name": "synthetic-add",
        "checkpoint_sha256": "a" * 64,
        "tokenizer_file_sha256": "b" * 64,
        "image": "python@sha256:" + "c" * 64,
        "sequence_length": 128,
        "max_new_tokens": 32,
        "seed": 7,
        "timeout_seconds": 2,
        "tasks": [
            {
                "id": "add",
                "family": "synthetic",
                "source_revision": "fixture-v1",
                "license": "MIT",
                "prompt": "Return raw Python defining add(a, b).",
                "tests": tests,
                "tests_sha256": canonical_digest(tests),
            }
        ],
    }
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(value))
    return path, value


class FixtureSandbox:
    """Scripted outcomes, never executes source."""

    def __init__(self, results):
        self.results = iter(results)
        self.files = {}
        self.commands = []

    async def write_file(self, path, contents):
        self.files[path] = contents

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        outcome = next(self.results)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            returncode=outcome, stdout="KLAB_TESTS_COMPLETED_fixture\n", stderr=""
        )


@pytest.mark.parametrize(
    ("responses", "status"),
    [
        ([0, 0], "passed"),
        ([0, 1], "test_failed"),
        ([65], "syntax_error"),
        ([127], "executor_error"),
        ([0, TimeoutError()], "timeout"),
        ([RuntimeError("fixture failure")], "executor_error"),
    ],
)
def test_sandbox_classification_and_provenance(responses, status, monkeypatch):
    monkeypatch.setattr(code_work.secrets, "token_hex", lambda _: "fixture")
    sandbox = FixtureSandbox(responses)
    result = asyncio.run(code_work.execute_python_case(sandbox, "answer", "tests", 2))
    assert result["status"] == status
    assert result["solution_sha256"] == canonical_digest("answer")
    assert result["tests_sha256"] == canonical_digest("tests")
    assert sandbox.files == {"/tmp/solution.py": "answer", "/tmp/tests.py": "tests"}
    for index, (command, options) in enumerate(sandbox.commands):
        expected = {"timeout": 2, "timeout_retry": False}
        if index == 1:
            expected["input"] = "KLAB_TESTS_COMPLETED_fixture\n"
        assert options == expected
        assert "KLAB_TESTS_COMPLETED_fixture" not in " ".join(command)


def test_tampered_tests_refused(declaration):
    _, value = declaration
    value["tasks"][0]["tests"] += "assert False"
    with pytest.raises(ValueError, match="digest mismatch"):
        code_work.CodeWorkDeclaration.model_validate(value)


def test_missing_executor_preserves_refusal_before_model_load(
    tmp_path, declaration, monkeypatch
):
    path, _ = declaration

    def absent(*args):
        raise code_work.ExecutorUnavailable("fixture: missing optional executor")

    monkeypatch.setattr(code_work, "preflight_executor", absent)
    monkeypatch.setattr(
        code_work,
        "load_run",
        lambda *a, **kw: pytest.fail("loaded model without executor"),
    )
    output = tmp_path / "result.json"
    result = code_work.run_code_work("run", tmp_path / "runs", None, path, output)
    assert result["status"] == "unavailable"
    assert result["pass_at_1"] is None
    assert not result["evidence_complete"]
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        code_work.run_code_work("run", tmp_path / "runs", None, path, output)


def test_native_generation_binding_and_complete_denominator(
    tmp_path, declaration, monkeypatch
):
    path, value = declaration
    native = SimpleNamespace(
        identity={"tokenizer_sha256": "b" * 64},
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=128)),
        model=object(),
        tokenizer=object(),
        device="cpu",
        engine=None,
    )
    monkeypatch.setattr(code_work, "preflight_executor", lambda _: {"fixture": True})

    def load(*args, **kwargs):
        assert kwargs["expected_checkpoint_sha256"] == "a" * 64
        return native

    monkeypatch.setattr(code_work, "load_run", load)

    def generate(*args, **kwargs):
        assert args[0] is native.model
        assert args[2] == value["tasks"][0]["prompt"]
        assert "assert" not in args[2]
        assert kwargs["strict_context"] and kwargs["seed"] == 7
        return SimpleNamespace(
            text=args[2] + "def add(a,b): return a+b",
            completion_tokens=12,
            forward_input_positions=33,
            finish_reason="stop",
        )

    monkeypatch.setattr(code_work, "generate_result", generate)

    def score(spec, solutions, directory, *, progress=None):
        assert solutions == {"add": "def add(a,b): return a+b"}
        return {"add": {"status": "test_failed"}}

    monkeypatch.setattr(code_work, "score_with_inspect", score)
    result = code_work.run_code_work(
        "run", tmp_path / "runs", None, path, tmp_path / "result.json"
    )
    assert result["status"] == "complete"
    assert result["pass_at_1"] == 0
    assert result["tasks"]["add"]["completion_tokens"] == 12


def test_actual_missing_inspect_preflight(declaration, monkeypatch):
    _, value = declaration

    def missing(_):
        raise code_work.importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr(code_work.importlib.metadata, "version", missing)
    monkeypatch.setattr(
        code_work.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("unexpected subprocess"),
    )
    with pytest.raises(code_work.ExecutorUnavailable, match="inspect-ai.*absent"):
        code_work.preflight_executor(
            code_work.CodeWorkDeclaration.model_validate(value)
        )


def test_missing_image_never_pulls(declaration, monkeypatch):
    _, value = declaration
    monkeypatch.setattr(
        code_work.importlib.metadata, "version", lambda _: code_work.INSPECT_VERSION
    )
    calls = []

    def local_only(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["image", "inspect"]:
            raise code_work.subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(code_work.subprocess, "run", local_only)
    with pytest.raises(code_work.ExecutorUnavailable, match="no pull attempted"):
        code_work.preflight_executor(
            code_work.CodeWorkDeclaration.model_validate(value)
        )
    assert calls == [
        ["docker", "compose", "version"],
        ["docker", "image", "inspect", value["image"]],
    ]


def test_scoring_failure_keeps_generated_and_partial_results(
    tmp_path, declaration, monkeypatch
):
    path, value = declaration
    monkeypatch.setattr(code_work, "preflight_executor", lambda _: {"fixture": True})
    native = SimpleNamespace(
        identity={"tokenizer_sha256": "b" * 64},
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=128)),
        model=object(),
        tokenizer=object(),
        device="cpu",
        engine=None,
    )
    monkeypatch.setattr(code_work, "load_run", lambda *a, **kw: native)
    monkeypatch.setattr(
        code_work,
        "generate_result",
        lambda *a, **kw: SimpleNamespace(
            text=value["tasks"][0]["prompt"] + "answer",
            completion_tokens=1,
            forward_input_positions=2,
            finish_reason="length",
        ),
    )

    def score(spec, solutions, directory, *, progress):
        progress("add", {"status": "test_failed"})
        raise code_work.ExecutorUnavailable("fixture cleanup failure")

    monkeypatch.setattr(code_work, "score_with_inspect", score)
    output = tmp_path / "result.json"
    result = code_work.run_code_work("run", tmp_path / "runs", None, path, output)
    assert result["status"] == "unavailable" and result["pass_at_1"] is None
    assert result["tasks"]["add"]["solution"] == "answer"
    assert result["tasks"]["add"]["status"] == "test_failed"
    assert json.loads(output.read_text()) == result


def test_early_successful_exit_without_test_completion_is_failure():
    class EarlyExitSandbox(FixtureSandbox):
        async def exec(self, command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = asyncio.run(
        code_work.execute_python_case(EarlyExitSandbox([]), "source", "tests", 2)
    )
    assert result["status"] == "test_failed"


@pytest.fixture
def native_ready(tmp_path, declaration, monkeypatch):
    path, value = declaration
    monkeypatch.setattr(code_work, "source_identity", lambda: {"sha256": "d" * 64})
    monkeypatch.setattr(code_work, "preflight_executor", lambda _: {"fixture": True})
    native = SimpleNamespace(
        identity={"tokenizer_sha256": "b" * 64, "source_identity_sha256": "e" * 64},
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=128)),
        model=object(),
        tokenizer=object(),
        device="cpu",
        engine=None,
    )
    monkeypatch.setattr(code_work, "load_run", lambda *a, **kw: native)
    monkeypatch.setattr(
        code_work,
        "generate_result",
        lambda *a, **kw: SimpleNamespace(
            text=value["tasks"][0]["prompt"] + "answer",
            completion_tokens=1,
            forward_input_positions=2,
            finish_reason="length",
        ),
    )
    return path, tmp_path / "result.json"


def test_interrupted_atomic_replace_preserves_generated_answer(
    native_ready, tmp_path, monkeypatch
):
    from klab import lab_records

    path, output = native_ready

    def interrupted_replace(*args):
        raise OSError("fixture interruption before rename")

    def score(*args, progress):
        monkeypatch.setattr(lab_records.os, "replace", interrupted_replace)
        progress("add", {"status": "passed"})

    monkeypatch.setattr(code_work, "score_with_inspect", score)
    with pytest.raises(OSError, match="before rename"):
        code_work.run_code_work("run", tmp_path / "runs", None, path, output)
    retained = json.loads(output.read_text())
    assert retained["tasks"]["add"]["status"] == "generated"
    assert retained["tasks"]["add"]["solution"] == "answer"
    assert retained["status"] == "incomplete"
    assert retained["pass_at_1"] is None


def test_progress_cannot_replace_another_invocation(
    native_ready, tmp_path, monkeypatch
):
    path, output = native_ready
    foreign = {"publication_id": "another-invocation", "status": "incomplete"}

    def score(*args, progress):
        output.write_text(json.dumps(foreign))
        return {"add": {"status": "passed"}}

    monkeypatch.setattr(code_work, "score_with_inspect", score)
    with pytest.raises(FileExistsError):
        code_work.run_code_work("run", tmp_path / "runs", None, path, output)
    assert json.loads(output.read_text()) == foreign


def test_evaluator_source_is_separate_from_training_source(
    native_ready, tmp_path, monkeypatch
):
    path, output = native_ready
    monkeypatch.setattr(
        code_work, "score_with_inspect", lambda *a, **kw: {"add": {"status": "passed"}}
    )
    result = code_work.run_code_work("run", tmp_path / "runs", None, path, output)
    assert result["evaluation_source_sha256"] == "d" * 64
    assert result["identity"]["source_identity_sha256"] == "e" * 64
    assert result["evidence_complete"]


def test_argv_echo_cannot_forge_completion_marker(monkeypatch):
    monkeypatch.setattr(code_work.secrets, "token_hex", lambda _: "fixture")

    class ArgvEchoSandbox(FixtureSandbox):
        async def exec(self, command, **kwargs):
            # Model the reported attack: read argv and exit zero without tests.
            return SimpleNamespace(returncode=0, stdout="\n".join(command), stderr="")

    result = asyncio.run(
        code_work.execute_python_case(ArgvEchoSandbox([]), "source", "tests", 2)
    )
    assert result["status"] == "test_failed"


def test_trusted_wrapper_reads_marker_before_tests(monkeypatch, capsys):
    import io
    import runpy
    import sys

    monkeypatch.setattr(code_work.secrets, "token_hex", lambda _: "fixture")
    sandbox = FixtureSandbox([0, 0])
    asyncio.run(code_work.execute_python_case(sandbox, "source", "tests", 2))
    command, options = sandbox.commands[1]
    seen = []
    monkeypatch.setattr(sys, "stdin", io.StringIO(options["input"]))
    monkeypatch.setattr(sys, "path", list(sys.path))

    def fake_tests(path, run_name):
        assert sys.stdin.read() == ""
        seen.append((path, run_name))

    monkeypatch.setattr(runpy, "run_path", fake_tests)
    # Execute only the trusted wrapper, with test/solution execution stubbed out.
    exec(compile(command[-1], "<trusted-code-work-wrapper>", "exec"), {})  # noqa: S102
    assert seen == [("/tmp/tests.py", "__main__")]
    assert capsys.readouterr().out == options["input"]
