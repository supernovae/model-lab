"""Execution-scored code benchmarks: sandboxed scoring and comparable records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from klab.probes import code_bench, code_sandbox

try:
    code_sandbox.require_sandbox()
    SANDBOX = None
except code_sandbox.SandboxUnavailable as error:
    SANDBOX = str(error).split(";")[0]
needs_sandbox = pytest.mark.skipif(
    SANDBOX is not None, reason=f"no isolating sandbox on this host: {SANDBOX}"
)
PROBLEMS = [
    {
        "task_id": "t/add",
        "prompt": "def add(a, b):\n",
        "test": "def check(f):\n    assert f(2, 3) == 5\n",
        "entry_point": "add",
    },
    {
        "task_id": "t/even",
        "prompt": "def is_even(n):\n",
        "test": "def check(f):\n    assert f(4) and not f(3)\n",
        "entry_point": "is_even",
    },
    {
        "task_id": "t/rev",
        "prompt": "def rev(s):\n",
        "test": "def check(f):\n    assert f('ab') == 'ba'\n",
        "entry_point": "rev",
    },
]
SOLUTIONS = {
    "good": {
        "t/add": "    return a + b\n",
        "t/even": "    return n % 2 == 0\n",
        "t/rev": "    return s\n",
    },
    "weak": {
        "t/add": "    return a - b\n",
        "t/even": "    while True:\n        pass\n",
        "t/rev": "    return s[::-1]\n",
    },
}


def _task(root: Path) -> Path:
    data = root / "problems.jsonl"
    data.write_text("".join(json.dumps(p) + "\n" for p in PROBLEMS))
    task = {
        "name": "tiny-exec",
        "repo_id": None,
        "revision": None,
        "data_file": data.name,
        "data_sha256": code_bench.sha256_file(data),
        "id_field": "task_id",
        "prompt": "{prompt}",
        "program": "{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n",
        "stop": ["\ndef"],
        "max_new_tokens": 32,
        "source": "test fixture",
    }
    path = root / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    return path


def _scripted(name: str, sha: str) -> SimpleNamespace:
    return SimpleNamespace(
        run=Path("/runs") / name,
        model=torch.nn.Linear(2, 2),
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=64)),
        identity={"run_id": name, "checkpoint_sha256": sha, "step": 1},
    )


@needs_sandbox
def test_bench_scores_by_execution_and_compares_within_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    task = _task(tmp_path)
    lab = tmp_path / "lab"
    monkeypatch.setattr(
        code_bench,
        "complete",
        lambda loaded, prompt, task: (
            SOLUTIONS[loaded.identity["run_id"]][
                next(p["task_id"] for p in PROBLEMS if p["prompt"] == prompt)
            ],
            3,
        ),
    )
    records = {}
    for name, sha in (("good", "a" * 64), ("weak", "b" * 64)):
        monkeypatch.setattr(
            code_bench,
            "load_bench_target",
            lambda *_, n=name, s=sha, **__: _scripted(n, s),
        )
        records[name], _ = code_bench.run_bench(
            name,
            str(task),
            lab_dir=lab,
            limits=code_sandbox.SandboxLimits(timeout_seconds=2, cpu_seconds=2),
            progress=lambda _: None,
        )
    statuses = {
        name: [p["status"] for p in record["problems"]]
        for name, record in records.items()
    }
    assert statuses == {
        "good": ["passed", "passed", "failed"],
        "weak": ["failed", "timeout", "passed"],
    }
    assert records["good"]["code_pass1"]["value"] == pytest.approx(2 / 3)
    group = records["good"]["code_pass1"]["benchmark_group"]
    assert group and records["weak"]["code_pass1"]["benchmark_group"] == group

    from klab.cli.main import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "compare",
            records["good"]["bench_id"],
            records["weak"]["bench_id"],
            "--lab-dir",
            str(lab),
            "--json",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    pair = next(
        p for p in report["comparisons"][0]["pairs"] if p["metric"] == "code_pass1"
    )
    assert pair["group"] == group
    assert pair["delta"] == pytest.approx(1 / 3)
    assert pair["status"] in {"higher", "within_noise"}


@needs_sandbox
def test_sandbox_contains_network_outside_writes_and_runaway_programs(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "escaped.txt"
    probe = (
        "import socket, sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "    sys.exit('network reachable')\n"
        "except OSError:\n"
        "    pass\n"
        "try:\n"
        f"    open({str(outside)!r}, 'w').write('x')\n"
        "    sys.exit('wrote outside its directory')\n"
        "except OSError:\n"
        "    pass\n"
        "open('inside.txt', 'w').write('ok')\n"
    )
    limits = code_sandbox.SandboxLimits(timeout_seconds=2, cpu_seconds=1)
    contained = code_sandbox.run_program(probe, limits)
    assert contained["status"] == "passed", contained["stderr_tail"]
    assert not outside.exists()
    forked = code_sandbox.run_program(
        "import os, time\n"
        "while True:\n"
        "    if os.fork() == 0:\n"
        "        time.sleep(30)\n",
        limits,
    )
    assert forked["status"] in {"failed", "timeout"}
    assert code_sandbox.run_program("while True: pass\n", limits)["status"] == "timeout"
