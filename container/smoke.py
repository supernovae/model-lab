"""Small direct-container CLI/web/WAL check; no models, downloads or training."""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path


def smoke(engine, image, output, revision):
    output.mkdir(parents=True, exist_ok=False)
    names = [
        "klab-smoke-" + uuid.uuid4().hex[:10] + suffix
        for suffix in (
            "-cli",
            "-web",
            "-writer",
            "-help",
            "-bare",
            "-invalid",
            "-compare",
            "-eval",
        )
    ]
    volumes = [names[0] + "-data", names[1] + "-data"]

    def call(*args, check=True, timeout=60):
        return subprocess.run(
            [engine, *args],
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )

    details = json.loads(call("image", "inspect", image).stdout)[0]
    assert details["Config"]["Labels"]["org.opencontainers.image.revision"] == revision
    (output / "image.json").write_text(json.dumps(details, indent=2))

    def start(index, command=()):
        args = [
            "run",
            "-d",
            "--name",
            names[index],
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--network=none",
            "--memory=4g",
            "--cpus=2",
            "--pids-limit=256",
            "--tmpfs=/tmp:size=64m",
            "-v",
            volumes[1 if index == 1 else 0] + ":/state",
        ]
        if index == 1:
            args += [
                "-v",
                volumes[0] + ":/observe:ro",
                "--health-cmd",
                "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2).read()\"",
                "--health-interval=1s",
                "--health-timeout=3s",
                "--health-start-period=5s",
            ]
        call(*args, image, *command)

    def wait(index, expected="0"):
        assert call("wait", names[index]).stdout.strip() == expected, call(
            "logs", names[index], check=False
        ).stdout

    try:
        for volume in volumes:
            call("volume", "create", volume)
        start(
            0,
            [
                "python",
                "-c",
                """import json,os,subprocess
from pathlib import Path
import klab,torch
assert klab.__file__.startswith('/opt/venv/')
assert os.getuid()==10001
assert torch.version.cuda is None and torch.version.hip is None
assert (torch.tensor([1,2])+1).tolist()==[2,3]
subprocess.run(['klab','--help'],check=True)
Path('/state/work/runs').mkdir()
print(json.dumps({'module':klab.__file__,'torch':torch.__version__,'uid':os.getuid()}))
""",
            ],
        )
        wait(0)
        for index, command, expected in (
            (3, (), "0"),
            (4, ["report", "--help"], "0"),
            (5, ["not-a-lab-command"], "2"),
            (6, ["compare", "--help"], "0"),
            (7, ["eval", "--help"], "0"),
        ):
            start(index, command)
            wait(index, expected)
            if index in (6, 7):
                # ImageMagick compare and the shell eval builtin must not win.
                assert f"usage: klab {command[0]}" in call("logs", names[index]).stdout
            state = json.loads(call("inspect", names[index]).stdout)[0]["State"]
            assert not state.get("Health"), state
        start(
            1,
            [
                "dashboard",
                "--address",
                "0.0.0.0",
                "--runs-dir",
                "/observe/work/runs",
                "--lab-dir",
                "/observe/work/lab",
                "--reports-dir",
                "/observe/work/reports",
                "--evidence-root",
                "/observe/evidence",
            ],
        )
        for _ in range(60):
            response = call(
                "exec",
                names[1],
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=1).read()",
                check=False,
                timeout=5,
            )
            state = json.loads(call("inspect", names[1]).stdout)[0]["State"]
            if (
                response.returncode == 0
                and state.get("Health", {}).get("Status") == "healthy"
            ):
                break
            time.sleep(0.5)
        else:
            raise AssertionError("web health timeout")
        start(
            2,
            [
                "python",
                "-c",
                """import sqlite3,time
c=sqlite3.connect('/state/work/runs/experiments.sqlite3')
c.execute('pragma journal_mode=WAL');c.execute('create table fixture(n integer)');c.commit()
for n in range(40):
 c.execute('insert into fixture values (?)',(n,));c.commit();time.sleep(.1)
""",
            ],
        )
        counts = []
        query = """from pathlib import Path
from klab.dashboard.queries import _connect
with _connect(Path('/observe/work/runs')) as c:
 print(c.execute('select count(*) from fixture').fetchone()[0])
"""
        for _ in range(20):
            value = call(
                "exec", names[1], "python", "-c", query, check=False, timeout=5
            )
            if value.returncode == 0:
                counts.append(int(value.stdout.strip()))
            if len(counts) >= 2 and counts[-1] > counts[0]:
                break
            time.sleep(0.1)
        assert len(counts) >= 2 and counts[-1] > counts[0], counts
        wait(2)
        denied = call(
            "exec",
            names[1],
            "python",
            "-c",
            "open('/observe/no-write','w')",
            check=False,
        )
        assert denied.returncode != 0
        call("stop", "--time", "10", names[1], timeout=20)
        state = json.loads(call("inspect", names[1]).stdout)[0]["State"]
        assert (
            state["ExitCode"] == 0 and not state["OOMKilled"] and not state["Running"]
        ), state
        (output / "result.json").write_text(
            json.dumps(
                {
                    "image": image,
                    "revision": revision,
                    "wal_counts": counts,
                    "web": state,
                },
                indent=2,
            )
        )
    finally:
        for name in names:
            for command in ("logs", "inspect"):
                result = call(command, name, check=False)
                (output / (name + "." + command)).write_text(
                    result.stdout + result.stderr
                )
            call("rm", "-f", name, check=False)
        for volume in volumes:
            call("volume", "rm", volume, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["docker", "podman"], default="docker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=Path("container-evidence"))
    args = parser.parse_args()
    smoke(args.engine, args.image, args.output, args.revision)
