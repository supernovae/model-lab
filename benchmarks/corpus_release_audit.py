"""Compare independent byte replay and receipt-backed Corpus audits on Linux."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import resource
import stat
import subprocess
import sys
import time
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.progress import recent_progress
from klab.experiments.artifacts import snapshot_dependencies
from klab.safe_io import symlinked
from klab.verification_proofs import writable_by_others


def process_io() -> dict[str, int]:
    with Path("/proc/self/io").open() as stream:
        return {name: int(value) for name, value in map(str.split, stream)}


def timed_audit(args: list[str]) -> None:
    from klab.cli.main import main

    sys.argv = ["klab", *args]
    before = process_io()
    own_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()) as output:
        main()
    seconds = time.perf_counter() - start
    after = process_io()
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(
        json.dumps(
            {
                "seconds": seconds,
                "user_seconds": (own.ru_utime - own_before.ru_utime)
                + (children.ru_utime - children_before.ru_utime),
                "system_seconds": (own.ru_stime - own_before.ru_stime)
                + (children.ru_stime - children_before.ru_stime),
                "peak_individual_rss_bytes": 1024
                * max(own.ru_maxrss, children.ru_maxrss),
                "logical_read_bytes": after["rchar:"] - before["rchar:"],
                "logical_write_bytes": after["wchar:"] - before["wchar:"],
                "storage_read_bytes": after["read_bytes:"] - before["read_bytes:"],
                "storage_write_bytes": after["write_bytes:"] - before["write_bytes:"],
                "audit_sha256": canonical_digest(json.loads(output.getvalue())),
            }
        )
    )


def evict_fixture(root: Path, release: Path) -> int:
    """Advise the kernel only for regular, private files in the fixture closure."""
    if (
        symlinked(root)
        or root.stat().st_uid != os.getuid()
        or root.stat().st_mode & 0o077
    ):
        raise ValueError("fixture eviction requires an owned private fixture root")
    root = root.resolve(strict=True)
    files = set()
    for path in (release, *snapshot_dependencies(release, "manifest.json")):
        if symlinked(path) or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"release dependency is outside the fixture: {path}")
        files.update(path.rglob("*") if path.is_dir() else (path,))
    sizes = 0
    for path in sorted(files):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or writable_by_others(info)
        ):
            raise ValueError(f"not an owned fixture file: {path}")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        sizes += info.st_size
    return sizes


def benchmark(release: Path, workspace: Path, *, workers: int, fixture: Path | None):
    release = release.absolute()
    workspace = workspace.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if workspace.is_relative_to(checkout) or workspace.is_relative_to(
        release.resolve()
    ):
        raise ValueError("use a new workspace outside the checkout and release")
    manifest = json.loads((release / "manifest.json").read_bytes())
    workspace.mkdir(mode=0o700)
    scratch = workspace / "scratch"
    scratch.mkdir(mode=0o700)
    proof_root = workspace / "proofs"
    proof_root.mkdir(mode=0o700)
    journal = scratch / f"verify-{manifest['release_id'][:16]}.progress.jsonl"
    env = {**os.environ, "XDG_CONFIG_HOME": str(workspace / "settings")}
    reports = []
    for mode in ("cold", "receipt"):
        advised = (
            evict_fixture(fixture.absolute(), release)
            if fixture and mode == "cold"
            else 0
        )
        before = journal.stat().st_size if journal.exists() else 0
        with (workspace / f"{mode}.log").open("xb") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_timed-audit",
                    "--work-dir",
                    str(proof_root),
                    "corpus",
                    "audit",
                    str(release),
                    "--scratch-dir",
                    str(scratch),
                    "--workers",
                    str(workers),
                    *(["--cold"] if mode == "cold" else []),
                ],
                env=env,
                stderr=log,
                stdout=subprocess.PIPE,
                check=True,
            )
        recorded = journal.exists() and journal.stat().st_size > before
        final = next(recent_progress(journal), {}) if recorded else {}
        reports.append(
            {
                "mode": mode,
                "advised_fixture_bytes": advised,
                "replay_journal_written": recorded,
                "replay_summary": final,
                **json.loads(result.stdout),
            }
        )
    if reports[0]["audit_sha256"] != reports[1]["audit_sha256"]:
        raise ValueError("audit results differ between byte replay and receipt mode")
    return {
        "release_id": manifest["release_id"],
        "workers": workers,
        "cache_policy": "fixture file DONTNEED advice" if fixture else "uncontrolled",
        "runs": reports,
    }


if __name__ == "__main__":
    if sys.argv[1:2] == ["_timed-audit"]:
        timed_audit(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("release", type=Path)
        parser.add_argument(
            "workspace", type=Path, help="New directory for receipts and scratch"
        )
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument(
            "--evict-fixture", type=Path, metavar="PRIVATE_FIXTURE_ROOT"
        )
        options = parser.parse_args()
        if sys.platform != "linux" or options.workers < 1:
            parser.error("requires Linux and positive worker count")
        print(
            json.dumps(
                benchmark(
                    options.release,
                    options.workspace,
                    workers=options.workers,
                    fixture=options.evict_fixture,
                ),
                indent=2,
            )
        )
