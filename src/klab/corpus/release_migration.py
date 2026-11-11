"""Explicit source-verified relocation of retained local corpus releases.

The source and target observations are new, domain-separated local evidence.
They never rewrite the historical release, execution records, or signatures.
The shared verification key trusts the service UID on both hosts, as does the
existing host-local proof store; it is not an independent third-party proof.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.preparation_inputs import corpus_work_root, local_scope
from klab.lab_records import write_json_atomic
from klab.training.manifest import sha256_file
from klab.verification_proofs import ProofStore

_SOURCE = "source-release-migration-v1"
_TARGET = "target-release-migration-v1"
_SOURCE_PROGRESS = "source-release-migration-progress-v1"
_TARGET_PROGRESS = "target-release-migration-progress-v1"


def _sign(store: ProofStore, observation: dict) -> dict:
    return {"observation": observation, "hmac": store.migration_signature(observation)}


def _read_signed(store: ProofStore, path: Path, kind: str) -> dict:
    if not store.trusted(path) or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("migration evidence is missing or not trusted")
    envelope = json.loads(path.read_bytes())
    if not isinstance(envelope, dict) or set(envelope) != {"observation", "hmac"}:
        raise ValueError("invalid migration evidence envelope")
    observation = envelope["observation"]
    if (
        not isinstance(observation, dict)
        or observation.get("kind") != kind
        or not isinstance(envelope["hmac"], str)
        or not hmac.compare_digest(
            envelope["hmac"], store.migration_signature(observation)
        )
    ):
        raise ValueError("migration evidence authentication failed")
    return observation


def _migration_dir(root: Path) -> Path:
    directory = root / "release-migrations"
    directory.mkdir(mode=0o700, exist_ok=True)
    if not ProofStore(root).trusted(directory):
        raise ValueError("migration directory is not owned and trusted")
    return directory


def _save(path: Path, envelope: dict, *, exclusive: bool = False) -> None:
    old = os.umask(0o077)
    try:
        write_json_atomic(path, envelope, exclusive=exclusive)
    finally:
        os.umask(old)


def _under(root: Path, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
        raise ValueError("migration input lies outside its source work root")
    return path


def _inputs(scope: dict, root: Path) -> tuple[dict[str, list], dict[str, int]]:
    roots: dict[str, list] = {}
    files: dict[str, int] = {}
    for mapping in (scope["request"]["inputs"], scope["source_bindings"]):
        for name, rows in mapping.items():
            path = _under(root, name)
            if name in roots and roots[name] != rows:
                raise ValueError("inconsistent source input fingerprints")
            roots[name] = rows
            for row in rows:
                if len(row) != 7 or not isinstance(row[0], str):
                    raise ValueError("invalid source input fingerprint")
                relative = Path(row[0])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("invalid source input member")
                member = path if row[0] == "." else path / relative
                if stat.S_ISREG(row[3]):
                    key = member.relative_to(root).as_posix()
                    if key in files and files[key] != row[4]:
                        raise ValueError("inconsistent source input size")
                    files[key] = row[4]
                elif not stat.S_ISDIR(row[3]):
                    raise ValueError("unsupported source input member")
    return roots, files


def _snapshot_digests(build: dict, release: Path, inputs: dict[str, int]) -> dict:
    """Reuse file hashes already bound by unchanged snapshot identities."""
    known = {}
    root = corpus_work_root(release)
    for snapshot in build["snapshots"]:
        directory = (
            release.parent.parent
            / "snapshots"
            / snapshot["source_id"]
            / snapshot["sha256"]
        )
        manifest_path = directory / "manifest.json"
        manifest_rel = manifest_path.relative_to(root).as_posix()
        if manifest_rel not in inputs:
            continue
        manifest = json.loads(manifest_path.read_bytes())
        identity = {
            "declaration_sha256": manifest["declaration_sha256"],
            "adapter": manifest["adapter"],
            "files": manifest["files"],
        }
        declaration = manifest["declaration"]
        acquisition = declaration["acquisition"]
        if declaration["kind"] == "wikimedia_dump" or (
            declaration["kind"] == "huggingface_dataset"
            and acquisition.get("retained_subset") is not None
        ):
            identity["retrieval"] = manifest["retrieval"]
        if (
            canonical_digest(identity) != snapshot["sha256"]
            or manifest["snapshot_sha256"] != snapshot["sha256"]
            or manifest["source_id"] != snapshot["source_id"]
        ):
            raise ValueError("snapshot manifest identity changed")
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe snapshot member")
            name = (directory / "files" / relative).relative_to(root).as_posix()
            if name in inputs:
                if inputs[name] != entry["size"]:
                    raise ValueError("snapshot size differs from execution binding")
                known[name] = entry["sha256"]
    return known


def _historical_evidence(
    release: Path, root: Path, origin_root: Path, scope: dict
) -> dict:
    """Bind the original publication completion and its proof or executor records."""
    store = ProofStore(root)
    publication = json.loads((release / "release-publication.json").read_bytes())
    evidence = {}

    def add(path: Path, expected: str | None = None) -> None:
        if not store.trusted(path) or not path.is_file():
            raise ValueError(f"historical migration evidence is untrusted: {path}")
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"historical migration evidence differs: {path}")
        evidence[path.relative_to(root).as_posix()] = {
            "sha256": digest,
            "size": path.stat().st_size,
        }

    stages = release.parent.parent / "build-stages" / canonical_digest(scope["request"])
    matches = []
    for candidate in stages.glob("completion-*.json"):
        if not store.trusted(candidate) or candidate.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("untrusted build completion record")
        event = json.loads(candidate.read_bytes())
        if event.get("sha256") == publication.get("completion_sha256"):
            if canonical_digest(event["record"]) != event["sha256"]:
                raise ValueError("historical completion identity differs")
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("historical publication completion missing or ambiguous")
    add(matches[0])
    method = publication.get("method")
    if method == "in_place_adopted_executor_build":
        reference = publication["execution_declaration"]
        old = _under(origin_root, reference["path"])
        declaration = root / old.relative_to(origin_root)
        add(declaration, reference["sha256"])
        record = json.loads(declaration.read_bytes())
        refs = record["evidence"]
        roles = {
            "job_spec",
            "job_report",
            "session_spec",
            "session_report",
            "container_report",
            "stdout",
            "stderr",
            "events",
        }
        if not isinstance(refs, dict) or set(refs) != roles:
            raise ValueError("historical executor evidence is incomplete")
        for ref in refs.values():
            old = _under(origin_root, ref["path"])
            add(root / old.relative_to(origin_root), ref["sha256"])
    elif method == "in_place_retained_native_build":
        expected = publication["historical_proof_sha256"]
        matches = []
        for candidate in store.directory.glob("*.json"):
            if store.trusted(candidate) and sha256_file(candidate) == expected:
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError("historical signed execution proof missing or ambiguous")
        add(matches[0], expected)
    else:
        raise ValueError("unsupported historical retained-release publication")
    return evidence


def _verify_source(
    release: Path, expected_id: str, checkout: Path, revision: str
) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source verifier revision must be a full commit ID")
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.strip()
    if actual != revision:
        raise ValueError("source verifier checkout revision differs")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            "src/klab",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    if dirty:
        raise ValueError("source verifier checkout has uncommitted code")
    env["PYTHONPATH"] = str(checkout / "src")
    code = (
        "import json,sys; from pathlib import Path; "
        "from klab.corpus.release import verify_release; "
        "m=verify_release(Path(sys.argv[1]),expected_id=sys.argv[2]); "
        "print(json.dumps({'release_id':m['release_id']}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(release), expected_id],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise ValueError(
            "source native release verification failed: "
            + (detail[-1][:500] if detail else f"exit {result.returncode}")
        )
    if json.loads(result.stdout).get("release_id") != expected_id:
        raise ValueError("source native release returned a different identity")


def _progress(
    store: ProofStore, path: Path, kind: str, release_id: str, root: Path
) -> dict:
    if path.exists() or path.is_symlink():
        state = _read_signed(store, path, kind)
        if (
            state.get("release_id") != release_id
            or state.get("root") != str(root)
            or state.get("authority") != _authority()
        ):
            raise ValueError(
                "migration progress belongs to a different release or root"
            )
        return state
    return {
        "kind": kind,
        "release_id": release_id,
        "root": str(root),
        "authority": _authority(),
        "files": {},
    }


def _authority() -> str:
    package = Path(__file__).resolve().parents[1]
    names = (
        "canonical.py",
        "verification_proofs.py",
        "training/manifest.py",
        "corpus/admission_inspection_cache.py",
        "corpus/completed_release.py",
        "corpus/preparation_inputs.py",
        "corpus/release.py",
        "corpus/release_migration.py",
        "corpus/retained_mixture.py",
    )
    return canonical_digest({name: sha256_file(package / name) for name in names})


def _hash_file(
    root: Path,
    name: str,
    size: int,
    expected: str | None,
    state: dict,
    store: ProofStore,
    progress: Path,
) -> str:
    path = _under(root, str(root / name))
    if not store.trusted(path) or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"untrusted or changed migration file: {path}")
    fingerprint = capture([path])[str(path)]
    cached = state["files"].get(name)
    if cached and cached.get("fingerprint") == fingerprint:
        digest = cached["sha256"]
    else:
        digest = sha256_file(path)
        if capture([path])[str(path)] != fingerprint:
            raise ValueError(f"migration file changed while hashing: {path}")
        if expected is not None and digest != expected:
            raise ValueError(f"migration file digest mismatch: {path}")
        state["files"][name] = {"sha256": digest, "fingerprint": fingerprint}
        _save(progress, _sign(store, state))
    if expected is not None and digest != expected:
        raise ValueError(f"migration cached digest mismatch: {path}")
    return digest


def attest_source(
    release: Path, *, verifier_checkout: Path, verifier_revision: str
) -> Path:
    """Verify the original release and sign a new source-side migration observation."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    build = json.loads((release / "build.json").read_bytes())
    manifest = json.loads((release / "manifest.json").read_bytes())
    release_id = manifest["release_id"]
    if (
        canonical_digest({k: v for k, v in manifest.items() if k != "release_id"})
        != release_id
    ):
        raise ValueError("source release manifest identity differs")
    scope = local_scope(build)
    if scope is None or build["build_id"] != release.name:
        raise ValueError("migration requires a retained local build")
    roots, files = _inputs(scope, root)
    store = ProofStore(root)
    directory = _migration_dir(root)
    progress = directory / (release_id + ".source-progress.json")
    output = directory / (release_id + ".source.json")
    _verify_source(release, release_id, Path(verifier_checkout), verifier_revision)
    unchanged(roots)
    metadata_paths = [
        release / name
        for name in ("build.json", "manifest.json", "release-publication.json")
    ]
    metadata_before = capture(metadata_paths)
    if output.exists() or output.is_symlink():
        old = _read_signed(store, output, _SOURCE)
        if (
            old.get("release_id") != release_id
            or old.get("source_root") != str(root)
            or old.get("release_relative_path") != release.relative_to(root).as_posix()
            or old.get("source_verifier_checkout")
            != str(Path(verifier_checkout).resolve(strict=True))
            or old.get("source_verifier_revision") != verifier_revision
            or old.get("migration_authority") != _authority()
            or old.get("build_sha256") != sha256_file(release / "build.json")
            or old.get("manifest_sha256") != sha256_file(release / "manifest.json")
            or old.get("publication_sha256")
            != sha256_file(release / "release-publication.json")
            or set(old.get("inputs", {})) != set(files)
        ):
            raise ValueError("existing source migration observation differs")
        return output
    state = _progress(store, progress, _SOURCE_PROGRESS, release_id, root)
    known = _snapshot_digests(build, release, files)
    historical = _historical_evidence(release, root, root, scope)
    historical_before = capture([root / name for name in historical])
    result = {}
    for name, size in sorted(files.items()):
        digest = known.get(name)
        basis = "recorded_snapshot_digest" if digest is not None else "source_sha256"
        if digest is None:
            digest = _hash_file(root, name, size, None, state, store, progress)
        result[name] = {"size": size, "sha256": digest, "basis": basis}
    unchanged(roots)
    _verify_source(release, release_id, Path(verifier_checkout), verifier_revision)
    observation = {
        "kind": _SOURCE,
        "release_id": release_id,
        "source_root": str(root),
        "release_relative_path": release.relative_to(root).as_posix(),
        "build_sha256": sha256_file(release / "build.json"),
        "manifest_sha256": sha256_file(release / "manifest.json"),
        "publication_sha256": sha256_file(release / "release-publication.json"),
        "source_verifier_checkout": str(Path(verifier_checkout).resolve(strict=True)),
        "source_verifier_revision": verifier_revision,
        "source_verification": "native_retained_release_verified",
        "migration_authority": _authority(),
        "inputs": result,
        "historical_evidence": historical,
    }
    unchanged(roots)
    unchanged(metadata_before)
    unchanged(historical_before)
    _save(output, _sign(store, observation), exclusive=True)
    return output


def _shape(rows: list) -> list:
    return [
        [row[0], stat.S_IFMT(row[3]), row[4] if stat.S_ISREG(row[3]) else None]
        for row in rows
    ]


def adopt_target(release: Path, source_observation: Path) -> Path:
    """Hash the mapped target bytes once and publish separate target evidence."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    store = ProofStore(root)
    source_path = Path(source_observation)
    source = _read_signed(store, source_path, _SOURCE)
    source_envelope = json.loads(source_path.read_bytes())
    if source_envelope.get("observation") != source or not hmac.compare_digest(
        source_envelope.get("hmac", ""), store.migration_signature(source)
    ):
        raise ValueError("source observation changed during adoption")
    metadata_paths = [
        release / name
        for name in ("build.json", "manifest.json", "release-publication.json")
    ]
    metadata_before = capture(metadata_paths)
    manifest = json.loads((release / "manifest.json").read_bytes())
    build = json.loads((release / "build.json").read_bytes())
    release_id = manifest["release_id"]
    old_root = Path(source["source_root"])
    if (
        not old_root.is_absolute()
        or old_root == Path(old_root.anchor)
        or ".." in old_root.parts
        or source.get("source_verification") != "native_retained_release_verified"
        or source.get("migration_authority") != _authority()
        or source["release_id"] != release_id
        or source["release_relative_path"] != release.relative_to(root).as_posix()
        or canonical_digest({k: v for k, v in manifest.items() if k != "release_id"})
        != release_id
        or build["build_id"] != release.name
        or canonical_digest(build["identity"]) != release.name
        or source["build_sha256"] != sha256_file(release / "build.json")
        or source["manifest_sha256"] != sha256_file(release / "manifest.json")
        or source["publication_sha256"]
        != sha256_file(release / "release-publication.json")
    ):
        raise ValueError("source observation and target release differ")
    scope = local_scope(build)
    if scope is None:
        raise ValueError("migration requires a retained local build")
    roots, inputs = _inputs(scope, old_root)
    if set(source["inputs"]) != set(inputs):
        raise ValueError("source observation input mapping differs")
    historical = _historical_evidence(release, root, old_root, scope)
    if historical != source["historical_evidence"]:
        raise ValueError("migrated historical evidence differs")
    historical_before = capture([root / name for name in historical])
    mapped_roots = []
    for name, old_rows in roots.items():
        path = root / Path(name).relative_to(old_root)
        actual = capture([path])[str(path)]
        if _shape(actual) != _shape(old_rows):
            raise ValueError(f"migrated input inventory differs: {path}")
        mapped_roots.append(path)
    directory = _migration_dir(root)
    progress = directory / (release_id + ".target-progress.json")
    output = directory / (release_id + ".target.json")
    if output.exists() or output.is_symlink():
        verify_migrated_release(release, manifest, expected_id=release_id)
        return output
    state = _progress(store, progress, _TARGET_PROGRESS, release_id, root)
    for name, size in sorted(inputs.items()):
        item = source["inputs"][name]
        if item["size"] != size:
            raise ValueError("migration input size differs")
        _hash_file(root, name, size, item["sha256"], state, store, progress)
    for name, item in sorted(build["files"].items()):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe release output path")
        relative = (release / name).relative_to(root).as_posix()
        _hash_file(root, relative, item["size"], item["sha256"], state, store, progress)
    paths = [
        *mapped_roots,
        *(release / name for name in build["files"]),
        release / "build.json",
        release / "manifest.json",
        release / "release-publication.json",
        *(root / name for name in historical),
    ]
    if any(directory.is_relative_to(p) for p in mapped_roots if p.is_dir()):
        raise ValueError("migration receipt directory overlaps a captured input")
    fingerprints = capture(paths)
    unchanged(metadata_before)
    unchanged(historical_before)
    for name, old_rows in roots.items():
        path = root / Path(name).relative_to(old_root)
        if _shape(fingerprints[str(path)]) != _shape(old_rows):
            raise ValueError(
                f"migrated input inventory changed during verification: {path}"
            )
    checked = set(inputs)
    checked.update(
        (release / name).relative_to(root).as_posix() for name in build["files"]
    )
    for name in checked:
        path = root / name
        if (
            not store.trusted(path)
            or capture([path])[str(path)] != state["files"][name]["fingerprint"]
        ):
            raise ValueError(f"migration file changed after hashing: {path}")
    observation = {
        "kind": _TARGET,
        "release_id": release_id,
        "target_root": str(root),
        "release_relative_path": release.relative_to(root).as_posix(),
        "source_observation": source_envelope,
        "target_fingerprints": fingerprints,
        "migration_authority": _authority(),
    }
    _save(output, _sign(store, observation), exclusive=True)
    return output


def verify_migrated_release(
    release: Path, manifest: dict, *, expected_id: str | None = None
) -> dict:
    """Verify a distinct migration receipt and unchanged target fingerprints."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    release_id = manifest["release_id"]
    store = ProofStore(root)
    receipt = root / "release-migrations" / (release_id + ".target.json")
    observation = _read_signed(store, receipt, _TARGET)
    source_envelope = observation["source_observation"]
    source = source_envelope["observation"]
    old_root = Path(source["source_root"])
    build = json.loads((release / "build.json").read_bytes())
    scope = local_scope(build)
    if (
        scope is None
        or not old_root.is_absolute()
        or old_root == Path(old_root.anchor)
        or ".." in old_root.parts
    ):
        raise ValueError("invalid migrated release scope")
    roots, inputs = _inputs(scope, old_root)
    expected_paths = {str(root / Path(name).relative_to(old_root)) for name in roots}
    for name in build["files"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe release output path")
        expected_paths.add(str(release / name))
    expected_paths.update(
        str(release / name)
        for name in ("build.json", "manifest.json", "release-publication.json")
    )
    expected_paths.update(
        str(_under(root, str(root / name))) for name in source["historical_evidence"]
    )
    if (
        source.get("kind") != _SOURCE
        or not hmac.compare_digest(
            source_envelope["hmac"], store.migration_signature(source)
        )
        or source.get("source_verification") != "native_retained_release_verified"
        or source["release_id"] != release_id
        or observation["release_id"] != release_id
        or observation["target_root"] != str(root)
        or observation["release_relative_path"] != release.relative_to(root).as_posix()
        or source["release_relative_path"] != observation["release_relative_path"]
        or set(source["inputs"]) != set(inputs)
        or set(observation["target_fingerprints"]) != expected_paths
        or any(
            not Path(name).is_relative_to(root) or ".." in Path(name).parts
            for name in observation["target_fingerprints"]
        )
        # A completed versioned receipt remains valid across compatible Lab
        # updates. Authority was checked while the two observations were made;
        # the signed evidence and target fingerprints remain the binding.
        or source.get("migration_authority") != observation.get("migration_authority")
        or canonical_digest({k: v for k, v in manifest.items() if k != "release_id"})
        != release_id
        or (expected_id is not None and expected_id != release_id)
    ):
        raise ValueError("migration receipt identity differs")
    unchanged(observation["target_fingerprints"])
    return manifest


def migrated_roots(release: Path, manifest: dict) -> tuple[Path, Path]:
    """Return the source and target roots bound by a verified migration."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    verify_migrated_release(release, manifest, expected_id=manifest["release_id"])
    receipt = root / "release-migrations" / (manifest["release_id"] + ".target.json")
    observation = _read_signed(ProofStore(root), receipt, _TARGET)
    return Path(observation["source_observation"]["observation"]["source_root"]), root


def migrated_dependencies(manifest: dict, work_root: Path) -> tuple[Path, ...] | None:
    """Return the target dependencies only for an authenticated migrated release."""
    receipt = (
        work_root / "release-migrations" / (manifest["release_id"] + ".target.json")
    )
    if not receipt.exists() and not receipt.is_symlink():
        return None
    release = (
        work_root / "corpora" / manifest["corpus_id"] / "builds" / manifest["build_id"]
    )
    verify_migrated_release(release, manifest, expected_id=manifest["release_id"])
    record = _read_signed(ProofStore(work_root), receipt, _TARGET)
    return (receipt, *(Path(name) for name in record["target_fingerprints"]))


_STAGE_SOURCE = "source-mixture-stage-migration-v1"
_STAGE_TARGET = "target-mixture-stage-migration-v1"
_STAGE_PROGRESS = "target-mixture-stage-migration-progress-v1"


def attest_mixture_stages(release: Path) -> Path:
    """Record source stage evidence without minting historical signatures."""
    from klab.corpus.retained_mixture import collect_mixture_stage_evidence

    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    store = ProofStore(root)
    manifest = json.loads((release / "manifest.json").read_bytes())
    build = json.loads((release / "build.json").read_bytes())
    release_id = manifest["release_id"]
    directory = _migration_dir(root)
    source_release_path = directory / (release_id + ".source.json")
    source_release = _read_signed(store, source_release_path, _SOURCE)
    if (
        source_release.get("release_id") != release_id
        or source_release.get("source_root") != str(root)
        or source_release.get("release_relative_path")
        != release.relative_to(root).as_posix()
        or source_release.get("build_sha256") != sha256_file(release / "build.json")
        or source_release.get("manifest_sha256")
        != sha256_file(release / "manifest.json")
        or source_release.get("publication_sha256")
        != sha256_file(release / "release-publication.json")
        or canonical_digest({k: v for k, v in manifest.items() if k != "release_id"})
        != release_id
        or canonical_digest(build["identity"]) != build["build_id"]
        or build["build_id"] != release.name
    ):
        raise ValueError("source release observation differs from current release")
    scope = local_scope(build)
    if scope is None:
        raise ValueError("stage migration requires a retained local build")
    roots, _ = _inputs(scope, root)
    unchanged(roots)
    source_before = capture(
        [
            source_release_path,
            release / "build.json",
            release / "manifest.json",
            release / "release-publication.json",
        ]
    )
    cohort = collect_mixture_stage_evidence(
        root, release.parent.parent, build["identity"]
    )
    stage_before = capture([root / name for name in cohort["files"]])
    unchanged(roots)
    unchanged(source_before)
    unchanged(stage_before)
    observation = {
        "kind": _STAGE_SOURCE,
        "release_id": release_id,
        "source_root": str(root),
        "release_relative_path": release.relative_to(root).as_posix(),
        "release_observation_sha256": sha256_file(source_release_path),
        "migration_authority": _authority(),
        "cohort": cohort,
        "limitation": "global stages may use owner-private native completions, not historical signatures",
    }
    output = directory / (release_id + ".stages.source.json")
    if output.exists() or output.is_symlink():
        if _read_signed(store, output, _STAGE_SOURCE) != observation:
            raise ValueError("existing source stage observation differs")
        return output
    _save(output, _sign(store, observation), exclusive=True)
    return output


def adopt_mixture_stages(release: Path, source_observation: Path) -> Path:
    """Verify mapped target stage bytes once and record target fingerprints."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    store = ProofStore(root)
    manifest = json.loads((release / "manifest.json").read_bytes())
    source_root, target_root = migrated_roots(release, manifest)
    if target_root != root:
        raise ValueError("target release root differs")
    source_path = Path(source_observation)
    source = _read_signed(store, source_path, _STAGE_SOURCE)
    envelope = json.loads(source_path.read_bytes())
    if envelope.get("observation") != source or not hmac.compare_digest(
        envelope.get("hmac", ""), store.migration_signature(source)
    ):
        raise ValueError("source stage observation changed during adoption")
    release_id = manifest["release_id"]
    source_release_path = root / "release-migrations" / (release_id + ".source.json")
    if (
        source.get("release_id") != release_id
        or source.get("source_root") != str(source_root)
        or source.get("release_relative_path") != release.relative_to(root).as_posix()
        or source.get("release_observation_sha256") != sha256_file(source_release_path)
        or source.get("migration_authority") != _authority()
    ):
        raise ValueError("source stage observation differs from migrated release")
    source_release = _read_signed(store, source_release_path, _SOURCE)
    target_release = _read_signed(
        store, root / "release-migrations" / (release_id + ".target.json"), _TARGET
    )
    if source_release != target_release["source_observation"]["observation"]:
        raise ValueError("source stage and release observations differ")
    cohort = source["cohort"]
    files = cohort["files"]
    if (
        not isinstance(files, dict)
        or not files
        or cohort["source_ids"] != sorted(manifest["build_identity"]["lock"])
    ):
        raise ValueError("source stage file mapping differs")
    directory = _migration_dir(root)
    output = directory / (release_id + ".stages.target.json")
    if output.exists() or output.is_symlink():
        verify_mixture_stage_migration(release, manifest)
        return output
    progress = directory / (release_id + ".stages.target-progress.json")
    state = _progress(store, progress, _STAGE_PROGRESS, release_id, root)
    for name, item in sorted(files.items()):
        _hash_file(root, name, item["bytes"], item["sha256"], state, store, progress)
    paths = [root / name for name in files]
    fingerprints = capture(paths)
    for name in files:
        path = _under(root, str(root / name))
        if (
            not store.trusted(path)
            or capture([path])[str(path)] != state["files"][name]["fingerprint"]
        ):
            raise ValueError(f"stage file changed after hashing: {path}")
    observation = {
        "kind": _STAGE_TARGET,
        "release_id": release_id,
        "target_root": str(root),
        "release_relative_path": release.relative_to(root).as_posix(),
        "source_observation": envelope,
        "target_fingerprints": fingerprints,
        "migration_authority": _authority(),
    }
    _save(output, _sign(store, observation), exclusive=True)
    return output


def verify_mixture_stage_migration(release: Path, manifest: dict) -> dict:
    """Return a mapped cohort only after both migration observations verify."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    store = ProofStore(root)
    source_root, _ = migrated_roots(release, manifest)
    release_id = manifest["release_id"]
    receipt = root / "release-migrations" / (release_id + ".stages.target.json")
    target = _read_signed(store, receipt, _STAGE_TARGET)
    envelope = target["source_observation"]
    source = envelope["observation"]
    source_release_path = root / "release-migrations" / (release_id + ".source.json")
    source_release = _read_signed(store, source_release_path, _SOURCE)
    target_release = _read_signed(
        store, root / "release-migrations" / (release_id + ".target.json"), _TARGET
    )
    if (
        source.get("release_observation_sha256") != sha256_file(source_release_path)
        or source_release != target_release["source_observation"]["observation"]
    ):
        raise ValueError("source stage and release observations differ")
    files = source["cohort"]["files"]
    paths = {str(_under(root, str(root / name))) for name in files}
    if (
        source.get("kind") != _STAGE_SOURCE
        or not hmac.compare_digest(envelope["hmac"], store.migration_signature(source))
        or source.get("release_id") != release_id
        or source.get("source_root") != str(source_root)
        or source.get("release_relative_path") != release.relative_to(root).as_posix()
        or target.get("release_id") != release_id
        or target.get("target_root") != str(root)
        or target.get("release_relative_path") != release.relative_to(root).as_posix()
        or source.get("migration_authority") != target.get("migration_authority")
        or set(target["target_fingerprints"]) != paths
        or source["cohort"]["source_ids"] != sorted(manifest["build_identity"]["lock"])
    ):
        raise ValueError("target stage migration receipt differs")
    unchanged(target["target_fingerprints"])

    def mapped(name):
        path = _under(source_root, name)
        relative = path.relative_to(source_root).as_posix()
        if relative not in files:
            raise ValueError("stage path is not covered by migration evidence")
        return str(root / relative)

    source_cohort = source["cohort"]
    evidence = source_cohort["evidence"]

    def mapped_proof(value):
        return {
            k: mapped(v) if k in {"completion_path", "cache_path", "proof_path"} else v
            for k, v in value.items()
        }

    return {
        "release_id": release_id,
        "release_path": str(release),
        "manifest": manifest,
        "documents": source_cohort["documents"],
        "references": mapped(source_cohort["references"]),
        "parents": mapped(source_cohort["parents"]),
        "origins": mapped(source_cohort["origins"]),
        "sources": [mapped(path) for path in source_cohort["sources"]],
        "source_ids": source_cohort["source_ids"],
        "evidence": {
            "references": mapped_proof(evidence["references"]),
            "deduplication": mapped_proof(evidence["deduplication"]),
            "origins": mapped_proof(evidence["origins"]),
            "sources": {
                key: mapped_proof(value) for key, value in evidence["sources"].items()
            },
        },
        "stage_migration_receipt": str(receipt),
    }


def migrated_bound_inputs(release: Path, manifest: dict) -> dict:
    """Map the original inputs already checked by the release migration."""
    release = Path(release).resolve(strict=True)
    root = corpus_work_root(release)
    old_root, _ = migrated_roots(release, manifest)
    store = ProofStore(root)
    source_path = (
        root / "release-migrations" / (manifest["release_id"] + ".source.json")
    )
    source = _read_signed(store, source_path, _SOURCE)
    if source.get("source_root") != str(old_root):
        raise ValueError("source input root differs")
    return {
        str(old_root / name): {"path": str(_under(root, str(root / name))), **item}
        for name, item in source["inputs"].items()
    }
