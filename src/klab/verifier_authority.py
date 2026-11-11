"""Versioned operational verifier authority scoped to typed receipt domains.

Semantic entry points are explicit; imports (including lazy function imports) in
those modules are discovered recursively inside each domain's semantic packages.
Neither source filenames nor receipt schema versions alone authenticate code.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from klab.canonical import canonical_digest

AUTHORITY_SCHEMA = 2
_COMMON = ("__init__", "verification_proofs", "verifier_authority", "training.manifest")
_ARTIFACT = ("experiments.artifacts", "experiments.plan")
_CORPUS = (
    "campaign.state",
    "config.models",
    "corpus.identity",
    "corpus.project",
    "corpus.provenance",
    "corpus.rights",
    "corpus.pipeline",
    "corpus.large_build",
    "corpus.release",
    "corpus.acquisition",
    "data.conversations",
)
_PREPARED = (
    "data.verification",
    "data.packing",
    "data.tokenizer",
    "data.local_stories",
    "data.datasets",
    "data.conversations",
    "data.allocation",
    "data.byte_hash",
    "config.models",
)
_ARRAY = ("data.verification", "data.packing", "config.models")

# These are verifier entry points, not an exhaustive manually maintained graph.
# Static imports from them and their imported semantic helpers are discovered
# recursively. CLI, UI and execution/training modules cannot enter a prepared
# authority through a convenience import from the verifier.
_CLOSURES: dict[str, tuple[str, ...]] = {
    "file": ("data.verification",),
    "prepared_array": _ARRAY,
    "stage_inventory_file": (
        "staging",
        "data.verification",
        "data.packing",
        "data.tokenizer",
        "config.models",
    ),
    "dispatch_cache_asset": (
        "workers.bundles",
        "workers.models",
        "staging",
        "owned_copy",
        "data.verification",
        "data.packing",
        "data.tokenizer",
        "config.models",
        "training.checkpoints",
        "training.continuation",
        "model.portable_engram",
        "corpus.export",
        "corpus.release",
        "experiments.artifacts",
    ),
    "checkpoint_member": (
        "training.checkpoints",
        "training.mlx_checkpoints",
        "data.verification",
        "config.models",
        "model.inspection",
    ),
    "run_artifact": (
        "data.verification",
        "evaluation.evidence",
        "training.checkpoints",
        "training.mlx_checkpoints",
        "model.inspection",
        "config.models",
    ),
    "ingested_run_artifact": (
        "data.verification",
        "experiments.evidence",
        "training.checkpoints",
        "training.mlx_checkpoints",
        "model.inspection",
        "config.models",
    ),
    "source_snapshot": _ARTIFACT + _CORPUS,
    "corpus_build": _ARTIFACT + _CORPUS,
    "corpus_release": _ARTIFACT + _CORPUS,
    "corpus_export": _ARTIFACT
    + _CORPUS
    + ("corpus.export", "config.loading", "data.tokenizer"),
    "tokenizer": _ARTIFACT
    + _CORPUS
    + (
        "corpus.export",
        "config.loading",
        "data.tokenizer",
        "data.local_stories",
        "recovery.evidence",
    ),
    "prepared_data": _ARTIFACT + _ARRAY,
    "stage_bundle": _ARTIFACT
    + _PREPARED
    + (
        "staging",
        "training.checkpoints",
        "training.mlx_checkpoints",
        "model.inspection",
        "data.encoding",
        "owned_copy",
    ),
    "checkpoint": _ARTIFACT
    + (
        "training.checkpoints",
        "training.mlx_checkpoints",
        "training.continuation",
        "model.inspection",
        "config.models",
        "data.verification",
    ),
    "capability_card": _ARTIFACT
    + ("evaluation.capabilities", "experiments.compiler", "config.models"),
    "prompt_set": _ARTIFACT,
}

# These existing imports belong to other operations in shared source files.
# The importing file itself is always digested. All *other* local imports,
# including newly added helpers outside the domain package, recurse by default.
# Entry roots for corpus/tokenizer/stage kinds explicitly include their own
# semantic verifier modules rather than inheriting irrelevant prepared-array work.
_COMMON_EXCLUDED_EDGES = frozenset(
    {
        ("verification_proofs", "experiments.artifacts"),
        ("training.manifest", "runtime"),
        ("data.local_stories", "engram.packs"),
        ("data", "data.tokenizer"),  # package convenience re-export
        ("recovery.evidence", "evaluation.suite"),
        ("recovery.evidence", "experiments.lock"),
    }
)
_ARRAY_EXCLUDED_EDGES = frozenset(
    {
        ("data.packing", "corpus.export"),
        # Only _prepare_data calls verify_mixture_dataset before packing and
        # binds the authenticated receipt digest into cache_identity. Loading
        # packed bytes checks that identity, manifest and member digests without
        # replaying source preparation. Semantic tokenizer/stage domains retain
        # this edge; the importing packing module remains digested in all kinds.
        # Tightening mixture verification does not retrospectively invalidate
        # warm packed-byte proofs: they rely on the preparation receipt identity.
        # Current semantic verification is a separate authority boundary.
        ("data.packing", "corpus.mixture"),
        ("data.packing", "data.allocation"),
        ("data.packing", "data.byte_hash"),
        ("data.packing", "data.conversations"),
        ("data.packing", "data.datasets"),
        ("data.packing", "data.encoding"),
        ("data.packing", "data.local_stories"),
        # Snapshot acquisition is used only by _prepare_data. The array/file
        # and prepared-inventory verifiers authenticate already packed bytes.
        # Tokenizer/stage authorities must retain this semantic source chain.
        ("data.packing", "data.sources"),
        ("data.packing", "data.tokenizer"),
        ("data.packing", "data.preparation_chunks"),
        ("data.packing", "data.preparation_telemetry"),
        ("data.packing", "data.preparation_workers"),
        ("data.packing", "experiments.source_compatibility"),
        ("data.packing", "progress"),
        ("data.packing", "resource_envelope"),
        ("data.packing", "workdir"),
        ("data.packing", "workspace_cleanup"),
        ("data.packing", "workspace_preflight"),
    }
)

_ARTIFACT_BRANCHES = {
    "dataset_snapshot": "data.sources",
    "source_snapshot": "corpus.acquisition",
    "corpus_build": "corpus.release",
    "corpus_release": "corpus.release",
    "corpus_export": "corpus.export",
    "tokenizer": "data.tokenizer",
    "prepared_data": "data.packing",
    "stage_bundle": "staging",
    "checkpoint": "training.checkpoints",
    "capability_card": "evaluation.capabilities",
}
_ARTIFACT_IMPORTS = frozenset(
    ("experiments.artifacts", name) for name in set(_ARTIFACT_BRANCHES.values())
)
_PLAN_OPERATIONAL_EDGES = frozenset(
    {
        ("experiments.plan", "campaign.plan"),
        ("experiments.plan", "recovery.provenance"),
        ("corpus.tokenizer_bakeoff", "corpus.cli"),
    }
)


def _excluded_edges(kind: str) -> frozenset[tuple[str, str]]:
    result = _COMMON_EXCLUDED_EDGES | _PLAN_OPERATIONAL_EDGES
    if kind == "source_snapshot":
        # Snapshot verification needs worker identity schemas transitively,
        # not the package's convenience exports for dispatch and transport.
        result |= {("workers", "workers.transport")}
    if kind in {"file", "prepared_array", "prepared_data"}:
        result |= _ARRAY_EXCLUDED_EDGES
    if kind in _ARTIFACT_BRANCHES:
        result |= _ARTIFACT_IMPORTS - {
            ("experiments.artifacts", _ARTIFACT_BRANCHES[kind])
        }
    elif kind == "prompt_set":
        result |= _ARTIFACT_IMPORTS
    return result


# Pinned below to the AST of each excluded import and all existing uses of its
# imported names. A new use in a verifier, or any change to an existing user,
# makes authority unavailable until the exclusion is explicitly re-audited.
# Context-transfer audit 2026-11-04: only _prepare_data changes.
# It validates short unpadded splits through MatchedStoryDataset; all byte-loader
# verifiers and their uses of excluded imports are unchanged. No exclusions added.
# Combined PR290/305 audit 2026-11-03: versus integrated PR290 only
# _prepare_data changes, to verify the explicit tokenizer origin.
# load_prepared_data and _verified_receipt ASTs are unchanged; semantic
# tokenizer/stage authorities retain origin verification and source closure.
# data.packing re-audited 2026-11-01: AST owner comparison against 903ad2f
# changes only _collect_streaming and _prepare_data, adding the producer-only
# _streaming_values helper. preparation_workers is only used by _prepare_data;
# it cannot authorize loading already packed bytes. New chunk scheduling,
# cancellation and telemetry are producer-only; load_prepared_data and
# _verified_receipt remain byte-for-byte unchanged. No excluded producer
# operation is reachable from those verifier entry points.
# data.packing re-audited 2026-10-29: stage-scoped cache identity and resumable
# token-target packing change producer uses only; load_prepared_data and
# _verified_receipt do not invoke any newly excluded preparation operation.
# data.packing edges re-audited 2026-10-10: since ba6370d only _prepare_data
# (token-mixture preparation) changed; no verifier gained a use of an excluded name.
# corpus.mixture audited 2026-10-11: its sole import/use is in _prepare_data;
# load_prepared_data/_verified_receipt and prepared artifact verification do not
# call it. Pin that import and its complete owner, as for the other preparation
# edges, so any new verifier use or owner change requires another audit.
# experiments.artifacts re-audited 2026-10-14: the new dataset_snapshot branch
# calls only data.sources.verify_snapshot (cold) and existing identity checks.
# Existing artifact branches and their excluded imports/uses are unchanged.
# The pins cover the complete dispatch owner. The new data.sources import is
# used only by dataset_snapshot, so exclude that branch for other artifact kinds
# using the existing branch map. Semantic domains retain their native source
# chains; prepared-data inventory proofs remain independent of preparation.
# data.packing re-audited 2026-11-03: the dataset projection in _prepare_data
# moved verbatim to prepared_dataset_identity. Inlining that helper restores
# the prior module AST exactly; no imports, verifier calls or exclusions changed.
# Re-pin the affected complete owners without broadening the excluded edges.
# verification_proofs record re-audited 2026-11-03: only a diagnostic event
# was added before the existing unavailable-key return. Removing that call
# restores the complete record AST; imported artifact uses, cold-evidence
# validation, trust checks and excluded edges are unchanged.
# verification_proofs record re-audited 2026-11-03: promoting receipt_path
# and record_event changes only method names. Reversing the public API renames
# and removing their docstrings restores the complete module AST exactly;
# artifact imports/uses, cold evidence, trust checks and exclusions are unchanged.
# data.packing re-audited 2026-11-04 against f61763f: reversing the
# _tokenizer_sha256 -> canonical_tokenizer_sha256 promotion and removing the
# producer-only map/tokenizer-object check in prepare_data restores the complete
# baseline module AST. Only _prepare_data and prepare_data owners changed;
# load_prepared_data and receipt validators are unchanged. The new tokenizer
# load is confined to prepare_data; no excluded edge gained a verifier use.
# Re-pin the existing complete-owner signatures without excluding new edges.
# Integration re-audit 2026-11-11: preserved main prefix block checks and proofs;
# only producer owners change. load_prepared_data and receipt validation ASTs
# remain identical to main; no exclusion edges are added or removed.
# Verified append producer re-audit: existing exclusion edge set is unchanged.
# New append production calls native collectors/preflight only; load_prepared_data,
# receipt verification and all verifier closure membership remain unchanged.
# Whole-owner pins below include append dispatch and producer-only helpers.
# data.packing re-audited 2026-11-11 (preparation worker budget): the only change
# is one keyword argument, token_budget, passed to PreparationWorkers inside the
# producer owner _prepare_data. load_prepared_data, receipt validators and the
# exclusion edge set are unchanged; no excluded edge gained a verifier use.
# Re-pin the existing complete-owner signatures.
_EXCLUSION_SIGNATURES: dict[tuple[str, str], str] = {
    (
        "data.packing",
        "data.preparation_workers",
    ): "3ba22c57267e4b03a754ae8b2d27cc409c619f8357359f39b64eba39f39c12a4",
    (
        "experiments.artifacts",
        "data.sources",
    ): "cd263ac6f69ea3a19b39c42efc17ada209ba14f90e06256bed022750a175f9ab",
    (
        "data.packing",
        "corpus.mixture",
    ): "efbc2c8df9856393623bbcd20e4a8463c4483ed5856db46075882141a993f968",
    (
        "workers",
        "workers.transport",
    ): "f9718ddb7cc69bf564b2455a22315ed442d7ee8885208060415a159caaa10084",
    (
        "data.packing",
        "data.tokenizer",
    ): "85a25da082875dd2d6b06c016d943c5f31bb8b34e7ba74b96b02dc6acc41088c",
    (
        "data.packing",
        "data.sources",
    ): "cdaca397132f56ecdf758945cb23f1a77df738f3785598d2b6d00da89214fea0",
    (
        "data",
        "data.tokenizer",
    ): "059401c5074a0f0dd2e82528082ea94222d5eda531f750ed942fdb5ede1cca99",
    (
        "data.local_stories",
        "engram.packs",
    ): "2846ac75f0ac04ef0eeb1c6e51e3a0d3a973278df2fa3d2bfc1a56e25d6c0f68",
    (
        "recovery.evidence",
        "evaluation.suite",
    ): "8a213bf93de72dcf87e89d0d32d0588294ab9b3b85431a9b6bb4651e620f6c7a",
    (
        "recovery.evidence",
        "experiments.lock",
    ): "2539a4683cda81622323231021f41ea11aa7d6490ce7fa7b738136b55242f9d8",
    (
        "data.packing",
        "corpus.export",
    ): "65b06a0512789c07b9e918426822d6aaf99c4b13cc6d4aa8276d7f5f70612a12",
    (
        "data.packing",
        "data.allocation",
    ): "59438a75d0e39f3825406bbbcbfc7aae36f2d4c89e907afd0aea1d364d01efd4",
    (
        "data.packing",
        "data.byte_hash",
    ): "ba5a65c4059dacd0c84a56fbf5378e5963e464352d63be88dea03361738a1238",
    (
        "data.packing",
        "data.conversations",
    ): "afbfd629d9069c78decd5ea392015933158428d8f12491da6629433437f2b97c",
    (
        "data.packing",
        "data.datasets",
    ): "c3a23408bafc968f228dd4c8a185dc2fa7a5185d630a085f29bb4f164f8d8d64",
    (
        "data.packing",
        "data.encoding",
    ): "c1581ab616281807342526b4b1f55fb9651f7e56cdbb6de33306552f91d5776e",
    (
        "data.packing",
        "data.local_stories",
    ): "b2c35e630407c4e40ce365161b00b7511c1179f5dd942bf7f2ec1a4d07e04e14",
    (
        "data.packing",
        "data.preparation_chunks",
    ): "c41db3c3dad3409ae38a87ed58a6304e5369e2ea7a06d47a04ed03e146e94801",
    (
        "data.packing",
        "data.preparation_telemetry",
    ): "aff8f7601dc0d0a23678eec4c1ae1e0dec0386ab70434f472f9dd0bb0781091b",
    (
        "data.packing",
        "experiments.source_compatibility",
    ): "9bce305af93bb593dda221fe31c6dbcb25521c74705a0699809248ef16f7500a",
    (
        "data.packing",
        "progress",
    ): "87df2512d30668e2d2b069a1356b552c35a99b720b7ea011145b07f7d05eb242",
    (
        "data.packing",
        "resource_envelope",
    ): "77af6d275b6a15c15aa840cfb96c97f8ab8e47b9552e339f6c0aff6dc75bd304",
    (
        "data.packing",
        "workdir",
    ): "e773a125cb320f33b1c7a256d84d8a664dd2290bed621153cae27c28818506c1",
    (
        "data.packing",
        "workspace_cleanup",
    ): "d598fd0d7474996e17426b85808ca993d0c6c8c8539e073fd4b0368829b7e43b",
    (
        "data.packing",
        "workspace_preflight",
    ): "3c815920a85ddd7f3e4c57d2b35ed03a8a578d6831ae48f4b2a81337e0dd0d79",
    (
        "corpus.tokenizer_bakeoff",
        "corpus.cli",
    ): "cc239d6512386c4722891b278a3dc81378ec2f99e017cd3f082187982db38e3e",
    (
        "experiments.artifacts",
        "corpus.acquisition",
    ): "e1e6a4cc26f3e46ee201fde7485c3794f7556d4825f62f9500554791374863e6",
    (
        "experiments.artifacts",
        "corpus.export",
    ): "6f722d95edbec8fa02693a80fb5a8545e902f57b008c7b3c16aa6d5740c515f1",
    (
        "experiments.artifacts",
        "corpus.release",
    ): "253c7ea8987c7d8dc24ce965cc5e2e2484eecece520938c449897c89bd619eec",
    (
        "experiments.artifacts",
        "data.packing",
    ): "fe6005e14f77906a167433dc8e59720011be4830f20ae0e59b6fac5f7055eeb4",
    (
        "experiments.artifacts",
        "data.tokenizer",
    ): "250a4be8d3b0c681884ef8fe9350cb97e8fa05c22b72aea3b9fdf6c41e15d78f",
    (
        "experiments.artifacts",
        "evaluation.capabilities",
    ): "f3a98af9a75509195051d748cdf1c26ab5c4acc171bda7807e83e25b53ccba12",
    (
        "experiments.artifacts",
        "staging",
    ): "3a60a538573b5d5a440ec74fd9d97dedfd326b10cf072c493b17db0b533f8592",
    (
        "experiments.artifacts",
        "training.checkpoints",
    ): "cade9b273545b50f6d0ba28780edf221585c24180763deac2c066989a60b100e",
    (
        "experiments.plan",
        "campaign.plan",
    ): "2a02bc1d20f9f43903a9c22c641279554787e2d1a3cdf38bea0bc732398ebd4b",
    (
        "experiments.plan",
        "recovery.provenance",
    ): "822b9fa9b416fa20e29b3b21509615e98cd35b6cfe30b14f147332da30b10696",
    (
        "training.manifest",
        "runtime",
    ): "faa8eb1b27c4053997a42f983367f87581c990e9afdb80bc257f01a098b130e6",
    (
        "verification_proofs",
        "experiments.artifacts",
    ): "3d50379d29878ab1369778d31ff3f5ec2b118f0f3f53c6d30dc5b71703abdcfb",
}


@lru_cache(maxsize=128)
def exclusion_signature(source: bytes, imported: str) -> str:
    """Compute the static review signature; this does not authorize an exclusion."""
    tree = ast.parse(source)
    declared: set[str] = set()
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == f"klab.{imported}":
            imports.append(ast.dump(node, include_attributes=False))
            declared.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == f"klab.{imported}":
                    imports.append(ast.dump(node, include_attributes=False))
                    declared.add(alias.asname or "klab")
    if not imports:
        raise ValueError(f"excluded verifier import disappeared: {imported}")

    def uses_alias(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in declared
            for child in ast.walk(node)
        )

    owners = [
        ast.dump(owner, include_attributes=False)
        for node in tree.body
        for owner in (node.body if isinstance(node, ast.ClassDef) else (node,))
        if not isinstance(owner, (ast.Import, ast.ImportFrom)) and uses_alias(owner)
    ]
    headers = [
        (
            node.name,
            [ast.dump(part, include_attributes=False) for part in header],
        )
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for header in (
            (*node.bases, *node.keywords, *node.decorator_list, *node.type_params),
        )
        if any(uses_alias(part) for part in header)
    ]
    return canonical_digest([*imports, *owners, *headers])


@dataclass(frozen=True)
class VerifierAuthority:
    schema: int
    kind: str
    version: int
    modules: dict[str, str]
    sha256: str


class _Imports(ast.NodeVisitor):
    def __init__(self, module: str, *, package: bool) -> None:
        self.module = module
        self.package = package
        self.modules: set[str] = set()
        self.required: set[str] = set()
        self.package_aliases: set[tuple[str, str]] = set()
        self.import_loaders = {"__import__", "import_module"}
        self.dynamic_evaluators = {"exec", "eval", "globals", "locals"}
        self.ignore_type_checking = False

    def visit_If(self, node: ast.If) -> None:
        if (
            self.ignore_type_checking
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.startswith("klab."):
                name = alias.name.removeprefix("klab.")
                self.modules.add(name)
                self.required.add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            parts = self.module.split(".")
            if not self.package:
                parts.pop()
            if node.level > len(parts):
                raise ValueError("verifier relative import escapes package")
            prefix = parts[: len(parts) - node.level + 1]
            base = ".".join((*prefix, *(node.module.split(".") if node.module else ())))
        elif node.module and node.module.startswith("klab."):
            base = node.module.removeprefix("klab.")
        elif node.module == "klab":
            base = ""
        else:
            return
        if base:
            self.modules.add(base)
            self.required.add(base)
        for alias in node.names:
            self.modules.add(f"{base}.{alias.name}" if base else alias.name)
            self.package_aliases.add((base, alias.name))

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Name) and function.id in self.dynamic_evaluators:
            raise ValueError("dynamic verifier code cannot authorize reuse")
        dynamic = (
            isinstance(function, ast.Name)
            and function.id in self.import_loaders
            or isinstance(function, ast.Attribute)
            and function.attr == "import_module"
        )
        if dynamic:
            target = node.args[0] if node.args else None
            if not isinstance(target, ast.Constant) or not isinstance(
                target.value, str
            ):
                raise ValueError("dynamic verifier import cannot authorize reuse")
            if target.value.startswith("."):
                raise ValueError(
                    "relative dynamic verifier import cannot authorize reuse"
                )
            if target.value.startswith("klab."):
                name = target.value.removeprefix("klab.")
                self.modules.add(name)
                self.required.add(name)
        self.generic_visit(node)


@lru_cache(maxsize=512)
def _imports(
    source: bytes, module: str, *, package: bool
) -> tuple[frozenset[str], frozenset[str], frozenset[tuple[str, str]]]:
    tree = ast.parse(source)
    nodes = tuple(ast.walk(tree))
    visitor = _Imports(module, package=package)
    type_bindings = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name == "TYPE_CHECKING":
                    type_bindings.append((node.module, alias.name))
                if node.module in {"importlib", "builtins"}:
                    if alias.name in {"import_module", "__import__"}:
                        visitor.import_loaders.add(name)
                    elif alias.name in {"exec", "eval", "globals", "locals"}:
                        visitor.dynamic_evaluators.add(name)
    call_functions = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    dynamic_names = visitor.import_loaders | visitor.dynamic_evaluators
    for node in nodes:
        indirect_dynamic_access = (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in dynamic_names
            or isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr in dynamic_names
        )
        if indirect_dynamic_access and id(node) not in call_functions:
            raise ValueError("indirect dynamic verifier access cannot authorize reuse")
    visitor.ignore_type_checking = (
        bool(type_bindings)
        and all(binding == ("typing", "TYPE_CHECKING") for binding in type_bindings)
        and not any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == "TYPE_CHECKING"
            or isinstance(node, ast.arg)
            and node.arg == "TYPE_CHECKING"
            for node in nodes
        )
    )
    visitor.visit(tree)
    return (
        frozenset(visitor.modules),
        frozenset(visitor.required),
        frozenset(visitor.package_aliases),
    )


@lru_cache(maxsize=128)
def _package_exports(source: bytes) -> frozenset[str]:
    exported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            exported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            exported.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            exported.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return frozenset(exported)


def supports_verifier(kind: object) -> bool:
    return isinstance(kind, str) and kind in _CLOSURES


def verifier_authority(kind: str, version: int) -> dict[str, object]:
    """Digest domain verifier roots and their non-excluded transitive local imports."""
    roots = _CLOSURES[kind]  # unsupported kinds never authorize warm reuse
    excluded = _excluded_edges(kind)
    package = Path(__file__).parent
    pending = list(_COMMON + roots)
    scheduled = set(pending)
    discovered_files: dict[Path, bool] = {}

    def is_source_file(path: Path) -> bool:
        if path not in discovered_files:
            discovered_files[path] = path.is_file()
        return discovered_files[path]

    files: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        path = package / (name.replace(".", "/") + ".py")
        if not is_source_file(path):
            path = package / name.replace(".", "/") / "__init__.py"
        source = (
            path.read_bytes()
        )  # missing code fails closed; never bless unknown bytes
        files[name] = hashlib.sha256(source).hexdigest()
        parts = name.split(".")
        for index in range(1, len(parts)):
            initializer = ".".join(parts[:index])
            if initializer not in scheduled and is_source_file(
                package / initializer.replace(".", "/") / "__init__.py"
            ):
                pending.append(initializer)
                scheduled.add(initializer)
        imports, required, aliases = _imports(
            source, name, package=path.name == "__init__.py"
        )
        for imported in imports:
            edge = name, imported
            if edge in excluded:
                if exclusion_signature(source, imported) != _EXCLUSION_SIGNATURES.get(
                    edge
                ):
                    raise ValueError(f"verifier exclusion needs review: {edge}")
                continue
            if imported in scheduled:
                continue
            module_path = package / imported.replace(".", "/")
            candidate = module_path.with_suffix(".py")
            initializer = module_path / "__init__.py"
            if is_source_file(candidate) or is_source_file(initializer):
                pending.append(imported)
                scheduled.add(imported)
            elif module_path.is_dir():
                # A namespace package has no initializer source to digest.
                continue
            elif imported in required:
                raise ValueError(f"missing verifier implementation module: {imported}")
            else:
                base, _, alias = imported.rpartition(".")
                if (base, alias) in aliases:
                    base_file = package / (base.replace(".", "/") + ".py")
                    if is_source_file(base_file):
                        continue  # a member of a verified module, not another module
                    base_init = package / base.replace(".", "/") / "__init__.py"
                    if not base:
                        base_init = package / "__init__.py"
                    if not is_source_file(base_init) or alias not in _package_exports(
                        base_init.read_bytes()
                    ):
                        raise ValueError(f"missing verifier package helper: {imported}")
    modules = dict(sorted(files.items()))
    authority = VerifierAuthority(
        schema=AUTHORITY_SCHEMA,
        kind=kind,
        version=version,
        modules=modules,
        sha256=canonical_digest(modules),
    )
    return asdict(authority)
