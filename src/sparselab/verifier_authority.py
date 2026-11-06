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

from sparselab.canonical import canonical_digest

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
# data.packing re-audited 2026-11-04 against f61763f: reversing the
# _tokenizer_sha256 -> canonical_tokenizer_sha256 promotion and removing the
# producer-only map/tokenizer-object check in prepare_data restores the complete
# baseline module AST. Only _prepare_data and prepare_data owners changed;
# load_prepared_data and receipt validators are unchanged. The new tokenizer
# load is confined to prepare_data; no excluded edge gained a verifier use.
# Re-pin the existing complete-owner signatures without excluding new edges.
# Verified append producer re-audit: existing exclusion edge set is unchanged.
# New append production calls native collectors/preflight only; load_prepared_data,
# receipt verification and all verifier closure membership remain unchanged.
# Whole-owner pins below include append dispatch and producer-only helpers.
_EXCLUSION_SIGNATURES: dict[tuple[str, str], str] = {
    (
        "data.packing",
        "data.preparation_workers",
    ): "e5b731976865f31d558447bc238794e331cbf82e6aa5583733846da8ef3e05e0",
    (
        "experiments.artifacts",
        "data.sources",
    ): "59400a133cadd8c521204da9dd44d4ad36716e9cd359f0a6b8637987540510f0",
    (
        "data.packing",
        "corpus.mixture",
    ): "2af16abca502bb82c157fef2aee80df5a70685cbdb0102fc583bd0164504ed59",
    (
        "workers",
        "workers.transport",
    ): "ba1975a626825a9a98a0d65eb1d480b239f2145df09127ba4b8bcf0392d0d3d1",
    (
        "data.packing",
        "data.tokenizer",
    ): "30be6bd1f5d0251b22ff4b7f4b4d7657890e872aca8cea5fe442dfadb8256170",
    (
        "data.packing",
        "data.sources",
    ): "dc3769e15a7341f951e05bfb1e7af663873d3c42749a24e43acf2aead95730a2",
    (
        "data",
        "data.tokenizer",
    ): "1f6ff5da53d2ea9a2093b49afba1c0b09cae934a331e27e6ac1ad28f840e9978",
    (
        "data.local_stories",
        "engram.packs",
    ): "f6ea05c786f5b8a4d5038426c1d314440b5051ee2395e524ff4ab6f8a819ee9e",
    (
        "recovery.evidence",
        "evaluation.suite",
    ): "d9c8038634e3c25c2c442ad561089c7f92f568499e3110fec48ac4bf0cba8722",
    (
        "recovery.evidence",
        "experiments.lock",
    ): "9ab4afdbe4cddab20f61bb2bc1e48314aa253f39348cfec242d083db410d697a",
    (
        "data.packing",
        "corpus.export",
    ): "671168408a9906e94efdc41a8ba54e0833407d28eb438a7d5a592a17cc67df61",
    (
        "data.packing",
        "data.allocation",
    ): "c0ab7e30055854ded1bc725e6aedb74dd372053d15a82de25e16088027090818",
    (
        "data.packing",
        "data.byte_hash",
    ): "1c41efaf18a6619eeee832d47d1e2604d72b305226153635270ae902395d709a",
    (
        "data.packing",
        "data.conversations",
    ): "6c6e354ed411f7adef4c22d76333563546c6cb75dd54356e0a57ed8b4d254c12",
    (
        "data.packing",
        "data.datasets",
    ): "ae708e1cf4f0902e44ec22cb75ff2f1f844ffda49b8216ae629de2e622ca6e2d",
    (
        "data.packing",
        "data.encoding",
    ): "86584acb9103d8920d08c1d04cf30fb57455dadd35ea2ce97fe15b8a126a7264",
    (
        "data.packing",
        "data.local_stories",
    ): "f02c6ed06a3bbed3711634f334ce4763e5af7008807e91775d2b3c61938365ae",
    (
        "data.packing",
        "data.preparation_chunks",
    ): "57b23fa77a23ed7e531fa8005d62af6a8ad90740d19088c93b2969d836f7287a",
    (
        "data.packing",
        "data.preparation_telemetry",
    ): "87bddd87d3adf9e6fea7aacc71be974ccc28f5fdf228abd0d7259fa2c6846c35",
    (
        "data.packing",
        "experiments.source_compatibility",
    ): "da30e3a2d720629d9f4f2a740405e1d2b24535f27f4eae3f63db725e4b257d2f",
    (
        "data.packing",
        "progress",
    ): "17675c284d532681502a7217e5c669a89824acc8c09b11c52d1cf637fccdc271",
    (
        "data.packing",
        "resource_envelope",
    ): "598361e1f96fa8eb4340bace59844fe8a6808af035ad43b137e9406886299fb9",
    (
        "data.packing",
        "workdir",
    ): "6e7d990f9d82a6c7c2b157b0c1bc533b4cd9964cb33624c7625cd6a94c86cf84",
    (
        "data.packing",
        "workspace_cleanup",
    ): "2e111f35237382d0fca9cc2d1e8de7fe7613f69d5911160e0ac42299f6ee5381",
    (
        "data.packing",
        "workspace_preflight",
    ): "a342c4da0dc49e441712bf974db8807e496f2699a80dafc04b54f67f270dea0d",
    (
        "corpus.tokenizer_bakeoff",
        "corpus.cli",
    ): "ad8963589fb231d03564e21e93e00054a8de9e19b92d357cadefa0e2c223271c",
    (
        "experiments.artifacts",
        "corpus.acquisition",
    ): "9744c3d2d0da2dec449dbc3a8a0faeb7ff42bd504822c3123188f57a05b66d26",
    (
        "experiments.artifacts",
        "corpus.export",
    ): "5458a195fe630b4c0a5cba2ce73f9a681197a0c1b1bc4b2d6542cf29c339a3a3",
    (
        "experiments.artifacts",
        "corpus.release",
    ): "c4c6aa6c331c2697495cc0b38d92f61aa5d3cad016c8ff3da04e9facb8950dfa",
    (
        "experiments.artifacts",
        "data.packing",
    ): "4e28f0e941104c03e44cd0153dfe8933e11f47379c4bd056c6fbb8714b172774",
    (
        "experiments.artifacts",
        "data.tokenizer",
    ): "1dcb44a3c9ded6ec87d429086c7c4c4873227c7d04e294fb1326780d47f1b822",
    (
        "experiments.artifacts",
        "evaluation.capabilities",
    ): "434d3459db0c03d03e54bedf84d91ab9482c867a2eff28b02935d2745b29dc6c",
    (
        "experiments.artifacts",
        "staging",
    ): "68b5c1bd897d08316bb55d37d6e616de97a3372da489a7d31ddc94d83b854727",
    (
        "experiments.artifacts",
        "training.checkpoints",
    ): "49b95d47ac8400b8905328e8cdc731ee5034f5b0b97f977803d39ef5f4879c40",
    (
        "experiments.plan",
        "campaign.plan",
    ): "2a02bc1d20f9f43903a9c22c641279554787e2d1a3cdf38bea0bc732398ebd4b",
    (
        "experiments.plan",
        "recovery.provenance",
    ): "eba675a70badc9a67800ed15c017efef6ca1a9293ce4a55650be73a4b5dcfbb5",
    (
        "training.manifest",
        "runtime",
    ): "9945e77afb4e51c21e9c61d4c966b4841b5ba8c3efcd3959d3d28fcca42d0eca",
    (
        "verification_proofs",
        "experiments.artifacts",
    ): "ef3237ce0f4e0bd2de58092bd5712316738fe25d49981b2993bda24f135d11ae",
}


@lru_cache(maxsize=128)
def exclusion_signature(source: bytes, imported: str) -> str:
    """Compute the static review signature; this does not authorize an exclusion."""
    tree = ast.parse(source)
    declared: set[str] = set()
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == f"sparselab.{imported}":
            imports.append(ast.dump(node, include_attributes=False))
            declared.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == f"sparselab.{imported}":
                    imports.append(ast.dump(node, include_attributes=False))
                    declared.add(alias.asname or "sparselab")
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
            if alias.name.startswith("sparselab."):
                name = alias.name.removeprefix("sparselab.")
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
        elif node.module and node.module.startswith("sparselab."):
            base = node.module.removeprefix("sparselab.")
        elif node.module == "sparselab":
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
            if target.value.startswith("sparselab."):
                name = target.value.removeprefix("sparselab.")
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
