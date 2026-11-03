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
_EXCLUSION_SIGNATURES: dict[tuple[str, str], str] = {
    (
        "data.packing",
        "data.preparation_workers",
    ): "4fa040e8a03c41d4938c306e2fe6bba3799f0e5e6b48ef08f8b681a31831cafb",
    (
        "experiments.artifacts",
        "data.sources",
    ): "59400a133cadd8c521204da9dd44d4ad36716e9cd359f0a6b8637987540510f0",
    (
        "data.packing",
        "corpus.mixture",
    ): "a6ea2b0defaf40bcc165169e042b384febe776c907922d6a51d12c854aa40617",
    (
        "workers",
        "workers.transport",
    ): "ba1975a626825a9a98a0d65eb1d480b239f2145df09127ba4b8bcf0392d0d3d1",
    (
        "data.packing",
        "data.tokenizer",
    ): "56d70d021458c2bf5eeac0b0713054447e16bb39cc8064b6110270c52176ed6f",
    (
        "data.packing",
        "data.sources",
    ): "14c60ab6de34770c6ba88c1e43b4632c5efe03306d2774267f8e1fc9af6180a7",
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
    ): "703ad792d314ea02ee0d8e1ed52483c802405ed67c916500001de0ec106722f0",
    (
        "data.packing",
        "data.allocation",
    ): "ece8a66f2d83755978b92407b330caacf7b350e919baa3d86cc49d706dfd52a5",
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
    ): "1457b1ab8973332c9f9608bf66878f767bdff4f4f2d74af1f9f21408adf2710c",
    (
        "data.packing",
        "data.local_stories",
    ): "be972f1dfaff8b42e223115343f88bccf69ecb8c6c58423d330e444d6d469aeb",
    (
        "data.packing",
        "data.preparation_chunks",
    ): "f9089bb58cd927c8b0f7928505c9ccdeeb90b593fcff80fdd64c7cf9d017e6d0",
    (
        "data.packing",
        "data.preparation_telemetry",
    ): "57e8c4f0515acb0cbf4bb34576394755f943d529b09a7bda15ee63a25f80a661",
    (
        "data.packing",
        "experiments.source_compatibility",
    ): "7b6a0c42202da4d0e7a7d17b018688820a00ddb06626a0d7fe52a73168f4d57c",
    (
        "data.packing",
        "progress",
    ): "d431bc7d6067b71a5bed94dbd0104bdd97ffbcd58e6614bf3bf985186852e78c",
    (
        "data.packing",
        "resource_envelope",
    ): "ba06e85038854041212c89aaddab9e5ae5123da6dd69ccb8b9855f174cc12238",
    (
        "data.packing",
        "workdir",
    ): "663292fe730983bd3dc49e88ba3684d1b8631ee4714f2256096c19353722407d",
    (
        "data.packing",
        "workspace_cleanup",
    ): "387d791e1615a692cd05e34b447f1a99c64885b9af73ef4d9d70ed2a9602189c",
    (
        "data.packing",
        "workspace_preflight",
    ): "4251e7df9076354737ac4ca2d13bc44baac0089383bb994fe52b2a88b5d76084",
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
    ): "87dcdcdc201f47a1883857ad24d645e5d4ec3dfdcd5ffb4e805cf025cd63f126",
}


@lru_cache(maxsize=128)
def _exclusion_signature(source: bytes, imported: str) -> str:
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
                if _exclusion_signature(source, imported) != _EXCLUSION_SIGNATURES.get(
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
