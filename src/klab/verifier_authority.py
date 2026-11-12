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
_ARTIFACT = ("experiments.artifacts",)
_CORPUS = (
    "config.models",
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
    "data.datasets",
    "data.conversations",
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
        "corpus.tokenizer_bakeoff",
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
        ("data", "data.tokenizer"),  # package convenience re-export
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
        ("data.packing", "data.byte_hash"),
        ("data.packing", "data.conversations"),
        ("data.packing", "data.datasets"),
        ("data.packing", "data.encoding"),
        # Snapshot acquisition is used only by _prepare_data. The array/file
        # and prepared-inventory verifiers authenticate already packed bytes.
        # Tokenizer/stage authorities must retain this semantic source chain.
        ("data.packing", "data.sources"),
        ("data.packing", "data.tokenizer"),
        ("data.packing", "data.preparation_chunks"),
        ("data.packing", "data.preparation_telemetry"),
        ("data.packing", "data.preparation_workers"),
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
}
_ARTIFACT_IMPORTS = frozenset(
    ("experiments.artifacts", name) for name in set(_ARTIFACT_BRANCHES.values())
)
_OPERATIONAL_EDGES = frozenset({("corpus.tokenizer_bakeoff", "corpus.cli")})


def _excluded_edges(kind: str) -> frozenset[tuple[str, str]]:
    result = _COMMON_EXCLUDED_EDGES | _OPERATIONAL_EDGES
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
    return result


# Pinned to the AST of each excluded import and every use of its imported
# names. A new use in a verifier, or any change to an existing owner, makes
# authority unavailable until the exclusion is reviewed and re-pinned.
_EXCLUSION_SIGNATURES: dict[tuple[str, str], str] = {
    (
        "data.packing",
        "data.preparation_workers",
    ): "652548cd9b6369a4285763d2b47e35011ee40cd571f14ed1d77658814fc60f1a",
    (
        "experiments.artifacts",
        "data.sources",
    ): "9354fdbb38a405e6230e9fe5142fca72dcfa3f448abefc4de158fa3e193557e4",
    (
        "data.packing",
        "corpus.mixture",
    ): "a4348052e395be388b3e9208faf3b0dbcbde6935c86076f18412944940dae465",
    (
        "workers",
        "workers.transport",
    ): "f9718ddb7cc69bf564b2455a22315ed442d7ee8885208060415a159caaa10084",
    (
        "data.packing",
        "data.tokenizer",
    ): "04e5168f58a9bcb52f467275837c480748c9001c825ffd852cb7fbac1bc5e7cd",
    (
        "data.packing",
        "data.sources",
    ): "a314fe512723056453f6b1d6dfd983c49bb0780172f0b1f6ffd8efda5ad06bd3",
    (
        "data",
        "data.tokenizer",
    ): "059401c5074a0f0dd2e82528082ea94222d5eda531f750ed942fdb5ede1cca99",
    (
        "data.packing",
        "corpus.export",
    ): "ad3d5a27399cbd004d73a0bd9b28947b14c4943836793528ff9ece0f66593c0b",
    (
        "data.packing",
        "data.byte_hash",
    ): "ba5a65c4059dacd0c84a56fbf5378e5963e464352d63be88dea03361738a1238",
    (
        "data.packing",
        "data.conversations",
    ): "392c1097c3c88740860ab036839cbb014478d40aaa3917e90748e9e89661595f",
    (
        "data.packing",
        "data.datasets",
    ): "0bbf6d3d2f08d9cf8371d3b30259682f05915087fba092250b2e327488842a18",
    (
        "data.packing",
        "data.encoding",
    ): "f3d612beb3f443c8ab9d208f014724544ab23cde1d68370cb1b11aa245133095",
    (
        "data.packing",
        "data.preparation_chunks",
    ): "55db8af83a0420cad30dbfe035d9a1626cbfce31401ddd8b47e6a07a7d86ea98",
    (
        "data.packing",
        "data.preparation_telemetry",
    ): "03c45dd6b3a980ae3c3d38bbb482712442e6c1590bba9033a2acab872f4f90bb",
    (
        "data.packing",
        "progress",
    ): "5eb83c244b7a4b94f6d1a8850c4f041e84cca76642524296bfd9dde4c54cdbff",
    (
        "data.packing",
        "resource_envelope",
    ): "823f3f2c416c2c1bace707c7fd18aa80fe71594a26d602e534e34ab32eb67eb4",
    (
        "data.packing",
        "workdir",
    ): "1f0ab102bf42545b33e722956b1c14330811ab61143b93ae1f65a013088a95a1",
    (
        "data.packing",
        "workspace_cleanup",
    ): "084043609715ba72e1b9f601f4e36213ca5666ed72e096c3af34ccd2c8834e08",
    (
        "data.packing",
        "workspace_preflight",
    ): "7a04302a03bcd021874ca9f3c979ba0be4b896e756b26672e26125c919c6de64",
    (
        "corpus.tokenizer_bakeoff",
        "corpus.cli",
    ): "cc239d6512386c4722891b278a3dc81378ec2f99e017cd3f082187982db38e3e",
    (
        "experiments.artifacts",
        "corpus.acquisition",
    ): "b180e30b88fbee9d9304009a57f499a6ff2f6eb760a5ea171131038b9092f3db",
    (
        "experiments.artifacts",
        "corpus.export",
    ): "65f10b6fecfccd25dae0cf40a12705242de6d71ffb18710c639d713a026037dd",
    (
        "experiments.artifacts",
        "corpus.release",
    ): "0b73a1e79b98751471fe107adbb9b770e7500210911e2eeb214b9814acd2949e",
    (
        "experiments.artifacts",
        "data.packing",
    ): "fe6005e14f77906a167433dc8e59720011be4830f20ae0e59b6fac5f7055eeb4",
    (
        "experiments.artifacts",
        "data.tokenizer",
    ): "ca3d690025dcc61773bc027ab652a774db7803b409d82565799d4b53de33271b",
    (
        "experiments.artifacts",
        "staging",
    ): "1251f82759d06e3151e6ac8a10341451fde9d112da98ab235035b1d7df5d05c0",
    (
        "experiments.artifacts",
        "training.checkpoints",
    ): "86d8022b8d245568476766a0128d631528c069e210823c35e28f04c7e996d563",
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


def local_import_closure(
    roots: tuple[str, ...], *, within: str, exclude: frozenset[str] = frozenset()
) -> dict[str, str]:
    """SHA-256 of every module reachable from ROOTS by static imports inside WITHIN.

    Names are relative to the ``klab`` package (``corpus.pipeline``). Imports
    leaving the WITHIN package, or naming an EXCLUDE module, are not followed.
    """
    package = Path(__file__).parent
    files: dict[str, str] = {}
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in files or name in exclude:
            continue
        path = package / (name.replace(".", "/") + ".py")
        source = path.read_bytes()  # a missing root fails rather than hashing nothing
        files[name] = hashlib.sha256(source).hexdigest()
        imports, _, _ = _imports(source, name, package=False)
        pending.extend(
            imported
            for imported in imports
            if imported.startswith(within + ".")
            and (package / (imported.replace(".", "/") + ".py")).is_file()
        )
    return dict(sorted(files.items()))
