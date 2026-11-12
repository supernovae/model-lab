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
# data.packing re-audited 2026-11-11 (require_prepared, merged over the worker
# budget re-pin above): prepare_data and _prepare_data gain a require_prepared
# flag that raises PreparedDataMissing on a cache miss, after the unchanged
# verified cache-hit return and before any production. A cache hit takes the same
# load_prepared_data path as before; receipt validators and the exclusion edge
# set are unchanged, and no excluded edge gained a verifier use. Re-pin the
# existing complete-owner signatures.
# data.packing re-audited 2026-11-11 (token store): _prepare_data gains a branch
# that takes a local_text split from the reusable token store instead of chunked
# tokenization; new producer-only helpers (_token_stores, _copy_from_token_store,
# _documents_encode_independently, token_store_root, the staging guard on
# prepare_data) use the existing writer mint. The published arrays, manifest and
# receipt are byte-identical to the chunked path (tested). The cache-hit path,
# load_prepared_data, receipt validators and the exclusion edge set are
# unchanged; no excluded edge gained a verifier use. Re-pin the existing
# complete-owner signatures.
# data.packing re-audited 2026-11-11 (one encoder pool): the streaming collector
# no longer passes a scratch directory to PreparationEncoder; no verifier edge
# changed.
# data.packing re-audited 2026-11-11 (binary mixture output):
# _collect_token_mixture copies raw int32 targets instead of parsing JSON rows
# and _prepare_data no longer stages mixture chunks; producer-only, no
# verifier edge changed.
# data.packing re-audited 2026-11-11 (generic dataset sources): the
# local_stories source and its verifier were removed; no excluded edge gained
# a new caller.
# experiments.artifacts, experiments.lock, data.tokenizer and data.sources re-
# audited 2026-11-11 (generic dataset sources): only the local_stories
# branches and the legacy snapshot import were removed; no excluded edge
# gained a new caller.
# data.packing re-audited 2026-11-11 (serial tokenization reason):
# _serial_encoding_reason names the failed eligibility check and prints it; no
# excluded edge gained a new caller.
# data.packing re-audited 2026-11-12 (preference pairs and chat templates):
# _prepare_data gained the local_preference branch (_collect_pairs), the chat
# template's special-token check and the shared packing_identity; producer-
# only, no excluded edge gained a new caller.
_EXCLUSION_SIGNATURES: dict[tuple[str, str], str] = {
    (
        "data.packing",
        "data.preparation_workers",
    ): "5c7806e994760bc86aa84aa5ca202ff351acfc2de8bd2a11263cfa855cf4d099",
    (
        "experiments.artifacts",
        "data.sources",
    ): "706bf76a2a036c3c83d38c58f02033cf366258ac01ed6df75c40563ba0a50cdc",
    (
        "data.packing",
        "corpus.mixture",
    ): "2fdebb54eb1d826b85277c540c17ef66e687899d568f3aed4088e627350de135",
    (
        "workers",
        "workers.transport",
    ): "f9718ddb7cc69bf564b2455a22315ed442d7ee8885208060415a159caaa10084",
    (
        "data.packing",
        "data.tokenizer",
    ): "e58279c1f697868b5ecbd7953686dab169b56e674bb3d841a06f78528b6e4189",
    (
        "data.packing",
        "data.sources",
    ): "f91fc487f75859a237e962ce704679d30d7092d679bd81adfc5c038de1b9f6b2",
    (
        "data",
        "data.tokenizer",
    ): "059401c5074a0f0dd2e82528082ea94222d5eda531f750ed942fdb5ede1cca99",
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
    ): "ebcae28af4ba59c167bcc102bdd901fee3dce3d9d48a25c8a2f62855c659409d",
    (
        "data.packing",
        "data.allocation",
    ): "ecfa93fde64a9f88e21d97a208100c97fc66cb0943244fb09795178c919f18ef",
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
    ): "27fc1395234d56f2ad212a18d885450456524ea472ef255c2fff3146a65abdd7",
    (
        "data.packing",
        "data.preparation_chunks",
    ): "4a0d0386dd9311c39038f8d0396228ef207295bc2e511e52352693851775d1d6",
    (
        "data.packing",
        "data.preparation_telemetry",
    ): "1522d1c8f8d47dd3edcd3d977ee22ec05269d233ae4de7414c4e69704c08fd74",
    (
        "data.packing",
        "experiments.source_compatibility",
    ): "a5758efc23187ee0727f3408d09ea97d613ba28951d3fb899b2dd7daa200ac35",
    (
        "data.packing",
        "progress",
    ): "05797bba775f0b59a65e3ea19ae0fdc7005b95a539d750db88cec1bf078ad273",
    (
        "data.packing",
        "resource_envelope",
    ): "002adcb2506c8a696658eabb3d04d79e32b030addf64afd3dc4d684ff333e551",
    (
        "data.packing",
        "workdir",
    ): "e0f73da63d0b441165991225e296c5850114812cc494cb635e6f2d176f0a8011",
    (
        "data.packing",
        "workspace_cleanup",
    ): "66fb2a041966232213546fad111e6d1e9b1b451efb2c654e6c69a63a5c1e43ce",
    (
        "data.packing",
        "workspace_preflight",
    ): "09cd03e501983cc06169e3fac0f4729f59620ab466ed01618d5d7ac9f7d8bf55",
    (
        "corpus.tokenizer_bakeoff",
        "corpus.cli",
    ): "cc239d6512386c4722891b278a3dc81378ec2f99e017cd3f082187982db38e3e",
    (
        "experiments.artifacts",
        "corpus.acquisition",
    ): "37c56efb4e040464c41a8c0a3437f9d6dcc2dd1407737527fc221e2651d78a37",
    (
        "experiments.artifacts",
        "corpus.export",
    ): "00fbfafe5ece02d10261a60ef6367f3facdd31192a0d866520c7d6a804f5baf9",
    (
        "experiments.artifacts",
        "corpus.release",
    ): "f4220be0bcfe6855e57f9be19cabb39bcc545367c1d262f7ef652bdba2fc607a",
    (
        "experiments.artifacts",
        "data.packing",
    ): "fe6005e14f77906a167433dc8e59720011be4830f20ae0e59b6fac5f7055eeb4",
    (
        "experiments.artifacts",
        "data.tokenizer",
    ): "582dea3c0d69825034105e4d746a616dda682efd18ac6c648b43d31539721135",
    (
        "experiments.artifacts",
        "evaluation.capabilities",
    ): "3ebc94b9a6f6da87d38a69843fb7ec43f66ef129980f949bf1b4249872ffe457",
    (
        "experiments.artifacts",
        "staging",
    ): "c77477899ebd374c17a271d85052b09a8f456abca33c0d07833ed22467e46707",
    (
        "experiments.artifacts",
        "training.checkpoints",
    ): "beb71cae5b1f352d21fa738ecc67142680b4dcc049b38ff3f2c1124509b489e4",
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
