"""Author existing portable descriptors from explicitly selected metadata bytes."""

import hashlib
import json
from pathlib import Path

from klab.declaration_context import canonical_reference, safe_path
from klab.documents import parse_document
from klab.experiments.recipe import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENTS,
    PortableRecipe,
    _metadata_bytes,
    snapshot_inventory,
)
from klab.experiments.recipe_export import publish_inventory


def create_recipe(
    root: Path,
    output: Path,
    *,
    profile: str,
    recipe_id: str,
    entrypoint: str,
    files: list[str],
    artifacts: Path | None = None,
    campaign: str | None = None,
) -> dict:
    root, output = Path(root).absolute(), Path(output).absolute()
    for component in (root, *root.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked recipe root: {component}")
    if not root.is_dir():
        raise ValueError("recipe root must be an existing directory")
    if len(files) > MAX_DOCUMENTS:
        raise ValueError("recipe inventory exceeds 64 documents")
    rows = []
    for member in files:
        role, separator, name = member.partition(":")
        if not separator:
            raise ValueError("each --file must be ROLE:RELATIVE_PATH")
        canonical_reference(name)
        data = _metadata_bytes(safe_path(root, name))
        rows.append(
            {"path": name, "role": role, "sha256": hashlib.sha256(data).hexdigest()}
        )
    document = {"id": recipe_id, "entrypoint": entrypoint, "files": rows}
    if profile == "preparation":
        from klab.experiments.preparation_recipe import PreparationRecipe

        model = PreparationRecipe
        document.update(
            portable_preparation_version=1,
            contracts={"campaign": 1, "run_config": 2, "tokenizer_config": 1},
        )
        if artifacts is not None or campaign is not None:
            raise ValueError(
                "preparation artifacts/connections belong in its native campaign"
            )
    else:
        identities = (
            {}
            if artifacts is None
            else parse_document(
                _metadata_bytes(artifacts).decode("utf-8"), source=artifacts
            )
        )
        document.update(
            contracts={"experiment_plan": 1, "run_config": 2}, artifacts=identities
        )
        if profile == "workflow":
            from klab.experiments.workflow_recipe import WorkflowRecipe

            model = WorkflowRecipe
            document.update(portable_workflow_version=1, campaign=campaign)
        elif profile == "matrix":
            model = PortableRecipe
            document["portable_recipe_version"] = 1
            if campaign is not None:
                raise ValueError("matrix profile has no campaign")
        else:
            raise ValueError("unknown recipe profile")
    descriptor = model.model_validate(document)
    raw = (json.dumps(descriptor.model_dump(mode="json"), indent=2) + "\n").encode()
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("recipe descriptor exceeds metadata document limit")
    # The existing loader enforces duplicate roles, exact native contracts,
    # bounded mapping documents, containment, digest integrity and self inventory.
    context = snapshot_inventory(root / "recipe.json", raw, descriptor.files)
    if campaign is not None and campaign not in context.documents:
        raise ValueError("campaign must be an inventoried metadata member")
    publish_inventory(output, "recipe.json", raw, context.documents)
    return {
        "format": "klab-recipe-authoring-v1",
        "output": str(output / "recipe.json"),
        "profile": profile,
        "files": len(rows) + 1,
        "executable": False,
        "validation": "inventory only; run the profile's native inspect with local bindings",
        "limitations": [
            "artifact identities are supplied declarations, not verified payloads",
            "no future hashes, runtime qualification, approval or execution are generated",
        ],
    }


def handle(args):
    try:
        result = create_recipe(
            args.root,
            args.output,
            profile=args.profile,
            recipe_id=args.id,
            entrypoint=args.entrypoint,
            files=args.file,
            artifacts=args.artifacts,
            campaign=args.campaign,
        )
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"klab: {error}") from None
    print(json.dumps(result, sort_keys=True, allow_nan=False))
