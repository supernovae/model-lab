"""Authenticated frozen document assignments with private operation-owned lookup."""

import shutil
from collections.abc import Mapping
from pathlib import Path

from klab.corpus.bounded_metadata import current_metadata_store
from klab.corpus.split_lookup import AssignmentLookup, stamp
from klab.training.manifest import sha256_file


def assignment_path(directory, reference):
    from klab.corpus.project import SplitAssignmentsReference

    reference = SplitAssignmentsReference.model_validate(reference).model_dump(
        mode="json"
    )
    path = Path(directory) / ("assignments-" + reference["sha256"] + ".jsonl")
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("missing or symlinked split assignments")
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("split assignment digest mismatch")
    return path


class Assignments(Mapping):
    def __init__(self, path, reference):
        store = current_metadata_store()
        key = (str(path), reference["sha256"], reference["rows"])
        lookup = store._assignment_lookups.get(key)
        if lookup is not None and lookup.source_stamp != stamp(path):
            raise ValueError("split assignments changed during lookup")
        if lookup is None:
            lookup = AssignmentLookup(Path(store.temporary.name), path, reference)
            store._assignment_lookups[key] = lookup
        self.index = lookup

    def __getitem__(self, key):
        return self.index[key]

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)


def split_policy(project):
    policy = project.splits.model_dump(mode="json")
    reference = policy.get("assignment_file")
    if reference is not None:
        from klab.corpus.project import project_path

        directory = project_path(project.root, project.config.splits).parent
        policy["assignments"] = Assignments(
            assignment_path(directory, reference), reference
        )
    return policy


def copy_assignments(project, destination):
    reference = project.splits.model_dump(mode="json").get("assignment_file")
    if reference is None:
        return
    from klab.corpus.project import project_path

    source = assignment_path(
        project_path(project.root, project.config.splits).parent, reference
    )
    target = Path(destination) / source.name
    with target.open("xb") as output, source.open("rb") as original:
        shutil.copyfileobj(original, output)
    assignment_path(destination, reference)


def verify_document_assignments(root, identity, documents):
    reference = identity["split"].get("assignment_file")
    if reference is None:
        return
    assignments = Assignments(assignment_path(root, reference), reference)
    for doc in documents:
        if assignments.get(doc["document_id"]) != doc["split"]:
            raise ValueError("document differs from frozen split assignment")
