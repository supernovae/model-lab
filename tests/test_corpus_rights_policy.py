"""Rights decisions retain exact evidence and fail closed on ambiguity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from klab.corpus.rights import RightsPolicy, resolve_file_rights


def policy(**changes: object) -> RightsPolicy:
    values: dict[str, object] = {
        "training_eligibility": "eligible",
        "redistribution_mode": "redistributable_under_source_terms",
        "spdx_expression": "MIT",
        "license_references": ("https://example.test/LICENSE",),
        "notices": ("Retain copyright and license notice",),
    }
    values.update(changes)
    return RightsPolicy.model_validate(values)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # One expression per license class and SPDX expression form.
        ("MIT", "eligible"),
        ("GPL-2.0-only WITH Linux-syscall-note", "eligible_with_obligations"),
        ("GPL-2.0+", "eligible_with_obligations"),
        (
            "(GPL-2.0 WITH Linux-syscall-note) OR BSD-3-Clause",
            "eligible_with_obligations",
        ),
        ("CC-BY-4.0", "eligible_with_obligations"),
        ("MIT OR Apache-2.0", "eligible"),
        ("MIT AND GPL-3.0-only", "eligible_with_obligations"),
        ("CC-BY-NC-SA-4.0", "review_required"),
        ("GPL-2.0-only WITH Unknown-exception", "review_required"),
        ("Not-A-License", "review_required"),
        ("MIT OR Not-A-License", "review_required"),
        ("MIT OR", "review_required"),
    ],
)
def test_exact_spdx_and_license_classes(expression: str, expected: str) -> None:
    rights = resolve_file_rights(
        policy(spdx_expression=expression),
        "docs/chapter.txt",
        f"// SPDX-License-Identifier: {expression}\ntext".encode(),
    )
    assert rights.detected_spdx_expression == expression
    assert rights.training_eligibility == expected
    assert rights.weight_license_status == "separate_analysis_required"
    assert rights.model_dump(mode="json")["license_references"] == [
        "https://example.test/LICENSE"
    ]
    assert rights.redistribution_mode == (
        "review_required"
        if expected == "review_required"
        else "redistributable_under_source_terms"
    )


def test_prospective_training_keeps_unrecognized_license_obligations_separate() -> None:
    declaration = policy(
        spdx_expression="Publisher-Proprietary",
        redistribution_mode="metadata_reconstruction_only",
    )
    legacy = resolve_file_rights(declaration, "articles/paper.txt", b"Public paper")
    prospective = resolve_file_rights(
        declaration,
        "articles/paper.txt",
        b"Public paper",
        prospective_private_research=True,
    )
    assert legacy.training_eligibility == "review_required"
    assert prospective.training_eligibility == "eligible_with_obligations"
    assert prospective.redistribution_mode == "metadata_reconstruction_only"
    blocked = resolve_file_rights(
        policy(
            training_eligibility="ineligible",
            training_restriction={"kind": "prohibited", "basis": "No ML training"},
        ),
        "articles/paper.txt",
        b"Public paper",
        prospective_private_research=True,
    )
    assert blocked.training_eligibility == "ineligible"


@pytest.mark.parametrize(
    ("mode", "eligibility"),
    [
        ("metadata_reconstruction_only", "eligible"),
        ("review_required", "eligible"),
    ],
)
def test_training_and_redistribution_are_separate(mode: str, eligibility: str) -> None:
    rights = resolve_file_rights(
        policy(redistribution_mode=mode), "docs/a.py", b"print(1)"
    )
    assert rights.training_eligibility == eligibility
    assert rights.redistribution_mode == mode
    assert rights.detected_spdx_expression is None


def test_missing_notice_and_declared_restrictions_cannot_be_relaxed() -> None:
    without_notice = resolve_file_rights(policy(notices=()), "docs/a.py", b"print(1)")
    assert without_notice.training_eligibility == "eligible_with_obligations"
    assert (
        resolve_file_rights(
            policy(training_eligibility="review_required"),
            "docs/a.py",
            b"# SPDX-License-Identifier: MIT",
        ).training_eligibility
        == "review_required"
    )
    prohibited = resolve_file_rights(
        policy(
            training_eligibility="ineligible",
            training_restriction={
                "kind": "prohibited",
                "basis": "No ML training under source contract",
            },
        ),
        "docs/a.py",
        b"# SPDX-License-Identifier: MIT",
    )
    assert prohibited.training_eligibility == "ineligible"
    assert "No ML training under source contract" in prohibited.reason
    assert prohibited.model_dump(mode="json")["training_restriction"] == {
        "kind": "prohibited",
        "basis": "No ML training under source contract",
    }
    conditional = resolve_file_rights(
        policy(
            training_eligibility="review_required",
            training_restriction={
                "kind": "conditional",
                "basis": "Written approval required",
            },
        ),
        "docs/a.py",
        b"# SPDX-License-Identifier: MIT",
    )
    assert conditional.training_eligibility == "review_required"
    assert "Written approval required" in conditional.reason


def test_file_copyright_is_retained_with_source_notices() -> None:
    rights = resolve_file_rights(
        policy(),
        "docs/a.py",
        b"# SPDX-License-Identifier: MIT\n# SPDX-FileCopyrightText: 2024 Example Authors\nprint(1)",
    )
    assert rights.notices == (
        "Retain copyright and license notice",
        "# SPDX-FileCopyrightText: 2024 Example Authors",
    )
    go = resolve_file_rights(
        policy(spdx_expression="BSD-3-Clause"),
        "src/main.go",
        b"// Copyright 2025 The Go Authors. All rights reserved.\n// Use of this source code is governed by a BSD-style license.",
    )
    assert go.notices[-1] == "// Copyright 2025 The Go Authors. All rights reserved."


@pytest.mark.parametrize(
    "path",
    [
        "vendor/lib/a.py",
        "third_party/lib/a.py",
    ],
)
def test_boundaries_require_explicit_path_allowlist(path: str) -> None:
    blocked = resolve_file_rights(policy(), path, b"# SPDX-License-Identifier: MIT")
    assert blocked.training_eligibility == "review_required"
    assert blocked.redistribution_mode == "review_required"
    assert blocked.boundary_flags
    allowed = resolve_file_rights(
        policy(allowed_boundary_paths=(path,)), path, b"# SPDX-License-Identifier: MIT"
    )
    assert allowed.training_eligibility == "eligible"
    assert (
        resolve_file_rights(
            policy(allowed_boundary_paths=("vendor/lib/",)),
            "vendor/lib/a.py",
            b"# SPDX-License-Identifier: MIT",
        ).training_eligibility
        == "eligible"
    )


def test_file_override_and_conflicting_evidence() -> None:
    override = resolve_file_rights(
        policy(), "docs/a.py", b"# SPDX-License-Identifier: GPL-3.0-only"
    )
    assert override.detected_spdx_expression == "GPL-3.0-only"
    assert override.training_eligibility == "eligible_with_obligations"
    duplicate = resolve_file_rights(
        policy(),
        "docs/a.py",
        b"// SPDX-License-Identifier: MIT\n// SPDX-License-Identifier: GPL-3.0-only",
    )
    assert duplicate.training_eligibility == "review_required"
    assert "conflicting" in duplicate.reason
    assert (
        resolve_file_rights(
            policy(spdx_expression=None), "docs/a.py", b"# SPDX-License-Identifier: MIT"
        ).training_eligibility
        == "eligible"
    )
    assert (
        resolve_file_rights(
            policy(spdx_expression="Bogus-1.0"), "docs/a.py", b"text"
        ).training_eligibility
        == "review_required"
    )


def test_nested_rust_metadata_directory_and_group_precedence() -> None:
    metadata = {
        "license-metadata.json": {
            "files": {
                "type": "root",
                "children": [
                    {
                        "type": "directory",
                        "name": ".",
                        "license": {"spdx": "MIT OR Apache-2.0"},
                        "children": [
                            {
                                "type": "directory",
                                "name": "library/core/src/unicode",
                                "license": {"spdx": "Unicode-3.0"},
                                "children": [
                                    {
                                        "type": "file",
                                        "name": "mod.rs",
                                        "license": {"spdx": "MIT"},
                                    }
                                ],
                            },
                            {
                                "type": "directory",
                                "name": "gcc/testsuite",
                                "license": {"spdx": "ISC"},
                                "children": [
                                    {
                                        "type": "group",
                                        "files": ["special.c"],
                                        "directories": ["legacy"],
                                        "license": {
                                            "spdx": "GPL-2.0-only WITH Linux-syscall-note"
                                        },
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        }
    }
    source = policy(
        spdx_expression="MIT OR Apache-2.0",
        nested_metadata_path="license-metadata.json",
    )
    root = resolve_file_rights(source, "src/main.rs", b"code", nested_metadata=metadata)
    assert root.training_eligibility == "eligible"
    assert root.detected_spdx_expression == "MIT OR Apache-2.0"
    exception = resolve_file_rights(
        source, "gcc/testsuite/special.c", b"code", nested_metadata=metadata
    )
    assert exception.detected_spdx_expression == "GPL-2.0-only WITH Linux-syscall-note"
    assert exception.training_eligibility == "eligible_with_obligations"
    assert (
        resolve_file_rights(
            source, "gcc/testsuite/legacy/a.c", b"code", nested_metadata=metadata
        ).detected_spdx_expression
        == exception.detected_spdx_expression
    )
    assert (
        resolve_file_rights(
            source, "gcc/testsuite/other.c", b"code", nested_metadata=metadata
        ).detected_spdx_expression
        == "ISC"
    )
    assert (
        resolve_file_rights(
            source, "library/core/src/unicode/mod.rs", b"code", nested_metadata=metadata
        ).detected_spdx_expression
        == "MIT"
    )
    unknown = resolve_file_rights(
        source, "library/core/src/unicode/other.rs", b"code", nested_metadata=metadata
    )
    assert unknown.detected_spdx_expression == "Unicode-3.0"
    assert unknown.training_eligibility == "review_required"


def test_nested_metadata_conflicts_and_flat_directory_scope() -> None:
    source = policy(
        nested_metadata_path="third_party/license-metadata.json",
        allowed_boundary_paths=("third_party/",),
    )
    metadata = {
        "third_party/license-metadata.json": {
            "files": [
                {"path": "lib/", "license": "BSD-3-Clause"},
                {"path": "lib/special.py", "license": "MIT"},
            ]
        }
    }
    rights = resolve_file_rights(
        source, "third_party/lib/special.py", b"code", nested_metadata=metadata
    )
    assert rights.detected_spdx_expression == "MIT"
    assert rights.training_eligibility == "eligible"
    conflict = resolve_file_rights(
        source,
        "third_party/lib/special.py",
        b"# SPDX-License-Identifier: Apache-2.0",
        nested_metadata=metadata,
    )
    assert conflict.training_eligibility == "review_required"
    assert "conflicting file license evidence" in conflict.reason
    duplicate = {
        "license-metadata.json": {
            "files": [
                {"path": "docs/a.py", "license": "MIT"},
                {"path": "docs/a.py", "license": "ISC"},
            ]
        }
    }
    assert (
        resolve_file_rights(
            policy(), "docs/a.py", b"code", nested_metadata=duplicate
        ).training_eligibility
        == "review_required"
    )


def test_policy_and_path_validation() -> None:
    with pytest.raises(ValidationError, match="basis"):
        policy(
            training_eligibility="ineligible",
            training_restriction={"kind": "prohibited", "basis": " "},
        )
    with pytest.raises(ValidationError, match="prohibited"):
        policy(training_restriction={"kind": "prohibited", "basis": "No training"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        policy(unexpected=True)
    with pytest.raises(ValidationError, match="nested_metadata_path"):
        policy(nested_metadata_path="../license-metadata.json")
    with pytest.raises(ValueError, match="unsafe rights path"):
        resolve_file_rights(policy(), "../outside.py", b"code")
