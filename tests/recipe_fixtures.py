"""A miniature of a real corpus recipe: its shape, not its scale.

The recipe mirrors the structure of a production project (strata made of
Hugging Face bounded-shard sources with per-row licenses and no file-level SPDX
expression, Git sources selected by include pattern, a held-out source per major
stratum picked by `hash_modulus`, a vendored path, a source-level SPDX
expression and an exact duplicate across train and held-out) so that
integration failures that only appear with that shape show up in tests. Every
byte is synthetic and every transfer is served by `MockHub`; nothing touches the
network.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import random
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from huggingface_hub import hf_hub_url

from klab.canonical import canonical_json

RELEASE_POLICY = {
    "duplicates": {
        "heldout_overlap": "drop_train_copy",
        "keys": ["raw", "normalized"],
        "resolution": "chain_smallest_id",
        "rights_conflict": "error",
    },
    "documents": {
        "empty_text": "error",
        "exclusions": "none",
        "markdown_kind": "first_declared",
        "row_files": "buffered",
    },
    "rights": {"training_use": "allowed_unless_explicitly_prohibited"},
    "build": {"strategy": "disk"},
}
STRICT_ELIGIBILITY = "explicit_eligibility_required"

# stratum -> release share; the mixture template quotas follow these shares.
STRATA = {
    "web": 0.40,
    "python": 0.20,
    "systems_code": 0.15,
    "math": 0.10,
    "reference": 0.15,
}
SOURCE_STRATA = {
    "web_fineweb": "web",
    "web_fineweb_heldout": "web",
    "code_python": "python",
    "code_python_heldout": "python",
    "code_go": "systems_code",
    "code_shell": "systems_code",
    "code_systems_heldout": "systems_code",
    "math_finemath": "math",
    "math_finemath_heldout": "math",
    "ref_mdn": "reference",
    "ref_python_docs": "reference",
    "ref_tldr_pages": "reference",
}
HELDOUT = {name for name in SOURCE_STRATA if name.endswith("_heldout")}

_WORDS = {
    "web": [
        "school",
        "learning",
        "teacher",
        "student",
        "reading",
        "science",
        "lesson",
        "history",
        "river",
        "garden",
        "market",
    ],
    "math": [
        "integral",
        "matrix",
        "prime",
        "theorem",
        "lemma",
        "vector",
        "series",
        "proof",
        "angle",
        "sum",
        "limit",
    ],
    "reference": [
        "request",
        "response",
        "header",
        "method",
        "module",
        "function",
        "return",
        "value",
        "default",
        "option",
    ],
    "code": [
        "buffer",
        "parse",
        "token",
        "render",
        "widget",
        "stream",
        "cache",
        "index",
        "queue",
        "handler",
    ],
}
_PERMISSIVE = ("MIT", "Apache-2.0", "BSD-3-Clause")


def selected_indices(
    revision: str, path: str, rows: int, modulus: int, remainders: list[int]
) -> list[int]:
    """Row indices a bounded shard keeps, by the acquisition's own hash rule."""
    return [
        index
        for index in range(rows)
        if int(hashlib.sha256(canonical_json([revision, path, index])).hexdigest(), 16)
        % modulus
        in remainders
    ]


def _prose(rng: random.Random, vocabulary: list[str], tag: str, sentences: int = 5):
    return " ".join(
        f"{rng.choice(vocabulary).capitalize()} {rng.choice(vocabulary)} "
        f"{rng.choice(vocabulary)} {tag} {rng.choice(vocabulary)}."
        for _ in range(sentences)
    )


def _code(rng: random.Random, language: str, tag: str) -> str:
    name = rng.choice(_WORDS["code"])
    if language == "Python":
        return (
            f"def {name}_{tag}(value):\n    # {rng.choice(_WORDS['code'])} step\n"
            f"    return value + {rng.randrange(100)}\n"
        )
    if language == "Go":
        return f"func {name.capitalize()}{tag}(n int) int {{\n\treturn n * {rng.randrange(9) + 1}\n}}\n"
    return f"{name}_{tag}() {{\n  echo {rng.choice(_WORDS['code'])} {rng.randrange(50)}\n}}\n"


class Shard:
    """One bounded shard: its path, its rows, its bytes and their SHA-256."""

    def __init__(self, path: str, rows: list[dict], *, parquet: bool = False):
        self.path = path
        self.rows = rows
        if parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            buffer = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(rows), buffer)
            self.body = buffer.getvalue()
        else:
            raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
            self.body = gzip.compress(raw, mtime=0)
        self.sha256 = hashlib.sha256(self.body).hexdigest()


class MockHub:
    """Serves pinned shard bodies and the Hub metadata acquisition cross-checks."""

    def __init__(self) -> None:
        self.shards: dict[str, Shard] = {}
        self.configs: dict[str, set[tuple[str, str]]] = {}
        self.requests: list[str] = []

    def register(self, repo: str, revision: str, shard: Shard, config: str) -> None:
        url = hf_hub_url(
            repo_id=repo, filename=shard.path, repo_type="dataset", revision=revision
        )
        self.shards[url] = shard
        self.configs.setdefault(repo, set()).add((config, "train"))

    @property
    def bodies(self) -> int:
        return sum(1 for url in self.requests if "/resolve/" in url)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hub = self

        class Response(io.BytesIO):
            status = 200

            def __init__(self, body: bytes, url: str):
                super().__init__(body)
                self.url = url
                self.headers = {"Content-Length": str(len(body))}

        class Opener:
            def open(self, request: urllib.request.Request, timeout: float = 30):
                url = request.full_url
                hub.requests.append(url)
                if url in hub.shards:
                    return Response(hub.shards[url].body, url)
                if "/paths-info/" in url:
                    repo = url.split("/api/datasets/")[1].split("/paths-info/")[0]
                    path = urllib.parse.parse_qs(request.data.decode())["paths"][0]
                    shard = next(
                        item
                        for known, item in hub.shards.items()
                        if f"/{repo}/resolve/" in known and item.path == path
                    )
                    body = [
                        {
                            "path": path,
                            "type": "file",
                            "size": len(shard.body),
                            "lfs": {"sha256": shard.sha256, "size": len(shard.body)},
                        }
                    ]
                    return Response(json.dumps(body).encode(), url)
                if "datasets-server.huggingface.co/splits" in url:
                    repo = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)[
                        "dataset"
                    ][0]
                    entries = [
                        {"dataset": repo, "config": config, "split": split}
                        for config, split in sorted(hub.configs[repo])
                    ]
                    return Response(json.dumps({"splits": entries}).encode(), url)
                raise AssertionError(f"unexpected request: {url}")

        monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "snapshot")
    return _git(repo, "rev-parse", "HEAD")


def _hf_rights(*, spdx: str | None) -> dict:
    rights = {
        "training_eligibility": "eligible_with_obligations",
        "redistribution_mode": "metadata_reconstruction_only",
        "weight_license_status": "separate_analysis_required",
        "license_references": ["https://example.org/license"],
        "notices": ["Synthetic fixture; no upstream rights claim."],
    }
    if spdx:
        rights["spdx_expression"] = spdx
    return rights


class Recipe:
    """A written recipe plus the handles tests use to change and inspect it."""

    def __init__(self, root: Path, hub: MockHub, expected: dict) -> None:
        self.root = root
        self.hub = hub
        self.expected = expected
        self.corpus = root / "corpus"

    @property
    def project(self) -> Path:
        return self.corpus / "corpus.yaml"

    def edit(self, name: str, change: Callable[[dict], None]) -> None:
        path = self.corpus / "sources" / f"{name}.yaml"
        value = yaml.safe_load(path.read_text())
        change(value)
        path.write_text(yaml.safe_dump(value, sort_keys=True))


def _heldout_shard(
    revision: str,
    path: str,
    make: Callable[[int], dict],
    leading: list[dict],
    *,
    total: int = 24,
    modulus: int = 3,
) -> tuple[Shard, int]:
    """A shard whose hash-selected rows are the leading documents, then `make(i)`.

    Every row the acquisition keeps is deliberate; the rest is never retained.
    Returns the shard and the number of rows selected.
    """
    picked = selected_indices(revision, path, total, modulus, [0])
    assert len(picked) >= len(leading), "enlarge the held-out fixture shard"
    rows = [
        {"text": f"unselected filler {index}", "metadata": {}} for index in range(total)
    ]
    for position, index in enumerate(picked):
        rows[index] = leading[position] if position < len(leading) else make(position)
    return Shard(path, rows), len(picked)


def write_recipe(root: Path, *, strict_rights: bool = False) -> Recipe:
    """Write the miniature recipe under `root` and return it with its mock hub.

    By default the release allows training unless a restriction is recorded, as
    a private-research recipe does. `strict_rights` requires explicit
    eligibility evidence instead, which Hugging Face shards with per-row
    licenses and no file-level SPDX expression cannot give.
    """
    rng = random.Random(20260101)
    hub = MockHub()
    corpus = root / "corpus"
    (corpus / "sources").mkdir(parents=True)
    (corpus / "transforms").mkdir()
    # What the design retains and drops, checked against the built release.
    expected: dict = {"kept": {}, "contaminated": 0}
    sources: dict[str, dict] = {}
    restriction = (
        {} if strict_rights else {"explicit_training_restriction": "none_found"}
    )

    def web_row(index: int, tag: str, vocabulary: list[str], language: str = "en"):
        url = f"https://example.org/{tag}/{index}"
        return {
            "text": _prose(rng, vocabulary, f"{tag}{index}"),
            "url": url,
            "metadata": {"url": url, "language": language},
        }

    def code_row(
        index: int, tag: str, language: str, *, license: str | None = None, path=None
    ):
        extension = {"Python": "py", "Go": "go", "Shell": "sh"}[language]
        return {
            "text": _code(rng, language, f"{tag}{index}"),
            "metadata": {
                "repo_name": f"org/{tag}-{index % 3}",
                "language": language,
                "license": license or _PERMISSIVE[index % 3],
                "path": path or f"src/{tag}{index}.{extension}",
            },
        }

    def hf_source(
        name: str,
        repo: str,
        config: str,
        shards: list[Shard],
        *,
        qualification: dict,
        spdx: str | None,
        domain: str,
        modulus: int = 1,
        scanned: int = 64,
    ) -> None:
        revision = hashlib.sha1(repo.encode()).hexdigest()
        for shard in shards:
            hub.register(repo, revision, shard, config)
        sources[name] = {
            "id": name,
            "kind": "huggingface_dataset",
            "canonical_uri": f"https://huggingface.co/datasets/{repo}",
            "revision": revision,
            "license": spdx or "Per-row permissive licenses; no dataset-level license",
            "license_url": "https://example.org/license",
            **restriction,
            "rights": _hf_rights(spdx=spdx),
            "domains": [domain],
            "document_kinds": [
                "code" if qualification["profile"] == "code" else "prose"
            ],
            "source_family": name,
            "acquisition": {
                "config": config,
                "split": "train",
                "text_field": "text",
                "max_rows": 10_000,
                "max_bytes": max(1 << 20, 8 * sum(len(shard.body) for shard in shards)),
                "bounded_shards": [
                    {
                        "path": shard.path,
                        "expected_sha256": shard.sha256,
                        "max_shard_bytes": len(shard.body),
                        "max_scanned_rows": scanned,
                        "hash_modulus": modulus,
                        "hash_remainders": [0],
                        "declared_config": config,
                        "declared_split": "train",
                    }
                    for shard in shards
                ],
                "qualification": qualification,
            },
        }

    def revision(repo: str) -> str:
        return hashlib.sha1(repo.encode()).hexdigest()

    web_q = {
        "profile": "web_text",
        "family_fields": ["metadata.url"],
        "language_field": "metadata.language",
        "allowed_languages": ["en"],
    }

    def code_q(language: str) -> dict:
        return {
            "profile": "code",
            "family_fields": ["metadata.repo_name"],
            "language_field": "metadata.language",
            "allowed_languages": [language],
            "license_field": "metadata.license",
            "license_labels": {label: label for label in _PERMISSIVE},
            "repository_field": "metadata.repo_name",
            "path_field": "metadata.path",
        }

    def count(name: str, kept: int) -> None:
        expected["kept"][name] = kept

    # Web: Parquet shards and a source-level SPDX expression, as the real dataset.
    web = [web_row(i, "web", _WORDS["web"]) for i in range(8)]
    # The same page crawled twice with different text: one canonical origin,
    # two documents. The release declares no canonical_origin duplicate key, so
    # both stay, and verification must not demand they share a representative.
    web[1] = {**web_row(1, "web", _WORDS["web"]), "url": web[0]["url"]}
    hf_source(
        "web_fineweb",
        "fixture/web-corpus",
        "default",
        [
            Shard("data/train-00000.parquet", web[:4], parquet=True),
            Shard("data/train-00001.parquet", web[4:], parquet=True),
        ],
        qualification=web_q,
        spdx="ODC-By-1.0",
        domain="web",
    )
    count("web_fineweb", 8)
    web_heldout, selected = _heldout_shard(
        revision("fixture/web-corpus"),
        "data/train-00039.parquet",
        lambda i: web_row(100 + i, "webheld", _WORDS["web"]),
        [],
    )
    web_heldout = Shard(web_heldout.path, web_heldout.rows, parquet=True)
    hf_source(
        "web_fineweb_heldout",
        "fixture/web-corpus",
        "default",
        [web_heldout],
        qualification=web_q,
        spdx="ODC-By-1.0",
        domain="web",
        modulus=3,
        scanned=64,
    )
    count("web_fineweb_heldout", selected)

    # Python: per-row licenses, no file-level SPDX expression anywhere.
    code_repo = "fixture/code-corpus"
    python = [code_row(i, "py", "Python") for i in range(14)]
    hf_source(
        "code_python",
        code_repo,
        "default",
        [
            Shard("stack-0001.json.gz", python[:7]),
            Shard("stack-0002.json.gz", python[7:]),
        ],
        qualification=code_q("Python"),
        spdx=None,
        domain="python",
    )
    count("code_python", 12)
    duplicates = [
        # The same document as a training row, byte for byte.
        {**python[0], "metadata": {**python[0]["metadata"], "repo_name": "org/mirror"}},
        # The same document after Unicode and line-ending normalization.
        {
            **python[1],
            "text": python[1]["text"].replace("\n", "\r\n"),
            "metadata": {**python[1]["metadata"], "repo_name": "org/mirror"},
        },
    ]
    python_heldout, selected = _heldout_shard(
        revision(code_repo),
        "stack-0099.json.gz",
        lambda i: code_row(200 + i, "pyheld", "Python"),
        duplicates,
    )
    hf_source(
        "code_python_heldout",
        code_repo,
        "default",
        [python_heldout],
        qualification=code_q("Python"),
        spdx=None,
        domain="python",
        modulus=3,
    )
    count("code_python_heldout", selected)
    expected["contaminated"] = 2

    # Systems code: two more per-row-license sources and a held-out one.
    go = [code_row(i, "go", "Go") for i in range(7)]
    hf_source(
        "code_go",
        code_repo,
        "default",
        [Shard("stack-0010.json.gz", go)],
        qualification=code_q("Go"),
        spdx=None,
        domain="systems_code",
    )
    count("code_go", 7)
    shell = [code_row(i, "sh", "Shell") for i in range(5)]
    hf_source(
        "code_shell",
        code_repo,
        "default",
        [Shard("stack-0011.json.gz", shell)],
        qualification=code_q("Shell"),
        spdx=None,
        domain="systems_code",
    )
    count("code_shell", 5)
    systems_heldout, selected = _heldout_shard(
        revision(code_repo),
        "stack-0098.json.gz",
        lambda i: code_row(300 + i, "goheld", "Go"),
        [],
    )
    hf_source(
        "code_systems_heldout",
        code_repo,
        "default",
        [systems_heldout],
        qualification=code_q("Go"),
        spdx=None,
        domain="systems_code",
        modulus=3,
    )
    count("code_systems_heldout", selected)

    # Math: a second repository, so the splits endpoint lists another config.
    math_repo = "fixture/math-corpus"
    math = [web_row(i, "math", _WORDS["math"]) for i in range(6)]
    hf_source(
        "math_finemath",
        math_repo,
        "plus",
        [Shard("plus/train-00000.json.gz", math)],
        qualification=web_q,
        spdx="ODC-By-1.0",
        domain="math",
    )
    count("math_finemath", 6)
    math_heldout, selected = _heldout_shard(
        revision(math_repo),
        "plus/train-00031.json.gz",
        lambda i: web_row(400 + i, "mathheld", _WORDS["math"]),
        [],
    )
    hf_source(
        "math_finemath_heldout",
        math_repo,
        "plus",
        [math_heldout],
        qualification=web_q,
        spdx="ODC-By-1.0",
        domain="math",
        modulus=3,
    )
    count("math_finemath_heldout", selected)

    # Reference: Git sources selected by include pattern, each with a
    # source-level SPDX expression; one carries a vendored directory.
    def front(title: str, body: str) -> str:
        return (
            f"---\ntitle: {title}\nslug: {title.lower()}\n---\n\n# {title}\n\n{body}\n"
        )

    def git_source(name: str, files: dict[str, str], include: list[str], spdx: str):
        repo = root / "git" / name
        commit = _commit(repo, files)
        sources[name] = {
            "id": name,
            "kind": "git",
            "canonical_uri": str(repo),
            "revision": commit,
            "license": spdx,
            "license_url": "https://example.org/license",
            **restriction,
            "rights": {
                **_hf_rights(spdx=spdx),
                "license_references": ["https://example.org/license"],
            },
            "domains": ["reference"],
            "document_kinds": ["markdown"],
            "source_family": name,
            "acquisition": {"include": include, "max_bytes": 1 << 20},
        }

    def reference(n: int, tag: str) -> str:
        return _prose(rng, _WORDS["reference"], f"{tag}{n}", sentences=6)

    git_source(
        "ref_mdn",
        {
            f"files/en-us/page{n}.md": front(f"Page{n}", reference(n, "mdn"))
            for n in range(4)
        },
        ["files/en-us/*.md"],
        "CC-BY-SA-2.5",
    )
    count("ref_mdn", 4)
    git_source(
        "ref_python_docs",
        {
            f"Doc/topic{n}.rst": f"Topic {n}\n=======\n\n{reference(n, 'py')}\n"
            for n in range(3)
        },
        ["Doc/*.rst"],
        "PSF-2.0",
    )
    count("ref_python_docs", 3)
    git_source(
        "ref_tldr_pages",
        {
            **{
                f"pages/common/tool{n}.md": f"# tool{n}\n\n{reference(n, 'tldr')}\n"
                for n in range(3)
            },
            "pages/vendor/bundled.md": "# bundled\n\nA vendored page.\n",
        },
        ["pages/*.md"],
        "CC-BY-4.0",
    )
    count("ref_tldr_pages", 3)

    for name, source in sources.items():
        (corpus / "sources" / f"{name}.yaml").write_text(yaml.safe_dump(source))
    (corpus / "transforms/lm.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "mini_lm",
                "version": "1",
                "kind": "lm_text",
                "parameters": {},
                "inputs": sorted(sources),
            }
        )
    )
    (corpus / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "unit": "source_document_family",
                "family_key": "source_family",
                "assignments": {
                    name: "validation" if name in HELDOUT else "train"
                    for name in sorted(sources)
                },
            }
        )
    )
    policy = copy.deepcopy(RELEASE_POLICY)
    if strict_rights:
        policy["rights"]["training_use"] = STRICT_ELIGIBILITY
    (corpus / "release.yaml").write_text(
        yaml.safe_dump(
            {
                "publication_mode": "metadata_reconstruction_only",
                "policy": policy,
                "mixture": dict(STRATA),
                "lm": {"selected": True, "training_splits": ["train", "validation"]},
                "chat": {"selected": False, "training_splits": []},
                "keep_nonredistributable_local": True,
                "normalizer": {
                    "structure_cleaning": True,
                    "front_matter_extra_keys": {},
                    "gutenberg_boilerplate": [],
                },
            }
        )
    )
    shard_bytes = sum(len(item.body) for item in hub.shards.values())
    (corpus / "corpus.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "mini-recipe",
                "sources": [f"sources/{name}.yaml" for name in sorted(sources)],
                "transforms": ["transforms/lm.yaml"],
                "splits": "splits.yaml",
                "release": "release.yaml",
                "transport_budget": {
                    "attempt_id": "mini-recipe-acquire-1",
                    "max_source_body_bytes": 8 * shard_bytes,
                    "max_metadata_body_bytes": 1 << 20,
                    "max_transfers": 100,
                    "max_retries_per_shard": 2,
                    "max_wall_seconds": 3600,
                    "max_disk_bytes": 1 << 30,
                },
            }
        )
    )
    return Recipe(root, hub, expected)


def write_training_files(recipe: Recipe, *, vocab_size: int) -> None:
    """The recipe's model, optimizer, pretraining and corpus-side training files.

    They keep the structure of the production recipe (sections composed with a
    local paths file, a stratified tokenizer sampling plan, a mixture template
    with `${...}` placeholders and an export base config) at a size a CPU trains
    in seconds.
    """
    root, corpus = recipe.root, recipe.corpus

    def write(path: Path, value: dict) -> None:
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    write(
        root / "model.yaml",
        {
            "vocab_size": vocab_size,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "ffn_dim": 64,
            "max_seq_len": 32,
            "rms_norm_eps": 1.0e-06,
            "tie_embeddings": True,
            "ffn": "dense",
            # Engram n-gram memory after block 0, as the production model has
            # after two of its blocks, with distinct prime table sizes.
            "memory": "ngram",
            "memory_injection": "after_block",
            "memory_block_indices": [0],
            "memory_ngram_size": 3,
            "memory_ngram_orders": [2, 3],
            "memory_hash_heads": 2,
            "memory_dim": 8,
            "memory_conv_kernel_size": 4,
            "memory_hash_seed": 42,
            "memory_table_size": 31,
            "memory_table_sizes": [31, 37, 41, 43],
            "memory_padding_token_id": 3,
            "memory_independent_init": True,
        },
    )
    write(root / "attention.yaml", {"kind": "dense", "implementation": "sdpa"})
    write(
        root / "optimizer.yaml",
        {
            "name": "adamw",
            "peak": 0.003,
            "floor": 0.0003,
            "warmup_steps": 1,
            "decay_steps": 2,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "eps": 1.0e-08,
            "memory_table_lr_multiplier": 5.0,
            "memory_table_weight_decay": 0.0,
        },
    )
    write(
        root / "pretrain.yaml",
        {
            "schema_version": 2,
            "name": "mini-pretrain",
            "seed": 42,
            "training": {
                "seq_len": 16,
                "micro_batch_size": 1,
                "gradient_accumulation": 1,
                "max_steps": 2,
                "max_tokens": 32,
                "grad_clip_norm": 1.0,
                "deterministic": True,
            },
            "evaluation": {"every_steps": None, "every_tokens": 16, "max_batches": 2},
            "checkpoint": {"every_steps": 2, "keep_periodic": False},
            "logging": {"every_steps": 1},
            "runtime": {"engine": "pytorch", "backend": "cpu", "precision": "fp32"},
            "staging": {"smoke_steps": 2, "warmup_steps": 5},
        },
    )
    write(
        corpus / "export-base.yaml",
        {
            "schema_version": 2,
            "name": "mini-export-base",
            "seed": 42,
            "model": {
                "vocab_size": vocab_size,
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 2,
                "ffn_dim": 64,
                "max_seq_len": 32,
            },
            "tokenizer": {"path": "./unbound/tokenizer.json"},
            "dataset": {
                "source": "local_text",
                "license": "Unbound; the export supplies the release's provenance",
                "train_path": "./unbound/train.jsonl",
                "validation_path": "./unbound/validation.jsonl",
                "cache_dir": "./unbound/cache",
                "train_max_documents": 100,
                "validation_max_documents": 100,
                "train_max_tokens": 1048576,
                "validation_max_tokens": 65536,
            },
            "attention": {"kind": "dense", "implementation": "sdpa"},
            "training": {
                "seq_len": 16,
                "micro_batch_size": 1,
                "gradient_accumulation": 1,
                "max_steps": 2,
                "max_tokens": 32,
            },
            "optimizer": {
                "name": "adamw",
                "peak": 0.003,
                "floor": 0.0003,
                "warmup_steps": 1,
            },
            "logging": {"root_dir": "./unbound/runs"},
        },
    )
    write(
        corpus / "tokenizer-sampling.yaml",
        {
            "source_strata": dict(SOURCE_STRATA),
            "byte_weights": {
                stratum: int(share * 100) for stratum, share in STRATA.items()
            },
            "max_fit_bytes": 2000,
            "audit_documents_per_stratum": 2,
            "max_audit_bytes": 65536,
        },
    )
    write(
        corpus / "mixture.template.yaml",
        {
            "schema_version": 1,
            "release_path": "${RELEASE_PATH}",
            "tokenizer_config": "${TOKENIZER_CONFIG}",
            "family_inventory": "${FAMILY_INVENTORY}",
            "source_strata": dict(SOURCE_STRATA),
            "target_quotas": {
                stratum: int(share * 1000) for stratum, share in STRATA.items()
            },
            "seed": 42,
            "max_exposures": 2,
            "tokenizer_origin_release_id": "${RELEASE_ID}",
        },
    )
