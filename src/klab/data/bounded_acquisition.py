"""Pinned whole-file HTTP acquisition with durable conservative reservations.

Response-body accounting is not wire accounting. Cache content reservations are
finite; aggregate owned storage is observed, never advertised as a hard quota.
No SDK streaming or global Hub cache is used by this profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.acquisition import (
    _ManualRedirect,
    _read_metadata_body,
    _read_with_deadline,
    _response_length,
)
from klab.corpus.transport_budget import TransportBudget
from klab.data.acquisition_scope import current_transport
from klab.engram.packs import _rename_noreplace
from klab.result_publication import sync_directory


class AcquisitionInterrupted(RuntimeError):
    """A durable attempt may be resumed; its reservations are never refunded."""


def validate_profile(lock, spec):
    if lock.get("format") != "klab-dataset-lock-v1":
        raise ValueError("acquisition requires native source lock v1")
    source, inventory = lock["source"], lock["provenance"]
    if (source["kind"], source["repo_id"], source["revision"], lock["lock_sha256"]) != (
        "huggingface",
        spec.repo_id,
        spec.revision,
        spec.lock_sha256,
    ):
        raise ValueError("acquisition authority differs from source lock")
    if inventory["loader"] != "parquet":
        raise ValueError("bounded acquisition initially supports scalar Parquet only")
    options = inventory.get("loader_options", {})
    # The lock producer records packaged-loader defaults, including explicit nulls.
    # Accept only inactive options; filtering/projection/scanner customization remain
    # unsupported, and the original provenance (and its lock identity) stays intact.
    inactive = {"columns", "filters", "fragment_scan_options"}
    if set(options) - {"batch_size", "features", "on_bad_files"} - inactive or any(
        options.get(name) is not None for name in inactive
    ):
        raise ValueError(
            "unsupported bounded Parquet loader option (including filters/projection)"
        )
    if options.get("on_bad_files", "error") != "error":
        raise ValueError("bounded Parquet cannot skip malformed files")
    if options.get("batch_size") is not None and (
        type(options["batch_size"]) is not int or options["batch_size"] <= 0
    ):
        raise ValueError("invalid Parquet batch size")
    _validate_features(options.get("features"))
    for rows in inventory["files"].values():
        for row in rows:
            if not row["path"].endswith(".parquet") or row["size"] <= 0:
                raise ValueError("bounded acquisition requires nonempty Parquet files")


def _validate_features(features):
    if features is not None:
        allowed = {
            "string",
            "large_string",
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
        }
        if not isinstance(features, dict) or any(
            not isinstance(value, dict)
            or set(value) != {"_type", "dtype"}
            or value["_type"] != "Value"
            or value["dtype"] not in allowed
            for value in features.values()
        ):
            raise ValueError(
                "media, external-file and non-scalar features are unavailable"
            )


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"acquisition requires regular nonsymlink file: {path}")


class DatasetTransport(TransportBudget):
    """Reuse native durable reservation/retry transactions, without a Forge Project."""

    def __init__(
        self, lock, spec, cache: Path, output: Path, work_dir: Path, binding: dict
    ):
        validate_profile(lock, spec)
        self.lock, self.spec = lock, spec
        self.cache, self.output, self.work_dir = cache, output, work_dir
        self.publication = output.with_name("." + output.name + ".work") / "publication"
        self.binding = hashlib.sha256(
            canonical_json(
                {
                    "format": "klab-dataset-transport-v1",
                    "lock": lock["lock_sha256"],
                    "admission": spec.model_dump(mode="json"),
                    "context": binding,
                    "cache": str(cache),
                    "output": str(output),
                }
            )
        ).hexdigest()
        self.root = work_dir / "execution" / "acquisition" / self.binding
        self.path = self.root / "transport.sqlite"
        self.cancel = self.root / "CANCEL"
        self.snapshot_limit = spec.max_observed_owned_bytes
        self.snapshot_roots = [(self.output, self.publication)]

    def validate_locations(self):
        # Reject existing redirects before creating directories or opening guards.
        # As elsewhere in native storage, this is not an adversarial filesystem sandbox.
        for path in (self.root, self.cache, self.output, self.publication.parent):
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError("symlink in acquisition storage location")
        for name in (
            "guard",
            "transport.sqlite",
            "transport.sqlite-journal",
            "transport.sqlite-wal",
            "transport.sqlite-shm",
            "receipt.json",
            "verification",
        ):
            if (self.root / name).is_symlink():
                raise ValueError("symlink in acquisition operational storage")

    def initialize(self):
        self.validate_locations()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        owner = self.cache / ".owner"
        if owner.exists():
            _regular(owner)
            if owner.read_text() != self.binding:
                raise ValueError("cache belongs to another acquisition admission")
        else:
            if any(self.cache.iterdir()):
                raise ValueError("acquisition cache lacks ownership")
            with owner.open("x") as handle:
                handle.write(self.binding)
                handle.flush()
                os.fsync(handle.fileno())
            sync_directory(self.cache)
        if not self.path.exists():
            # Existing payloads without their ledger cannot reset spent authority.
            if (
                any(p.name != ".owner" for p in self.cache.iterdir())
                or self.output.exists()
                or self.publication.parent.exists()
            ):
                raise ValueError(
                    "acquisition ledger missing for existing payload/journal"
                )
            with self.path.open("xb"):
                pass
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute(
                    "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)"
                )
                now = time.time()
                state = {
                    "binding": self.binding,
                    "created_at": now,
                    "deadline": now + self.spec.max_wall_seconds,
                    "source_charged": 0,
                    "source_actual": 0,
                    "metadata_charged": 0,
                    "metadata_actual": 0,
                    "transfers": [],
                    "failures": [],
                    "requests": 0,
                    "scanned_rows": 0,
                    "scanned_rows_observed": 0,
                    "verification_bytes": 0,
                    "cache_hits": 0,
                    "peak_owned_bytes": 0,
                    "peak_owned_inodes": 0,
                }
                db.execute("INSERT INTO state VALUES (1, ?)", (json.dumps(state),))
        _regular(self.path)
        self._read()
        self.observe()

    # This admission schema deliberately has no corpus feasibility gate.
    @property
    def _total_body_limit(self):
        return None

    @property
    def _initial_body_limit(self):
        return None

    def _validate_state(self, state):
        super()._validate_state(state)
        for field, maximum in (
            ("requests", self.spec.max_requests),
            ("scanned_rows", self.spec.max_scanned_rows),
            ("verification_bytes", self.spec.max_verification_bytes),
            ("scanned_rows_observed", state["scanned_rows"]),
            ("cache_hits", 2**63 - 1),
            ("peak_owned_bytes", 2**63 - 1),
            ("peak_owned_inodes", 2**63 - 1),
        ):
            if type(state.get(field)) is not int or not 0 <= state[field] <= maximum:
                raise ValueError("invalid acquisition operational counter")

    def consume(self, field, amount, maximum):
        def update(state):
            if amount < 0 or amount > maximum - state[field]:
                raise ValueError(f"acquisition allowance exhausted: {field}")
            state[field] += amount

        self._mutate(update)

    def remaining_seconds(self):
        if self.cancel.exists():
            raise AcquisitionInterrupted(
                "acquisition cancellation requested; journal retained"
            )
        return super().remaining_seconds()

    def observe(self):
        self.remaining_seconds()
        size = count = 0
        for path in self.work_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in owned acquisition storage")
            count += 1
            if path.is_file():
                size += path.stat().st_size

        def update(state):
            state["peak_owned_bytes"] = max(state["peak_owned_bytes"], size)
            state["peak_owned_inodes"] = max(state["peak_owned_inodes"], count)

        self._mutate(update)
        if (
            size > self.spec.max_observed_owned_bytes
            or count > self.spec.max_observed_owned_inodes
        ):
            raise ValueError("observed owned-storage stop exceeded (not a hard quota)")

    def _cache_inventory(self):
        size = count = 0
        for path in self.cache.iterdir():
            _regular(path)
            if path.name == ".owner":
                if path.read_text() != self.binding:
                    raise ValueError("cache ownership changed")
                continue
            if (
                path.suffix not in {".bin", ".partial"}
                or len(path.stem) != 64
                or any(c not in "0123456789abcdef" for c in path.stem)
            ):
                raise ValueError("unexpected acquisition cache member")
            size += path.stat().st_size
            count += 1
        if size > self.spec.max_cache_bytes or count > self.spec.max_cache_files:
            raise ValueError("acquisition cache allowance exceeded")
        return size, count

    def _url(self, row):
        return f"https://huggingface.co/datasets/{quote(self.spec.repo_id, safe='/')}/resolve/{self.spec.revision}/{quote(row['path'], safe='/')}"

    def _open(self, url):
        from klab.hf_auth import hub_auth_kwargs

        auth = hub_auth_kwargs()
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "KybernModelLab-Bounded-Dataset/1",
        }
        if auth.get("token"):
            headers["Authorization"] = "Bearer " + auth["token"]
        opener = urllib.request.build_opener(_ManualRedirect())
        for _ in range(6):
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.username
                or parsed.password
                or parsed.port not in (None, 443)
            ):
                raise ValueError("acquisition redirect requires ordinary HTTPS")
            if not parsed.hostname or not (
                parsed.hostname == "huggingface.co"
                or parsed.hostname.endswith(".huggingface.co")
                or parsed.hostname == "hf.co"
                or parsed.hostname.endswith(".hf.co")
                or parsed.hostname.endswith(".xethub.hf.co")
            ):
                raise ValueError(
                    "acquisition redirect origin is not an admitted Hub delivery host"
                )
            if parsed.hostname != "huggingface.co":
                headers.pop("Authorization", None)
            self.consume("requests", 1, self.spec.max_requests)
            try:
                response = opener.open(
                    urllib.request.Request(url, headers=headers),
                    timeout=min(30, self.remaining_seconds()),
                )
            except HTTPError as error:
                with error:
                    _read_metadata_body(error, self)
                raise
            if (
                response.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                response.close()
                raise ValueError("HTTP content encoding is unavailable")
            if response.status not in {301, 302, 303, 307, 308}:
                if response.status != 200:
                    with response:
                        _read_metadata_body(response, self)
                    raise ValueError("unexpected acquisition HTTP status")
                return response
            with response:
                _read_metadata_body(response, self)
                location = response.headers.get("Location")
            if not location:
                raise ValueError("acquisition redirect lacks Location")
            destination = urljoin(url, location)
            if urlsplit(destination).netloc != parsed.netloc:
                headers.pop("Authorization", None)
            url = destination
        raise ValueError("too many acquisition redirects")

    def _authenticate(self, path, row):
        _regular(path)
        if path.stat().st_size != row["size"]:
            raise ValueError("cached pinned file size mismatch")
        self.consume(
            "verification_bytes", row["size"], self.spec.max_verification_bytes
        )
        sha = hashlib.sha256()
        git = hashlib.sha1(b"blob " + str(row["size"]).encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                self.remaining_seconds()
                sha.update(chunk)
                git.update(chunk)
        if "lfs_sha256" in row:
            valid = sha.hexdigest() == row["lfs_sha256"]
        else:
            valid = git.hexdigest() == row["blob_id"]
        if not valid:
            raise ValueError("pinned file typed content hash mismatch")
        return sha.hexdigest()

    def file(self, row):
        self.observe()
        self._cache_inventory()
        key = canonical_digest(row)
        target = self.cache / (key + ".bin")
        partial = target.with_suffix(".partial")
        if target.exists():
            self._authenticate(target, row)
            self._mutate(lambda state: state.update(cache_hits=state["cache_hits"] + 1))
            return target
        if partial.exists():
            _regular(partial)
            partial.unlink()  # only this owned incomplete file; transfer charge retained
        size, count = self._cache_inventory()
        if (
            row["size"] > self.spec.max_cache_bytes - size
            or count >= self.spec.max_cache_files
        ):
            raise ValueError("acquisition cache reservation unavailable")
        index = self.reserve_transfer(
            self.spec.repo_id, row["path"], row["size"], allow_verified_retry=True
        )
        try:
            with self._open(self._url(row)) as response:
                if (
                    response.headers.get("Transfer-Encoding")
                    or _response_length(response) != row["size"]
                ):
                    raise ValueError("pinned response requires exact Content-Length")
                with partial.open("xb", buffering=0) as output:
                    remaining = row["size"]
                    while remaining:
                        requested = min(65536, remaining)
                        chunk = _read_with_deadline(response, requested, self)
                        if not chunk or len(chunk) > requested:
                            raise OSError("short or oversized pinned response")
                        self.charge_transfer_actual(index, len(chunk))
                        output.write(chunk)
                        remaining -= len(chunk)
                        self.observe()
                    os.fsync(output.fileno())
            self._authenticate(partial, row)
            _rename_noreplace(partial, target)
            sync_directory(self.cache)
            self.finish_transfer(index, success=True)
        except (
            OSError,
            TimeoutError,
            KeyboardInterrupt,
            AcquisitionInterrupted,
        ) as error:
            self.finish_transfer(index, success=False, interrupted=True)
            raise AcquisitionInterrupted(str(error)) from error
        except BaseException:
            self.finish_transfer(index, success=False)
            raise
        return target

    def stream(self, source, split, inventory):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Features
        from datasets.table import table_cast

        if inventory != self.lock["provenance"]:
            raise ValueError("transport inventory changed")
        options = inventory.get("loader_options", {})
        features = (
            Features.from_dict(options["features"]) if options.get("features") else None
        )
        for row in inventory["files"][split]:
            path = self.file(row)
            with pq.ParquetFile(path, pre_buffer=False, buffer_size=65536) as parquet:
                schema = parquet.schema_arrow
                inferred = Features.from_arrow_schema(schema)
                _validate_features(inferred.to_dict())
                allowed = (
                    pa.types.is_boolean,
                    pa.types.is_integer,
                    pa.types.is_floating,
                    pa.types.is_string,
                    pa.types.is_large_string,
                )
                if any(
                    not any(test(field.type) for test in allowed) for field in schema
                ):
                    raise ValueError("Parquet profile requires scalar/text columns")
                if source.text_field not in schema.names or not (
                    pa.types.is_string(schema.field(source.text_field).type)
                    or pa.types.is_large_string(schema.field(source.text_field).type)
                ):
                    raise ValueError("Parquet text field must be a string")
                if (
                    features is not None
                    and features.arrow_schema != schema.remove_metadata()
                ):
                    raise ValueError("Parquet profile does not support feature casts")
                # Count all decoded rows before any selection/deduplication. No filters.
                batch_size = min(
                    1024,
                    options.get("batch_size") or 1024,
                    max(1, self.spec.max_scanned_rows - self._read()["scanned_rows"]),
                )
                batches = parquet.iter_batches(batch_size=batch_size, use_threads=False)
                try:
                    remaining_rows = parquet.metadata.num_rows
                    while remaining_rows:
                        # Reserve before decoding; a crash never refunds row work.
                        reserve = min(batch_size, remaining_rows)
                        self.consume(
                            "scanned_rows", reserve, self.spec.max_scanned_rows
                        )
                        batch = next(batches)
                        if batch.num_rows <= 0 or batch.num_rows > reserve:
                            raise ValueError("Parquet row batch exceeds reservation")
                        remaining_rows -= batch.num_rows
                        self._mutate(
                            lambda state, rows=batch.num_rows: state.update(
                                scanned_rows_observed=state["scanned_rows_observed"]
                                + rows
                            )
                        )
                        self.observe()
                        table = pa.Table.from_batches([batch])
                        if features is not None:
                            table = table_cast(table, features.arrow_schema)
                        yield from table.to_pylist()
                finally:
                    batches.close()

    def before_verification(self, path):
        total = current = 0
        for alternatives in self.snapshot_roots:
            root = next((root for root in alternatives if root.exists()), None)
            if root is None:
                continue
            for member in root.iterdir():
                _regular(member)
                size = member.stat().st_size
                total += size
                if root == path.parent:
                    current += size
        if total > self.snapshot_limit:
            raise ValueError(
                "acquired snapshots exceed cumulative snapshot byte admission"
            )
        self.consume("verification_bytes", current, self.spec.max_verification_bytes)
        self.observe()

    def scratch(self):
        path = self.root / "verification"
        path.mkdir(exist_ok=True)
        return path

    @contextmanager
    def scope(self):
        token = current_transport.set(self)
        try:
            yield
        finally:
            current_transport.reset(token)

    def receipt(self):
        return {
            **super().receipt(),
            "format": "klab-dataset-transport-v1",
            "observed_body_bytes_may_be_incomplete_after_crash": True,
            "wire_bytes": None,
            "hard_disk_quota": False,
            "hard_decoder_memory_limit": False,
            "expanded_bytes": None,
        }
