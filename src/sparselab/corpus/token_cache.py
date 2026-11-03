"""Durable content tokenization, independent of mixture quotas and ordering.

Only encoding is reused. Current release rights, split isolation, deduplication
and scheduling are still checked by the mixture producer on every invocation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from sparselab.canonical import canonical_json
from sparselab.corpus.mixture_index import MAX_ROW_BYTES


class TokenCache:
    def __init__(self, root: Path, tokenizer_sha: str):
        root.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(
            canonical_json(["encode-no-special-v1", tokenizer_sha])
        ).hexdigest()
        path = root / f"{name}.sqlite"
        if path.is_symlink():
            raise ValueError("token cache must not be a symlink")
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA cache_size=-4096;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS tokens (
                content TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL
            );
        """)
        self.pending = 0
        self.hits = self.misses = 0
        self.lookup_seconds = self.write_seconds = self.encode_seconds = 0.0

    def lookup(self, content, *, count_miss=True, decode=True):
        started = time.monotonic()
        row = self.db.execute(
            "SELECT payload, digest FROM tokens WHERE content=?", (content,)
        ).fetchone()
        self.lookup_seconds += time.monotonic() - started
        if row is None:
            if count_miss:
                self.misses += 1
            return None
        payload, digest = row
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("cached token payload digest mismatch")
        self.hits += 1
        return self.decode_ids(payload) if decode else payload

    @staticmethod
    def decode_ids(payload):
        ids = json.loads(payload)
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid cached token IDs")
        return ids

    def payload_bytes(self, content):
        row = self.db.execute(
            "SELECT length(payload) FROM tokens WHERE content=?", (content,)
        ).fetchone()
        return row[0] if row is not None else 0

    def encode_many(self, texts, encode_missing):
        """Return cached encodings in input order and batch only missing content."""
        keys, cached, missing = [], {}, {}
        for text in texts:
            content = hashlib.sha256(text.encode("utf-8")).hexdigest()
            keys.append(content)
            if content in cached or content in missing:
                continue
            ids = self.lookup(content)
            if ids is None:
                missing[content] = text
            else:
                cached[content] = ids

        if missing:
            started = time.monotonic()
            results = encode_missing(list(missing.values()))
            self.encode_seconds += time.monotonic() - started
            if len(results) != len(missing):
                raise ValueError("tokenizer returned incorrect batch length")
            started = time.monotonic()
            for content, ids in zip(missing, results, strict=True):
                payload = canonical_json(ids)
                if len(payload) > MAX_ROW_BYTES:
                    raise ValueError("mixture encoded record-byte cap exceeded")
                self.db.execute(
                    "INSERT INTO tokens VALUES(?,?,?)",
                    (content, payload, hashlib.sha256(payload).hexdigest()),
                )
                self.pending += 1
                if self.pending >= 256:
                    self.db.commit()
                    self.pending = 0
                cached[content] = ids
            self.write_seconds += time.monotonic() - started
        return [cached[content] for content in keys]

    def encode(self, tokenizer, text):
        return self.encode_many(
            [text],
            lambda texts: [
                tokenizer.encode(value, add_special_tokens=False).ids for value in texts
            ],
        )[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Valid encodings survive quota failure or cooperative interruption.
        try:
            self.db.commit()
        finally:
            self.db.close()
