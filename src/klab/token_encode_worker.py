"""Tokenizer worker for the token store; imports no torch and no data package.

Run as ``python -m klab.token_encode_worker``. Each worker owns one
single-threaded tokenizer and speaks length-prefixed binary frames on its
standard streams. A request is a run of raw JSONL lines; the worker parses and validates
them itself, so the parent only reads bytes. A reply is the documents' token
IDs, each document followed by EOS, as raw int32, plus raw uint32 arrays of
per-document token counts, text sizes and line sizes. No JSON or Python integer
list crosses the process boundary. The parent and the worker are the same trusted program.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import sys
from typing import BinaryIO

import numpy as np

_tokenizer = None
_eos = 0


def initialize(tokenizer_json: str, eos: int) -> None:
    global _tokenizer, _eos
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer

    _tokenizer = Tokenizer.from_str(tokenizer_json)
    _eos = eos


class LineError(ValueError):
    """A source line was refused; ``index`` is its position within the request."""

    def __init__(self, index: int, kind: str, detail: tuple[int, ...] = ()) -> None:
        super().__init__(kind)
        self.index, self.kind, self.detail = index, kind, detail


def local_text_record(line: bytes) -> str:
    """The text of one ``{"text": ...}`` JSONL line, or ``LineError``."""
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LineError(0, "json") from error
    if (
        not isinstance(row, dict)
        or set(row) != {"text"}
        or not isinstance(row["text"], str)
        or not row["text"].strip()
    ):
        raise LineError(0, "record")
    return row["text"]


def encode_lines(
    chunk: bytes, line_limit: int, source_bytes: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse and tokenize whole JSONL lines.

    Returns ``(ids, token_counts, text_bytes, line_bytes)``; token counts
    include each document's EOS. ``line_limit`` is the raw-line cap (negative
    for none) and ``source_bytes`` the cap on a document's UTF-8 size.
    """
    assert _tokenizer is not None
    # Split on newline only, exactly as ``readline`` does.
    parts = chunk.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    texts = []
    for index, line in enumerate(lines):
        if 0 <= line_limit < len(line):
            raise LineError(index, "line_cap", (line_limit,))
        try:
            texts.append(local_text_record(line))
        except LineError as error:
            raise LineError(index, error.kind) from error
    text_bytes = np.fromiter(
        (len(text.encode("utf-8")) for text in texts), dtype=np.uint32, count=len(texts)
    )
    if source_bytes is not None and len(texts) and int(text_bytes.max()) > source_bytes:
        index = int(np.argmax(text_bytes > source_bytes))
        raise LineError(index, "document_cap", (int(text_bytes[index]), source_bytes))
    encodings = _tokenizer.encode_batch(texts, add_special_tokens=False)
    counts = np.fromiter(
        (len(encoding.ids) + 1 for encoding in encodings),
        dtype=np.uint32,
        count=len(encodings),
    )
    ids = np.empty(int(counts.sum()), dtype=np.int32)
    position = 0
    for encoding, count in zip(encodings, counts.tolist(), strict=True):
        ids[position : position + count - 1] = encoding.ids
        ids[position + count - 1] = _eos
        position += count
    line_bytes = np.fromiter(
        (len(line) for line in lines), dtype=np.uint32, count=len(lines)
    )
    return ids, counts, text_bytes, line_bytes


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)


def read_frame(stream: BinaryIO) -> bytes | None:
    header = stream.read(8)
    if not header:
        return None
    (size,) = struct.unpack("<Q", header)
    payload = stream.read(size)
    if len(header) != 8 or len(payload) != size:
        raise EOFError("truncated token worker frame")
    return payload


def main() -> None:
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    setup = read_frame(source)
    assert setup is not None
    initialize(*pickle.loads(setup))
    write_frame(sink, b"ready")
    sink.flush()
    while (request := read_frame(source)) is not None:
        line_limit, source_bytes = struct.unpack("<qq", request[:16])
        try:
            arrays = encode_lines(
                request[16:], line_limit, None if source_bytes < 0 else source_bytes
            )
        except LineError as error:
            write_frame(sink, b"line")
            write_frame(sink, pickle.dumps((error.index, error.kind, error.detail)))
        except Exception as error:  # noqa: BLE001 - tokenizers raises plain Exception
            write_frame(sink, b"error")
            write_frame(sink, str(error)[:512].encode("utf-8"))
        else:
            write_frame(sink, b"ok")
            for array in arrays:
                write_frame(sink, array.tobytes())
        sink.flush()


if __name__ == "__main__":
    main()
