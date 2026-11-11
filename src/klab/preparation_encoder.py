"""Minimal preparation tokenizer child; imports no packing or torch."""

from __future__ import annotations

import json
import sys

from tokenizers import Tokenizer


def main() -> None:
    tokenizer = Tokenizer.from_file(sys.argv[1])
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    for line in sys.stdin.buffer:
        try:
            texts = json.loads(line)
            if not isinstance(texts, list) or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("batch must contain strings")
            if len(texts) == 1:
                encodings = [tokenizer.encode(texts[0], add_special_tokens=False)]
            else:
                encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            result = [encoding.ids for encoding in encodings]
        except Exception as error:  # noqa: BLE001 - tokenizers reports Rust failures as Exception
            result = {"error": str(error)[:512]}
        sys.stdout.buffer.write(
            json.dumps(result, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
