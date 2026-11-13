"""Deterministic PDF text extraction, run in a child process under a wall-time limit.

``python -I -m klab.corpus.pdf_text PDF VERSION MAX_PAGES`` writes the page
count, a newline and the UTF-8 text to stdout. A refusal exits with status 2 and
its reason on stderr. The parent kills the child at its deadline, so a PDF that
stalls the parser is refused rather than hanging acquisition or verification.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path


def extract(path: Path, version: str, max_pages: int) -> tuple[bytes, int]:
    """Page order, LF newlines, trailing spaces removed, pages joined by a blank line."""
    import pypdf

    if pypdf.__version__ != version:
        raise ValueError(f"PDF text extraction is pinned to pypdf {version}")
    try:
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password")
        pages = len(reader.pages)
        if pages > max_pages:
            raise ValueError("PDF exceeds max_pages")
        texts = []
        for page in reader.pages:
            text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
            text = "\n".join(
                line.rstrip() for line in text.replace("\x00", "").split("\n")
            )
            if text.strip():
                texts.append(text.strip("\n"))
    except pypdf.errors.PdfReadError as error:
        raise ValueError(f"invalid PDF: {error}") from error
    if not texts:
        raise ValueError("PDF has no extractable text layer (OCR is unsupported)")
    return ("\n\n".join(texts) + "\n").encode("utf-8", errors="replace"), pages


def main(arguments: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    path, version, max_pages = arguments
    try:
        text, pages = extract(Path(path), version, int(max_pages))
    except ValueError as error:
        sys.stderr.write(str(error)[:500])
        return 2
    sys.stdout.buffer.write(b"%d\n" % pages + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
