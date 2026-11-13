"""Row placement for ``document_start`` token mixtures.

Shared by mixture materialization, which records the layout in its receipt,
and by packing, which places the masked filler; neither imports the other.
"""

from __future__ import annotations


class DocumentStartRows:
    """Place records of a target stream so long documents start on a row.

    A record of at least its stratum's minimum (its EOS included) starts at a
    row boundary of the target stream, after masked filler; every other record
    follows the previous one directly. A record's targets therefore stay
    contiguous, and a long record occupies consecutive rows in order.
    """

    def __init__(self, row_tokens: int, minimums: dict[str, int]) -> None:
        self.width, self.minimums = row_tokens, minimums
        self.position = self.filler = 0
        self.strata = {
            stratum: {"min_document_tokens": minimum, "documents": 0, "rows": 0}
            for stratum, minimum in sorted(minimums.items())
        }
        # A long record whose last row is partly free holds that row alone
        # only if the next record starts on a fresh row.
        self.open_row: tuple[str, int] | None = None

    def place(self, stratum: str, tokens: int) -> int:
        """Return the filler before this record and advance past it."""
        minimum = self.minimums.get(stratum)
        starts_row = minimum is not None and tokens >= minimum
        gap = -self.position % self.width if starts_row else 0
        self.position += gap
        self.filler += gap
        if self.open_row is not None:
            owner, row = self.open_row
            if self.position // self.width != row:
                self.strata[owner]["rows"] += 1
            self.open_row = None
        if starts_row:
            self.strata[stratum]["documents"] += 1
            self.strata[stratum]["rows"] += tokens // self.width
            if tokens % self.width:
                self.open_row = (stratum, (self.position + tokens) // self.width)
        self.position += tokens
        return gap

    def summary(self) -> dict:
        strata = {stratum: dict(value) for stratum, value in self.strata.items()}
        if self.open_row is not None:
            strata[self.open_row[0]]["rows"] += 1
        return {
            "layout": "document_start",
            "row_tokens": self.width,
            "rows": -(-self.position // self.width),
            "filler_positions": self.filler,
            # Rows whose every target belongs to one record of at least the
            # stratum minimum.
            "strata": strata,
        }


def document_rows(layout: dict) -> DocumentStartRows:
    """The placement a receipt's ``row_layout`` declares."""
    return DocumentStartRows(
        layout["row_tokens"],
        {
            stratum: value["min_document_tokens"]
            for stratum, value in layout["strata"].items()
        },
    )
