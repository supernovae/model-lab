"""Random access to packed training positions over sealed mixture segments."""

from __future__ import annotations

import hashlib
from array import array
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from klab.corpus.mixture_schedule import (
    SCHEDULE_RANGE,
    TRAIN_SCHEDULE,
    VerifiedSchedule,
    open_verified_schedule_member,
    require_schedule_generation,
    schedule_token_counts,
    verify_schedule,
)
from klab.corpus.release_verify import verification_settings
from klab.data.row_layout import packed_mixture_records
from klab.data.tokenizer import load_tokenizer
from klab.safe_io import read_file_range
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.config.models import RunConfig
    from klab.data.packing import PreparedData
    from klab.verification_proofs import ProofStore, VerificationMode

_PAGE_RECORDS = 256
_PAGES = 256
_SEGMENT_HANDLES = 16


class ReferencedTokenArray:
    """The contiguous packer's token positions, resolved a slice at a time.

    Landmarks are one packed offset per schedule page. Within-page ends and
    locators are computed on demand. Document-start gaps are sparse row/size
    pairs, so there is no corpus-sized table of Python records or token IDs.
    """

    ndim = 1
    dtype = np.dtype(np.int32)

    def __init__(
        self,
        schedule: VerifiedSchedule,
        records: Path,
        receipt: dict,
        *,
        seq_len: int,
        bos: int,
        pad: int,
    ) -> None:
        self.schedule = schedule
        self.token_dtype = schedule.token_dtype
        self.token_width = self.token_dtype.itemsize
        self.bos, self.pad = bos, pad
        self.records = receipt["records"]
        self.landmarks = array("Q")
        self.gap_rows = array("Q")
        self.gap_sizes = array("Q")
        self.pages = OrderedDict()
        self.segments = OrderedDict()
        position = 1
        with (
            closing(schedule.ranges()) as ranges,
            closing(packed_mixture_records(records, receipt, seq_len)) as placement,
        ):
            for row, ((_, _, scheduled), (count, gap)) in enumerate(
                zip(ranges, placement, strict=True)
            ):
                if scheduled != count:
                    raise ValueError("mixture schedule range differs from its record")
                if row % _PAGE_RECORDS == 0:
                    self.landmarks.append(position)
                if gap:
                    self.gap_rows.append(row)
                    self.gap_sizes.append(gap)
                position += gap + count
        self.end = position
        self.length = 1 + -(-(position - 1) // seq_len) * seq_len
        schedule.require_unchanged()
        with ExitStack() as stack:
            self.schedule_file = stack.enter_context(
                open_verified_schedule_member(schedule.schedule)
            )
            self.resources = stack.pop_all()

    def __len__(self) -> int:
        return self.length

    def _page(self, page: int):
        if page not in self.pages:
            first = page * _PAGE_RECORDS
            count = min(_PAGE_RECORDS, self.records - first)
            raw = read_file_range(
                self.schedule_file,
                first * SCHEDULE_RANGE.size,
                count * SCHEDULE_RANGE.size,
            )
            require_schedule_generation(
                self.schedule.schedule, self.schedule_file.fileno()
            )
            ends = schedule_token_counts(raw).astype(np.uint64)
            gap_start = bisect_left(self.gap_rows, first)
            gap_end = bisect_left(self.gap_rows, first + count, lo=gap_start)
            if gap_start != gap_end:
                rows = np.asarray(self.gap_rows)[gap_start:gap_end] - first
                ends[rows] += np.asarray(self.gap_sizes)[gap_start:gap_end]
            ends.cumsum(out=ends)
            ends += self.landmarks[page]
            if len(self.pages) == _PAGES:
                self.pages.popitem(last=False)
            self.pages[page] = ends.tolist(), raw
        self.pages.move_to_end(page)
        return self.pages[page]

    def _segment(self, ordinal: int):
        if ordinal not in self.segments:
            if len(self.segments) == _SEGMENT_HANDLES:
                self.segments.popitem(last=False)[1][1].close()
            with ExitStack() as stack:
                stream = stack.enter_context(
                    open_verified_schedule_member(self.schedule.tokens[ordinal])
                )
                self.segments[ordinal] = stream, stack.pop_all()
        self.segments.move_to_end(ordinal)
        return self.segments[ordinal][0]

    def _content(self, segment: int, offset: int, count: int) -> np.ndarray:
        stream = self._segment(segment)
        raw = read_file_range(
            stream, self.token_width * offset, self.token_width * count
        )
        require_schedule_generation(self.schedule.tokens[segment], stream.fileno())
        return np.frombuffer(raw, dtype=self.token_dtype)

    def __getitem__(self, key: slice) -> np.ndarray:
        start, stop, step = key.indices(self.length)
        if step != 1:
            raise ValueError("reference training tokens require a contiguous slice")
        values = None
        position = start
        if position == 0:
            position = 1
        limit = min(stop, self.end)
        if position < limit:
            page = bisect_right(self.landmarks, position) - 1
            ends, ranges = self._page(page)
            row = bisect_right(ends, position)
            while position < limit:
                segment, offset, count = SCHEDULE_RANGE.unpack_from(
                    ranges, row * SCHEDULE_RANGE.size
                )
                end = ends[row]
                document = end - count
                if document <= position == start < stop < end:
                    content = self._content(
                        segment, offset + start - document, stop - start
                    )
                    return (
                        content if self.token_width == 4 else content.astype(self.dtype)
                    )
                if values is None:
                    values = np.full(stop - start, self.pad, dtype=self.dtype)
                if position < document:
                    # Filler is PAD, with EOS immediately before the new document.
                    if document <= limit:
                        values[document - 1 - start] = self.schedule.eos
                    position = min(document, limit)
                content_end = min(end - 1, limit)
                if position < content_end:
                    values[position - start : content_end - start] = self._content(
                        segment, offset + position - document, content_end - position
                    )
                if end <= limit:
                    values[end - 1 - start] = self.schedule.eos
                position = min(end, limit)
                row += 1
                if position < limit and row == len(ends):
                    page += 1
                    ends, ranges = self._page(page)
                    row = 0
        if values is None:
            values = np.full(max(0, stop - start), self.pad, dtype=self.dtype)
        if start == 0 and len(values):
            values[0] = self.bos
        return values

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        with self.resources as resources:
            for _, stack in self.segments.values():
                resources.callback(stack.close)


def reference_training_tokens(
    config: RunConfig,
    data: PreparedData,
    *,
    tokenizer_path: Path,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> ReferencedTokenArray:
    """Open the exact reference mixture bound by the verified prepared bundle."""
    from klab.corpus.mixture import TRAIN_RECORDS, MixtureDeclaration

    dataset = config.dataset
    receipt = data.manifest.get("token_mixture")
    declaration = dataset.mixture_declaration_path
    output = dataset.mixture_output_path
    if (
        not isinstance(receipt, dict)
        or "token_schedule" not in receipt
        or output is None
        or declaration is None
        or dataset.train_path != output / TRAIN_SCHEDULE
    ):
        raise ValueError(
            "training.mixture_loader=references requires a reference mixture schedule"
        )
    if (
        data.manifest["cache_identity"]["packing"]["mixture_row_tokens"]
        != config.training.seq_len
    ):
        raise ValueError(
            "reference training sequence length differs from the prepared mixture"
        )
    raw = declaration.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != receipt["declaration_sha256"]
        or sha256_file(tokenizer_path) != receipt["tokenizer_sha256"]
    ):
        raise ValueError("reference training inputs differ from the prepared mixture")
    spec = MixtureDeclaration.model_validate(yaml.safe_load(raw))
    tokenizer = load_tokenizer(tokenizer_path)
    bos, eos, pad = (
        tokenizer.token_to_id(f"<{name}>") for name in ("bos", "eos", "pad")
    )
    if any(value is None for value in (bos, eos, pad)):
        raise ValueError("reference token mixture requires BOS, EOS and PAD")
    schedule = data.mixture_schedule
    if schedule is not None:
        if (
            schedule.schedule.path != dataset.train_path
            or schedule.schedule.sha256 != receipt["token_schedule"]["sha256"]
            or schedule.eos != eos
        ):
            raise ValueError(
                "reference schedule proof differs from the prepared mixture"
            )
    else:
        with verification_settings(
            proof_store=proof_store, verification_mode=verification_mode
        ):
            schedule = verify_schedule(
                output,
                spec.referenced_cache_root(output),
                receipt,
                eos,
            )
    source = ReferencedTokenArray(
        schedule,
        output / TRAIN_RECORDS,
        receipt,
        seq_len=config.training.seq_len,
        bos=bos,
        pad=pad,
    )
    if len(source) != len(data.train):
        source.__exit__(None, None, None)
        raise ValueError("reference training length differs from the prepared mixture")
    return source
