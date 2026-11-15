"""Size Engram n-gram tables against the statistics of real token data.

``klab data engram-sizing`` hashes a bounded sample of a token stream with the
exact address functions the model uses (``klab.address_hash``) and reports, per
(site, order) stream and per candidate table size, how much of the traffic
shares or is dominated by other n-grams' rows.

Method. The pre-modulo address of an n-gram is a 63-bit mix; two different
n-grams share it with probability about 2**-60, so distinct addresses count
distinct n-grams. Counting is exact on the sampled tokens (``np.unique``);
memory is about 40 bytes per sampled token. For a prime ``p`` an n-gram ``g``
with count ``f_g`` lands in row ``g mod p``; with row traffic ``R`` its
*purity* is ``f_g / R``. An occurrence is *shared* when another n-gram uses its
row and *clobbered* when other n-grams carry more than half of the row's
traffic. Under a uniformly random hash the shared probability is
``1 - (1 - 1/p)**(D - 1)`` for ``D`` distinct n-grams, which the report prints
beside the measured value.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from klab.address_hash import (
    engram_address_finish,
    engram_address_step,
    engram_multipliers,
    engram_table_sizes,
)

DEFAULT_MAX_TOKENS = 1 << 25
DEFAULT_BLOCKS = 64
DEFAULT_LADDER = tuple(1 << power for power in range(14, 25))
TOP_COUNTS = (1_000, 10_000, 100_000, 1_000_000)
PREFIX_FRACTIONS = (8, 4, 2, 1)
# Adam keeps two fp32 moments; the gradient and weights are fp32 as well.
TRAINING_BYTES_PER_PARAMETER = 16


@dataclass(frozen=True)
class EngramSetup:
    """The addressing and shape settings the report is computed for."""

    orders: tuple[int, ...]
    heads: int
    seed: int
    vocab_size: int
    padding_token_id: int
    sites: tuple[int, ...]
    memory_dim: int
    token_mapping: tuple[int, ...] | None = None
    seq_len: int | None = None
    configured_sizes: tuple[int, ...] | None = None
    total_parameters: int | None = None

    @property
    def streams(self) -> int:
        return len(self.orders) * self.heads

    @classmethod
    def from_config(cls, config) -> EngramSetup:
        model = config.model
        if model.memory != "ngram":
            raise ValueError("the config has no n-gram memory (model.memory: ngram)")
        mapping = model.memory_token_mapping
        from klab.model.inspection import parameter_inventory

        return cls(
            orders=model.memory_ngram_orders or (model.memory_ngram_size,),
            heads=model.memory_hash_heads,
            seed=model.memory_hash_seed,
            vocab_size=model.vocab_size,
            padding_token_id=model.memory_padding_token_id,
            sites=model.memory_block_sites or (0,),
            memory_dim=model.memory_dim,
            token_mapping=None if mapping is None else mapping.token_ids,
            seq_len=config.training.seq_len,
            configured_sizes=model.memory_table_sizes,
            total_parameters=parameter_inventory(config).total,
        )


def open_tokens(path: Path) -> list[np.ndarray]:
    """Memory-map token IDs from a prepared root, token store, .npy or raw int32."""
    path = Path(path)
    if path.is_dir():
        if (path / "train.npy").is_file():
            return [np.load(path / "train.npy", mmap_mode="r", allow_pickle=False)]
        shards = sorted(path.glob("shard-*.ids"))
        if not shards:
            raise ValueError(
                f"{path} is neither a prepared root (train.npy) nor a token store"
            )
        return [np.memmap(shard, dtype=np.int32, mode="r") for shard in shards]
    if path.suffix == ".npy":
        return [np.load(path, mmap_mode="r", allow_pickle=False)]
    if not path.is_file():
        raise ValueError(f"token file not found: {path}")
    return [np.memmap(path, dtype=np.int32, mode="r")]


def sample_blocks(
    arrays: Sequence[np.ndarray],
    max_tokens: int,
    blocks: int,
    block_multiple: int = 1,
) -> list[np.ndarray]:
    """Take evenly spaced contiguous blocks covering the whole input.

    Every block is a sequence start for hashing. Inputs no longer than
    ``max_tokens`` are used whole. Block starts and sizes are multiples of
    ``block_multiple`` (the training sequence length) so windows stay aligned.
    """
    lengths = [len(a) for a in arrays]
    total = sum(lengths)
    if total == 0:
        raise ValueError("no tokens to analyze")
    if total <= max_tokens:
        return [np.asarray(a, dtype=np.int64) for a in arrays]
    size = max(block_multiple, max_tokens // blocks // block_multiple * block_multiple)
    count = max(1, max_tokens // size)
    starts = [int(i * (total - size) / max(count - 1, 1)) for i in range(count)]
    starts = [s // block_multiple * block_multiple for s in starts]
    offsets = np.cumsum([0, *lengths])
    result = []
    for start in starts:
        index = int(np.searchsorted(offsets, start, side="right")) - 1
        local = start - int(offsets[index])
        piece = np.asarray(arrays[index][local : local + size], dtype=np.int64)
        if len(piece):
            result.append(piece)
    return result


def ngram_keys(
    tokens: np.ndarray,
    order: int,
    multipliers: Sequence[int],
    padding: int,
    seq_len: int | None,
) -> np.ndarray:
    """Pre-modulo addresses of every position, as ``TokenNgramMemory`` computes them.

    Positions inside the first ``order - 1`` of a block or training window read
    the padding ID, exactly like the model's left padding.
    """
    position = np.arange(len(tokens))
    if seq_len:
        position = position % seq_len
    address = np.zeros(len(tokens), dtype=np.int64)
    for offset in range(order):
        shifted = np.full(len(tokens), padding, dtype=np.int64)
        if offset < len(tokens):
            shifted[offset:] = tokens[: len(tokens) - offset]
        shifted[position < offset] = padding
        address = engram_address_step(address, shifted, multipliers[offset])
    return address


@dataclass
class OrderCounts:
    """Exact n-gram counts of one (site, order) stream on the sample."""

    order: int
    tokens: int
    keys: np.ndarray
    counts: np.ndarray
    first_position: np.ndarray

    @property
    def distinct(self) -> int:
        return len(self.counts)

    def prefix_distinct(self) -> list[tuple[int, int]]:
        points = []
        for divisor in PREFIX_FRACTIONS:
            size = self.tokens // divisor
            points.append((size, int(np.count_nonzero(self.first_position < size))))
        return points

    def heaps(self) -> tuple[float, float] | None:
        """Fit ``D = k * n**beta`` through the prefix points; None if degenerate."""
        points = [(n, d) for n, d in self.prefix_distinct() if n > 0 and d > 0]
        if len(points) < 2 or len({n for n, _ in points}) < 2:
            return None
        x = np.log([n for n, _ in points])
        y = np.log([d for _, d in points])
        beta, intercept = np.polyfit(x, y, 1)
        return float(math.exp(intercept)), float(beta)

    def summary(self) -> dict:
        ordered = np.sort(self.counts)[::-1]
        singletons = self.counts == 1
        fit = None
        low, high = 10, min(self.distinct, 10_000)
        if high - low >= 2:
            ranks = np.log(np.arange(low, high + 1))
            slope = np.polyfit(ranks, np.log(ordered[low - 1 : high]), 1)[0]
            fit = float(-slope)
        heaps = self.heaps()
        return {
            "order": self.order,
            "tokens": self.tokens,
            "distinct": self.distinct,
            "singleton_ngrams": int(singletons.sum()),
            "singleton_ngram_share": float(singletons.mean()),
            "singleton_occurrence_share": float(singletons.sum() / self.tokens),
            "top_mass": {
                str(k): float(ordered[:k].sum() / self.tokens)
                for k in TOP_COUNTS
                if k < self.distinct
            },
            "zipf_slope": fit,
            "distinct_by_prefix": [
                {"tokens": n, "distinct": d} for n, d in self.prefix_distinct()
            ],
            "heaps_exponent": None if heaps is None else heaps[1],
            "_heaps": heaps,
        }


def count_orders(
    blocks: Sequence[np.ndarray], setup: EngramSetup, site: int
) -> list[OrderCounts]:
    """Count the distinct n-gram addresses of each order at one site."""
    multipliers = engram_multipliers(
        setup.seed, max(setup.orders), setup.vocab_size, layer_id=site
    )
    mapping = None if setup.token_mapping is None else np.asarray(setup.token_mapping)
    padding = setup.padding_token_id
    if mapping is not None:
        padding = int(mapping[padding])
    result = []
    for order in setup.orders:
        parts = []
        for block in blocks:
            if len(block) and (block.min() < 0 or block.max() >= setup.vocab_size):
                raise ValueError("token IDs fall outside the model vocabulary")
            ids = block if mapping is None else mapping[block]
            parts.append(ngram_keys(ids, order, multipliers, padding, setup.seq_len))
        keys = np.concatenate(parts)
        unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
        result.append(OrderCounts(order, len(keys), unique, counts, first))
    return result


def head_statistics(
    stream: OrderCounts, prime: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Measured collision statistics of one table; also returns per-ngram flags."""
    rows = engram_address_finish(stream.keys, prime)
    distinct_per_row = np.bincount(rows, minlength=prime)
    load = np.bincount(rows, weights=stream.counts, minlength=prime)
    shared = distinct_per_row[rows] > 1
    purity = stream.counts / load[rows]
    clobbered = purity < 0.5
    tokens = stream.tokens
    distinct = stream.distinct
    occupied = int(np.count_nonzero(distinct_per_row))
    stats = {
        "rows": prime,
        "occupancy": occupied / prime,
        "expected_occupancy": 1 - (1 - 1 / prime) ** distinct,
        "shared_occurrences": float(stream.counts[shared].sum() / tokens),
        "expected_shared_occurrences": 1 - (1 - 1 / prime) ** (distinct - 1),
        "clobbered_occurrences": float(stream.counts[clobbered].sum() / tokens),
        "mean_purity": float((stream.counts * purity).sum() / tokens),
        "largest_row_share": float(load.max() / tokens),
    }
    return stats, shared, clobbered


def evaluate_table(stream: OrderCounts, primes: Sequence[int]) -> dict:
    """Per-head and all-heads statistics for the heads of one order."""
    heads = []
    shared_all = clobbered_all = None
    for prime in primes:
        stats, shared, clobbered = head_statistics(stream, prime)
        heads.append(stats)
        shared_all = shared if shared_all is None else shared_all & shared
        clobbered_all = (
            clobbered if clobbered_all is None else clobbered_all & clobbered
        )
    counts = stream.counts
    return {
        "heads": heads,
        "shared_all_heads": float(counts[shared_all].sum() / stream.tokens),
        "shared_all_heads_if_independent": float(
            np.prod([h["shared_occurrences"] for h in heads])
        ),
        "clobbered_all_heads": float(counts[clobbered_all].sum() / stream.tokens),
        "clobbered_all_heads_if_independent": float(
            np.prod([h["clobbered_occurrences"] for h in heads])
        ),
    }


def projected_shared(distinct: float, primes: Sequence[int]) -> float:
    """Random-hash probability an occurrence shares a row on every head."""
    return float(np.prod([-math.expm1(-(distinct - 1) / prime) for prime in primes]))


def parameter_cost(
    rows: int, setup: EngramSetup, total: int | None
) -> dict[str, float | int | None]:
    parameters = rows * setup.memory_dim
    return {
        "rows": rows,
        "parameters": parameters,
        "weight_gib_fp32": parameters * 4 / 1024**3,
        "training_state_gib": parameters * TRAINING_BYTES_PER_PARAMETER / 1024**3,
        "share_of_model": None if not total else parameters / total,
    }


def _assign_sizes(requested: Sequence[int], used: set[int]) -> list[int]:
    """Smallest unused prime at or above each requested size, in order."""
    result = []
    for size in requested:
        prime = size
        while True:
            prime = engram_table_sizes(prime, 1)[0]
            if prime not in used:
                break
            prime += 1
        used.add(prime)
        result.append(prime)
    return result


def analyze(
    blocks: Sequence[np.ndarray],
    setup: EngramSetup,
    *,
    candidates: Sequence[int] = DEFAULT_LADDER,
    max_clobbered: float = 0.25,
    target_tokens: int | None = None,
) -> dict:
    """Report statistics for every site, order and candidate table size."""
    candidate_sizes = sorted(set(candidates))
    site_reports = []
    verdict: dict[int, int | None] = {}
    for site_index, site in enumerate(setup.sites):
        counted = count_orders(blocks, setup, site)
        order_reports = []
        for stream_index, stream in enumerate(counted):
            summary = stream.summary()
            heaps = summary.pop("_heaps")
            projected = None
            if target_tokens and heaps:
                projected = heaps[0] * target_tokens ** heaps[1]
            tables = []
            for size in candidate_sizes:
                primes = engram_table_sizes(size, setup.heads)
                row = {"requested": size, **evaluate_table(stream, primes)}
                if projected:
                    row["projected_shared_all_heads"] = projected_shared(
                        projected, primes
                    )
                row["rows"] = sum(primes)
                tables.append(row)
            if setup.configured_sizes:
                start = site_index * setup.streams + stream_index * setup.heads
                primes = setup.configured_sizes[start : start + setup.heads]
                row = {"requested": None, **evaluate_table(stream, primes)}
                row["rows"] = sum(primes)
                if projected:
                    row["projected_shared_all_heads"] = projected_shared(
                        projected, primes
                    )
                summary["configured"] = row
            summary["projected_distinct_at_target"] = projected
            summary["tables"] = tables
            meeting = [
                t["requested"]
                for t in tables
                if t["clobbered_all_heads"] <= max_clobbered
            ]
            summary["smallest_meeting_target"] = meeting[0] if meeting else None
            order_reports.append(summary)
            key = site * 1000 + stream.order
            verdict[key] = summary["smallest_meeting_target"]
        site_reports.append({"site": site, "orders": order_reports})
    report = {
        "setup": {
            "orders": list(setup.orders),
            "hash_heads": setup.heads,
            "seed": setup.seed,
            "vocab_size": setup.vocab_size,
            "padding_token_id": setup.padding_token_id,
            "sites": list(setup.sites),
            "memory_dim": setup.memory_dim,
            "mapped": setup.token_mapping is not None,
            "seq_len": setup.seq_len,
        },
        "sample_tokens": sum(len(b) for b in blocks),
        "sample_blocks": len(blocks),
        "target_tokens": target_tokens,
        "criterion": {"max_clobbered_all_heads": max_clobbered},
        "sites": site_reports,
    }
    report["recommendation"] = recommend(report, setup)
    return report


def recommend(report: dict, setup: EngramSetup) -> dict:
    """Smallest candidate per order meeting the target at every site."""
    requested: list[int | None] = []
    for order_index in range(len(setup.orders)):
        sizes = [
            site["orders"][order_index]["smallest_meeting_target"]
            for site in report["sites"]
        ]
        requested.append(None if None in sizes else max(sizes))
    unmet = [o for o, r in zip(setup.orders, requested, strict=True) if r is None]
    if unmet:
        largest = max(t["requested"] for t in report["sites"][0]["orders"][0]["tables"])
        return {
            "met": False,
            "unmet_orders": unmet,
            "message": (
                f"no candidate up to {largest:,} rows per head met the target for "
                f"order(s) {unmet}; raise --sizes, loosen the target, or accept that "
                "those streams act as a hashed bag of frequent n-grams"
            ),
        }
    used: set[int] = set()
    per_site = []
    for _ in setup.sites:
        site_sizes = []
        for size in requested:
            site_sizes.extend(_assign_sizes([size] * setup.heads, used))
        per_site.extend(site_sizes)
    return {
        "met": True,
        "per_order_requested_rows": dict(zip(setup.orders, requested, strict=True)),
        "memory_table_sizes": per_site,
        "memory_table_size": per_site[0],
        "cost": parameter_cost(sum(per_site), setup, setup.total_parameters),
    }


def render(report: dict) -> str:
    """Human-readable summary of an ``analyze`` report."""
    lines = []
    setup = report["setup"]
    lines.append(
        f"Engram sizing on {report['sample_tokens']:,} sampled tokens "
        f"({report['sample_blocks']} blocks); orders {setup['orders']}, "
        f"{setup['hash_heads']} hash heads, sites {setup['sites']}, "
        f"memory_dim {setup['memory_dim']}"
    )
    lines.append(
        "  clobbered = another n-gram set carries over half of the row's traffic; "
        "shared = any other n-gram uses the row"
    )
    first = report["sites"][0]
    spread = (
        max(
            abs(
                a["tables"][i]["clobbered_all_heads"]
                - b["orders"][j]["tables"][i]["clobbered_all_heads"]
            )
            for b in report["sites"][1:]
            for j, a in enumerate(first["orders"])
            for i in range(len(a["tables"]))
        )
        if len(report["sites"]) > 1
        else None
    )
    for site in report["sites"][:1]:
        for item in site["orders"]:
            lines.append(f"\nSite {site['site']}, order {item['order']}")
            lines.append(
                f"  distinct {item['distinct']:,}; singletons "
                f"{item['singleton_ngram_share']:.1%} of n-grams, "
                f"{item['singleton_occurrence_share']:.1%} of occurrences; "
                f"Zipf slope {item['zipf_slope'] if item['zipf_slope'] is None else round(item['zipf_slope'], 2)}; "
                f"Heaps exponent "
                f"{item['heaps_exponent'] if item['heaps_exponent'] is None else round(item['heaps_exponent'], 2)}"
            )
            lines.append(
                "  top-k mass: "
                + ", ".join(f"{k}: {v:.1%}" for k, v in item["top_mass"].items())
            )
            if item.get("projected_distinct_at_target"):
                lines.append(
                    f"  projected distinct at {report['target_tokens']:,} tokens: "
                    f"{item['projected_distinct_at_target']:,.0f} (Heaps fit, estimate)"
                )
            lines.append(
                "  rows/head  occupancy  shared(1 head) measured/expected  "
                "clobbered(1 head)  clobbered(all)  shared(all)"
            )
            tables = [*item["tables"]]
            if "configured" in item:
                tables.append(item["configured"])
            for table in tables:
                head = table["heads"][0]
                label = (
                    "configured"
                    if table["requested"] is None
                    else f"{table['requested']:,}"
                )
                lines.append(
                    f"  {label:>10}  {head['occupancy']:>8.1%}  "
                    f"{head['shared_occurrences']:>8.1%} / {head['expected_shared_occurrences']:<8.1%}"
                    f"           {head['clobbered_occurrences']:>8.1%}"
                    f"          {table['clobbered_all_heads']:>8.1%}"
                    f"      {table['shared_all_heads']:>8.1%}"
                )
    if spread is not None:
        lines.append(
            f"\nThe other sites use different multipliers; their clobbered(all) "
            f"differs from site {first['site']} by at most {spread:.1%} (all sites are "
            "in --json and in the recommendation)."
        )
    lines.append(
        f"\nTarget: clobbered on all heads <= "
        f"{report['criterion']['max_clobbered_all_heads']:.0%} of occurrences"
    )
    rec = report["recommendation"]
    if not rec["met"]:
        lines.append(rec["message"])
    else:
        cost = rec["cost"]
        lines.append(
            "Smallest sizes meeting it (rows per head): "
            + ", ".join(
                f"order {o}: {r:,}" for o, r in rec["per_order_requested_rows"].items()
            )
        )
        lines.append("Set model.memory_table_sizes to:")
        lines.append(f"  {rec['memory_table_sizes']}")
        share = cost["share_of_model"]
        lines.append(
            f"Cost: {cost['rows']:,} rows, {cost['parameters']:,} parameters"
            + ("" if share is None else f" ({share:.1%} of the model)")
            + f"; {cost['weight_gib_fp32']:.2f} GiB fp32 weights, about "
            f"{cost['training_state_gib']:.2f} GiB with gradients and Adam state"
        )
    lines.append(
        "Measured on the sample only: rerun on the real training tokens, and treat "
        "the sample as a lower bound on collisions for a larger corpus."
    )
    return "\n".join(lines)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item)


def _setup(args: argparse.Namespace) -> EngramSetup:
    if args.config is not None:
        from klab.config.loading import load_config

        base = EngramSetup.from_config(load_config(args.config))
    else:
        base = None
    given = {
        "orders": args.orders,
        "heads": args.heads,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "padding_token_id": args.padding_token_id,
        "sites": args.sites,
        "memory_dim": args.memory_dim,
        "seq_len": args.seq_len,
    }
    if base is None:
        missing = [
            k
            for k in ("orders", "heads", "vocab_size", "memory_dim")
            if given[k] is None
        ]
        if missing:
            raise ValueError(
                f"give --config or {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        defaults = {"seed": 0, "padding_token_id": 0, "sites": (0,), "seq_len": None}
        values = {
            k: (v if v is not None else defaults.get(k)) for k, v in given.items()
        }
        return EngramSetup(**values)
    from dataclasses import replace

    return replace(base, **{k: v for k, v in given.items() if v is not None})


def _handle(args: argparse.Namespace) -> None:
    try:
        setup = _setup(args)
        arrays = open_tokens(args.tokens)
        blocks = sample_blocks(arrays, args.max_tokens, args.blocks, setup.seq_len or 1)
        report = analyze(
            blocks,
            setup,
            candidates=args.sizes or DEFAULT_LADDER,
            max_clobbered=args.max_clobbered,
            target_tokens=args.target_tokens,
        )
    except (OSError, ValueError) as error:
        print(f"Engram sizing blocked: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))


def register_parser(data_commands: argparse._SubParsersAction) -> None:
    command = data_commands.add_parser(
        "engram-sizing",
        help="Measure n-gram collisions on token data and size Engram tables",
    )
    command.add_argument(
        "tokens",
        type=Path,
        help="prepared root (train.npy), token store directory, .npy or raw int32 file",
    )
    command.add_argument("--config", type=Path, help="run config with n-gram memory")
    command.add_argument("--orders", type=_parse_ints, help="for example 2,3")
    command.add_argument("--heads", type=int)
    command.add_argument("--seed", type=int)
    command.add_argument("--vocab-size", type=int)
    command.add_argument("--padding-token-id", type=int)
    command.add_argument("--sites", type=_parse_ints, help="memory block indices")
    command.add_argument("--memory-dim", type=int)
    command.add_argument("--seq-len", type=int)
    command.add_argument(
        "--sizes",
        type=_parse_ints,
        help="candidate rows per head (default: powers of two, 16384 to 16777216)",
    )
    command.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    command.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    command.add_argument(
        "--target-tokens",
        type=int,
        help="training tokens, to project distinct n-grams with a Heaps fit",
    )
    command.add_argument("--max-clobbered", type=float, default=0.25)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
