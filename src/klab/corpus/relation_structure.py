"""Bounded, deterministic structural views over the canonical relation index."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import groupby

from klab.corpus.relations import ASSERTION_COLUMNS, ASSERTION_JOINS

_PAGE_SIZE = 4096
_SCHEMA = """
CREATE TABLE structure.groups (
    first INTEGER NOT NULL, second INTEGER NOT NULL, assertion INTEGER NOT NULL,
    PRIMARY KEY (first,second,assertion)
) WITHOUT ROWID;
CREATE TABLE structure.edges (
    parent INTEGER NOT NULL, child INTEGER NOT NULL, PRIMARY KEY (parent,child)
) WITHOUT ROWID;
CREATE TABLE structure.nodes (
    node INTEGER PRIMARY KEY, remaining INTEGER NOT NULL, depth INTEGER NOT NULL
);
CREATE TABLE structure.ready (node INTEGER PRIMARY KEY);
"""


@contextmanager
def structural_scratch(db, staging, max_bytes, result):
    """Place the compact grouping and hierarchy tables on the stage filesystem."""
    path = staging / "structure.sqlite"
    db.execute("ATTACH DATABASE ? AS structure", (str(path),))
    try:
        db.executescript(f"""
            PRAGMA structure.page_size={_PAGE_SIZE};
            PRAGMA structure.journal_mode=OFF;
            PRAGMA structure.synchronous=OFF;
            PRAGMA structure.cache_size=-8192;
            PRAGMA structure.max_page_count={max_bytes // _PAGE_SIZE};
            {_SCHEMA}
        """)
        db.execute("PRAGMA temp_store=FILE")
        yield
        result["scratch_bytes"] = path.stat().st_size
    finally:
        db.execute("DETACH DATABASE structure")
        path.unlink(missing_ok=True)


def _hierarchy_depth(db, split, predicate, max_depth):
    # Multiple qualified assertions for one pair are one topology edge.
    db.execute(
        """
        INSERT OR IGNORE INTO structure.edges
        SELECT object, subject FROM assertions WHERE split=? AND predicate=? AND object IS NOT NULL
        """,
        (split, predicate),
    )
    db.execute(
        "INSERT OR IGNORE INTO structure.nodes SELECT parent,0,0 FROM structure.edges"
    )
    db.execute(
        """
        INSERT INTO structure.nodes SELECT child,1,0 FROM structure.edges WHERE true
        ON CONFLICT(node) DO UPDATE SET remaining=remaining+1
        """
    )
    db.execute(
        "INSERT INTO structure.ready SELECT node FROM structure.nodes WHERE remaining=0"
    )
    while ready := db.execute(
        """SELECT r.node,n.depth FROM structure.ready r
        JOIN structure.nodes n USING(node) ORDER BY r.node LIMIT 1"""
    ).fetchone():
        node, depth = ready
        db.execute("DELETE FROM structure.ready WHERE node=?", (node,))
        db.execute(
            """UPDATE structure.nodes SET remaining=remaining-1,depth=max(depth,?)
            WHERE node IN (SELECT child FROM structure.edges WHERE parent=?)""",
            (depth + 1, node),
        )
        db.execute(
            """INSERT INTO structure.ready
            SELECT n.node FROM structure.edges e JOIN structure.nodes n ON n.node=e.child
            WHERE e.parent=? AND n.remaining=0""",
            (node,),
        )
        if depth > max_depth:
            raise ValueError("hierarchy exceeds the declared maximum depth")
    if db.execute("SELECT 1 FROM structure.nodes WHERE remaining>0 LIMIT 1").fetchone():
        raise ValueError("hierarchy predicate contains a directed cycle")


def _populate_groups(db, split, options):
    for table in ("groups", "edges", "nodes", "ready"):
        db.execute(f"DELETE FROM structure.{table}")
    if options.target == "hierarchy":
        predicate = db.execute(
            "SELECT id FROM predicates WHERE name=?", (options.hierarchy_predicate,)
        ).fetchone()
        if predicate is None:
            raise ValueError("hierarchy predicate is absent from the canonical index")
        _hierarchy_depth(db, split, predicate[0], options.max_depth)
        db.execute(
            """
            INSERT INTO structure.groups SELECT n.depth,a.object,a.id
            FROM assertions a JOIN structure.nodes n ON a.object=n.node
            WHERE a.split=? AND a.predicate=?
            """,
            (split, predicate[0]),
        )
        return
    db.execute(
        """INSERT INTO structure.groups SELECT predicate,object,id
        FROM assertions WHERE split=? AND object IS NOT NULL""",
        (split,),
    )


def _chunks(rows, max_rows, max_bytes):
    chunk = []
    used = 0
    for row in rows:
        # Each character occupies at most four UTF-8 bytes. Leave room for IDs and JSON.
        weight = 4 * len(row["qualifiers"]) + 2048
        if chunk and (len(chunk) == max_rows or used + weight > max_bytes):
            yield chunk
            chunk = []
            used = 0
        used += weight
        chunk.append(row)
    if chunk:
        yield chunk


def structural_groups(db, split, options, max_bytes):
    """Yield bounded groups, followed by unchanged triples outside their scope."""
    with db:
        _populate_groups(db, split, options)
    rows = db.execute(
        f"""SELECT g.first AS group_first,g.second AS group_second,{ASSERTION_COLUMNS}
        FROM structure.groups g CROSS JOIN assertions a ON a.id=g.assertion {ASSERTION_JOINS}
        ORDER BY g.first,g.second,g.assertion"""
    )
    for (first, _), group in groupby(
        rows, key=lambda row: (row["group_first"], row["group_second"])
    ):
        for chunk in _chunks(group, options.group_size, max_bytes):
            if options.target == "hierarchy":
                yield chunk, ("hierarchy", first)
            elif len(chunk) > 1:
                yield chunk, ("factorized",)
            else:
                yield chunk, ()
    remainder = (
        "(a.object IS NULL OR p.name<>?)"
        if options.target == "hierarchy"
        else "a.object IS NULL"
    )
    parameters = (
        (split, options.hierarchy_predicate)
        if options.target == "hierarchy"
        else (split,)
    )
    for row in db.execute(
        f"""SELECT {ASSERTION_COLUMNS} FROM assertions a {ASSERTION_JOINS}
        WHERE a.split=? AND {remainder} ORDER BY a.id""",
        parameters,
    ):
        yield (row,), ()


def encode_structure(tag, triples):
    """Factor out the predicate and object, preserving qualified occurrences."""
    subject, predicate, value, *tail = triples[0]
    members = [subject if not tail else [subject, *tail]]
    for subject, _, _, *tail in triples[1:]:
        members.append(subject if not tail else [subject, *tail])
    return [*tag, predicate, value["id"], members]


def expand_structure(sequence):
    """Recover triples from the serialized structural view, in member order."""
    if sequence[0] == "hierarchy":
        _, _, predicate, object_id, members = sequence
    elif sequence[0] == "factorized":
        _, predicate, object_id, members = sequence
    else:
        raise ValueError("unknown structural relation representation")
    value = {"kind": "entity", "id": object_id}
    for member in members:
        if isinstance(member, str):
            yield [member, predicate, value]
        else:
            subject, *tail = member
            yield [subject, predicate, value, *tail]
