from __future__ import annotations

from shard_assignment import assign, load_durations, shard_of


def test_every_file_runs_in_exactly_one_shard_whatever_the_count() -> None:
    durations = {
        f"tests/test_{name}.py": seconds
        for name, seconds in zip("abcdefg", (90, 40, 40, 30, 10, 5, 1), strict=True)
    }
    files = [*durations, "tests/test_new.py", "tests/test_also_new.py"]
    for count in (1, 2, 3, 4):
        placed = assign(durations, count)
        for file in files:
            owners = [
                index
                for index in range(1, count + 1)
                if shard_of(file, count, placed) == index
            ]
            assert len(owners) == 1, (count, file)


def test_the_recorded_table_balances_the_shards_to_within_one_file() -> None:
    durations = load_durations()
    assert durations, "tests/shard_durations.json is missing or empty"
    for count in (2, 3):
        placed = assign(durations, count)
        loads = [
            sum(seconds for file, seconds in durations.items() if placed[file] == index)
            for index in range(1, count + 1)
        ]
        # Longest-first placement is within the largest single item of an even split.
        assert max(loads) - min(loads) <= max(durations.values()), (count, loads)
