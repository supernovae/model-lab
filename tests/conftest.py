"""Process-wide CPU limits for the test suite.

This module is imported before test collection in both the xdist controller and
its workers.  Child processes inherit the environment established here.
"""

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _name in _CPU_THREAD_ENVIRONMENT:
    os.environ[_name] = "1"

_torch = sys.modules.get("torch")
if isinstance(_torch, ModuleType):
    _torch.set_num_threads(1)


# `-n auto` (pyproject addopts): one worker per usable CPU, capped so each gets
# ~2 GiB of currently available RAM (a worker plus a `klab try` child
# peaks near 1 GiB). Threads stay at 1 per worker above, so a 48-core box runs
# ~48 workers and a 4-vCPU CI runner or small laptop runs 4 or fewer.
_WORKER_RAM_BYTES = 2 * 1024**3


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    import psutil

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cpus = os.cpu_count() or 1
    available = psutil.virtual_memory().available
    return max(1, min(cpus, available // _WORKER_RAM_BYTES))


# Host-RAM planning (``klab.host_capacity``) refuses work when measured
# available RAM minus a reserve cannot fit one worker. Under xdist, and on a
# box shared with other jobs, that measurement depends on whatever else is
# running, so unrelated tests failed with "inadequate measured RAM after
# reserve". Every Model Lab RAM reader (reserves from ``total`` as well as
# checks on ``available``) goes through ``host_capacity.measure_memory``, so
# pinning it gives tests one fixed, ample host whatever the physical machine.
# A test that monkeypatches ``psutil.virtual_memory`` itself
# (tests/test_host_capacity.py and friends) still gets its own value, so
# low-memory and reserve logic stays covered.
_GIB = 1024**3
_TEST_HOST_RAM_ENV = "KLAB_TEST_HOST_RAM"
_CHILD_BOOTSTRAP = Path(__file__).with_name("child_bootstrap")


@pytest.fixture(autouse=True)
def _deterministic_host_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    from klab import host_capacity

    host_capacity_measure = host_capacity.measure_memory
    original = psutil.virtual_memory
    # Children launched by tests (``python -c``/``-m klab``) get the same
    # fixed host from a tests-only sitecustomize (tests/child_bootstrap); the
    # shipped measure_memory has no override, so this never reaches production.
    monkeypatch.setenv(_TEST_HOST_RAM_ENV, f"{64 * _GIB}:{48 * _GIB}")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            filter(None, (str(_CHILD_BOOTSTRAP), os.environ.get("PYTHONPATH")))
        ),
    )

    def measured() -> object:
        if psutil.virtual_memory is not original:
            return psutil.virtual_memory()
        return host_capacity_measure()._replace(total=64 * _GIB, available=48 * _GIB)

    monkeypatch.setattr(host_capacity, "measure_memory", measured)


# Fast/full split. Tests listed in tests/slow_tests.txt (by base node id, so
# every parametrized case follows) get the `slow` marker; `-m "not slow"` is
# the fast suite. See docs/test-speed.md.
_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")
_SLOW = frozenset(
    line.strip()
    for line in _SLOW_LIST.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)


# Known failures with an open issue (tests/known_failures.txt): skipped with
# the precise reason until the issue lands, never silently weakened.
_KNOWN = {
    node.strip(): reason.strip()
    for node, _, reason in (
        line.partition("|")
        for line in Path(__file__)
        .with_name("known_failures.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        base = item.nodeid.split("[", 1)[0]
        module = base.split("::", 1)[0]
        if base in _SLOW or module in _SLOW:
            item.add_marker(pytest.mark.slow)
        reason = _KNOWN.get(base) or _KNOWN.get(module)
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"known failure: {reason}"))
