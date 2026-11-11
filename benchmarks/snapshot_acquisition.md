# Snapshot acquisition batching: CPU evidence

Base: `b3a1630ec09dbddae36301b0acef4122623ed939`.
Offline synthetic stream, 5,000 training + 32 validation records, about 530 UTF-8
bytes per record, unique across splits. Includes source locking, snapshot journal,
export, cold verification, fsync and publication. No corpus/model downloads or
remote machine access. Python 3.14.7, Linux cloud filesystem; not a storage-device
qualification or a prediction of the Workshop run.

| Measurement | Base | Batched |
| --- | ---: | ---: |
| Elapsed seconds, run 1 | 2.061411 | 0.955744 |
| Elapsed seconds, run 2 | 1.847597 | 0.545331 |
| Elapsed seconds, run 3 | 2.019588 | 0.687784 |
| Median seconds | 2.019588 | 0.687784 |
| SQLite COMMIT statements | 5,034 | 24 |
| Capacity scans | 20,131 | 70 |
| Process peak RSS, KiB, range | 323,068–323,216 | 321,544–321,768 |
| Final owned file bytes | 14,258,291 | 14,258,291 |

Median speedup: **2.94×**. Every run produced manifest digest
`84655f6e019f33a8a9ab1ec0a7d0103cf57a70a92ff18065c3faeeffc9017a3e`.
Final bytes include the retained work journal and published snapshot, not a
peak-disk measurement. RSS includes Python, imports and the existing cached CPU
Torch package; it does not isolate SQLite memory. No OS caches were dropped.
Small timings vary with shared-host load; reduced commits/scans are deterministic
for this fixture, but time-triggered flushes can increase them on slower systems.

Reproduce in the provisioned locked CPU environment:

```sh
git show b3a1630ec09dbddae36301b0acef4122623ed939:src/klab/data/sources.py > /tmp/snapshot-base.py
uv run --locked --no-sync python benchmarks/snapshot_acquisition.py --baseline-source /tmp/snapshot-base.py
uv run --locked --no-sync python benchmarks/snapshot_acquisition.py
```

The baseline option executes explicitly selected trusted repository code. This
session used `UV_CACHE_DIR=/workspace/.uv-cache` and added the already-present CPU
Torch cache archive to `PYTHONPATH`, because the provisioned environment omitted
Torch. Nothing was installed or downloaded for these tests.

Focused coverage exercises deterministic split/dedup output, committed-prefix
source drift rejection, exception and abrupt-process-exit recovery at record,
byte and cooperative time boundaries, capacity loss before commit and during
export, small work allowances, oversized allowed records and rejected resource
caps. Three fork-based crash tests emit Python's multithreaded-fork deprecation
warning; all child processes exited as expected. Full suites, network acquisition,
accelerator work and real Workshop validation remain outside this check.

Limits: batching does not remove upstream decode allocations, transport ledger
costs or source replay on resume. FULL synchronization is unchanged. A pending
group is not durable progress. Capacity observations are not filesystem quotas;
other writers can consume space between checks. The 250 ms boundary is cooperative
and cannot interrupt blocked reads, commits or filesystem calls. SQLite readers
can still contend with commit/cache-spill locks. Validate throughput and peak
resources through the actual Workshop workflow before claiming host performance.
