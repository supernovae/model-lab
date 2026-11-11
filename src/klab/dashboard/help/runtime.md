# Runtime and precision

A run records its execution engine, backend, device identity, framework versions, and declared versus tested precision capabilities. `auto` precision remains a configuration request; only a recorded probe makes a backend/precision combination tested.

CPU, CUDA/ROCm, XPU, and MPS/Metal do not have interchangeable allocator measurements. A missing device measurement is **unavailable**, not zero. MPS and MLX use unified memory, so device capacity and host RAM must never be added as two independent pools.

The dashboard shows requested precision from the resolved configuration separately from the effective precision recorded by the runtime. An unavailable probe result is not evidence that the request executed.

Runtime forecasting is operational telemetry, not a guarantee. The Training
and Runtime pages show target-based live progress, optimizer-only ETA, the
latest bounded snapshot, and the separate final phase observation. Missing
history/device identity leaves planning unavailable; stale progress suspends
ETA rather than rewriting or stopping a run. The dashboard is read-only and
does not tune batch settings. See `docs/runtime.md` in the repository for
matching rules and limitations.
