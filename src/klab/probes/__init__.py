"""Fast-fail probe battery: cheap, versioned checks that filter ideas early.

``klab probe RUN [--vs BASELINE] [--config fast|standard|full]`` scores one
checkpoint with a fixed suite (held-out loss and calibration, top-1 agreement
and divergence vs. a baseline, degeneration of greedy generations, reworded
in-context fact recall, needle-in-context retrieval and optional
lm-evaluation-harness tasks), cheapest first, stopping early on a hard failure.
The result carries the suite's versioned digest and a machine-readable verdict
with a recommended next action. See docs/probes.md.
"""
