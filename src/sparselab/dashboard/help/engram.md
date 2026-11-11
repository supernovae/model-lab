# Reading Engram training diagnostics

Values are recorded by the trainer from forward-pass memory diagnostics and
weighted by valid prediction targets across the update's microbatches. Even
lookup and address counts can therefore be fractional; they are not lifetime
counters. Missing or non-finite values are unavailable, never zero.

- **Gate mean** shows the learned gate output, not a gradient or quality score.
- **Value norm / hidden norm** are mean vector L2 norms at the adapter.
- **Table utilization** is observed distinct addresses divided by capacity in
  that forward observation, not cumulative learned coverage.
- **Bucket reuse / collision count** include repeated input addresses. They
  cannot distinguish useful repeats from distinct n-grams hashing together.
- **Maximum address fraction** measures address concentration, not recall.

Multiport runs emit `engram/port/<block index>/…` when supported. The producer's
aggregate sums lookup and unique counts across ports, derives reuse/utilization,
and averages gate and norm measurements across ports. It is not another port.
Single-port and older runs may only have aggregate measurements.

These diagnostics do not establish that memory weights receive gradients or
improve predictions. Gradient/update checks and held-out evaluations are separate
evidence. No such values are inferred by this dashboard.
