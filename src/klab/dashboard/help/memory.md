# Training memory

Training memory is shown as disjoint estimate buckets: resident weights, gradients, optimizer state, retained activations, attention working tensors, temporary workspace, and headroom. Expert and memory-table figures are subsets of those buckets, never extra memory added to the total.

Native allocator peaks are useful where the runtime exposes them. Sampled peaks, especially MPS measurements, are lower bounds. Process RSS is host memory and is not a second device-memory total.

MLX reports Metal active, peak, and cache memory separately. Its cache value is a framework cache measurement, not allocator-reserved memory or driver allocation. Missing, sampled, and native values are shown as different kinds of evidence and are never substituted for one another.
