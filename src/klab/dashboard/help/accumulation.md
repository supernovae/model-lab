# Gradient accumulation

Gradient accumulation combines several microbatches into one successful optimizer update. Losses are weighted by valid target counts, while clipping, learning-rate scheduling, and the optimizer step happen once at the update boundary.

The effective batch and effective targets per update are observed values. A short final window may be smaller than the configured nominal batch.
