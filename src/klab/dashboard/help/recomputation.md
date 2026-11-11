# Activation recomputation

Activation recomputation discards selected forward intermediates and reruns those blocks during backward. It can reduce retained activation memory while increasing compute. It is not a disk checkpoint and does not make an interrupted run resumable by itself.

The recompute call ratio records extra block forwards. A memory or throughput difference is evidence only when matched runs record both conditions.
