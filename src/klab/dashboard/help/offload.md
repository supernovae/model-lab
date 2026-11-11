# Activation offload

Activation offload copies eligible autograd-saved activation storage to pageable
host memory and restores it for backward. It differs from activation
recomputation, which reruns computation, and from disk checkpoints, which
preserve durable restart state.

The synchronous reference path measures bytes sent to host and returned to the
device, peak live host storage, and synchronized transfer time. Transfer time
remains part of the update duration. Parameter/buffer storage and their aliases
are never offloaded; activation aliases share a single host copy until Autograd
releases every saved view.

A successful runtime probe proves only a tiny roundtrip/backward on that
accelerator. It does not prove capacity relief. MPS uses unified memory and
therefore reports no additional physical capacity from host offload. Any host
budget must account for the live copied activations in process RSS, and a
one-run measurement must not invent a no-offload baseline.
