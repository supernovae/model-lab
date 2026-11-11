# Disk checkpoints and lineage

A verified checkpoint generation is an immutable disk snapshot. `latest` and `best` are lookup pointers; the generation manifest and file hashes are the durable evidence. A local best is the lowest evaluated, verified generation physically present in this run.

A resumed child can carry an ancestor lineage-best record. That ancestor may not be present locally, so it is displayed as lineage metadata rather than a local checkpoint pointer. Promotion copies canonical weights into fresh training state; it is not full resume.

PyTorch and MLX use the same generation manifest, canonical weight inventory, aliases, and file-hash verification. Each engine has its own native training-state codec: PyTorch uses its validated native state, while MLX stores validated typed JSON plus optimizer safetensors. Native MLX checkpoint files are resumable state, not training-only artifacts; offline verification does not require importing MLX.
