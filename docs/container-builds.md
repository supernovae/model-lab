# Accelerator image builds

Linux accelerator variants are independent images in the public
`ghcr.io/supernovae/model-lab` package. A backend participates only when
`container/backends/NAME/config.json` and its complete hashed wheel lock exist.
Blocked backend status documents do not activate builds. macOS Metal needs native
delivery and is not provided by these Linux images.

Pull requests run standard-library planner tests and validate configured input
identities without downloading framework wheels. On trusted main, available
backends run independently in parallel; a manual workflow dispatch on main can
retry publication while reusing existing dependency digests; dispatch does not refresh them. Each job has a 60-minute ceiling. A first dependency build can
require several GB of disk/network even though no GPU or experiment runs.

Dependency images use `deps-BACKEND-SHA256` tags. The identity includes the complete
hashed wheel lock, backend config, every backend-local recipe/pin file, dependency Dockerfile (including pinned Python
3.14.4/OS and uv 0.12.17 image digests), build backend hash lock, validation program,
build-context allowlist. Registry lookup resolves
an existing identity to an exact digest; only an explicit manifest-unknown response or the exact requested reference
followed by `: not found` allows bootstrap. Generic missing-tool/reference text
does not. Authentication and transport failures stop the job. Workflow
concurrency serializes main runs, preventing competing writes of the same tag.
Existing dependency tags are never overwritten. The shared dependency recipe
and `.dockerignore` are conservative identity inputs even for a custom backend;
changes to either rebuild all enabled bases. Application-only edits do not. To refresh dependencies, review a
change to their pinned inputs; there is no unpinned refresh switch.

Application source, project metadata, README, license, startup and application Dockerfile changes do
not change the dependency identity. The application starts from its exact registry
dependency digest and installs only Lab, offline and without dependency resolution.
It does not copy the entire environment into another application layer. Shared application packages must retain their tested `uv.lock` versions; only
the Torch/Triton backend stack may diverge. The planner refuses unrelated version
upgrades before any build. New project dependencies must also update the committed backend lock; otherwise the application
compatibility check refuses publication. `uv pip
check` verifies the installed package requirements. Failed dependency compatibility,
framework-build verification or application smoke prevents application publication.

The smoke checks installed framework version/build identity, a two-element CPU
tensor and installed CLI help in a network-disabled, nonroot, read-only container
with finite memory/CPU/PID/tmpfs limits. It provides no GPU-device or driver
compatibility evidence. No GPU passthrough or model workload runs in CI. The same
tested local application image is pushed as `BACKEND-sha-COMMIT`; the retained JSON
artifact records source commit, dependency key/digest and application digest.
The application tag may change on an explicit rerun of the same source revision;
use the recorded digest for deployment. CPU delivery and its viewer smoke remain separate.

Publication uses the workflow's built-in GitHub token on main with package write
permission confined to the publication job. Only public Lab sources enter the
allowlisted context. The package's existing visibility setting applies to these
tags; no workflow changes repository or package visibility.

A backend may provide its own `container/backends/NAME/dependencies.Dockerfile`
when its supported userspace requires a different OS or an explicitly audited
source metapackage. That recipe and all its local input files participate in the
dependency identity. This does not change the thin application layer contract.

Local lightweight checks (no installs):

```sh
python3 -m unittest discover -s container -p 'test_image_plan.py'
python3 container/image_plan.py plan cuda
```

The planner command needs that backend's committed configuration and lock. Building
an image does not establish that the worker's driver, hardware or native Lab
backend supports every operation; use the backend-specific delivery notes before
an independently authorized device test.

### Build storage

Dependency installs hardlink uv cache files into the venv on the same build
filesystem, then clear the cache in the same RUN. The workflow uses the buildx
`docker` driver to build directly into the daemon store, avoiding a second
large image export from a separate builder store. It requires at least 12 GiB
free in `/var/lib/docker` before building and refuses with a clear diagnostic
otherwise; it never deletes host tools or unrelated images. This is a minimum
preflight, not a guarantee against concurrent disk consumption or future wheel
growth. The locally tested ROCm image is about 8.1 GB; GitHub bootstrap disk
capacity remains subject to the first trusted-main build.
