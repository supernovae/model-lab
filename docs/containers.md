# Run Model Lab in a CPU container

The image installs `klab` directly. No host Python, uv, harness or Workshop
installation is needed. The [uv installation](../README.md#first-run) remains an
alternative. The current image targets Linux amd64 and CPU only.

## Upgrade notice for existing viewers

**Breaking change for the mutable `:cpu` tag:** after pulling the new CLI-first
image and recreating a container, an invocation without a command prints help
and exits 0 instead of starting the viewer. The default HTTP healthcheck is also
removed. Update viewer launch commands to pass `dashboard` and the explicit
healthcheck shown below before upgrading. Existing containers and images pinned
by an old digest keep their original behavior; pulling a tag does not change an
already-created container.

## Run the CLI and a separate web viewer

After publication and the one-time package visibility action below:

```sh
podman pull ghcr.io/supernovae/model-lab:cpu
IMAGE=ghcr.io/supernovae/model-lab:cpu

# A fresh job volume; this only prints help, without data/model downloads.
podman run --name lab-help --network=none --memory=4g --cpus=2 --pids-limit=256 \
  -v lab-job:/state "$IMAGE" --help

# Observe that job; keep viewer metadata separate from read-only job evidence.
podman run -d --name lab-web --memory=2g --cpus=1 --pids-limit=256 \
  -v lab-web:/state -v lab-job:/observe:ro \
  -p 127.0.0.1:8501:8501 \
  --health-cmd="python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2).read()\"" \
  --health-interval=10s --health-timeout=3s --health-start-period=30s \
  "$IMAGE" dashboard --address 0.0.0.0 \
  --runs-dir /observe/work/runs --lab-dir /observe/work/lab \
  --reports-dir /observe/work/reports --evidence-root /observe/evidence
```

Open <http://127.0.0.1:8501>. Stop the viewer with `podman stop lab-web`. Retain
logs with `podman logs lab-help` and `podman logs lab-web`; inspect exits with
`podman inspect`. Docker accepts the same commands with `docker` in place of
`podman`. An invocation with no arguments prints `klab --help` and exits. Bare verbs
and options go to Model Lab: `"$IMAGE" report TRY_ID` or `"$IMAGE" --help`.
Explicit `"$IMAGE" klab ...` still works. `klab`, `python`, `python3`,
`sh`, `bash`, and `streamlit` run with their original argv. Use an explicit path
for another executable, for example `"$IMAGE" /usr/bin/compare ...`. All other
bare words and options go to Model Lab: `compare` and `eval` always mean Lab
commands, even when an executable or shell builtin has the same name.
No arguments are evaluated as shell text.

CLI jobs have no image-level HTTP healthcheck. The viewer above opts into one
for its explicitly selected port; change the health URL too when using another
`dashboard --port`. A health response means the server is reachable, not that
an experiment succeeded. The dashboard launcher replaces itself with Streamlit,
so container stop signals reach the server and its exit status is retained.

For real work, use a fresh job directory/volume and the native `klab try`,
`report`, `probe`, `compare` or `explore` command from the existing guides.
Mount local configs/data at their declared container paths. `/state/work` is the
default work root; home, caches and temporary files also stay under `/state`.
The image starts offline; source acquisition requires an explicitly chosen
network-enabled invocation and overriding the relevant `HF_*_OFFLINE` variables.
No corpus, model or experiment is shipped or downloaded by the smoke test.

Existing worker-local storage such as `/srv/klab/jobs/JOB_ID` can replace
`lab-job` in the volume arguments. Create that fresh directory yourself under
the owned root, use a simple ID without slashes, and use a separate
`/srv/klab/viewers/VIEWER_ID` for viewer state. On rootless Podman add
`--userns=keep-id:uid=10001,gid=10001` to both commands so host-owned bind paths
match image UID 10001. On Docker use `--user "$(id -u):$(id -g)"` instead.
No chmod/chown or mount reconfiguration is needed. Identical paths on two workers
are separate local storage; this viewer does not aggregate remote machines.

Keep the localhost port mapping. The viewer is observational: writes to job
evidence (including interactive Explorer cache writes) are unavailable through
its read-only mount. Its native SQLite reader sees the live WAL/SHM files.

## Build locally

From the public Lab checkout:

```sh
podman build --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t klab-cpu:local .
python3 container/smoke.py --engine podman --image klab-cpu:local \
  --revision "$(git rev-parse HEAD)" --output /tmp/klab-cpu-smoke-new
```

Docker can build the same Dockerfile. Python 3.14 and uv base images are pinned by
digest. The frozen CPU dependency layer precedes the public application source
layer. The final image contains the installed public Lab package and dependencies,
not Git history, credentials, external repositories or development environments.
The smoke uses tiny CPU/SQLite fixtures, bounded containers, retained logs and
cleanup. It does not establish scientific results. CPU/memory/PID limits require
engine support on the chosen host.

## Publication and anonymous pulls

PR CI builds and smokes without publishing. Successful main builds publish the
same tested image through `GITHUB_TOKEN`, without rebuilding, as `:cpu` and
`:cpu-sha-FULL_COMMIT`. The workflow records the immutable registry digest;
use `ghcr.io/supernovae/model-lab@sha256:...` to pin an execution.
No PAT is needed by this workflow or by anonymous users of a public package.

**One-time owner action after first publication:** open the new `model-lab`
container package on GitHub, choose **Package settings → Change visibility →
Public**, and confirm. A public source repository alone does not make the package
public. See [GitHub's package visibility documentation](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).
This workflow does not change visibility or any existing private package.
Anonymous pull availability must be checked after this action; until then use a
local build instead of creating a token just to run the image.

For AMD gfx1100 packaging, see [the ROCm image guide](containers-rocm.md).
