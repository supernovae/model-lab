#!/bin/bash
set -euo pipefail
# No implicit idle evidence, output reuse, or automatic next arm.
# Usage: execute-arm.sh ARM PREPARED_DIR FRESH_IDLE_JSON NEW_OUTPUT_DIR
SOURCE=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="/home/byron/Documents/Codex/2026-11-04/task-7/intel-26.40/python:$SOURCE/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_LOGS=recompiles,graph_breaks
export LD_LIBRARY_PATH=/home/byron/Documents/Codex/2026-11-04/task-7/intel-26.40/prefix/usr/lib/x86_64-linux-gnu:/home/byron/Documents/Codex/2026-11-04/task-7/intel-26.40/prefix/usr/local/lib
export ZE_ENABLE_ALT_DRIVERS=/home/byron/Documents/Codex/2026-11-04/task-7/intel-26.40/prefix/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
exec /mnt/sparselab/runtime/xpu-2.15.0-compare-20261103-r1/bin/python -B "$SOURCE/tools/document_microbatch/run_arm.py" --plan "$2/plan.json" --arm "$1" --preflight-receipt "$2/$1-admission.json" --host-idle-evidence "$3" --output "$4" --runtime-profile /mnt/sparselab/operations/lab86-torch215-compare-20261103-r1/runtime-profile.json --envelope /mnt/sparselab/projects/devmind/experiments/reasoning-spine-continuation-20261104-v1/resource-envelope.json --execute
