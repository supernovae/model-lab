"""Native HIP forward/backward kernels for selected-block attention."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from klab.workdir import ensure_work_dir

_MAX_HEAD_DIM = 2048

_HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
constexpr int kThreads = 256;
constexpr int kMaxValuesPerThread = 8;

__device__ float reduce_sum(float value, float* scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__global__ void selected_forward_kernel(
    const float* q, const float* k, const float* v, const bool* membership,
    float* output, float* logsum, int batch, int heads, int length, int dim,
    int block_size, int block_count,
    int64_t qs0, int64_t qs1, int64_t qs2, int64_t qs3,
    int64_t ks0, int64_t ks1, int64_t ks2, int64_t ks3,
    int64_t vs0, int64_t vs1, int64_t vs2, int64_t vs3) {
    const int row = blockIdx.x;
    const int t = row % length;
    const int h = (row / length) % heads;
    const int b = row / (length * heads);
    const int lane = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(dim));
    const float* query = q + b*qs0 + h*qs1 + t*qs2;
    const int membership_row = t * block_count;
    float accum[kMaxValuesPerThread] = {};
    __shared__ float scratch[kThreads];
    __shared__ float maximum_shared;
    __shared__ float denominator_shared;
    __shared__ float alpha_shared;
    __shared__ float weight_shared;
    if (lane == 0) {
        maximum_shared = -INFINITY;
        denominator_shared = 0.0f;
    }
    __syncthreads();

    for (int block = 0; block < block_count; ++block) {
        if (!membership[membership_row + block]) continue;
        const int end = min((block + 1) * block_size, t + 1);
        for (int p = block * block_size; p < end; ++p) {
            const float* key = k + b*ks0 + h*ks1 + p*ks2;
            const float* value = v + b*vs0 + h*vs1 + p*vs2;
            float partial = 0.0f;
            for (int item = 0; item < kMaxValuesPerThread; ++item) {
                const int d = lane + item * kThreads;
                if (d < dim) partial += query[d*qs3] * key[d*ks3];
            }
            const float dot = reduce_sum(partial, scratch);
            if (lane == 0) {
                const float score = dot * scale;
                const float next_maximum = fmaxf(maximum_shared, score);
                alpha_shared = denominator_shared == 0.0f
                    ? 0.0f : expf(maximum_shared - next_maximum);
                weight_shared = expf(score - next_maximum);
                denominator_shared = denominator_shared * alpha_shared + weight_shared;
                maximum_shared = next_maximum;
            }
            __syncthreads();
            for (int item = 0; item < kMaxValuesPerThread; ++item) {
                const int d = lane + item * kThreads;
                if (d < dim) {
                    const int local = item;
                    accum[local] = accum[local] * alpha_shared + weight_shared * value[d*vs3];
                }
            }
            __syncthreads();
        }
    }
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
        const int d = lane + item * kThreads;
        if (d < dim) output[row * dim + d] = accum[item] / denominator_shared;
    }
    if (lane == 0) logsum[row] = maximum_shared + logf(denominator_shared);
}

__global__ void selected_backward_query_kernel(
    const float* q, const float* k, const float* v, const bool* membership,
    const float* output, const float* logsum, const float* grad_output,
    float* grad_q, int batch, int heads, int length, int dim,
    int block_size, int block_count,
    int64_t qs0, int64_t qs1, int64_t qs2, int64_t qs3,
    int64_t ks0, int64_t ks1, int64_t ks2, int64_t ks3,
    int64_t vs0, int64_t vs1, int64_t vs2, int64_t vs3,
    int64_t gs0, int64_t gs1, int64_t gs2, int64_t gs3) {
    const int row = blockIdx.x;
    const int t = row % length;
    const int h = (row / length) % heads;
    const int b = row / (length * heads);
    const int lane = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(dim));
    const float* query = q + b*qs0 + h*qs1 + t*qs2;
    const float* grad = grad_output + b*gs0 + h*gs1 + t*gs2;
    float accum[kMaxValuesPerThread] = {};
    __shared__ float scratch[kThreads];
    __shared__ float probability_shared;
    __shared__ float ds_shared;
    __shared__ float delta_shared;
    const float* result = output + row * dim;
    float delta_part = 0.0f;
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
        const int d = lane + item * kThreads;
        if (d < dim) delta_part += grad[d*gs3] * result[d];
    }
    const float delta = reduce_sum(delta_part, scratch);
    if (lane == 0) delta_shared = delta;
    __syncthreads();
    const float lse = logsum[row];
    for (int block = 0; block < block_count; ++block) {
        if (!membership[t * block_count + block]) continue;
        const int end = min((block + 1) * block_size, t + 1);
        for (int p = block * block_size; p < end; ++p) {
            const float* key = k + b*ks0 + h*ks1 + p*ks2;
            const float* value = v + b*vs0 + h*vs1 + p*vs2;
            float score_part = 0.0f;
            float value_part = 0.0f;
            for (int item = 0; item < kMaxValuesPerThread; ++item) {
                const int d = lane + item * kThreads;
                if (d < dim) {
                    score_part += query[d*qs3] * key[d*ks3];
                    value_part += grad[d*gs3] * value[d*vs3];
                }
            }
            const float dot = reduce_sum(score_part, scratch);
            const float grad_value_dot = reduce_sum(value_part, scratch);
            if (lane == 0) {
                probability_shared = expf(dot * scale - lse);
                ds_shared = probability_shared * (grad_value_dot - delta_shared) * scale;
            }
            __syncthreads();
            for (int item = 0; item < kMaxValuesPerThread; ++item) {
                const int d = lane + item * kThreads;
                if (d < dim) accum[item] += ds_shared * key[d*ks3];
            }
            __syncthreads();
        }
    }
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
        const int d = lane + item * kThreads;
        if (d < dim) grad_q[row * dim + d] = accum[item];
    }
}

__global__ void selected_backward_key_value_kernel(
    const float* q, const float* k, const float* v, const bool* membership,
    const float* output, const float* logsum, const float* grad_output,
    float* grad_k, float* grad_v, int batch, int heads, int length, int dim,
    int block_size, int block_count,
    int64_t qs0, int64_t qs1, int64_t qs2, int64_t qs3,
    int64_t ks0, int64_t ks1, int64_t ks2, int64_t ks3,
    int64_t vs0, int64_t vs1, int64_t vs2, int64_t vs3,
    int64_t gs0, int64_t gs1, int64_t gs2, int64_t gs3) {
    const int row = blockIdx.x;
    const int p = row % length;
    const int h = (row / length) % heads;
    const int b = row / (length * heads);
    const int lane = threadIdx.x;
    const int key_block = p / block_size;
    const float scale = rsqrtf(static_cast<float>(dim));
    const float* key = k + b*ks0 + h*ks1 + p*ks2;
    const float* value = v + b*vs0 + h*vs1 + p*vs2;
    float key_accum[kMaxValuesPerThread] = {};
    float value_accum[kMaxValuesPerThread] = {};
    __shared__ float scratch[kThreads];
    __shared__ float probability_shared;
    __shared__ float ds_shared;
    for (int t = p; t < length; ++t) {
        if (!membership[t * block_count + key_block]) continue;
        const int query_row = (b * heads + h) * length + t;
        const float* query = q + b*qs0 + h*qs1 + t*qs2;
        const float* grad = grad_output + b*gs0 + h*gs1 + t*gs2;
        const float* result = output + query_row * dim;
        float score_part = 0.0f;
        float value_part = 0.0f;
        float delta_part = 0.0f;
        for (int item = 0; item < kMaxValuesPerThread; ++item) {
            const int d = lane + item * kThreads;
            if (d < dim) {
                score_part += query[d*qs3] * key[d*ks3];
                value_part += grad[d*gs3] * value[d*vs3];
                delta_part += grad[d*gs3] * result[d];
            }
        }
        const float dot = reduce_sum(score_part, scratch);
        const float grad_value_dot = reduce_sum(value_part, scratch);
        const float delta = reduce_sum(delta_part, scratch);
        if (lane == 0) {
            probability_shared = expf(dot * scale - logsum[query_row]);
            ds_shared = probability_shared * (grad_value_dot - delta) * scale;
        }
        __syncthreads();
        for (int item = 0; item < kMaxValuesPerThread; ++item) {
            const int d = lane + item * kThreads;
            if (d < dim) {
                key_accum[item] += ds_shared * query[d*qs3];
                value_accum[item] += probability_shared * grad[d*gs3];
            }
        }
        __syncthreads();
    }
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
        const int d = lane + item * kThreads;
        if (d < dim) {
            grad_k[row * dim + d] = key_accum[item];
            grad_v[row * dim + d] = value_accum[item];
        }
    }
}

extern "C" int sparse_forward(
    const float* q, const float* k, const float* v, const bool* membership,
    float* output, float* logsum, int batch, int heads, int length, int dim,
    int block_size, const int64_t* qs, const int64_t* ks, const int64_t* vs,
    void* stream_handle) {
    const int block_count = (length + block_size - 1) / block_size;
    const int rows = batch * heads * length;
    hipLaunchKernelGGL(selected_forward_kernel, dim3(rows), dim3(kThreads), 0,
        reinterpret_cast<hipStream_t>(stream_handle),
        q, k, v, membership, output, logsum, batch, heads, length, dim,
        block_size, block_count,
        qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3]);
    return static_cast<int>(hipGetLastError());
}

extern "C" int sparse_backward(
    const float* q, const float* k, const float* v, const bool* membership,
    const float* output, const float* logsum, const float* grad_output,
    float* grad_q, float* grad_k, float* grad_v, int batch, int heads,
    int length, int dim, int block_size, const int64_t* qs, const int64_t* ks,
    const int64_t* vs, const int64_t* gs, void* stream_handle) {
    const int block_count = (length + block_size - 1) / block_size;
    const int rows = batch * heads * length;
    const hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
    hipLaunchKernelGGL(selected_backward_query_kernel, dim3(rows), dim3(kThreads), 0, stream,
        q, k, v, membership, output, logsum, grad_output, grad_q,
        batch, heads, length, dim, block_size, block_count,
        qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3], gs[0], gs[1], gs[2], gs[3]);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return static_cast<int>(error);
    hipLaunchKernelGGL(selected_backward_key_value_kernel, dim3(rows), dim3(kThreads), 0, stream,
        q, k, v, membership, output, logsum, grad_output, grad_k, grad_v,
        batch, heads, length, dim, block_size, block_count,
        qs[0], qs[1], qs[2], qs[3], ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3], gs[0], gs[1], gs[2], gs[3]);
    return static_cast<int>(hipGetLastError());
}

extern "C" const char* sparse_error_string(int error) {
    return hipGetErrorString(static_cast<hipError_t>(error));
}
"""


@lru_cache(maxsize=4)
def _extension(arch: str):
    ensure_work_dir()
    if not torch.version.hip:
        raise RuntimeError("native HIP sparse attention requires a ROCm PyTorch build")
    compiler = shutil.which("hipcc")
    if compiler is None:
        raise RuntimeError("native HIP sparse attention requires hipcc on PATH")
    digest = hashlib.sha256(
        f"{arch}\n{torch.version.hip}\n{_HIP_SOURCE}".encode()
    ).hexdigest()[:16]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    build_dir = cache_root / "klab" / "hip" / digest
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "sparse_attention.hip"
    library_path = build_dir / "sparse_attention.so"
    sdk_root = Path(torch.__file__).resolve().parent.parent / "_rocm_sdk_core"
    sdk_lib = sdk_root / "lib"
    runtime_library = sdk_lib / "libamdhip64.so.7"
    if not runtime_library.exists():
        raise RuntimeError(f"ROCm HIP runtime library is missing: {runtime_library}")
    object_path = build_dir / "sparse_attention.o"
    linker = sdk_root / "lib" / "llvm" / "bin" / "clang++"
    with (build_dir / "build.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library_path.exists():
            source_path.write_text(_HIP_SOURCE)
            compile_command = [
                compiler,
                "-O3",
                "-fPIC",
                f"--offload-arch={arch}",
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ]
            link_command = [
                str(linker),
                "-shared",
                "-fPIC",
                str(object_path),
                str(runtime_library),
                f"-Wl,-rpath,{sdk_lib}",
                "-o",
                str(library_path),
            ]
            for command in (compile_command, link_command):
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"HIP sparse attention build failed:\n{error.stdout}{error.stderr}"
                    ) from error

    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.c_void_p
    stride_pointer = ctypes.POINTER(ctypes.c_int64)
    library.sparse_forward.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        pointer,
    ]
    library.sparse_forward.restype = ctypes.c_int
    library.sparse_backward.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        pointer,
    ]
    library.sparse_backward.restype = ctypes.c_int
    library.sparse_error_string.argtypes = [ctypes.c_int]
    library.sparse_error_string.restype = ctypes.c_char_p
    return library


def _pointer(tensor: Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _strides(tensor: Tensor):
    return (ctypes.c_int64 * 4)(*(int(value) for value in tensor.stride()))


def _check_launch(library, status: int) -> None:
    if status:
        message = library.sparse_error_string(status).decode()
        raise RuntimeError(f"HIP sparse attention kernel launch failed: {message}")


class _HIPSelectedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: Tensor, k: Tensor, v: Tensor, membership: Tensor, block_size: int
    ) -> Tensor:
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        logsum = torch.empty(q.shape[:-1], dtype=q.dtype, device=q.device)
        q_strides, k_strides, v_strides = _strides(q), _strides(k), _strides(v)
        stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        library = _extension(torch.cuda.get_device_properties(q.device).gcnArchName)
        status = library.sparse_forward(
            _pointer(q),
            _pointer(k),
            _pointer(v),
            _pointer(membership),
            _pointer(output),
            _pointer(logsum),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            block_size,
            q_strides,
            k_strides,
            v_strides,
            stream,
        )
        _check_launch(library, status)
        ctx.save_for_backward(q, k, v, membership, output, logsum)
        ctx.block_size = block_size
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q, k, v, membership, output, logsum = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q, grad_k, grad_v = (
            torch.empty(item.shape, dtype=item.dtype, device=item.device)
            for item in (q, k, v)
        )
        q_strides, k_strides, v_strides = _strides(q), _strides(k), _strides(v)
        grad_strides = _strides(grad_output)
        stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        library = _extension(torch.cuda.get_device_properties(q.device).gcnArchName)
        status = library.sparse_backward(
            _pointer(q),
            _pointer(k),
            _pointer(v),
            _pointer(membership),
            _pointer(output),
            _pointer(logsum),
            _pointer(grad_output),
            _pointer(grad_q),
            _pointer(grad_k),
            _pointer(grad_v),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            ctx.block_size,
            q_strides,
            k_strides,
            v_strides,
            grad_strides,
            stream,
        )
        _check_launch(library, status)
        return grad_q, grad_k, grad_v, None, None


def selected_attention(
    q: Tensor, k: Tensor, v: Tensor, membership: Tensor, block_size: int
) -> Tensor:
    """Run selected-block attention with deterministic query/key-owned HIP gradients."""
    if q.shape[-1] > _MAX_HEAD_DIM:
        raise ValueError(
            f"native HIP sparse attention supports head dimensions up to {_MAX_HEAD_DIM}"
        )
    if any(value.dtype != torch.float32 for value in (q, k, v)):
        raise ValueError("native HIP sparse attention currently requires FP32 Q/K/V")
    return _HIPSelectedAttention.apply(q, k, v, membership, block_size)
