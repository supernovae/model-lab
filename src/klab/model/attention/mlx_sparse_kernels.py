"""FP32 selected-block attention with explicit first-order Metal gradients.

Queries own forward/dQ rows; keys own dK/dV rows. No atomics, gathered K/V
copies, token-square scores, or host position loop are required. The reverse
key-owned scan checks the compact block-membership matrix before doing math.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_FORWARD = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint T = q_shape[2], H = q_shape[1];
    const uint b = row / (H * T), h = (row / T) % H, t = row % T;
    constexpr uint C = (D + 31) / 32;
    const float scale = 1.0f / metal::sqrt(float(D));
    const size_t qb = b*q_strides[0] + h*q_strides[1] + t*q_strides[2];
    float qv[C], acc[C];
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        qv[c] = d < D ? q[qb + d*q_strides[3]] : 0.0f;
        acc[c] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint s=0; s<ids_shape[1]; ++s) {
        const uint block = ids[t*ids_strides[0] + s*ids_strides[1]];
        if (!membership[t*membership_strides[0] + block*membership_strides[1]]) continue;
        for (uint offset=0; offset<BS; ++offset) {
            const uint p = block*BS + offset;
            if (p > t || p >= T) break;
            const size_t kb = b*k_strides[0] + h*k_strides[1] + p*k_strides[2];
            const size_t vb = b*v_strides[0] + h*v_strides[1] + p*v_strides[2];
            float dot = 0.0f;
            for (uint c=0; c<C; ++c) {
                const uint d = lane + 32*c;
                if (d < D) dot += qv[c] * k[kb + d*k_strides[3]];
            }
            const float score = metal::simd_sum(dot) * scale;
            const float next_maximum = metal::max(maximum, score);
            const float previous_scale = metal::exp(maximum - next_maximum);
            const float probability_scale = metal::exp(score - next_maximum);
            denominator = denominator*previous_scale + probability_scale;
            for (uint c=0; c<C; ++c) {
                const uint d = lane + 32*c;
                if (d < D) acc[c] = acc[c]*previous_scale + probability_scale*v[vb + d*v_strides[3]];
            }
            maximum = next_maximum;
        }
    }
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        if (d < D) result[size_t(row)*D + d] = acc[c] / denominator;
    }
    if (lane == 0) logsum[row] = maximum + metal::log(denominator);
"""

_DQUERY = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint T = q_shape[2], H = q_shape[1];
    const uint b = row / (H * T), h = (row / T) % H, t = row % T;
    constexpr uint C = (D + 31) / 32;
    const float scale = 1.0f / metal::sqrt(float(D));
    const size_t qb = b*q_strides[0] + h*q_strides[1] + t*q_strides[2];
    const size_t gb = b*go_strides[0] + h*go_strides[1] + t*go_strides[2];
    float qv[C], gv[C], acc[C], local_delta = 0.0f;
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        qv[c] = d < D ? q[qb + d*q_strides[3]] : 0.0f;
        gv[c] = d < D ? go[gb + d*go_strides[3]] : 0.0f;
        acc[c] = 0.0f;
        if (d < D) local_delta += gv[c] * result[size_t(row)*D + d];
    }
    const float delta = metal::simd_sum(local_delta);
    const float dl = glogsum[b*glogsum_strides[0] + h*glogsum_strides[1] + t*glogsum_strides[2]];
    for (uint s=0; s<ids_shape[1]; ++s) {
        const uint block = ids[t*ids_strides[0] + s*ids_strides[1]];
        if (!membership[t*membership_strides[0] + block*membership_strides[1]]) continue;
        for (uint offset=0; offset<BS; ++offset) {
            const uint p = block*BS + offset;
            if (p > t || p >= T) break;
            const size_t kb = b*k_strides[0] + h*k_strides[1] + p*k_strides[2];
            const size_t vb = b*v_strides[0] + h*v_strides[1] + p*v_strides[2];
            float dot = 0.0f, gv_dot = 0.0f;
            for (uint c=0; c<C; ++c) {
                const uint d = lane + 32*c;
                if (d < D) {
                    dot += qv[c]*k[kb + d*k_strides[3]];
                    gv_dot += gv[c]*v[vb + d*v_strides[3]];
                }
            }
            const float probability = metal::exp(metal::simd_sum(dot)*scale - logsum[row]);
            const float ds = probability * (metal::simd_sum(gv_dot) - delta + dl) * scale;
            for (uint c=0; c<C; ++c) {
                const uint d = lane + 32*c;
                if (d < D) acc[c] += ds*k[kb + d*k_strides[3]];
            }
        }
    }
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        if (d < D) dq[size_t(row)*D + d] = acc[c];
    }
    if (lane == 0) deltas[row] = delta;
"""

_DKEY_VALUE = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint T = q_shape[2], H = q_shape[1];
    const uint b = row / (H*T), h = (row/T) % H, p = row % T;
    constexpr uint C = (D + 31) / 32;
    const float scale = 1.0f / metal::sqrt(float(D));
    const size_t kb = b*k_strides[0] + h*k_strides[1] + p*k_strides[2];
    const size_t vb = b*v_strides[0] + h*v_strides[1] + p*v_strides[2];
    float kv[C], vv[C], kacc[C], vacc[C];
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        kv[c] = d < D ? k[kb + d*k_strides[3]] : 0.0f;
        vv[c] = d < D ? v[vb + d*v_strides[3]] : 0.0f;
        kacc[c] = vacc[c] = 0.0f;
    }
    for (uint t=p; t<T; ++t) {
        if (!membership[t*membership_strides[0] + (p/BS)*membership_strides[1]]) continue;
        const uint qr = (b*H+h)*T+t;
        const size_t qb = b*q_strides[0] + h*q_strides[1] + t*q_strides[2];
        const size_t gb = b*go_strides[0] + h*go_strides[1] + t*go_strides[2];
        float qv[C], gv[C], dot = 0.0f, gv_dot = 0.0f;
        for (uint c=0; c<C; ++c) {
            const uint d = lane + 32*c;
            qv[c] = d < D ? q[qb+d*q_strides[3]] : 0.0f;
            gv[c] = d < D ? go[gb+d*go_strides[3]] : 0.0f;
            dot += qv[c]*kv[c];
            gv_dot += gv[c]*vv[c];
        }
        const float probability = metal::exp(metal::simd_sum(dot)*scale - logsum[qr]);
        const float dl = glogsum[b*glogsum_strides[0] + h*glogsum_strides[1] + t*glogsum_strides[2]];
        const float ds = probability*(metal::simd_sum(gv_dot)-deltas[qr]+dl)*scale;
        for (uint c=0; c<C; ++c) {
            kacc[c] += ds*qv[c];
            vacc[c] += probability*gv[c];
        }
    }
    for (uint c=0; c<C; ++c) {
        const uint d = lane + 32*c;
        if (d < D) {
            dk[size_t(row)*D+d] = kacc[c];
            dv[size_t(row)*D+d] = vacc[c];
        }
    }
"""


@lru_cache(maxsize=16)
def selected_attention(block_size: int):
    """Build the explicit first-order rule once per selected-block geometry."""
    forward = mx.fast.metal_kernel(
        name="klab_sparse_forward",
        input_names=["q", "k", "v", "ids", "membership"],
        output_names=["result", "logsum"],
        source=_FORWARD,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )
    dquery = mx.fast.metal_kernel(
        name="klab_sparse_dquery",
        input_names=[
            "q",
            "k",
            "v",
            "ids",
            "membership",
            "result",
            "logsum",
            "go",
            "glogsum",
        ],
        output_names=["dq", "deltas"],
        source=_DQUERY,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )
    dkey_value = mx.fast.metal_kernel(
        name="klab_sparse_dkey_value",
        input_names=["q", "k", "v", "membership", "logsum", "go", "glogsum", "deltas"],
        output_names=["dk", "dv"],
        source=_DKEY_VALUE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )

    def arguments(q):
        return {
            "template": [("D", q.shape[-1]), ("BS", block_size)],
            "grid": (q.shape[0] * q.shape[1] * q.shape[2] * 32, 1, 1),
            "threadgroup": (32, 1, 1),
        }

    @mx.custom_function
    def attention(q, k, v, ids, membership):
        if any(value.dtype != mx.float32 for value in (q, k, v)):
            raise ValueError("native sparse Metal kernels currently require FP32 Q/K/V")
        output, logsum = forward(
            inputs=[q, k, v, ids, membership],
            output_shapes=[q.shape, q.shape[:-1]],
            output_dtypes=[mx.float32, mx.float32],
            **arguments(q),
        )
        return output, logsum

    @attention.vjp
    def backward(primals, cotangents, outputs):
        q, k, v, ids, membership = primals
        go, glogsum = cotangents
        result, logsum = outputs
        dq, deltas = dquery(
            inputs=[q, k, v, ids, membership, result, logsum, go, glogsum],
            output_shapes=[q.shape, q.shape[:-1]],
            output_dtypes=[mx.float32, mx.float32],
            **arguments(q),
        )
        dk, dv = dkey_value(
            inputs=[q, k, v, membership, logsum, go, glogsum, deltas],
            output_shapes=[k.shape, v.shape],
            output_dtypes=[mx.float32, mx.float32],
            **arguments(q),
        )
        return dq, dk, dv, mx.zeros_like(ids), mx.zeros_like(membership)

    return attention
