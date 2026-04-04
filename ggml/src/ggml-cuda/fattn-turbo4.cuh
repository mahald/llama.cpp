#pragma once

// TurboQuant 4-bit Flash Attention helpers — Lazy K + Lazy V
//
// Both K and V WHT operations are deferred to avoid per-position overhead:
//
// K path: Instead of inverse WHT on every K vector, apply a FORWARD WHT
//   to Q once (before the KV loop). Then Q_wht · centroids = Q · WHT_inv(centroids).
//   Per-position K cost: centroid unpack + dot + norm multiply. No WHT.
//
// V path: Accumulate attention-weighted centroids*norm in WHT space.
//   Apply a single inverse WHT to VKQ after the KV loop.
//   Per-position V cost: centroid unpack + norm multiply. No WHT.
//
// Result: WHT cost is O(1) per head instead of O(context_length).

// Lloyd-Max centroids for 3-bit quantization — must match ggml-turbo-quant.c
static __device__ const float fattn_turbo_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// WHT sign arrays — must match ggml-turbo-quant.c / dequantize.cuh
static __device__ const float fattn_turbo_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __device__ const float fattn_turbo_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// ============================================================
// Helper: unpack one 3-bit centroid (fallback, used rarely)
// ============================================================
static __device__ __forceinline__ float turbo4_unpack_centroid(
        const uint8_t * __restrict__ qs, int j) {
    const int bit_offset = j * 3;
    const int byte_idx   = bit_offset / 8;
    const int bit_pos    = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < QK_TURBO4 * 3 / 8) {
        raw |= (uint16_t)qs[byte_idx + 1] << 8;
    }
    const int idx = (raw >> bit_pos) & 0x7;
    return fattn_turbo_centroids[idx];
}

// ============================================================
// Batch decode: 8 centroids from byte-aligned position
//
// For K vec_dot: elem_start = lane * 8, so bit_offset = lane * 24
// which is always byte-aligned (24 % 8 = 0).
// One uint32_t load covers all 24 bits (8 × 3-bit indices).
// ============================================================
static __device__ __forceinline__ void turbo4_batch_unpack_8(
        const uint8_t * __restrict__ qs, int elem_start, float * __restrict__ out) {
    const int byte_start = (elem_start * 3) / 8;  // = lane * 3
    // Load 3 bytes (24 bits = 8 × 3-bit indices), no alignment requirement
    const uint32_t raw = (uint32_t)qs[byte_start]
                       | ((uint32_t)qs[byte_start + 1] << 8)
                       | ((uint32_t)qs[byte_start + 2] << 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = fattn_turbo_centroids[(raw >> (i * 3)) & 0x7];
    }
}

// ============================================================
// Batch decode: 4 centroids from arbitrary position
//
// For V dequant (ne=4): 12 bits needed, uint16_t covers all cases
// even with non-zero bit_pos (max bit_pos + 12 = 16).
// ============================================================
static __device__ __forceinline__ void turbo4_batch_unpack_4(
        const uint8_t * __restrict__ qs, int local_i, float * __restrict__ out) {
    const int bit_offset = local_i * 3;
    const int byte_idx   = bit_offset / 8;
    const int bit_pos    = bit_offset % 8;
    // Load 2 bytes (16 bits covers 12 data bits + up to 4 bit_pos offset)
    const uint32_t raw = (uint32_t)qs[byte_idx] | ((uint32_t)qs[byte_idx + 1] << 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = fattn_turbo_centroids[(raw >> (bit_pos + i * 3)) & 0x7];
    }
}

// ============================================================
// Helper: in-register butterfly for local WHT stages
// ============================================================
template <int N>
static __device__ __forceinline__ void turbo4_local_wht(float * vals) {
#pragma unroll
    for (int h = 1; h < N; h *= 2) {
#pragma unroll
        for (int j = 0; j < N; j += 2 * h) {
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float a = vals[j + k];
                const float b = vals[j + k + h];
                vals[j + k]     = a + b;
                vals[j + k + h] = a - b;
            }
        }
    }
}

// ============================================================
// Helper: warp-shuffle butterfly for inter-thread WHT stages
// ============================================================
template <int N>
static __device__ __forceinline__ void turbo4_shuffle_wht(float * vals, int lane) {
#pragma unroll
    for (int h = N; h < 128; h *= 2) {
        const int lane_mask = h / N;
        const bool is_top = (lane & lane_mask) == 0;

#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float partner = __shfl_xor_sync(0xFFFFFFFF, vals[i], lane_mask);
            vals[i] = is_top ? (vals[i] + partner) : (partner - vals[i]);
        }
    }
}


// ============================================================
// Q pre-transformation: forward WHT on Q registers
//
// Applied once before the KV loop. Transforms Q so that:
//   Q_wht · centroids = Q · WHT_inv(centroids)
//
// Math: Q_wht = inv_sqrt * diag(s2) * H * diag(s1) * Q
//   where H is the Hadamard butterfly (symmetric: H = H^T)
//
// After this, the K vec_dot just dots raw centroids with Q_wht.
//
// Template params must match the kernel's nthreads_KQ and Q_reg layout.
// nthreads_KQ = 16, each thread has (D/2)/16 half2 pairs = 4 per block.
// ============================================================
template <int D, int ncols, int nthreads_KQ>
static __device__ __forceinline__ void turbo4_pretransform_Q(
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2 Q_reg[][D/2/nthreads_KQ]
#else
    float2 Q_reg[][D/2/nthreads_KQ]
#endif
) {
    constexpr int n_k_blocks = D / QK_TURBO4;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int kq_lane = threadIdx.x % nthreads_KQ;  // 0..15
    const int elem_start = kq_lane * 8;

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int b = 0; b < n_k_blocks; ++b) {
            // Extract 8 float values from 4 half2/float2 pairs
            float qvals[8];
#ifdef V_DOT2_F32_F16_AVAILABLE
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f2 = __half22float2(Q_reg[j][b * 4 + i]);
                qvals[2*i]     = f2.x;
                qvals[2*i + 1] = f2.y;
            }
#else
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qvals[2*i]     = Q_reg[j][b * 4 + i].x;
                qvals[2*i + 1] = Q_reg[j][b * 4 + i].y;
            }
#endif

            // Forward WHT: s1 first, butterfly, then s2 * inv_sqrt
            // This is the transpose of the inverse WHT (which does s2, butterfly, s1)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qvals[i] *= fattn_turbo_s1[elem_start + i];
            }

            turbo4_local_wht<8>(qvals);
            turbo4_shuffle_wht<8>(qvals, kq_lane);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qvals[i] *= inv_sqrt_128 * fattn_turbo_s2[elem_start + i];
            }

            // Store back to Q_reg
#ifdef V_DOT2_F32_F16_AVAILABLE
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                Q_reg[j][b * 4 + i] = make_half2(__float2half(qvals[2*i]), __float2half(qvals[2*i + 1]));
            }
#else
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                Q_reg[j][b * 4 + i] = make_float2(qvals[2*i], qvals[2*i + 1]);
            }
#endif
        }
    }
}


// ============================================================
// K vec_dot: FAST mode — Q is pre-transformed, just dot centroids
//
// Q_v contains WHT-transformed Q values (from turbo4_pretransform_Q).
// Per-position cost: centroid unpack + dot product + norm scale.
// No WHT butterfly, no sign array lookups in the hot loop.
//
// Math: Q_wht · c * norm = Q · K_full
//   (inv_sqrt and sign arrays are baked into Q_wht)
// ============================================================
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    static_assert(D == 128 || D == 256 || D == 512,
                  "turbo4 FA vec_dot requires D=128, 256, or 512");
    static_assert(nthreads == 16, "turbo4 FA vec_dot requires nthreads_KQ=16");
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int n_blocks = D / QK_TURBO4;

    const int lane = threadIdx.x % nthreads;
    const int elem_start = lane * 8;

    float sum = 0.0f;

#pragma unroll
    for (int b = 0; b < n_blocks; ++b) {
        const block_turbo4_0 * block = (const block_turbo4_0 *)(K_c) + b;
        const float norm = __half2float(block->norm);

        // Unpack raw centroids (no WHT, no sign arrays)
        // Batch unpack 8 raw centroids (one uint32_t load)
        float kval[8];
        turbo4_batch_unpack_8(block->qs, elem_start, kval);

        // Dot with pre-transformed Q (inv_sqrt and signs baked in)
        float block_sum = 0.0f;

#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 * Q_h2 = (const half2 *) Q_v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 q_f2 = __half22float2(Q_h2[b * 4 + i]);
            block_sum += kval[2*i + 0] * q_f2.x;
            block_sum += kval[2*i + 1] * q_f2.y;
        }
#else
        const float2 * Q_f2 = (const float2 *) Q_v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            block_sum += kval[2*i + 0] * Q_f2[b * 4 + i].x;
            block_sum += kval[2*i + 1] * Q_f2[b * 4 + i].y;
        }
#endif
        sum += block_sum * norm;
    }

    return sum;
}


// ============================================================
// Batch decode: 16 centroids from byte-aligned position
//
// For V dequant (ne=16, D=512): local_i is multiple of 16,
// bit_offset = local_i * 3 is multiple of 48 (byte-aligned).
// 48 bits needed, uint64_t covers it.
// ============================================================
static __device__ __forceinline__ void turbo4_batch_unpack_16(
        const uint8_t * __restrict__ qs, int local_i, float * __restrict__ out) {
    const int byte_start = (local_i * 3) / 8;
    // Load 6 bytes (48 bits = 16 × 3-bit indices)
    const uint64_t raw = (uint64_t)qs[byte_start]
                       | ((uint64_t)qs[byte_start + 1] << 8)
                       | ((uint64_t)qs[byte_start + 2] << 16)
                       | ((uint64_t)qs[byte_start + 3] << 24)
                       | ((uint64_t)qs[byte_start + 4] << 32)
                       | ((uint64_t)qs[byte_start + 5] << 40);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = fattn_turbo_centroids[(raw >> (i * 3)) & 0x7];
    }
}


// ============================================================
// V dequantize: LAZY mode — centroid lookup + norm only, NO WHT
// ============================================================
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(
    const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {

    static_assert(ne == 4 || ne == 8 || ne == 16,
                  "turbo4 FA V dequant requires ne=4, 8, or 16");

    const block_turbo4_0 * block = (const block_turbo4_0 *) ((const char *)vx);

    const int64_t ib  = i0 / QK_TURBO4;
    const int local_i = (int)(i0 % QK_TURBO4);
    const block_turbo4_0 * blk = block + ib;

    const float norm = __half2float(blk->norm);

    // Batch decode centroids based on ne
    float vals[ne];
    if constexpr (ne == 4) {
        turbo4_batch_unpack_4(blk->qs, local_i, vals);
    } else if constexpr (ne == 8) {
        turbo4_batch_unpack_8(blk->qs, local_i, vals);
    } else {
        turbo4_batch_unpack_16(blk->qs, local_i, vals);
    }

    // Scale by norm
#pragma unroll
    for (int i = 0; i < ne; ++i) {
        vals[i] *= norm;
    }

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 * dummy = nullptr; GGML_UNUSED(dummy);
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0 + 1]));
        }
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = vals[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type for turbo4 V dequant");
    }
}


// ============================================================
// Post-processing: apply deferred inverse WHT to accumulated VKQ
//
// Called once after the KV loop. Converts VKQ from WHT space
// to real space via inverse WHT per 128-element block.
// ============================================================
template <int D, int ncols, int nthreads_V, int V_rows_per_thread>
static __device__ __forceinline__ void turbo4_post_process_VKQ(
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2 VKQ[][D/2/nthreads_V]
#else
    float2 VKQ[][D/2/nthreads_V]
#endif
) {
    constexpr int n_v_blocks = D / QK_TURBO4;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int v_lane  = threadIdx.x;
    const int local_i = v_lane * V_rows_per_thread;

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int vb = 0; vb < n_v_blocks; ++vb) {
            const int vkq_base = vb * ((QK_TURBO4 / 2) / nthreads_V);

            float vals[V_rows_per_thread];
#ifdef V_DOT2_F32_F16_AVAILABLE
#pragma unroll
            for (int i = 0; i < V_rows_per_thread; i += 2) {
                const float2 f2 = __half22float2(VKQ[j][vkq_base + i/2]);
                vals[i]   = f2.x;
                vals[i+1] = f2.y;
            }
#else
#pragma unroll
            for (int i = 0; i < V_rows_per_thread; i += 2) {
                vals[i]   = VKQ[j][vkq_base + i/2].x;
                vals[i+1] = VKQ[j][vkq_base + i/2].y;
            }
#endif

            // Inverse WHT: s2 → butterfly → s1 * inv_sqrt
#pragma unroll
            for (int i = 0; i < V_rows_per_thread; ++i) {
                vals[i] *= fattn_turbo_s2[local_i + i];
            }

            turbo4_local_wht<V_rows_per_thread>(vals);
            turbo4_shuffle_wht<V_rows_per_thread>(vals, v_lane);

#pragma unroll
            for (int i = 0; i < V_rows_per_thread; ++i) {
                vals[i] *= inv_sqrt_128 * fattn_turbo_s1[local_i + i];
            }

#ifdef V_DOT2_F32_F16_AVAILABLE
#pragma unroll
            for (int i = 0; i < V_rows_per_thread; i += 2) {
                VKQ[j][vkq_base + i/2] = make_half2(__float2half(vals[i]), __float2half(vals[i+1]));
            }
#else
#pragma unroll
            for (int i = 0; i < V_rows_per_thread; i += 2) {
                VKQ[j][vkq_base + i/2] = make_float2(vals[i], vals[i+1]);
            }
#endif
        }
    }
}
