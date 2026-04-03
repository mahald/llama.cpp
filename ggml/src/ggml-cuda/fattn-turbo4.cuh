#pragma once

// TurboQuant 4-bit Flash Attention helpers
// Warp-shuffle WHT: no shared memory needed.
//
// K vec_dot: nthreads_KQ=16, 8 elements/thread, 128 total
//   - 3 local butterfly stages (h=1,2,4)
//   - 4 warp-shuffle stages (h=8,16,32,64)
//
// V dequant: nthreads_V=32, 4 elements/thread, 128 total
//   - 2 local butterfly stages (h=1,2)
//   - 5 warp-shuffle stages (h=4,8,16,32,64)

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
// Helper: unpack one 3-bit centroid from packed qs[] array
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
// Helper: in-register butterfly for local WHT stages
// ============================================================
template <int N>
static __device__ __forceinline__ void turbo4_local_wht(float * vals) {
    // Butterfly stages where h < N (all pairs are within this thread's elements)
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
//   N = elements per thread
//   lane = thread's position within the cooperating group
// ============================================================
template <int N>
static __device__ __forceinline__ void turbo4_shuffle_wht(float * vals, int lane) {
    // For 128 elements total, max h = 64.
    // Shuffle stages start at h = N (first stage needing inter-thread exchange).
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
// K vec_dot: dot(inverse_WHT(centroids) * norm, Q)
//
// Template params match other vec_dot functions.
// D = head dimension (must be 128 for turbo4)
// nthreads = nthreads_KQ (must be 16 for turbo4)
//
// Q_v = half2 or float2 array, (D/2)/nthreads entries per thread
//       Thread t holds Q elements [D/nthreads * t, D/nthreads * t + D/nthreads - 1]
//       = 8 contiguous Q values as 4 half2 or float2 pairs
//
// Each of the 16 threads holds 8 K elements. WHT via 3 local + 4 shuffle stages.
// ============================================================
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    static_assert(D == 128, "turbo4 FA vec_dot requires D=128");
    static_assert(nthreads == 16, "turbo4 FA vec_dot requires nthreads_KQ=16");
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    const block_turbo4_0 * block = (const block_turbo4_0 *) K_c;
    const float norm = __half2float(block->norm);

    const int lane = threadIdx.x % nthreads; // 0..15
    const int elem_start = lane * 8;          // first of 8 elements this thread owns

    // 1. Unpack 8 centroids
    float kval[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        kval[i] = turbo4_unpack_centroid(block->qs, elem_start + i);
    }

    // 2. Inverse WHT (direction=1): s_first=s2, s_second=s1

    // Multiply by s2 (first sign array for inverse)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        kval[i] *= fattn_turbo_s2[elem_start + i];
    }

    // Local butterfly stages: h=1,2,4 (within 8 elements)
    turbo4_local_wht<8>(kval);

    // Shuffle butterfly stages: h=8,16,32,64
    turbo4_shuffle_wht<8>(kval, lane);

    // Scale by 1/sqrt(128) and multiply by s1 (second sign array for inverse)
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        kval[i] *= inv_sqrt_128 * fattn_turbo_s1[elem_start + i];
    }

    // 3. Scale by norm
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        kval[i] *= norm;
    }

    // 4. Dot product with Q
    //    Q_v has 4 entries (half2 or float2), covering 8 Q values.
    float sum = 0.0f;

#ifdef V_DOT2_F32_F16_AVAILABLE
    const half2 * Q_h2 = (const half2 *) Q_v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 q_f2 = __half22float2(Q_h2[i]);
        sum += kval[2*i + 0] * q_f2.x;
        sum += kval[2*i + 1] * q_f2.y;
    }
#else
    const float2 * Q_f2 = (const float2 *) Q_v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += kval[2*i + 0] * Q_f2[i].x;
        sum += kval[2*i + 1] * Q_f2[i].y;
    }
#endif // V_DOT2_F32_F16_AVAILABLE

    return sum;
}


// ============================================================
// V dequantize: inverse_WHT(centroids) * norm → output ne elements
//
// Called by all 32 warp threads simultaneously for the same V block.
// Thread threadIdx.x handles elements [i0, i0+ne-1] where i0 = threadIdx.x * ne.
// With ne=4 and 32 threads: 128 elements total.
//
// WHT via 2 local + 5 shuffle stages.
// ============================================================
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(
    const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {

    static_assert(ne == 4, "turbo4 FA V dequant requires ne=4 (32 threads × 4 = 128)");

    const block_turbo4_0 * block = (const block_turbo4_0 *) ((const char *)vx);

    // Determine which block this i0 belongs to.
    // turbo4: QK=128, so block = i0/128, local offset = i0%128.
    // In the FA kernel, V + k*nb21 already points to the right row.
    // Since QK_TURBO4=128=D, i0 is the local offset within the single block.
    const int64_t ib  = i0 / QK_TURBO4;
    const int local_i = (int)(i0 % QK_TURBO4);
    const block_turbo4_0 * blk = block + ib;

    const float norm = __half2float(blk->norm);
    const int lane = local_i / ne; // = threadIdx.x for the standard calling pattern

    // 1. Unpack 4 centroids
    float vals[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = turbo4_unpack_centroid(blk->qs, local_i + i);
    }

    // 2. Inverse WHT (direction=1): s_first=s2, s_second=s1

    // Multiply by s2
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] *= fattn_turbo_s2[local_i + i];
    }

    // Local butterfly stages: h=1,2 (within 4 elements)
    turbo4_local_wht<4>(vals);

    // Shuffle butterfly stages: h=4,8,16,32,64
    turbo4_shuffle_wht<4>(vals, lane);

    // Scale by 1/sqrt(128), multiply by s1, scale by norm
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] *= inv_sqrt_128 * fattn_turbo_s1[local_i + i] * norm;
    }

    // 3. Output
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 * dummy = nullptr; GGML_UNUSED(dummy); // suppress unused warning
#pragma unroll
        for (int l0 = 0; l0 < 4; l0 += 2) {
            ((half2 *) dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0 + 1]));
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            ((float *) dst)[l] = vals[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type for turbo4 V dequant");
    }
}
