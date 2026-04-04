#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

// TurboQuant Lloyd-Max centroids for N(0, 1/d), d=128
// Must match CENTROIDS_3BIT in ggml-turbo-quant.c exactly
static __device__ const float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Midpoints between adjacent centroids for nearest-centroid search
// Must match nearest_centroid_3bit() in ggml-turbo-quant.c
static __device__ __forceinline__ int turbo_nearest_centroid_3bit(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

// TurboQuant 3-bit: PolarQuant with L2-norm scaling
// 3-bit index split: low 2 bits in qs[] (4 per byte), high 1 bit in signs[] (8 per byte)
// Dequant: value = CENTROIDS_3BIT[idx] * norm
// Graph applies TURBO_WHT rotation before SET_ROWS, so we just do centroid quantization here.
static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ x, block_turbo3_0 * __restrict__ y) {
    // Compute L2 norm
    float sum_sq = 0.0f;
    for (int i = 0; i < QK_TURBO3; i++) {
        sum_sq += x[i] * x[i];
    }
    const float norm = sqrtf(sum_sq);
    y->norm = __float2half(norm);

    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Zero output buffers
    for (int j = 0; j < QK_TURBO3 / 4; j++) { y->qs[j] = 0; }
    for (int j = 0; j < QK_TURBO3 / 8; j++) { y->signs[j] = 0; }

    for (int i = 0; i < QK_TURBO3; i++) {
        const float val = x[i] * inv_norm;

        const int idx = turbo_nearest_centroid_3bit(val);

        // Low 2 bits → qs[], 4 values per byte
        y->qs[i >> 2] |= (uint8_t)((idx & 0x3) << ((i & 3) * 2));

        // High 1 bit → signs[], 8 values per byte
        if (idx & 0x4) {
            y->signs[i >> 3] |= (uint8_t)(1 << (i & 7));
        }
    }
}

// WHT sign arrays for turbo4 quantize — must match CPU and turbo-wht.cu exactly
static __device__ const float turbo_wht_s1_cpy[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};
static __device__ const float turbo_wht_s2_cpy[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// Optimized serial WHT for single-thread device functions (quantize path)
// Fully unrolled butterfly with fused sign-multiply passes
// direction 0 = forward, 1 = inverse
static __device__ __forceinline__ void turbo_wht_128_serial(float * __restrict__ x, int direction) {
    const float * s_first  = (direction == 0) ? turbo_wht_s1_cpy : turbo_wht_s2_cpy;
    const float * s_second = (direction == 0) ? turbo_wht_s2_cpy : turbo_wht_s1_cpy;

    // Fused: sign multiply + first butterfly stage (h=1)
#pragma unroll
    for (int j = 0; j < 128; j += 2) {
        const float a = x[j]     * s_first[j];
        const float b = x[j + 1] * s_first[j + 1];
        x[j]     = a + b;
        x[j + 1] = a - b;
    }

    // Butterfly stages h=2..64 (fully unrolled)
#pragma unroll
    for (int h = 2; h < 128; h *= 2) {
#pragma unroll
        for (int j = 0; j < 128; j += h * 2) {
#pragma unroll
            for (int k = 0; k < h; k++) {
                const float a = x[j + k];
                const float b = x[j + k + h];
                x[j + k]     = a + b;
                x[j + k + h] = a - b;
            }
        }
    }

    // Fused: scale + second sign multiply
    constexpr float scale = 0.08838834764831845f; // 1/sqrt(128)
#pragma unroll
    for (int i = 0; i < 128; i++) {
        x[i] *= scale * s_second[i];
    }
}

// TurboQuant 4-bit: 3-bit PolarQuant indices + 1-bit QJL signs
// With WHT rotation before centroid search (matches CPU ggml-turbo-quant.c)
// Optimized: batch 3-bit packing via uint32_t accumulator
static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ x, block_turbo4_0 * __restrict__ y) {
    // 1. Compute L2 norm
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < QK_TURBO4; i++) {
        sum_sq += x[i] * x[i];
    }
    const float norm = sqrtf(sum_sq);

    y->norm  = __float2half(norm);
    y->rnorm = __float2half(0.0f);

    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // 2. Normalize into working buffer
    float tmp[128];
#pragma unroll
    for (int i = 0; i < QK_TURBO4; i++) tmp[i] = x[i] * inv_norm;

    // 3. Forward WHT — Gaussianizes the distribution
    turbo_wht_128_serial(tmp, 0);

    // 4. Centroid search + batch 3-bit packing
    // Process 8 elements at a time: 8 × 3 = 24 bits fits in 3 bytes
#pragma unroll
    for (int j = 0; j < QK_TURBO4 / 8; j++) { y->signs[j] = 0; }

    // Pack 8 centroids at a time into 3 bytes
#pragma unroll
    for (int group = 0; group < QK_TURBO4 / 8; group++) {
        uint32_t packed = 0;
        uint8_t  sign_byte = 0;
        const int base = group * 8;

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int idx = turbo_nearest_centroid_3bit(tmp[base + i]);
            packed |= ((uint32_t)(idx & 0x7)) << (i * 3);

            if (tmp[base + i] >= 0.0f) {
                sign_byte |= (uint8_t)(1 << i);
            }
        }

        // Write 3 packed bytes (24 bits)
        const int byte_start = group * 3;
        y->qs[byte_start]     = (uint8_t)(packed & 0xFF);
        y->qs[byte_start + 1] = (uint8_t)((packed >> 8) & 0xFF);
        y->qs[byte_start + 2] = (uint8_t)((packed >> 16) & 0xFF);

        y->signs[group] = sign_byte;
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void cpy_blck_f32_turbo3_0(const char * cxi, char * cdsti) {
    quantize_f32_turbo3_0_block((const float *)cxi, (block_turbo3_0 *)cdsti);
}

static __device__ void cpy_blck_f32_turbo4_0(const char * cxi, char * cdsti) {
    quantize_f32_turbo4_0_block((const float *)cxi, (block_turbo4_0 *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
