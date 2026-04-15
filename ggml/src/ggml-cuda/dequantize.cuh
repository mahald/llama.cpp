#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// TurboQuant Lloyd-Max centroids for N(0, 1/d), d=128
// Must match CENTROIDS_3BIT in ggml-turbo-quant.c exactly
static __device__ const float turbo_centroids_3bit_dq[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// TurboQuant 3-bit PolarQuant dequantize
// 3-bit index split: low 2 bits in qs[] (4 per byte), high 1 bit in signs[] (8 per byte)
// Dequant: value = CENTROIDS_3BIT[idx] * norm
// Matches dequantize_row_turbo3_0 in ggml-turbo-quant.c
static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;

    const float norm = __half2float(x[ib].norm);

    // Element iqs
    {
        const int i = iqs;
        const int lo = (x[ib].qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
        const int hi = (x[ib].signs[i >> 3] >> (i & 7)) & 0x1;
        const int idx = lo | (hi << 2);
        v.x = turbo_centroids_3bit_dq[idx] * norm;
    }
    // Element iqs + 1
    {
        const int i = iqs + 1;
        const int lo = (x[ib].qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
        const int hi = (x[ib].signs[i >> 3] >> (i & 7)) & 0x1;
        const int idx = lo | (hi << 2);
        v.y = turbo_centroids_3bit_dq[idx] * norm;
    }
}

// TurboQuant 4-bit dequantize: 3-bit PolarQuant indices (sequential packing) + 1-bit QJL signs
// Per-element version (NO WHT) — used by legacy GET_ROWS template.
// For WHT-correct dequant, use dequantize_row_turbo4_0_cuda below.
static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;

    const float norm = __half2float(x[ib].norm);

    // Element iqs
    {
        const int i = iqs;
        const int bit_offset = i * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_pos    = bit_offset % 8;
        uint16_t raw = (uint16_t)x[ib].qs[byte_idx];
        if (byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            raw |= (uint16_t)x[ib].qs[byte_idx + 1] << 8;
        }
        const int idx = (raw >> bit_pos) & 0x7;
        v.x = turbo_centroids_3bit_dq[idx] * norm;
    }
    // Element iqs + 1
    {
        const int i = iqs + 1;
        const int bit_offset = i * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_pos    = bit_offset % 8;
        uint16_t raw = (uint16_t)x[ib].qs[byte_idx];
        if (byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            raw |= (uint16_t)x[ib].qs[byte_idx + 1] << 8;
        }
        const int idx = (raw >> bit_pos) & 0x7;
        v.y = turbo_centroids_3bit_dq[idx] * norm;
    }
}

// WHT sign arrays for full-block dequant — must match CPU and turbo-wht.cu exactly
static __device__ const float turbo_wht_s1_dq[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};
static __device__ const float turbo_wht_s2_dq[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// Full-block turbo4 dequant with inverse WHT (serial, single thread)
// Dequantizes all 128 elements of one block, applies inverse WHT, scales by norm.
// Use this for GET_ROWS and any path that needs correct reconstruction.
static __device__ void dequantize_block_turbo4_0_full(const block_turbo4_0 * __restrict__ x, float * __restrict__ out) {
    const float norm = __half2float(x->norm);

    // 1. Unpack all 128 centroid values
    float tmp[128];
    for (int j = 0; j < QK_TURBO4; j++) {
        const int bit_offset = j * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_pos    = bit_offset % 8;
        uint16_t raw = (uint16_t)x->qs[byte_idx];
        if (byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            raw |= (uint16_t)x->qs[byte_idx + 1] << 8;
        }
        const int idx = (raw >> bit_pos) & 0x7;
        tmp[j] = turbo_centroids_3bit_dq[idx];
    }

    // 2. Inverse WHT (direction = 1: s_first=s2, s_second=s1)
    for (int i = 0; i < 128; i++) tmp[i] *= turbo_wht_s2_dq[i];

    for (int h = 1; h < 128; h *= 2) {
        for (int j = 0; j < 128; j += h * 2) {
            for (int k = 0; k < h; k++) {
                float a = tmp[j + k];
                float b = tmp[j + k + h];
                tmp[j + k]     = a + b;
                tmp[j + k + h] = a - b;
            }
        }
    }

    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) tmp[i] *= scale;
    for (int i = 0; i < 128; i++) tmp[i] *= turbo_wht_s1_dq[i];

    // 3. Scale by norm
    for (int i = 0; i < 128; i++) out[i] = tmp[i] * norm;
}
