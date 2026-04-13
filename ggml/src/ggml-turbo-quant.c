/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * ROTATION STRATEGY: Walsh-Hadamard Transform (WHT) rotation is applied
 * INSIDE quantize/dequantize for turbo4 (QK=128).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- Lloyd-Max centroids for 3-bit quantization ---------- */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static int nearest_centroid_3bit(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

/* ---------- Walsh-Hadamard Transform ---------- */
static const float turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static const float turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

static void turbo_wht_128(float * x, int direction) {
    const float * s_first  = (direction == 0) ? turbo_wht_s1 : turbo_wht_s2;
    const float * s_second = (direction == 0) ? turbo_wht_s2 : turbo_wht_s1;

    for (int i = 0; i < 128; i++) x[i] *= s_first[i];

    for (int h = 1; h < 128; h *= 2) {
        for (int j = 0; j < 128; j += h * 2) {
            for (int k = 0; k < h; k++) {
                const float a = x[j + k];
                const float b = x[j + k + h];
                x[j + k]     = a + b;
                x[j + k + h] = a - b;
            }
        }
    }

    const float scale = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 128; i++) x[i] *= scale;
    for (int i = 0; i < 128; i++) x[i] *= s_second[i];
}


/* ---------- TURBO3_0: 3-bit (QK=32) — no WHT ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        const float * xb = x + block * QK_TURBO3;
        float sum_sq = 0.0f;
        for (int i = 0; i < QK_TURBO3; i++) sum_sq += xb[i] * xb[i];
        const float norm = sqrtf(sum_sq);
        y[block].norm = GGML_FP32_TO_FP16(norm);
        const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        memset(y[block].qs, 0, QK_TURBO3 / 4);
        memset(y[block].signs, 0, QK_TURBO3 / 8);
        for (int i = 0; i < QK_TURBO3; i++) {
            const float val = xb[i] * inv_norm;
            const int idx = nearest_centroid_3bit(val);
            y[block].qs[i >> 2] |= (uint8_t)((idx & 0x3) << ((i & 3) * 2));
            if (idx & 0x4) {
                y[block].signs[i >> 3] |= (uint8_t)(1 << (i & 7));
            }
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            const int lo = (x[block].qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
            const int hi = (x[block].signs[j >> 3] >> (j & 7)) & 0x1;
            const int idx = lo | (hi << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_TURBO3 == 0);
    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}


/* ---------- TURBO4_0: 4-bit (QK=128) with WHT ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * xb = x + block * d;

        float sum_sq = 0.0f;
        for (int i = 0; i < d; i++) sum_sq += xb[i] * xb[i];
        const float norm = sqrtf(sum_sq);

        y[block].norm  = GGML_FP32_TO_FP16(norm);
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);

        const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

        float tmp[128];
        for (int i = 0; i < d; i++) tmp[i] = xb[i] * inv_norm;

        turbo_wht_128(tmp, 0);

        memset(y[block].qs, 0, d * 3 / 8);
        memset(y[block].signs, 0, d / 8);

        for (int i = 0; i < d; i++) {
            const int idx = nearest_centroid_3bit(tmp[i]);
            const int bit_offset = i * 3;
            const int byte_idx   = bit_offset / 8;
            const int bit_pos    = bit_offset % 8;
            y[block].qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
            }
            if (tmp[i] >= 0.0f) {
                y[block].signs[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);

        float tmp[128];
        for (int j = 0; j < d; j++) {
            const int bit_offset = j * 3;
            const int byte_idx   = bit_offset / 8;
            const int bit_pos    = bit_offset % 8;
            uint16_t raw = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            const int idx = (raw >> bit_pos) & 0x7;
            tmp[j] = CENTROIDS_3BIT[idx];
        }

        turbo_wht_128(tmp, 1);

        for (int j = 0; j < d; j++) {
            y[block * d + j] = tmp[j] * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_TURBO4 == 0);
    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size), n_per_row);
    }
    return nrows * row_size;
}
