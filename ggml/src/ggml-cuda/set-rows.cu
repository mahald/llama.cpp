#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// Warp-cooperative turbo4 quantization kernel
// 16 threads collaborate on one 128-element block via warp shuffles.
// Each thread handles 8 elements: load, normalize, WHT, centroid search, pack.
template <typename idx_t, int THREADS_PER_BLOCK, int GROUPS_PER_CUDA_BLK>
static __global__ void k_set_rows_turbo4_coop(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne_total,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00,
        const uint3   ne01,
        const uint3   ne02,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {

    const int group_id = threadIdx.x / THREADS_PER_BLOCK;
    const int lane     = threadIdx.x % THREADS_PER_BLOCK;
    const int64_t i    = (int64_t)blockIdx.x * GROUPS_PER_CUDA_BLK + group_id;

    if (i >= ne_total) {
        return;
    }

    // Index decomposition (same as original kernel)
    const int64_t i_base = i * QK_TURBO4;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_turbo4_0);

    const float * src_block = src0_row + i00;
    block_turbo4_0 * y = dst_row_ptr + i00 / QK_TURBO4;

    // --- Cooperative quantization: 16 threads, 8 elements each ---
    const int elem_start = lane * 8;

    // 1. Load 8 elements and compute partial sum_sq
    float vals[8];
    float partial_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        vals[k] = src_block[elem_start + k];
        partial_sq += vals[k] * vals[k];
    }

    // 2. Warp-reduce sum_sq across 16 threads
#pragma unroll
    for (int offset = 8; offset >= 1; offset >>= 1) {
        partial_sq += __shfl_xor_sync(0xFFFFFFFF, partial_sq, offset);
    }
    const float norm = sqrtf(partial_sq);
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // 3. Normalize
#pragma unroll
    for (int k = 0; k < 8; k++) {
        vals[k] *= inv_norm;
    }

    // 4. Forward WHT via warp shuffles (same as FA read path)
    //    s1 first, butterfly, then scale * s2
#pragma unroll
    for (int k = 0; k < 8; k++) {
        vals[k] *= turbo_wht_s1_cpy[elem_start + k];
    }

    // Local butterfly (h=1..4, within 8 elements)
#pragma unroll
    for (int h = 1; h < 8; h *= 2) {
#pragma unroll
        for (int j = 0; j < 8; j += 2 * h) {
#pragma unroll
            for (int k = 0; k < h; k++) {
                const float a = vals[j + k];
                const float b = vals[j + k + h];
                vals[j + k]     = a + b;
                vals[j + k + h] = a - b;
            }
        }
    }

    // Shuffle butterfly (h=8..64, across threads)
#pragma unroll
    for (int h = 8; h < 128; h *= 2) {
        const int lane_mask = h / 8;
        const bool is_top = (lane & lane_mask) == 0;

#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float partner = __shfl_xor_sync(0xFFFFFFFF, vals[k], lane_mask);
            vals[k] = is_top ? (vals[k] + partner) : (partner - vals[k]);
        }
    }

    constexpr float scale = 0.08838834764831845f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        vals[k] *= scale * turbo_wht_s2_cpy[elem_start + k];
    }

    // 5. Centroid search + batch pack (8 centroids → 3 bytes)
    uint32_t packed = 0;
    uint8_t  sign_byte = 0;

#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int idx = turbo_nearest_centroid_3bit(vals[k]);
        packed |= ((uint32_t)(idx & 0x7)) << (k * 3);
        if (vals[k] >= 0.0f) {
            sign_byte |= (uint8_t)(1 << k);
        }
    }

    // 6. Write output — each thread writes its own bytes (no conflicts)
    const int byte_start = lane * 3;
    y->qs[byte_start]     = (uint8_t)(packed & 0xFF);
    y->qs[byte_start + 1] = (uint8_t)((packed >> 8) & 0xFF);
    y->qs[byte_start + 2] = (uint8_t)((packed >> 16) & 0xFF);
    y->signs[lane] = sign_byte;

    // Thread 0 writes norm
    if (lane == 0) {
        y->norm  = __float2half(norm);
        y->rnorm = __float2half(0.0f);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_quant<idx_t, block_turbo3_0, QK_TURBO3, quantize_f32_turbo3_0_block>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        // Warp-cooperative turbo4 quantization: 16 threads per 128-element block
        // Uses warp-shuffle WHT instead of serial single-thread WHT
        constexpr int THREADS_PER_BLOCK = 16;
        constexpr int CUDA_BLK = 256;
        constexpr int GROUPS_PER_CUDA_BLK = CUDA_BLK / THREADS_PER_BLOCK;

        GGML_ASSERT(ne00 % QK_TURBO4 == 0);
        const int64_t ne_total_t4 = (ne00 * ne01 * ne02 * ne03) / QK_TURBO4;
        const int num_cuda_blocks = (ne_total_t4 + GROUPS_PER_CUDA_BLK - 1) / GROUPS_PER_CUDA_BLK;

        const int64_t s01 = nb01/sizeof(float);
        const int64_t s02 = nb02/sizeof(float);
        const int64_t s03 = nb03/sizeof(float);
        const int64_t s10 = nb10/sizeof(idx_t);
        const int64_t s11 = nb11/sizeof(idx_t);
        const int64_t s12 = nb12/sizeof(idx_t);
        const int64_t s1  = nb1;
        const int64_t s2  = nb2;
        const int64_t s3  = nb3;

        if (ne_total_t4 > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

            k_set_rows_turbo4_coop<idx_t, THREADS_PER_BLOCK, GROUPS_PER_CUDA_BLK>
                <<<num_cuda_blocks, CUDA_BLK, 0, stream>>>(
                src0_d, src1_d, (block_turbo4_0*)dst->data,
                ne_total_t4, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12, s1, s2, s3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
