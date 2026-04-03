#include "turbo-wht.cuh"

// Sign arrays - must match CPU ops.cpp turbo_wht_s1/s2 exactly
static __device__ const float turbo_wht_s1[128] = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static __device__ const float turbo_wht_s2[128] = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

// One block per 128-element group, 128 threads per block
static __global__ void k_turbo_wht_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_groups,
        const int direction) {

    const int64_t group = (int64_t)blockIdx.x;
    if (group >= n_groups) return;

    const int tid = threadIdx.x; // 0..127

    const float * s_first  = (direction == 0) ? turbo_wht_s1 : turbo_wht_s2;
    const float * s_second = (direction == 0) ? turbo_wht_s2 : turbo_wht_s1;

    // Shared memory for butterfly
    __shared__ float x[128];

    // Load + first signs
    x[tid] = src[group * 128 + tid] * s_first[tid];
    __syncthreads();

    // WHT butterfly (7 stages: h = 1, 2, 4, 8, 16, 32, 64)
    for (int h = 1; h < 128; h *= 2) {
        // Each thread handles one butterfly pair
        const int block_size = h * 2;
        const int block_idx = tid / block_size;
        const int offset = tid % block_size;

        if (offset < h) {
            const int j = block_idx * block_size + offset;
            const float a = x[j];
            const float b = x[j + h];
            x[j]     = a + b;
            x[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize + second signs
    const float inv_sqrt_128 = 0.08838834764831845f;
    dst[group * 128 + tid] = x[tid] * inv_sqrt_128 * s_second[tid];
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->ne[0] % 128 == 0);

    const float * src_d = (const float *)src->data;
    float * dst_d = (float *)dst->data;

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_total = ggml_nelements(src);
    const int64_t n_groups = n_total / 128;

    cudaStream_t stream = ctx.stream();

    k_turbo_wht_f32<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_groups, direction);
}
