# TurboQuant Speed Optimization Ideas

**Goal**: Make turbo KV cache faster than q8_0/f16 at attention, not just close the gap. turbo3 reads 3 bits/element vs q8_0's 8 or f16's 16 — a 2.7-5.3x bandwidth advantage currently wasted by dequanting to fp16 before compute.

**Current state**: Prefill 98.8% of q8_0, decode 95-97%. Close enough for launch but not a selling point.

**When attention speed matters:**
- Decode at 100K+: attention becomes bandwidth-bound, turbo's 3-bit reads dominate
- MoE models: FFN is cheaper (fewer active params), KV bandwidth is a bigger fraction
- Prefill: attention is larger fraction of total time

**What we tried and ruled out:**
- Experiment #88: Native VEC decode (read turbo directly) — 1% SLOWER, scalar math can't compete with tensor core MMA
- Experiment #89: Inverse FWHT K dequant — 0% speed, 2-7% KLD regression from fp16 precision loss
- Both confirm: FFN dominates decode at <32K on dense models; attention optimizations invisible

---

## Tier 0: New from NVIDIA Research (April 2026)

### Skip Softmax (tile-level attention skipping)
**Source**: TensorRT-LLM 2026 (developer.nvidia.com/blog)
**Idea**: During flash attention tiling, compare each tile's local max logit to running global max. If `m_local - m_global < -lambda`, skip the entire tile: no softmax, no BMM2, no V HBM load. ~50% of tiles skippable at long context. 1.4x throughput. Threshold auto-scales as `scale_factor / context_length`.
**Why it's #1**: This is strictly more powerful than our sparse V dequant. Sparse V skips individual positions after computing attention weights. Skip Softmax skips entire tiles BEFORE V is even loaded. They're complementary — Skip Softmax for tile-level, sparse V for position-level within surviving tiles.
**Implementation**: Our vec kernel already tracks KQ_max across tile iterations. Adding a skip check at the tile boundary is ~10 lines. The key insight: after computing KQ scores for a tile, if ALL scores in the tile are far below the running max, the entire V load + accumulation can be skipped.
**Estimated effort**: 1-2 days for prototype, 1 week for tuning threshold.

### BitDecoding TC-Aware Quant Layout (prefill)
**Source**: BitDecoding (HPCA 2026, arXiv:2503.18773)
**Idea**: Store quantized KV in tensor core fragment order. Use `ldmatrix` to load into registers where values already match MMA fragment layout. Dequant via `lop3` produces correctly-ordered operands. No reshuffling needed. 7.5x over FP16 FlashDecoding.
**Why**: Our current prefill dequants turbo->fp16 buffer then reads it back for MMA. This eliminates that round-trip entirely.
**Estimated effort**: 3-4 weeks. High difficulty.

### Marlin lop3 Dequant
**Source**: IST-DASLab Marlin (github.com/IST-DASLab/marlin)
**Idea**: Single PTX `lop3` instruction does ternary logic (AND+OR) to place quantized bits into FP16 mantissa with correct exponent. Two INT4->FP16 conversions per INT32 register. For our turbo3 (3 bits split across qs+signs), a lop3 chain could replace current bit extraction + centroid lookup.
**Estimated effort**: 1-2 weeks.

### LiquidGEMM IMAD+XOR (validates our approach)
**Source**: LiquidGEMM (SC'25, arXiv:2509.01229)
**Idea**: Rotation-based quant + IMAD+XOR dequant = 2 instructions per 4 elements. 10x fewer instructions than QServe. They independently converged on rotation before quantization (like our FWHT).
**Relevance**: Confirms our architecture. Their IMAD approach could apply to our turbo3 dequant.

### XOR Swizzle for Shared Memory
**Source**: Flash Attention bank conflict analysis (lubits.ch/flash/Part-4)
**Idea**: `swizzled_col = row ^ col` eliminates SMEM bank conflicts. 2x bandwidth utilization. Our fattn-vec kernel uses linear shared memory indexing — should audit.
**Estimated effort**: 1 day to audit, 1-2 days to implement if needed.

---

## Tier 1: Previously Identified, Most Promising

### ADC — Asymmetric Distance Computation for K
**Paper**: VecInfer (arXiv:2510.06175)
**Idea**: Precompute `Q_subvec * centroid` lookup table (8 entries for turbo3). Each K element's contribution is one table lookup instead of dequant+FMA. 256 lookups+adds vs 256 FMA. Bandwidth: 3 bits vs 16 bits per element.
**Blocker**: PQ incompatible with FWHT (350-1466% worse). Scalar VQ ADC (8 entries) trivially fits registers. Q must be pre-transformed (already is). Math works — question is whether scalar-ADC gives enough speedup vs table construction overhead.
**Estimated effort**: 2-3 weeks.

### BitDecoding-style MMA + Dequant Pipelining
**Papers**: BitDecoding (HPCA 2026), TurboMind (arXiv:2508.15601)
**Idea**: While tensor cores execute MMA on tile N, CUDA cores dequant tile N+1 in registers. Software pipeline hides dequant latency completely. `lop3` for bit manipulation, `ldmatrix` for TC layout.
**Blocker**: Requires restructuring fattn-mma-f16.cu. turbo3 qs+signs layout maps well to lop3.
**Estimated effort**: 3-4 weeks.

### INT8 Tensor Core Attention
**Papers**: SageAttention (ICLR 2025), SageAttention2 (ICML 2025)
**Idea**: Dequant turbo3 to INT8, use `mma u8.u8.s32` (2x throughput of fp16 MMA on Ampere). Prefill could be 1.5-2x FASTER than f16.
**Blocker**: Double-quantization error. K smoothing helps. Needs PPL validation.
**Estimated effort**: 2-3 weeks.

---

## Tier 2: Solid but Harder

### Fused Register Dequant (no fp16 intermediate)
Dequant turbo->fp16 directly in MMA fragment registers. Eliminates fp16 buffer write+read round-trip. Subsumes BitDecoding pipelining if done right. Requires deep PTX + `ldmatrix` swizzling knowledge. 3-4 weeks.

### Fast Encode (Lloyd-Max fallback for TCQ)
Lloyd-Max scalar quantize as fast encode alternative to Viterbi. Same bitstream format, ~10x faster encode (fully parallel). 0.5-1% PPL cost. Runtime flag. 1 week.

### TCQ Codebook -> Shared Memory
turbo3_tcq vec kernel dequant accesses 512-entry `__constant__` codebook. 128 threads hitting different entries serializes on constant cache. Loading 2KB codebook into shared memory gives 32-bank parallel access. 30 minutes to prototype.

### Warp-Level Cooperative FWHT for Encode
set-rows encoder does FWHT sequentially in one thread (~896 FLOPs per group). Warp-level FWHT via `__shfl_xor_sync`: 32 threads x 4 elements = 128 elements per warp. h=1,2 in-register; h=4,8,16,32,64 via shuffles. ~20 shuffle ops vs ~896 sequential FLOPs. ~45x faster. Only matters for prefill/batch encode.

### Vectorized Turbo3 Bit Extraction
Use `uint32_t` loads to grab 16 2-bit values at once instead of per-byte extraction. Micro-opt, 2-5% faster dequant. Only visible at very long context.

---

## Tier 3: Speculative

### Approximate Attention in Quantized Space
Compute approximate Q*K scores directly from 3-bit symbols, identify top-k K positions, only dequant those V rows. Extends sparse V from threshold-based to ranking-based.

### Warp-Specialized Producer/Consumer Pipeline
Dedicate warps to reading+dequanting (producer) vs MMA (consumer), connected via shared memory. Persistent kernels. Maximum overlap. Complex.

### Fused Q Rotation + Attention
Merge FWHT rotation kernel with vec attention kernel. Eliminates one kernel launch + one global memory round-trip for Q. But Q rotation is already negligible (experiment #89).

### CUDA Graphs
Package entire attention layer (FWHT + vec attn + V un-rotation) as CUDA Graph. Eliminates per-kernel launch overhead. 10-30% at short context where launch overhead dominates. NVIDIA already contributed this to upstream llama.cpp (10-15% reported).

---

## Key Learnings from Failed Experiments

1. **fp16 truncation in original domain worse than rotated domain** — FWHT makes values uniform, better for both quantization AND fp16. Don't inv_fwht before fp16 cast. (#89)
2. **Scalar VEC can't beat tensor core MMA** even with 5x bandwidth savings. Any "faster than q8_0" path must use tensor cores. (#88)
3. **FFN dominates decode at <32K on dense models** — attention optimizations invisible. Only matters at long context or on MoE. (#88, #89)
4. **Q rotation is free** — FWHT kernel on Q is negligible. Don't optimize it. (#89)
5. **TheTom tried 14 dequant-level optimizations, all failed** — constant memory LUT is at hardware floor. Bottleneck is how MANY values are dequantized, not HOW. This validates sparse V / Skip Softmax direction.

---

## Priority Order (what to try first)

1. **Skip Softmax** — biggest win, easiest to implement, works on Ampere
2. **TCQ codebook -> shared memory** — 30 min prototype, benchmark immediately
3. **XOR swizzle audit** — quick check, potentially significant
4. **lop3 dequant** — moderate effort, reduces instruction count
5. **INT8 tensor core attention** — high reward for prefill, needs quality validation
6. **BitDecoding TC-aware layout** — highest theoretical gain for prefill, hardest to implement
7. **ADC lookup** — interesting but scalar-VQ version may not be enough gain
8. **Fast encode (Lloyd-Max TCQ fallback)** — easy win for prefill-heavy workloads

---

## Beyond KV Cache: Full llama.cpp Decode Pipeline

### Where Time Actually Goes (single-user decode, RTX 3090)

| Component | % of decode time | llama.cpp vs best engine | Notes |
|-----------|-----------------|--------------------------|-------|
| Weight GEMM (mul_mat) | 85-90% | **40-80% slower** than Marlin/ExLlama | THE bottleneck |
| Kernel launch overhead | 5-10% | 10-15% recoverable w/ CUDA Graphs | Hundreds of launches per token |
| CPU graph construction | 2-5% | SGLang/vLLM V1 solved this | llama_graph_build per token |
| Attention (our domain) | 2-5% @ 2K | Competitive | Grows with context |
| Small ops (RMSNorm, RoPE) | <2% | Other engines fuse these | Separate kernel each |

### Why llama.cpp's Weight GEMM is Slow

ExLlamaV3 is **85% faster** than llama.cpp on RTX 3090 — and the gap is almost entirely weight GEMM:

1. **No fused dequant-GEMM**: llama.cpp dequants quantized weights → fp16 buffer → cuBLAS. ExLlama dequants directly in registers → tensor core MMA. Eliminates entire memory round-trip.
2. **No offline weight layout**: Marlin/ExLlama pre-arrange weight bits at quantize time so runtime is `cp.async` + `ldmatrix` (2 instructions). llama.cpp does layout work at runtime.
3. **dp4a vs tensor core**: llama.cpp MMQ uses INT8 dp4a with F32 accumulation (60 TFLOPS on q2_K). cuBLAS achieves 221 TFLOPS. Tensor core MMA is 4x faster.
4. **No tensor parallelism**: Can't split across GPUs.

### Techniques Other Engines Use That llama.cpp Doesn't

**Fused operations** (TensorRT-LLM, LMDeploy):
- AllReduce + RMSNorm + Quantize → 1 kernel
- QKV projection + RoPE + Reshape → 1 kernel (XQA)
- Residual add + RMSNorm → 1 kernel
- llama.cpp: each is separate launch

**CUDA Graphs** (SGLang, vLLM, NVIDIA contrib):
- Eliminate 15-25% CPU scheduling overhead
- SGLang: multi-stream capture, piecewise per-layer graphs
- NVIDIA contributed basic CUDA Graphs to llama.cpp (10-15%)
- llama.cpp rebuilds graph dynamically, limiting reuse

**Persistent batch tensors** (vLLM V1):
- Cache input tensors between tokens, only apply diffs
- Avoids reconstructing full tensor each step

**MoE-as-grouped-GEMM** (TensorRT-LLM):
- Send ALL tokens to ALL experts, mask outputs
- Grouped GEMMs are memory-bound so redundant compute is "free"
- Eliminates dispatch/reduction overhead
- llama.cpp dispatches per-expert

**FlashInfer GQA decode trick**:
- Use prefill (multi-query) kernel for GQA decode
- Tensor cores instead of CUDA cores
- 3x faster than PageAttention at batch_size=64
- llama.cpp uses CUDA-core vec kernel for all decode

**Speculative decoding** (all major engines):
- Draft model generates N candidates, main verifies in one pass
- 2-5x effective throughput
- llama.cpp has this but underutilized

---

## Competitive Landscape (April 2026)

### Speed Hierarchy (single-user decode, consumer GPU)
1. **ExLlamaV3** — fastest, 85% over llama.cpp. EXL3 uses QTIP/TCQ (validates our approach!)
2. **TensorRT-LLM** — fastest compiled, not interactive-friendly
3. **LMDeploy/TurboMind** — 93% BW utilization, best quantized attention kernel
4. **SGLang** — best with prefix caching (radix tree), 29% over vLLM
5. **vLLM** — batch throughput focus, broadest ecosystem
6. **llama.cpp** — most portable, competitive at batch 1, slowest weight GEMM

### KV Cache Quant Support
| Engine | Minimum | Sub-4-bit? |
|--------|---------|-----------|
| vLLM | FP8 | No |
| SGLang | FP4 | No |
| TRT-LLM | NVFP4 | No |
| LMDeploy | INT4 | No |
| ExLlamaV3 | 2-bit | Yes (basic) |
| **llama.cpp+turbo** | **TCQ 2-bit** | **Yes (unique)** |

**Nobody does sub-4-bit KV cache with quality in production. We're alone here.**

### ExLlamaV3 Uses TCQ (External Validation)
turboderp's EXL3 format is based on QTIP (Cornell RelaxML):
- 128x128 Hadamard transform (= our FWHT)
- Viterbi-optimal encoding through bitshift trellis (= our TCQ)
- Procedural codebook via MCG constant (we use trained codebooks)
- 16x16 tiles matching tensor core dimensions
Validates that TCQ + Hadamard is production-viable on GPU.

### BitDecoding Validates TC for Low-Bit KV
First tensor core system for low-bit KV cache attention:
- 4x speedup at 4-bit, 7x+ at 2-bit over FlashDecoding-v2
- Query reshape: [1, (gq, hkv)] → [gq, hkv] turns GEMV into GEMM
- lop3 dequant with 75316420 interleaving pattern
- Software pipeline: mma.sync tile i, ldmatrix+dequant tile i+1
Directly applicable to our turbo3/turbo4.

### TurboMind Has Best Quantized Attention
93% bandwidth utilization via 3-stage ILP overlap:
1. `mma.sync` on tile k (tensor cores)
2. I2F + FMA dequant on tile k+1 (ALUs)
3. `cp.async` prefetch tile k+2 (LD/ST)
64% more instructions but only 3% more wall time.
Their Adaptive Head Alignment for mixed FP16-Q/INT4-KV is directly applicable.

---

## Our Value Proposition

**We're not competing on raw tok/s** — llama.cpp at batch 1 is already competitive with vLLM.

**We're competing on context-per-VRAM at high quality:**
- turbo3_tcq at 3.25 bpv vs q8_0 at 8 bpv = 2.5x more context
- Quality: subjectively BETTER than q8_0 at 34K (regularization effect)
- No other engine offers sub-4-bit KV cache with TCQ quality

**Future speed opportunity**: If we port turbo types to BitDecoding-style TC kernels, we get both the memory savings AND faster attention at long context. That's the endgame.
