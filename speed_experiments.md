# Speed Experiments — Consolidated from All Research

Every speed optimization idea from: original brainstorming, NVIDIA kernel research, inference engine reverse-engineering, and llama.cpp pipeline analysis. Organized by effort level.

**Current state** (Qwen3.5-27B Q6_K, RTX 3090, turbo3_tcq):
- Decode: 29.9 tok/s (96.5% of q8_0's 31.0)
- Prefill: 1125 tok/s (98.8% of q8_0's 1135)
- **85-90% of decode time is weight GEMM, NOT attention**

**KEY FINDING (2026-04-05)**: Attention is **<1% of decode time** on Qwen3.5-27B at ALL tested context lengths (up to 65K). Turbo3_tcq is only 0.3% slower at 65K vs 2K. Even f16 KV drops only 0.7% at 32K. ALL attention-only optimizations (S3, S5-S11, S14-S18) are invisible on this model. Only weight GEMM, pipeline, or prefill optimizations can show measurable results.

---

## Tier 0: Quick Tests (minutes to hours)

### S1. GGML_CUDA_FORCE_MMQ=ON rebuild
**Source**: llama.cpp build flag analysis
**What**: MMQ kernels may beat cuBLAS for Q6_K specifically on Ampere (sm86). Different codepath for weight GEMM.
**Effort**: 1 hour (rebuild + benchmark)
**Expected**: 0-10% decode change. Quick A/B test.
**Test**: Rebuild with `-DGGML_CUDA_FORCE_MMQ=ON`, run llama-bench tg64 at 4K/16K/32K.

### S2. GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 env var
**Source**: llama.cpp runtime flag
**What**: Force cuBLAS to use FP16 compute instead of FP32 accumulation. RTX 3090 FP16 tensor cores are 2x faster than FP32.
**Effort**: 5 minutes (env var, no rebuild)
**Expected**: 5-15% decode speedup if weight GEMM is cuBLAS-bound. Small PPL impact from FP16 accumulation.
**Test**: `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1 llama-bench ...`

### S3. TCQ codebook -> shared memory ❌ TESTED — NO IMPROVEMENT
**Source**: Original brainstorming
**What**: turbo3_tcq vec kernel dequant accesses 512-entry `__constant__` codebook. 128 threads hitting different entries serializes on constant cache (32B broadcast granularity). Loading 2KB codebook into shared memory gives 32-bank parallel access.
**Result**: 0% improvement at 2K/16K/32K context. Constant cache is NOT the bottleneck — 2KB fits entirely in 64KB constant cache on Ampere. Confirms TheTom's finding: bottleneck is HOW MANY values are dequantized, not HOW.

### S4. Speculative decoding (zero code changes) ❌ TESTED — SLOWER
**Source**: All major engines use this
**What**: Qwen3.5-2B Q4_K_M as draft + 27B Q6_K as target.
**Result (2026-04-05)**: Draft on CPU (no VRAM for both models): ~20 tok/s effective vs 31 tok/s normal. 35% slower. Draft model on CPU (26 tok/s) is the bottleneck. Qwen3-0.6B failed (token mismatch with Qwen3.5). Both models don't fit in 24GB VRAM simultaneously. Prior experiment #31 with GPU draft: 28.78 vs 31.0 (7% slower).
**Dead end**: 27B Q6_K leaves <3GB VRAM. No room for draft model on GPU. CPU draft too slow.

---

## Tier 1: Small Changes (1-3 days)

### S5. Skip Softmax (tile-level attention skipping) ❌ TESTED — NO IMPROVEMENT
**Source**: TensorRT-LLM 2026
**What**: Skip entire KV tiles when all scores are far below running max (threshold=20, exp(-20)≈2e-9).
**Result**: 0% improvement at 2K/16K/32K/65K. Correctly implemented but attention is <1% of decode time on Qwen3.5-27B (4 KV heads). Even at 65K, only 0.3% of decode is attention. Weight GEMM dominates >99%.
**NOTE**: This optimization IS valid — just invisible on this model. Would show gains on MoE (cheaper FFN), many KV heads, or 100K+ context.

### S6. XOR swizzle audit for shared memory ⏭️ SKIPPED — attention <1% on test model
**Source**: Flash Attention bank conflict analysis, BitDecoding
**What**: `swizzled_col = row ^ col` eliminates shared memory bank conflicts. 2x bandwidth utilization (93% wasted → 23% wasted). Our fattn-vec kernel uses linear shared memory indexing.
**Expected**: 0-30% attention kernel speedup, but attention is <1% of decode. Invisible on Qwen3.5-27B.
**Note**: Valid optimization for models with many KV heads or 100K+ context. Not worth testing on current hardware/model.

### S7. FP16 QK reduction ⏭️ SKIPPED — attention <1% on test model
**Source**: FlashInfer `allow_fp16_qk_reduction`
**What**: Use FP16 instead of FP32 for Q*K dot product accumulation. 50% speedup reported on RTX 4090.
**Expected**: Up to 50% attention speedup, but attention is <1% of decode. Invisible on Qwen3.5-27B.
**Note**: Would show gains at very long context (100K+) or many KV heads.

### S8. FireQ LUT formalization ⏭️ SKIPPED — attention <1% on test model
**Source**: FireQ (arXiv:2505.11594)
**What**: Precompute `codebook[state] * norm` LUT to remove multiply from inner loop.
**Expected**: 0-5% attention speedup. Invisible when attention is <1% of decode.

---

## Tier 2: Medium Effort (1-2 weeks)

### S9. lop3 dequant for turbo3/turbo4 ⏭️ SKIPPED — attention <1% on test model
**Source**: Marlin kernel (IST-DASLab), used in vLLM
**What**: `lop3` PTX instruction for bit extraction. 20-40% fewer dequant instructions.
**Expected**: Only visible at long context, attention <1% of decode on Qwen3.5-27B.
**Note**: Worth revisiting for models with many KV heads or 100K+ context.

### S10. LiquidGEMM IMAD+XOR dequant ⏭️ SKIPPED — attention <1% + TCQ incompatible
**Source**: LiquidGEMM (SC'25, arXiv:2509.01229)
**What**: 2-instruction dequant via IMAD+XOR. Requires linearly scaled values (incompatible with TCQ codebook).
**Note**: Blocked by both TCQ codebook non-linearity AND attention being <1% of decode.

### S11. Vectorized turbo3 bit extraction ⏭️ SKIPPED — attention <1% on test model
**Source**: Original brainstorming
**What**: `uint32_t` loads for batch bit extraction. 2-5% dequant speedup.
**Expected**: Only visible at very long context, attention <1% of decode on Qwen3.5-27B.

### S12. Fast encode — greedy fallback for TCQ ❌ TESTED — DEAD END
**Source**: Original brainstorming, Duster's TBQ encode is fully parallel
**What**: Greedy encode as fast alternative to Viterbi. Pick locally optimal trellis output at each step.
**Result**: Single-thread greedy: +8% prefill speed but PPL 17.09 (3x worse). Multi-start greedy (512 threads, argmin best): PPL 14.74 (still 2.5x worse) and NO speed gain (same compute as Viterbi, just trades syncthreads for parallelism). Greedy cannot match Viterbi quality because it lacks global path optimization. Dead end.
**Code**: Still in turbo-quant-cuda.cuh behind `TURBO_TCQ_FAST_ENCODE=1` env var (harmless, off by default).

### S13. Warp-level cooperative FWHT for encode ⏭️ LOW PRIORITY
**Source**: Original brainstorming
**What**: Warp-level FWHT via `__shfl_xor_sync` instead of shared memory + syncthreads.
**Analysis**: Current FWHT already parallelized (64 threads, 7 syncthreads). Warp shuffles would save 5 syncthreads, but FWHT is only ~10% of TCQ encode time — Viterbi (256 syncthreads) dominates. Saving 5/270 syncthreads ≈ 2% of encode ≈ 0.2% of prefill. Not worth the effort.
**Note**: Experiment #73 already captured the big FWHT win (+12.6% prefill). Diminishing returns from here.

---

## Tier 3: High Effort (2-4 weeks)

### S14. Scalar ADC for K dot product ⏭️ SKIPPED — attention <1% on test model
**Source**: VecInfer (arXiv:2510.06175), original brainstorming
**What**: Precompute `Q * centroid` LUT. Lookup instead of dequant+FMA.
**Expected**: Combines bandwidth + compute reduction. But attention <1% on Qwen3.5-27B.
**Note**: Math approach is sound. Would matter with many KV heads or very long context.

### S15. INT8 tensor core attention ⏭️ SKIPPED — attention <1% on test model
**Source**: SageAttention (ICLR 2025), SageAttention2 (ICML 2025)
**What**: `mma u8.u8.s32` for 2x attention throughput.
**Expected**: Only visible in prefill or very long context decode. Decode invisible on Qwen3.5-27B.

### S16. BitDecoding-style MMA + dequant pipelining ⏭️ SKIPPED — attention <1% on test model
**Source**: BitDecoding (HPCA 2026), TurboMind
**What**: 3-stage pipeline: MMA tile k + dequant tile k+1 + prefetch tile k+2.
**Expected**: Would eliminate dequant overhead, but attention is <1% of decode on Qwen3.5-27B.

### S17. Fused register dequant ⏭️ SKIPPED — attention <1% on test model
**Source**: Original brainstorming, BitDecoding
**What**: Dequant turbo→fp16 directly in MMA fragment registers.
**Expected**: Eliminates fp16 buffer round-trip, but invisible when attention <1%.

### S18. BitDecoding TC-aware quant layout ⏭️ SKIPPED — attention <1% on test model
**Source**: BitDecoding (HPCA 2026)
**What**: Store KV in tensor core fragment order. 7.5x over FP16 FlashDecoding for prefill.
**Expected**: Huge prefill speedup but format-breaking change. Would help prefill, invisible for decode.

---

## Tier 4: Architectural / Long-term (months)

### S19. Marlin-style Q6_K GEMM kernel ❌ PROFILED — AT HARDWARE WALL + FUSED QUANTIZE ❌ TESTED
**Source**: Inference engine landscape research
**What**: Rewrite weight GEMM kernel with Marlin-style pipelining, lop3 dequant, fused register ops.
**Profile result (ncu, 2026-04-05)**: MMVQ kernel already at **88-94% peak DRAM bandwidth** for large layers (17408, 12288, 5120 rows). Only small layers (1024 rows) are at 50% due to tail effects. No fp16 buffer exists — current kernel already does in-register dequant with DP4A. 40 registers/thread, 46 active warps.
**Revised estimate**: 0-5% gain possible (only from small-matrix tail effects). Original 40-80% estimate was wrong — based on ExLlamaV3's advantage which comes from full-stack optimization (persistent kernels, fused ops), not just the weight GEMM kernel.
**Conclusion**: Not worth weeks of kernel engineering for 0-5% gain. The kernel is already near-optimal.

### S30. Viterbi double-buffered cost + global bt ✅ MERGED — +0.6% MoE, +0.5% dense
**Source**: ncu profiling of Viterbi GPU underutilization (experiment/viterbi-opt)
**What**: Move 32KB backtrace from shared memory to global, double-buffer cost arrays to eliminate 2/3 of __syncthreads (384→128 per group). Byte-packed bt replaces nibble-packed (removes even/odd thread sync).
**Result (2026-04-04)**:
- PPL: **bit-exact** (6.2186 baseline = 6.2186 optimized)
- Dense (Qwen3.5-27B): 29.56→29.71 tok/s (+0.5%)
- MoE (Qwen3.5-35B-A3B): 126.22→126.97 tok/s (+0.6%)
- TCQ overhead reduced from 4.6% to 4.1% vs turbo3 (no TCQ)
- Shared memory: 35KB→5KB per block
- Bank conflict COST_PAD fix tried and reverted — ALU overhead > bank conflict savings
**Note**: Fundamental limit: 4-8 blocks on 82 SMs = 95% GPU idle during Viterbi. No per-block optimization can fix this.

### S20. Kernel fusion (AllReduce+RMSNorm, QKV+RoPE) — PARTIAL TEST: f32 activation in MMVQ ❌
**Source**: TensorRT-LLM, LMDeploy
**What**: llama.cpp launches separate kernels for RMSNorm, RoPE, residual add, etc. TRT-LLM fuses AllReduce+RMSNorm+Quantize into one kernel, QKV projection+RoPE+Reshape into one kernel.
**Effort**: Weeks per fusion.
**Expected**: 5-15% total by eliminating kernel launch overhead and intermediate memory traffic.
**Note**: llama.cpp's modular architecture makes this hard. Each op is a separate ggml node.
**Tested (2026-04-04)**: Three approaches attempted:
1. **S20 f32 FMA** (replace DP4A): -17.4% (24.82 tok/s). Compute-bound without DP4A.
2. **S19 in-register quantize** (keep DP4A, quantize per-thread): -31.5% (20.57 tok/s). SFU + rpb=1 redundancy.
3. **S19 shared memory quantize** (keep DP4A, cooperative quantize): -71.0% (8.70 tok/s). Phase 1 overhead (940 ns/block) exceeds Phase 2 work (260 ns/block) with rpb=1.
**Conclusion**: The separate `quantize_q8_1` kernel is already optimal. Per-block redundant quantization always loses to quantize-once-distribute-via-L2. The 2.3% kernel launch overhead is irreducible without CUDA Graphs (see S21).

### S21. CUDA Graphs for full decode pipeline
**Source**: NVIDIA contribution to llama.cpp, SGLang, vLLM
**What**: Package entire decode step as CUDA Graph. Eliminates per-kernel launch overhead (hundreds of launches per token). NVIDIA already contributed basic CUDA Graphs to upstream llama.cpp (10-15% reported).
**Effort**: 1-2 weeks for basic, more for piecewise/multi-stream.
**Expected**: 10-15% decode speedup. SGLang uses multi-stream capture, piecewise per-layer graphs.
**Note**: llama.cpp rebuilds graph dynamically per token, limiting reuse. Need fixed-shape decode path.

### S22. FlashInfer GQA decode trick
**Source**: FlashInfer (MLSys 2025 Best Paper)
**What**: Use prefill (multi-query) kernel for GQA decode. Tensor cores instead of CUDA cores. 3x faster than PageAttention at batch_size=64.
**Effort**: 2-3 weeks to adapt to our kernel structure.
**Expected**: Only matters at batch>1. At batch=1, VEC is appropriate. But for multi-user serving, this is significant.
**How**: For Qwen3.5-27B with 24 Q heads / 4 KV heads (GQA ratio 6:1), reshape Q to treat 6 queries as a "batch" dimension and use MMA instead of VEC for decode.

### S23. Persistent batch tensors (vLLM V1 trick)
**Source**: vLLM V1 engine
**What**: Cache input tensors between tokens, only apply diffs. Avoids reconstructing the full input tensor each decode step.
**Effort**: Significant llama.cpp architecture changes.
**Expected**: 2-5% from reduced CPU overhead.

### S24. Warp-specialized producer/consumer pipeline
**Source**: Original brainstorming, FlashAttention-3 Hopper
**What**: Dedicate warps to reading+dequanting (producer) vs MMA (consumer), connected via shared memory. Producer warps use few registers (donate rest to consumers). Persistent kernels.
**Effort**: 4-6 weeks.
**Expected**: Maximum overlap of memory and compute. Complex to implement and debug.
**Blocker**: Hopper-specific (warp specialization via `setmaxnreg`). On Ampere, use software pipeline (S16) instead.

### S25. Approximate attention in quantized space
**Source**: Original brainstorming
**What**: Compute approximate Q*K scores directly from 3-bit symbols (without dequant), identify top-k K positions, only dequant those V rows. Extends sparse V (#51) from threshold-based to ranking-based.
**Effort**: 3-4 weeks.
**Expected**: Combined bandwidth and compute savings. Speculative.

### S26. Blackwell native FP4/FP6 tensor cores
**Source**: NVIDIA Blackwell architecture
**What**: On B200/RTX5090, turbo3's 3-bit values could be zero-padded to 4-bit and use native FP4 tensor cores. `tcgen05.mma` with E2M1 inputs. 7703 TFLOPS at FP4. Sub-byte TMA handles packing/unpacking in hardware.
**Effort**: Medium (when targeting Blackwell hardware).
**Expected**: Eliminates dequant bottleneck entirely for Q*K matmul.
**Blocker**: Requires Blackwell GPU (RTX 5090 / B200).

### S27. MoE-as-grouped-GEMM
**Source**: TensorRT-LLM
**What**: Send ALL tokens to ALL experts, mask outputs. Grouped GEMMs are memory-bound so redundant compute is "free". Eliminates dispatch/reduction overhead. llama.cpp dispatches per-expert.
**Effort**: Weeks. Major architecture change.
**Expected**: Significant for MoE models (Qwen3.5-35B-A3B, Gemma-4-26B).

### S28. Fused Q rotation + attention kernel
**Source**: Original brainstorming
**What**: Merge FWHT rotation kernel with vec attention kernel. Eliminates one kernel launch + one global memory round-trip for Q.
**Effort**: 1-2 weeks.
**Expected**: Near zero. Q rotation is already negligible (confirmed by experiment #89). Only worth doing for code cleanliness.
**Status**: Deprioritized based on profiling data.

### S29. Persistent mega-kernel (Mirage)
**Source**: Inference engine research
**What**: Entire LLM forward pass in single CUDA kernel. Eliminates all kernel launch overhead. 1.0-1.7x reported.
**Effort**: Months. Fundamental architecture rewrite.
**Expected**: 1.0-1.7x total throughput. Unrealistic for llama.cpp's modular design.

---

## Already Tested — Results

| # | Experiment | Result | Why |
|---|-----------|--------|-----|
| #88 | Native VEC decode (read turbo directly) | **1% SLOWER** | Scalar math can't compete with tensor core MMA even with 5x bandwidth savings |
| #89 | Inverse FWHT K dequant | **0% speed, KLD regression** | fp16 truncation worse in original domain than rotated domain |
| #72 | Chunked cuBLAS GEMM prefill | **1-5% SLOWER** | Fused MMA avoids materializing O(nq*nkv) score matrix |
| #71 | Native vec_dot TCQ inline | **0% improvement** | Dequant-to-f16 path already fast enough |
| #49 | Parallel blocks tuning (split-K) | **NO EFFECT** | Attention <5% of decode — FFN dominates |
| #25b | Sign+magnitude encoding | **NEUTRAL** | Bottleneck is bandwidth, not ALU/register pressure |
| #31 | Speculative decoding (2B→27B) | **SLOWER** | Poor acceptance rate for this model pair |
| #32 | Fused quant in QKV projection | **Deferred** | SET_ROWS already negligible, prefill at 98.8% |
| #18 | SAS softmax optimization | **Dropped** | 5-15% attn speedup × <5% attn share = <0.75% total |

---

## Key Learnings

1. **85-90% of decode is weight GEMM** — attention optimizations are invisible at <32K on dense models
2. **Weight quant directly determines decode speed** — Q4_K_M→Q6_K = 24.6% slower, proportional to weight size. KV quant = fixed ~4.5% overhead.
3. **Attention is <1% of decode at ALL contexts tested (up to 65K)** on Qwen3.5-27B (4 KV heads). S3, S5 both confirmed 0% improvement.
4. **Greedy TCQ encode can't match Viterbi quality** — 2.5x PPL regression with multi-start greedy (S12). Same compute as Viterbi (trades syncthreads for parallelism), 3090 handles both equally well.
5. **Speculative decoding needs VRAM headroom** — 27B Q6_K (21GB) leaves no room for GPU draft on 24GB. CPU draft is 35% slower (S4).
6. **Scalar VEC can't beat tensor core MMA** even with 5x bandwidth savings (#88)
7. **fp16 in rotated domain > original domain** — FWHT makes values uniform, better for fp16 (#89)
8. **Q rotation is free** — FWHT on Q is negligible vs FFN (#89)
9. **TheTom tried 14 dequant-level optimizations, all failed** — constant memory LUT is at hardware floor. Bottleneck is HOW MANY values are dequantized, not HOW (#51 sparse V is the right direction)
10. **Fused MMA beats cuBLAS GEMM** for attention — don't try to replace flash attention with cuBLAS (#72)

---

## Experiment Status Summary (2026-04-05)

| # | Experiment | Status | Result |
|---|-----------|--------|--------|
| S1 | MMQ rebuild | ❌ Tested | No improvement |
| S2 | CUBLAS_COMPUTE_16F | ❌ Tested | No improvement |
| S3 | Codebook→SMEM | ❌ Tested | No improvement (attn <1%) |
| S4 | Speculative decoding | ❌ Tested | 35% slower (OOM for dual models) |
| S5 | Skip Softmax | ❌ Tested | No improvement (attn <1%) |
| S6-S11 | Attention dequant opts | ⏭️ Skipped | Attention <1%, invisible on test model |
| S12 | Greedy TCQ encode | ❌ Tested | PPL 2.5x worse, no speed gain |
| S13 | Warp FWHT | ⏭️ Low priority | FWHT is <10% of encode; Viterbi is bottleneck |
| S14-S18 | Attention kernel opts | ⏭️ Skipped | Attention <1%, invisible on test model |
| S19 | Marlin-style GEMM + fused quantize | ❌ Tested | At 88-94% DRAM peak; fused quantize: -31.5% (in-reg), -71% (SMEM) |
| S20 | Kernel fusion (f32 MMVQ) | ❌ Tested | -17.4% (DP4A→FMA tips kernel to compute-bound) |
| S30 | Viterbi double-buffer | ✅ Merged | +0.6% MoE, +0.5% dense, bit-exact |
| S21 | CUDA Graphs | ✅ Already done | +3.1% decode |
| S22-S29 | Architecture-level | ❓ Not attempted | Weeks-months of work each |

---

## Priority Ranking (Updated 2026-04-04)

**Hardware wall reached.** MMVQ kernel at 88-94% DRAM bandwidth. All feasible kernel-level optimizations tested. No single-user decode speedup remaining on RTX 3090 + Qwen3.5-27B Q6_K.

**Only paths to further speed gains:**
1. **Different hardware** — Blackwell FP4 tensor cores (S26) would eliminate dequant bottleneck entirely. Requires RTX 5090/B200.
2. **Different model** — MoE models have larger attention fraction; skipped attention opts (S6-S18) may show gains. MoE-as-grouped-GEMM (S27) is a real win for serving.
3. **Different workload** — Batch>1 serving: FlashInfer GQA trick (S22) gives 3x with tensor cores. Persistent batch tensors (S23) saves 2-5%.
4. **Use smaller weight quant** — Q5_K_M gives +13.5% decode with minor quality cost. Zero code changes.

**All tested — dead ends on this model/GPU:**
- S1 (MMQ), S2 (FP16 compute), S3 (SMEM codebook), S4 (spec decode), S5 (skip softmax), S12 (greedy encode), S19 (Marlin GEMM + fused quantize ×3), S20 (f32 MMVQ)

**Merged wins:**
- S21 (CUDA Graphs): +3.1%
- S30 (Viterbi double-buffer): +0.5%

**Potentially useful on different models/context:**
- S5 (Skip Softmax): valid for many KV heads or 100K+ context
- S14 (Scalar ADC): valid for long context attention
- S15 (INT8 TC): valid for prefill-heavy workloads
- S22 (GQA trick): valid for batch>1 serving
