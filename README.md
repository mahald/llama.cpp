# Turbo4 on Gemma 4 26B A4B: 120 t/s with 3.8x KV compression (RTX 3090)

Got turbo4 working on Gemma 4 26B A4B with **zero speed penalty** vs f16 KV. The key challenge was Gemma 4's variable head dimensions (256-dim on SWA layers, 512-dim on global layers) — the existing turbo4 FA kernels only supported D=128.

**Hardware:** RTX 3090 (24GB), Ryzen 9 5950X, 96GB RAM, Windows, MSVC + CUDA 13.0

## Results

| Config | Speed (32K ctx) | KV VRAM (32K) | KV VRAM (256K) | Fits 24GB at 256K? |
|---|---|---|---|---|
| f16 KV + FA | 120 t/s | ~1.2 GB | ~9.4 GB | No (25.4GB total) |
| q8_0 KV + FA | 122 t/s | ~0.6 GB | ~4.7 GB | Yes |
| turbo4 KV + FA | **120 t/s** | ~0.4 GB | ~2.5 GB | Yes (18.5GB total) |

Turbo4 is the only config that runs Gemma 4 at **full 256K context** on a 24GB card at full speed.

## The optimization journey

Starting from the existing turbo4 FA kernels (D=128 only), extending to D=256/512 gave 63 t/s — about half of f16. Six optimizations brought it to parity:

| Optimization | Speed | vs f16 |
|---|---|---|
| Baseline turbo4 K+V (full WHT per position) | 63 t/s | 53% |
| + Lazy V: defer inverse WHT to single post-loop pass | 72 t/s | 60% |
| + Lazy K: forward WHT on Q once, dot raw centroids | 80 t/s | 67% |
| + Batch centroid decode: uint32/uint64 loads | 96 t/s | 80% |
| + Optimized write path: unrolled WHT + batch 3-bit packing | 104 t/s | 87% |
| + Warp-cooperative write kernel: 16-thread shuffle WHT | **120 t/s** | **100%** |

### Key ideas

**Lazy V (deferred WHT):** Instead of inverse WHT on every V vector during attention, accumulate attention-weighted centroids in WHT space. Apply one inverse WHT to the accumulated VKQ after the loop. Exploits linearity: `WHT_inv(Σ aᵢvᵢ) = Σ aᵢ·WHT_inv(vᵢ)`. Reduces V WHT cost from O(context_length) to O(1).

**Lazy K (pre-transformed Q):** Instead of inverse WHT on every K vector, apply forward WHT to Q once before the loop. Math: `Q · WHT_inv(K_raw) = WHT_fwd(Q) · K_raw`. Per-position K cost becomes just centroid unpack + dot + norm multiply. Also O(1).

**Batch centroid decode:** For K (8 elements, byte-aligned): load 3 bytes into uint32_t → 8 centroids via shifts. For V: uint16_t (ne=4), uint32_t (ne=8), uint64_t (ne=16). Eliminates per-element bit_offset/byte_idx computation.

**Warp-cooperative write kernel:** The original KV write path ran one CUDA thread per 128-element turbo4 block with a serial WHT butterfly. Replaced with 16 threads per block using warp-shuffle WHT — same approach as the FA read path. ~10-16x faster per block.

## Files modified (all in `ggml/src/ggml-cuda/`)

- `fattn-turbo4.cuh` — Lazy K (Q pretransform) + lazy V (deferred WHT) + batch centroid decode
- `fattn-vec.cuh` — Q pretransform call before KV loop, VKQ post-process after loop
- `fattn-vec-turbo4.cu` — D=256/512 kernel instantiations
- `fattn.cu` — Dispatch table + kernel selection for D=256/512 turbo4
- `cpy-utils.cuh` — Fused butterfly + unrolled WHT + batch 3-bit packing
- `set-rows.cu` — Warp-cooperative turbo4 quantize kernel (16 threads/block)

## Correctness

Model produces identical-quality output between f16 and turbo4 — verified on math (17×23=391, three methods shown), reasoning (general relativity explanations), and general chat. No quality degradation observed.

## Transparency note

The CUDA kernel development was done mostly using Claude 4.6 Opus. I directed the project, chose the model and optimization targets, ran all builds and benchmarks, and validated correctness, but the actual kernel code was written by the AI. I found it genuinely interesting that every time the current iteration of turbo4 wasn't as fast as the f16 version, all I had to do was ask Claude if there was a clever way to make it faster, and it always found a way. So all credit for this work (and for most of this post except for this Transparency note) goes to Claude.

## Fork

[LINK TO YOUR GITHUB REPO HERE]

Built on top of the existing TurboQuant llama.cpp fork. To test:

```bash
cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release

build/bin/llama-server -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mmproj mmproj-gemma-4-26B-A4B-it-f16.gguf \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --flash-attn on --ctx-size 262144 \
  --n-gpu-layers 99 --host 0.0.0.0 --port 8080
```

The lazy K/V approach should generalize to any model with head_dim > 128 — it's not Gemma 4 specific.
