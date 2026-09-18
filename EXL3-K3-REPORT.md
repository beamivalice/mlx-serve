# EXL3-K3 report

Status: reset onto 5b1eaae6. Diagnosis only (no kernel edits). The 64k prefill dip is a K4-specific **48k resident-bandwidth cliff**. K3 MTP forced-d2 and AUTO 4k are on record.

Measurement sha 467b630 (mem-line diagnostic on 5b1eaae6).

## Files changed

- `src/expert_exl3.zig`: `kFromPackedDim`; K2/K3 fixture embeds and shared decode helper. Host unpack was already K-generic.
- `src/fixtures/exl3_k2_linear.safetensors`, `src/fixtures/exl3_k3_linear.safetensors`: 128x128 MUL1 fixtures from ponyexl3 CPU direct quantize.
- `src/expert_exl3_kernels.zig`: tile decode and shape checks on K in {2,3,4}. Template `KBITS`, config caches keyed by K (GEMM key has no nwin). Cooperative indexed GEMV uses PACKED_W funnel for K!=4. NAX `nax_wfrag` stays K4; `nax_wfrag_k<K>` is the funnel twin. Prefill dispatches NAX for every K when G17+ and out_dim % 128 == 0.
- `src/expert_quant.zig`: `parseExpertQuant` admits k in {2,3,4} mul1; `kFromPackedDim`. No config-parse shard scan.
- `src/model.zig`: `expert_quant_k` on config from the expert_quant block. Enum stays `.exl3_k4`.
- `tests/convert_qwen38_flash_next_exl3.py`: `--from-exl3` accepts K2/K3/K4 per stacked bank, prints a per-K histogram, writes modal `expert_quant.k`.
- `src/generate.zig`: per-chunk `[prefill-trace] mem ... active_bytes= cache_bytes=` after eval, before `mlx_clear_cache` (diagnosis; not a kernel).
- `EXL3-K3-REPORT.md`.

## Tests added

- `exl3 K3 packed fixture decodes to the library inner and public f16` — `src/expert_exl3.zig`
- `exl3 K2 packed fixture decodes to the library inner and public f16` — `src/expert_exl3.zig`
- `exl3 packed dim 16 times K is K in 2,3,4` — `src/expert_exl3.zig`
- `exl3 K3 Metal inner GEMV matches the host tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K2 Metal inner GEMV matches the host tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K3 Metal inner GEMV matches host on production shape` — `src/expert_exl3_kernels.zig`
- `exl3 K3 cooperative indexed GEMV matches host MUL1 tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K2 cooperative indexed GEMV matches host MUL1 tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K3 cooperative indexed GEMV matches host MUL1 on production shape` — `src/expert_exl3_kernels.zig`
- `exl3 K2 cooperative indexed GEMV matches host MUL1 on production shape` — `src/expert_exl3_kernels.zig`
- `exl3 packedK accepts last dim 16 times 2,3,4` — `src/expert_exl3_kernels.zig`
- `exl3 K3 sorted GEMM matches host MUL1 on small shape` — `src/expert_exl3_kernels.zig`
- `exl3 expert_quant admits K2 K3 K4 mul1 and refuses other k or codebook` — `src/expert_quant.zig`
- `exl3 kFromPackedDim maps last dim 16*K` — `src/expert_quant.zig`
- `test_restack_mixed_k_per_tensor_keeps_each_last_dim` — `tests/convert_qwen38_flash_next_exl3.py`
- Production-shape cooperative GEMV uses `expectGemvEnvelope` (K4 f32-reference bar) for K3 and K2. The E=16 run test prints K3 vs K4 and asserts K3 * 100 < K4 * 115.
- `exl3 NAX K3 GEMM within 1.15x of K4 at C=2048 and 8192` — `src/expert_exl3_kernels.zig`
- `prefill-trace mem line names active_bytes and cache_bytes` — `src/generate.zig`

## Red-first evidence

Host, missing fixture:

```
.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="exl3 K3 packed"
src/expert_exl3.zig:442:37: error: unable to open 'fixtures/exl3_k3_linear.safetensors': FileNotFound
```

Loader, K3 refused:

```
.zig-toolchain/zig test src/expert_quant.zig -OReleaseFast --test-filter "admits K2"
FAIL (ExpertLayoutUnsupported)
```

Converter, mixed K:

```
RuntimeError: not K4 packed dim [8, 8, 48]
```

Production-shape K3 cooperative GEMV (before the PACKED_W funnel): `expectGemvEnvelope` failed at H 2560, I 640, top-k 10. Small shape 128 passed. Root cause was the cooperative body's tile stride hard-coded at 32 words (K4).

NAX K3 1.15x test, before `nax_wfrag_k`: `C=2048 K4 6935 us K3 33165 us` FAIL (K3 still on SIMD GEMM). After the port, same test green.

## Suite counts

On 2f876392 + K-generic (fb37abe6), before NAX port:

```
exl3 filter 61/61
mtp filter 124 passed; 1 skipped; 0 failed
```

On e5e47e83 (NAX K-generic):

```
.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="exl3" && ./zig-out/tests/test
62/62 passed.
```

NAX one-proj (e5e47e83): C=2048 K4 622 us K3 632 us (1.016x); C=8192 K4 1715 us K3 1899 us (1.107x); bar 1.15. Envelope `K4 185 us K3 188 us`. K4 small-shape GEMM still matches host.

KLD-era envelope on 6b3c5671: `K4 157 us K3 159 us`. Previous session 351/352 µs.

## Commit sha

Integration 5b1eaae6 (e5e47e83 + 801cbfac merged). Diagnostic + live 467b630.

## Live table

Pack `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-exl3k3-8bit` (never written this session). Histogram from restack `{2: 0, 3: 75264, 4: 0}` modal=3 (48 layers + MTP × 512 × 3). `expert_quant`: `{format: exl3, k: 3, codebook: mul1, out_scales: svh, source: restack}`.

Sizes measured this session:

| pack | total GiB | ngram GiB | excl. n-gram GiB | expert shards GiB |
|---|---|---|---|---|
| K3 | 78.87 | 29.80 | 49.07 | 43.52 |
| K4 | 93.23 | 29.80 | 63.42 | 57.87 |

KLD vs `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-16x512-raw`, `--kv-quant off --no-mtp`, JSON `/tmp/exl3-k3-kld.json` and `/tmp/exl3-k3-kld-ngram.json`. Mean to-first-EOS (7186 positions) is the table number.

| Quant | Weights GiB excl. n-gram | Mean KLD to-EOS | Top-1 to-EOS | NLL to-EOS | Mean KLD all | Top-1 all |
|---|---|---|---|---|---|---|
| EXL3 K3 (6b3c5671) | 49.07 | 0.09817 | 0.9066 | 0.4261 | 0.09000 | 0.9149 |
| EXL3 K3 + bf16 n-gram | 49.07 | 0.09149 | 0.9062 | 0.4208 | 0.08418 | 0.9152 |
| EXL3 K4 original | ~60 | 0.0676 | 0.924 | — | — | — |
| EXL3 K4 original + bf16 n-gram | ~60 | 0.0612 | 0.924 | — | — | — |
| EXL3 K4 current integration path | — | 0.0679 | — | — | — | — |
| affine 4/8 | — | 0.0814 | — | — | — | — |
| affine 4/8 + bf16 n-gram | — | 0.0749 | — | — | — | — |

Greedy determinism: two boots, prompt "Explain how a B-tree index speeds up a database range scan, step by step.", max_tokens 128, temperature 0, kv off, no MTP. Byte-identical 531 bytes, sha256 `0fc74a2c41e4e56cb7b7a9bf797085769249183edc9b55a6ab9135f0e87a25e8`, 128 tokens, finish_reason length both boots. Load at boot 3.86 / 3.73.

New-standard speed on e5e47e83 (NAX K-generic, 2f876392 parent), kv8, `--no-mtp --no-vision --prefix-cache-entries 0 --prefill-chunk 8192 --ctx-size 131072`, 3 boots interleaved, one fleeter lease per boot, nonce prefix, every row 512 completion tokens finish_reason=length. Affine-ab not re-run; affine medians from `results-20260918-2318.jsonl` (same prompt files, same protocol).

| ctx | K3 prefill / decode-512 med (e5e47e83) | K4 prefill / decode-512 med (e5e47e83) | affine 4/8 (20260918) | K3/K4 | K3/affine |
|---|---|---|---|---|---|
| 4k | 1472.8 / 50.1 | 1520.1 / 51.8 | 2123.1 / 58.1 | 0.969 / 0.967 | 0.694 / 0.862 |
| 16k | 1510.5 / 50.3 | 1601.1 / 52.8 | 2011.6 / 59.5 | 0.943 / 0.952 | 0.751 / 0.845 |
| 64k | 1523.1 / 48.5 | 1336.2 / 52.3 | 2046.2 / 58.0 | 1.140 / 0.928 | 0.744 / 0.837 |

K3 per-boot prefill 4k: 1418.4 / 1488.8 / 1472.8. Decode-512 4k: 50.1 / 50.8 / 49.5. load1 2.24 / 2.10 / 3.16.
K4 per-boot prefill 4k: 1466.7 / 1520.1 / 1551.9. Decode-512 4k: 51.6 / 51.8 / 52.3. load1 3.35 / 3.45 / 3.15.

On 6b3c5671 (SIMD GEMM, no NAX for K3) the same protocol was K3 4k 233.4 / 49.2 vs K4 762.6 / 51.4 (K3/K4 prefill 0.306). NAX port is a 6.3x K3 prefill lift at 4k.

### 64k pack-differential (467b630)

One 64k boot per pack, one lease each, `--prefill-trace`, chunk 8192, kv8, no MTP. Both: `[exl3-gemm] win=32 aligned=1 mixed=0` NAX, n=81920. nwin K4 2794 / K3 2788. Overall: K4 1470.4 tok/s / K3 1506.5 tok/s (this boot; e5e47e83 3-boot medians were K4 1336 / K3 1523).

Per 8192-token chunk, ms and mem after eval before `mlx_clear_cache`. `eval_ms=0` (KV eval <1 ms). Active grows **+2013265920 B (+1.875 GiB) per 8k chunk on BOTH packs** (KV, not trellis). cache_bytes trajectory is **identical** (~6.02 → 4.76 → 5.28 GB).

| pos | K4 ms | K3 ms | K4/K3 | K4 active GB | K3 active GB | K4 cache GB | K3 cache GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–8k | 5000 | 5343 | 0.936 | 64.60 | 50.53 | 5.61 | 5.61 |
| 8–16k | 4724 | 5242 | 0.901 | 66.47 | 52.40 | 4.44 | 4.44 |
| 16–24k | 4846 | 5370 | 0.902 | 68.35 | 54.27 | 4.52 | 4.52 |
| 24–32k | 5053 | 5354 | 0.944 | 70.22 | 56.15 | 4.55 | 4.55 |
| 32–40k | 5138 | 5409 | 0.950 | 72.10 | 58.02 | 4.64 | 4.64 |
| 40–48k | 5207 | 5418 | 0.961 | 73.97 | 59.90 | 4.80 | 4.80 |
| 48–56k | 6050 | 5452 | **1.110** | 75.85 | 61.77 | 4.89 | 4.89 |
| 56–64k | 6454 | 5507 | **1.172** | 77.72 | 63.65 | 4.92 | 4.92 |
| 64–70k | 5001 | 3242 | 1.542 | 78.60 | 64.53 | 3.18 | 3.17 |

K3 is flat (5343→5507 = 1.03×). K4 is faster until 48k, then a cliff 5207→6050→6454. Not nwin, not cache_bytes, not KV growth (same +1.875 GiB/chunk). Tile bytes **128 vs 96** (64 vs 48 u16). Weight delta **14.07 GB** (first-chunk active 64.60 vs 50.53).

**Term: 48k resident-bandwidth cliff (K4-only).** At pos 49152 K4 active is 75.85 GB vs K3 61.77. The K4 aligned 32-word NAX tile (128 B) plus 14 GB extra weights share the unified-memory bus with growing KV. Early chunks are compute-bound (aligned `nax_wfrag` wins). After ~48k the extra 33% trellis bytes become a DRAM tax on the same bus as attention KV, and K4 8k-chunk time jumps 21% while K3 stays flat.

**Fix (prefill executor):** (1) split the 8k-chunk timer into attention vs expert GEMM at pos 8k vs 56k on both packs — if GEMM inflates, cap NAX occupancy or use the K3 funnel occupancy when `kv_len>48k`; if attention inflates, the 14 GB K4 weights + 128 B tiles are starving KV, so stage expert tiles through a K-independent scratch or pipeline KV away from GEMM. (2) Do not chase cache_bytes (identical). (3) Do not chase nwin (2794 vs 2788).

### K3 MTP 4k (467b630)

One lease each, persist 0, kv8, handbook 4k, 512 tokens, finish=length. K4 numbers from decode report on 2f876392 (EXL3-DECODE-REPORT).

| row | K3 tok/s | K3 avg/round | K3 depth | K3 round_ms | K4 tok/s (decode rpt) | K3/K4 |
|---|---:|---:|---:|---:|---:|---:|
| forced-d2 (`MLX_SERVE_MTP_FORCE_DEPTH=2`) | 64.782 | 1.21 | 6 (unclamped) | 35.03 | 65.995 | **0.982** |
| AUTO (cap 2, no FORCE_DEPTH) | 50.233 | 1.09 | 2 | 44.75 | 61.754 | **0.813** |

Forced-d2 is K4-parity. AUTO lost the MTP gain (50.2 ≈ serial 50.1 on e5e47e83); log shows `[mtp] adaptive depth cap 2` and a two-chunk regime gate. One boot.

## Open questions

- K3 KLD 0.0982 / top-1 0.907 (6b3c5671) vs K4 0.0676 / affine 0.0814.
- Prefill executor: confirm whether the 48k cliff is attention or GEMM (layer ubench at 8k vs 56k).
- K3 AUTO MTP 0.813 vs K4 is one boot; forced-d2 is 0.982. Cap-2 climb vs two-chunk gate.
- Shared-expert/attention K5 out of scope.

## Comments

none added

## Ports

| what | source |
|---|---|
| K-bit funnel pair window (`bit0 = first*K + K + 256*K - 16`, wrap `packed_words = 8*K`) | `polarrust-metal-shaders/metal/common/exl3.metal` `polar_exl3_codeword_pair_window` / `polar_exl3_funnel_pair` |
| Same unpack on host | `ponyexl3/ref/trellis.py` `unpack_trellis_tile` |
| MUL1 codebook (unchanged) | `ponyexl3/ref/codebook.py`; Metal `polar_exl3_mul1_decode` |
| Direct quantize for fixtures | `ponyexl3/convert/direct.py` `quantize_inner_matrix_direct` + `regularize_public_weight` |
| Tensor-core 16×16 perm (unchanged) | `ponyexl3/ref/perm.py` `tensor_core_perm` |
| NAX generic-K lane decode (`tau_0`, funnel per oracle thread) | `polarrust-metal-shaders/metal/common/exl3_nax.metal` `polar_exl3_nax_decode_frag_k` |
