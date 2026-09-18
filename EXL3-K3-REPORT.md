# EXL3-K3 report

Status: rebased onto 2f876392. K3 prefill uses the NAX rows GEMM (`GEMM_NAX_SOURCE`) with K-generic tile decode; K4 keeps the aligned nax_wfrag body. Live six-number rows on e5e47e83.

Start sha 238fb8d7, branch agent/exl3-k3b. KLD/greedy on 6b3c5671. Speed on e5e47e83. HEAD e5e47e83. Routed-expert trellis in the K3 pack is uniform K3.

## Files changed

- `src/expert_exl3.zig`: `kFromPackedDim`; K2/K3 fixture embeds and shared decode helper. Host unpack was already K-generic.
- `src/fixtures/exl3_k2_linear.safetensors`, `src/fixtures/exl3_k3_linear.safetensors`: 128x128 MUL1 fixtures from ponyexl3 CPU direct quantize.
- `src/expert_exl3_kernels.zig`: tile decode and shape checks on K in {2,3,4}. Template `KBITS`, config caches keyed by K (GEMM key has no nwin). Cooperative indexed GEMV uses PACKED_W funnel for K!=4. NAX `nax_wfrag` stays K4; `nax_wfrag_k<K>` is the funnel twin. Prefill dispatches NAX for every K when G17+ and out_dim % 128 == 0.
- `src/expert_quant.zig`: `parseExpertQuant` admits k in {2,3,4} mul1; `kFromPackedDim`. No config-parse shard scan.
- `src/model.zig`: `expert_quant_k` on config from the expert_quant block. Enum stays `.exl3_k4`.
- `tests/convert_qwen38_flash_next_exl3.py`: `--from-exl3` accepts K2/K3/K4 per stacked bank, prints a per-K histogram, writes modal `expert_quant.k`.
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

KLD/greedy binary: 6b3c5671.

Rebased onto 2f876392:

- 61d45cf4 host K2/K3 decode fixtures
- 4936ea49 kFromPackedDim
- b5a6e2a1 loader admits K in {2,3,4} MUL1
- 0f10d085 `--from-exl3` restack
- 48c82bea config parse does not open shards
- cfe14ea1 live KLD/greedy/speed report (6b3c5671 numbers)
- fb37abe6 K-generic cooperative readers (GEMM key keeps k, drops nwin)
- b1831e07 record kernel-port sha
- e5e47e83 K-generic NAX tile decode (HEAD, live speed binary)

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

## Open questions

- K3 KLD 0.0982 / top-1 0.907 (6b3c5671) is worse than K4 0.0676 / 0.924 and worse than affine 4/8 0.0814. Expected of 3-bit routed experts. No K3 MTP claim until this is accepted.
- K3 64k prefill 1.140x of same-binary K4 (e5e47e83): K4 64k prefill dipped to 1336 vs K3 1523. Decode stays ~0.93x.
- Affine ratios use the 20260918-2318 affine-ab row, not a same-session affine boot.
- Shared-expert and attention in the 305bpw checkpoint are K5. Out of scope.

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
