# EXL3-K3 report

Status: kernels, loader, converter, restack, live KLD, greedy determinism, and new-standard speed rows done on 6b3c5671. Rebase onto integration 728af897 is the next commit on this branch.

Start sha 238fb8d7, branch agent/exl3-k3b, clone head at measurement 6b3c5671. Routed-expert trellis in the K3 pack is uniform K3. Non-expert trellis stays in the 4/8 trunk (shared-expert/attn K5, MTP fc K4, indexer K3).

## Files changed

- `src/expert_exl3.zig`: `kFromPackedDim`; K2/K3 fixture embeds and shared decode helper. Host unpack was already K-generic.
- `src/fixtures/exl3_k2_linear.safetensors`, `src/fixtures/exl3_k3_linear.safetensors`: 128x128 MUL1 fixtures from ponyexl3 CPU direct quantize.
- `src/expert_exl3_kernels.zig`: tile decode and shape checks on K in {2,3,4}. Template `KBITS`, config caches keyed by K. Last dim `16*K`. Cooperative indexed GEMV uses PACKED_W funnel for K!=4 (`bit0 = first*K + K + 256*K - 16`, wrap `packed_words = 8*K`). NAX stays K4-only. Per-expert fallback deleted.
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

## Suite counts

This session, head 6b3c5671:

```
.zig-toolchain/zig build -Doptimize=ReleaseFast
.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="exl3" && ./zig-out/tests/test
36/36 passed.
```

Envelope print from that run: `exl3 indexed GEMV H=2560 I=640 topk=10: K4 157 us K3 159 us` (K3/K4 = 1.013, bar 1.15). Previous session on the same test printed K3 351 µs vs K4 352 µs (parity under the K4 envelope). Ratio holds; absolute µs tracks box load.

Converter `--self-test`: 15 tests OK.

## Commit sha

- 32f2bf55 host K2/K3 decode fixtures
- bcf0800f Metal tile decode parameterized on K
- 5e134f18 loader admits K in {2,3,4} MUL1
- 46983e81 `--from-exl3` restack K2/K3/K4
- fa2bab27 config parse does not open trellis shards
- 6b3c5671 K3 cooperative GEMV uses PACKED_W funnel (measurement binary)
- this report: filled after commit

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
| EXL3 K3 (this binary) | 49.07 | 0.09817 | 0.9066 | 0.4261 | 0.09000 | 0.9149 |
| EXL3 K3 + bf16 n-gram | 49.07 | 0.09149 | 0.9062 | 0.4208 | 0.08418 | 0.9152 |
| EXL3 K4 original | ~60 | 0.0676 | 0.924 | — | — | — |
| EXL3 K4 original + bf16 n-gram | ~60 | 0.0612 | 0.924 | — | — | — |
| EXL3 K4 current integration path | — | 0.0679 | — | — | — | — |
| affine 4/8 | — | 0.0814 | — | — | — | — |
| affine 4/8 + bf16 n-gram | — | 0.0749 | — | — | — | — |

Greedy determinism: two boots, prompt "Explain how a B-tree index speeds up a database range scan, step by step.", max_tokens 128, temperature 0, kv off, no MTP. Byte-identical 531 bytes, sha256 `0fc74a2c41e4e56cb7b7a9bf797085769249183edc9b55a6ab9135f0e87a25e8`, 128 tokens, finish_reason length both boots. Load at boot 3.86 / 3.73.

New-standard speed, same session, same binary 6b3c5671, kv8, `--no-mtp --no-vision --prefix-cache-entries 0 --prefill-chunk 8192 --ctx-size 131072`, 3 boots interleaved K3 then K4, nonce prefix per request, every row 512 completion tokens finish_reason=length. This branch does not carry aligned-32 windows or split-K 2 (those land on rebase).

| ctx (prompt tokens) | K3 prefill med / decode-512 med | K4 prefill med / decode-512 med | K3/K4 prefill / decode |
|---|---|---|---|
| 4k (4569 / 4568) | 233.4 / 49.2 | 762.6 / 51.4 | 0.306 / 0.957 |
| 16k (17631) | 242.3 / 49.8 | 765.7 / 51.9 | 0.316 / 0.960 |
| 64k (69938 / 69941) | 244.1 / 49.4 | 710.7 / 49.0 | 0.343 / 1.008 |

Per-boot load1: K3 1.74 / 2.39 / 1.83; K4 2.00 / 0.86 / 1.83.

Per-boot prefill 4k: K3 240.9 / 233.4 / 225.3; K4 762.6 / 793.3 / 665.7.
Per-boot decode-512 4k: K3 49.9 / 49.2 / 47.6; K4 51.4 / 51.6 / 48.5.

Decode matches the envelope test (K3 ≈ K4). Prefill on this branch is 0.31x of same-binary K4: the sorted GEMM still walks K3 tiles through the funnel, while K4 keeps the aligned 32-word path. Coordinator K4 4k prefill 1576 tok/s is the integration head with aligned-32 windows, not this binary.

## Open questions

- K3 KLD 0.0982 / top-1 0.907 is worse than K4 0.0676 / 0.924 and worse than affine 4/8 0.0814. Expected of 3-bit vs 4-bit routed experts; not a kernel-parity failure (envelope holds). No K3 MTP claim until this is accepted.
- K3 prefill 0.31x of K4 on 6b3c5671. After rebase the K-generic funnel must land on the run-aligned 32-row window GEMM; remeasure at the new standard before quoting a K3 prefill ratio against integration K4.
- Shared-expert and attention in the 305bpw checkpoint are K5. Out of scope (K in {2,3,4} for routed experts).

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
