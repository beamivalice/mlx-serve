# EXL3 K4 routed experts for Qwen3.8-Flash-Next (qwen4_exp)

Status: converter, loader, host decoder, and Metal inner/indexed GEMV are in tree. Live conversion and timed KLD/speed are blocked on a box grant.

## Format (phase 0)

K=4 trellis: each 16×16 tile stores 256 weights at 4 bits each = 1024 bits = 64 uint16 = 32 uint32. Packing writes K fresh bits per weight into 16 spans of 16 values, then SWAP16 per uint32. Unpack recovers 16-bit sliding windows (`bitshift` trellis): for thread t in 0..127 a (K+16)-bit funnel from two u32 words yields codewords `cw[2t]`, `cw[2t+1]`. Packed last dim is `256*K/16 = 64`.

Codebook: MCG. `mixed = codeword * 0xCBAC1FED`; `pair = 0x3B603B60 ^ (mixed & 0x8FFF8FFF)`; the two packed f16 lanes add in f32 and round back to f16. MUL1 is implemented on the host (`0x83DCD12D` + byte-sum + `0x1EEE`/`0xC931` fma) but this pack does not use it. MCG is the codebook the Metal kernels and the 4.00bpw expert conversions actually ship.

suh/svh: f16 vectors of length in_features / out_features. Encode folds the codebook scale (~1.2437) and per-row/col RMS into these vectors, plus random signs. Decode applies:

1. `x' = H128(suh ⊙ round16(x))` (round16 after the transform)
2. inner GEMV against the decoded 16×16 tiles in row-major (tensor-core perm inverted)
3. `y = round16(svh ⊙ H128(inner))`

H128 is the Sylvester Hadamard of order 128, scaled by `1/sqrt(128) = 0.08838834764831845`. Host public reconstruct left-multiplies inner by H then suh, then right-multiplies by H then svh. GEMV applies the same transforms to activations.

Per expert projection tensors:

- `trellis` U16 `[in_tiles, out_tiles, 64]`
- `suh` F16 `[in_features]`
- `svh` F16 `[out_features]`

Stacked per-layer dialect, one bank per projection:

```
language_model.model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.trellis  [E, in_tiles, out_tiles, 64]
language_model.model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.suh      [E, in]
language_model.model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.svh      [E, out]
```

Same prefix as the affine pack (`switch_mlp`, `language_model.model.`). Gather kernels index expert `e` on axis 0 without assembling per-expert modules. MTP uses the same names under `language_model.mtp.layers.0.mlp.switch_mlp.*`.

`config.json` block:

```
"expert_quant": {"format":"exl3","k":4,"codebook":"mcg","mcg_multiplier":3417055213, ...}
```

Hermetic host decoder: `src/expert_exl3.zig`. Fixture `src/fixtures/exl3_k4_linear.safetensors` (copy also under `tests/fixtures/`, gitignored there): 128×128 public matrix regularized and quantized K4 MCG, plus inner and public f16 truth.

## Converter

`tests/convert_qwen38_flash_next_exl3.py`

- Input: bf16 HF dir + the mixed 4/8 pack.
- Plan: files with only expert affine banks are dropped; files with no expert keys are hard-linked; mixed shards are rewritten with remaining tensors copied as raw bytes.
- Experts: HF `mlp.experts.gate_up_proj` `[E,2I,H]` split and transposed to public `[H,I]`; `down_proj` `[E,H,I]` transposed to `[I,H]`; stacked on axis 0.
- Default quantizer: LDLQ (Hessian from 256 Gaussian rows in the inner dim when no capture file is passed). `--quantizer direct` is the synthetic-test path (Metal tile search, no Hessian).
- Calibration: synthetic tests used `none-direct`. Production default without `--calibration` is `ldlq-gaussian-256`. A captured row file (`--calibration *.npy`, shape `[N, in]`) is recorded as `captured-rows`. Wikidata/wikitext capture of the real MLP inputs is the live-conversion job (box grant).
- Unit tests: plan classification + tiny 1-layer/2-expert/128-d pack layout, hardlink inodes, `expert_quant` block, trellis shape `[2,8,8,64]`. `python3 tests/convert_qwen38_flash_next_exl3.py --self-test` → 4/4.

## Loader

`Layout.exl3_k4` on the expert-layout enum. Detected from stacked `*.trellis`/`*.suh`/`*.svh` keys. `parseExpertQuant` admits only `format=exl3`, `k=4`, `codebook=mcg`; anything else is `error.ExpertLayoutUnsupported`. Named 503: `expert_layout_unsupported` via `loadRefusalFor` + `loadErrorFromName`. Streaming this layout is refused (resident pack). Resident load binds trellis/suh/svh onto the existing switch_* weight/scales/biases slots and skips the bf16 pre-transpose. `[expert-exl3] engaged` is logged once from `parseConfig` when the index is EXL3 K4.

## Kernels

Host: tile unpack, MCG decode, tensor-core perm, public reconstruct, activation-side project (prepare H128 + inner GEMV + finish H128).

Metal (`mlx_fast_metal_kernel`, shape-keyed config cache for the single-expert GEMV):

| kernel | what | parity |
|---|---|---|
| `mlxserve_exl3_k4_mcg_gemv` | inner GEMV, K4 MCG, 1 row | bit-exact vs host inner decode on 128×128 fixture |
| `mlxserve_exl3_k4_mcg_gemv_indexed` | same, `slots` gather, trellis `[E,…]` | compiled; used by MoE arm |
| `mlxserve_exl3_prepare_h128` | per-expert suh ⊙ x then Sylvester H128 | naive 128-thread matvec |
| `mlxserve_exl3_finish_h128` | H128 then ⊙ svh | naive 128-thread matvec |

Shapes tested: inner GEMV `in=128, out=128`, 1 row. Host project vs dense `x @ W_public` rel RMS < 0.02 on that fixture.

MoE decode: `moeSwigluIndexed` = prepare+indexed GEMV+finish on gate and up, SiLU(gate)*up, same on down, score-weighted sum. Hooked in `moeMLP2WithRouter` when `expert_layout == .exl3_k4`. Decode (B*S=1) is one shot; prefill loops tokens (correctness arm, not the speed arm).

The naive inner GEMV unpacks 256 codewords per output lane per tile; it is the parity kernel, not the 66 tok/s kernel. A polar-style simdgroup K4 body (32-word tile, 8 accumulators/lane) is the speed port still owed.

## Prefill arm chosen

Not measured on a 512-row synthetic layer yet (would load real expert banks or a large synthetic). Current arm: per-token `moeSwigluIndexed` (decode kernel). Alternative still open: decode selected experts to f16 then `gather_mm`. The rows4-indexed port is not in tree. Prefill numbers: n/a.

## Live results

Not run. Box not granted. Affine 4-bit reference row to beat: mean KLD 0.0814 to-EOS (0.0749 with bf16 n-gram table). Resident affine speed: warm loop ~66 tok/s, 4k prefill ~2110 tok/s, kv8, no MTP.

| Quant | Weights GB | Mean KLD | Top-1 | NLL |
|---|---|---|---|---|
| affine 4/8 (existing) | (pack) | 0.0814 / 0.0749 w/ bf16 ngram | — | — |
| EXL3 K4 experts | not converted | not run | not run | not run |

Speed vs affine pack: not run.

## Ports

Every algorithm taken from another tree (do not copy these names into mlx-serve source; this list is for NOTICE):

| what | source file |
|---|---|
| MCG codebook (`* 0xCBAC1FED`, LOP3 0x6A = `c ^ (a & b)`, f16 pair add) | `polarrust-metal-shaders/metal/common/exl3.metal` (`polar_exl3_mcg_decode`); host twin `polarrust-layers/src/kernels/primitives/exl3.rs` `decode_codeword` / `decode_finite_half_pair`; CPU oracle `ponyexl3/ref/codebook.py` `decode_3inst` |
| MUL1 codebook (host only) | same Metal file `polar_exl3_mul1_decode`; `ponyexl3/ref/codebook.py` |
| Trellis pack/unpack, K-bit funnel, SWAP16 | `ponyexl3/ref/trellis.py` `pack_trellis_tile` / `unpack_trellis_tile`; host `polarrust-layers/.../exl3.rs` `decode_tile_codewords` |
| Tensor-core 16×16 perm | `ponyexl3/ref/perm.py` `tensor_core_perm`; Metal `polar_exl3_tile_position` |
| suh/svh + H128 contract | `ponyexl3/ref/reconstruct.py` `reconstruct_public_weights`; `ponyexl3/ref/hadamard.py`; Metal prepare/finish in `exl3.metal` |
| Regularize + LDLQ + direct tile search | `ponyexl3/convert/regularize.py`, `hessian.py` `ldlq_inner_matrix`, `direct.py` `quantize_inner_matrix_direct` |
| Indexed gather-by-expert idea | Metal `polar_exl3_gemv_indexed_pair_f16_f32`, `polar_exl3_gemm_rows4_indexed_f16_f32` (algorithm only; our first kernel is a scalar unpack GEMV) |
| mlx_fast_metal_kernel cache | `src/expert_bf16_kernels.zig` |

## Files changed

- `src/expert_exl3.zig` (host codec + tests)
- `src/expert_exl3_kernels.zig` (Metal GEMV/prepare/finish/MoE)
- `src/expert_quant.zig` (layout + expert_quant parse)
- `src/expert_stream.zig` (exl3 byte bill)
- `src/model.zig` (parseConfig engage)
- `src/model_discovery.zig` (streaming index)
- `src/model_registry.zig` / `src/server.zig` / `src/scheduler.zig` (named refusal)
- `src/transformer.zig` (bind trellis, moeExl3)
- `src/mtp.zig` (trellis triple)
- `src/tests.zig` (imports)
- `src/fixtures/exl3_k4_linear.safetensors`
- `tests/convert_qwen38_flash_next_exl3.py`
- `EXL3-REPORT.md`

## Tests added

- `exl3 MCG codebook maps a zero codeword to the finite half pair`
- `exl3 K4 packed fixture decodes to the library inner and public f16` (also host project vs dense)
- `exl3 K4 Metal inner GEMV matches the host tile decode`
- `exl3 expert_quant admits uniform K4 mcg and refuses any other codebook or k`
- layout map resolves `exl3_k4`; `expert_layout_unsupported` 503
- converter `--self-test` 4 tests (plan + synthetic layout/hardlinks)

## Red-first evidence

1. Host MCG stub returned 0; test expected 16224 (`half(float(0x3B60)+float(0x3B60))`). Then decode implemented.
2. Converter stubs raised `NotImplementedError` (4 errors). Then plan/convert implemented; 4/4.
3. Metal inner GEMV test compiled and matched host tile decode bit-exact on the fixture.

## Suite counts

`zig build test-build -Doptimize=ReleaseFast` succeeded. `./zig-out/tests/test`: **2691 passed, 174 skipped, 0 failed**. Filtered `exl3 K4` 4/4. Converter `--self-test` 4/4.

## Commit sha

HEAD `84a25c08`. Stack: `b578cb05` host decoder, `5b9c48fb` converter, `b3e3e219` loader, `259f3b16` GEMV kernel, `84a25c08` indexed MoE hook. Clone base `7264fe15`.

## Open questions

- Naive GEMV will not hit the 15% speed bar; need the simdgroup K4 indexed-pair port and a rows4 prefill kernel.
- LDLQ default uses Gaussian Hessian unless `--calibration` is passed; live conversion should capture MLP-input rows from the bf16 teacher (wikitext2 / the pack's own prompt file).
- Prefill token loop is a correctness arm only; measure vs decode-to-f16+`gather_mm` on a synthetic layer when the box is granted.
- suh/svh Hadamard Metal path is a naive matvec; F16 rounding vs host FWHT/Sylvester association needs a dedicated parity test at 2560-d.
- MTP EXL3 experts load via trellis triples; the MTP forward reuses the trunk MoE hook.

## Comments

none added
