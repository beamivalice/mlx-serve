# EXL3 K4 routed experts for Qwen3.8-Flash-Next (qwen4_exp)

Status: converter, loader, host decoder, and Metal inner/indexed GEMV are in tree. Live conversion and timed KLD/speed are blocked on a box grant.

## Format (phase 0)

Trellis packing at K=4: a tile is 16×16 = 256 weights. Each weight contributes **4 bits**, so the tile is 1024 bits = **64 uint16** = **32 uint32**. Pack walks 16 spans of 16 values, stuffing the low-K bits of each codeword from the top of a 32-bit buffer; whenever fewer than 16 bits remain it emits the high halfword. After the spans, the uint32 view is SWAP16 (`(w<<16)|(w>>16)`). Unpack is the inverse bitshift trellis: for thread t in 0..127 a (K+16)-bit funnel from two u32 words (wrapped mod 32) yields 16-bit sliding windows `cw[2t]`, `cw[2t+1]`. Last dim is always `256*K/16 = 64`.

Codebook: **MCG**, not MUL1. `mixed = (codeword & 0xFFFF) * 0xCBAC1FED`; `pair = 0x3B603B60 ^ (mixed & 0x8FFF8FFF)`; the two packed f16 lanes are added in f32 and rounded back to f16. Why MCG: it is the codebook every 4.00bpw expert conversion of this family ships (`mcg_multiplier = 3417055213`); MUL1 (`* 0x83DCD12D`, byte-sum, then `0x1EEE`/`0xC931`) is a different reconstruction and is not required. The host implements MUL1 only as a decoder twin.

suh / svh: f16 vectors, length `in_features` and `out_features`. There is **no separate per-tensor scalar**; the encode-time codebook scale `1.24371088` is folded into suh by `regularize_public_weight`. Decode:

1. `x' = round16( H128( suh ⊙ round16(x) ) )`  — left / input axis
2. inner GEMV against 16×16 tiles in **row-major** (tensor-core perm inverted)
3. `y = round16( svh ⊙ H128(inner) )`  — right / output axis

H128 is the **Sylvester** matrix `H[i,j] = (-1)^{popcount(i∧j)} / sqrt(128)` with `1/sqrt(128) = 0.08838834764831845`, applied with left-to-right f32 accumulation (not BLAS association). Host public reconstruct left-multiplies inner by that H then suh, then right-multiplies by H then svh. The first public-reconstruct pass used an in-place FWHT; it missed the fixture by 1 ULP, so the host switched to the popcount Sylvester matvec. Activation-side GEMV applies the same H to activations.

Per expert projection (one linear):

| tensor | dtype | shape |
|---|---|---|
| `trellis` | U16 | `[in/16, out/16, 64]` |
| `suh` | F16 | `[in]` |
| `svh` | F16 | `[out]` |

Stacked per-layer dialect, E on axis 0 so gather is `bank[eid]`. Production geometry **E=512, hidden=2560, inter=640**:

| projection | in | out | trellis | suh | svh |
|---|---|---|---|---|---|
| gate | 2560 | 640 | `[512, 160, 40, 64]` U16 | `[512, 2560]` F16 | `[512, 640]` F16 |
| up | 2560 | 640 | `[512, 160, 40, 64]` U16 | `[512, 2560]` F16 | `[512, 640]` F16 |
| down | 640 | 2560 | `[512, 40, 160, 64]` U16 | `[512, 640]` F16 | `[512, 2560]` F16 |

Names: `language_model.model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}`. MTP: `language_model.mtp.layers.0.mlp.switch_mlp.*`. Same `switch_mlp` / `language_model.model.` prefix as the affine pack. Bytes: each trellis bank is 512×160×40×64×2 = 400 MiB, so three projections ≈ 1.2 GiB trellis + ~6.5 MiB suh/svh per layer; 48 layers ≈ 57.6 GiB experts.

`config.json`: `"expert_quant": {"format":"exl3","k":4,"codebook":"mcg","mcg_multiplier":3417055213,"calibration":"imatrix-diagonal"}`.

Host decoder: `src/expert_exl3.zig`. Fixture: `src/fixtures/exl3_k4_linear.safetensors`.

## Converter

`tests/convert_qwen38_flash_next_exl3.py`

- Input: bf16 HF dir + the mixed 4/8 pack.
- Plan: files with only expert affine banks are dropped; files with no expert keys are hard-linked; mixed shards are rewritten with remaining tensors copied as raw bytes.
- Experts: HF `mlp.experts.gate_up_proj` `[E,2I,H]` split and transposed to public `[H,I]`; `down_proj` `[E,H,I]` transposed to `[I,H]`; stacked on axis 0.

ponyexl3 entry points:

- Direct (picked): `regularize_public_weight` then `quantize_inner_matrix_direct` on a **concatenated-out batch** of N same-shape experts (`_quantize_direct_batch`). Metal trellis search is already GPU-resident and splits on scratch; stacking out-features is the grouped-search analogue of `ldlq_quantize_group`.
- LDLQ (slower, no quality win here): `prepare_hessian_for_ldl(diag(v))` + `block_ldl` + `ldlq_inner_matrix`.

Measured on one 2560×640 expert, imatrix-like `v`, 256 rows `X[:,i]~N(0,√v_i)`:

| arm | weighted relRMS | output relRMS | time |
|---|---:|---:|---:|
| affine 4-bit g64 (`mx.quantize`) | 0.09110 | 0.09126 | — |
| **direct K4 MCG** | **0.06868** | **0.06881** | **0.353 s** |
| direct, rows × √v | 0.06869 | 0.06883 | — |
| LDLQ diag(v) | 0.06868 | 0.06881 | 0.594 s |

All EXL3 arms beat affine-4. **Pick direct** (fastest). Row-√v imatrix weighting did not move the needle.

Batched direct projections/s (2560×640, Metal):

| N | wall | proj/s | full pack (73,728) |
|---|---:|---:|---:|
| 16 | 5.173 s | **3.09** | **6.63 h** |
| 32 | 10.422 s | 3.07 | 6.67 h |
| 64 | 21.290 s | 3.01 | 6.80 h |

N=16 is the plateau. 6.63 h > 4 h, so conversion is **resumable**: `model-exl3-L{layer:02d}-{gate,up,down}.safetensors`, skip if header shape validates. A 2 h window does ~3.09×7200 ≈ 22k projections ≈ **14 layers**. ~4 windows of 2 h finish the pack.

Imatrix collect (step 0, still grant-gated) checkpoints `*.safetensors.layers/LXX.safetensors` and skips complete layers. Tag stays `imatrix-diagonal` when `--imatrix` is passed (used for LDLQ / zero-routed listing); the picked search arm is direct.

Unit tests: plan + layout + imatrix-diagonal + resume skip. `--self-test` → 12/12.

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

MoE decode: `moeSwigluIndexed` = prepare+indexed GEMV+finish on gate and up, SiLU(gate)*up, same on down, score-weighted sum. Hooked in `moeMLP2WithRouter` when `expert_layout == .exl3_k4`. **rows ≤ 16** (decode + verify): one-shot or per-row loop. **rows > 16**: `moePrefill` (rows kernel).

The naive inner GEMV unpacks 256 codewords per output lane per tile; it is the parity kernel, not the 66 tok/s kernel. A polar-style simdgroup K4 body (32-word tile, 8 accumulators/lane) is the speed port still owed.

## Prefill arm chosen

Per-row indexed GEMV is used only for **rows ≤ 16** (decode + verify widths). Above that, `moePrefill` (expand rows × topk, one indexed GEMV family — the rows kernel).

Measured on a synthetic layer, 512 rows, E=4, H=I=128, topk=2, GPU:

| arm | time |
|---|---|
| (b) rows kernel (`moePrefill`) | **197 ms** |
| (a) decode-to-f16 once + `gather_mm` (public banks already on GPU) | **13 ms** |

**Pick (a)** by the number: 15× faster when public W is materialized. Production still runs (b) until a Metal full-matrix decode writes those banks per chunk (host reconstruct of 2560×640 × hundreds of experts is not a prefill). Next grant work: Metal decode-full of unique experts, then `gather_mm`.

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

HEAD after this round (batched direct + resumable shards). Clone base `7264fe15`.

## Open questions

- Naive GEMV will not hit the 15% speed bar; need the simdgroup K4 indexed-pair port and a rows4 prefill kernel.
- Convert is 6.63 h at 3.09 proj/s; needs ~4× 2 h windows. Imatrix collect still grant-gated.
- Production prefill should switch to decode-to-bf16 + gather_mm once Metal weight decode exists (synthetic pick is 13 ms vs 197 ms).
- suh/svh Hadamard Metal path is a naive matvec; F16 rounding vs host FWHT/Sylvester association needs a dedicated parity test at 2560-d.
- MTP EXL3 experts load via trellis triples; the MTP forward reuses the trunk MoE hook.

## Comments

none added
