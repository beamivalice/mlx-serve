# EXL3-K3 report

Status: kernels, loader, converter, and restack done. KLD gated on BOX-GRANT (BOX-REQUEST.md written; no grant). Start sha 238fb8d7, branch agent/exl3-k3.

Routed-expert trellis tensors in `/Users/beam/llm/models/Qwen3.8-Flash-Next-exl3-305bpw` are **uniform K3** (75264/75264). Non-expert trellis is mixed (shared-expert/attn K5, MTP fc K4, indexer K3) and stays in the 4/8 trunk.

## Files changed

- `src/expert_exl3.zig`: `kFromPackedDim`; K2/K3 fixture embeds and shared decode helper. Host unpack was already K-generic.
- `src/fixtures/exl3_k2_linear.safetensors`, `src/fixtures/exl3_k3_linear.safetensors`: 128x128 MUL1 fixtures from ponyexl3 CPU direct quantize. Inner bits match ponyexl3 `reconstruct_inner`. Public bits match host Sylvester H128.
- `src/expert_exl3_kernels.zig`: tile decode and shape checks parameterized on K in {2,3,4}. Template `KBITS`, config caches keyed by K. Last dim `16*K`. NAX stays K4-only. Dispatch structure unchanged.
- `src/expert_quant.zig`: `parseExpertQuant` admits k in {2,3,4} mul1; `kFromPackedDim`; `scanExl3TrellisK` reads per-tensor last dim from safetensors headers.
- `src/model.zig`: `expert_quant_k` on config; parseConfig records modal K from the trellis scan. Enum stays `.exl3_k4`.
- `tests/convert_qwen38_flash_next_exl3.py`: `--from-exl3` accepts K2/K3/K4 per stacked bank, prints a per-K histogram, writes modal `expert_quant.k`.
- `EXL3-K3-REPORT.md`, `BOX-REQUEST.md`.

## Tests added

- `exl3 K3 packed fixture decodes to the library inner and public f16` — `src/expert_exl3.zig`
- `exl3 K2 packed fixture decodes to the library inner and public f16` — `src/expert_exl3.zig`
- `exl3 packed dim 16 times K is K in 2,3,4` — `src/expert_exl3.zig`
- `exl3 K3 Metal inner GEMV matches the host tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K2 Metal inner GEMV matches the host tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K3 Metal inner GEMV matches host on production shape` — `src/expert_exl3_kernels.zig`
- `exl3 K3 cooperative indexed GEMV matches host MUL1 tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K2 cooperative indexed GEMV matches host MUL1 tile decode` — `src/expert_exl3_kernels.zig`
- `exl3 K2 cooperative indexed GEMV matches host MUL1 on production shape` — `src/expert_exl3_kernels.zig`
- `exl3 K3/K2 cooperative indexed GEMV runs at production shape E16` — `src/expert_exl3_kernels.zig`
- `exl3 packedK accepts last dim 16 times 2,3,4` — `src/expert_exl3_kernels.zig`
- `exl3 K3 sorted GEMM matches host MUL1 on small shape` — `src/expert_exl3_kernels.zig`
- `exl3 expert_quant admits K2 K3 K4 mul1 and refuses other k or codebook` — `src/expert_quant.zig`
- `exl3 kFromPackedDim maps last dim 16*K` — `src/expert_quant.zig`
- `exl3 scanExl3TrellisK loads a mixed-K two-layer pack` — `src/expert_quant.zig`
- `test_restack_mixed_k_per_tensor_keeps_each_last_dim` — `tests/convert_qwen38_flash_next_exl3.py`

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

## Suite counts

```
.zig-toolchain/zig build test-build -Doptimize=ReleaseFast && ./zig-out/tests/test
2726 passed; 174 skipped; 0 failed.
```

Converter `--self-test`: 15 tests OK.

## Restack

Destination: `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-exl3k3-8bit`

- Histogram: `{2: 0, 3: 75264, 4: 0}` modal=3 (48 layers + MTP × 512 × 3).
- `expert_quant`: `{format: exl3, k: 3, codebook: mul1, out_scales: svh, source: restack}`
- Bit-exact vs source: L0e0 gate, L0e137 up, L7e10 down, L47e511 gate, L0e0 down — all match.
- Weights GB: total 84.69; experts only 46.72; resident excluding n-gram 52.69 (`ngram_table.bin` 32.00).

## KLD

Not run. `BOX-REQUEST.md` present, `ls BOX-GRANT.md` empty. Other executor holds/held the box. No model server started.

## Commit sha

(filled after commits)

## Open questions

- Indexed GEMV vs host envelope at K3 production (H 2560, I 640, topk=10) failed `expectGemvEnvelope`; naive GEMV at that shape matches host, and indexed matches host on 128. E=16 is run-only for K3 indexed, same as the existing K4 E=16 run test. Reviewer should check whether the generic-K indexed body needs a tighter production parity test.
- KLD/determinism/prefill numbers: blocked on BOX-GRANT.
- Shared-expert and attention in the 305bpw checkpoint are K5 (`head_bits: 5`). Out of scope (K in {2,3,4} for routed experts).

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
