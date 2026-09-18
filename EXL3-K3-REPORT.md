# EXL3-K3 report

Status: in progress. Start sha 238fb8d7, branch agent/exl3-k3.

## Files changed

- `src/expert_exl3.zig`: K2/K3 fixture embeds and shared decode helper. Host unpack/decode was already K-generic; tests lock K2/K3/K4.
- `src/fixtures/exl3_k2_linear.safetensors`: 128x128 MUL1 K2 linear from ponyexl3 CPU direct quantize. Inner bits match ponyexl3 `reconstruct_inner`. Public bits match host Sylvester H128 (same contract as the K4 fixture).
- `src/fixtures/exl3_k3_linear.safetensors`: same for K3 (packed last dim 48).

## Tests added

- `exl3 K3 packed fixture decodes to the library inner and public f16` in `src/expert_exl3.zig`
- `exl3 K2 packed fixture decodes to the library inner and public f16` in `src/expert_exl3.zig`

## Red-first evidence

Command: `.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="exl3 K3 packed"`

```
src/expert_exl3.zig:442:37: error: unable to open 'fixtures/exl3_k3_linear.safetensors': FileNotFound
const fixture_k3_bytes = @embedFile("fixtures/exl3_k3_linear.safetensors");
```

After the fixtures existed, inner matched the reference dequant; public missed host H128 by tens of ULPs until public was rewritten with the host Sylvester (K4 fixture already used that contract; current ponyexl3 `preapply_had_*` is BLAS-associated).

Command: `.zig-toolchain/zig test src/expert_exl3.zig -OReleaseFast --test-filter "packed fixture"`

```
All 3 tests passed.
```

## Suite counts

(pending full suite)

## Commit sha

(pending)

## Open questions

- 305bpw download still incomplete (7 `*.incomplete` shards). Restack and KLD gated on that and the box.
- Whether the 3.05bpw checkpoint is uniform K3 or mixed K is unknown until shards finish.

## Comments

none added

## Ports

- Trellis K-bit funnel unpack: `ponyexl3/ref/trellis.py` `unpack_trellis_tile`; Metal twin `polar_exl3_codeword_pair_window` / `polar_exl3_funnel_pair` in `polarrust-metal-shaders/metal/common/exl3.metal`
- MUL1 codebook (unchanged): `ponyexl3/ref/codebook.py`; Metal `polar_exl3_mul1_decode`
- Direct quantize for fixtures: `ponyexl3/convert/direct.py` `quantize_inner_matrix_direct` + `regularize_public_weight`
