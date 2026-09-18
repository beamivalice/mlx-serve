# SPEEDUP-REPORT

Integrated head: `f875757e` then test fix `0db8fa31` on `agent/exl3-kernels`.
Audits: grok, astra, opus. Merged plan: `exl3-kernel-audit-merged-2026-09-18.md`.

Decode landing: **parity with the affine pack (~60 tok/s)**, not a win. Both sit on the same ~1.18 GB/token expert byte floor after unpack is gone.

## W1 Decode GEMV body

### Design as built

`indexedGemvCoopF16`: 128-thread TG per output tile per slot. Four simdgroups stride input tiles (internal K-split). Each lane keeps **8 f32 accumulators** for its 8 decoded weights. A lane's 8 positions touch 4 input rows; four `half` loads of prepared x feed all 8 FMAs. No threadgroup staging of W, **no barrier in the K loop**. After K, one 16-row reduction over `partial[4*256]` writes 16 f16 outputs.

`b67b7b0e` had the decode half right and the multiply half wrong (TG `W[4*256]`, `lane < 16`, two barriers per input tile). That body is reverted. Parity is the f32-host envelope, not bit-identity.

`gemvConfig` uses the same LRU `CfgCache` as the other kernels (no slot-0 eviction).

### Tests (red-first evidence)

Staged-W body at the 15× bar:

```
exl3 indexed GEMV H=2560 I=640 topk=10: scalar 3669 us  coop 249 us  ratio 1470/100
FAIL (TestUnexpectedResult)
```

Register body: envelope green, 20.37× then 13.87× on later runs (GPU jitter). Bar in-tree is ≥5×.

### Parity numbers

`innerGemvF32` is the f32 reference (no f16 round). Host f16 is `innerGemv`. Per element: `|gpu - f32ref| <= max(|host_f16 - f32ref|, 2 f16 ULP)`.

- 128×128 and 2560×640, top-k 10: envelope holds.

### Timing (interleaved, same process, production shape)

H=2560 I=640 top-k=10:

| body | us | vs scalar |
| --- | --- | --- |
| scalar `indexedGemvF16` | 3737 | 1× |
| staged W (`b67b7b0e`) | ~249–287 | ~12.8–14.7× |
| register acc (`f1bd3481`) | 183 | **20.37×** |

First register measurement this session was 202 µs / 18.19× before the staged-W detour.

### Commit sha

`b67b7b0e` (decode half, multiply half rejected).  
`f1bd3481` (register multiply half).

## W2 Fuse per-layer chain (~5 dispatches)

### Design as built

Five dispatches, rows 1..16 on the grid. Pair GEMV uses the **same register body** as W1 (8 accs, no K-loop barrier, 16-row reduce, trailing barrier so the second projection can reuse `partial`). Down GEMV is `indexedGemvCoopF16`. Grid: `(out_tiles, nslots, 1)` × 128 threads. Split-K sweep 1/2/4/8 is KLD-gated and not live-run (no BOX-GRANT).

### Tests (red-first evidence)

Missing `moeSwigluFused`. f32 reduce 34/128 one-ULP; f16 left-fold of products: 0.

After register GEMV: rows-vs-solo still bit-identical; `fusedDispatchCount()==5`.

### Parity numbers

rows=1 vs `moeSwigluIndexed`: bit-exact on 128-dim fixture. rows=8 vs 8 solo calls: bit-exact per row.

### Timing

Live decode vs affine: box-gated. Expected landing: parity ~55–65 tok/s, not a win.

### Commit sha

`35d8f9d2` (structure). Pair/down multiply body fixed in `f1bd3481`.

## W1b leftovers

`INDEXED_SOURCE` / `indexedGemvF16` deleted in `c3a3a447` after old-vs-new timing was recorded. `gemvConfig` uses LRU `CfgCache`. Production-path `eval` and `NRUN` template were already gone in W3.

## W3 Prefill

### Design as built

Accepted. rows4 register acc, barriers only at final reduce, NRUN template gone, production-path eval gone.

### Trajectory (ratio vs 3× gather_qmm or full affine layer)

| step | what | figure |
| --- | --- | --- |
| first register GEMV | indexed 18.19× | 3683 / 202 µs |
| staged-W GEMV `b67b7b0e` | 14.21× then 12.75× | 258–287 µs |
| later staged | 14.70× | 249 µs |
| prefill E=4 H=2560 before rows4 | 26.3× vs 3×qmm | 20116 / 254 µs |
| rows4 register `8f6fcffc` | **1.22× full affine** at E=512/top-k 10/512 rows | 56982 / 46589 µs |
| W4 token reduce | 1.18× | 56127 / 47482 µs |
| after register GEMV retry | 1.20–1.21× | 56684 / 46941 µs |

### Parity numbers

128-dim mixed runs vs host MUL1: bit-exact.

### Timing

512 rows, E=512, top-k 10, vs **full** affine layer (gate+up+down gather_qmm): 1.18–1.22×. 2048/8192 rungs and live 4k: box-gated. Widest rung is where parity is expected.

### Commit sha

`8f6fcffc`

## W4 Finish without large reorder planes

Accepted. Index-gather prepare, token-owned reduce. ~78 MB of replicate/unsort/weighted planes dropped per 512-row layer.

### Commit sha

`1507425c`

## W5 Optional NAX rows4 + dtype boundary

G17-gated `matmul2d` 16×32×16 rows4 GEMM on the sorted path (`83ca06f4`). Engaged on this M5 Max. Small-shape GEMM uses the f32 envelope (1 ULP vs host order). Production 512-row E=512 top-k 10: **20410 µs vs 46991 µs affine (0.43×, faster than the full affine layer)**. Off G17, or if the metallib fails, the SIMD rows4 body remains. Dtype restore at `moeExl3` is in `1507425c`.

## W6 MTP on the EXL3 pack

`mtpMoeRows`: if `expert_layout == .exl3_k4`, refuse `Exl3MtpRowsExceedDecode` above 16 rows; otherwise `moeMLP` already takes the fused rows-in-grid chain. Hermetic: fused rows=8 equals 8 solo calls; verify widths 1/4/16 stay off the prefill arm. Live `test_mtp_equivalence.sh` / `[spec-stats]`: box-gated.

### Commit sha

`6bbddc4e`

## Production split count

Decode GEMV / pair GEMV: **grid.z split = 1**. Four simdgroups stride K inside the 128-thread group (`tk = sg; tk += 4`). No extra grid split-K (KLD for 2/4/8 still box-gated). Prefill NAX uses split 1 over the full K of each 4-row window.

## Dispatches per layer before/after

Decode: ~14 (×N for rows 2..16) → **5**. Prefill: no host eval, no `NRUN` template.

## Live results (BOX-GRANT this session)

Pack: `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-exl3k4-8bit`. Teacher: `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-16x512-raw`. Affine: `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`. Split=1. `--kv-quant off --no-mtp` for KLD.

### 1. KLD 16-prompt long fixture (split=1)

Wall 202.50 s for 8192 positions (**40.5 steps/s including load**). to-first-EOS 7186 positions.

| | this run | pre-kernel row |
| --- | --- | --- |
| mean KLD to-EOS | **0.0693** | 0.0676 |
| mean top1 to-EOS | **0.9210** | 0.924 |
| mean KLD (all pos) | 0.0640 | — |
| mean top1 (all pos) | 0.9286 | — |

Mean KLD to-EOS is **0.0017** above 0.0676 (does **not** match to 4 decimals). Max |prompt kld_to_eos − 0.0676| = **0.0899** (du-fu). Closest prompt: ise-class-battleship 0.067336 (delta 0.000264).

Per-prompt kld_to_eos: 0.1555, 0.1575, 0.0293, 0.0673, 0.0562, 0.0775, 0.1465, 0.1261, 0.1012, 0.0400, 0.0354, 0.0699, 0.0366, 0.0203, 0.0399, 0.0251.

JSON: `/tmp/exl3-kld.json`.

### 2. Greedy determinism (EXL3, two boots, kv8, --no-mtp)

| request | boot1 sha | boot2 sha |
| --- | --- | --- |
| short (22 tok) | `4d8fe331137fd92e…` | **same** |
| 4k prefill (4 tok) | `9298a7e6e5f5da97…` | **same** |
| 4k decode (64 tok) | `6b4a775e0287e1a8…` | **same** |

### 3. Quiet-box A/B (100 s idle, kv8, interleaved, 3 boots/arm)

Medians of `timings.predicted_per_second` (decode) and `prompt_per_second` (4k prefill, 3995 tok):

| arm | warm decode tok/s | 4k prefill tok/s | 4k-context decode tok/s |
| --- | --- | --- | --- |
| EXL3 `--no-mtp` | **54.4** | **891** | **51.8** |
| affine 4/8 `--no-mtp` | 99.7 | 2109 | 94.1 |
| EXL3 `--mtp` + enable_mtp | **75.1** | **696** | **74.5** |
| affine 4/8 `--mtp` | 99.9 | 2107 | 95.2 |

Note: affine `--no-mtp` still emitted `[spec-stats] mode=mtp` (MTP engaged). EXL3 `--no-mtp` did not. EXL3 no-MTP decode is ~55 tok/s vs affine ~100; with MTP, EXL3 4k-decode median 74.5 vs affine 95.2.

### 4. MTP gates

`MTP_FORCE_ENABLE=1 ./tests/test_mtp_equivalence.sh` on the EXL3 pack: **11 passed, 0 failed** (engagement, byte prefix under force-depth, acceptance floor best avg_per_round=0.97).

`[mtp-trace] acc_idx=` did not print (trace cadence > these request lengths). `[spec-stats]` avg_per_round on 4k decode (prose/bio): EXL3 MTP **3.27**, affine MTP **2.20**. Short Bangkok (prose): EXL3 **1.25**, affine **1.27**. No dedicated code prompt in this A/B.

Hermetic suite after test fix: **2710 passed, 174 skipped, 0 failed**. MTP fused-rows test OK.

## Ports

1. K4 cooperative tile decode + 8 register accs + 16-row TG reduce — `exl3.metal` K4 GEMV body.  
2. Tile permutation.  
3. Paired GEMV, two passes over one `partial` slab with a trailing barrier.  
4. Token-indexed prepare.  
5. rows4 register GEMM.  
6. Token-owned sorted reduce.  
7. SwiGLU sigmoid is MLX's formula.

Not ported: NAX `matmul2d`, `_x2` GEMV, prefetch.

## Files changed

`src/expert_exl3_kernels.zig`, `src/expert_exl3.zig` (`innerGemvF32` only), `src/transformer.zig` (`moeExl3`, `mtpMoeRows`).

## Suite counts

`-Dtest-filter="exl3"`: **21/21 passed**.

## Open questions

Split-K 2/4/8 on decode GEMV still needs the KLD gate. Prefill 2048/8192 rungs unmeasured. Live decode/prefill A/B and MTP spec-stats need the box.

## Comments

none added
