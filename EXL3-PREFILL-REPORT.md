# EXL3 Prefill Report

Run: `exl3-prefill`
Base: `fe5df6af` (`refs/keep/exl3-kernels-r4`)
Box: Apple M5 Max

## Files changed

- `src/expert_exl3_kernels.zig` (run-aligned window table from buildRuns; default 32 after 4k/8k 3-boot win)
- `NOTICE` (decode-full entry already removed)

## Tests added

- Item 0: both in-tree prefill benches assert `max_e >= 400` and loop C in {512, 2048} at E=512 / top-k 10.
- Item 1: `exl3 sorted GEMM 16-row windows match 4-row per row` (mixed runs, n=32, bit identity). NAX default and SIMD via `MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1`.
- Item 2: `exl3 moePrefill matches staged sorted chain` (old two-prepare / MLX SwiGLU / inv-reduce vs new path, bit identity).
- Item 3: `exl3 decode-once transient bytes at production dims`; `exl3 decode-full K4 matches host inner`; `exl3 decode-once GEMM matches 16-row windows on long run`; `exl3 decode-once stage ubench C=2048 sweep N`.

## Red-first evidence

- Item 0: `max_e >= 400` on E=4 benches: `FAIL (TestUnexpectedResult)`. Green after E=512 / top-k 10.
- Item 1: 16-row grid with kernel still `win * 4u`: `expected 17898, found 0` (`FAIL TestExpectedEqual`). Green after `start = win * WIN` and in-kernel run walk.
- Item 2: characterization `exl3 moePrefill matches staged sorted chain` green on the old body, still green after pair-prepare / int32 / scatter / mid+downFinishReduce.
- Item 3: decode-full with grid in threadgroups (not threads) wrote zeros: `expected 47816, found 0`. Green after `set_grid(out_tiles * 128, ...)`.

## Suite counts

- Item 0 filtered `-Dtest-filter="exl3"`: 22 passed.
- Item 1 filtered `-Dtest-filter="exl3"`: 23 passed, 0 failed (includes new identity test). SIMD-arm rerun of sorted GEMM tests: 4/4.
- Item 2 filtered `-Dtest-filter="exl3"`: 24 passed, 0 failed.
- Item 3 filtered `-Dtest-filter="exl3"`: 28 passed, 0 failed.

## Commit sha

- item 0: `26332d0a37ba48c6a65dfdd86dbbc5da93aea253`
- item 1: `ccb4916c5f8feb9abbb79deee39b4bf1cc6e8180`
- item 2: `f158aec058838d253787158d931c547d173f21e3`

## Live table (every row with box, chunk, engagement lines)

Protocol: 100 s idle, kv8, `--no-mtp`, affine through `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-affine-ab`, EXL3 `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-exl3k4-8bit`, interleaved 3 boots, medians of `timings.prompt_per_second`. `MLX_SERVE_NGRAM_WARM=0`. Box: Apple M5 Max. ~4085 prompt tokens. Chunk 8192, one chunk.

### Baseline (4-row windows)

| arm | boot1 | boot2 | boot3 | median tok/s | chunk | engagement |
| --- | --- | --- | --- | --- | --- | --- |
| EXL3 | 755.8 | 853.7 | 885.8 | **853.7** | 8192 (1 chunk) | `[args] kv-quant: affine 8-bit (group=64)`; `[expert-exl3] engaged`; `[qwen4] MTP head loaded ... drafts armed by --mtp`; `[short-gen] ... path=serial`; `[prefill-trace] tokens=4088 chunks=1 chunk_size=8192`; `[model-settings]` absent; `[spec-stats] mode=` absent |
| affine-ab | 2077.4 | 1871.2 | 2065.5 | **2065.5** | 8192 (1 chunk) | same kv8 / serial / MTP-head-loaded-not-armed; no `[expert-exl3]` |

### After item 1 (16-row windows + run walk)

| arm | boot1 | boot2 | boot3 | median tok/s | chunk | engagement |
| --- | --- | --- | --- | --- | --- | --- |
| EXL3 | 1157.7 | 1466.0 | 1464.5 | **1464.5** | 8192 (1 chunk) | `[args] kv-quant: affine 8-bit (group=64)`; `[expert-exl3] engaged`; `[qwen4] MTP head loaded ... drafts armed by --mtp`; `[short-gen] ... path=serial`; `[prefill-trace] tokens=4083 chunks=1 chunk_size=8192`; `[model-settings]` absent; `[spec-stats] mode=` absent |
| affine-ab | 2074.7 | 2100.6 | 2110.5 | **2100.6** | 8192 (1 chunk) | same kv8 / serial / MTP-head-loaded-not-armed; `[prefill-trace] tokens=4084 chunks=1 chunk_size=8192` |

Ratio 1464.5 / 2100.6 = **0.697x**. Bar 1580 tok/s (0.75x of 2109, or 0.75x of 2100.6 = 1575). Still short.

### After item 2 (paired prepare, int32 order, scatter, mid/finish fuse)

| arm | boot1 | boot2 | boot3 | median tok/s | chunk | engagement |
| --- | --- | --- | --- | --- | --- | --- |
| EXL3 | 1186.7 | 1454.4 | 1470.3 | **1454.4** | 8192 (1 chunk) | `[args] kv-quant: affine 8-bit (group=64)`; `[expert-exl3] engaged`; `[qwen4] MTP head loaded ... drafts armed by --mtp`; `[short-gen] ... path=serial`; `[prefill-trace] tokens=4088 chunks=1 chunk_size=8192`; `[model-settings]` absent; `[spec-stats] mode=` absent |
| affine-ab | 2070.7 | 2097.4 | 2108.4 | **2097.4** | 8192 (1 chunk) | same kv8 / serial / MTP-head-loaded-not-armed; `[prefill-trace] tokens=4083 chunks=1 chunk_size=8192` |

Ratio 1454.4 / 2097.4 = **0.693x**. Still short of 1580. Item 3 is required.

### After item 3 (decode-once wired into moePrefill — not kept)

Load averages are 1-minute from `uptime` at Model ready.

| arm | boot1 | boot2 | boot3 | median tok/s | chunk | loadavg 1m | engagement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXL3 | 191.5 | 215.0 | 229.5 | **215.0** | 8192 (1 chunk) | 2.24 / 2.35 / 2.78 | kv8; `[expert-exl3] engaged`; MTP head loaded not armed; `path=serial`; `[prefill-trace] tokens=4088 chunks=1 chunk_size=8192 chunked=21300ms`; `[model-settings]` absent; `[spec-stats] mode=` absent |
| affine-ab | 2083.4 | 2092.9 | 1693.0 | **2083.4** | 8192 (1 chunk) | 2.51 / 2.23 / 3.46 | kv8; serial; MTP head loaded not armed; `[prefill-trace] chunked=1929ms` |

Ratio 215 / 2083 = **0.103x**. Regression vs item 2 (1454 tok/s). moePrefill restored to the 16-row window path. `decodeFull` / `innerGemmByRuns` stay in-tree and tested.

Transient bytes (bound per group, production H=2560 I=640): **64 * 2560 * 640 * 2 = 104857600 (100.0 MiB)**. Independent of chunk width once any long run exists; C=512 typically all-short so 0; C=2048 and C=4096 hit the 100 MiB cap.

KLD: production path is item 2 (windows). Decode-once was not left in moePrefill, so logits are unchanged vs the 0.0693 / 0.921 reference. KLD compare was not re-run on the 215 tok/s wiring.

Hermetic serialized 512-row prefill dispatch (E=16 / top-k 10 / H=2560 / I=640):

| dispatch | 4-row ms | 16-row ms |
| --- | --- | --- |
| sort | 0.251 | 0.236 |
| token_prepare | 0.237 | 0.237 |
| gemm_gate | 2.740 | 1.021 |
| gemm_up | 2.726 | 1.058 |
| gemm_down | 3.071 | 1.381 |
| token_reduce | 0.311 | 0.235 |
| **sum** | **9.336** | **4.168** |

In-tree production-shape (E=512 / top-k 10):

| C | 4-row layer us | 16-row layer us |
| --- | --- | --- |
| 512 | 15714 | 6954 |
| 2048 | 38806 | 15121 |

`exl3 512-row E=512 topk=10 layer within 2x affine` after item 1: 6648 us vs 41507 us (0.16x).

## Open questions

- Auto chunk is 8192 on both arms once the affine settings row is neutralized.
- Item 1 landed 1465 tok/s vs 1580 bar. Item 2 (scatter, int32 slots, paired prepare, cache gemmNaxOn, fuse planes) is required.

## Comments

none added

## Ports (reference ideas taken, for NOTICE)

- 16-row NAX fill of the existing 16x32x16 `matmul2d` tile (`act[4+c]` = row `origin.y+8`); run walk one pass per distinct expert in the window.
- Prefill mid/finish fused onto the existing decode mid + down-finish-reduce leaves (no new arithmetic).
- Full-trellis K4 MUL1 decode into a dense f16 [out, in] buffer (ExLlamaV3 codec / EXL3 Metal decode-full leaf); stock `mlx_matmul` over a padded expert group.

## Item 0 — Measurement

Status: done.

## Item 1 — 16-row windows + in-kernel run walk

Status: done. Live median 1465 tok/s. Below 1580.

## Item 2 — Fixed costs, bit-identical

Status: done. Live median 1454 tok/s (flat vs item 1 within boot noise). Below 1580.

## Item 3 — decode-once occupancy retry

Grid is now one simdgroup per 16x16 tile (`in_tiles * 32, out_tiles, n_exp`). Bit-exact vs host and vs 16-row windows.

### Per-stage (one gate/up projection, production H=2560 I=640)

C=2048 (nslots=20480), after warmup:

| N (decode-once if run >= N) | window us | decode-once us | decode_full ms/group | matmul_pad ms/group | pad/live rows | host_group |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 5172 | 27772 | 0.83–0.87 (first 4.4) | 1.06–1.66 | 3712/2497 .. 3648/2548 (64 exp) | 0.075 ms **eval=1** |
| 32 | 5172 | 19694 | 0.82–0.89 | 1.03–1.13 | similar | 0.063 ms eval=1 |
| 64 | 5172 | 4553 | (no long runs) | — | — | 0.054 ms eval=1 |

C=8192 (nslots=81920): window **13786 us**. N=16 decode-once **58713 us**. decode_full 0.65–3.2 ms/group, matmul_pad 2.4–2.8 ms/group (pad ~9200 vs live ~8100, ~50 experts). host_group 0.141 ms eval=1.

Token-major reduce is unchanged (`downFinishReduce` after scatter). Host grouping eval inside the layer is a red flag (one `mlx_array_eval` of sorted eids per `innerGemmByRuns`).

Winner of the N sweep: **N=64** (falls through to windows) at C=2048; at C=8192 even N=16 loses 4x. PROFILED arm still loses. `moePrefill` stays on 16-row windows.

### Chunk sweep (16-row windows, not decode-once)

8k prompt ~8085 tok, 4k ~4055 tok. One boot each. Admission width matches `--prefill-chunk`. Default sizer is 8192.

| prompt | chunk | EXL3 tok/s | affine tok/s | EXL3 peak_bytes | affine peak_bytes | loadavg 1m EXL3 |
| --- | --- | --- | --- | --- | --- | --- |
| 8k | 2048 | 1273 | 1741 | 69.3e9 | 74.2e9 | 2.88 |
| 8k | 4096 | 1319 | **1860** | 69.9e9 | 74.8e9 | 4.04 |
| 8k | 8192 | **1438** | 1712 | 71.3e9 | 76.1e9 | 4.29 |
| 4k | 2048 | 926 | 1518 | 68.3e9 | 74.3e9 | 3.98 |
| 4k | 4096 | 1228 | **1846** | 68.9e9 | 74.8e9 | 3.11 |
| 4k | 8192 | **1396** | 1738 | 68.9e9 | 74.8e9 | 3.03 |

3-boot 8k, EXL3 at its best (8192) vs affine at its best (4096):

| arm | boot1 | boot2 | boot3 | median | loadavg 1m |
| --- | --- | --- | --- | --- | --- |
| EXL3 8192 | 1446.9 | 1567.0 | 1558.6 | **1558.6** | 2.96 / 2.58 / 2.92 |
| affine 4096 | 1650.6 | 1938.6 | 1939.8 | **1938.6** | 2.76 / 3.16 / 2.86 |

Default sizer picks 8192, which **is** EXL3's winner. Affine prefers 4096. The old 583 vs 891 at chunk 4096 was the 4-row path and is void; on 16-row windows, wider chunks help EXL3 (926 → 1228 → 1396 at 4k; 1273 → 1319 → 1438 at 8k).
