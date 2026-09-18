# EXL3 Prefill Report

Run: `exl3-prefill`
Base: `fe5df6af` (`refs/keep/exl3-kernels-r4`)
Box: Apple M5 Max

## Files changed

- `src/expert_exl3_kernels.zig` (item 0 benches; item 1 16-row windows + run walk)

## Tests added

- Item 0: both in-tree prefill benches assert `max_e >= 400` and loop C in {512, 2048} at E=512 / top-k 10.
- Item 1: `exl3 sorted GEMM 16-row windows match 4-row per row` (mixed runs, n=32, bit identity). NAX default and SIMD via `MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1`.

## Red-first evidence

- Item 0: `max_e >= 400` on E=4 benches: `FAIL (TestUnexpectedResult)`. Green after E=512 / top-k 10.
- Item 1: 16-row grid with kernel still `win * 4u`: `expected 17898, found 0` (`FAIL TestExpectedEqual`). Green after `start = win * WIN` and in-kernel run walk.

## Suite counts

- Item 0 filtered `-Dtest-filter="exl3"`: 22 passed.
- Item 1 filtered `-Dtest-filter="exl3"`: 23 passed, 0 failed (includes new identity test). SIMD-arm rerun of sorted GEMM tests: 4/4.

## Commit sha

- item 0: `26332d0a37ba48c6a65dfdd86dbbc5da93aea253`
- item 1: (pending)

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

## Item 0 — Measurement

Status: done.

## Item 1 — 16-row windows + in-kernel run walk

Status: done. Live median 1465 tok/s. Below 1580.

## Item 2 — Fixed costs, bit-identical

Status: in progress

## Item 3 — decode-once + dense f16 GEMM

Status: pending (only if still < 1580 after 1+2)
