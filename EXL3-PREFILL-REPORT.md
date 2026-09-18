# EXL3 Prefill Report

Run: `exl3-prefill`
Base: `fe5df6af` (`refs/keep/exl3-kernels-r4`)
Box: Apple M5 Max

## Files changed

- `src/expert_exl3_kernels.zig` (item 0 benches only)

## Tests added

- Both in-tree prefill benches now assert `max_e >= 400` and loop C in {512, 2048} at E=512 / top-k 10.

## Red-first evidence

- `max_e >= 400` on the old E=4 / top-k 2 benches: `FAIL (TestUnexpectedResult)` for
  `exl3 512-row prefill: decode-to-f16 gather_mm vs rows kernel` and
  `exl3 512-row production-shape sorted gemm vs affine gather_qmm`.
- After re-parameterization: both green at C=512 and C=2048.

## Suite counts

- Filtered `-Dtest-filter="exl3"`: 22 passed, 0 skipped, 0 failed.
- Full unfiltered suite: pending (gpu held by `exl3-decode` after item 0 live A/B).

## Commit sha

(pending item 0 commit)

## Live table (every row with box, chunk, engagement lines)

Box: Apple M5 Max. Protocol: 100 s idle, kv8, `--no-mtp`, affine through `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-affine-ab`, EXL3 `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-exl3k4-8bit`, interleaved 3 boots, medians of `timings.prompt_per_second`. `MLX_SERVE_NGRAM_WARM=0`. ~4085 prompt tokens.

| arm | boot1 | boot2 | boot3 | median tok/s | chunk | engagement |
| --- | --- | --- | --- | --- | --- | --- |
| EXL3 | 755.8 | 853.7 | 885.8 | **853.7** | 8192 (1 chunk) | `[args] kv-quant: affine 8-bit (group=64)`; `[expert-exl3] engaged`; `[qwen4] MTP head loaded ... drafts armed by --mtp`; `[short-gen] ... path=serial`; `[prefill-trace] tokens=4088 chunks=1 chunk_size=8192`; `[model-settings]` absent; `[spec-stats] mode=` absent |
| affine-ab | 2077.4 | 1871.2 | 2065.5 | **2065.5** | 8192 (1 chunk) | `[args] kv-quant: affine 8-bit (group=64)`; `[qwen4] MTP head loaded ... drafts armed by --mtp`; `[short-gen] ... path=serial`; `[prefill-trace] tokens=4082 chunks=1 chunk_size=8192`; `[model-settings]` absent; `[spec-stats] mode=` absent; no `[expert-exl3]` |

Ratio 853.7 / 2065.5 = **0.413x**. Bar 0.75x of this affine median is **1549 tok/s** (brief 1580 used 2109).

Supporting 2k (2225 tok, same protocol, one-chunk): EXL3 632.3 / 770.0 / 840.4 median 770; affine 1769.4 / 1825.3 / 1814.4 median 1814.

Hermetic serialized 512-row prefill dispatch (E=16 / top-k 10 / H=2560 / I=640, `ubench_force`):

| dispatch | ms |
| --- | --- |
| sort | 0.251 |
| token_prepare | 0.237 |
| gemm_gate | 2.740 |
| gemm_up | 2.726 |
| gemm_down | 3.071 |
| token_reduce | 0.311 |
| **sum** | **9.336** |

In-tree production-shape (E=512 / top-k 10, isolated eval):

| C | EXL3 layer us | affine one-proj us | vs 3x-qmm |
| --- | --- | --- | --- |
| 512 | 15714 | 2168 | 2.41x |
| 2048 | 38806 | 3244 | 3.98x |

`exl3 512-row E=512 topk=10 layer within 2x affine`: 15476 us vs 41403 us full 3x gather_qmm (0.37x).

## Open questions

- Auto chunk is 8192 on both arms once the affine settings row is neutralized (old 512-row affine chunk was the ctx=524288 bar). Live 4k is one 4088-row GEMM window count, not 512-row chunks.
- Per-layer live table at `--prefill-chunk 512` not yet taken (gpu granted to `exl3-decode` on unlock).

## Comments

none added

## Ports (reference ideas taken, for NOTICE)

(none yet)

## Item 0 — Measurement

Status: done (live 4k A/B + bench re-param). Full suite and 512-row live ubench still pending GPU.

## Item 1 — 16-row windows + in-kernel run walk

Status: pending

## Item 2 — Fixed costs, bit-identical

Status: pending

## Item 3 — decode-once + dense f16 GEMM

Status: pending (only if still < 1580 after 1+2)
