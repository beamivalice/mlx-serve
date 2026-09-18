# EXL3 Prefill Report

Run: `exl3-prefill`
Base: `fe5df6af`
Box: Apple M5 Max

## Files changed

- `src/expert_exl3_kernels.zig` — stride and run-aligned window tables; `innerGemmSortedWinAlign`; default WIN=32 aligned after four-arm live medians.

## Tests added

- `exl3 run-aligned windows match stride per row` (mixed runs + tails, WIN 16 and 32, bit identity). NAX default; SIMD via `MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1`.
- `exl3 window 16 vs 32 production C=2048` prints four-arm serialized stats at C=2048 and C=8192 (nwin, mixed, decodes, us).

## Red-first evidence

- Identity test requires mixed stride windows (`st.mixed > 0`) and zero mixed after alignment (`al.mixed == 0`), then per-row `expectEqual` vs stride. Green on NAX and SIMD.
- If the aligned table had straddled runs, bits would differ from stride (same arithmetic per row only when each row uses its own expert's trellis once).

## Suite counts

- `-Dtest-filter="exl3"`: **27 passed**, 0 failed.
- SIMD rerun of `exl3 run-aligned windows match stride per row`: passed.

## Commit sha

(filled after commit)

## Live table (every row with box, chunk, engagement lines)

Box: Apple M5 Max. kv8, `--no-mtp`, affine-ab symlink, `--prefill-chunk 8192`. `MLX_SERVE_NGRAM_WARM=0`. Admission width=8192 admit. MTP head loaded not armed. Loadavg is 1-minute at Model ready.

### Serialized one-projection (H=2560 I=640, sorted slots)

| C | arm | us | nwin | mixed | decodes |
| --- | --- | --- | --- | --- | --- |
| 2048 | stride-16 | 6429 | 1280 | 484 | 1764 |
| 2048 | aligned-16 | 4626 | 1533 | 0 | 1533 |
| 2048 | aligned-32 | 4653 | 972 | 0 | 972 |
| 2048 | stride-32 | 4537 | 640 | 499 | 1140 |
| 8192 | stride-16 | 15874 | 5120 | 476 | 5596 |
| 8192 | aligned-16 | 15129 | 5351 | 0 | 5351 |
| 8192 | aligned-32 | 11878 | 2802 | 0 | 2802 |
| 8192 | stride-32 | 12509 | 2560 | 493 | 3053 |

Mixed = windows whose 16/32 consecutive sorted slots contain more than one expert. Decodes = trellis passes (one per run inside a window). Alignment zeros mixed; WIN=32 cuts decodes further.

### 4k (~4054 tok), 3 boots, chunk 8192

| arm | boot1 | boot2 | boot3 | median | loadavg 1m |
| --- | --- | --- | --- | --- | --- |
| stride-16 | 1466.1 | 1530.4 | 1064.7 | **1466.1** | 1.70 / 3.01 / 3.65 |
| aligned-16 | 1554.4 | 1623.8 | 1567.1 | **1567.1** | 2.78 / 3.54 / 3.87 |
| aligned-32 | 1678.2 | 1730.5 | 1728.0 | **1728.0** | 2.77 / 3.30 / 3.30 |
| stride-32 | 1689.0 | 1688.5 | 1167.7 | **1688.5** | 2.81 / 3.33 / 3.41 |
| affine-ab | 2121.0 | 2148.5 | 2122.0 | **2122.0** | 3.01 / 3.34 / 3.81 |

stride-16 median 1466 matches the 1465 baseline. Winner: **aligned-32** 1728. Ratio vs affine 1728/2122 = **0.81×**. 4k bar 1575 **met**.

### 8k (~8084 tok), 3 boots, chunk 8192

| arm | boot1 | boot2 | boot3 | median | loadavg 1m |
| --- | --- | --- | --- | --- | --- |
| stride-16 | 980.8 | 1149.0 | 1064.7 | **1064.7** | 3.71 / 3.29 / 5.05 |
| aligned-16 | 1316.9 | 1375.5 | 1377.6 | **1375.5** | 4.02 / 3.65 / 5.46 |
| aligned-32 | 1513.5 | 1517.8 | 1471.1 | **1513.5** | 3.71 / 3.55 / 4.76 |
| stride-32 | 1347.3 | 1172.9 | 1582.5 | **1347.3** | 3.63 / 3.77 / 4.66 |
| affine-ab | 1783.4 | 2058.6 | 1503.5 | **1783.4** | 3.39 / 3.45 / 4.73 |

Winner: **aligned-32** 1514. 8k loadavg 3.3–5.5 (other compilers on box); stride-16 8k is below the earlier 1559 8k row for that reason. Default set to aligned WIN=32.

### 16k / `--prefill-chunk 16384`

Sizer still pins **width=8192** (`needed=8386 MB … verdict=admit`). EXL3 1082 tok/s (16141 tok, loadavg 4.37), affine 1526 tok/s (loadavg 4.96). One boot each.

## Open questions

- Window table is host `buildRuns` after eids eval (one per GEMM). Metal cannot atomically write an mlx input buffer.
- 16384 is not admitted.

## Comments

none added

## Ports (reference ideas taken, for NOTICE)

none
