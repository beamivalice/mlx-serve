# EXL3 Prefill Report

Run: `exl3-prefill`
Head: `11ca7ebc` (integration)
Box: Apple M5 Max
A/B shas: f54 `ba1503e3` / 728 `89d6f8db`
Six-number sha256: `c9b797bccd843a7b2ace32187e99bbcbf4ed254756fb5abae3373dc93ac2f4e2`

## Files changed

none (A/B + baseline only; Astra study 3 names the next lever)

## Tests added

none

## Red-first evidence

n/a

## Suite counts

not re-run (no source change)

## Commit sha

- Head `11ca7ebc828076616975531e66698961b4ec6d5c`

## Live table

### Interleaved 3-boot A/B, 4k prefill only

Same session, alternating boots, warm then 3× `prompt-4k` `max_tokens=8`. One lease.

**f54b6fdf** sha `ba1503e37e02c2872b5ba86fdfc5658abaf6b9ec89ce2f78e491509bf210fbf9`

| boot | load1 | 3 req tok/s | boot median |
| --- | --- | --- | --- |
| 1 | 2.38 | 1612.8 / 1687.8 / 1678.3 | 1678.3 |
| 2 | 1.88 | 1574.2 / 1654.3 / 1586.7 | 1586.7 |
| 3 | 1.53 | 1562.9 / 1634.6 / 1591.8 | 1591.8 |
| overall median of boot medians | | | **1591.8** |

**728af897** sha `89d6f8db9340c3fcc274df41dd797a8640c7e3b740031b23c3c8f75234f1d32f`

| boot | load1 | 3 req tok/s | boot median |
| --- | --- | --- | --- |
| 1 | 2.10 | 1584.7 / 1803.1 / 1665.3 | 1665.3 |
| 2 | 1.82 | 1577.8 / 1639.0 / 1575.3 | 1577.8 |
| 3 | 1.46 | 1217.8 / 1345.9 / 1304.2 | 1304.2 |
| overall median of boot medians | | | **1577.8** |

DELTA (f54−728)/f54 = **0.88%** (< 5%). **VERDICT: session noise, no offender.** No inner Opus-2 bisect. No fix.

### Six-number pinned baseline on `11ca7ebc`

sha256 `c9b797bc`. 3 boots, EXL3 then affine-ab each boot, `max_tokens=512`, `completion_tokens=512`, `finish_reason=length` every row. Chunk 8192.

**EXL3**

| ctx | prefill b1 / b2 / b3 | prefill median | decode b1 / b2 / b3 | decode median | loadavg |
| --- | --- | --- | --- | --- | --- |
| 4k | 1569.0 / 1434.8 / 1466.4 | **1466.4** | 51.86 / 52.14 / 53.40 | **52.14** | 2.08 / 2.64 / 1.60 |
| 16k | 1570.6 / 1380.7 / 1520.3 | **1520.3** | 51.33 / 51.53 / 53.96 | **51.53** | same |
| 64k | 1191.0 / 1237.5 / 1372.8 | **1237.5** | 50.82 / 52.40 / 52.69 | **52.40** | same |

**affine-ab** (same session)

| ctx | prefill b1 / b2 / b3 | prefill median | decode b1 / b2 / b3 | decode median | loadavg |
| --- | --- | --- | --- | --- | --- |
| 4k | 1257.1 / 1134.6 / 1970.6 | **1257.1** | 56.22 / 57.07 / 58.28 | **57.07** | 2.42 / 2.44 / 1.81 |
| 16k | 1454.7 / 1476.5 / 1838.7 | **1476.5** | 57.56 / 59.41 / 59.13 | **59.13** | same |
| 64k | 1776.5 / 1895.7 / 1892.9 | **1892.9** | 55.65 / 57.99 / 57.66 | **57.66** | same |

Affine 4k/16k boots 1–2 are cold vs boot 3 (1971 / 1839). Ratios at medians: 4k 1466/1257 = **1.17×**, 16k 1520/1476 = **1.03×**, 64k 1238/1893 = **0.65×**. Decode 0.91 / 0.87 / 0.91. 4k bar 1575: **not met** at this session’s EXL3 median 1466 (boot 1 was 1569).

## Open questions

- Astra study 3 on `11ca7ebc` names the next prefill lever; none started here.
- Affine 4k first two boots look compile-cold on this binary; boot 3 1971 is the familiar band.
- 64k EXL3 1238 vs affine 1893 is the kv climb, unchanged by the A/B.

## Comments

none added

## Ports

none
