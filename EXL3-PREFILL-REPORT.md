# EXL3 Prefill Report

Run: `exl3-prefill`
Integration base: `728af897` (`feat/qwen4-exl3`)
Rebased unique commit: `e207616f`
Box: Apple M5 Max
Binary sha256: `cf84a38b66f9b890a85eccc1b46d5654a2b032e9182422fd946d91413deef00d`

Rebase kept upstream selector (`d68bea4c`), activation dtype (`43cf388e` = B15), window capacity refuse (`2aac27da`), window-table frees (`ac5ae2c2`). Dropped as duplicates: `7e838510` (B15), `72fc82d7` selector pin, `9c9b3480` serial GPU table. Unique work replayed: host-table already on `728af897`; GEMM config reuse (`nwin` out of the cache key, `set_grid` per dispatch) plus per-chunk `[prefill-trace]` and engagement log.

## Files changed

- `src/expert_exl3_kernels.zig` — `GemmSortedKey` is `(in,out,rows,win)`; grid set on every apply; `[exl3-gemm] win= aligned= nwin= mixed= n=` once at `n>=2048`.
- `src/generate.zig` — `[prefill-trace] chunk pos= end= width= ms=` per chunk.

## Tests added

- `exl3 aligned GEMM reuses config across nwin`

## Red-first evidence

- Long-run then short-run at n=40 WIN=32: aligned short matches stride after the small-grid call (stale grid would drop windows).

## Suite counts

- `-Dtest-filter="exl3"`: **46 passed**, 0 failed.
- `-Dtest-filter="mtp"`: **123 passed**, 1 skipped, 0 failed.

## Commit sha

- `e207616fe70766c07759d705c48ae5e1d09351f6` on `728af897`

## Live table

Affine-ab **reused from the earlier GPU-table session** (2101.3 / 2173.1 / 2114.8 prefill, 57.36 / 58.69 / 58.39 decode-512). Same session as this rebase was impossible (new head, new boots). Shared prompts, nonce prefix, `max_tokens` 512, `completion_tokens=512`, `finish_reason=length` on every six-number and WIN-replay row.

### Rebased six-number (`e207616f`, sha `cf84a38b`)

Engagement: `[exl3-gemm] win=32 aligned=1 nwin=1670 mixed=0 n=45660`. Chunk 8192.

| ctx | tokens | prefill b1 / b2 / b3 | prefill median | decode b1 / b2 / b3 | decode median | loadavg 1m |
| --- | --- | --- | --- | --- | --- | --- |
| 4k | 4567 / 4565 / 4570 | 1618.8 / 1359.7 / 1397.0 | **1397.0** | 53.48 / 51.40 / 52.05 | **52.05** | 2.16 / 1.56 / 1.55 |
| 16k | 17626 / 17630 / 17629 | 1594.7 / 1430.3 / 1448.1 | **1448.1** | 53.87 / 52.32 / 52.92 | **52.92** | same |
| 64k | 69939 / 69938 / 69938 | 1317.6 / 1300.2 / 1326.4 | **1317.6** | 50.52 / 51.68 / 51.70 | **51.68** | same |

vs frozen affine: 4k 1397/2101 = **0.665×**; 16k 1448/2173 = **0.666×**; 64k 1318/2115 = **0.623×**. Decode 0.91 / 0.90 / 0.89. 4k bar 1575: **not met** on this head (boot 1 was 1619). Prior host-table-only head (`e92aeaba`) was 1691 / 1628 / 1519.

### 64k penalty

Uninstrumented boot 1 full-width chunks (ms): 5606, 5318, 5354, 5362, 5858, 6491, 6782, 7375, tail 4847 (4402 tok). Climbs after kv≈32k.

`QWEN4_PROFILE_FWD=all` one 64k boot (sha `cf84a38b`, max_tokens 16, prefill 1519 tok/s — extra evals flatten wall):

| kv | mlp (MoE) ms | attn ms | gdn ms | ple ms |
| --- | --- | --- | --- | --- |
| 8192 | 2462 | 1351 | 1013 | 294 |
| 16384 | 2419 | 1002 | 997 | 227 |
| 32768 | 2428 | 1061 | 999 | 186 |
| 49152 | 2434 | 1126 | 1005 | 159 |
| 65536 | 2345 | 1118 | 966 | 479 |

**Growing term is not MoE-prefill.** mlp is flat ~2.42 s/chunk (host eids eval sits inside it). Uninstrumented climb after 32k is kv-length attention/QSA; PROFILE's per-block evals compress that slope. Parallel GPU window table **not built** (would not beat a flat mlp).

### Study-2 WIN 16 vs 32 (identical content)

Same binary `cf84a38b`. Nonces `rep{N}-4k` / `rep{N}-16k` shared across arms. Tokens 4542 / 17603 every row. Selector: `win=16 nwin=3056–3062` and `win=32 nwin=1659–1662`, mixed=0.

| arm | ctx | prefill b1 / b2 / b3 | prefill median | decode median |
| --- | --- | --- | --- | --- |
| w16 | 4k | 1463.9 / 1556.7 / 1283.9 | **1463.9** | 53.7 |
| w32 | 4k | 1551.8 / 1558.2 / 1418.2 | **1551.8** | 49.8 |
| w16 | 16k | 1507.9 / 1492.8 / 1253.3 | **1492.8** | 53.8 |
| w32 | 16k | 1621.8 / 1350.0 / 1377.5 | **1377.5** | 50.5 |

Boot 1 paired: 32 wins (1552/1622 vs 1464/1508). Boots 2–3 noisy (load 1.8–2.4). Default 32 kept. No width >32. Occupancy: NAX still two dest chains, 32 f32/lane; no Metal occupancy API wired; none promoted.

## Open questions

- Integration-head 4k 1397 vs prior 1691 and vs 1800. Decode-track f32 reduce is on this head.
- 16k/64k 8k-class 1680 not met (1448 / 1318).
- Cache-resident partial decode next; decode-once dead; chunk 16384 not admitted.

## Comments

none added

## Ports

none
