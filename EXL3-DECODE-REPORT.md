# EXL3-DECODE-REPORT (round 8) — 64k prefill growth diagnosis

No kernel edits. Integration base `dec5319d`. Measured sha `ce0c3f55` (prefill-trace mem line after `mlx_clear_cache`). One lease per pack. `prompt-64k.txt`, nonce, max_tokens 16, `--prefill-trace`, `QWEN4_PROFILE_QSA=1`, no `NGRAM_WARM` override. Chunk 8192. kv8. Box Apple M5 Max.

## EXL3 prefill transients (geometry, not kv)

C=8192, topk=10, H=2560, I=640, win=32, nslots=81920. Same every full chunk.

| plane | shape / dtype | bytes |
|---|---|---:|
| sorted slot tables | nslots u32 ×3 (order, i32, take) | 0.98 MB |
| window starts+nlives | ~nwin u32 ×2, nwin≈2560 | 0.02 MB |
| pair-prepare yg,yu | 2 × [nslots,H] f16 | 839 MB |
| GEMM gate+up | 2 × [nslots,I] f16 | 210 MB |
| GEMM down + scatter | 2 × [nslots,H] f16 | 839 MB |

No f32 split planes on the prefill GEMM (those are decode). Overlapping defers ~1.5 GB peak, **O(chunk) not O(kv)**. Matches PROFILE_FWD mlp flat at ~2.42 s.

## Allocator picture (after every chunk, after `mlx_clear_cache`)

Cadence: **clear every chunk**. Both packs `cache_bytes=0` after every clear. The pool is not the grower.

### Affine (`diag-64k-aff`, 2025.6 tok/s)

| pos | wall ms | eval ms | active GB | cache | Δ active |
|---:|---:|---:|---:|---:|---:|
| 0–8192 | 4203 | 7 | 74.376 | 0 | — |
| 8192–16384 | 3907 | 6 | 74.376 | 0 | 0 |
| 16384–24576 | 3892 | 6 | 74.376 | 0 | 0 |
| 24576–32768 | 3938 | 8 | 74.376 | 0 | 0 |
| 32768–40960 | 3978 | 6 | 74.376 | 0 | 0 |
| 40960–49152 | 4003 | 5 | 74.376 | 0 | 0 |
| 49152–57344 | 4140 | 4 | 74.376 | 0 | 0 |
| 57344–65536 | 4136 | 5 | 74.376 | 0 | 0 |
| 65536–69940 | 2237 | 1 | 74.240 | 0 | −0.14 |

Wall ~3.9–4.2 s, flat. `/props` after: active 72.95 GB, cache 1.14 GB (decode/idle pool), peak 77.16 GB.

### EXL3 (`diag-64k-exl3`, 1672 tok/s under QSA evals)

| pos | wall ms | eval ms | active GB | cache | Δ active |
|---:|---:|---:|---:|---:|---:|
| 0–8192 | 5007 | 2 | 69.315 | 0 | — |
| 8192–16384 | 4675 | 10 | 71.328 | 0 | **+2.013 GB** |
| 16384–24576 | 4715 | 8 | 73.341 | 0 | **+2.013 GB** |
| 24576–32768 | 4750 | 1 | 75.355 | 0 | **+2.013 GB** |
| 32768–40960 | 4814 | 8 | 77.368 | 0 | **+2.013 GB** |
| 40960–49152 | 4858 | 7 | 79.381 | 0 | **+2.013 GB** |
| 49152–57344 | 4899 | 9 | 81.394 | 0 | **+2.013 GB** |
| 57344–65536 | 4960 | 8 | 83.408 | 0 | **+2.013 GB** |
| 65536–69939 | 3049 | 2 | 84.354 | 0 | +0.946 GB |

Δ is **exactly 1920 MiB per 8192 tokens** (245760 B/token), live `active_bytes`, not the cache pool. Tail 4403 tok × same rate ≈ 0.946 GB. `/props` after: active 83.06 GB, cache 1.14 GB, peak 86.41 GB (from 65.9 GB before the prompt).

eval_ms stays 1–10 ms: the extra 2 GB is **already materialized**, not a growing `mlx_eval` of the KV vector.

## QSA share (12 full-attn layers, `indexer_compress_ratio=4`)

Profile evals flatten wall (known). Numbers still name the kv term. 12 layers/chunk.

| kv after chunk | affine select sum / med | affine gather sum / med | EXL3 select sum / med | EXL3 gather sum / med |
|---:|---:|---:|---:|---:|
| 16384 | 2762 / 228.5 | 707 / 58.8 | 619 / 51.4 | 698 / 57.9 |
| 32768 | 2806 / 232.9 | 744 / 61.8 | 656 / 54.7 | 738 / 61.4 |
| 49152 | 2871 / 238.5 | 776 / 64.7 | 697 / 58.1 | 776 / 64.6 |
| 65536 | 2957 / 244.8 | 822 / 68.5 | 728 / 60.4 | 807 / 66.5 |

EXL3 QSA total 1317 → 1535 ms/chunk (+218 ms). Wall 4675 → 4960 (+285 ms) under profile evals. Uninstrumented climb 5.3 → 7.4 s (+2.1 s) is the same graph **without** per-op eval: lazy QSA over growing `nb=kv/4`.

Dense score sheet `[S=8192, nb]` f32: 67 MB at kv=8k → **537 MB at kv=64k**. New size every chunk.

## Growing term

**Named: EXL3 keeps 1920 MiB of live tensors per 8192-token chunk (`active_bytes`, cache_bytes=0). Affine keeps 0.** That is not MoE (O(chunk) f16 planes, mlp flat). It is not `mlx_clear_cache` cadence (already every chunk; pool empty). Compute slope without profile evals is the **lazy QSA select/gather graph on `nb=kv/4`**, which PROFILE_FWD/QSA evals hide.

## Proposed fix (prefill executor)

1. **Stop the 1920 MiB/chunk retain.** Affine pre-sizes the KV/QSA slab; EXL3 is appending live tensors. Find the handle (full-attn KV, GDN history, QSA pooled keys, or a dequant view) and reuse one preallocated plane. `clear_cache` cannot fix `active_bytes`.
2. **Bound QSA select.** Tile `nb` (already tiles `S`) and reuse one `[S, tile]` f32 score workspace instead of a new `[S, kv/4]` sheet per chunk. Or cap the sheet at `indexer_budget=2048` blocks.
3. Do not change prefill GEMM dtype or window tables for this; they do not grow with kv.
4. Keep the per-chunk mem line (`[prefill-trace] mem ... active_bytes= cache_bytes=`) until the retain is gone.

## Files changed

- `src/generate.zig` — `mlxPrefillMem` / `formatPrefillMemLine` after chunk clear (diagnosis only)
- `EXL3-DECODE-REPORT.md`

## Tests added

- `prefill-trace mem line names active_bytes and cache_bytes`

## Red-first evidence

undeclared `formatPrefillMemLine`

## Suite counts

that test green. Full suite not re-run.

## Commit sha

- integration: `dec5319d`
- measured: `ce0c3f55`

## Live table

Box Apple M5 Max. `[admission] width=8192`. kv affine 8. `[expert-exl3] engaged`. `[exl3-gemm] win=32 aligned=1 nwin=2788 mixed=0 n=81920`. Logs `/tmp/exl3-decode-logs/diag-64k-{aff,exl3}.log` and `*-server.log`.

## Open questions

- Which live tensor is the 1920 MiB/chunk (KV append vs QSA pooled vs dequant view) — needs an allocation name, not more wall clocks.
- Affine QSA select is ~4× EXL3 (228 vs 51 ms) but affine wall stays flat; EXL3’s problem is retain + lazy graph, not select ms under eval.

## Comments

none added

## Ports

none
