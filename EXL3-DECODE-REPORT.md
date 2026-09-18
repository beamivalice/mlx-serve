# EXL3-DECODE-REPORT — chunk-boundary graph retention

Reset onto `b125e3c2` (NROWS template no longer keys the sorted GEMM). Restored `lib/ds4`, `lib/mlx-src`, `lib/mlxc-src`, `lib/opencode2-mlx-serve` worktree-only symlinks; not git-added. One GPU lease.

## Files changed

- `src/transformer.zig` — `capturePrefillHidden` / `capturePrefillHiddenLast` (`materializedOwnedCopy` + eval); `evalCadencePoint` names `ssm_state` as well as `conv_state`; qwen4 cadence flushes a pending hc write before the named eval; two-chunk retention test
- `src/generate.zig` — chunk-boundary eval vector names `chunk_hidden_all`
- `EXL3-DECODE-REPORT.md`

No per-layer `moePrefill` eval.

## Tests added

- `two-chunk prefill capture drops the chunk parent after the boundary eval`

A large parent `[1,2048,2048] f16` is viewed as an 8-row slice (the capture/checkpoint class). Two chunks keep those captures. After the dummy cadence eval (h only), live bytes must fall under one parent. Red when capture was `mlx_array_set` of the slice; green after the owned copy.

## Red-first evidence

```
two-chunk prefill capture drops the chunk parent after the boundary eval...FAIL (TestUnexpectedResult)
```

then green after `capturePrefillHidden`.

## Suite counts

```
.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="exl3" && ./zig-out/tests/test
65/65 passed.
```

Two-chunk test green (separate filter).

## Commit sha

(filled at commit)

## Live table

Box Apple M5 Max. Pack `Qwen3.8-Flash-Next-MLX-Serve-exl3k4-8bit`. kv8, `--no-mtp --no-vision`, width=8192, handbook prompts, nonce, `max_tokens=16`. `[expert-exl3] engaged`. Binary `4b2a7a9c3e826b91a8c96322e0e399a98ea7f67137597da6bbe7756291f4c9bc`. Logs `/tmp/exl3-decode-logs/leakfix-64k.log`. MTP `loaded=false` on this boot (so `chunk_hidden_all` is empty).

Prefill tok/s: **4k 1545.1 / 16k 1601.7 / 64k 1269.0**. 4k is in line with the 1573 bar (prior extra-eval boot 1529).

64k active_bytes still **+2013265920 per 8192-token chunk** (`cache_bytes=0`), ms/chunk 5.0→8.3 s after kv≈32k. The `--no-mtp` leak is not the MTP hidden capture. The two-chunk guard still holds for the view-capture class.

## Open questions

- `--no-mtp` 64k still accumulates 1920 MiB/chunk after naming `ssm_state` in cadence, flushing hc pending before cadence, and owned-copying captures. Next handle is whatever remains live after that named vector (not `moePrefill` after free; that hermetic was green).
- Re-take 64k with MTP on to confirm `chunk_hidden_all` no longer pins a parent.

## Comments

none added

## Ports

none
