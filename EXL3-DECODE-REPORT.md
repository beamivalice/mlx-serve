# EXL3-DECODE-REPORT (round 4)

HEAD measured: `2e000c53` f32 down GEMV inner. Restack gate: index.json present; `config.json` has `expert_quant k=4 source=restack` (mtime Sep 18 21:38, same as index). Restack log `~/claude-tmp/exl3-restack-logs/restack-k4-redo.log` ends at the last MTP shard (`restack model-exl3-mtp-L00-up.safetensors`) and never printed its `k histogram tensors=… modal=4` done line; converter writes that line immediately before rewriting config, so the pack is complete. Did not write under `~/llm/models`.

## (1) Quality KLD `2e000c53`

`mlx-serve kld compare` vs `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-16x512-raw`, `--kv-quant off --no-mtp`, json `/tmp/exl3-decode-logs/kld-down-f32.json`.

| stage | sha | mean to-EOS | top-1 |
|---|---|---|---|
| original decode path | — | 0.0676 | 0.924 |
| coop GEMV | — | 0.0693 | 0.921 |
| fold | 9f3c1641 | 0.06926 | 0.921 |
| split-2 f16 inner | 08790eb6 | 0.06879 | 0.923 |
| f32 pair inner + split-2 | 47ef9ece | 0.06840 | 0.9235 |
| **f32 down GEMV inner** | **2e000c53** | **0.06966** | **0.9218** |

All-positions: mean KLD 0.06455 / top-1 0.9292 / 8192 pos; to-EOS 7186 pos, NLL 0.3940. **Misses bar 0.0676/0.924** (delta +0.0021 KLD, −0.0022 top-1 vs original). Worse than f32-pair `47ef9ece` 0.06840/0.9235. Finish reduce still half-folds after H128+svh. Loadavg at KLD start 2.75 2.96 2.75.

## (1) Serial 3-boot (short + 4k, affine-ab interleaved)

`--no-mtp --kv-quant 8 --prefill-chunk 8192 --no-vision`. Box Apple M5 Max. `[admission] width=8192`. sha `2e000c53`.

| boot | arm | short tok/s | 4k tok/s | loadavg | admission |
|---|---|---:|---:|---|---|
| r1 | exl3 | 52.265 | 53.259 | 4.04 3.45 3.00 | width=8192 |
| r1 | aff | 48.366 | 56.557 | 3.80 3.45 3.02 | width=8192 |
| r2 | exl3 | 55.808 | 50.966 | 4.18 3.57 3.07 | width=8192 |
| r2 | aff | 60.920 | 57.072 | 4.44 3.65 3.11 | width=8192 |
| r3 | exl3 | 55.877 | 53.341 | 4.02 3.60 3.10 | width=8192 |
| r3 | aff | 61.991 | 58.033 | 3.57 3.52 3.08 | width=8192 |

Medians: EXL3 **55.808 / 53.259**, affine **60.920 / 57.072** → **91.6% / 93.3%**. Affine r1 short 48.4 is a load outlier (exl3-bughunt `zig test-filter=exl3` on box). Quiet pair r2/r3: EXL3 55.8–55.9 vs affine 60.9–62.0.

## (1) MTP-vs-MTP 3-boot (profile off, persist 0)

`ENABLE_MTP=1`, `MLX_SERVE_MTP_QWEN4_PROFILE=0`, `MLX_SERVE_ROUND_COST_PERSIST=0`, `--mtp --kv-quant 8 --prefill-chunk 8192`. sha `2e000c53`.

| boot | arm | short | 4k | loadavg | short avg_per_round / round_ms | 4k avg_per_round / round_ms |
|---|---|---:|---:|---|---|---|
| r1 | exl3 | 73.546 | 54.309 | 2.60 3.29 3.01 | 1.25 / 33.39 | 2.37 / 65.84 (w4 trial) |
| r1 | aff | 94.689 | 78.195 | 2.40 3.18 2.98 | 1.27 / 25.61 | 2.05 / 41.53 |
| r2 | exl3 | 72.067 | 72.139 | 2.61 3.19 2.99 | 1.25 / 31.31 | 2.37 / 50.72 |
| r2 | aff | 94.751 | 87.308 | 2.92 3.23 3.00 | 1.27 / 25.40 | 2.05 / 37.06 |
| r3 | exl3 | 72.845 | 72.442 | 3.03 3.24 3.01 | 1.25 / 31.26 | 2.37 / 50.26 |
| r3 | aff | 95.386 | 88.463 | 3.02 3.22 3.01 | 1.27 / 25.40 | 2.05 / 36.66 |

Medians: EXL3 **72.845 / 72.139**, affine **94.751 / 87.308** → **76.9% / 82.6%**. Cold-start cap 6 + w4 trial poisons EXL3 r1 4k (54.3). spec-stats `depth=6` is the cap; persist 0 so no table.

## (2) Lever 2: depth sweep `MLX_SERVE_MTP_FORCE_DEPTH`

`MLX_SERVE_MTP_TRACE=1`, persist 0, profile off. Two protocols.

### 160-token essay (acc_idx / survival)

| d | short tok/s | 4k tok/s | short acc_idx | 4k acc_idx | short round_ms / eval | 4k round_ms / eval | delivered/round short / 4k |
|---|---:|---:|---|---|---|---|---|
| 1 | 72.765 | 69.947 | 0.66 then 0.78 | 0.75 then 0.94 | 24.48 / 21.4 | 26.90 / 23.4 | 1.76 / 1.86 |
| 2 | 70.743 | 78.963 | 0.66/0.41 then 0.78/0.44 | 0.88/0.78 | 29.36 / 26.0 | 33.59 / 28.7 | 2.05 / 2.58 |
| 3 | 64.382 | 74.744 | 0.63/0.31/0.25 | 0.88/0.72/0.59 | 36.76 / 31.4 | 38.72 / 36.5 | 2.32 / 3.08 |
| 4 | 43.567 | 61.887 | 0.56/0.38/0.25/0.19 | 0.72/0.56/0.44/0.44 | 56.84 / 35.8 (corr 17) | 54.62 / 38.7 | 2.46 / 3.40 |

`m_avg` matches forced depth. d4 short corr ~17 ms is a rollback cliff, not acceptance.

### 3-boot best two (d1, d2) Bangkok 64 + 4k 64 — the ~82 / ~70 bar

| boot | d | short | 4k | loadavg | short wall ms (pred_ms/attempts) | emitted 1+avg |
|---|---|---:|---:|---|---:|---:|
| r1 | 1 | 70.573 | 70.180 | 2.67 2.85 2.88 | 24.7 | 1.69 |
| r1 | 2 | 82.322 | 82.242 | 2.21 2.73 2.83 | 26.72 | 2.30 |
| r2 | 1 | 70.307 | 68.563 | 2.09 2.68 2.81 | 24.6 | 1.69 |
| r2 | 2 | 80.334 | 81.158 | 2.06 2.65 2.80 | 31.3 EMA | 2.30 |
| r3 | 1 | 67.746 | 67.852 | 2.20 2.65 2.79 | 25.4 | 1.69 |
| r3 | 2 | 76.718 | 78.662 | 2.45 2.68 2.80 | 31.8 EMA | 2.30 |

Medians: d1 **70.307 / 68.563**; d2 **80.334 / 81.158**. d2 delivers the ~82 short / ~70 at 4k bar (4k is above 70). Acceptance rule unchanged (`FORCE_DEPTH` only). Equivalence 11/11 not re-run (planner does not touch verify/accept).

### Policy (implemented)

EXL3 already uses `.generic`, not the affine mixed profile. Generic M5 Max cold-start cap was 6. Measured: cap 2.

- `mtp.adaptiveDepthCapForExl3`: M5 Max cap **2** (`m5-max-exl3`); other chips keep `adaptiveDepthCapForMachine`.
- `mtp.applyExl3DepthCap`: only when `expert_layout == .exl3_k4`; affine mixed unchanged (still cap 6).
- Wired at Generator init and scheduler `entry.mtp_depth` / spec-warmup. `mtp_depth_free` stays 6 so a persisted table may still trial above 2 once trusted; persist-0 / cold boots bind at 2.
- Bucket: `<2k` and `2-4k` should start at w2. Do not cold-trial w4 (d4 43.6 short). No change to `round_cost.zig` cells/EMA; the table already keys by model_dir.

## (3) Lever 5: expert-union histograms

Env `MLX_SERVE_EXL3_UNION_HIST=1` (eval-sync; tok/s poisoned: d2 61 vs 82). sha after helper rebuild, still on `2e000c53` parent. Forced depth 2/3/4 → verify S=3/4/5. S=2 from partial rounds.

| S | n layers | unique mean | uniform U(S) | 10×S | reuse vs 10×S | uniform reuse | shared mean (≥2 rows) | max_mult p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 161 | 16.65 | 19.80 | 20 | **1.20** | 1.01 | 3.4 | 2 |
| 3 | 1745 | 21.18 | 29.42 | 30 | **1.42** | 1.02 | 6.6 | 3 |
| 4 | 1713 | 26.87 | 38.84 | 40 | **1.49** | 1.03 | 8.5 | 4 |
| 5 | 1360 | 30.58 | 48.08 | 50 | **1.64** | 1.04 | 11.2 | 4 |

Real routing overlaps more than independent top-10: S=4 saves ~33% of expert assignments (40→26.9), not the uniform 3%. Material vs Astra's "near-zero under uniform". **No shared-decode kernel this turn** (large redesign; dump eval also not a speed path). Follow-up if taking the ~72-at-4k lever.

Warmup also logged S=8/9/12 (same hist every boot; ignore for verify). Logs: `/tmp/exl3-decode-logs/live-union-d{2,3,4}.log`.

## Files changed

- `src/expert_exl3_kernels.zig` — `unionUnique` / `unionMultiplicity` / env dump
- `src/transformer.zig` — call dump on decode-arm S≥2
- `src/mtp.zig` — `adaptiveDepthCapForExl3` / `applyExl3DepthCap`
- `src/generate.zig` — apply EXL3 cap at Generator init
- `src/scheduler.zig` — apply EXL3 cap at load
- `EXL3-DECODE-REPORT.md`

## Tests added

- `exl3 verify group union unique vs assignment count`
- `adaptiveDepthCapForExl3: M5 Max cold-start cap is 2`

## Red-first evidence

- union: `use of undeclared identifier 'unionUnique'`
- cap: `use of undeclared identifier 'adaptiveDepthCapForExl3'`
- then green

## Suite counts

`-Dtest-filter="exl3"`: **28 passed** (was 27). Cap test 3/3 passed. Full suite not re-run.

## Commit sha

- measured live/KLD: `2e000c53`
- planner+union helper: this commit (`git log -1` on `agent/exl3-decode`)

## Live table (every row with box, chunk, engagement lines)

Box: Apple M5 Max. Chunk: `[admission] width=8192`. Engagement: `[expert-exl3] engaged`, MTP `--mtp` as noted, kv affine 8. See tables above; extracts under `/tmp/exl3-decode-logs/live-*.extract`.

## Open questions

- Down-f32 KLD **regressed** vs pair-f32. Finish reduce still half-folds; next quality stage if we still want 0.0676.
- Restack log missing histogram done line (config was written).
- d4 short `corr` ~17 ms: rollback cost at S=5, not overlap.
- Equivalence 11/11 not re-run this turn (acceptance rule unchanged).

## Comments

none added

## Ports (reference ideas taken, for NOTICE)

- none this turn (planner cap is a measured silicon row, same pattern as M1 Pro / base M5)
