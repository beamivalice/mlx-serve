# EXL3-DECODE-REPORT (round 5)

Reset onto integration `63367221` (`git fetch … feat/qwen4-exl3 && git reset --hard FETCH_HEAD`). Dropped `2e000c53` (f32 down inner). Cap-2 + union hist present on pair-f32 `47ef9ece`. Down GEMV inner is f16 (`half(sum)`). Restored local `lib/ds4` (and mlx-src/mlxc-src/opencode2) worktree symlinks emptied by reset; not git-added.

Compile on `63367221`: `.zig-toolchain/zig build -Doptimize=ReleaseFast` succeeded.
`-Dtest-filter="exl3"`: **27/27**. `-Dtest-filter="mtp"`: **125 passed, 1 skipped**.

## (1) Equivalence 11/11 on `63367221`

`MTP_FORCE_ENABLE=1 MTP_TEST_MODEL=<exl3k4 pack> tests/test_mtp_equivalence.sh` → **10 passed, 1 failed**.

PASS: no-mtp baseline, auto-load, chat non-stream/stream, messages (near-tie gap 0.125), acceptance floor avg_per_round=1.00, fused QK-norm, packed prework, enable_mtp:false, EV ext_rounds=28.

FAIL: `[adaptive kill switch]: depth=2 ext_rounds=0 (want depth=3 ext_rounds=0)`. `applyExl3DepthCap` also clamps `MLX_SERVE_MTP_ADAPTIVE=0` (DEFAULT_DEPTH 3) to 2. AUTO path is the intended cap-2; kill-switch contract is leaked. Loadavg at lock 6.36 6.72 4.92.

## (1) AUTO MTP-vs-MTP 3-boot, cap-2 live, no FORCE_DEPTH

`--mtp --kv-quant 8 --prefill-chunk 8192 --no-vision`. Profile off. Box Apple M5 Max. `[admission] width=8192`. `[mtp] adaptive depth cap 2 (m5-max-exl3 row, default 6)`. sha `63367221`.

### persist 0

| boot | arm | short | 4k | loadavg | depth | short avg_per_round | 4k avg_per_round |
|---|---|---:|---:|---|---:|---:|---:|
| r1 | exl3 | 77.885 | 77.851 | 1.61 3.04 3.77 | 2 | 0.92 | 1.46 |
| r1 | aff | 98.111 | 64.720 | 1.81 2.97 3.72 | 6 | 1.27 | 2.05 |
| r2 | exl3 | 78.019 | 77.831 | 1.86 2.85 3.65 | 2 | 0.92 | 1.46 |
| r2 | aff | 97.968 | 85.631 | 2.02 2.85 3.64 | 6 | 1.27 | 2.05 |
| r3 | exl3 | 77.684 | 77.286 | 2.00 2.77 3.58 | 2 | 0.92 | 1.46 |
| r3 | aff | 97.163 | 88.980 | 1.83 2.70 3.54 | 6 | 1.27 | 2.05 |

Medians: EXL3 **77.885 / 77.831**, affine **97.968 / 85.631**. 4k **beats ~70**. Short **77.9 vs ~82** (FORCE_DEPTH=2 was 80.3; AUTO starts `mtp_depth_current=1` and climbs; short is 12 rounds). Affine r1 4k 64.7 is a w3 trial outlier.

### persist 1 (warmup request then timed short+4k)

| boot | short | 4k | loadavg | depth | short avg | 4k avg |
|---|---:|---:|---|---:|---:|---:|
| r1 | 68.471 | 77.285 | 1.72 2.87 3.67 | 2 | 0.92 / 1.00 | 1.29 |
| r2 | 71.310 | 81.601 | 2.08 2.83 3.62 | 2 | 0.92 / 1.00 | 1.67 |
| r3 | 79.746 | 77.875 | 1.81 2.65 3.51 | 2 | 0.92 / 1.09 | 1.37 |

Median **71.310 / 77.875**. 4k holds ~78–82; short noisier (w1 trials on the matured `<2k` table). persist 0 is the stabler cap-2 AUTO cell.

## (2) Quality: finish reduce f32 through H128+svh+score, T(·) at store

Down inner stays f16. REDUCE_SOURCE: `threadgroup float vals`, float score accum, `y = T(a0)` at the end.

TDD: `exl3 down finish reduce keeps f32 through score fold` — red (half vals / `half a0`), then green. `moePrefill matches staged sorted chain` 1–8 ULP vs half staged tokenReduce (envelope). Exl3 filter **28/28**.

KLD vs teacher, kv off, no MTP, `/tmp/exl3-decode-logs/kld-reduce-f32.json` (dirty tree on `63367221` + this change):

| stage | sha | mean to-EOS | top-1 |
|---|---|---|---|
| original | — | 0.0676 | 0.924 |
| f32 pair inner + split-2 | 47ef9ece / 63367221 | 0.06840 | 0.9235 |
| **f32 finish reduce** | **this commit** | **0.06787** | **0.9228** |

Moved: KLD **0.06840 → 0.06787** (−0.00053), +0.00027 vs original 0.0676. top-1 0.9235 → 0.9228 (−0.00073 vs pair, −0.0012 vs 0.924). **Not a null.** Bar 0.0676/0.924 not fully met (KLD close, top-1 still short). No serial cost measured this stage.

## (3) Union kernel

No rows-kernel redesign. Astra study 2 owns the shared-decode plan.

## Files changed

- `src/expert_exl3_kernels.zig` — REDUCE_SOURCE f32 vals + score fold; test; moePrefill 8-ULP envelope
- `EXL3-DECODE-REPORT.md`

## Tests added

- `exl3 down finish reduce keeps f32 through score fold`

## Red-first evidence

- `exl3 down finish reduce keeps f32 through score fold` failed without output (half vals / half a0)
- then green after float vals + float accum

## Suite counts

On `63367221` before this change: exl3 **27/27**, mtp **125 pass / 1 skip**.
After REDUCE_SOURCE: exl3 **28/28**. Full suite not re-run.

## Commit sha

- reset/verify/live AUTO/equiv: `63367221`
- finish-reduce f32: this commit

## Live table (every row with box, chunk, engagement)

Box Apple M5 Max. `[admission] width=8192`. `[expert-exl3] engaged`. kv affine 8. Cap-2 log on EXL3 AUTO boots. See tables above. Extracts `/tmp/exl3-decode-logs/live-auto-*.extract`. Equiv `/tmp/exl3-decode-logs/equiv-11.log`.

## Open questions

- AUTO short 77.9 vs FORCE d2 80.3: climb from depth 1 on a 12-round short. Starting `mtp_depth_current` at 2 would close it.
- 11/11 kill-switch: cap-2 should not bind `MLX_SERVE_MTP_ADAPTIVE=0`.
- top-1 still 0.9228 vs 0.924 after reduce-f32; KLD is the closer of the two.

## Comments

none added

## Ports

none
