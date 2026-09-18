# EXL3-DECODE-REPORT (round 7)

Reset onto `2f876392` (`feat/qwen4-exl3`: cap-scope `a9598bd0` + B13 `00ef132a` + prefill config reuse). Shared-decode (`6e523afb`, `62520f81`) and half-product acc (`f84a794d`) dropped with the reset. Start-at-2 never on this head. Restored `lib/ds4`, `lib/mlx-src`, `lib/mlxc-src`, `lib/opencode2-mlx-serve` worktree symlinks; not git-added.

Quality chain stays at f32 finish reduce (bar 0.0679 / 0.923). 0.0676 bisect closed as exhausted.

Leases: AUTO one 20m lock, unlock; forced-d2 one 20m lock, unlock. Not a combined lease.

## Shared-decode null (deleted)

Whole-forward, shared ON, handbook 512-decode, round 6 (contaminated for merging, kept as the null):

| ctx | serial EXL3 | shared-ON MTP EXL3 | delta |
|---|---:|---:|---:|
| 4k | 53.682 | 45.054 | −8.6 tok/s (−16%) |
| 16k | 54.353 | 48.016 | −6.3 tok/s (−12%) |
| 64k | 52.521 | 49.651 | −2.9 tok/s (−5%) |

Live MTP widths on EXL3 AUTO/d2 are S≈2 (cap 2 / force 2). That is a loss vs per-row serial. S=3..5 were not a winning in-situ arm (no live EXL3 path drafts that wide under cap-2). Kernel, descriptor, buckets, and tests deleted with the reset. No width kept behind a predicate.

## Half-product float acc null (deleted)

KLD to-EOS **0.068517 / top-1 0.9222** vs bar 0.0676 / 0.924 (worse than f32 finish 0.06787 / 0.9228). Dropped.

## New-standard MTP on clean `2f876392`

Handbook prompts `/Users/beam/claude-tmp/exl3-bench/prompt-{4k,16k,64k}.txt`, nonce prefix, max_tokens 512. Every row **completion_tokens=512**, **finish_reason=length**. kv8, width=8192, persist 0, no `MLX_SERVE_NGRAM_WARM=0`. Affine first, interleaved, 3 boots, medians. Box Apple M5 Max.

`spec-stats depth=` is the **cap** (`mtp_depth`), not the round width. AUTO EXL3 prints depth=2 (cap-2). FORCE_DEPTH=2 unclamps the cap (prints 6) but every round drafts exactly 2 (`accepts / (attempts * 2)` matches `per_draft_pct`).

### AUTO (cap 2, persist 0) — one lease

| arm | boot | loadavg | 4k tok/s avg/ms | 16k tok/s avg/ms | 64k tok/s avg/ms |
|---|---|---|---|---|---|
| aff | r1 | 2.11 2.00 1.86 | 76.192 1.23/33.54 | 74.470 1.28/29.40 | 70.167 1.12/27.83 |
| exl3 | r1 | 2.78 2.21 1.95 | 60.327 0.82/37.70 | 58.204 1.43/36.48 | 60.660 1.12/35.57 |
| aff | r2 | 2.20 2.23 1.99 | 72.474 1.35/34.53 | 74.969 1.42/29.38 | 76.721 1.37/30.63 |
| exl3 | r2 | 1.66 2.15 1.99 | 61.754 0.94/28.73 | 63.202 0.90/34.30 | 66.498 1.20/31.87 |
| aff | r3 | 1.95 2.33 2.09 | 72.783 1.17/34.59 | 75.264 1.59/37.74 | 78.205 1.67/35.63 |
| exl3 | r3 | 2.35 2.47 2.17 | 61.801 1.23/38.85 | 66.629 1.27/35.96 | 63.674 0.85/27.58 |

Medians: affine decode **72.783 / 74.969 / 76.721**, avg **1.23 / 1.42 / 1.37**, round_ms **34.53 / 29.40 / 30.63**, cap 6. EXL3 decode **61.754 / 63.202 / 63.674**, avg **0.94 / 1.27 / 1.12**, round_ms **37.70 / 35.96 / 31.87**, cap 2. Decode ratio **84.8% / 84.3% / 83.0%**. `[mtp] adaptive depth cap 2`. `[expert-exl3] engaged`.

### Forced d2 — one lease

Both packs `MLX_SERVE_MTP_FORCE_DEPTH=2` (every round drafts 2).

| arm | boot | loadavg | 4k tok/s avg/ms | 16k tok/s avg/ms | 64k tok/s avg/ms |
|---|---|---|---|---|---|
| aff | r1 | 1.84 2.18 2.09 | 76.013 1.13/28.05 | 82.591 1.26/27.55 | 78.928 1.30/29.66 |
| exl3 | r1 | 2.82 2.39 2.18 | 59.529 1.19/33.97 | 69.794 1.28/32.42 | 65.931 1.21/33.44 |
| aff | r2 | 2.83 2.55 2.27 | 76.898 1.19/28.77 | 79.181 1.18/27.97 | 81.605 1.29/28.29 |
| exl3 | r2 | 2.54 2.57 2.30 | 66.271 1.17/34.05 | 69.639 1.24/32.85 | 67.962 1.27/33.54 |
| aff | r3 | 2.21 2.40 2.26 | 76.691 1.12/28.02 | 83.144 1.24/27.12 | 83.242 1.34/28.39 |
| exl3 | r3 | 1.87 2.27 2.22 | 65.995 1.16/33.69 | 71.334 1.27/32.43 | 67.710 1.21/33.32 |

Medians: affine decode **76.691 / 82.591 / 81.605**, avg **1.13 / 1.24 / 1.30**, round_ms **28.05 / 27.55 / 28.39**. EXL3 decode **65.995 / 69.794 / 67.710**, avg **1.17 / 1.27 / 1.21**, round_ms **33.97 / 32.43 / 33.44**. Decode ratio **86.1% / 84.5% / 83.0%**.

## EXL3/affine MTP ratio and cause

Handbook affine MTP is **73–77 AUTO / 77–83 forced-d2**, not the 95–98 short-prose band. Acceptance on this prompt is low: affine avg_per_round **1.13–1.42** (handbook) vs ~2+ on short prose.

At **matched force-d2**, acceptance is the same (EXL3 1.17/1.27/1.21 vs affine 1.13/1.24/1.30). EXL3 **round_ms is 18–21% higher** (34.0/32.4/33.4 vs 28.1/27.6/28.4). The MTP ratio **83–86%** is verify-row cost, not acceptance.

AUTO ratio is the same **83–85%**: affine’s extra cap-6 width does not pay on this prompt (affine forced-d2 is *faster* than affine AUTO). EXL3 cap-2 is not the limiter here.

vs serial EXL3 53.7/54.4/52.5, clean AUTO 61.8/63.2/63.7 is still a real MTP gain.

## Files changed

- `EXL3-DECODE-REPORT.md` only (reset dropped shared-decode, half-product acc, start-at-2)

## Tests added

none this turn

## Red-first evidence

n/a (reset; no new kernel)

## Suite counts

not re-run this turn

## Commit sha

- integration reset: `2f876392`
- cap-scope (on that head): `a9598bd0`
- B13 (on that head): `00ef132a`
- tables measured on `2f876392`

## Live table

Box Apple M5 Max. `[admission] width=8192`. kv affine 8. `[expert-exl3] engaged`. AUTO: `[mtp] adaptive depth cap 2`. Logs `/tmp/exl3-decode-logs/clean-{auto,d2}.log`. JSON `/tmp/exl3-decode-logs/live-clean-{auto,d2}-*-std.json`.

## Open questions

- Shared-decode is a journal null; no width kept.
- Half-product acc is a journal null; quality stays f32 finish reduce.
- Handbook MTP is acceptance-limited for affine; EXL3 gap at matched d2 is round_ms.

## Comments

none added

## Ports

none
