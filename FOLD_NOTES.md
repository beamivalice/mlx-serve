# FOLD_NOTES — folding the split-K QSA kernel into longctx-mega-k

Base: `longctx-mega-k` @ de41ffc (= longctx-mega-cand-nk, 75 commits on a93e2c0, kernel-free, suite green).
Dry run performed on scratch branch `scratch-fold-dry` in worktree `.claude/worktrees/mega-k-scratch`
(reset afterwards; nothing kept). Range dry-run: `0af2a49..a24b392` (15 commits) — the OWNER'S FINAL
SHA LIST WILL DIFFER; these notes are about the SHAPE of the conflicts, which is sha-independent.

## Dry-run result

`git cherry-pick 0af2a49..a24b392` onto de41ffc: **14 of 15 commits apply clean, 1 conflict.**

Net delta vs de41ffc after resolution: exactly the kernel range's own delta —
  docs/gotchas/engine-mlx.md  +80
  src/main.zig                 +5
  src/transformer.zig      +1770 / -43
(3 files, 1812 insertions, 43 deletions). No stray drift.

### src/transformer.zig — NO CONFLICT (expected one, got none)

The sheet knob (fccea47, `MLX_SERVE_QSA_SCORE_SHEET_MB`) and the kernel's config caches
(M17/M18 `QsaSelectCfgKey` / `QsaAttnCfgKey` LRU) touch **different regions** of the file and
auto-merge across all 9 kernel commits + 6 audit fixes. Nothing to hand-resolve.
NOTE: auto-merge is not a compile. The scratch tree is being compiled to prove it (see below).

### src/main.zig — NO CONFLICT

L20/L25 (`51f2b7b`) adds the main-thread QSA env resolution call; applies clean.

### docs/gotchas/engine-mlx.md — ONE CONFLICT (commit 61dd860, the split-K story)

Purely ADDITIVE: both sides append a new `##` section at the same insertion point.
HEAD's side = the mega's own sections (`## Speculation never compared itself with the serial
token it replaces (MtpAdaptive)` … `## A spec cache that is not KV-only persists both halves or
neither`); theirs = `## Split-K rescued the QSA sparse-attn kernel …`.

**RESOLUTION: keep BOTH, HEAD's block first, then the kernel's.** Mechanically: strip the three
conflict markers from the file and `git add` it. DO NOT use `git checkout --theirs` — that
silently discards HEAD's ~430 lines of adaptive/spec-cache stories (I did this on the first pass;
the tell is a `-430` line count on engine-mlx.md in `git diff --stat de41ffc HEAD`).

Verification after resolving: `git diff --stat de41ffc HEAD -- docs/gotchas/engine-mlx.md`
must read `+80` and **zero deletions**.

The second docs commit (`46708ad`/`54cfeeb`, "OPEN — split-K passed every synthetic parity bar and
still moved real output") applies clean but is now STALE: the kernel was EXONERATED at 09:55
(main itself is 4/8 on this prompt). On fold, rewrite that section's verdict line to the
exoneration finding rather than leaving an OPEN item in the PR.

## Files the kernel range does NOT touch — authored by hand on FOLD

`git diff --stat 0af2a49 a24b392 -- CLAUDE.md CHANGELOG.md` is EMPTY. Both need new content:

### CLAUDE.md — headroom plan (cap 100,000 B; de41ffc is at 99,273 B ⇒ 727 B free)

Three rules are owed, ~1,000-1,200 B if written as three fresh bullets. Plan:

1. **New bullet (~350 B)** — the split-K kernel + levers:
   `- **A fused sparse-attn kernel over one row's key range STARVES the GPU (12 threadgroups on
     40 cores); the grid must carry the SPLIT** (qwen4 QSA, S=6 −13% / S=3 −8% per verify forward
     at 62k; `MLX_SERVE_QSA_ATTN_KERNEL/_NSPLIT/_MIN_S`, default NSPLIT 16): the merge is the
     delicate half — split-count invariance {1,8,16,64} is the bar, not the kernel body alone.`
2. **Clause onto the existing QSA bullet** (line ~258, `QSA GATHERS selected blocks…`, ~120 B):
   name the fused attn kernel as the arm above the gather and the M12 unification (mask arm and
   gather path now select with the SAME exact tie rule; a prefill tail chunk 2 ≤ S < 16 falls to
   the mask arm, so the two rules were observable as a whole-output fork at S=1).
3. **Clause onto the existing `Kernel testing` bullet** (~150 B) — the 09:55 lesson:
   a greedy first-divergence bar cannot CONVICT without the UPSTREAM reference on the same prompt
   (main 4/8 over 0.15 nats on this prompt; the kernel-free tree's 0/8 was the lucky outlier).

Budget: ~620 B added vs 727 B free ⇒ fits, but with <110 B of slack. To restore margin, trim
~400 B of prose whose detail already lives one hop away (never a symbol, never a rule):
   - line 11 `containers/agent-shell-mlxserve/` row (333 B) → compress to one clause, detail is
     already in that directory's own README + docs/reference.md;
   - line 12 `website/` row (159 B) → the Design/Guards pointers duplicate docs/reference.md.
Full stories go to `docs/gotchas/engine-mlx.md` (already carried by the kernel's own docs commits).
Verify with `wc -c CLAUDE.md` < 100000 before the suite.

### CHANGELOG.md — re-add the split-K entry under the provisional marker

Under the same unreleased/provisional heading the nk branch already uses:
   - split-K sparse-attention kernel for qwen4_exp verify widths: **S=6 −13% / S=3 −8% per verify
     forward at 62k**; default `NSPLIT` 16; levers `MLX_SERVE_QSA_ATTN_KERNEL`,
     `MLX_SERVE_QSA_ATTN_NSPLIT`, `MLX_SERVE_QSA_ATTN_MIN_S`.
   - the exoneration note, stated as a bar and not a boast: on the 62.7k prompt the greedy
     first-divergence tally is **main 4/8 over 0.15 nats vs the kernel 3/8** — the absolute bar is
     the prompt's own near-tie landscape, so the kernel is judged RELATIVE to upstream.

## Post-fold sequence

`bash post_fold.sh` in this worktree (wiped-cache ReleaseFast build, full suite, swift build,
stage root_mega3, --version). Edit-only until the coordinator's GO for anything that boots.

## Exact edit sites located (for FOLD)

CHANGELOG.md (in this worktree):
  - `### Notes` bullet "The fused split-K sparse-attention kernel … is PARKED, not shipped" —
    REPLACE with a `### Highlights` entry (perf numbers + levers) and the exoneration note.
    The provisional marker is the HTML comment under `## v26.9.2 (unreleased)`.

PR_BODY_DRAFT.md (`/Users/beam/claude-tmp/bench-qwen4-ladder/PR_BODY_DRAFT.md`, edit-only, that file only):
  - line 1/3 header: rename branch to `longctx-mega-k`, new sha + commit count, drop
    "with the split-K sparse-attention kernel removed (see Known limits)"; new CLAUDE.md size.
  - `## Summary`: add the split-K bullet next to the fused-select bullet (line ~9).
  - `## Levers and defaults` table (line ~100-105): add `MLX_SERVE_QSA_ATTN_KERNEL` (on),
    `MLX_SERVE_QSA_ATTN_NSPLIT` (16), `MLX_SERVE_QSA_ATTN_MIN_S`.
  - line ~117 "Parked: the split-K sparse-attention kernel" under Known limits: DELETE from
    Known limits; move the ubench table (line ~125, S=6 −13% / S=3 −8%, v1 +15.6/+36.0%) under
    `## Evidence`, and add a **Correctness** paragraph:
      hermetic 2-ulp parity vs the reference's own scale; split-count invariance {1,8,16,64};
      greedy first-divergence table on the 62.7k prompt, 8 nonces, vs each tree's OWN serial —
      main (a93e2c0) 4/8 over 0.15 nats (max .375), kernel off 3/8, NSPLIT=1 2/8, NSPLIT=16 3/8
      (max .375) — the absolute 0.15 bar does not discriminate here because the flips are the
      PROMPT's near-tie landscape, present on upstream without any of this work; the kernel is
      therefore judged RELATIVE to upstream and is no worse.
    Keep the honest caveat that S=1 was never priced (both arms ran the decode gather).

## Dry-run compile result (2026-09-05 10:01)

The folded tree (de41ffc + `0af2a49..a24b392`, conflict resolved as above) BUILDS clean:
ReleaseFast exit 0, `zig-out/bin/mlx-serve --version` = 26.9.1-dev / mlx 0.32.2 / nax on.
So the transformer.zig auto-merge (sheet knob vs the kernel's config caches) is real, not just
textual. Suite NOT run on the scratch (that is post_fold.sh's job on the real fold).

Dry-run commits discarded: `scratch-fold-dry` deleted, `mega-k-scratch` reset to detached de41ffc.
The scratch worktree is kept as a warm compile sandbox; it holds nothing of the fold.

## FOLD has TWO inputs (coordinator, 2026-09-05) — ORDER IS FIXED

1. `nk-fix-reserve` (branch off de41ffc; the #353 reservation over-reserves when `max_tokens` is
   omitted at long context: 374k no-max_tokens -> 503 after prefill). Touches server.zig +
   transformer.zig ESTIMATOR paths. Applies to `longctx-mega-k` FIRST.
2. The kernel owner's FINAL sequence (not a24b392 — shas will differ). Applies SECOND.
3. `post_fold.sh`.

CHANGELOG: add a line for the reserve fix under the provisional marker in `## v26.9.2 (unreleased)`
`### Fixes`, next to the two existing #353 bullets — a long prefill with no `max_tokens` reserved
for an output budget it was never going to use and refused itself after paying for the prefill.

RISK FLAGGED: my dry run proved the kernel range auto-merges `src/transformer.zig` against
**de41ffc**. With the reserve fix landed first, transformer.zig has new estimator edits, so the
kernel range may now conflict there — the dry run does NOT cover that combination. Mitigation:
re-run the dry run (scratch worktree, warm .zig-cache) as soon as `nk-fix-reserve` exists, BEFORE
FOLD, so the fold is never the first time we see that interaction. Branch does not exist yet
(`git rev-parse --verify nk-fix-reserve` fails as of this note).

## Poll note (2026-09-05)

`nk-fix-reserve` was CREATED at de41ffc with 0 commits (branch exists, fix not yet committed).
The existence poll is therefore not a readiness signal; re-armed on CONTENT
(`git rev-list --count de41ffc..nk-fix-reserve > 0`). Dry run runs when that fires.

## DRY RUN 2 — reserve fix FIRST, then the kernel range (2026-09-05)

`nk-fix-reserve` = de41ffc + ONE commit `4140416` ("the reservation buys the PREFILL's
coexistence, not an unbounded generation (#353 follow-up)"). Touches:
.zig-toolchain(+1) CLAUDE.md(+2) docs/gotchas/server-http.md(+67) src/prefix_cache.zig(+18)
src/scheduler.zig(+41) src/server.zig(+202) src/transformer.zig(+80).

Result on scratch branch `scratch-fold2`:
  STEP 1 `git cherry-pick 4140416` onto de41ffc — CLEAN, no conflict.
  STEP 2 `git cherry-pick 0af2a49..a24b392` — 14 clean, the SAME ONE conflict as dry run 1
    (docs/gotchas/engine-mlx.md, additive; strip markers, HEAD's block then the kernel's).
  NO NEW CONFLICT. The kernel range's delta on top of the reserve fix is byte-for-byte the
  delta it had on bare de41ffc: +1812/-43 over engine-mlx.md(+80) main.zig(+5)
  transformer.zig(+1770/-43). The reserve fix's +80 transformer.zig lines are in a different
  region from the kernel's, so the auto-merge holds. 16 commits, 9 files, +2209/-57 total.

### TWO PROBLEMS FOUND — both must be handled at FOLD

**P1 — `.zig-toolchain` is committed as a SYMLINK BLOB (mode 120000) pointing at
`/Users/beam/llm/mlx-serve/.zig-toolchain`.** A machine-local absolute path; it must NOT ship in
a PR. `.gitignore:5` is `/.zig-toolchain/` (trailing slash) so it matches the DIRECTORY, not a
symlink of that name — which is why the worktree-setup symlink got staged. On FOLD: drop it
(`git rm --cached .zig-toolchain` in a fixup, or ask the fix's owner to amend). Also worth
widening the ignore to `/.zig-toolchain` so no future worktree repeats it.

**P2 — CLAUDE.md headroom is now 92 B, not 727.** The reserve fix adds 2 bullets and spends 635 B
net: de41ffc 99,273 -> 99,908 B. My kernel rules need ~620 B, so the trim is no longer optional
and must be roughly TWICE the earlier plan: free ~1,050 B to land the three kernel rules with
real slack. Planned trims (prose only, never a symbol or a rule; detail already one hop away):
  - Detail-map `containers/agent-shell-mlxserve/` row (333 B) -> one clause;
  - Detail-map `website/` row (159 B) -> drop the Design/Guards pointers (in docs/reference.md);
  - if still short, compress the `Stack` paragraph (468 B) and the debugging grep list (450 B),
    both of which duplicate docs/reference.md and the `## Prompt-based skills` section.
Assert < 100,000 B with post_fold.sh's cap check BEFORE the suite.

## Coordinator rulings recorded (2026-09-05) — apply at FOLD

### P1 — `.zig-toolchain`
1. Fixup folded into the reserve commit: `git rm --cached .zig-toolchain` (its owner is also
   being told, so the amend may already be upstream — check before duplicating).
2. SEPARATE one-line commit widening `.gitignore:5` from `/.zig-toolchain/` to `/.zig-toolchain`
   (matches the directory OR a symlink of that name).
3. Scan-pin: **SKIP — the repo has no conventions scan file.** Checked: the "source scan" idiom
   here is per-topic Zig tests reading source TEXT inside the file they guard
   (`src/server.zig:11447`, `src/transformer.zig:41003`); there is no repo-wide tracked-path
   scan and `tests/test_release_workflow_gates.sh` parses release.yml only — no `git ls-files`
   guard exists anywhere in tests/. A tracked-path property does not fit either idiom, so per
   the ruling ("else skip") no scan is added. The widened .gitignore is the guard.

### P2 — CLAUDE.md trim, approved; levers IN ORDER, stop when >= 300 B headroom
  (1) MERGE the reserve fix's TWO new bullets into ONE <= 3-line bullet (it spent 635 B for what
      should be ~300) — keeping BOTH symbol sets: `KVCache.RESERVE_GEN_HEADROOM` 8192, the
      omitted-max_tokens/ctx-prompt clamp, `reclaimableBytes` = residency - largest entry,
      `fitsAfterEviction`, `error.PrefillDoesNotFit` -> named 400 (never the MLX-abandon 503).
  (2) Detail-map rows `containers/agent-shell-mlxserve/` (333 B) and `website/` (159 B).
  (3) The debugging grep list (450 B).
  (4) The `Stack` paragraph (468 B) ONLY if still short.
Constraints: never drop a symbol, lever, threshold or rule; every bullet <= 3 lines;
final < 100,000 B with >= 300 B headroom. Record before/after byte counts here.

BYTE LEDGER (fill at FOLD):
  de41ffc                        99,273 B
  + reserve fix (4140416)        99,908 B   (+635, 2 new bullets)
  after lever 1 (merge)          ______ B
  after lever 2 (detail map)     ______ B
  after lever 3 (grep list)      ______ B
  + 3 kernel rules (~620 B)      ______ B
  FINAL (must be < 99,700)       ______ B

## Dry run 2 compile result (2026-09-05 10:25)

Combined tree (de41ffc + 4140416 + `0af2a49..a24b392`, 16 commits) BUILDS clean: ReleaseFast
exit 0, binary 13,082,648 B, `--version` = 26.9.1-dev / mlx 0.32.2. So both the reserve fix's
and the kernel's `src/transformer.zig` edits coexist at compile level, not just textually.
Suite NOT run (post_fold.sh's job on the real fold).
Scratch reset: `scratch-fold2` deleted, `mega-k-scratch` detached at de41ffc, clean.

## FOLD PART A — EXECUTED (2026-09-05)

Coordinator's sha list was superseded TWICE mid-flight (2ef2571/1bcd484 -> 6fc7a7a/0edafea).
Picked the corrected pair. `.zig-toolchain` is NOT tracked on the amended commits
(`git ls-files | grep -c zig-toolchain` = 0), so P1's `git rm --cached` is MOOT; only the
.gitignore widening shipped.

Commits on longctx-mega-k (de41ffc + 4):
  0ecc190  fix(admission): the reservation buys the PREFILL's coexistence  (= 6fc7a7a)
  4e1806a  test(qwen4): PLE rollback leaves no freed-but-non-null QSA handle (= 0edafea)
  049b4f6  chore: .gitignore /.zig-toolchain/ -> /.zig-toolchain
  591a03e  docs: CLAUDE.md trims + CHANGELOG entry for the reserve fix

### BYTE LEDGER (actual)
  de41ffc                                   99,273 B
  + reserve fix (one 3-line bullet)         99,665 B   (+392)
  - lever 2a containers row -> pointer      99,522 B   (-143; detail MOVED to docs/reference.md +598 B)
  - lever 2b website row -> pointer         99,455 B   (-67)
  - lever 3 post-mortem bullet prose        99,403 B   (-52)
  - lever 4 Stack paragraph prose           99,390 B   (-13)
  CURRENT                                   99,390 B   free-to-cap 610
Lever 1 (merge the reserve bullets) was ALREADY DONE by the fix's owner — it lands as ONE
3-line bullet, so there was nothing to merge.

### SHORTFALL — needs a coordinator decision before PART B
Target was: final < 99,700 B after the kernel's rules, i.e. <= ~99,080 B before them.
Actual 99,390 B — SHORT BY ~310 B. The four approved levers yielded 275 B, not the ~1,050 B
estimated, because that estimate priced the rows by their FULL length while the ruling forbids
dropping a symbol, and these rows are almost entirely symbols (paths, guard filenames, version
pins, grep tokens). What is left in them is connective prose worth tens of bytes, not hundreds.
The one lever that DID pay was RELOCATION (containers detail -> docs/reference.md), which the
growth policy already sanctions and which loses nothing from the repo.
Options put to the coordinator: (a) relocate more detail the same way; (b) accept <300 B
headroom; (c) write the kernel as ONE bullet + one clause instead of three rules.
