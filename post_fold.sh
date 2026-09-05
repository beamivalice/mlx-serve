#!/bin/bash
# post-fold verification for longctx-mega-k. Edit-only: builds, tests, stages a root.
# Boots NOTHING. Run only after the kernel fold is committed.
set -uo pipefail
W=/Users/beam/llm/mlx-serve/.claude/worktrees/mega-k
Z=/Users/beam/llm/mlx-serve/.zig-toolchain/zig
O=/Users/beam/claude-tmp/bench-qwen4-ladder
L=$W/post_fold.log
: > "$L"
say(){ echo "=== $* ===" | tee -a "$L"; }

cd "$W" || exit 1
say "TIP $(git log --oneline -1)  COUNT $(git rev-list --count a93e2c0..HEAD)"
say "CLAUDE.md $(wc -c < CLAUDE.md) B (cap 100000)"
[ "$(wc -c < CLAUDE.md)" -lt 100000 ] || { echo "FAIL: CLAUDE.md over cap" | tee -a "$L"; exit 1; }

say "wiped-cache ReleaseFast build"
rm -rf "$W/.zig-cache"
"$Z" build -Doptimize=ReleaseFast 2>&1 | tee -a "$L" | tail -20
[ -x "$W/zig-out/bin/mlx-serve" ] || { echo "FAIL: no binary" | tee -a "$L"; exit 1; }

say "--version"
"$W/zig-out/bin/mlx-serve" --version 2>&1 | tee -a "$L"

say "full suite (zig build test)"
"$Z" build test 2>&1 | tee -a "$L" | tail -40

say "swift build"
( cd "$W/app" && swift build 2>&1 ) | tee -a "$L" | tail -15

say "stage root_mega3"
mkdir -p "$O/root_mega3/zig-out/bin"
ln -sfn /Users/beam/llm/mlx-serve/lib "$O/root_mega3/lib"
cp "$W/zig-out/bin/mlx-serve" "$O/root_mega3/zig-out/bin/mlx-serve"
ls -la "$O/root_mega3" "$O/root_mega3/zig-out/bin" | tee -a "$L"
"$O/root_mega3/zig-out/bin/mlx-serve" --version 2>&1 | tee -a "$L"

say "DONE — grep '$L' for pass/fail counts"
