#!/bin/bash
# An MLX error costs ONE REQUEST, never the server (issue #353).
#
# Before the mlx-c error handler was replaced, a Metal working-set OOM went
# `mlx_error(...)` -> `mlx_error_handler_default_` -> exit(-1): no status line,
# no connection close, every in-flight request gone with the process. The
# invariant now is the pair — the failing request answers with a NAMED memory
# error, and the NEXT request on the same server succeeds.
#
# `MLX_SERVE_MLX_FAULT_CHUNK=<n>` latches a synthetic Metal OOM at the n-th
# `mlx.checkError` of the process (the prefill chunk loop's checkpoint) and
# disarms itself, so one boot exercises both halves.
#
# Usage: ./tests/test_mlx_error_recovery.sh [model_dir] [port]
set -u
MODEL=${1:-"$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"}
PORT=${2:-8151}
BASE="http://127.0.0.1:$PORT"
PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

[ -d "$MODEL" ] || { echo "SKIP: model not found at $MODEL"; exit 0; }
[ -x ./zig-out/bin/mlx-serve ] || { echo "FAIL: build with -Doptimize=ReleaseFast first"; exit 1; }

LOG=$(mktemp -t mlxerr).log
MLX_SERVE_MLX_FAULT_CHUNK=1 ./zig-out/bin/mlx-serve serve --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done

req() { # $1 = prompt
  curl -s -o /tmp/mlxerr_body.json -w '%{http_code}' -m 300 \
    -H 'content-type: application/json' \
    -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":false}" \
    "$BASE/v1/chat/completions"
}

echo "[1] the request that hits the injected MLX error is refused, by name"
CODE=$(req "Count to three.")
# 503 is the memory class; the shape that must NEVER appear is an empty reply
# from a dead socket, so a body is as load-bearing as the code.
case "$CODE" in
  503) ok "injected MLX OOM answered 503 ($(head -c 120 /tmp/mlxerr_body.json))" ;;
  500) ok "injected MLX error answered 500 (non-memory class)" ;;
  000|"") bad "no HTTP response — the server died (the #353 symptom)" ;;
  *)   bad "unexpected status $CODE: $(head -c 200 /tmp/mlxerr_body.json)" ;;
esac
grep -q "FAULT INJECTION armed" "$LOG" || bad "injector never armed — the env hook moved"
grep -q "\[mlx\] " "$LOG" || bad "no [mlx] line logged for the latched error"

echo "[2] the server is still serving: the NEXT request succeeds"
kill -0 $SRV 2>/dev/null || bad "server process is gone"
CODE2=$(req "Say hello.")
if [ "$CODE2" = "200" ] && grep -q '"content"' /tmp/mlxerr_body.json; then
  ok "second request answered 200 with content"
else
  bad "second request status $CODE2: $(head -c 200 /tmp/mlxerr_body.json)"
fi

echo "[3] the fault is ONE-SHOT: a third request is unaffected"
CODE3=$(req "Say goodbye.")
[ "$CODE3" = "200" ] && ok "third request answered 200" || bad "third request status $CODE3"

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "---- $PASS passed, $FAIL failed ----"
[ "$FAIL" -eq 0 ]
