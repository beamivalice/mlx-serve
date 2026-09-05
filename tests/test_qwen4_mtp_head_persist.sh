#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp): the in-checkpoint MTP head's committed
# history rides the prefix cache.
#
# The head is NOT KV-only — it owns a QSA index-key history and pooled block
# bank beside its own KV — so before this it was neither committed nor
# restored, and a prefix-cache hit drafted from `qwen4MtpReset`: an EMPTY
# head at a 62.7k-token cursor. Measured (62.7k prose prompt, auto MTP):
# cold prefill m_avg 2.94 / acc 1.59 -> 54.1 tok/s, the SAME prompt as a
# cache hit m_avg 1.00 / acc 0.59 -> 52.2 tok/s, i.e. serial (51.1).
#
# What this asserts is the INVARIANT, never a checkpoint's acceptance:
#   - the second turn is a hot-cache hit AND the head is restored (log line),
#   - the restored-head answer matches the persist-OFF answer tie-aware,
#   - `MLX_SERVE_MTP_HEAD_PERSIST=0` restores the old behaviour exactly (no
#     restore line, still a correct answer).
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_mtp_head_persist.sh [port]
set -u
MODEL="${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit}"
PORT="${1:-11413}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
DIR="$HOME/claude-tmp/qwen4-head-persist"
mkdir -p "$DIR"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
[ -f "$MODEL/ngram_table.bin" ] || { echo "SKIP: pack has no ngram_table.bin"; exit 0; }
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

# Cleanup is armed HERE — before anything can start a server — and is never
# disarmed. An earlier version set the trap inside run_arm AFTER the boot and
# cleared it with `trap - EXIT` at the end of each arm, so any failure outside
# that window (including `set -u` killing the script before the first boot)
# left an engine running and blocked the next executor's port. `SPID` is
# initialised so `set -u` cannot make the handler itself the failure.
SPID=""
stop_srv() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  SPID=""
}
trap stop_srv EXIT INT TERM

# One long prompt with a needle, well past the QSA budget so the head's
# history is the thing under test and not an incidental short window.
body() { python3 -c "
import json,sys
filler=('The archivist catalogued the shelves in the long hall. ')*700
print(json.dumps({'messages':[{'role':'user','content':filler+' The secret code is PELICAN-42. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':24,'temperature':0,'enable_thinking':False,'enable_mtp':True}))"; }

run_arm() { # $1 = arm name, $2 = MLX_SERVE_MTP_HEAD_PERSIST value
  # ONE declaration PER LINE. `local a="$1" b="$DIR/$a.log"` expands every word
  # on the line before any of the assignments take effect, so the third
  # initialiser read `$name` while it was still unset and `set -u` killed the
  # script at the top of the first arm.
  local name="$1"
  local persist="$2"
  local log="$DIR/$name.log"
  local u="http://127.0.0.1:$PORT"
  local b
  MLX_SERVE_MTP_HEAD_PERSIST="$persist" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 \
    --port "$PORT" --log-level info --mtp --prefix-cache-entries 4 > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$log" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$log"; exit 1; }; sleep 2; done
  b=$(body)
  # Turn 1 cold-prefills and COMMITS the head; turn 2 is the cache hit.
  curl -s -m 1800 "$u/v1/chat/completions" -H 'content-type: application/json' -d "$b" >/dev/null
  curl -s -m 1800 "$u/v1/chat/completions" -H 'content-type: application/json' -d "$b" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$DIR/$name.answer"
  # Through the SAME helper the trap uses, so the two can never disagree about
  # what "stopped" means; the trap stays armed for the next arm.
  stop_srv
}

echo "[1] persistence ON: the second turn restores the head"
run_arm on 1
check "hot-cache hit on turn 2" "$(grep -c '\[hot-cache\] reused' "$DIR/on.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "MTP head restored" "$(grep -c '\[qwen4\] MTP head restored' "$DIR/on.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "head restore never declined" "$(grep -c '\[qwen4\] MTP head restore declined' "$DIR/on.log")" "0"
check "needle recovered on the restored turn" "$(grep -c 'PELICAN-42' "$DIR/on.answer")" "1"

echo "[2] MLX_SERVE_MTP_HEAD_PERSIST=0 restores the old behaviour"
run_arm off 0
check "hot-cache hit on turn 2" "$(grep -c '\[hot-cache\] reused' "$DIR/off.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "no head restore line" "$(grep -c '\[qwen4\] MTP head restore' "$DIR/off.log")" "0"
check "needle still recovered (blind head costs acceptance, never a token)" "$(grep -c 'PELICAN-42' "$DIR/off.answer")" "1"

echo "[3] the restored head answers what the blind head answers"
# Greedy verify decides every emitted token on BOTH arms — drafts steer
# acceptance only — so a restored head must not change the answer.
check "restored == blind (first 20 chars)" "$(head -c 20 "$DIR/on.answer")" "$(head -c 20 "$DIR/off.answer")"

echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
