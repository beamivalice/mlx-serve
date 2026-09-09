# Batched decode: make it visible, make it cover qwen3_5 MoE, then spec + batching

## Where we stand

- Concurrent requests are always admitted and decode on the inference thread together. `--max-concurrent` only sizes the submit queue (`cap + 32`) and the prefix-cache floor. Default 1 prints no "continuous batching" line, so users read it as "one request at a time" and start a second instance.
- Batched decode (one forward for N slots, weights read once) engages per slot via `Scheduler.batchable`:
  - dense pure-attention archs (`modelBatchable`);
  - GDN trunks via `supportsBatchedGdnDecode` -> `forwardMoeBatchedDecode`: `qwen3_5` dense and `qwen4_exp`. `qwen3_5_moe` is refused by `ModelConfig.supportsBatchedGdnDecode` (`isMoe() and !isQwen4()`), so Qwen3.5-35B-A3B / Qwen3.8 MoE packs decode serially interleaved: N users share one stream's tok/s.
- Per-slot drop-outs to serial: spec-decode dispatching (MTP/DFlash/PLD), grammar/JSON mode, logprobs, module-head release pending, pad-waste cap (`batchedKvKeepCount`, waste > 1.5x sends the longest slots serial).
- Only tells today: one-shot `[batched] ... engaged (slots=N)` in the log, `[batched] pad-waste cap` lines. Nothing says why a slot went serial, nothing in `/props`, `/v1/models` or metrics.
- Prefill is never batched; a cold prefill yields to decode ticks at chunk boundaries (`interleaveDecodeTick`). Out of scope.

## Step 1: observability

Goal: a user can answer "is my server batching, and if not, why" without reading source.

- [x] 1. `Scheduler.batchable` returns a reason enum instead of bool (`BatchVerdict`: `ok`, `spec_active`, `grammar`, `logprobs`, `head_release_pending`, `embedded_engine`, `arch`, `pad_waste`). The bool callers read `== .ok`. Pure, unit-tested per arm.
- [x] 2. Log once per slot when it first decodes serial beside another live slot: `[batched] slot N serial: <reason>` (debounced per slot, never per tick). The pad-waste line already exists; route it through the same reason.
- [x] 3. `/props` gains `"batching":{"supported":bool,"reason":"<arch|ok>","max_group":N}` from the loaded model's config (`modelBatchable || supportsBatchedGdnDecode`). No-model variant reports `supported:false`.
- [x] 4. `/v1/models` rows: `"batched_decode": bool` at top level, both emitters (same rule as `context_length`).
- [x] 5. `--metrics`: `mlx_serve:batched_group_size` (gauge, last tick's group size) and `mlx_serve:decode_serial_reason_total{reason=...}` counter. Zero cost when off.
- [x] 6. Serve-mode startup line prints for every `--max-concurrent`, including 1: `Concurrency: N in-flight, batched decode <on|off (reason)>`.
- [x] 7. App: Settings shows the batching verdict next to the `maxConcurrent` stepper (read from `/props`), Stepper default left at 1: the flag only sizes the submit queue, so a bigger default changes nothing (explainer rewritten to say so).

Tests:
- unit: `BatchVerdict` per arm on synthetic slots (`std.testing`, no MLX).
- `tests/test_models_capabilities.sh`: `batched_decode` present on both emitters, true on a dense pack, false on a GGUF.
- `tests/test_batched_equivalence.sh`: assert the new serial-reason line appears when a PLD-armed slot joins, absent for two plain slots.

## Step 2: batched decode for qwen3_5_moe

Status: DONE. Live bars run 2026-09-08: `test_batched_equivalence.sh` 5/5 and `test_concurrent_throughput.sh` 1.62x on Qwen3.6-35B-A3B.

Goal: Qwen3.5 / Qwen3.8 MoE packs batch like qwen4_exp does.

Why it should be small: `forwardMoeBatchedDecode` is already batch-generic and qwen4_exp already runs routed experts through it. The only per-slot state on `qwen3_5_moe` is the GDN pair, which the path already merges/splits. The exclusion is a predicate, not a missing kernel.

- [x] 1. Red: extend the `supportsBatchedGdnDecode` config test with a `qwen3_5_moe` config expecting true; extend the transformer-side predicate test (the `.moe => if (self.qwen4 == null) return false` arm) with a fixture that has `moe_layers` in `.moe` shape and no qwen4 module.
- [x] 2. Green: `ModelConfig.supportsBatchedGdnDecode` drops the `!isQwen4()` conjunct for the qwen3_5 family (keep refusing hy3/laguna/lfm2_moe/bailing, which have other state); transformer predicate accepts `.moe` layers whenever the trunk is a plain qwen3_5 MoE (shared expert included). Audit `forwardMoeBatchedDecode` for anything that reads `self.qwen4` unconditionally on the MoE arm (PLE window, QSA keys) and gate it on the module.
- [x] 3. Check the MoE decode kernel at N rows: `useGatherQmvDecode` gates on `batch*seq == 1`; at N>1 the routed experts must take the sorted `gather_qmm` path (the same one verify rows take). Confirm no per-row loop lands (CLAUDE.md: a per-row `gatherQmv` loop at S>1 lost).
- [x] 4. Spec interplay: MTP default is OFF on MoE, PLD may arm from the prompt score. `slotTicksRegular` already lets a runtime-disabled PLD slot batch. No change, but the equivalence script must run one arm with PLD armed to show the serial-reason line from step 1.
- [x] 5. Server-side clamp in `serve()` reads the shared predicate already, so it follows.

Tests / bar:
- `tests/test_batched_equivalence.sh` on a qwen3_5_moe pack (Qwen3.5-35B-A3B or the Qwen3.8 MoE 4-bit pack from the external drive): `MLX_SERVE_FORCE_BATCHED=1` N=1 byte-identical to serial greedy; real two-stream arm logs `[batched] gdn batched decode engaged (slots=2)` and acquits near-ties at <= 0.15 nats (same bar as the qwen3_5 dense run).
- `tests/test_concurrent_throughput.sh` on the same pack: aggregate tok/s at N=2 and N=4 vs N=1 serial, ReleaseFast, same boot. Record in `benchmarks.md` as a new row, not a new column.
- Prefix-cache restore + batched decode on a GDN MoE: two slots restored from different entries, both correct (the batched tick advances `moe_seq_offset`; the rope-at-0 bug was fixed on qwen3_5 dense, the MoE arm inherits it but the test must run).

Risks:
- Memory: N x expert activations plus the padded KV transient (`MAX_PAD_WASTE`). The pad-waste cap already bounds the KV side; the expert side is `top_k` x hidden per row and small at N <= 8.
- qwen4_exp-specific code inside the batched forward that silently assumes a `qwen4` module. Grep every `self.qwen4.?` on that path before flipping the predicate.

## Step 4: spec decode inside a batched group — DONE (2026-09-08/09)

Measured gate first (`tests/bench_batched_spec.sh`): on the dense 27B a batched verify pays (verify rows near-free until the 8-row kernel cliff); on the MoEs a verify row is expert bytes, so Flash Next and the 35B-A3B gain nothing from batching the verify itself.

- [x] `nextMtp` split into `mtpRoundBegin` / `mtpRoundVerify` / `mtpRoundFinish` (`MtpRoundState`); finish handles padded rows (`verify_len > 1 + m`).
- [x] `forwardMoeBatchedVerify`: `[N, S]` rows, per-slot causal mask (`buildBatchedDecodeMask(…, seq_len)`), per-row SSM capture split (`splitSsmToSlots`).
- [x] Scheduler group tick (`runBatchedMtpTick`), 7-row budget off-NAX (`mtpGroupRowCap`, `mtpSubGroupSize`, `Generator.mtp_group_cap`), crowd arm (`mtp_plain_tick` with hidden capture) at 4+ slots.
- [x] Flash Next: per-request head state (`Qwen4MtpState`, `qwen4MtpActivate`) so MTP slots are no longer exclusive; rounds stay solo (interleave at 2, plain at 3+).
- [x] Bars: `tests/test_mtp_batched.sh` on both packs; `tests/test_mtp_equivalence.sh` 11/11 on the 27B; `tests/test_qwen4_exp.sh` 40/40; unit suite green.
- [ ] Follow-up: qwen4 batched verify needs PLE spec capture + QSA mask at S > 1 in the batched forward (ceiling ~+25% at N=2 by the row-cost curve).
- [ ] Follow-up: the same 8-row cliff caps PLAIN batching at N=8 off-NAX (27B: 81 tok/s at N=6, 66 at N=8); a row-chunked split-K fallback or an M 8..16 lane is the lever.

Numbers: `~/claude-tmp/batch-0908/ladder.csv`, chart `concurrency_chart.html`.

### Original sketch

Today a slot that dispatches MTP/DFlash/PLD leaves the group and decodes serial, so on qwen4_exp with `--mtp` the second user costs the first their speculation. Two options, pick after measuring:

- Option A, cheap: keep spec slots serial but stop starving the group. Alternate ticks (one batched tick for the plain group, one spec round for the exclusive slot) is already what the scheduler does; the win would only be scheduling fairness. Probably not worth code.
- Option B, real: a batched VERIFY forward. Every slot in the group contributes `1 + draft_len` rows; the forward becomes `[N, S]` with per-slot rope offsets and per-slot KV, which the batched path already models per row for S=1. Needs: draft rows padded to the group's widest draft (mask the rest), per-slot accept/rollback on the shared GDN state (`ssmRollbackFromCapture` at the verify width, per row), and the round-cost table gaining a group dimension. MTP head for qwen4 is module-owned (one cache per model), so B is DFlash/PLD only on qwen4_exp until the head gets per-slot caches.

Gate on measurement first: on qwen4_exp a verify row is bytes (S=1/2/4 = 16/22/31 ms), so a batched verify at N=2, S=2 is expected near 2 serial forwards. If the N=2 batched plain tick already lands under 1.3x a serial tick, option B's ceiling is small and the answer is to keep spec off past N=1 (document it in step 1's reason line as `spec_active`).
