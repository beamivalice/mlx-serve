# EXL3 Prefill Report

Run: `exl3-prefill`
Head: `2f876392` (integration, config-reuse already merged)
Box: Apple M5 Max
Six-number binary sha256: `3f090c2ff652b0a14d696c0f59a910fc7c5cd9c6dd404a5d1f7eced83c767f42`

## Files changed

none this turn (bisect + re-measure only; no decode revert)

## Tests added

none (no code fix: 5815028d and 43cf388e are not the 17% drop)

## Red-first evidence

n/a — no failing identity to land a prefill-only dtype/bank split. See live table.

## Suite counts

not re-run this turn (no source change)

## Commit sha

- Head `2f87639291ca4f65170fdc7068e1bc221359e7ed`
- Bisect control patch was uncommitted on `449e3e24` (config-reuse only)

## Live table

Method for bisect: one boot, warm sentence, 3× `prompt-4k`, `max_tokens=8`, median `prompt_per_second`, one lease per build, same session. Affine reused (2101 / 2173 / 2115).

### Bisect 4k (this session)

| point | git | sha256 | load1 | prefill 3 req | median |
| --- | --- | --- | --- | --- | --- |
| (1) 449e3e24+reuse | 449e3e24 | c9c424b2 | 2.88 | 1581 / 1802 / 1781 | **1781** |
| (2) f32 finish reduce | 5815028d | 634ff0d6 | 1.76 | 1715 / 1804 / 1806 | **1804** |
| (3) Opus 1 end | f54b6fdf | 32e1b6d3 | 2.02 | 1700 / 1800 / 1778 | **1778** |
| (3a) pre-dtype | 9a2e2e30 | 23b03572 | 2.34 | 1675 / 1736 / 1741 | **1736** |
| (3b) activation dtype | 43cf388e | 5877c008 | 3.22 | 1645 / 1730 / 1731 | **1730** |
| (4) Opus 2 | 728af897 | 3f2e4f3b | 1.80 | 1059 / 1658 / 1657 | **1657** |
| (5) cap-scope B13 | a9598bd0 | f2f06b3f | 2.04 | 1505 / 1661 / 1668 | **1661** |

**5815028d is not the 4k offender** (1804, above control). **43cf388e is not isolated** (1730 vs parent 1736). The listed drop vs control is **Opus 2 (`f54b6fdf` 1778 → `728af897` 1657, ~7%)**. First 728af request 1059 is a cold Metal specialization, not the 17%. Cap-scope `a9598bd0` is flat with 728af.

The 17% (1397 vs 1691) was a **3-boot six-number session** with boots 1619 / 1360 / 1397. Not reproduced as 17% here.

### Six-number re-take on `2f876392`

sha256 `3f090c2f`. 3 boots, `max_tokens=512`, `completion_tokens=512`, `finish_reason=length`. Engagement `win=32 aligned=1 nwin=1671 mixed=0 n=45680`.

| ctx | prefill b1 / b2 / b3 | median | decode median | loadavg |
| --- | --- | --- | --- | --- |
| 4k | 1711.3 / 1505.1 / 1572.6 | **1572.6** | 53.3 | 1.31 / 2.28 / 1.72 |
| 16k | 1647.8 / 1605.1 / 1500.3 | **1605.1** | 54.2 | same |
| 64k | 1466.7 / 1426.4 / 1420.4 | **1426.4** | 52.5 | same |

vs frozen affine 2101 / 2173 / 2115: 0.748× / 0.739× / 0.675×. 4k bar 1575: **at the line** (1573). Prior host-table-only 1691; prior integration six-number 1397 was the noisy session.

### 64k climb vs the 4k bisect

Boot 1 full-width chunks ms: 4821, 4639, 4832, 4882, 5143, 5644, 5885, 6785. Same kv-dependent climb after ~32k. **Not explained by 5815028d** (that commit’s 4k is 1804). PROFILE_FWD already showed mlp (MoE, including host eids eval) flat; climb sits outside MoE/attn/gdn block sums (QSA/KV working set). No `/props` per-chunk series this turn (one 64k request, server not instrumented between chunks).

## Open questions

- 728af897 1657 vs 43cf388e 1730 in the 1-boot method: remaining ~4% not pinned to a single diff (union-hist latch and table-frees are error-path only).
- 4k 1573 vs 1800 stretch.
- 64k climb after kv 32k is still the EXL3-specific kv term.

## Comments

none added

## Ports

none
