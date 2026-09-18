const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const exl3 = @import("expert_exl3.zig");
const io_util = @import("io_util.zig");

var ubench_force: bool = false;

fn exl3UbenchOn() bool {
    if (ubench_force) return true;
    const p = std.c.getenv("MLX_SERVE_EXL3_LAYER_UBENCH") orelse return false;
    return p[0] == '1';
}

fn ubenchEval(a: mlx.mlx_array, name: []const u8) !void {
    if (!exl3UbenchOn()) return;
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(a));
    const ns = sw.read();
    std.debug.print("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
    log.info("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
}

pub const DECODE_ROWS_MAX: usize = 16;

pub fn usesPrefillArm(rows: usize) bool {
    return rows > DECODE_ROWS_MAX;
}

pub const RunTable = struct {
    start: []u32,
    len: []u32,
    eid: []u32,
    n: u32,
};

pub fn buildRuns(alloc: std.mem.Allocator, experts: []const u32) !RunTable {
    if (experts.len == 0) return .{ .start = &.{}, .len = &.{}, .eid = &.{}, .n = 0 };
    var n: u32 = 1;
    var i: usize = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) n += 1;
    }
    const start = try alloc.alloc(u32, n);
    const len = try alloc.alloc(u32, n);
    const eid = try alloc.alloc(u32, n);
    var r: u32 = 0;
    start[0] = 0;
    eid[0] = experts[0];
    i = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) {
            len[r] = @intCast(i - start[r]);
            r += 1;
            start[r] = @intCast(i);
            eid[r] = experts[i];
        }
    }
    len[r] = @intCast(experts.len - start[r]);
    return .{ .start = start, .len = len, .eid = eid, .n = n };
}

const GEMV_SOURCE: [:0]const u8 =
    \\uint gid = uint(thread_position_in_grid.x);
    \\if (gid >= uint(ODIM)) return;
    \\constexpr uint TILE = 16u;
    \\constexpr uint IN_TILES = uint(IDIM) / TILE;
    \\constexpr uint OUT_TILES = uint(ODIM) / TILE;
    \\const uint tn = gid / TILE;
    \\const uint local = gid % TILE;
    \\float acc = 0.0f;
    \\for (uint tk = 0u; tk < IN_TILES; tk++) {
    \\  const device ushort* tile = trellis + (tk * OUT_TILES + tn) * 64u;
    \\  ushort cw[256];
    \\  const uint word_count = 32u;
    \\  for (uint th = 0u; th < 128u; th++) {
    \\    const int bit0 = int(th) * 8 + 4 + 1024 - 16;
    \\    const int bit2 = bit0 + 4 + 16;
    \\    const int index0 = bit0 / 32;
    \\    const int index1 = (bit2 - 1) / 32;
    \\    const uint shift = uint((index1 + 1) * 32 - bit2);
    \\    const uint a = uint(tile[(uint(index0) % word_count) * 2u]) | (uint(tile[(uint(index0) % word_count) * 2u + 1u]) << 16u);
    \\    const uint b = uint(tile[(uint(index1) % word_count) * 2u]) | (uint(tile[(uint(index1) % word_count) * 2u + 1u]) << 16u);
    \\    const ulong merged = (ulong(a) << 32) | ulong(b);
    \\    const uint funnel = uint(merged >> shift);
    \\    cw[th * 2u] = ushort((funnel >> 4u) & 0xffffu);
    \\    cw[th * 2u + 1u] = ushort(funnel & 0xffffu);
    \\  }
    \\  for (uint slot = 0u; slot < 256u; slot++) {
    \\    const uint lane = slot / 8u;
    \\    const uint s = slot % 8u;
    \\    const uint row0 = (lane & 3u) * 2u;
    \\    const uint col0 = lane >> 2u;
    \\    uint pos;
    \\    switch (s) {
    \\      case 0u: pos = row0 * 16u + col0; break;
    \\      case 1u: pos = (row0 + 1u) * 16u + col0; break;
    \\      case 2u: pos = (row0 + 8u) * 16u + col0; break;
    \\      case 3u: pos = (row0 + 9u) * 16u + col0; break;
    \\      case 4u: pos = row0 * 16u + col0 + 8u; break;
    \\      case 5u: pos = (row0 + 1u) * 16u + col0 + 8u; break;
    \\      case 6u: pos = (row0 + 8u) * 16u + col0 + 8u; break;
    \\      default: pos = (row0 + 9u) * 16u + col0 + 8u; break;
    \\    }
    \\    if ((pos % 16u) != local) continue;
    \\    const uint mixed = uint(cw[slot]) * 0x83DCD12Du;
    \\    const uint pair_sums = (mixed & 0x00FF00FFu) + ((mixed >> 8u) & 0x00FF00FFu);
    \\    const uint byte_sum = 0x6400u + (pair_sums & 0xFFFFu) + (pair_sums >> 16u);
    \\    const half hh = as_type<half>(ushort(byte_sum));
    \\    const half inv = as_type<half>(ushort(0x1EEEu));
    \\    const half bias = as_type<half>(ushort(0xC931u));
    \\    const float w = float(fma(hh, inv, bias));
    \\    acc += float(x[tk * TILE + (pos / 16u)]) * w;
    \\  }
    \\}
    \\y[gid] = half(acc);
;

const GEMM_SORTED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[2 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = win * 4u;
    \\const uint ntot = uint(NROWS);
    \\const uint n = (start >= ntot) ? 0u : min(4u, ntot - start);
    \\const uint eid0 = (n == 0u) ? 0u : uint(eids[start]);
    \\uint same = 1u;
    \\for (uint r = 1u; r < n; r++) {
    \\  if (uint(eids[start + r]) != eid0) same = 0u;
    \\}
    \\const uint pg = sg >> 1u;
    \\const uint th = sg & 1u;
    \\const uint first = th * 128u + lane * 4u;
    \\uint pos[4];
    \\uint irow[4];
    \\for (uint s = 0u; s < 4u; s++) {
    \\  const uint src = first + s;
    \\  const uint ln = src >> 3u;
    \\  const uint sl = src & 7u;
    \\  const uint row0 = (ln & 3u) * 2u;
    \\  const uint col0 = ln >> 2u;
    \\  uint p;
    \\  switch (sl) {
    \\    case 0u: p = row0 * 16u + col0; break;
    \\    case 1u: p = (row0 + 1u) * 16u + col0; break;
    \\    case 2u: p = (row0 + 8u) * 16u + col0; break;
    \\    case 3u: p = (row0 + 9u) * 16u + col0; break;
    \\    case 4u: p = row0 * 16u + col0 + 8u; break;
    \\    case 5u: p = (row0 + 1u) * 16u + col0 + 8u; break;
    \\    case 6u: p = (row0 + 8u) * 16u + col0 + 8u; break;
    \\    default: p = (row0 + 9u) * 16u + col0 + 8u; break;
    \\  }
    \\  pos[s] = p;
    \\  irow[s] = p >> 4u;
    \\}
    \\float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    \\if (n > 0u && same) {
    \\  for (uint tk = pg; tk < IT; tk += 2u) {
    \\    const device uint* words = (const device uint*)(trellis + ((((size_t)eid0 * (size_t)IT + tk) * (size_t)OT + ot) * 64u));
    \\    const int bits = 4;
    \\    const int b0 = int(first) * bits + bits + 1024 - 16;
    \\    const int b2 = b0 + bits + 16;
    \\    const int i0 = b0 / 32;
    \\    const int i1 = (b2 - 1) / 32;
    \\    const uint sh0 = uint((i1 + 1) * 32 - b2);
    \\    const ulong m0 = ((ulong)words[uint(i0) % 32u] << 32) | (ulong)words[uint(i1) % 32u];
    \\    const uint f0 = uint(m0 >> sh0);
    \\    const int b3 = int(first + 2u) * bits + bits + 1024 - 16;
    \\    const int b5 = b3 + bits + 16;
    \\    const int i2 = b3 / 32;
    \\    const int i3 = (b5 - 1) / 32;
    \\    const uint sh1 = uint((i3 + 1) * 32 - b5);
    \\    const ulong m1 = ((ulong)words[uint(i2) % 32u] << 32) | (ulong)words[uint(i3) % 32u];
    \\    const uint f1 = uint(m1 >> sh1);
    \\    const uint2 lo = uint2((f0 >> 4u) & 0xffffu, f0 & 0xffffu);
    \\    const uint2 hi = uint2((f1 >> 4u) & 0xffffu, f1 & 0xffffu);
    \\    const uint2 mlo = lo * uint2(0x83DCD12Du);
    \\    const uint2 mhi = hi * uint2(0x83DCD12Du);
    \\    const uint2 pslo = (mlo & uint2(0x00FF00FFu)) + ((mlo >> uint2(8u)) & uint2(0x00FF00FFu));
    \\    const uint2 pshi = (mhi & uint2(0x00FF00FFu)) + ((mhi >> uint2(8u)) & uint2(0x00FF00FFu));
    \\    const uint2 bslo = uint2(0x6400u) + (pslo & uint2(0xFFFFu)) + (pslo >> uint2(16u));
    \\    const uint2 bshi = uint2(0x6400u) + (pshi & uint2(0xFFFFu)) + (pshi >> uint2(16u));
    \\    const half2 hlo = as_type<half2>(ushort2(bslo & uint2(0xFFFFu)));
    \\    const half2 hhi = as_type<half2>(ushort2(bshi & uint2(0xFFFFu)));
    \\    const half2 inv = as_type<half2>(ushort2(0x1EEEu));
    \\    const half2 bias = as_type<half2>(ushort2(0xC931u));
    \\    const float2 wlo = float2(fma(hlo, inv, bias));
    \\    const float2 whi = float2(fma(hhi, inv, bias));
    \\    const float4 wt = float4(wlo.x, wlo.y, whi.x, whi.y);
    \\    const uint ib = tk * TILE;
    \\    for (uint s = 0u; s < 4u; s++) {
    \\      const size_t col = (size_t)(ib + irow[s]);
    \\      float4 a = float4(0.0f);
    \\      a.x = float(x[(size_t)start * (size_t)(IDIM) + col]);
    \\      if (n > 1u) a.y = float(x[(size_t)(start + 1u) * (size_t)(IDIM) + col]);
    \\      if (n > 2u) a.z = float(x[(size_t)(start + 2u) * (size_t)(IDIM) + col]);
    \\      if (n > 3u) a.w = float(x[(size_t)(start + 3u) * (size_t)(IDIM) + col]);
    \\      acc[s] = fma(a, float4(wt[s]), acc[s]);
    \\    }
    \\  }
    \\} else if (n > 0u) {
    \\  for (uint rr = 0u; rr < n; rr++) {
    \\    const uint eid = uint(eids[start + rr]);
    \\    for (uint tk = pg; tk < IT; tk += 2u) {
    \\      const device uint* words = (const device uint*)(trellis + ((((size_t)eid * (size_t)IT + tk) * (size_t)OT + ot) * 64u));
    \\      const int bits = 4;
    \\      const int b0 = int(first) * bits + bits + 1024 - 16;
    \\      const int b2 = b0 + bits + 16;
    \\      const int i0 = b0 / 32;
    \\      const int i1 = (b2 - 1) / 32;
    \\      const uint sh0 = uint((i1 + 1) * 32 - b2);
    \\      const ulong m0 = ((ulong)words[uint(i0) % 32u] << 32) | (ulong)words[uint(i1) % 32u];
    \\      const uint f0 = uint(m0 >> sh0);
    \\      const int b3 = int(first + 2u) * bits + bits + 1024 - 16;
    \\      const int b5 = b3 + bits + 16;
    \\      const int i2 = b3 / 32;
    \\      const int i3 = (b5 - 1) / 32;
    \\      const uint sh1 = uint((i3 + 1) * 32 - b5);
    \\      const ulong m1 = ((ulong)words[uint(i2) % 32u] << 32) | (ulong)words[uint(i3) % 32u];
    \\      const uint f1 = uint(m1 >> sh1);
    \\      const uint2 lo = uint2((f0 >> 4u) & 0xffffu, f0 & 0xffffu);
    \\      const uint2 hi = uint2((f1 >> 4u) & 0xffffu, f1 & 0xffffu);
    \\      const uint2 mlo = lo * uint2(0x83DCD12Du);
    \\      const uint2 mhi = hi * uint2(0x83DCD12Du);
    \\      const uint2 pslo = (mlo & uint2(0x00FF00FFu)) + ((mlo >> uint2(8u)) & uint2(0x00FF00FFu));
    \\      const uint2 pshi = (mhi & uint2(0x00FF00FFu)) + ((mhi >> uint2(8u)) & uint2(0x00FF00FFu));
    \\      const uint2 bslo = uint2(0x6400u) + (pslo & uint2(0xFFFFu)) + (pslo >> uint2(16u));
    \\      const uint2 bshi = uint2(0x6400u) + (pshi & uint2(0xFFFFu)) + (pshi >> uint2(16u));
    \\      const half2 hlo = as_type<half2>(ushort2(bslo & uint2(0xFFFFu)));
    \\      const half2 hhi = as_type<half2>(ushort2(bshi & uint2(0xFFFFu)));
    \\      const half2 inv = as_type<half2>(ushort2(0x1EEEu));
    \\      const half2 bias = as_type<half2>(ushort2(0xC931u));
    \\      const float2 wlo = float2(fma(hlo, inv, bias));
    \\      const float2 whi = float2(fma(hhi, inv, bias));
    \\      const float4 wt = float4(wlo.x, wlo.y, whi.x, whi.y);
    \\      const uint ib = tk * TILE;
    \\      for (uint s = 0u; s < 4u; s++) {
    \\        const float av = float(x[(size_t)(start + rr) * (size_t)(IDIM) + ib + irow[s]]);
    \\        acc[s][rr] = fma(av, wt[s], acc[s][rr]);
    \\      }
    \\    }
    \\  }
    \\}
    \\for (uint rr = 0u; rr < 4u; rr++) {
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint s = 0u; s < 4u; s++) {
    \\    partial[pg * 256u + pos[s]] = acc[s][rr];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lid < 16u && rr < n) {
    \\    float sum = 0.0f;
    \\    for (uint r = 0u; r < 16u; r++) {
    \\      const uint p = r * 16u + lid;
    \\      sum += partial[p] + partial[256u + p];
    \\    }
    \\    y[(size_t)(start + rr) * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\  }
    \\}
;

const GEMM_NAX_HEADER: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\static inline half2 mul1_pair(uint2 cw) {
    \\  const uint2 mixed = cw * uint2(0x83DCD12Du);
    \\  const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
    \\  const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
    \\  const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
    \\  return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
    \\}
    \\using nfrag = vec<half, 8>;
    \\static inline nfrag nax_wfrag(const device uint *words, uint lane) {
    \\  const uint source0 = (lane & 16u) + ((lane & 7u) << 1u);
    \\  const uint2 current = *(const device uint2 *)(words + source0);
    \\  const uint previous = words[(source0 + 31u) & 31u];
    \\  const uint slot = ((lane >> 3u) & 1u) * 2u;
    \\  const ulong w0 = ((ulong)previous << 32) | (ulong)current.x;
    \\  const ulong w1 = ((ulong)current.x << 32) | (ulong)current.y;
    \\  const uint s0 = 28u - slot * 4u;
    \\  const uint s1 = 28u - (slot + 4u) * 4u;
    \\  const half2 p00 = mul1_pair(uint2(uint(w0 >> s0) & 0xffffu, uint(w0 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p01 = mul1_pair(uint2(uint(w1 >> s0) & 0xffffu, uint(w1 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p10 = mul1_pair(uint2(uint(w0 >> s1) & 0xffffu, uint(w0 >> (s1 - 4u)) & 0xffffu));
    \\  const half2 p11 = mul1_pair(uint2(uint(w1 >> s1) & 0xffffu, uint(w1 >> (s1 - 4u)) & 0xffffu));
    \\  return nfrag(p00.x, p00.y, p01.x, p01.y, p10.x, p10.y, p11.x, p11.y);
    \\}
    \\static inline short2 nax_origin(uint lane) {
    \\  const short qid = short(lane >> 2u);
    \\  return short2(short(((qid & 2) | short(lane & 1u)) * 4), short((qid & 4) | short((lane >> 1u) & 3u)));
    \\}
;

const GEMM_NAX_SOURCE: [:0]const u8 =
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = win * 4u;
    \\const uint ntot = uint(NROWS);
    \\const uint n = (start >= ntot) ? 0u : min(4u, ntot - start);
    \\if (n == 0u) return;
    \\uint eid0 = uint(eids[start]);
    \\uint same = 1u;
    \\for (uint r = 1u; r < n; r++) {
    \\  if (uint(eids[start + r]) != eid0) same = 0u;
    \\}
    \\constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroup> op;
    \\auto left = op.get_left_input_cooperative_tensor<half, half, float>();
    \\auto right = op.get_right_input_cooperative_tensor<half, half, float>();
    \\auto destination = op.get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
    \\const short2 origin = nax_origin(lane);
    \\const uint output_base = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
    \\const uint npass = same ? 1u : n;
    \\for (uint pass = 0u; pass < npass; pass++) {
    \\  const uint row0 = same ? start : (start + pass);
    \\  const uint nlive = same ? n : 1u;
    \\  const uint eid = same ? eid0 : uint(eids[row0]);
    \\  const bool active = origin.y < short(nlive);
    \\  for (uint s = 0u; s < destination.get_capacity(); s++) destination[s] = 0.0f;
    \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * 64u);
    \\  for (uint tk = 0u; tk < IT; tk++) {
    \\    const uint input_base = tk * TILE;
    \\    nfrag act;
    \\    for (short c = 0; c < 4; c++) {
    \\      const uint ic = input_base + uint(origin.x + c);
    \\      act[c] = active ? x[(size_t)(row0 + uint(origin.y)) * (size_t)(IDIM) + ic] : half(0.0h);
    \\      act[4 + c] = half(0.0h);
    \\    }
    \\    const device uint *words0 = trellis_e + ((size_t)tk * (size_t)OT + output_base / 16u) * 32u;
    \\    const nfrag w0 = nax_wfrag(words0, lane);
    \\    const nfrag w1 = nax_wfrag(words0 + 32u, lane);
    \\    for (short s = 0; s < 8; s++) {
    \\      left[s] = act[s];
    \\      right[s] = w0[s];
    \\      right[8 + s] = w1[s];
    \\    }
    \\    op.run(left, right, destination);
    \\  }
    \\  if (active) {
    \\    for (short ct = 0; ct < 2; ct++) {
    \\      for (short c = 0; c < 4; c++) {
    \\        const uint col = output_base + uint(ct) * 16u + uint(origin.x + c);
    \\        if (col < uint(ODIM)) {
    \\          y[(size_t)(row0 + uint(origin.y)) * (size_t)(ODIM) + col] = half(destination[uint(ct) * 8u + uint(c)]);
    \\        }
    \\      }
    \\    }
    \\  }
    \\}
;

const TOKEN_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint orig = uint(order[slot]);
    \\const uint row = orig / uint(TOPK);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[yb + lane] = half((s0 + s2) * sc);
    \\y[yb + lane + 32u] = half((s1 + s3) * sc);
    \\y[yb + lane + 64u] = half((s0 - s2) * sc);
    \\y[yb + lane + 96u] = half((s1 - s3) * sc);
;

const TOKEN_REDUCE_SOURCE: [:0]const u8 =
    \\uint col = uint(thread_position_in_grid.x);
    \\uint row = uint(thread_position_in_grid.y);
    \\if (col >= uint(ODIM)) return;
    \\half acc = half(0.0f);
    \\for (uint k = 0u; k < uint(TOPK); k++) {
    \\  const uint orig = row * uint(TOPK) + k;
    \\  const uint si = uint(inv[orig]);
    \\  const half p = half(float(d[(size_t)si * (size_t)(ODIM) + col]) * float(half(sc[orig])));
    \\  acc += p;
    \\}
    \\y[(size_t)row * (size_t)(ODIM) + col] = acc;
;

const PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc);
    \\y[xb + lane + 32u] = half((s1 + s3) * sc);
    \\y[xb + lane + 64u] = half((s0 - s2) * sc);
    \\y[xb + lane + 96u] = half((s1 - s3) * sc);
;

const FINISH_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\float4 v = float4(
    \\  float(inner[xb + lane]),
    \\  float(inner[xb + lane + 32u]),
    \\  float(inner[xb + lane + 64u]),
    \\  float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc * float(svh[sb + lane]));
    \\y[xb + lane + 32u] = half((s1 + s3) * sc * float(svh[sb + lane + 32u]));
    \\y[xb + lane + 64u] = half((s0 - s2) * sc * float(svh[sb + lane + 64u]));
    \\y[xb + lane + 96u] = half((s1 - s3) * sc * float(svh[sb + lane + 96u]));
;

var gemm_sorted_kernel: ?mlx.mlx_fast_metal_kernel = null;
var prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var finish_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_engaged: bool = false;

const GemvKey = struct { in_dim: c_int, out_dim: c_int };

fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.cfgs, 0..) |c, i| {
                if (c == null) {
                    victim = i;
                    break;
                }
                if (self.used[i] < oldest) {
                    oldest = self.used[i];
                    victim = i;
                }
            }
            if (self.cfgs[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
            self.cfgs[victim] = cfg;
            self.keys[victim] = key;
            self.tick += 1;
            self.used[victim] = self.tick;
        }
    };
}

const IndexedKey = struct { in_dim: c_int, out_dim: c_int, topk: c_int };
const UnaryKey = struct { dim: c_int, topk: c_int };
const GemmSortedKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int };
var indexed_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var prepare_cfgs: CfgCache(UnaryKey, 8) = .{};
var finish_cfgs: CfgCache(UnaryKey, 8) = .{};
var gemm_sorted_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemm_nax_failed: bool = false;
const PairPrepKey = struct { in_dim: c_int, nslots: c_int, topk: c_int };
const PairGemvKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int };
const MidKey = struct { dim: c_int, nslots: c_int };
const ReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int };
var pair_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var pair_gemv_cfgs: CfgCache(PairGemvKey, 8) = .{};
var mid_cfgs: CfgCache(MidKey, 8) = .{};
var reduce_cfgs: CfgCache(ReduceKey, 8) = .{};
var token_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_reduce_cfgs: CfgCache(ReduceKey, 8) = .{};
var token_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fused_dispatches: u32 = 0;

pub fn resetFusedDispatchCount() void {
    fused_dispatches = 0;
}

pub fn fusedDispatchCount() u32 {
    return fused_dispatches;
}

fn gpuArch(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

fn gemmNaxOn() bool {
    if (gemm_nax_failed) return false;
    if (std.c.getenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '1') return false;
    }
    var buf: [128]u8 = undefined;
    const arch = gpuArch(&buf) orelse return false;
    var i: usize = 0;
    while (i + 2 < arch.len) : (i += 1) {
        const a = arch[i] | 32;
        const b = arch[i + 1] | 32;
        if (a == 'g' and b == '1' and arch[i + 2] >= '7' and arch[i + 2] <= '9') return true;
    }
    return false;
}

fn getGemmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (gemm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis", "eids" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_k4_gemm_nax",
        in_vec,
        out_vec,
        GEMM_NAX_SOURCE,
        GEMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) {
        gemm_nax_failed = true;
        return error.MetalKernelCompileFailed;
    }
    gemm_nax_kernel = kernel;
    return kernel;
}

fn getGemmSortedKernel() !mlx.mlx_fast_metal_kernel {
    if (gemm_sorted_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis", "eids" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_k4_gemm_sorted",
        in_vec,
        out_vec,
        GEMM_SORTED_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gemm_sorted_kernel = kernel;
    return kernel;
}

pub fn innerGemmSorted(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 2 or tsh.len != 4) return error.BadExl3Shape;
    const n = xsh[0];
    const in_dim = xsh[1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const nwin = @divFloor(n + 3, 4);
    const key = GemmSortedKey{ .in_dim = in_dim, .out_dim = out_dim, .rows = n };
    if (gemmNaxOn() and @rem(out_dim, 128) == 0) {
        if (getGemmNaxKernel()) |nk| {
            const ncfg = gemm_nax_cfgs.get(key) orelse blk: {
                const c = mlx.mlx_fast_metal_kernel_config_new();
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, out_dim }, 2, .float16));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_dim, nwin, 1));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NROWS", n));
                gemm_nax_cfgs.put(key, c);
                break :blk c;
            };
            const ninputs = [_]mlx.mlx_array{ x, trellis, eids };
            const ninputs_vec = mlx.mlx_vector_array_new_data(&ninputs, ninputs.len);
            defer _ = mlx.mlx_vector_array_free(ninputs_vec);
            var noutputs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(noutputs);
            if (mlx.mlx_fast_metal_kernel_apply(&noutputs, nk, ninputs_vec, ncfg, s) == 0 and mlx.mlx_vector_array_size(noutputs) == 1) {
                var nout = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(nout);
                try mlx.check(mlx.mlx_vector_array_get(&nout, noutputs, 0));
                return nout;
            }
            gemm_nax_failed = true;
        } else |_| {
            gemm_nax_failed = true;
        }
    }
    const cfg = gemm_sorted_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, out_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, nwin, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NROWS", n));
        gemm_sorted_cfgs.put(key, c);
        break :blk c;
    };
    const inputs = [_]mlx.mlx_array{ x, trellis, eids };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGemmSortedKernel(), inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

fn getPrepareKernel() !mlx.mlx_fast_metal_kernel {
    if (prepare_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "suh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_prepare_h128",
        in_vec,
        out_vec,
        PREPARE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    prepare_kernel = kernel;
    return kernel;
}

fn getFinishKernel() !mlx.mlx_fast_metal_kernel {
    if (finish_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "inner", "svh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_finish_h128",
        in_vec,
        out_vec,
        FINISH_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    finish_kernel = kernel;
    return kernel;
}

fn applyUnary(
    s: mlx.mlx_stream,
    kernel: mlx.mlx_fast_metal_kernel,
    inputs: []const mlx.mlx_array,
    cfg: mlx.mlx_fast_metal_kernel_config,
) !mlx.mlx_array {
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub fn prepareIndexed(s: mlx.mlx_stream, x_in: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const topk = mlx.getShape(slots)[0];
    const xsh = mlx.getShape(x_in);
    const in_dim = xsh[xsh.len - 1];
    var x = x_in;
    var owned = false;
    if (xsh.len == 1) {
        var b = mlx.mlx_array_new();
        const shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_broadcast_to(&b, x_in, &shape, 2, s));
        x = b;
        owned = true;
    }
    defer if (owned) {
        _ = mlx.mlx_array_free(x);
    };
    const blocks = @divExact(in_dim, 128);
    const key = UnaryKey{ .dim = in_dim, .topk = topk };
    const cfg = prepare_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        prepare_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getPrepareKernel(), &.{ x, suh, slots }, cfg);
}

pub fn finishIndexed(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(inner);
    const topk = sh[0];
    const out_dim = sh[1];
    const blocks = @divExact(out_dim, 128);
    const key = UnaryKey{ .dim = out_dim, .topk = topk };
    const cfg = finish_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        finish_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getFinishKernel(), &.{ inner, svh, slots }, cfg);
}

fn getGemvKernel() !mlx.mlx_fast_metal_kernel {
    if (gemv_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_k4_mcg_gemv",
        in_vec,
        out_vec,
        GEMV_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gemv_kernel = kernel;
    return kernel;
}

var inner_gemv_cfgs: CfgCache(GemvKey, 8) = .{};

fn gemvConfig(in_dim: c_int, out_dim: c_int) !mlx.mlx_fast_metal_kernel_config {
    const key = GemvKey{ .in_dim = in_dim, .out_dim = out_dim };
    if (inner_gemv_cfgs.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{out_dim};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    inner_gemv_cfgs.put(key, cfg);
    return cfg;
}

const INDEXED_COOP_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint SPLITS = 1u;
    \\const uint eid = uint(slots[slot]);
    \\const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\uint pos[8];
    \\pos[0] = prow * 16u + pcol;
    \\pos[1] = (prow + 1u) * 16u + pcol;
    \\pos[2] = (prow + 8u) * 16u + pcol;
    \\pos[3] = (prow + 9u) * 16u + pcol;
    \\pos[4] = prow * 16u + pcol + 8u;
    \\pos[5] = (prow + 1u) * 16u + pcol + 8u;
    \\pos[6] = (prow + 8u) * 16u + pcol + 8u;
    \\pos[7] = (prow + 9u) * 16u + pcol + 8u;
    \\const uint row0 = pos[0] >> 4u;
    \\const uint row1 = pos[1] >> 4u;
    \\const uint row2 = pos[2] >> 4u;
    \\const uint row3 = pos[3] >> 4u;
    \\float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\const device uint* trellis_e = (const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * 64u);
    \\const size_t xb = (size_t)slot * (size_t)(IDIM);
    \\for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\  const float in0 = float(x[xb + tk * TILE + row0]);
    \\  const float in1 = float(x[xb + tk * TILE + row1]);
    \\  const float in2 = float(x[xb + tk * TILE + row2]);
    \\  const float in3 = float(x[xb + tk * TILE + row3]);
    \\  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\    const uint2 mixed = cw * uint2(0x83DCD12Du);
    \\    const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
    \\    const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
    \\    const half2 hh = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
    \\    const half2 inv = as_type<half2>(ushort2(0x1EEEu));
    \\    const half2 bias = as_type<half2>(ushort2(0xC931u));
    \\    const float2 w = float2(fma(hh, inv, bias));
    \\    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\  }
    \\}
    \\for (uint si = 0u; si < 8u; si++) {
    \\  partial[sg * 256u + pos[si]] = acc[si];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (lid < 16u) {
    \\  float sum = 0.0f;
    \\  for (uint r = 0u; r < 16u; r++) {
    \\    const uint p = r * 16u + lid;
    \\    for (uint g = 0u; g < SGS; g++) {
    \\      sum += partial[g * 256u + p];
    \\    }
    \\  }
    \\  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;

var indexed_coop_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getIndexedCoopKernel() !mlx.mlx_fast_metal_kernel {
    if (indexed_coop_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_k4_mul1_gemv_indexed",
        in_vec,
        out_vec,
        INDEXED_COOP_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    indexed_coop_kernel = kernel;
    return kernel;
}

pub fn indexedGemvCoopF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    if (tsh[1] * 16 != in_dim or tsh[3] != 64) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk };
    const cfg = indexed_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        indexed_coop_cfgs.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getIndexedCoopKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub fn projectIndexed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x, suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try indexedGemvCoopF16(s, prepared, trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots);
}

const PAIR_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const uint row = slot / uint(TOPK);
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\const float x0 = float(x[xb + lane]);
    \\const float x1 = float(x[xb + lane + 32u]);
    \\const float x2 = float(x[xb + lane + 64u]);
    \\const float x3 = float(x[xb + lane + 96u]);
    \\float4 v = float4(
    \\  x0 * float(suhg[sb + lane]),
    \\  x1 * float(suhg[sb + lane + 32u]),
    \\  x2 * float(suhg[sb + lane + 64u]),
    \\  x3 * float(suhg[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float sc = 0.08838834764831845f;
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\yg[yb + lane] = half((s0 + s2) * sc);
    \\yg[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yg[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yg[yb + lane + 96u] = half((s1 - s3) * sc);
    \\v = float4(
    \\  x0 * float(suhu[sb + lane]),
    \\  x1 * float(suhu[sb + lane + 32u]),
    \\  x2 * float(suhu[sb + lane + 64u]),
    \\  x3 * float(suhu[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yu[yb + lane] = half((s0 + s2) * sc);
    \\yu[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yu[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yu[yb + lane + 96u] = half((s1 - s3) * sc);
;

const PAIR_GEMV_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\const uint eid = uint(slots[slot]);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\uint pos[8];
    \\pos[0] = prow * 16u + pcol;
    \\pos[1] = (prow + 1u) * 16u + pcol;
    \\pos[2] = (prow + 8u) * 16u + pcol;
    \\pos[3] = (prow + 9u) * 16u + pcol;
    \\pos[4] = prow * 16u + pcol + 8u;
    \\pos[5] = (prow + 1u) * 16u + pcol + 8u;
    \\pos[6] = (prow + 8u) * 16u + pcol + 8u;
    \\pos[7] = (prow + 9u) * 16u + pcol + 8u;
    \\const uint row0 = pos[0] >> 4u;
    \\const uint row1 = pos[1] >> 4u;
    \\const uint row2 = pos[2] >> 4u;
    \\const uint row3 = pos[3] >> 4u;
    \\const size_t xb = (size_t)slot * (size_t)(IDIM);
    \\for (uint proj = 0u; proj < 2u; proj++) {
    \\  const device half *x = (proj == 0u) ? xg : xu;
    \\  const device ushort *trellis = (proj == 0u) ? tg : tu;
    \\  device half *y = (proj == 0u) ? yg : yu;
    \\  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * 64u);
    \\  for (uint tk = sg; tk < IT; tk += SGS) {
    \\    const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\    const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\    const float in0 = float(x[xb + tk * TILE + row0]);
    \\    const float in1 = float(x[xb + tk * TILE + row1]);
    \\    const float in2 = float(x[xb + tk * TILE + row2]);
    \\    const float in3 = float(x[xb + tk * TILE + row3]);
    \\    const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\      const uint2 mixed = cw * uint2(0x83DCD12Du);
    \\      const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
    \\      const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
    \\      const half2 hh = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
    \\      const half2 inv = as_type<half2>(ushort2(0x1EEEu));
    \\      const half2 bias = as_type<half2>(ushort2(0xC931u));
    \\      const float2 w = float2(fma(hh, inv, bias));
    \\      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\    }
    \\  }
    \\  for (uint si = 0u; si < 8u; si++) {
    \\    partial[sg * 256u + pos[si]] = acc[si];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lid < 16u) {
    \\    float sum = 0.0f;
    \\    for (uint r = 0u; r < 16u; r++) {
    \\      const uint p = r * 16u + lid;
    \\      for (uint g = 0u; g < SGS; g++) {
    \\        sum += partial[g * 256u + p];
    \\      }
    \\    }
    \\    y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;

const MID_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float sc = 0.08838834764831845f;
    \\float4 v = float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\const half g0 = half((s0 + s2) * sc * float(svhg[sb + lane]));
    \\const half g1 = half((s1 + s3) * sc * float(svhg[sb + lane + 32u]));
    \\const half g2 = half((s0 - s2) * sc * float(svhg[sb + lane + 64u]));
    \\const half g3 = half((s1 - s3) * sc * float(svhg[sb + lane + 96u]));
    \\v = float4(float(iu[xb + lane]), float(iu[xb + lane + 32u]), float(iu[xb + lane + 64u]), float(iu[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\const half u0 = half((s0 + s2) * sc * float(svhu[sb + lane]));
    \\const half u1 = half((s1 + s3) * sc * float(svhu[sb + lane + 32u]));
    \\const half u2 = half((s0 - s2) * sc * float(svhu[sb + lane + 64u]));
    \\const half u3 = half((s1 - s3) * sc * float(svhu[sb + lane + 96u]));
    \\const half ysig0 = 1 / (1 + exp(abs(g0)));
    \\const half ysig1 = 1 / (1 + exp(abs(g1)));
    \\const half ysig2 = 1 / (1 + exp(abs(g2)));
    \\const half ysig3 = 1 / (1 + exp(abs(g3)));
    \\const half sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
    \\const half sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
    \\const half sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
    \\const half sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
    \\const half silu0 = half(float(g0) * float(sig0));
    \\const half silu1 = half(float(g1) * float(sig1));
    \\const half silu2 = half(float(g2) * float(sig2));
    \\const half silu3 = half(float(g3) * float(sig3));
    \\const half h0 = half(float(silu0) * float(u0));
    \\const half h1 = half(float(silu1) * float(u1));
    \\const half h2 = half(float(silu2) * float(u2));
    \\const half h3 = half(float(silu3) * float(u3));
    \\v = float4(float(h0) * float(suhd[sb + lane]), float(h1) * float(suhd[sb + lane + 32u]), float(h2) * float(suhd[sb + lane + 64u]), float(h3) * float(suhd[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yd[xb + lane] = half((s0 + s2) * sc);
    \\yd[xb + lane + 32u] = half((s1 + s3) * sc);
    \\yd[xb + lane + 64u] = half((s0 - s2) * sc);
    \\yd[xb + lane + 96u] = half((s1 - s3) * sc);
;

const REDUCE_SOURCE: [:0]const u8 =
    \\threadgroup half vals[16 * 128];
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint row = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint base = block * 128u;
    \\const uint slot = row * uint(TOPK) + sg;
    \\const uint eid = uint(slots[slot]);
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float scv = 0.08838834764831845f;
    \\float4 v = float4(float(inner[xb + lane]), float(inner[xb + lane + 32u]), float(inner[xb + lane + 64u]), float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\vals[sg * 128u + lane] = half((s0 + s2) * scv * float(svh[sb + lane]));
    \\vals[sg * 128u + lane + 32u] = half((s1 + s3) * scv * float(svh[sb + lane + 32u]));
    \\vals[sg * 128u + lane + 64u] = half((s0 - s2) * scv * float(svh[sb + lane + 64u]));
    \\vals[sg * 128u + lane + 96u] = half((s1 - s3) * scv * float(svh[sb + lane + 96u]));
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  half a0 = half(0.0f);
    \\  half a1 = half(0.0f);
    \\  half a2 = half(0.0f);
    \\  half a3 = half(0.0f);
    \\  for (uint k = 0u; k < uint(TOPK); k++) {
    \\    const float w = float(half(sc[row * uint(TOPK) + k]));
    \\    a0 += half(float(vals[k * 128u + lane]) * w);
    \\    a1 += half(float(vals[k * 128u + lane + 32u]) * w);
    \\    a2 += half(float(vals[k * 128u + lane + 64u]) * w);
    \\    a3 += half(float(vals[k * 128u + lane + 96u]) * w);
    \\  }
    \\  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
    \\  y[yb + lane] = a0;
    \\  y[yb + lane + 32u] = a1;
    \\  y[yb + lane + 64u] = a2;
    \\  y[yb + lane + 96u] = a3;
    \\}
;

var pair_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pair_gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getNamedKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header.ptr, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = kernel;
    return kernel;
}

fn applyOuts(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, n_out: usize) !mlx.mlx_vector_array {
    fused_dispatches += 1;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != n_out) {
        _ = mlx.mlx_vector_array_free(outputs_vec);
        return error.MetalKernelBadOutputCount;
    }
    return outputs_vec;
}

fn pairPrepare(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = try getNamedKernel(&pair_prepare_kernel, "mlxserve_exl3_pair_prepare", &ins, &outs, PAIR_PREPARE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    var b = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn pairGemv(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const out_tiles = @divExact(out_dim, 16);
    const key = PairGemvKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots };
    const cfg = pair_gemv_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = [_]c_int{ nslots, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        pair_gemv_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "xg", "xu", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = try getNamedKernel(&pair_gemv_kernel, "mlxserve_exl3_pair_gemv", &ins, &outs, PAIR_GEMV_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ xg, xu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    var b = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn midSwigluPrep(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots };
    const cfg = mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&mid_kernel, "mlxserve_exl3_mid_swiglu", &ins, &outs, MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn downFinishReduce(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int) !mlx.mlx_array {
    const key = ReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk };
    const cfg = reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = if (rows == 1) [_]c_int{out_dim} ++ [_]c_int{0} else [_]c_int{ rows, out_dim };
        const ndim: usize = if (rows == 1) 1 else 2;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, ndim, .float16));
        const blocks = @divExact(out_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * blocks, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "inner", "svh", "slots", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&reduce_kernel, "mlxserve_exl3_down_reduce", &ins, &outs, REDUCE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ inner, svh, slots, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn prepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array, order: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !mlx.mlx_array {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suh", "slots", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_prepare_kernel, "mlxserve_exl3_token_prepare", &ins, &outs, TOKEN_PREPARE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suh, slots, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn tokenReduce(s: mlx.mlx_stream, d: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int) !mlx.mlx_array {
    const key = ReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk };
    const cfg = token_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        const sh = [_]c_int{ rows, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_dim, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "d", "inv", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_reduce_kernel, "mlxserve_exl3_token_reduce", &ins, &outs, TOKEN_REDUCE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ d, inv, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn moeSwigluFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const ssh = mlx.getShape(slots);
    const tsh = mlx.getShape(gate_t);
    const nslots = ssh[0];
    const hidden: c_int = if (xsh.len == 1) xsh[0] else xsh[xsh.len - 1];
    const rows: c_int = if (xsh.len == 1) 1 else xsh[0];
    const topk = @divExact(nslots, rows);
    const inter = tsh[2] * 16;
    const prep = try pairPrepare(s, x, gate_suh, up_suh, slots, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    try ubenchEval( prep[0], "pair_prepare");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(prep[1]));
    const inners = try pairGemv(s, prep[0], prep[1], gate_t, up_t, slots, hidden, inter, nslots);
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try ubenchEval( inners[0], "pair_gemv");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(inners[1]));
    const down_x = try midSwigluPrep(s, inners[0], inners[1], gate_svh, up_svh, down_suh, slots, inter, nslots);
    defer _ = mlx.mlx_array_free(down_x);
    try ubenchEval( down_x, "mid");
    const down_inner = try indexedGemvCoopF16(s, down_x, down_t, slots);
    defer _ = mlx.mlx_array_free(down_inner);
    try ubenchEval( down_inner, "down_gemv");
    fused_dispatches += 1;
    const out = try downFinishReduce(s, down_inner, down_svh, slots, scores, hidden, rows, topk);
    try ubenchEval( out, "reduce");
    return out;
}

pub fn moeSwigluIndexed(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
) !mlx.mlx_array {
    const g = try projectIndexed(s, x, gate_t, gate_suh, gate_svh, slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try projectIndexed(s, x, up_t, up_suh, up_svh, slots);
    defer _ = mlx.mlx_array_free(u);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    const d = try projectIndexed(s, h, down_t, down_suh, down_svh, slots);
    defer _ = mlx.mlx_array_free(d);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, .float16, s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_expand_dims(&sc2, sc, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

fn repeatRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, topk: c_int) !mlx.mlx_array {
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(rows), 1, .int32, s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, ar, &[_]c_int{ rows, 1 }, 2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    const shape = [_]c_int{ rows, topk };
    try mlx.check(mlx.mlx_broadcast_to(&wide, col, &shape, 2, s));
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_reshape(&idx, wide, &[_]c_int{ rows * topk }, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, x, idx, 0, s));
    return out;
}

fn projectSorted(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    var sorted_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_x);
    try mlx.check(mlx.mlx_take_axis(&sorted_x, x, order, 0, s));
    const prepared = try prepareIndexed(s, sorted_x, suh, sorted_slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, sorted_slots);
    defer _ = mlx.mlx_array_free(inner);
    const finished = try finishIndexed(s, inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(finished);
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, finished, inv, 0, s));
    return out;
}

fn projectSortedWithRuns(
    s: mlx.mlx_stream,
    x_sorted: mlx.mlx_array,
    trellis: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,
    slots_sorted: mlx.mlx_array,
) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x_sorted, suh, slots_sorted);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, slots_sorted);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots_sorted);
}

pub fn moePrefill(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const rows = xsh[0];
    const hidden = xsh[1];
    const nslots = rows * topk;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_u);
    try mlx.check(mlx.mlx_astype(&order_u, order, .uint32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    try ubenchEval( sorted_slots, "sort");
    const g_prep = try prepareFromTokens(s, x, gate_suh, sorted_slots, order_u, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(g_prep);
    const u_prep = try prepareFromTokens(s, x, up_suh, sorted_slots, order_u, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(u_prep);
    try ubenchEval( g_prep, "token_prepare");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(u_prep));
    const g_inner = try innerGemmSorted(s, g_prep, gate_t, sorted_slots);
    defer _ = mlx.mlx_array_free(g_inner);
    try ubenchEval( g_inner, "gemm_gate");
    const u_inner = try innerGemmSorted(s, u_prep, up_t, sorted_slots);
    defer _ = mlx.mlx_array_free(u_inner);
    try ubenchEval( u_inner, "gemm_up");
    const g = try finishIndexed(s, g_inner, gate_svh, sorted_slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try finishIndexed(s, u_inner, up_svh, sorted_slots);
    defer _ = mlx.mlx_array_free(u);
    if (exl3UbenchOn()) {
        try mlx.check(mlx.mlx_array_eval(g));
        try mlx.check(mlx.mlx_array_eval(u));
    }
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    const d_sorted = try projectSortedWithRuns(s, h, down_t, down_suh, down_svh, sorted_slots);
    defer _ = mlx.mlx_array_free(d_sorted);
    try ubenchEval( d_sorted, "gemm_down");
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var inv_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_u);
    try mlx.check(mlx.mlx_astype(&inv_u, inv, .uint32, s));
    const out = try tokenReduce(s, d_sorted, inv_u, scores, hidden, rows, topk);
    try ubenchEval( out, "token_reduce");
    return out;
}

pub fn prefillDecodeGatherMm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_pub: mlx.mlx_array,
    up_pub: mlx.mlx_array,
    down_pub: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    rows: c_int,
    hidden: c_int,
    inter: c_int,
    topk: c_int,
) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &[_]c_int{ rows, 1, 1, hidden }, 4, s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_gather_mm(&g4, x4, gate_pub, no_idx, slots, false, s));
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_gather_mm(&up4, x4, up_pub, no_idx, slots, false, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_squeeze(&g, g4, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_squeeze(&u, up4, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    var h4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h4);
    try mlx.check(mlx.mlx_reshape(&h4, h, &[_]c_int{ rows, topk, 1, inter }, 4, s));
    var d4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d4);
    try mlx.check(mlx.mlx_gather_mm(&d4, h4, down_pub, no_idx, slots, false, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_squeeze(&d, d4, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, mlx.mlx_array_dtype(d), s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, sc, &[_]c_int{ rows, topk, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 1, false, s));
    return out;
}

pub fn innerGemvF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 1 or tsh.len != 3) return error.BadExl3Shape;
    const in_dim = xsh[0];
    const out_dim = tsh[1] * 16;
    if (tsh[0] * 16 != in_dim or tsh[2] != 64) return error.BadExl3Shape;
    const cfg = try gemvConfig(in_dim, out_dim);
    const inputs_arr = [_]mlx.mlx_array{ x, trellis };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGemvKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gemv_engaged) {
        gemv_engaged = true;
        log.info("[expert-exl3] engaged in={d} out={d}\n", .{ in_dim, out_dim });
    }
    return out;
}

pub fn moeSwigluHost(
    alloc: std.mem.Allocator,
    x: []const f32,
    gate_t: []const u16,
    gate_suh: []const u16,
    gate_svh: []const u16,
    up_t: []const u16,
    up_suh: []const u16,
    up_svh: []const u16,
    down_t: []const u16,
    down_suh: []const u16,
    down_svh: []const u16,
    slots: []const u32,
    weights: []const f32,
    hidden: usize,
    inter: usize,
    packed_n: usize,
    in_tiles_h: usize,
    out_tiles_i: usize,
) ![]f32 {
    const topk = slots.len;
    const y = try alloc.alloc(f32, hidden);
    @memset(y, 0);
    const transformed = try alloc.alloc(f32, hidden);
    defer alloc.free(transformed);
    const inner = try alloc.alloc(f32, @max(hidden, inter));
    defer alloc.free(inner);
    const gate_y = try alloc.alloc(f32, inter);
    defer alloc.free(gate_y);
    const up_y = try alloc.alloc(f32, inter);
    defer alloc.free(up_y);
    const h = try alloc.alloc(f32, inter);
    defer alloc.free(h);
    const down_y = try alloc.alloc(f32, hidden);
    defer alloc.free(down_y);
    const tstride_gu = in_tiles_h * out_tiles_i * packed_n;
    const tstride_d = out_tiles_i * in_tiles_h * packed_n;
    for (0..topk) |k| {
        const e = slots[k];
        const g_off = e * tstride_gu;
        const u_off = e * tstride_gu;
        const d_off = e * tstride_d;
        exl3.project(x, gate_t[g_off..][0..tstride_gu], gate_suh[e * hidden ..][0..hidden], gate_svh[e * inter ..][0..inter], hidden, inter, 4, .mcg, transformed, inner[0..inter], gate_y);
        exl3.project(x, up_t[u_off..][0..tstride_gu], up_suh[e * hidden ..][0..hidden], up_svh[e * inter ..][0..inter], hidden, inter, 4, .mcg, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        exl3.project(h, down_t[d_off..][0..tstride_d], down_suh[e * inter ..][0..inter], down_svh[e * hidden ..][0..hidden], inter, hidden, 4, .mcg, inner[0..inter], transformed, down_y);
        const w = weights[k];
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
    return y;
}

test "exl3 K4 Metal inner GEMV matches the host tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const inner_meta = parsed.value.object.get("inner").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const i_off: usize = @intCast(inner_meta.get("data_offsets").?.array.items[0].integer);
    const i_end: usize = @intCast(inner_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const inner_bits = std.mem.bytesAsSlice(u16, data[i_off..i_end]);
    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const xf16 = try alloc.alloc(u16, 128);
    for (x, xf16) |v, *b| b.* = exl3.f32ToF16Bits(v);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{128}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis_bits.ptr, &[_]c_int{ 8, 8, 64 }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    var host: [128]f32 = undefined;
    @memset(&host, 0);
    for (0..128) |o| {
        var acc: f32 = 0;
        for (0..128) |i| acc += exl3.f16BitsToF32(xf16[i]) * exl3.f16BitsToF32(inner_bits[i * 128 + o]);
        host[o] = exl3.f16BitsToF32(exl3.f32ToF16Bits(acc));
    }
    for (0..128) |i| {
        const bits: u16 = @bitCast(src[i]);
        try t.expectEqual(exl3.f32ToF16Bits(host[i]), bits);
    }
}

test "exl3 buildRuns groups sorted expert ids" {
    const t = std.testing;
    const ids = [_]u32{ 3, 3, 3, 7, 7, 1 };
    const runs = try buildRuns(t.allocator, &ids);
    defer t.allocator.free(runs.start);
    defer t.allocator.free(runs.len);
    defer t.allocator.free(runs.eid);
    try t.expectEqual(@as(u32, 3), runs.n);
    try t.expectEqual(@as(u32, 0), runs.start[0]);
    try t.expectEqual(@as(u32, 3), runs.len[0]);
    try t.expectEqual(@as(u32, 3), runs.eid[0]);
    try t.expectEqual(@as(u32, 3), runs.start[1]);
    try t.expectEqual(@as(u32, 2), runs.len[1]);
    try t.expectEqual(@as(u32, 7), runs.eid[1]);
    try t.expectEqual(@as(u32, 5), runs.start[2]);
    try t.expectEqual(@as(u32, 1), runs.len[2]);
    try t.expectEqual(@as(u32, 1), runs.eid[2]);
}

test "exl3 prefill arm is used only above 16 rows" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(2));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
    try t.expect(usesPrefillArm(512));
}

test "exl3 512-row prefill: decode-to-f16 gather_mm vs rows kernel" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const pub_meta = parsed.value.object.get("public").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const p0: usize = @intCast(pub_meta.get("data_offsets").?.array.items[0].integer);
    const p1: usize = @intCast(pub_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const pub_bits = std.mem.bytesAsSlice(u16, data[p0..p1]);
    const E: c_int = 4;
    const R: c_int = 512;
    const topk: c_int = 2;
    const dim: c_int = 128;
    const stacked_t = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    const stacked_suh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_svh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_pub = try alloc.alloc(u16, @intCast(E * dim * dim));
    var e_i: c_int = 0;
    while (e_i < E) : (e_i += 1) {
        const tb: usize = @intCast(e_i);
        @memcpy(stacked_t[tb * trellis_bits.len ..][0..trellis_bits.len], trellis_bits);
        @memcpy(stacked_suh[tb * suh_bits.len ..][0..suh_bits.len], suh_bits);
        @memcpy(stacked_svh[tb * svh_bits.len ..][0..svh_bits.len], svh_bits);
        @memcpy(stacked_pub[tb * pub_bits.len ..][0..pub_bits.len], pub_bits);
    }
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, @intCast(R * dim));
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    for (scores_h) |*v| v.* = 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{ R * topk }, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const slots2 = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{ R, topk }, 2, .uint32);
    defer _ = mlx.mlx_array_free(slots2);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{ R * topk }, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const pub_a = mlx.mlx_array_new_data(stacked_pub.ptr, &[_]c_int{ E, dim, dim }, 3, .float16);
    defer _ = mlx.mlx_array_free(pub_a);
    const warm = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(warm));
    _ = mlx.mlx_array_free(warm);
    var t_rows = io_util.Stopwatch.init(t.io);
    var it: usize = 0;
    while (it < 20) : (it += 1) {
        const rows_out = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(rows_out));
        _ = mlx.mlx_array_free(rows_out);
    }
    const rows_ns = t_rows.read() / 20;
    var w_oi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_oi);
    try mlx.check(mlx.mlx_transpose_axes(&w_oi, pub_a, &[_]c_int{ 0, 2, 1 }, 3, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, w_oi, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    const n_tok: c_int = R * topk;
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var xrep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrep);
    try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ n_tok, 1, dim }, 3, s));
    var slots_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_i);
    try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
    const no_idx = mlx.mlx_array{ .ctx = null };
    var qmm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qmm);
    try mlx.check(mlx.mlx_gather_qmm(&qmm, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    try mlx.check(mlx.mlx_array_eval(qmm));
    var t_q = io_util.Stopwatch.init(t.io);
    it = 0;
    while (it < 20) : (it += 1) {
        var qmm2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qmm2, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qmm2));
        _ = mlx.mlx_array_free(qmm2);
    }
    const qmm_ns = t_q.read() / 20;
    const ratio_x100: u64 = if (qmm_ns == 0) 0 else (rows_ns * 100) / (qmm_ns * 3);
    std.debug.print("exl3 512-row E=4 H=128 topk=2: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
        rows_ns / 1000,
        qmm_ns / 1000,
        ratio_x100,
    });
    try t.expect(qmm_ns > 0);
}

test "exl3 512-row production-shape sorted gemm vs affine gather_qmm" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 4;
    const R: c_int = 512;
    const topk: c_int = 2;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_n);
    const tr_d = try alloc.alloc(u16, @intCast(E * (I / 16) * (H / 16) * 64));
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scores_h) |*v| v.* = 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{ R * topk }, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{ R * topk }, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(warm));
    _ = mlx.mlx_array_free(warm);
    var t_g = io_util.Stopwatch.init(t.io);
    var it: usize = 0;
    while (it < 5) : (it += 1) {
        const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
    }
    const gemm_ns = t_g.read() / 5;
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_random_normal(&dense, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, dense, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var xrep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrep);
    try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
    var slots_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_i);
    try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
    const no_idx = mlx.mlx_array{ .ctx = null };
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    try mlx.check(mlx.mlx_array_eval(q0));
    var t_q = io_util.Stopwatch.init(t.io);
    it = 0;
    while (it < 5) : (it += 1) {
        var q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&q, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(q));
        _ = mlx.mlx_array_free(q);
    }
    const qmm_ns = t_q.read() / 5;
    const ratio_x100: u64 = if (qmm_ns == 0) 0 else (gemm_ns * 100) / (qmm_ns * 3);
    std.debug.print("exl3 512-row E=4 H=2560 I=640 topk=2: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
        gemm_ns / 1000,
        qmm_ns / 1000,
        ratio_x100,
    });
    try t.expect(qmm_ns > 0);
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 10;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, topk * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    const xf = try alloc.alloc(f32, dim);
    const host16 = try alloc.alloc(f32, dim);
    const host32 = try alloc.alloc(f32, dim);
    for (0..topk) |k| {
        for (0..dim) |i| xf[i] = exl3.f16BitsToF32(xh[k * dim + i]);
        const e: usize = slots_h[k];
        exl3.innerGemv(stacked[e * tile_n ..][0..tile_n], xf, dim, dim, 4, .mul1, host16);
        exl3.innerGemvF32(stacked[e * tile_n ..][0..tile_n], xf, dim, dim, 4, .mul1, host32);
        for (0..dim) |o| {
            const gpu = @as(f32, @floatCast(src[k * dim + o]));
            try expectGemvEnvelope(gpu, host16[o], host32[o]);
        }
    }
}

fn f16Ulp(v: f32) f32 {
    const h: f16 = @floatCast(v);
    const bits: u16 = @bitCast(h);
    if ((bits & 0x7fff) >= 0x7c00) return 0.5;
    const up: u16 = bits + 1;
    const hu: f16 = @bitCast(up);
    return @abs(@as(f32, @floatCast(hu)) - @as(f32, @floatCast(h)));
}

fn expectGemvEnvelope(gpu: f32, host_f16: f32, ref: f32) !void {
    const eg = @abs(gpu - ref);
    const eh = @abs(host_f16 - ref);
    if (eg <= eh) return;
    if (eg <= 2 * f16Ulp(@max(@abs(ref), 1e-8))) return;
    return error.TestExpectedEqual;
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 on production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    const xf = try alloc.alloc(f32, in_dim);
    const host16 = try alloc.alloc(f32, out_dim);
    const host32 = try alloc.alloc(f32, out_dim);
    for (0..topk) |k| {
        for (0..in_dim) |i| xf[i] = exl3.f16BitsToF32(xh[k * in_dim + i]);
        const e: usize = slots_h[k];
        exl3.innerGemv(stacked[e * tile_n ..][0..tile_n], xf, in_dim, out_dim, 4, .mul1, host16);
        exl3.innerGemvF32(stacked[e * tile_n ..][0..tile_n], xf, in_dim, out_dim, 4, .mul1, host32);
        for (0..out_dim) |o| {
            const gpu = @as(f32, @floatCast(src[k * out_dim + o]));
            try expectGemvEnvelope(gpu, host16[o], host32[o]);
        }
    }
}

test "exl3 K4 cooperative indexed GEMV runs at production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(29);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const warm_new = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    try mlx.check(mlx.mlx_array_eval(warm_new));
    _ = mlx.mlx_array_free(warm_new);
    var new_ns: u64 = 0;
    var it: usize = 0;
    while (it < 10) : (it += 1) {
        var sw = io_util.Stopwatch.init(t.io);
        const b = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
        try mlx.check(mlx.mlx_array_eval(b));
        new_ns += sw.read();
        _ = mlx.mlx_array_free(b);
    }
    new_ns /= 10;
    std.debug.print("exl3 indexed GEMV H=2560 I=640 topk=10: coop {d} us\n", .{
        new_ns / 1000,
    });
    try t.expect(new_ns > 0);
}

test "exl3 layer ubench production shape rows=1 and 512" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    ubench_force = true;
    defer {
        ubench_force = false;
    }
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const g_n = in_tiles * out_tiles * 64;
    const d_n = out_tiles * in_tiles * 64;
    const tr_g = try alloc.alloc(u16, E * g_n);
    const tr_d = try alloc.alloc(u16, E * d_n);
    const suh_g = try alloc.alloc(u16, E * in_dim);
    const svh_g = try alloc.alloc(u16, E * out_dim);
    const suh_d = try alloc.alloc(u16, E * out_dim);
    const svh_d = try alloc.alloc(u16, E * in_dim);
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_tiles), @intCast(in_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const x1h = try alloc.alloc(u16, in_dim);
    for (x1h) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sl1 = try alloc.alloc(u32, topk);
    const sc1 = try alloc.alloc(f32, topk);
    for (sl1, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc1) |*v| v.* = 0.1;
    const x1 = mlx.mlx_array_new_data(x1h.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const slots1 = mlx.mlx_array_new_data(sl1.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots1);
    const scores1 = mlx.mlx_array_new_data(sc1.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores1);
    ubench_force = false;
    const warm1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1);
    try mlx.check(mlx.mlx_array_eval(warm1));
    _ = mlx.mlx_array_free(warm1);
    ubench_force = true;
    std.debug.print("exl3-ubench rows=1\n", .{});
    const y1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1);
    try mlx.check(mlx.mlx_array_eval(y1));
    _ = mlx.mlx_array_free(y1);
    const R: usize = 512;
    const xnh = try alloc.alloc(u16, R * in_dim);
    for (xnh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sln = try alloc.alloc(u32, R * topk);
    const scn = try alloc.alloc(f32, R * topk);
    for (sln, 0..) |*v, i| v.* = @intCast(i % E);
    for (scn) |*v| v.* = 0.1;
    const xn = mlx.mlx_array_new_data(xnh.ptr, &[_]c_int{ @intCast(R), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(xn);
    const slotsn = mlx.mlx_array_new_data(sln.ptr, &[_]c_int{@intCast(R * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slotsn);
    const scoresn = mlx.mlx_array_new_data(scn.ptr, &[_]c_int{@intCast(R * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scoresn);
    ubench_force = false;
    const warmn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(warmn));
    _ = mlx.mlx_array_free(warmn);
    ubench_force = true;
    std.debug.print("exl3-ubench rows=512\n", .{});
    const yn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(yn));
    _ = mlx.mlx_array_free(yn);
}

test "exl3 fused decode chain matches indexed SwiGLU on one row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(41);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    const scores_h = try alloc.alloc(f32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    resetFusedDispatchCount();
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores);
    defer _ = mlx.mlx_array_free(fused);
    const n_disp = fusedDispatchCount();
    const old = try moeSwigluIndexed(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores);
    defer _ = mlx.mlx_array_free(old);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    var c_o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_o);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_contiguous(&c_o, old, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    try mlx.check(mlx.mlx_array_eval(c_o));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    const so = mlx.mlx_array_data_float16(c_o) orelse return error.F16Unreadable;
    for (0..dim) |i| {
        const a: u16 = @bitCast(sf[i]);
        const b: u16 = @bitCast(so[i]);
        try t.expectEqual(b, a);
    }
    try t.expectEqual(@as(u32, 5), n_disp);
}

test "exl3 fused decode chain rows match N solo calls" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(43);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, rows * topk);
    const scores_h = try alloc.alloc(f32, rows * topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores);
    defer _ = mlx.mlx_array_free(fused);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const sl = mlx.mlx_array_new_data(slots_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sl);
        const sc = mlx.mlx_array_new_data(scores_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const solo = try moeSwigluIndexed(s, x1, tr, suh, svh, tr, suh, svh, tr, suh, svh, sl, sc);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        for (0..dim) |i| {
            const a: u16 = @bitCast(sf[r * dim + i]);
            const b: u16 = @bitCast(ss[i]);
            try t.expectEqual(b, a);
        }
    }
}

test "exl3 MTP MoE rows stay on the fused decode arm" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(4));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
}

test "exl3 sorted GEMM matches host MUL1 on small shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = @embedFile("fixtures/exl3_k4_linear.safetensors");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = [_]u32{ 0, 0, 0, 0, 2, 2, 1, 1 };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSorted(s, x_arr, tr_arr, eid_a);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    const xf = try alloc.alloc(f32, dim);
    const host16 = try alloc.alloc(f32, dim);
    const host32 = try alloc.alloc(f32, dim);
    for (0..n) |r| {
        for (0..dim) |i| xf[i] = exl3.f16BitsToF32(xh[r * dim + i]);
        const e: usize = eids[r];
        exl3.innerGemv(stacked[e * tile_n ..][0..tile_n], xf, dim, dim, 4, .mul1, host16);
        exl3.innerGemvF32(stacked[e * tile_n ..][0..tile_n], xf, dim, dim, 4, .mul1, host32);
        for (0..dim) |o| {
            const gpu = @as(f32, @floatCast(src[r * dim + o]));
            try expectGemvEnvelope(gpu, host16[o], host32[o]);
        }
    }
}

test "exl3 512-row E=512 topk=10 layer within 2x affine" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const R: c_int = 512;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_g_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_d_n: usize = @intCast(E * (I / 16) * (H / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_g_n);
    const tr_d = try alloc.alloc(u16, tr_d_n);
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scores_h) |*v| v.* = 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{ R * topk }, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{ R * topk }, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(warm));
    _ = mlx.mlx_array_free(warm);
    var t_g = io_util.Stopwatch.init(t.io);
    var it: usize = 0;
    while (it < 3) : (it += 1) {
        const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
    }
    const gemm_ns = t_g.read() / 3;
    var dense_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_g);
    try mlx.check(mlx.mlx_random_normal(&dense_g, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cg);
    try mlx.check(mlx.mlx_contiguous(&w_cg, dense_g, false, s));
    var triple_g = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_g);
    try mlx.check(mlx.mlx_quantize(&triple_g, w_cg, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqg);
    var wscg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscg);
    var wbig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbig);
    try mlx.check(mlx.mlx_vector_array_get(&wqg, triple_g, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscg, triple_g, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbig, triple_g, 2));
    var dense_d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_d);
    try mlx.check(mlx.mlx_random_normal(&dense_d, &[_]c_int{ E, H, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cd);
    try mlx.check(mlx.mlx_contiguous(&w_cd, dense_d, false, s));
    var triple_d = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_d);
    try mlx.check(mlx.mlx_quantize(&triple_d, w_cd, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqd);
    var wscd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscd);
    var wbid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbid);
    try mlx.check(mlx.mlx_vector_array_get(&wqd, triple_d, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscd, triple_d, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbid, triple_d, 2));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var xrep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrep);
    try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
    var xdi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xdi);
    try mlx.check(mlx.mlx_random_normal(&xdi, &[_]c_int{ R * topk, 1, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var slots_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_i);
    try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
    const no_idx = mlx.mlx_array{ .ctx = null };
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    try mlx.check(mlx.mlx_array_eval(q0));
    var t_q = io_util.Stopwatch.init(t.io);
    it = 0;
    while (it < 3) : (it += 1) {
        var qg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qg, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qg));
        _ = mlx.mlx_array_free(qg);
        var qu = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qu, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qu));
        _ = mlx.mlx_array_free(qu);
        var qd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qd, xdi, wqd, wscd, wbid, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qd));
        _ = mlx.mlx_array_free(qd);
    }
    const affine_ns = t_q.read() / 3;
    const ratio_x100: u64 = if (affine_ns == 0) 0 else (gemm_ns * 100) / affine_ns;
    std.debug.print("exl3 512-row E=512 H=2560 I=640 topk=10: layer {d} us  affine-3x-gather_qmm {d} us  ratio {d}/100\n", .{
        gemm_ns / 1000,
        affine_ns / 1000,
        ratio_x100,
    });
    try t.expect(ratio_x100 <= 200);
}
