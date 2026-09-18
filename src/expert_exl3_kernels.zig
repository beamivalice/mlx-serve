const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const exl3 = @import("expert_exl3.zig");

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
    \\threadgroup float W[256];
    \\threadgroup float Xs[8 * 16];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint run = uint(threadgroup_position_in_grid.y);
    \\uint lane = uint(thread_index_in_threadgroup);
    \\if (run >= uint(NRUN)) return;
    \\constexpr uint TILE = 16u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint ROWS = 4u;
    \\const uint start = run_start[run];
    \\const uint rlen = run_len[run];
    \\const uint eid = run_eid[run];
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
    \\for (uint r0 = 0u; r0 < rlen; r0 += ROWS) {
    \\  const uint n = min(ROWS, rlen - r0);
    \\  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\  if (lane < 32u) { Xs[lane] = 0.0f; Xs[32u + lane] = 0.0f; Xs[64u + lane] = 0.0f; Xs[96u + lane] = 0.0f; }
    \\  for (uint tk = 0u; tk < IT; tk++) {
    \\    const device ushort* tile = trellis + (((size_t)eid * (size_t)IT + tk) * (size_t)OT + ot) * 64u;
    \\    const device uint* words = (const device uint*)tile;
    \\    const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\    const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\    for (uint s = 0u; s < 8u; s++) {
    \\      const uint cw = uint(merged >> sh[s]) & 0xffffu;
    \\      const uint mixed = cw * 0x83DCD12Du;
    \\      const uint pair_sums = (mixed & 0x00FF00FFu) + ((mixed >> 8u) & 0x00FF00FFu);
    \\      const uint byte_sum = 0x6400u + (pair_sums & 0xFFFFu) + (pair_sums >> 16u);
    \\      const half hh = as_type<half>(ushort(byte_sum));
    \\      const half inv = as_type<half>(ushort(0x1EEEu));
    \\      const half bias = as_type<half>(ushort(0xC931u));
    \\      W[pos[s]] = float(fma(hh, inv, bias));
    \\    }
    \\    for (uint t = 0u; t < 4u; t++) {
    \\      const uint lin = lane + t * 32u;
    \\      const uint rr = lin / 16u;
    \\      const uint kk = lin % 16u;
    \\      if (rr < n) Xs[rr * 16u + kk] = float(x[(size_t)(start + r0 + rr) * (size_t)(IDIM) + tk * TILE + kk]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (lane < 16u) {
    \\      float4 a = float4(acc[0], acc[1], acc[2], acc[3]);
    \\      for (uint k = 0u; k < TILE; k++) {
    \\        const float w = W[k * TILE + lane];
    \\        a = fma(float4(Xs[k], Xs[16u + k], Xs[32u + k], Xs[48u + k]), float4(w), a);
    \\      }
    \\      acc[0] = a.x; acc[1] = a.y; acc[2] = a.z; acc[3] = a.w;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\  if (lane < 16u) {
    \\    for (uint r = 0u; r < n; r++) {
    \\      y[(size_t)(start + r0 + r) * (size_t)(ODIM) + ot * TILE + lane] = half(acc[r]);
    \\    }
    \\  }
    \\}
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
var gemv_cfgs: [8]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var gemv_keys: [8]GemvKey = @splat(.{ .in_dim = 0, .out_dim = 0 });

fn getGemmSortedKernel() !mlx.mlx_fast_metal_kernel {
    if (gemm_sorted_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis", "run_start", "run_len", "run_eid" };
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
    runs: RunTable,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 2 or tsh.len != 4 or runs.n == 0) return error.BadExl3Shape;
    const n = xsh[0];
    const in_dim = xsh[1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const start_a = mlx.mlx_array_new_data(runs.start.ptr, &[_]c_int{@intCast(runs.n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(start_a);
    const len_a = mlx.mlx_array_new_data(runs.len.ptr, &[_]c_int{@intCast(runs.n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(len_a);
    const eid_a = mlx.mlx_array_new_data(runs.eid.ptr, &[_]c_int{@intCast(runs.n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, out_dim }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_tiles * 32, @intCast(runs.n), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NRUN", @intCast(runs.n)));
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const inputs = [_]mlx.mlx_array{ x, trellis, start_a, len_a, eid_a };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGemmSortedKernel(), inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    try mlx.check(mlx.mlx_array_eval(out));
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
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{ topk, in_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * blocks, topk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    return applyUnary(s, try getPrepareKernel(), &.{ x, suh, slots }, cfg);
}

pub fn finishIndexed(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(inner);
    const topk = sh[0];
    const out_dim = sh[1];
    const blocks = @divExact(out_dim, 128);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{ topk, out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * blocks, topk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
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

fn gemvConfig(in_dim: c_int, out_dim: c_int) !mlx.mlx_fast_metal_kernel_config {
    for (gemv_cfgs, 0..) |c, i| {
        if (c != null and gemv_keys[i].in_dim == in_dim and gemv_keys[i].out_dim == out_dim) return c.?;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{out_dim};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    var slot: usize = 0;
    while (slot < gemv_cfgs.len and gemv_cfgs[slot] != null) : (slot += 1) {}
    if (slot == gemv_cfgs.len) slot = 0;
    if (gemv_cfgs[slot]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
    gemv_cfgs[slot] = cfg;
    gemv_keys[slot] = .{ .in_dim = in_dim, .out_dim = out_dim };
    return cfg;
}

const INDEXED_SOURCE: [:0]const u8 =
    \\uint gid = uint(thread_position_in_grid.x);
    \\uint slot = uint(thread_position_in_grid.z);
    \\if (gid >= uint(ODIM)) return;
    \\constexpr uint TILE = 16u;
    \\constexpr uint IN_TILES = uint(IDIM) / TILE;
    \\constexpr uint OUT_TILES = uint(ODIM) / TILE;
    \\const uint eid = uint(slots[slot]);
    \\const device ushort* trellis_e = trellis + eid * IN_TILES * OUT_TILES * 64u;
    \\const uint tn = gid / TILE;
    \\const uint local = gid % TILE;
    \\float acc = 0.0f;
    \\for (uint tk = 0u; tk < IN_TILES; tk++) {
    \\  const device ushort* tile = trellis_e + (tk * OUT_TILES + tn) * 64u;
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
    \\  for (uint si = 0u; si < 256u; si++) {
    \\    const uint lane = si / 8u;
    \\    const uint s = si % 8u;
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
    \\    const uint mixed = uint(cw[si]) * 0x83DCD12Du;
    \\    const uint pair_sums = (mixed & 0x00FF00FFu) + ((mixed >> 8u) & 0x00FF00FFu);
    \\    const uint byte_sum = 0x6400u + (pair_sums & 0xFFFFu) + (pair_sums >> 16u);
    \\    const half hh = as_type<half>(ushort(byte_sum));
    \\    const half inv = as_type<half>(ushort(0x1EEEu));
    \\    const half bias = as_type<half>(ushort(0xC931u));
    \\    const float w = float(fma(hh, inv, bias));
    \\    acc += float(x[(size_t)slot * (size_t)(IDIM) + tk * TILE + (pos / 16u)]) * w;
    \\  }
    \\}
    \\y[(size_t)slot * (size_t)(ODIM) + gid] = half(acc);
;

var indexed_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getIndexedKernel() !mlx.mlx_fast_metal_kernel {
    if (indexed_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "trellis", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_k4_mcg_gemv_indexed",
        in_vec,
        out_vec,
        INDEXED_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    indexed_kernel = kernel;
    return kernel;
}

pub fn indexedGemvF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const topk = ssh[0];
    if (tsh[1] * 16 != in_dim or tsh[3] != 64) return error.BadExl3Shape;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{ topk, out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, topk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getIndexedKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub fn projectIndexed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x, suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try indexedGemvF16(s, prepared, trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots);
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

fn projectSorted(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, alloc: std.mem.Allocator) !mlx.mlx_array {
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
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, sorted_slots, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const n: usize = @intCast(mlx.getShape(slots)[0]);
    const ids = mlx.mlx_array_data_uint32(contig) orelse return error.U32Unreadable;
    const runs = try buildRuns(alloc, ids[0..n]);
    const inner = try innerGemmSorted(s, prepared, trellis, runs);
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
    runs: RunTable,
) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x_sorted, suh, slots_sorted);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, runs);
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
    const xrep = try repeatRows(s, x, rows, topk);
    defer _ = mlx.mlx_array_free(xrep);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    var sorted_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_x);
    try mlx.check(mlx.mlx_take_axis(&sorted_x, xrep, order, 0, s));
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, sorted_slots, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const n: usize = @intCast(mlx.getShape(slots)[0]);
    const ids = mlx.mlx_array_data_uint32(contig) orelse return error.U32Unreadable;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const runs = try buildRuns(arena.allocator(), ids[0..n]);
    const g = try projectSortedWithRuns(s, sorted_x, gate_t, gate_suh, gate_svh, sorted_slots, runs);
    defer _ = mlx.mlx_array_free(g);
    const u = try projectSortedWithRuns(s, sorted_x, up_t, up_suh, up_svh, sorted_slots, runs);
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
    const d_sorted = try projectSortedWithRuns(s, h, down_t, down_suh, down_svh, sorted_slots, runs);
    defer _ = mlx.mlx_array_free(d_sorted);
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_take_axis(&d, d_sorted, inv, 0, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, .float16, s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, sc, &[_]c_int{ rows, topk, 1 }, 3, s));
    var d3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d3);
    try mlx.check(mlx.mlx_reshape(&d3, d, &[_]c_int{ rows, topk, hidden }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d3, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 1, false, s));
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
    const io_util = @import("io_util.zig");
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
    const io_util = @import("io_util.zig");
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
