const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const exl3 = @import("expert_exl3.zig");

pub const DECODE_ROWS_MAX: usize = 16;

pub fn usesPrefillArm(rows: usize) bool {
    return rows > DECODE_ROWS_MAX;
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
    \\    const uint mixed = uint(cw[slot]) * 0xCBAC1FEDu;
    \\    const uint pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    \\    const half lo = as_type<half>(ushort(pair));
    \\    const half hi = as_type<half>(ushort(pair >> 16u));
    \\    const float w = float(half(float(lo) + float(hi)));
    \\    acc += float(x[tk * TILE + (pos / 16u)]) * w;
    \\  }
    \\}
    \\y[gid] = half(acc);
;

const PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.z);
    \\uint i = uint(thread_position_in_threadgroup.x);
    \\if (i >= 128u) return;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\float acc = 0.0f;
    \\for (uint k = 0u; k < 128u; k++) {
    \\  const uint bits = popcount(uint(i & k));
    \\  const float sign = (bits & 1u) ? -1.0f : 1.0f;
    \\  acc += sign * 0.08838834764831845f * float(x[(size_t)slot * (size_t)(IDIM) + base + k]) * float(suh[eid * uint(IDIM) + base + k]);
    \\}
    \\y[(size_t)slot * (size_t)(IDIM) + base + i] = half(acc);
;

const FINISH_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.z);
    \\uint i = uint(thread_position_in_threadgroup.x);
    \\if (i >= 128u) return;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\float acc = 0.0f;
    \\for (uint k = 0u; k < 128u; k++) {
    \\  const uint bits = popcount(uint(i & k));
    \\  const float sign = (bits & 1u) ? -1.0f : 1.0f;
    \\  acc += sign * 0.08838834764831845f * float(inner[(size_t)slot * (size_t)(ODIM) + base + k]);
    \\}
    \\y[(size_t)slot * (size_t)(ODIM) + base + i] = half(acc * float(svh[eid * uint(ODIM) + base + i]));
;

var prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var finish_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_engaged: bool = false;

const GemvKey = struct { in_dim: c_int, out_dim: c_int };
var gemv_cfgs: [8]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var gemv_keys: [8]GemvKey = @splat(.{ .in_dim = 0, .out_dim = 0 });

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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * blocks, 1, topk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * blocks, 1, topk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
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
    \\    const uint mixed = uint(cw[si]) * 0xCBAC1FEDu;
    \\    const uint pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    \\    const half lo = as_type<half>(ushort(pair));
    \\    const half hi = as_type<half>(ushort(pair >> 16u));
    \\    const float w = float(half(float(lo) + float(hi)));
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
    const g = try projectIndexed(s, xrep, gate_t, gate_suh, gate_svh, slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try projectIndexed(s, xrep, up_t, up_suh, up_svh, slots);
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
    var t_rows = io_util.Stopwatch.init(t.io);
    const rows_out = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(rows_out));
    const rows_ns = t_rows.read();
    _ = mlx.mlx_array_free(rows_out);
    var t_mm = io_util.Stopwatch.init(t.io);
    const mm_out = try prefillDecodeGatherMm(s, x_arr, pub_a, pub_a, pub_a, slots2, scores, R, dim, dim, topk);
    try mlx.check(mlx.mlx_array_eval(mm_out));
    const mm_ns = t_mm.read();
    _ = mlx.mlx_array_free(mm_out);
    std.debug.print("exl3 512-row synthetic E=4 H=128 topk=2: rows-kernel {d} ms  decode+gather_mm {d} ms\n", .{
        rows_ns / 1_000_000,
        mm_ns / 1_000_000,
    });
    try t.expect(rows_ns > 0);
    try t.expect(mm_ns > 0);
}
