#!/usr/bin/env python3
"""Replace routed-expert affine banks in a qwen4_exp 4/8 pack with stacked EXL3 K4.

Input: bf16 HF checkpoint + the existing mixed 4/8 pack. Output: a new pack dir
where every routed gate/up/down bank is EXL3 K4 (MCG), stacked [E, ...] per
projection so gather kernels index expert e on axis 0. Every non-expert file
from the 4/8 pack is hard-linked; mixed shards are rewritten with the expert
tensors dropped and remaining tensors copied as raw bytes. config.json carries
expert_quant = {format: exl3, k: 4, codebook: mcg}.

Default quantization is LDLQ with a captured Hessian; --quantizer direct is
the synthetic-test path. Calibration rows, when captured, are the MLP input
(hidden) for gate/up and the SwiGLU activation (intermediate) for down.

  python3 tests/convert_qwen38_flash_next_exl3.py --self-test
  python3 tests/convert_qwen38_flash_next_exl3.py \\
      --hf /Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next \\
      --pack /Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit \\
      --dst /path/to/out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import write_safetensors_raw  # noqa: E402
from convert_qwen38_flash_next import read_header, read_raw, rename  # noqa: E402

K = 4
CODEBOOK = "mcg"
PACKED_K4 = 256 * K // 16
HAD = 128
MCG_MULT = 0xCBAC1FED

SWITCH = ".mlp.switch_mlp."
HF_GATE_UP = ".mlp.experts.gate_up_proj"
HF_DOWN = ".mlp.experts.down_proj"


def is_pack_expert_key(key: str) -> bool:
    return SWITCH in key


def exl3_keys(base: str) -> tuple[str, str, str]:
    return base + ".trellis", base + ".suh", base + ".svh"


def mlx_switch_base(hf_key: str, proj: str) -> str:
    nk = rename(hf_key)
    if nk.endswith(HF_GATE_UP):
        return nk[: -len("experts.gate_up_proj")] + f"switch_mlp.{proj}_proj"
    if nk.endswith(HF_DOWN):
        return nk[: -len("experts.down_proj")] + "switch_mlp.down_proj"
    raise ValueError(hf_key)


def plan_pack(pack_index: dict) -> dict:
    wm = pack_index["weight_map"]
    files: dict[str, dict[str, list[str]]] = {}
    keep_keys: list[str] = []
    expert_bases: list[str] = []
    for key, fname in wm.items():
        slot = files.setdefault(fname, {"expert": [], "other": []})
        if is_pack_expert_key(key):
            slot["expert"].append(key)
            if key.endswith(".weight"):
                expert_bases.append(key[: -len(".weight")])
        else:
            slot["other"].append(key)
            keep_keys.append(key)
    hardlink, rewrite, drop = [], [], []
    for fname, slot in files.items():
        if slot["expert"] and slot["other"]:
            rewrite.append(fname)
        elif slot["expert"]:
            drop.append(fname)
        else:
            hardlink.append(fname)
    hardlink.sort()
    rewrite.sort()
    drop.sort()
    exl3 = []
    for base in sorted(set(expert_bases)):
        exl3.extend(exl3_keys(base))
    return {
        "hardlink": hardlink,
        "rewrite": rewrite,
        "drop": drop,
        "keep_keys": keep_keys,
        "exl3_keys": exl3,
        "files": files,
    }


def _ensure_lib() -> None:
    lib = os.environ.get("EXL3_CONVERT_LIB", "/Users/beam/llm/ponyexl3")
    if lib not in sys.path:
        sys.path.insert(0, lib)


def _quantize_public(
    public: np.ndarray,
    *,
    k: int,
    codebook: str,
    quantizer: str,
    calibration: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_lib()
    from ponyexl3.convert.direct import quantize_inner_matrix_direct
    from ponyexl3.convert.hessian import block_ldl, capture_hessian, ldlq_inner_matrix, prepare_hessian_for_ldl
    from ponyexl3.convert.regularize import regularize_public_weight
    from ponyexl3.ref.codebook import CodebookMode

    cb = CodebookMode.MCG if codebook == "mcg" else CodebookMode.MUL1
    reg = regularize_public_weight(public.astype(np.float32), seed=seed)
    suh = reg.suh.astype(np.float16)
    svh = reg.svh.astype(np.float16)
    if quantizer == "direct":
        packed, _, _ = quantize_inner_matrix_direct(
            reg.inner, k=k, cb=cb, search_backend="metal", return_states=False
        )
        return packed.astype(np.uint16), suh, svh
    rows = public.shape[0]
    if calibration is None:
        rng = np.random.default_rng(seed + 17)
        acts = rng.standard_normal((256, rows), dtype=np.float32)
    else:
        acts = np.asarray(calibration, dtype=np.float32)
        if acts.shape[1] != rows:
            raise ValueError(f"calibration features {acts.shape[1]} != {rows}")
    prepared = prepare_hessian_for_ldl(capture_hessian(acts))
    ldl = block_ldl(prepared.hessian)
    result = ldlq_inner_matrix(
        reg.inner,
        ldl.l,
        k=k,
        cb=cb,
        hessian=prepared.hessian,
        search_backend="metal",
        collect_states=False,
        compute_proxy=False,
    )
    return result.packed.astype(np.uint16), suh, svh


def _stack_experts(
    bank: np.ndarray,
    *,
    k: int,
    codebook: str,
    quantizer: str,
    calibration: np.ndarray | None,
    seed: int,
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    e, out_dim, in_dim = bank.shape
    trellis_list, suh_list, svh_list = [], [], []
    for ei in range(e):
        public = np.ascontiguousarray(bank[ei].T)
        packed, suh, svh = _quantize_public(
            public, k=k, codebook=codebook, quantizer=quantizer, calibration=calibration, seed=seed + ei
        )
        trellis_list.append(packed)
        suh_list.append(suh)
        svh_list.append(svh)
    trellis = np.stack(trellis_list, axis=0)
    suh = np.stack(suh_list, axis=0)
    svh = np.stack(svh_list, axis=0)
    return {
        "trellis": ("U16", trellis.shape, np.ascontiguousarray(trellis).tobytes()),
        "suh": ("F16", suh.shape, np.ascontiguousarray(suh).tobytes()),
        "svh": ("F16", svh.shape, np.ascontiguousarray(svh).tobytes()),
    }


def _copy_raw_tensors(src_file: Path, keys: list[str]) -> dict:
    header, data_off = read_header(src_file)
    out = {}
    for key in keys:
        meta = header[key]
        raw = read_raw(src_file, data_off, meta)
        out[key] = (meta["dtype"], tuple(meta["shape"]), np.ascontiguousarray(raw).tobytes())
    return out


def convert_pack(
    hf_dir: str | Path,
    pack_dir: str | Path,
    dst: str | Path,
    *,
    quantizer: str = "ldlq",
    k: int = K,
    codebook: str = CODEBOOK,
    calibration: np.ndarray | None = None,
) -> dict:
    hf_dir = Path(hf_dir)
    pack_dir = Path(pack_dir)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    pack_index = json.loads((pack_dir / "model.safetensors.index.json").read_text())
    plan = plan_pack(pack_index)
    for fname in plan["hardlink"]:
        src = pack_dir / fname
        out = dst / fname
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(src, out)
    for name in sorted(os.listdir(pack_dir)):
        if name.startswith("."):
            continue
        if name.startswith("model-") and name.endswith(".safetensors"):
            continue
        if name in ("model.safetensors.index.json", "config.json"):
            continue
        src = pack_dir / name
        if not src.is_file():
            continue
        out = dst / name
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(src, out)
    for i, fname in enumerate(plan["rewrite"]):
        other_keys = plan["files"][fname]["other"]
        tensors = _copy_raw_tensors(pack_dir / fname, other_keys)
        write_safetensors_raw(str(dst / fname), tensors)
    hf_index_path = hf_dir / "model.safetensors.index.json"
    if hf_index_path.exists():
        hf_map = json.loads(hf_index_path.read_text())["weight_map"]
    else:
        keys = list(read_header(hf_dir / "model.safetensors")[0])
        hf_map = {key: "model.safetensors" for key in keys}
    weight_map = {key: pack_index["weight_map"][key] for key in plan["keep_keys"]}
    for key, pack_name in list(weight_map.items()):
        if pack_name in plan["drop"]:
            raise RuntimeError(f"keep key {key} pointed at dropped shard {pack_name}")
    expert_out: dict[str, tuple] = {}
    seed = 0
    for hf_key, hf_file in hf_map.items():
        if hf_key.endswith("experts.gate_up_proj"):
            header, data_off = read_header(hf_dir / hf_file)
            arr = read_raw(hf_dir / hf_file, data_off, header[hf_key])
            if arr.dtype == np.uint16:
                from convert_dsv4_weights import bf16_to_f32
                arr = bf16_to_f32(arr)
            arr = np.asarray(arr, dtype=np.float32)
            half = arr.shape[1] // 2
            gate = np.ascontiguousarray(arr[:, :half])
            up = np.ascontiguousarray(arr[:, half:])
            for proj, bank in (("gate", gate), ("up", up)):
                base = mlx_switch_base(hf_key, proj)
                stacked = _stack_experts(
                    bank, k=k, codebook=codebook, quantizer=quantizer, calibration=calibration, seed=seed
                )
                seed += bank.shape[0]
                for suffix, triple in stacked.items():
                    expert_out[f"{base}.{suffix}"] = triple
        elif hf_key.endswith("experts.down_proj"):
            header, data_off = read_header(hf_dir / hf_file)
            arr = read_raw(hf_dir / hf_file, data_off, header[hf_key])
            if arr.dtype == np.uint16:
                from convert_dsv4_weights import bf16_to_f32
                arr = bf16_to_f32(arr)
            arr = np.asarray(arr, dtype=np.float32)
            base = mlx_switch_base(hf_key, "down")
            stacked = _stack_experts(
                arr, k=k, codebook=codebook, quantizer=quantizer, calibration=calibration, seed=seed
            )
            seed += arr.shape[0]
            for suffix, triple in stacked.items():
                expert_out[f"{base}.{suffix}"] = triple
    shard = "model-exl3-00001.safetensors"
    write_safetensors_raw(str(dst / shard), expert_out)
    for key in expert_out:
        weight_map[key] = shard
    total = 0
    for fname in sorted(set(weight_map.values())):
        total += os.path.getsize(dst / fname)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
    ))
    cfg = json.loads((pack_dir / "config.json").read_text())
    cfg["expert_quant"] = {
        "format": "exl3",
        "k": int(k),
        "codebook": codebook,
        "mcg_multiplier": MCG_MULT,
        "quantizer": quantizer,
        "calibration": (
            "captured-rows" if calibration is not None
            else ("ldlq-gaussian-256" if quantizer == "ldlq" else "none-direct")
        ),
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return plan


class PlanTests(unittest.TestCase):
    def test_pack_expert_keys_are_the_switch_mlp_banks(self):
        self.assertTrue(is_pack_expert_key(
            "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight"))
        self.assertTrue(is_pack_expert_key(
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.scales"))
        self.assertFalse(is_pack_expert_key(
            "language_model.model.layers.3.mlp.shared_expert.down_proj.weight"))
        self.assertFalse(is_pack_expert_key(
            "language_model.model.layers.3.mlp.gate.weight"))

    def test_stacked_dialect_names_three_tensors_per_projection(self):
        trellis, suh, svh = exl3_keys(
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj")
        self.assertEqual(trellis, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis")
        self.assertEqual(suh, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh")
        self.assertEqual(svh, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh")

    def test_plan_classifies_hardlink_rewrite_and_drop(self):
        idx = {
            "weight_map": {
                "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
            }
        }
        plan = plan_pack(idx)
        self.assertEqual(plan["hardlink"], ["model-00001.safetensors"])
        self.assertEqual(plan["drop"], ["model-00002.safetensors"])
        self.assertEqual(plan["rewrite"], ["model-00003.safetensors"])
        self.assertIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis", plan["exl3_keys"])
        self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", plan["keep_keys"])
        self.assertIn("language_model.model.layers.0.mlp.gate.weight", plan["keep_keys"])


class LayoutTests(unittest.TestCase):
    def test_synthetic_pack_layout_and_hardlinks(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf = td / "hf"
            pack = td / "pack"
            dst = td / "out"
            hf.mkdir()
            pack.mkdir()
            e, hidden, inter = 2, 128, 128
            gate_up = rng.standard_normal((e, 2 * inter, hidden)).astype(np.float32)
            down = rng.standard_normal((e, hidden, inter)).astype(np.float32)
            write_safetensors_raw(str(hf / "model.safetensors"), {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                    "F32", gate_up.shape, np.ascontiguousarray(gate_up).tobytes()),
                "model.language_model.layers.0.mlp.experts.down_proj": (
                    "F32", down.shape, np.ascontiguousarray(down).tobytes()),
            })
            (hf / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "text_config": {"num_hidden_layers": 1, "hidden_size": hidden,
                                "moe_intermediate_size": inter, "num_experts": e},
            }))
            (hf / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
                    "model.language_model.layers.0.mlp.experts.down_proj": "model.safetensors",
                }
            }))
            other = rng.standard_normal((4,)).astype(np.float32)
            write_safetensors_raw(str(pack / "model-00001.safetensors"), {
                "language_model.model.embed_tokens.weight": (
                    "F32", other.shape, np.ascontiguousarray(other).tobytes()),
            })
            dummy = np.zeros((e, 8), dtype=np.uint32)
            write_safetensors_raw(str(pack / "model-00002.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
            })
            router = rng.standard_normal((e, hidden)).astype(np.float32)
            write_safetensors_raw(str(pack / "model-00003.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.gate.weight": (
                    "F32", router.shape, np.ascontiguousarray(router).tobytes()),
            })
            (pack / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 1},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
                },
            }))
            (pack / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
                "text_config": {"num_hidden_layers": 1, "hidden_size": hidden,
                                "moe_intermediate_size": inter, "num_experts": e},
            }))
            (pack / "tokenizer.json").write_text("{}")
            convert_pack(hf, pack, dst, quantizer="direct")
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["format"], "exl3")
            self.assertEqual(cfg["expert_quant"]["k"], 4)
            self.assertEqual(cfg["expert_quant"]["codebook"], "mcg")
            self.assertEqual(
                os.stat(dst / "model-00001.safetensors").st_ino,
                os.stat(pack / "model-00001.safetensors").st_ino,
            )
            self.assertEqual(
                os.stat(dst / "tokenizer.json").st_ino,
                os.stat(pack / "tokenizer.json").st_ino,
            )
            self.assertFalse((dst / "model-00002.safetensors").exists())
            idx = json.loads((dst / "model.safetensors.index.json").read_text())
            wm = idx["weight_map"]
            for proj in ("gate", "up", "down"):
                base = f"language_model.model.layers.0.mlp.switch_mlp.{proj}_proj"
                for suffix in (".trellis", ".suh", ".svh"):
                    self.assertIn(base + suffix, wm)
                self.assertNotIn(base + ".weight", wm)
            self.assertIn("language_model.model.layers.0.mlp.gate.weight", wm)
            self.assertEqual(wm["language_model.model.embed_tokens.weight"], "model-00001.safetensors")
            expert_file = wm["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"]
            header, _data_off = read_header(dst / expert_file)
            self.assertEqual(
                tuple(header["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"]["shape"]),
                (e, hidden // 16, inter // 16, PACKED_K4),
            )
            mix_file = wm["language_model.model.layers.0.mlp.gate.weight"]
            mix_header, mix_off = read_header(dst / mix_file)
            self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight", mix_header)
            got = read_raw(dst / mix_file, mix_off, mix_header["language_model.model.layers.0.mlp.gate.weight"])
            np.testing.assert_array_equal(got, router)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--hf", default=None)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--quantizer", default="ldlq", choices=("ldlq", "direct"))
    ap.add_argument("--calibration", default=None)
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if not (args.hf and args.pack and args.dst):
        ap.error("--hf --pack --dst are required unless --self-test")
    cal = None
    if args.calibration:
        cal = np.load(os.path.expanduser(args.calibration))
    convert_pack(args.hf, args.pack, args.dst, quantizer=args.quantizer, calibration=cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
