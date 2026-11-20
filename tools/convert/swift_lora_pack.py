#!/usr/bin/env python3
"""Convert the Flash-Next Swift rank-32 factors to Basalt projection order."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from checkpoint import LAYER
from model_sections import GDN_QK_ROWS, gdn_index, read_safetensors_header

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_writer import GGUFWriter
from model_file import ModelFile


KEY = re.compile(r"base_model\.model\.model\.language_model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight")
RANK = 32
PROJECTIONS = {
    "attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
}
WEIGHT = re.compile(r"blk\.(?:[0-9]|[1-3][0-9]|4[0-7])\.([^.]+)\.weight")


def base_shapes(path):
    pack = ModelFile(str(path))
    meta = json.loads(pack.read("meta"))["gguf_kv"]
    if meta["general.architecture"] != "qwen4exp" or meta["qwen4exp.block_count"] != 48:
        raise ValueError("expected a 48-layer Flash-Next base pack")
    expected = {}
    for line in pack.read("dense.index").decode().splitlines():
        cols = line.split()
        if cols and (match := WEIGHT.fullmatch(cols[0])) and match[1] in PROJECTIONS:
            expected[cols[0]] = tuple(map(int, cols[7:9]))
    if len(expected) != 300:
        raise ValueError(f"expected 300 base projections, found {len(expected)}")
    return expected


def convert(source, model, output):
    config = json.loads((source / "adapter_config.json").read_text())
    if (config["base_model_name_or_path"], config["peft_type"], config["r"], config["lora_alpha"]) != (
        "Qwen/Qwen3.8-Flash-Next", "LORA", RANK, RANK
    ) or any(config.get(key) for key in (
        "use_rslora", "use_dora", "fan_in_fan_out", "lora_bias", "modules_to_save", "rank_pattern", "alpha_pattern"
    )) or config.get("bias") != "none":
        raise ValueError("expected unscaled Swift rank-32 PEFT factors")

    path = source / "adapter_model.safetensors"
    header, data_start = read_safetensors_header(path)
    writer = GGUFWriter()
    writer.add("general.architecture", "qwen4exp")
    writer.add("general.type", "adapter")
    writer.add("adapter.type", "lora")
    writer.add("adapter.lora_alpha", float(RANK), "f32")

    missing = {(name, side): (RANK, inputs) if side == "A" else (outputs, RANK)
               for name, (inputs, outputs) in base_shapes(model).items() for side in ("A", "B")}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        match = KEY.fullmatch(key)
        if match is None or match[2] + ".weight" not in LAYER:
            raise ValueError(f"unexpected Swift tensor: {key}")
        suffix, _, layout = LAYER[match[2] + ".weight"]
        name, side = f"blk.{int(match[1])}.{suffix}", match[3]
        shape = missing.pop((name, side), None)
        start, end = entry["data_offsets"]
        if shape != tuple(entry["shape"]) or entry["dtype"] != "BF16" or end - start != 2 * math.prod(shape):
            raise ValueError(f"unexpected Swift factor shape or dtype: {key}")

        bits = np.fromfile(path, dtype="<u2", count=math.prod(shape), offset=data_start + start).reshape(shape)
        if np.any((bits & 0x7f80) == 0x7f80):
            raise ValueError(f"nonfinite Swift factor: {key}")
        if side == "B" and layout in ("vrows", "zrows"):
            bits = bits[gdn_index(shape[0], GDN_QK_ROWS if layout == "vrows" else 0), :]
        elif side == "A" and layout == "vcols":
            bits = bits[:, gdn_index(shape[1], 0)]
        writer.add_raw(f"{name}.lora_{side.lower()}", list(shape[::-1]), "BF16", bits.tobytes())

    if missing:
        raise ValueError(f"missing Swift factors: {sorted(missing)}")
    writer.write(output)
    print(f"{len(writer.tensors)} BF16 factors in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("base_pack", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.adapter_dir, args.base_pack, args.output)
