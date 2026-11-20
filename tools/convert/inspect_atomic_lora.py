#!/usr/bin/env python3
"""Check the AtomicChat rank-one adapter against a Flash-Next Basalt pack."""

import argparse
import hashlib
import json
import mmap
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
from model_file import ModelFile

DENSE_WRITER = re.compile(r"blk\.\d+\.(ffn_down_shexp|ssm_out|attn_output|ple_value)\.weight")
EMBEDDING = "token_embd.weight"


def expected_sites(pack):
    meta = json.loads(pack.read("meta"))["gguf_kv"]
    if meta["general.architecture"] != "qwen4exp":
        raise ValueError("expected a Flash-Next qwen4exp base pack")
    dims = {}
    for line in pack.read("dense.index").decode().splitlines():
        if line and not line.startswith("#"):
            cols = line.split()
            if cols[0] == EMBEDDING or DENSE_WRITER.fullmatch(cols[0]):
                dims[cols[0]] = tuple(map(int, cols[7:9]))
    block_count = meta["qwen4exp.block_count"]
    experts = meta["qwen4exp.expert_count"]
    hidden = meta["qwen4exp.embedding_length"]
    expert_hidden = meta["qwen4exp.expert_feed_forward_length"]
    if block_count != 48 or len(dims) != 98 or EMBEDDING not in dims:
        raise ValueError("unexpected Flash-Next writer layout")
    for layer in range(block_count):
        dims[f"blk.{layer}.ffn_down_exps.weight"] = (expert_hidden, hidden, experts)
    return dims


def inspect(adapter_path, model_path):
    pack = ModelFile(str(model_path))
    sites = expected_sites(pack)
    adapter = GGUFFile(adapter_path)
    if tuple(adapter.metadata.get(key) for key in
             ("general.architecture", "general.type", "adapter.type", "adapter.lora_alpha")) != (
                 "qwen4exp", "adapter", "lora", 1.0):
        raise ValueError("expected a qwen4exp rank-one LoRA with alpha 1")
    expected = {}
    for name, (inputs, outputs, *experts) in sites.items():
        a = (1, outputs) if name == EMBEDDING else (inputs, 1, *experts)
        expected[name + ".lora_a"] = a
        expected[name + ".lora_b"] = (1, inputs if name == EMBEDDING else outputs, *experts)

    size = adapter_path.stat().st_size
    tensors = sorted(adapter.tensors, key=lambda item: item.offset)
    previous_end = adapter.data_start
    with adapter_path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        digest = hashlib.sha256(data).hexdigest()
        output = None
        output_count = 0
        for tensor in tensors:
            shape = expected.pop(tensor.name, None)
            if shape != tuple(tensor.shape) or tensor.type_id != 0:
                raise ValueError(f"unexpected tensor, shape or type: {tensor.name}")
            start = adapter.data_start + tensor.offset
            end = start + 4 * tensor.elements
            if start < previous_end or end > size or tensor.offset % adapter.alignment:
                raise ValueError(f"invalid tensor span: {tensor.name}")
            previous_end = end
            # Construct an ndarray over an ordinary bytes slice so errors can close the mapping.
            values = np.frombuffer(data[start:end], dtype="<f4")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values: {tensor.name}")
            if tensor.name.endswith(".lora_b"):
                rows = values.view("<u4").reshape(-1, shape[1])
                if output is None:
                    output = rows[0].copy()
                if not np.all(rows == output):
                    raise ValueError(f"output factor differs from the shared vector: {tensor.name}")
                output_count += len(rows)
    if expected:
        raise ValueError(f"missing adapter tensor: {next(iter(expected))}")
    return dict(adapter_sha256=digest, base_meta_sha256=hashlib.sha256(pack.read("meta")).hexdigest(),
                adapter_bytes=size, sites=len(sites), tensors=len(tensors), output_factor_vectors=output_count,
                unique_output_factor_vectors=1, shared_output_bytes=int(output.nbytes),
                compact_factor_bytes=sum(4 * t.elements for t in tensors if t.name.endswith(".lora_a"))
                + int(output.nbytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.adapter, args.model), indent=2))


if __name__ == "__main__":
    main()
