"""Pack Inco Flash-Next DFlash2 projections as GGML Q4_0, with BF16 selector tables at the end."""
import json, struct, sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit("usage: dflash2_pack.py model.safetensors output-directory")
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

N = 2560
Q4 = ["fc.weight", "candidate_selector.hidden_projection.weight"]
for l in range(5):
    p = f"layers.{l}."
    Q4 += [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
           p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
           p + "mlp.down_proj.weight", p + "attention_conv.kernel_projection.weight",
           p + "mlp_conv.kernel_projection.weight"]
F32 = ["hidden_norm.weight", "norm.weight"]
for l in range(5):
    p = f"layers.{l}."
    F32 += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight",
            p + "attention_conv.base_kernel", p + "mlp_conv.base_kernel"]
BF16 = ["candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"]

hdr = None
with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

def load(name):
    info = hdr[name]
    dt = info["dtype"]; shape = info["shape"]
    with open(SRC, "rb") as f:
        f.seek(base + info["data_offsets"][0])
        raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dt == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise ValueError((name, dt))

def q4_0(array):
    blocks = array.reshape(-1, 32).astype(np.float32)
    largest = np.take_along_axis(blocks, np.argmax(np.abs(blocks), axis=1)[:, None], axis=1)
    scale = largest / -8.0
    norm = np.divide(blocks, scale, out=np.zeros_like(blocks), where=scale != 0)
    quants = np.clip((norm + 8.5).astype(np.int16), 0, 15).astype(np.uint8)
    out = np.empty((blocks.shape[0], 18), dtype=np.uint8)
    out[:, :2] = scale.astype('<f2').view(np.uint8)
    out[:, 2:] = quants[:, :16] | (quants[:, 16:] << 4)
    return out.tobytes()

idx = []
blob = bytearray()
def put(name, kind, arr_bytes, rows, cols):
    idx.append(f"{name} {kind} {rows} {cols} {len(blob)} {len(arr_bytes)}")
    blob.extend(arr_bytes)

for name in Q4:
    a = load(name)
    assert a.shape[1] % 32 == 0, (name, a.shape)
    put(name, "q4_0", q4_0(a), a.shape[0], a.shape[1])
for name in F32:
    a = load(name).astype(np.float32)
    rows, cols = (a.shape[0], a.shape[1]) if a.ndim == 2 else (a.size, 1)
    put(name, "f32", np.ascontiguousarray(a).tobytes(), rows, cols)
for name in BF16:
    a = load(name)
    u = a.astype(np.float32).view(np.uint32) >> 16
    put(name, "bf16", u.astype(np.uint16).tobytes(), a.shape[0], a.shape[1])

(OUT / "dense.bin").write_bytes(bytes(blob))
(OUT / "dense.txt").write_text("\n".join(idx) + "\n")
cfg = json.loads((SRC.parent / "config.json").read_text())
dc = cfg["dflash_config"]
(OUT / "config.json").write_text(json.dumps({
    "arch": "dflash2", "hidden": cfg["hidden_size"], "layers": cfg["num_hidden_layers"],
    "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"],
    "intermediate": cfg["intermediate_size"], "vocab": cfg["vocab_size"],
    "block_size": dc["block_size"], "mask_token_id": dc["mask_token_id"],
    "conv_kernel": dc["conv_kernel_size"], "conv_group": dc["conv_group_size"],
    "selector_rank": dc["selector_rank"], "selector_top_k": dc["selector_top_k"],
    "target_layers": dc["target_layer_ids"], "swa": cfg.get("sliding_window", 0),
    "rms_eps": cfg["rms_norm_eps"], "rope_theta": cfg["rope_parameters"]["rope_theta"],
}, indent=1) + "\n")
print(f"{OUT}: {len(blob)} bytes, {len(idx)} tensors")
