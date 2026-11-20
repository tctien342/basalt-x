# Model conversion tools

This directory contains the Python converters for Basalt model packs and their component artifacts. The converters read source model files and write separate output packs. They do not modify the input checkpoint.

Run the Flash-Next DFlash2 converter with NumPy installed:

    python3 tools/convert/dflash2_pack.py /path/model.safetensors /path/draft-pack

It reads the pinned Inco checkpoint and its adjacent config.json. The output directory contains the tensor manifest and weights. Q4_0 projections and FP32 norms come first. The BF16 selector tables follow and are read into host memory only.

Inspect the AtomicChat rank-one adapter and its base pack before runtime integration:

    python3 tools/convert/inspect_atomic_lora.py /path/adapter.gguf /path/model.basalt

The checker reports SHA-256, validates the full FP32 A/B site set and verifies that the output factor is identical across every routed expert and dense writer. It reads the existing GGUF and Basalt indexes and does not write or load model weights into an engine.

Convert the Swift rank-32 Flash-Next adapter with NumPy installed:

    python3 tools/convert/swift_lora_pack.py /path/peft-adapter /path/model.basalt /path/swift.gguf

The converter checks all factor shapes against the base pack, preserves BF16, and orders the GDN value heads to match the deployed projection layouts. It writes a separate GGUF adapter.
