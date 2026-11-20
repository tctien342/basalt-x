# MMQ kernels

This directory carries the vendored ggml CUDA quantizer and matrix products used by Basalt's routed prefill. `quantize.cu` packs rows for the scale layout selected by the weight type. The optional SwiGLU loader consumes gate and up rows on the same CUDA stream and uses the shared packing body. Callers provide device buffers and retain their ownership until stream completion.
