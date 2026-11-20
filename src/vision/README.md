# Vision encoders

The vision pipeline loads multimodal projector weights with `mmproj.cpp`,
prepares images in `preprocess.cpp` and video in `video.cpp`, and computes
embeddings in `encoder_cpu.cpp` or `encoder_cuda.cu`. CPU matrix products and
CUDA kernels live beside their respective encoders. `mrope_positions.cpp`
assigns text and image positions for consumers of the embeddings.

The serving worker can stage weights in pinned host memory and attach the CUDA
encoder to a temporary expert-cache region. Before the region can be reused,
`CudaEncoder::detach` fences its private stream and returns the fence status.
The caller owns the region and must keep it unavailable to other work if the
fence fails.
