# Vision interfaces

These headers expose multimodal projector weights, preprocessing, image and
video input, rotary positions, and CPU and CUDA embedding encoders. See
`src/vision` for their implementation.

A caller using `CudaEncoder::attach` owns its CUDA arena. Before reclaiming that
arena, it must check that `CudaEncoder::detach` successfully fenced the encoder
stream. The independently allocated `init` path owns its memory.
