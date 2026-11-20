# Prefill interfaces

Headers in this directory expose shared prefill and expert transport operations to model-specific callers. `moe_mmq.hpp` defines routed products and activation packing. `peer_share.hpp` owns peer launch buffers and accepts a fused packing callback or a custom float activation on the supplied producer stream. Implementations and CUDA dependencies live in `src/prefill`.

Callers that can read packed FP16 results opt in per run and use the actual result format reported by PeerShare.

For mixed direct transport, a caller can request a packed tail. Read its actual row count with `results_fp16_rows()`. The FP32 prefix and the compact FP16 tail each start at their original row address.

The optional peer MMQ post-down callback receives the original expert-list offset and runs on the peer producer stream before publishing the result. Flash-Next applies the LoRA delta here before FP16 packing or FP32 transport.
