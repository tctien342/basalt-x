# Module notes

Shared routed prefill kernels and peer result transport. Direct peer output preserves the existing producer and consumer events. Routed MMQ packs SwiGLU directly into the down projection input on the producer stream.

The opt-in direct Q8 path packs the layer into main-card scratch and copies the exact slab to peer-card scratch when routed rows exceed source tokens. It uses the existing readiness and reuse events. Custom products, sparse routes and unavailable peer access use mapped host activations.

The optional FP16 result path packs on the peer output stream before DMA. Its single scratch buffer uses the same stream for packing and copying. The caller can query the last run's format.

A caller may restrict direct packing to a tail of the peer result interval. A launch crossing the split returns a full-precision prefix and a compact tail in order on the existing output stream. The original readiness and reuse events cover both ranges.

The optional peer MMQ post-down callback receives the original expert-list offset and runs on the peer producer stream before publishing the result. Flash-Next applies the LoRA delta here before FP16 packing or FP32 transport.
