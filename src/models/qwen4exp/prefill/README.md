# Module notes

Target and draft prefill use the same value basis as their decode paths. Each target output returns to the original basis before gating.

Asynchronous stage handoffs capture the launch thread’s RoPE context and restore any enclosing scope on completion.

A stage can bind Swift's immutable device factors and graph-stable scale. Dense prefill dispatch adds the Swift delta from the original FP16 activation rows after the base projection, reusing the dequantization scratch on the same stream. The caller drains outstanding stage work before changing the scale.

Flash-Next prefill owns the temporary stage pipeline. Direct hand-off waits for the destination owner and preserves the host draft relay. Primary and peer routed experts use the shared SwiGLU-to-Q8_1 MMQ packer.

Successful initial layout and relayout record bytes carved for chunk and GEMM work. CUDA and host allocations outside that carve have their own lifetimes.

In the overlap experiment, FP16 and INT8 QSA stage cached pages on the copy stream and lets QSA selection run during the copy, then waits before appending and attending. An activation handoff uses the existing host slot while the peer still owns its destination.

The peer_cold_mmq experiment uses two peer MMQ stage buffers for short IQ4_XS prompts. CPU assignments and resident peer work stay fixed. Selected pinned host misses upload directly into the peer group layout and return through the existing peer result fence.

The `peer_cold_mmq_ranked` experiment moves complete groups for supported native MMQ layers outside the existing IQ4_XS route. It prioritizes avoided primary uploads and accepts a group only when the estimated PCIe bottleneck falls. The original group geometry and old FP16 peer rows are retained. Newly moved rows return in FP32.

Direct MMQ prefill can return FP16 peer results through the existing row interval. The combine widens those rows while preserving its FP32 accumulation order.

Opt-in packed transport reuses inactive mixed scratch for the direct layer handoff and draft relay. A CPU expert can consume the GPU-generated AVX2 Q8_K representation with a float fallback for nonfinite inputs.

`peer_cold_copy_balance` changes only where eligible cold experts are staged for short IQ4_XS prompts. It estimates both H2D paths and the packed peer return using measured rates and accepts moves that lower the estimated slower path. Router output, resident and CPU assignments, and the staging fences retain their existing contracts.

`route_shared_overlap` can record a completion event after the route read for short grouped prefill. Host grouping waits for that event while the GPU executes the shared expert on its existing stream. Routed products and the final combine keep their original data dependencies.

`peer_ahead_cap` limits speculative primary uploads to 32 experts for short prefill with a direct cold MMQ peer. The existing ring geometry and slot lifetimes stay fixed. Expert routing and demand loads use their existing paths.

The opt-in `qsa_selective_stage` path stages selected FP16 or INT8 historical pages before appending new cells for eligible cached chunks of 64 to 2560 tokens. Chunks up to 640 tokens can use it after 32768 historical tokens. Chunks up to 256 tokens can use it after 16384 historical tokens. It reuses score scratch after selection and leaves attention and the resident map unchanged. Other prompt shapes keep dense staging.

With `qsa_identity_index`, prefill skips score launches when a selection batch covers every available token. It also skips query-index projection when this holds for the whole chunk. Index key projection and append, ascending selection IDs and attention are retained.

With DFlash2, the completed residuals at five tap layers stage the current prompt tail for draft KV injection. Each stage writes its own tap rows into the same pinned host ring.

The optional AtomicChat PLE value delta is applied after projection and before PLE postprocessing. Its factors and scale are bound per device before graph capture.

Native prefill projections can apply a rank-one delta from the same FP16 activation rows used by GEMM. The optional adapter is bound to the GEMM context and selected by the writer name.

The prefill context can bind a per-device LoRA adapter after initialization. Local routed experts apply their down delta to the grouped result before the router combine. The FP16 path uses the same intermediate as its base down GEMM. The MMQ path reconstructs SwiGLU in existing scratch before applying the delta.

The optional peer MMQ post-down callback receives the original expert-list offset and runs on the peer producer stream before publishing the result. Flash-Next applies the LoRA delta here before FP16 packing or FP32 transport.

Optional CPU LoRA jobs borrow host factors. The pool computes one input-factor dot per routed SwiGLU row before activation quantization and adds the delta to the down output before publication.

The base alias keeps its fused streamed expert kernels. Adapted streamed experts use MMQ, which exposes the SwiGLU rows used by the LoRA correction.

An optional device page table translates sequence-local FP16 host KV blocks to physical host backing in append, residency fill and prefill staging. The caller owns the table and backing and must keep both live until their CUDA work drains. The fixed-session path uses identity addressing.

Direct MTP prefill checks the draft destination range before launching GPU writes that could overlap a frozen recovery tail.
