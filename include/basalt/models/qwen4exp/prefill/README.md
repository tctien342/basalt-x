# Module notes

A stage can bind Swift's immutable device factors and graph-stable scale. Dense prefill dispatch adds the Swift delta from the original FP16 activation rows after the base projection, reusing the dequantization scratch on the same stream. The caller drains outstanding stage work before changing the scale.

Flash-Next prefill interfaces. A successor retains its host hand-off slot for draft relay even when activations land directly in its device rows.

Workspace bytes describe the carved chunk workspace. The borrowed flag tells telemetry when those bytes are already part of another allocation.

The FP16 peer combine reads only the peer interval in the packed format. Local and CPU expert rows retain their float layout.

The packed transport helpers encode CPU Q8_K input or convert FP16 residual rows on the caller's stream. The draft relay overload can read compact pinned FP16 rows.

The optional DFlash2 sink captures only the current prompt tail and supports the temporary peer prefill stages.

Native prefill projections can apply a rank-one delta from the same FP16 activation rows used by GEMM. The optional adapter is bound to the GEMM context and selected by the writer name.

The prefill context can bind a per-device LoRA adapter after initialization. Local routed experts apply their down delta to the grouped result before the router combine. The FP16 path uses the same intermediate as its base down GEMM. The MMQ path reconstructs SwiGLU in existing scratch before applying the delta.

The optional peer MMQ post-down callback receives the original expert-list offset and runs on the peer producer stream before publishing the result. Flash-Next applies the LoRA delta here before FP16 packing or FP32 transport.

Optional CPU LoRA jobs borrow host factors. The pool computes one input-factor dot per routed SwiGLU row before activation quantization and adds the delta to the down output before publication.

Adapter-enabled sizing reserves full MMQ scratch alongside the streamed fused layout.
