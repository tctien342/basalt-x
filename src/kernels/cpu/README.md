# CPU expert kernels

`native_expert.cpp` dispatches GGUF expert projections and activation quantization. The IQ, K-quant, NVFP4 and Q2 translation units contain ISA-specific row kernels. `pool.cpp` schedules CPU expert jobs. `expert_layout.cpp` derives row shapes and CPU capabilities. The extracted ggml dot and quantizer code lives in `ggml/`.

The entry points are declared under `include/basalt/kernels/cpu`. Keep per-token arithmetic independent of batch width. `BASALT_CPU_Q5_MULTI=1` enables the experimental paired Q5_0 down projection on AVX2 systems without AVX-512. It reads the original packed weights and writes the existing output row ranges. Expert ownership, row mapping and buffer lifetimes are handled by the caller.

`BASALT_CPU_AVX2_TASKS` accepts 4 or 12 in isolated partition experiments. The default remains eight tasks per worker. AVX-512 uses its own existing partition factor.

Optional CPU LoRA jobs borrow host factors. The pool computes one input-factor dot per routed SwiGLU row before activation quantization and adds the delta to the down output before publication.

Grouped CPU experts can borrow the decode window adapter scale vector. Dispatch copies each original token’s scale into its expert job row before starting the workers.
