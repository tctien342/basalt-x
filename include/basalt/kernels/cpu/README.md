# CPU expert interfaces

These headers expose CPU expert quantization, layout, dot kernels and the worker pool. Implementations live in `src/kernels/cpu`. `ExpertJobMulti` borrows its activations, output rows and optional LoRA factors until the worker batch completes. The pool computes the low-rank scalar once per SwiGLU row and applies it to disjoint down-projection rows.

Grouped CPU experts can borrow the decode window adapter scale vector. Dispatch copies each original token’s scale into its expert job row before starting the workers.
