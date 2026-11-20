# Flash-Next decode

Rotated INT8 scalar decode uses the same value basis at append and inverse transform.

The native dense loader can fill a packed host image using bounded GGUF reads. It binds matrix references to the matching GPU slab offsets and keeps projection scratch on the GPU.

This folder owns the native projection, draft logic and verification modules. The verifier uses CUDA graphs and the existing target session state to accept or reject candidate tokens. See verify/README.md for verifier details.

BASALT_QFN_ROUTE_TRACE optionally writes a diagnostic file of native int32 records: layer, token count, top-k, CPU pool microseconds, all expert IDs and all dispatch kinds. It opens a new file and is intended for synthetic isolated runs.

The optional DFlash2 backend borrows the target projection and embedding. Its Q4_0 weights and bounded FP16 draft KV live on the target decode GPU. The selector codebooks and prompt staging live in host memory. An exact still-live prefix can continue the draft ring. Other restores reset it and seed from fresh prompt rows. The target's checkpoint and KV precision follow their existing settings.

The drafter captures a full proposal round on its own stream after a direct warmup. Pinned I/O keeps the anchor and position request-local across replays. Short tails run directly. Set `BASALT_DFLASH_GRAPH=0` to measure the direct path.

Draft attention splits the FP16 ring into eight 256-slot chunks, merges FP32 softmax partials, and reuses one scratch allocation across layers. The noise block contributes once.

The compact noise block runs four rows including the anchor. It offers up to three proposals while the verified feature ring and its eight-row capture path keep the same shape. `BASALT_DFLASH_COUPLED=1` lets sampled proposals use the target seed and absolute verifier position with the draft logits. The verifier still chooses the emitted tokens independently. `BASALT_DFLASH_CONF` controls the proposal confidence threshold at startup and defaults to 0.75. `BASALT_DFLASH_PROFILE=1` records per-stage CUDA timings and the offered-width histogram.

The optional cost-based selector weighs conditional confidence against the measured cost of each verifier width. It changes the length of the proposed prefix.

Draft overlap queues the drafter on its own stream while the caller commits already verified rows. The drafter waits before the next window reuses captured features.

The native embedding gathers can bind per-device AtomicChat factors before graph capture. Both batched and single-token gathers apply the correction on the same stream before returning the embedding to their caller.

Optional CPU LoRA jobs borrow host factors. The pool computes one input-factor dot per routed SwiGLU row before activation quantization and adds the delta to the down output before publication.

The optional native `--atomic-lora` path loads a validated Flash-Next adapter at startup. Embeddings, prefill and verifier bindings share the resident device factors. The startup scale is zero. Adapted requests still require the request binding and cache isolation gate.

Layer-ordered bindings support portable pinned FP16 K/V backing or INT8 codes with both FP16 scale arrays. The owner and page-table backing outlive the borrower, which retains its own GPU residency map.

A shared-residency binding also borrows the owner’s GPU pool and map. Use the corresponding session sizing mode to omit private resident KV and map storage. Bind only after the caller has wired the shared addressing path for that session.

Batch decode passes per-row AtomicChat scales to embedding, PLE value and shared expert down. Solo callers use the request scalar.

Grouped CPU experts can borrow the decode window adapter scale vector. Dispatch copies each original token’s scale into its expert job row before starting the workers.

With shared target KV enabled, MTP backup rings allocate on first copy. The graph working ring still allocates during startup.

Sessions borrowing shared FP16 residency also borrow the immutable full-context RoPE table from the primary session.

Shared FP16 borrowers map private QSA pooled keys on demand at stable virtual addresses. They keep recurrence and partial indexer rows separate for each sequence.

The INT8 decode gather translates selected logical cells through the shared page map when a session has shared residency. It keeps private and identity layouts on their original path.

The opt-in INT8 rotation selection applies consistently to decode, draft and prompt-stage storage. Q4 keeps its existing transform.
