# Module notes

The rotated INT8 verifier uses the selected value width for both V append and the fused inverse, gate and pack.

The rotated INT8 verifier packs Q8_1 output directly from the inverse Hadamard and gate epilogue. It waits for the query-ready event before reading the gate.

Verifier window and indexer commit graphs are captured per RoPE context. Both the standalone and pipelined commits use the profile of their staged window.

Windows without a Swift row use the original verifier graph variant. A Swift window has a separate capture key and reads row-specific scales in batch and a device scalar in solo.

Swift uses independent graph-stable rank scratch and per-slot decode scales. The verifier applies Q/K/V and GDN QKV/Z corrections before updating session state. Shared-expert gate/up corrections precede SwiGLU, and the down correction precedes the scalar gate.

Solo commit graph setup reports separate capture and instantiation errors.

The primary workspace byte count comes from the successful arena allocation used by the verifier and commit graphs.

Flash-Next verification builds graph-stable decode plans. The single-peer direct row arena uses the existing completion word and keeps CPU expert rows in mapped host memory.

The direct single-peer result arena can store FP16, with the existing completion fence and a FP32 combine. Mapped CPU and multi-peer results retain their original layout.

With DFlash2, graph capture records the five completed-layer feature means into graph-stable scratch. Only verified prompt or accepted decode rows feed the draft KV.

The max-thinking verifier shares the target session with the normal verifier. It shares the configured peer tiers while keeping its own K=23 plans, scratch and completion state, requires device-planned execution within 128 routed entries, and drains commits before either verifier gives up ownership of the session.

The optional AtomicChat PLE value delta is applied after projection and before PLE postprocessing. Its factors and scale are bound per device before graph capture.

Verifier LoRA corrections read the gated attention and normalized GDN activations and apply shared-expert down deltas before the scalar gate. The split GDN projection requires its own adapter integration before binding.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The primary verifier applies the optional routed LoRA correction from the grouped gate/up scratch after down projection. Destination indices preserve router order for both cached and staged expert rows.

The max verifier allocates for the configured target K and uses K=10 before the configured final layer range. Routing, CPU jobs, peer results and the combine use the actual layer stride.

Segmented QSA postprocessing and residency fills accept FP16 and INT8. Each recorded attention launch uses one format across its slots.

Eligible multi-row INT8+ROT batch windows without Swift use the existing QSA branch streams. The V-ready event follows its Hadamard transform. The K stream prepares K before joining V and updating the segmented KV store. Query rotation precedes attention, and the existing indexer and KV joins protect slot-owned state. The attention merge and inverse-rotation output path are unchanged.

Batch staging expands the slot adapter scales into graph-stable mapped rows. Set the slot scale before staging a request window. Attention and recurrence output projections and grouped local expert corrections consume these rows.

Batch decode passes per-row AtomicChat scales to embedding, PLE value and shared expert down. Solo callers use the request scalar.

The peer expert window reads adapter scales from portable mapped staging, indexed by the original token row. Solo windows use the existing peer-device scalar.

Segmented commit position staging is indexed by segment within the current window. Its size and copy traffic depend on verifier width. Session state and QSA slot descriptors still use logical slot IDs.

Batch PLE stages history pointers by segment of the current window. Captured addresses and pointer table size depend only on verifier width.

QSA descriptor and indexer snapshot storage is sized by verifier width. Each window stages its segment pointers and retains them through commit.

The initialized shared verifier can register another compatible session on the scheduler thread. The caller owns the session and keeps it live through all windows and commits. Registration leaves CUDA graph buffers in place.

Runtime shared-slot registration creates a borrower of the main residency map and publishes its scheduler state after verifier and MTP registration succeed.

The rotated INT8 verifier prepares attention K and V with an orthonormal Hadamard transform, uses the segmented writer for slot-local cache and indexer ownership, and applies the inverse transform before gating attention output. The indexer remains in its original basis.

Startup and dynamic shared-slot attachment require matching rotation, format and physical backing. For INT8 the residency and host bindings include K, V and both scale arrays.

Peer GDN sizing and verifier startup share the same tensor eligibility checks. An unsupported layer contributes zero bytes to the peer split budget.

Rotated INT8 reuses the combined query preparation and prepared K/V append kernels. The solo verifier uses staged attention when the exact target kernel supports its geometry and width. The inverse transform still runs before the gate, and the draft-specific wide kernel stays outside the target path.

Grouped verifier calls with a null output pointer run and commit known input rows without applying the per-row sampler.

The adaptive PCIe staging flag allocates an initially zero staging table for one persistent peer in an unsplit batch verifier. The stage stride fits the largest expert in a mixed-quant pack, while each layer copies its own blob length. A staging window refreshes aligned pinned aliases per expert and the fraction upload precedes its captured graph. The staged group applies AtomicLoRA output correction.

Adaptive staging captures separate zero-share and positive-share graphs. Automatic tables are computed from per-layer CPU service, local GPU completion, observed peer acknowledgements, and staged copy and grouped-compute timing. The positive graph is selected only when the last comparable window's misses are predicted to benefit. Both route and copy consumers read the same immutable table. A sparse layer can stage one whole expert above the soft ceiling during bootstrap when conservative copy and compute estimates leave twice the usual margin. An explicit hard cap still applies. Pinned comparisons still use a byte table referenced to average blob size.

Direct peer result arenas also allocate adaptive timing when legacy wait statistics are disabled. The verifier passes the per-layer mapped words to the existing done-word wait. The tagged path continues to measure inside combine.
