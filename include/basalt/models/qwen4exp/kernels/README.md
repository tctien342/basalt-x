# Flash-Next CUDA interfaces

Value rotation width is process-wide for rotated INT8, including the fused draft KV append and the verifier gate pack.

The decode post descriptors can request Hadamard on the query after RoPE and on INT8 K/V before append.

A RoPE context scope selects immutable analytic parameters on the calling thread and restores the enclosing profile.

Swift uses independent graph-stable rank scratch and per-slot decode scales. The verifier applies Q/K/V and GDN QKV/Z corrections before updating session state. Shared-expert gate/up corrections precede SwiGLU, and the down correction precedes the scalar gate.

The Swift rank-32 launcher applies BF16 factors to FP16 or FP32 activations and accumulates into FP32 projection rows. The caller owns fixed-address scratch and a scalar or per-row device scale.

These headers expose model kernels and CUDA launch arguments to the verifier, prefill and serving paths. The peer decode result format is part of the launch contract, and the combine uses the same format after peer completion.

The residency-biased selector takes 10 through 23 picks per row of 512 logits.

The selective FP16 or INT8 prompt staging launcher takes the indexer's cell IDs and a caller-owned scratch bitmap. It writes selected pages at their logical indices without modifying the resident page table.

Fused HC routing supports top-10 and top-23. Top-23 uses the generic selector’s ordered FP64 sums and stable ties.

The AtomicChat rank-one launchers add FP32 projection and token embedding deltas in place. Inputs, output factors, routing IDs and destination rows are caller-owned device buffers. The caller validates IDs against the factor table and schedules each correction before consuming the output.

The optional device adapter owns one immutable upload per CUDA device and a fixed-address request-scale scalar. Before updating that scalar, the caller must drain previous users on the device. A zero device scale leaves output rows untouched.

The embedding launcher also accepts one host token ID, which avoids a device token buffer for a single-row gather.

The optional AtomicChat PLE value delta is applied after projection and before PLE postprocessing. Its factors and scale are bound per device before graph capture.

Native prefill projections can apply a rank-one delta from the same FP16 activation rows used by GEMM. The optional adapter is bound to the GEMM context and selected by the writer name.

Verifier LoRA corrections read the gated attention and normalized GDN activations and apply shared-expert down deltas before the scalar gate. The split GDN projection requires its own adapter integration before binding.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The primary verifier applies the optional routed LoRA correction from the grouped gate/up scratch after down projection. Destination indices preserve router order for both cached and staged expert rows.

The persistent peer window carries a layer boundary for mixed K. The combine and residency-biased router support the entire thinking K range.

An optional device page table translates sequence-local FP16 host KV blocks to physical host backing in append, residency fill and prefill staging. The caller owns the table and backing and must keep both live until their CUDA work drains. The fixed-session path uses identity addressing.

Segmented QSA postprocessing and residency fills accept FP16 and INT8. Each recorded attention launch uses one format across its slots.

AtomicChat projection and embedding launchers also accept a device scale per activation row. The grouped down launcher takes the routing width and indexes those scales by the original token row before expert grouping. Scale storage is caller-owned and must remain live until the CUDA stream completes.

Batch decode passes per-row AtomicChat scales to embedding, PLE value and shared expert down. Solo callers use the request scalar.

The peer expert window reads adapter scales from portable mapped staging, indexed by the original token row. Solo windows use the existing peer-device scalar.

Segmented commit position staging is indexed by segment within the current window. Its size and copy traffic depend on verifier width. Session state and QSA slot descriptors still use logical slot IDs.

Batch PLE stages history pointers by segment of the current window. Captured addresses and pointer table size depend only on verifier width.

QSA descriptor and indexer snapshot storage is sized by verifier width. Each window stages its segment pointers and retains them through commit.

INT8 gather accepts an optional logical-to-physical page table. A null table preserves identity page IDs. The owner keeps it live while CUDA work reads the mapping.

The QSA post interface accepts prepared K/V values while retaining append and indexer processing. This allows rotated INT8 to reuse the segmented writer.

The optional MoE wait stats buffer includes last-window staged copy and grouped-compute stamps by layer.

For the batch staging cost model, each layer owns mapped time words for route, local finish, peer acknowledgment, grouped finish and copy block boundaries. Other call sites pass no timing buffer.

The direct-peer wait launcher can write route-relative timing into caller-owned mapped words. It observes the existing peer done word and expects a plain stream launch after local work. Legacy wait statistics use a separate buffer.
