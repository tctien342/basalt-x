# Flash-Next CUDA kernels

The value Hadamard launchers support contiguous blocks of 64, 128 or 256 channels. Query and key transforms retain 256 channels.

The rotated INT8 verifier epilogue restores each attention head with inverse Hadamard, applies its gate, then packs native Q8_1 blocks in one launch.

The launch thread can scope analytic RoPE to native or YaRN ×2. The startup profile stays immutable.

Swift uses independent graph-stable rank scratch and per-slot decode scales. The verifier applies Q/K/V and GDN QKV/Z corrections before updating session state. Shared-expert gate/up corrections precede SwiGLU, and the down correction precedes the scalar gate.

The Swift rank-32 launcher applies BF16 factors to FP16 or FP32 activations and accumulates into FP32 projection rows. The caller owns fixed-address scratch and a scalar or per-row device scale.

This directory implements decode MoE, attention, recurrence and their CUDA launchers. Interfaces live under include/basalt/models/qwen4exp/kernels. The direct peer decode path can pack expert results in FP16 and widen them in the FP32 reduction after the existing fence.

The device-planned combine has vector paths for top-10 and top-23 routing. Peer rows use the direct completion fence when available. The scalar fallback handles other widths up to 32 experts.

The selective staging kernels use one bitmap bit per historical page. They copy each selected page from its resident slot or the mapped host pool with coalesced vector loads.

Fused HC routing supports top-10 and top-23. Top-23 uses the generic selector’s ordered FP64 sums and stable ties.

The AtomicChat rank-one launchers add FP32 projection and token embedding deltas in place. Inputs, output factors, routing IDs and destination rows are caller-owned device buffers. The caller validates IDs against the factor table and schedules each correction before consuming the output.

The optional device adapter owns one immutable upload per CUDA device and a fixed-address request-scale scalar. Before updating that scalar, the caller must drain previous users on the device. A zero device scale leaves output rows untouched.

The embedding launcher also accepts one host token ID, which avoids a device token buffer for a single-row gather.

The optional AtomicChat PLE value delta is applied after projection and before PLE postprocessing. Its factors and scale are bound per device before graph capture.

Native prefill projections can apply a rank-one delta from the same FP16 activation rows used by GEMM. The optional adapter is bound to the GEMM context and selected by the writer name.

Verifier LoRA corrections read the gated attention and normalized GDN activations and apply shared-expert down deltas before the scalar gate. The split GDN projection requires its own adapter integration before binding.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The primary verifier applies the optional routed LoRA correction from the grouped gate/up scratch after down projection. Destination indices preserve router order for both cached and staged expert rows.

The vector combine for intermediate K retains the peer and CPU completion fences. Persistent peer routing and warm-up use the current and predicted layer widths respectively.

An optional device page table translates sequence-local FP16 host KV blocks to physical host backing in append, residency fill and prefill staging. The caller owns the table and backing and must keep both live until their CUDA work drains. The fixed-session path uses identity addressing.

Segmented QSA postprocessing and residency fills accept FP16 and INT8. Each recorded attention launch uses one format across its slots.

Shared segmented FP16 residency resolves all rows in one epoch against physical page IDs. The caller must bind a common map and page backing for every slot and bound the window working set to the resident capacity.

Shared residency invalidates only the named physical pages after the owning GPU work drains. Snapshot restore invalidates the pages it overwrites. Sequence reset preserves the other sequences’ resident pages.

AtomicChat projection and embedding launchers also accept a device scale per activation row. The grouped down launcher takes the routing width and indexes those scales by the original token row before expert grouping. Scale storage is caller-owned and must remain live until the CUDA stream completes.

Batch decode passes per-row AtomicChat scales to embedding, PLE value and shared expert down. Solo callers use the request scalar.

The peer expert window reads adapter scales from portable mapped staging, indexed by the original token row. Solo windows use the existing peer-device scalar.

Segmented commit position staging is indexed by segment within the current window. Its size and copy traffic depend on verifier width. Session state and QSA slot descriptors still use logical slot IDs.

Batch PLE stages history pointers by segment of the current window. Captured addresses and pointer table size depend only on verifier width.

The INT8 append, miss-fill and gather kernels accept the optional shared physical page mapping. Both codes and both scales use the same page. Shared residency miss IDs are already physical when copying the four arrays.

The segmented INT8 post kernel can take prepared attention K/V rows from the Hadamard path while keeping indexer updates and tail snapshots on each row’s owning slot.

The device-plan staged fetch uses the same tier classification as route publication. A peer-staging layout uses one uncapped persistent peer and the existing fixed staging limits.

Peer staging ranks eligible mapped CPU misses by the number of rows that reuse each distinct expert, with router order breaking ties. Legacy single-card staging keeps its router order.

The opt-in layer cost model timestamps the normal combine wait and staged copy blocks without adding a peer or host wait. Both route and copy use its graph-scoped staging count table.

A direct-peer window takes its local and peer timestamps inside the existing plain done-word wait. It preserves the earlier local stamp from a staged window and records peer observation before waiting on CPU results. There is no added kernel or wait.
