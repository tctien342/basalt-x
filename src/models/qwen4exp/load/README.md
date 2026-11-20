# Qwen Flash-Next loader

The Swift option loads a validated rank-32 adapter before native expert sizing so both relevant devices size their tiers around the immutable factor allocations. It uses the unsplit target verifier and the normal staged prompt pipeline.

`--no-mtp` skips selecting the embedded Flash-Next drafter before weights and expert-tier budgeting. The target verifier retains its configured window.

The loader owns model buffers, device sessions and expert-tier startup. In adaptive vision mode it warms the encoder on the peer before expert placement and retains the weights in pinned host RAM. The GPU encoder arena exists only while an ENC command borrows expert slots.
The experimental startup_parallel_vision flag stages vision weights in pinned host RAM on a worker after reserving the text weight arena. Text weight upload runs on the startup thread, which joins the worker before creating CUDA vision resources.
The startup diagnostic reports host staging and join wait, creation, temporary attachment, preprocessing, encode, and teardown times in seconds.
The experimental startup_parallel_peer_warmup flag overlaps warm-up on the peer with text session initialization when the entire text session is on CUDA0. The startup thread joins after expert source attachment and before sizing either expert cache.
The experimental `startup_vision_warmup_512` flag uses a 512-square startup image in adaptive mode. It still attaches the full 2,048-token arena and releases it before expert sizing. Runtime input geometry is unchanged.
The experimental `startup_direct_prompt_image` flag builds the prompt-split weight image in pinned RAM without borrowing expert slots during startup. Each split request still uploads the image into the same expert-cache tail before the stage runs.
The experimental `startup_parallel_expert_banks` flag overlaps CUDA0 and peer expert upload for one non-exclusive peer backed by an immutable shared arena. The primary slot map is available before peer selection. Both uploads finish and verify data before any request or startup prompt-split loan can consume them.
The bank startup diagnostic reports token-path setup, resident planning and allocation, split-link probes and the split RAM budget in seconds.
The experimental `startup_pcie_probe_64m` flag uses 64 MiB instead of 256 MiB for individual and concurrent startup link probes. Warm-up and burst counts, bandwidth estimates and prompt-split decisions follow the same code.
The experimental `startup_parallel_split_reserve` flag touches prompt-split host reservations on a worker while both expert banks are filled on the same shared-arena layout. It joins before any split loan or allocation uses the reservation. File-backed complement planning retains synchronous reservations.
The experimental `startup_split_reserve_page_touch` flag commits the same split reservations page by page when set to 1. Mode 2 requests Linux writable prefaulting and falls back to explicit page writes. The later pinned allocations still replace the holds through `host_alloc`.
The experimental `startup_split_prefault_after_peer_alloc` flag starts mode 2 just before peer DMA, after peer scratch, pinned allocations and slots are ready. The reservation still joins before a split consumes it.

Adapter-enabled startup reserves MMQ fallback scratch before prefill sizing.

The isolated shared_kv_pool experiment binds streamed FP16 or INT8 target sessions to slot zero’s backing on the main decode device. Per-slot tables and model state remain separate. This loader path still sizes logical sessions at startup and supports peer prefill with a single decode stage. The native shared mode accepts a logical session count wider than one verifier window. Each window still fits its bounded row scratch.

With shared target KV enabled, MTP backup rings allocate on first copy. The graph working ring still allocates during startup.

Runtime shared-slot registration creates a borrower of the main residency map and publishes its scheduler state after verifier and MTP registration succeed.

Shared KV can start with one session and grow on demand. The default seed count is one when the pool is enabled.

The runtime slot deleter releases its QSA virtual indexer ranges before freeing the slot arena.

The shared_kv_expert_lending experiment segments the main cache for a single decode stage, including configurations with a separate peer expert tier.

The loader releases the image warm-up arena before sizing the peer expert cache. With adaptive peer vision and BASALT_GPU1_VISION_OVERLAP=1, native serving loans the encoder a tail interval only while it is needed.

The immutable shared expert arena qualifies as a complete host copy for staged peer prefill and solo auto sizing. File-backed expert modes still require a ready RAM complement.

Startup and dynamic shared-slot attachment require matching rotation, format and physical backing. For INT8 the residency and host bindings include K, V and both scale arrays.

Peer expert sizing reserves prompt buffers and budgets the vocabulary and GDN splits independently. Ineligible GDN tensors leave their unused budget available to experts.

With experimental `startup_early_shared_arena` and readable prefault enabled, a worker prepares the existing shared mapping after text KV allocation and vision host staging, before draft and head weight loading. CUDA registration waits for the usual source stage after the text session and head allocations. The startup thread joins before consuming the arena.
The worker prefaults already resident ranges early. Encountering a range with nonresident pages defers the remaining prefault work until the same post-head permit.
The optional cold two-lane worker applies only to the deferred ranges after that permit.
It can issue a bounded swap readahead hint for each range before faulting and registering the bytes.

Serving publishes the already measured simultaneous H2D copy bandwidth as `INFO pcie_h2d_combined_mbps` and the number of probed devices. The loader reuses the existing warm-up and bursts.
