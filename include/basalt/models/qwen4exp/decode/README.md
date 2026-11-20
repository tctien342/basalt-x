# Flash-Next decode interfaces

`NativeDense::load` can stage a caller-owned host image of an external device slab. It publishes references into the device slab and allocates projection scratch on the current device.

Verifier and MTP graph storage keeps separate captures for startup, native or YaRN ×2 RoPE.

Windows without a Swift row use the original verifier graph variant. A Swift window has a separate capture key and reads row-specific scales in batch and a device scalar in solo.

Swift uses independent graph-stable rank scratch and per-slot decode scales. The verifier applies Q/K/V and GDN QKV/Z corrections before updating session state. Shared-expert gate/up corrections precede SwiGLU, and the down correction precedes the scalar gate.

`shared_kv_pages.hpp` reserves pool capacity and assigns physical page IDs on demand. `shared_kv_admission.hpp` queues reservations and permits one backfill ahead of a blocked head. These CPU components assume one scheduler owner. The runtime must drain GPU users before releasing a lease. The conversation snapshot interface can defer pooled device rows only when the scheduler retains the immutable source index until the dedicated copy stream completes.

These headers declare session state, native projections, draft and verification. The verify.hpp interface owns target CUDA graphs and commit. The mtp.hpp interface describes the packed MTP layer and reports CUDA allocation status during shared slot registration. The dflash2.hpp interface describes the alternative block drafter.

The verifier exposes the allocated bytes of its main device arena. This excludes the optional profiling allocations and session-state backing.

The K=23 and K=10 target verifiers share the session and peer cache, with separate graphs and completion state.

The optional DFlash2 object borrows the target head and embedding and is registered for the lifetime of the engine and its graph captures. Prompt feature rows and verified decode rows enter its bounded ring. The target verifier owns acceptance and its own KV cache.

The drafter owns its pinned round staging and full-round CUDA graph until engine teardown. Prefix resets leave those addresses intact.

The drafter arena owns the per-chunk attention partials. The graph keeps the same workspace for all layers.

The compact draft arena holds four noise rows and three proposal rows. Verified feature capture and context injection still hold eight rows. The optional timing events belong to the drafter and are reused across rounds.

The selector can borrow a read-only verify cost estimate while ranking the proposed window lengths.

The optional enqueue callback runs on the caller device after draft submission and before draft synchronization.

The native embedding gathers can bind per-device AtomicChat factors before graph capture. Both batched and single-token gathers apply the correction on the same stream before returning the embedding to their caller.

Verifier LoRA corrections read the gated attention and normalized GDN activations and apply shared-expert down deltas before the scalar gate. The split GDN projection requires its own adapter integration before binding.

Optional CPU LoRA jobs borrow host factors. The pool computes one input-factor dot per routed SwiGLU row before activation quantization and adds the delta to the down output before publication.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The optional native `--atomic-lora` path loads a validated Flash-Next adapter at startup. Embeddings, prefill and verifier bindings share the resident device factors. The startup scale is zero. Adapted requests still require the request binding and cache isolation gate.

A widened verifier can specify the first widened layer when initialized. Earlier layers use K=10.

A mapped FP16 QSA session exposes the scheduler-owned CPU mirror of its device table for synchronous checkpoint transfers. Both views must describe the same live page lease.

SharedKvTable publishes the materialized part of a lease at a fixed CUDA address per sequence and device and updates the session layers’ CPU mirror. Its session invalidation path waits on the caller-ordered stream before clearing those mirrors. The scheduler must release the lease only after all devices succeed.

A parked lease can return unused reservation capacity while retaining all materialized prefix pages. Reacquiring capacity preserves the page IDs but may relocate the CPU vector, so a growing lease must republish its CPU mirror before checkpoint code uses it.

Batch staging expands the slot adapter scales into graph-stable mapped rows. Set the slot scale before staging a request window. Attention and recurrence output projections and grouped local expert corrections consume these rows.

Batch decode passes per-row AtomicChat scales to embedding, PLE value and shared expert down. Solo callers use the request scalar.

Grouped CPU experts can borrow the decode window adapter scale vector. Dispatch copies each original token’s scale into its expert job row before starting the workers.

With shared target KV enabled, MTP backup rings allocate on first copy. The graph working ring still allocates during startup.

Shared batch MTP binds the request scale to its draft store. Store switches update mapped embedding scales after the previous stream drains.

Batch PLE stages history pointers by segment of the current window. Captured addresses and pointer table size depend only on verifier width.

QSA descriptor and indexer snapshot storage is sized by verifier width. Each window stages its segment pointers and retains them through commit.

The initialized shared verifier can register another compatible session on the scheduler thread. The caller owns the session and keeps it live through all windows and commits. Registration leaves CUDA graph buffers in place.

Runtime shared registration preserves existing draft slot IDs and moves main-store metadata to the new final index. The scheduler calls registration between windows.

Runtime shared-slot registration creates a borrower of the main FP16 residency map and publishes its scheduler state after verifier and MTP registration succeed.

When a shared slot loses its parked prefix, the scheduler drains and releases its draft backup ring and deferred rows. The graph working state stays allocated.

Borrower QSA states reserve a fixed address for pooled keys. The scheduler maps backing through the admitted extent before starting any reader and releases backing after draining the lease.

Shared slot registration can query draft-ring backing bytes before allocating the slot. Admission can query missing rings and provision them before prompting.

Shared KV tables expose whether every stage has published the backing pages of a candidate checkpoint. Cross-slot restore uses that check before ranking checkpoints.

The conversation snapshot interface can select a saved checkpoint in a single-stage host image and shrink its K/V payload before restore into a shorter shared FP16 lease.

QSA host bindings preserve the owner’s storage format, pools and residency map. An INT8 binding requires both code and both scale arrays before it can publish a borrowed session.

The verifier can update the opt-in one-peer batch PCIe share between completed windows. Its last-window counters report distinct eligible and staged experts and staged payload bytes.

The verifier keys staged and zero-share graphs separately, including whether cost observations are recorded. The zero-share graph has no staging copy or staged group. Returning to the unchanged ordinary graph is immediate for ineligible windows. Timing survives mapped alias placement updates; batch-end counters expose why the model assigned zero or positive budget.

The snapshot save API accepts an optional stop callback for a disposable idle capture. A stopped capture never publishes a partial image.

The split shared INT8 capture seals a private checkpoint chain, draft and indexer payload, and copies its page IDs on the scheduler. The finish call uses CPU aliases only. The caller keeps the source backing frozen and reserved through the worker completion fence.

Frozen MTP snapshots protect a page-rounded host interval through a draft-ready future. Draft writers check their actual launch or restore range.
