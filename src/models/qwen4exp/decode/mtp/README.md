# Flash-Next MTP draft layer

Rotated INT8 MTP keeps Q and K at 256 channels while sharing the selected V width with draft prefill.

Rotated INT8 draft preparation fuses query split, normalization, RoPE and Hadamard. The KV post kernel applies K normalization and RoPE, rotates K/V and appends their codes and scales.

MTP graph caches partition prompt ingestion, draft rounds and chain steps by host RoPE context.

Rotated INT8 single-row drafts with an attention window of at least 8192 cells use the wide tensor-core attention kernel. The attention output is inverse-rotated before the gate and Q8 activation pack. The target verifier still decides accepted tokens.

`load.cpp` allocates the graph working state and draft buffers. `stores.cpp` saves and restores sequence rings when the drafter changes slots. `batching.cpp` feeds deferred rows, and `graphs.cpp` and the forward modules execute draft windows on the owned CUDA stream. Interfaces live in `include/basalt/models/qwen4exp/decode/mtp.hpp`.

Shared target KV mode allocates missing seed and main draft backup rings during admission and reserves rings for newly grown slots during registration. Captured working state addresses remain stable. A failed allocation releases the partial ring, and destruction releases all completed rings. Legacy startup allocates its rings eagerly.

Shared batch MTP binds the request scale to its draft store. Store switches update mapped embedding scales after the previous stream drains.

Runtime shared registration preserves existing draft slot IDs and moves main-store metadata to the new final index. The scheduler calls registration between windows and can reclaim an expert segment before retrying a CUDA out-of-memory result.

Shared draft stores allocate deferred activation rows on first use. If that allocation runs out of memory, the drafter processes the incoming rows immediately.

When a shared slot loses its parked prefix, the scheduler drains and releases its deferred rows and retains its reserved draft backup ring. The graph working state stays allocated.

Shared slot admission includes the draft-ring backing estimate before allocating the slot.

Draft ring allocation errors report the failing CUDA operation and allocation size.

An overlapping draft graph, prefill, ring save or restore waits for the draft-tail copy fence. Disjoint writes retain their ordinary scheduling.
