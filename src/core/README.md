# Module notes

A newer snapshot with a different key only supersedes a prior committed snapshot after its own index has been published and revalidated.

The peer loader invokes an optional callback after scratch and expert slots are settled and before issuing their copies.

Host reservations optionally use Linux writable prefaulting or touch one byte of each system page, including the final page. The kernel mode falls back to explicit page touch if unavailable. The held pointers are freed before pinned buffers are allocated.

Shared GPU resources and session copy helpers. StagedCopy keeps the host and VMM paths and can queue a legacy device pair directly on the destination stream.

The packed weight loader can write converted slices to a caller-owned RAM image using the same offsets as the CUDA arena. The loader still stores CUDA addresses in the weight table.

`FileExpertSource` builds the resident CPU expert complement from the active placement plan. On Linux, `BASALT_CPU_EXPERT_INTERLEAVE=1` binds only this arena to the allowed NUMA nodes before reads or registration. It leaves expert order, logical CPU/GPU tiers, P2P and CUDA staging allocations to the existing paths.

Grouped disk stores serialize writes and restores around a common retention limit. Eviction compares recency across all participating indexes while each directory keeps its own snapshot identity and part references.

With experimental `BASALT_BCS_DIRECT_RESTORE=1`, the store reads full aligned K/V pieces straight into private snapshot buffers and verifies the original checksums before returning the image. Staging remains for partial pieces, unaligned ranges and checkpoint state.
With experimental `BASALT_BCS_DIRECT_WRITE=1`, the writer can use complete aligned pieces from the immutable queued image. Discontiguous or padded ranges still use staging. The file layout, checksums and publication sequence stay the same.

An expert arena on Linux tmpfs publishes its ready marker only after the initial load has completed. Later workers reuse a ready, model-matched payload under the population lock and recreate their CUDA host registrations.
The experimental `startup_shared_arena_read_prefault` flag prefaults readable page tables for a published shared mapping before CUDA registration. It logs the kernel result and uses the normal registration path after either result. Unpublished arenas still register before loading their payload.
The startup diagnostic reports shared arena identity, lock/layout and map/register times in seconds before any payload copy.
Concurrent reads of an attached immutable expert arena update its diagnostic read count with relaxed atomic operations.

The store checks room for each new aligned container after part deduplication. On filesystem pressure the writer can evict old snapshots within the same cache group and filesystem. A failed write puts that store on a 30-second enqueue cooldown. Queued-image restores copy from their pinned host image without waiting for the shared file lock.
Grouped writers serialize space use. A durable restore holds the group's file-retention lock, while payload writes proceed separately. Publication verifies that no concurrent corrupt read invalidated a shared part.

During early startup the published-source owner holds the population lock and waits for the caller's registration permit after CPU prefaulting. Canceling the permit unmaps the private view.
The early worker bounds each readable prefault call to 256 MiB so CUDA host allocations can run between calls. The synchronous source stage still advises the whole range at once.
The worker checks page residency before each early call. At the first range with nonresident pages or an unavailable residency query, it waits for the existing permit before advising that range and the remainder. Residency is a snapshot of the mapping.
With optional `startup_arena_cold_two_prefault`, two CPU lanes advise disjoint parts of the deferred remainder after the permit. The owner joins both before CUDA registration.
Optional `startup_arena_cold_readahead` sends a best-effort swap read hint before each deferred 256 MiB part. The blocking prefault still checks completion and the same CUDA registration follows.
