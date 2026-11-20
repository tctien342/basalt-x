# Module notes

A recovery capture may inspect retained K/V only through the session that restored it. Admission checks its incremental memory requirement before taking ownership of the retained buffers.

Core interfaces and CUDA ownership helpers. The direct peer helper limits experimental copies to eligible devices and legacy device allocations or mapped loans with an explicit access grant.

`PeerExperts::open` can schedule independent startup work after reserving its slots and buffers, immediately before upload.

`host_plan` holds committed pages until an allocation replaces the reservation. The page-touch and kernel-prefault modes only commit writable pages. The reservation contents are private.

`ArenaExpertSource` permits concurrent blob reads once its immutable arena is published. Its owner still serializes open and destruction with readers.

`WeightTable::load` can write a caller-owned RAM image while binding reference addresses to a device arena. The caller uploads that image before any CUDA consumer uses the references.

The RAM conversation cache can donate K/V from a normal eviction to the session that kept an unchanged prefix. The donor is identified by a live cache entry and invalidated when that entry leaves, the session changes, or the retained prefix is rewritten.

A retained RAM donor can share snapshot segments with an eligible capture after admission. Normal buffer copies own independent storage. An explicit shared copy detaches a segment before mutation, and the cache continues charging each image its full logical capacity.

Conversation buffer segment allocations of at least 4 KiB are aligned for direct store reads. Smaller segments retain normal allocator behavior.

Disk conversation stores can share one configured byte budget across independently identified directories. A prefix match identifies its entry across index erasures.

Check store queue capacity before starting an expensive snapshot capture. A failed background write temporarily stops new submissions; the file format and existing reuse rules are unchanged.

An optional registration gate uses a published mapping and transfers CUDA access to its worker after the caller permits it.

A completed unsplit conversation can supply a recovery writer view with independent metadata and explicitly shared checkpoint and K/V payloads. Finish deferred checkpoint writes before creating the view. The existing buffer copy still owns independent bytes. A mutable checkpoint accessor detaches a shared backing before returning it.

A completed idle recovery image may be retained within the existing byte budget without consuming a parked slot. An exact handoff moves the saved image and KV together. Dirty or foreign owners drop the exact metadata, leaving only eligible prefix reuse.
