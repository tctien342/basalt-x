# Conversation snapshots

The alternate rotated INT8 V widths have distinct snapshot layout tags. V256 keeps its existing tag.

`conversation_state.cpp` captures and restores model running state and assembles session snapshots. `conversation_snapshot.cpp` copies authoritative K/V and indexer payloads using the model session's layout. `conversation_checked.hpp` provides checked snapshot arithmetic. Callers and entry points use `basalt/models/qwen4exp/decode/conversation_snapshot.hpp`.

The caller fences state writes before capture. Publishing a snapshot requires all of its layers to succeed. Restore validates geometry and buffer extents before changing the session. `park_parallel_copy` captures long snapshots across four independent layer lanes and can read streamed K/V from the CPU alias after the fence. Device-resident buffers retain the CUDA copy. Snapshot layout, memory admission and restore behavior are shared with serial capture.

`park_parallel_restore` restores long snapshots across independent QSA layers. The existing device fence precedes the copies, and a new device fence completes worker transfers and resident-map resets before the serial draft and running-state restore. Other restore callers keep their existing copy path.

`park_capture_lanes12` uses twelve independent lanes for long captures with fresh backing and at least twelve QSA layers. Captures with retained backing and restores keep four lanes. The capture and publication fences are shared.

`park_shared_restore8` uses eight restore lanes only when the incoming parked image still shares a segment. Other restores and all capture choices keep their prior lane counts and fences.

Mapped FP16 and INT8 checkpoint transfers use the session CPU page table, coalesce adjacent physical pages and preserve logical page order in the saved image. The scheduler keeps mappings and backing pinned through capture, restore and peer transfer. INT8 codes and both scale arrays use the same page map.

A host snapshot can be trimmed to a retained running-state checkpoint before shared FP16 restore. The operation keeps only the rounded K/V prefix and matching indexer rows and does not touch device state or change the snapshot format.

The optional idle snapshot stop check runs before and after the device fence, between KV buffer segments and between page runs inside a segment. Parallel copy lanes join before returning and publication requires a complete image.

The shared INT8 seal validates a single source page map and copies pooled rows plus running state and draft. The deferred finish reads the mapped host codes and scales without CUDA calls. It publishes only a complete image and can stop between segments. Runtime ownership and admission must keep the source lease immutable until the finish returns. The frozen-slot seal can leave contiguous pooled device rows on the same lease. Its worker reads the full index extent, including the moving spare row, on a nonblocking stream and drains that stream before publishing or retiring the lease.

A frozen shared seal can copy the mutable prefix of a contiguous mapped draft image. Its worker copies the remaining draft bytes first into the existing dense representation.
