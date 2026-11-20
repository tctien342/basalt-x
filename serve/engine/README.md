# Engine adapters

On workers with RoPE profiles, shared slot affinity also requires the same 256K or 512K profile. Different profiles can hold capacity concurrently.

Inherited engine flags use the native parser-only probe when advertised. Older engines use GPU-plan for those probes. The final model plan always checks GPU placement.
With `BASALT_PARSER_PROBE_CACHE` set to a writable file in the bridge's own directory, successful parser-only checks persist across bridge restarts. Cache hits require matching executable, model metadata and exact argument group. Unknown, malformed and failed checks run normally.

This folder wraps native command transport, model packs, and image input. Adaptive vision uses the existing normalized-image cache for both backends. A single uncached image below about 1,024 image tokens uses CPU. A larger image or multiple distinct misses use one native peer loan. Cached images require no encoder work.

For Flash-Next max thinking, the API supplies the native GEN key `think_close` with the special closing token ID.

The private Boolean sampling fields `_atomic_lora` and `_swift_lora` serialize to the corresponding `atomic_lora=1` and `swift_lora=1` native keys. The server alias dispatcher sets either or both. Slot affinity uses the full combination.

Native and adaptive image requests use attempt-specific temporary files and atomically publish the normalized image cache entry. Adaptive vision sends videos through the CPU sidecar. The upstream dynamic encoder has its own video path.

The slot admission queue provides a capacity-aware policy for a shared KV pool. Grant the oldest fitting waiter and allow one bypass when the head cannot fit. Callers serialize the check with charging the request’s reservation. If the child advertises `shared_kv_mixed_adapters=1`, different adapter modes can acquire capacity at the same time. Otherwise requests for the other mode wait for the active cohort to drain and use the same bypass budget. Shared slot selection scores session and prefix affinity only for the matching adapter. An assignment to a different adapter clears the old API prefix hint. When the worker advertises `shared_kv_grow=1`, the first eligible waiter can ask for a slot using `BGROW` while owning control. A refusal leaves the request queued without spending its bypass.

Both solo and batch select eligible slots by affinity. A shared slot can match a rewritten continuation when its common prefix is longer than the discarded tail. The native engine still validates its checkpoints. Without affinity, prefer the shortest held prefix and break ties by recency.

Cancellation after a successful growth reply leaves the new slot idle and available for the next request. While the current engine advertises a frozen source, compatible alternate-profile text prefers a fitting nonzero slot or plain growth. The reported old pages are charged only for that preference. If an alternate cannot fit or grow, normal admission can still retire the source. A text request changing both RoPE and adapter also prefers a fitting nonzero slot when slot zero holds a long cached prefix. The extra page charge is used for this placement decision. A continuation without a matching session key can reuse slot zero when its compatible token prefix clears the long-context threshold and beats the unowned choices. The reader handles the INFO message without the slot condition lock and each process owns its own mailbox.

A shared batch request whose indexer or MTP draft rings cannot allocate VRAM releases control and its provisional reservation before waiting for a running slot to finish. It retries normal admission after that slot completes. When no other slot can release resources, the request receives a resource error.

The experimental BASALT_SHARED_SOLO=1 bridge mode requires a worker with slot-zero shared SGEN lease settlement. An idle slot-zero request can start in native solo. On contention it keeps its reservation and continues on BGEN after the native STOP boundary. A full-grant text request on slot zero can return to native solo after other work drains. It retains the slot and page charge through BDONE, and resumes BGEN on the same slot if another request has arrived. The bridge checks cancellation and engine identity again after reacquiring native control.

The experimental BASALT_SHARED_KV_INCREMENTAL=1 bridge mode can grant bounded text decode runs under shared-pool pressure. At least one active slot must hold a full grant before a partial grant can be issued. A length boundary on a partial grant returns the same API generator to admission. A parked request can take a full grant whenever it fits. Another partial grant requires a different full owner so the same cohort cannot repeatedly prefill a large paused stream. Native prefix reuse is opportunistic as the pool may evict parked pages. Images keep full reservations.

With native vision_overlap capability, adaptive image work uses an identified ENCA reply queue. The engine's sole stdout reader dispatches those replies separately from token and control lines. Capacity refusals and timeouts use the existing CPU fallback. Videos keep their existing path.

A native worker advertising `scheduler_queue=1` receives coalesced control and admission wait counts, parked admission count, oldest age in milliseconds, and free shared KV pages. The publisher binds the updates to that worker process. Native scheduler snapshots are read separately from requests and reset on replacement.

Load narration preserves each startup KV allocation record for metrics after progress completes. A new process gets a fresh set of records.

The opt-in recovery coordinator receives unexpected native EOF after the old worker's queues close. It reloads through the service FIFO for that worker identity. Explicit shutdown does not trigger recovery, and a corrupt output run kills the worker without flushing its current cache.

Set `BASALT_OVERLAP_HINT_FILE` to an absolute path to an independently qualified single-row overlap profile. The launcher checks expiry and fingerprints the executable, inputs and launch configuration. It sends the matched row only to the child at startup. The bridge records a rejection before starting an asynchronous compare-and-delete of the unchanged row. A restart in the same bridge skips the row even while the write is pending. No request data is written.

For independent online overlap learning experiments, set `BASALT_OVERLAP_LEARN_DIR` to an existing absolute directory owned by that bridge. One file per compatible launch fingerprint retains up to sixteen workload scopes. The native worker reports one settled outcome per request. A durable candidate requires two wins for the same setting, is dropped after a valid rejection, and remains subject to fresh measurement after restart. Writes run outside the reply pump. Use a separate directory for each service instance. Metrics read a locked summary of unexpired in-memory scopes and the write state. Outcomes stay in memory across native child restarts, but the file stores only the entries. The API omits the profile path and compatibility fingerprint.

A profiled Flash-Next request emits a trusted `rope_profile` option and requires native capability `rope_context_profiles=1`. The alias cap covers prompt plus output and is checked locally, including ID conversion and continuation. Page admission still uses the shared pool geometry.

With the native grouped_short_prefill capability, BGEN adds an advisory hint while another short prompt is already waiting on control. The native request still validates compatibility and cache reuse. The native BPF line is routed to its owning slot. It updates that request's fresh prompt count, reused prefix and elapsed prefill time when a deferred grouped prompt finishes. A zero-output cancelled slot drops its affinity hint during cleanup.

The grouped prefill hint can use a short suffix of an exact prefix held in the claimed slot when another request is waiting. It remains advisory. The native worker validates the restored prefix and actual suffix before deferral.

For an advertised cold recovery path, the first replacement of a long idle slot-zero conversation on another RoPE profile may ask for one additional slot. The bridge counts both conversations, uses the existing ticket to gate hinted BGROW, and retains conventional slot-zero admission on refusal.
