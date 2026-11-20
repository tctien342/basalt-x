# Qwen Flash-Next serving

The experiment flag int8_kv_v_rot_width accepts 64, 128 or 256. Alternate widths have distinct conversation identities.

Idle RAM recovery can move the active session's validated retained K/V into its capture. The memory guard checks additional allocation, and the recovery queue accepts one long image within its configured budget. After a successful submission for the active unsplit owner, the live cache can retain shared snapshot KV segments for its next capture. The existing owner, rewind and budget checks still apply. Writes to a shared segment detach before changing it.

Scheduler snapshot storage counters include both RoPE profiles. The expert arena counter reports the loaded RAM payload or the resident complement when experts use mmap.

Profile-aware tuner scopes keep measurements for different RoPE contexts separate. Setup warms the native graphs and the first YaRN request captures the graphs it uses.

With context profiles enabled, native RoPE is capped at 262,144 cells and YaRN ×2 at 524,288. The profile and adapter select the parked cache and the owning RAM recovery or NVMe namespace. Each persistence tier keeps its shared byte budget across the enabled namespaces.

A shared KV lease records its RoPE context. Batch windows group slots with compatible RoPE and use the cursor to give skipped slots the next window. Active prefill workers inherit their request context.

On the shared cold recovery path, an idle text owner in slot zero can keep its existing lease across a simultaneous RoPE and adapter change when the next request fits in a nonzero slot. The solo checkpoint chain and draft store become ordinary cached slot state under the outgoing key. This handoff does not require a RAM recovery image.

Idle RAM recovery gives pending input priority during both cold and retained copies. Interrupted captures leave the last committed image available. Ordinary recovery publication continues after an uninterrupted capture.

Windows without a Swift row use the original verifier graph variant. A Swift window has a separate capture key and reads row-specific scales in batch and a device scalar in solo.

Configure `--swift-lora /absolute/swift.gguf` to enable Swift, with `--atomic-lora` available independently. The two request selectors identify base, Uncen, Swift and combined states. Solo and prompt stages bind fixed device strengths. Shared decode stages per-row strengths, including when a window contains different modes. Prompt affinity, live KV, RAM parking, disk cache and recovery use the complete mode. Disk and recovery each retain one shared byte budget across their enabled adapter namespaces.

INT8 with `int8_kv_rot=1` has its own disk and RAM recovery identity. Snapshot metadata also distinguishes its rotated K/V representation from plain INT8.

This folder owns native request admission, prompt and decode state, image encoding and temporary GPU loans.

Shared decode groups text slots in the same verifier window. Image slots retain their own rotary position tables and take individual windows in the same rotation. A window that changes the device table drains GPU work before switching and restores the current prefill table before returning to an interrupted prompt read. The slot releases its host table when it settles. The shared_rope_audit experiment logs ownership and positions for isolated synthetic overlap tests.

For isolated synthetic tests, the shared_kv_audit_parked experiment fingerprints each parked slot at a shared-KV settlement fence. It reads committed pages and session state and synchronizes devices. Use it only in qualification builds because a full digest adds GPU-to-host traffic and CPU work. The dispatcher uses the loader's sessions, expert tiers and model geometry.

The image request prepares image rows and mRoPE positions before prefill. The main card injects image rows into the first stage. Each temporary peer stage binds a local copy of the request positions until the stage has stopped, then restores its previous binding. Position copies use CUDA peer access when supported and otherwise upload the host copy. Text routing and the layer pricing policy are shared with image requests. A two-card layer split that exceeds peer capacity tries smaller peer layer ranges at attention-group boundaries before falling back to the main card.

Opt-in direct activation and state transfers use peer permissions on the mapped segments covering the stage loan. The permissions expire when the segments are unmapped. CUDA work and expert restoration retain their existing request boundaries.

For synthetic profiling, `--exp prefill_route_log=1` adds a binary histogram beside the configured draft log. Records contain two uint64 values (fresh token count and entry count), then uint32 routed counts in layer-major order.

An explicit `--no-mtp` can retain a multi-row target verifier without the embedded drafter. Use `--suffix-draft 0` to isolate target-only decoding.

The optional FP16 draft relay packs only rows consumed by MTP. The target session and its checkpoint state retain their original representations.

`mtp_longer_proposals` scales the solo MTP proposal threshold consistently on the host and device. It leaves the verifier limit, target sampling and request-local acceptance feedback in place.

`park_evicted_kv` can reuse an exact prefix from a snapshot the RAM cache was already evicting. The option is restricted to long single-session captures without a layer split or disk spill. The previous entry is eligible only when the running session actually kept its prefix.

`park_share_kv` borrows unchanged segments from a donor still parked after normal RAM admission. The same owner and rewrite bounds as eviction reuse apply. Physical admission subtracts only uniquely owned retained backing, and shared segments detach before any write.

With `park_move_short_checkpoints`, a full single-session switch at up to 4096 live tokens moves the existing host checkpoints into the RAM entry after capture and admission. Rewinds, incoming images, layer splits and disk parking keep their existing copies. A refused or failed insertion returns ownership to the active session.

The compact DFlash2 variant offers up to three proposals and still captures every verifier row. The request sampler settings are available to its opt-in proposal policy. Confidence changes the proposed chain, while target verification and sampling retain their existing policy.

`--exp dflash_suffix_first=1` lets the existing suffix policy choose a strong repeated continuation before launching a neural draft. The verified feature ring still consumes the committed target rows and the target verifies the selected suffix tokens normally.

`--exp dflash_cost_policy=1` passes the existing verifier cost estimate to the neural selector. Fixed-confidence selection remains the default.

`--exp dflash_overlap_commit=1` overlaps draft work with phase-one target commit after the first emitted token. Suffix-first keeps its early emission ordering.

With `--exp thinking_max=1`, single-slot unsplit Flash-Next requests route thinking through a separate K=23 verifier. Prompt rows and the closing marker use K=10. Both verifiers use the configured peer expert tier and prepare their own graphs before READY. Without inner MTP, the drafter ingests committed thinking rows until ordinary decoding resumes.

The research switch `thinking_max_mtp=1` enables MTP inside that block for greedy and sampled text without grammar or logit bias. Thinking accepts only drafts that match the target draw. It checks at most five rows and ends the window before any proposed closing token. Other requests retain the one-row thinking path.

The optional native `--atomic-lora` path loads a validated Flash-Next adapter at startup. Embeddings, prefill and verifier bindings share the resident device factors. The startup scale is zero. Adapted requests still require the request binding and cache isolation gate.

AtomicChat variants park the outgoing session before switching scales. The base and adapted variants have independent RAM caches and disk namespaces. Both disk namespaces share the configured byte budget and evict the least recently used conversations across modes. Disk identity includes the entire loaded factor set and the scale.

Temporary peer prefill stages track the selected adapter scale for streamed expert routing.

Max thinking uses `--target-k-thinking-num` (10–23, default 16) on the final `--target-k-last-layer-num` layers (1–48, default 16). Earlier layers use K=10. Inner MTP fits up to eight rows in 128 routed pairs per layer.

When a peer prefill stage overwrites a shared FP16 sequence’s host pages, handoff invalidates only the physical pages covered by that sequence’s prefill range after the peer work completes.

Shared KV batch scheduling rotates across runnable sequences and fits each window to its verifier row budget. Shared KV batch completion settles the lease before BDONE. A new reservation can evict idle leases under pressure. Reclamation fences the main device and uses the stable sequence tables to invalidate pages.

The experimental `--exp adaptive_decode_rows=1` option compares the existing speculative row budget against a nearby budget in stable text decode batches. It uses committed tokens over complete window time, including draft and commit work, and tracks each slot's emission timing and token rate. Prefill, vision, slot changes and context buckets reset the trial. The verifier's physical capacity, slot rotation and at least one target row per selected slot still apply.

The experimental `adaptive_engine_scheduler` policy tracks phase transitions across solo prefill, overlapping prefill, solo decode and batch decode. Overlapping prefill's chunk and decode budget take turns adjusting at measured native emission boundaries. Batch trials use the live queue pressure class when available and reset when it changes. The runtime publishes coalesced phase, active knobs, latest decision, queue and expert routing counters on the `SCHEDSTAT` telemetry line.

Scheduler snapshots include mapped shared indexer bytes without counting alias pointers twice, pinned KV, pinned peer weights and file bytes from both adapter namespaces. Peer prefill scratch and vision loans are slices of the expert bank. Dense arenas and adapter payload sizes come from the initialized runtime.

The pinned KV counter includes allocations from the target and draft paths. Prefill and vision layouts can overlap their loan spans. The latest batch output gap is the largest same-request interval observed in the last completed native batch window.

The coordinated overlap policy keeps the old decode budget as its floor and measures committed decode tokens over each prefill/decode cycle. A drop below the calibration rate favors a larger decode quantum. Chunk downshifts still take precedence when emission latency is high.

An optional `adaptive_scheduler_warm_start=1` retains up to sixteen decode trial winners across cohorts within the running process. A sufficiently recent winner for the same workload can become the first trial after fresh calibration. The cohort's prior timings and request identity never enter the table. A remembered setting requires the same promotion checks, and a failed replay removes it.

Warm trials require fresh queue advice. The profile groups context length, slot count, physical row capacity, adapter mix and queue pressure. Instantaneous draft demand only determines whether a saved setting has an effect.

Decode probes move one row at a time. A trial cannot qualify when the early and late incumbent blocks disagree on throughput. A promoted limit returns to physical capacity and recalibrates when demand exceeds what was observed during its trial.

Set `adaptive_decode_rows=0` explicitly for observation with the physical row limit. The scheduler reports completed joint batch-window count and time, with a subtotal for windows where an adaptive cap actually removed a requested row. These counters are process-wide and exclude overlap with prefill.

Grouped CPU experts can borrow the decode window adapter scale vector. Dispatch copies each original token’s scale into its expert job row before starting the workers.

The request finishing prefill binds its adapter scale to the verifier slot before the slot enters a decode window.

A shared KV lease records the adapter mode. Changing modes on a slot retires the parked prefix and fences GPU work before publishing new pages.

A peer prefill stage created after adapter selection initializes both its device adapter bindings and its CPU expert scale from the active prompt mode.

Shared batch MTP binds the request scale to its draft store. Store switches update mapped embedding scales after the previous stream drains.

Shared KV startup binds AtomicChat PLE on every slot and advertises mixed adapter admission when the adapter is loaded. Native restore accepts a slot checkpoint only if its retained adapter matches the request.

Shared KV prompt reads alternate prefill and decode at chunk boundaries because both paths mutate the same device residency metadata.

After the last decode slot settles during a fresh shared prompt read, a completed chunk can hand the remaining tokens to the normal solo planner. The last batch decoder's adaptive expert transactions are settled before the request resizes its scratch loan. The request can then run the suffix across both GPUs using the prefix already committed to its session.

Request views retain stable addresses when dynamic slot growth appends a view.

Runtime shared-slot registration creates a borrower of the main FP16 residency map and publishes its scheduler state after verifier and MTP registration succeed.

The shared KV worker advertises `shared_kv_grow=1`. At a command boundary, `BGROW` requests one additional slot. The reply is `BGROWN <slot>` or `BGROWN -1` when the worker cannot grow. The caller serializes the operation with request dispatch.

When a shared slot loses its parked prefix, the scheduler drains and releases its draft backup ring and deferred rows. The graph working state stays allocated.

A parked prefix larger than the newly admitted extent is retired before publication, so it cannot leave the slot holding pages beyond its admission charge.

Shared decode bounds each GPU window by the worst-case number of selected QSA pages per row and the physical resident page count. The logical queue can still hold more sequences than a window can run.

The shared AtomicChat batch summary reports how many completed verifier windows contained both adapter modes.

Shared slot admission budgets seed and main draft rings and prepares them along with indexer backing before publishing its KV lease. Expert lending accounts for the existing CUDA reserve before mapping this backing, leaving space for lazily loaded prefill kernels. Startup warms the shared main solo commit graph while the stable session addresses are available and before long requests grow their backing. Prefix parking keeps the backing for retained cells, and slot retirement releases it after draining GPU work.

If indexer backing cannot grow, admission retires idle prefix leases one at a time and retries before reporting resource pressure. Active leases remain live.

Single-sequence shared decode windows let MTP consume the verifier output directly. The window still flushes its draft store and drains the MTP stream before verifier reuse. Mixed windows keep the activation snapshot.

Shared native SGEN on slot zero settles its lease before DONE. The retained prefix is the consumed solo sequence. An interrupted prefill releases its lease.

With shared_kv_expert_lending enabled, shared slot and indexer growth can reclaim primary expert-cache segments after draining readers and adaptive copies. If CUDA refuses a new draft ring for lack of memory despite the estimated fit, growth may reclaim another segment and retry once before publishing the slot. Idle command boundaries refill evicted expert slots from RAM when backing is available again.

The shared_kv_live_prefill experiment applies the existing chunk interleave to fresh shared-pool prompt parts over 1,024 tokens when decode slots are active. Auto prefill sizes the interleaved layout for up to 2,048 tokens under the existing expert-loan ceiling. The request uses 1,024-token slices initially, then adjusts in 256-token steps within the layout using a reference from the first three native decode gaps. It reduces the slice on slow gaps and grows only after four observations with margin. Slice changes reuse the current layout. Idle prefill keeps the existing placement and fit planner.

With `adaptive_prefill_quantum=1`, the decode share moves gradually between 25 and 75 percent of the last slice's elapsed reading time based on margin to the observed decode gap. An interleaved slice still runs at least one decode window. With `adaptive_short_prefill=1`, complete single-part solo text readings near the short-read cutoff inform the choice between verifier and batched prefill. Estimates are separated by adapter and KV residency and require four observations on both paths. The configured cutoff remains an upper bound for verifier reads.

The isolated tuner can hold a fixed shared-overlap chunk with `tuner_overlap_chunk` or a fixed decode share with `tuner_overlap_decode_pct`. Zero uses the normal controller. Positive chunk requests align to 256 and clamp to the prepared layout and existing loan limits. Positive decode share clamps to 25–100 percent of the preceding slice and takes effect only between complete windows. Set the other dimension to a fixed value when comparing a sweep. These controls apply only to the existing adaptive shared-overlap path. The trial log records the requested and effective slice, allocated layout, measured decode windows and actual decode time for a completed overlap segment.

`tuner_overlap_trial_chunk` is an opt-in online replay of an independently qualified chunk. Two unscored incumbent slices let the overlap settle. The trial measures the incumbent, tries the proposed chunk, then measures the incumbent again. Diagnostics distinguish cohort and queue changes, invalid observations and baseline drift. After promotion the immediate gap guard uses the envelope of both matched incumbent blocks. Sustained rate checks still compare against each incumbent. An optional positive `tuner_overlap_gap_ms` supplies an independently qualified absolute gap budget for the experiment, shared by both measured incumbent blocks and the candidate. Complete cycles run at 50 percent decode share. A trial admits only an unchanged shared decoder cohort, the ready layout and fresh empty-queue advice. The trial can start at the next eligible slice boundary once admission advice clears. It rejects proposals outside the prepared layout, compares lead progress for each slot and restores the usual controller when the proposal fails. The decision is reported in the scheduler diagnostics. A changed cohort invalidates its timing samples. The explicit experiment revalidates its input each time it enters a compatible overlap.

Idle shared slots park and spill through the adapter that owns the slot's retained KV, even while a different adapter is bound for request prefill. Shutdown drains the paired RAM and disk namespaces using the same ownership.
Shutdown flushes the writer before submitting each image and can use the configured disk budget for one physically admitted capture. Background persistence keeps the 4 GiB in-memory queue limit.

ENCA jobs use one background encoder. During peer prefill the vision arena sits below the stage weights and above the session buffers within the stage's existing expert-tail loan. The weights keep their startup addresses. During decode the scheduler drains the current window and lends a temporary peer tail. Both paths close vision admission and drain its stream before restoring experts. Reply lines carry the job identity.

Retiring the shared lease for slot zero clears both its batch metadata and the solo conversation chain. The checkpoint selector verifies the shared page extent before borrowing a prefix from another slot.

When BASALT_RAM_RECOVERY_DIR points to a tmpfs directory, idle dispatch publishes previous completed conversation state there using the existing snapshot format. The two adapters share its budget. Restore uses the best compatible prefix from the RAM recovery and NVMe stores.

An optional qualified overlap hint starts as an advisory and is scoped to text requests with matching prefill adapter, context, work size, prepared layout and active decoder shapes. It excludes decoder sequences that retain image coordinates and pauses hinted trials on vision work. It arms only under the existing fresh empty-queue gate and uses the full calibration and confirmation path at 50% decode share. A valid rejected observation clears the hint for this worker and reports its scope for durable retirement. Cohort interruption only discards the trial.

The independent online learning experiment starts one adjacent 256-token candidate above the incumbent when a fresh empty-queue text overlap has enough work and the prepared layout fits. An existing scoped candidate can replace that first probe and is revalidated. A complete winning episode requires two healthy post-promotion blocks, a stable decoder cohort, successful prompt settlement and refill, and a completed gap within the calibrated envelope through each old decoder's first emission after settlement. The incomplete last prefill slice does not contribute a rate sample. The native process emits at most one learning outcome for a request. A rejected candidate is disabled for the compatible scope in the current process.

A shared FP16 request can reuse a parked checkpoint when the full historical branch exceeds its admitted host pages. The restore path keeps the granted lease and trims only the incoming host image to the validated checkpoint that fits. Peer prefill does not change the single decode-stage shared pool contract.

For a promoted autonomous overlap trial, the final chunk can be divided at a 256-token tail so the decoders receive an ordinary window before checkpoint/refill. The controller checks that partial cycle against the existing gap limit and excludes it from the rate comparison. Native logs the terminal slice and actual next-emission settlement gap.

With RAM caching enabled, ordinary idle sessions no longer get an extra NVMe snapshot. Existing eviction, disk fallback and shutdown paths still persist eligible images. Idle RAM recovery queues one captured image without waiting for its tmpfs publication.

Idle snapshot attempts back off for ten seconds after RAM admission refuses every capture. The RAM recovery and disk tiers track refusals independently, report the first refusal in each episode, and recheck admission on retry. Shutdown spills still use the normal admission checks.

The opt-in grouped_short_prefill experiment lets compatible active slots share a grouped verifier window for short resident text tails. Each segment commits its known tokens with its own KV and draft state. A prompt window alternates with decode when both are runnable. The feature starts a group only for a cold short prompt with another short waiter. A compatible pending prompt can accept the last cold arrival without a waiter. Warm restores and isolated followers keep their ordinary prompt read.

The worker reports isolated startup H2D copy results in decimal MB/s by CUDA ordinal. These values describe a warmed CUDA copy-engine calibration. A missing probe omits the field.

With adaptive_pcie_fraction enabled, steady device-planned multi-slot decode uses a per-layer makespan model. It updates estimates from completed compatible windows and looks up the staged count using the current routed miss count. The 20% soft ceiling can be exceeded only with measured copy/compute and enough timing margin. The optional pcie_tuner_cap is a hard fractional ceiling. Grouped prefill uses zero adaptive staging. Scheduler telemetry reports the realized fraction and payload.

The opt-in comparison gate pcie_tuner_pin_q8 uses fixed staging tables in units of 1/256. It skips model timing and retains the hard cap. Automatic estimates reset on verifier row-shape changes. Expert placement refreshes the mapped aliases and current miss counts. The finished batch logs counts for each budget decision, including censored peer timing. No separate trial requests run during serving.

Direct P2P decode supplies budget timing through the existing peer done-word wait, including when FP16 peer rows are enabled. The realized fraction still reflects completed expert staging.

A shared resident continuation may join a grouped short tail after its cached prefix is restored in the owning slot. Native bounds the actual suffix, checks the restored position and leaves partially resumed reads and checkpoint boundaries on their existing paths.

Batch suffix lookup sizes each slot’s history and trigram index from its full prepared prompt and decode limit. It includes deferred prompt rows and the seed token, and retains the configured context ceiling for oversized native limits.

The solo suffix index is initialized for each request at decode entry using its prepared prompt and output limit. Startup does not allocate a context-wide table, and a short request does not clear the previous request’s table.

Cold solo RAM recovery without retained KV can yield at KV copy boundaries to a queued text request explicitly switching RoPE profiles. Before changing profiles, the completed outgoing park queues a shared view in the original namespace if it needs recovery. The handoff also covers input arriving before the idle capture begins. Existing recovery and cache budgets govern admission.

An immediately following RoPE switch can consume the outgoing owner’s complete idle recovery image without another draft or KV capture. The full image is retained only within the existing RAM cache budget.

The isolated `cold_shared_recovery=1` path holds slot-zero target pages after a completed solo text turn while admitting a nonzero BGEN on another RoPE profile with the same adapter. It seals the draft and running state before switching, overlaps the device index read and CPU target copy with the replacement request, then retires the source before slot-zero publication, a lookup in the outgoing cache namespace, maintenance or shutdown. Text BGEN in another namespace can publish on a separate slot while the copy runs. BGROW may allocate another table and session over the unchanged backing. STOP and BSTOP remain immediate. Pressure reclaims other idle slots first and waits for the frozen lease only when needed. A process crash before the worker publishes retains only the previous durable recovery prefix.

With one slot, a compatible engine can advertise a minimum retained prefix and accept BGROW with the next RoPE profile. The hinted command yields the disposable cold idle copy, and native growth verifies the committed solo owner before allocating. A refused growth keeps the ordinary slot-zero admission path. The worker sends the frozen source page count and cache namespace in an INFO message before acknowledging the first replacement. It clears the count after the source lease settles, including quota and source-namespace retirement.

The cold recovery seal takes the page-rounded incoming budget from mapped draft KV and defers the rest. An overlapping old-slot flush or restore waits for draft preparation independently of target capture.
