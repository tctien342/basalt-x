# Persistent serving

A model-specific context cap applies after image placement and is intersected with the configured engine or slot limit. Prompt clipping, prefill and output admission use that limit.

`Runtime` (`state.hpp`) holds what every model shares: the input, the conversation chain, the slots. `Request`
(`request.hpp`) runs one request's phases in order. A model answers through their virtual calls, which run per request,
per command or per batch window; `Request::decode_loop` (`decode_loop.hpp`) is instantiated once per model, so a window
pays no indirect call. `Flow::next` returns to command dispatch; `Flow::error` ends serving; `Flow::quit` runs shutdown.

| Concern | Implementation |
| --- | --- |
| Lifecycle and shutdown | `runtime.cpp` |
| Command dispatch and scheduling | `commands.cpp` |
| Input framing, tokenization and watchdog | `protocol.cpp` |
| Request order | `request.cpp` |
| Request keys, token ids and grammar | `request_parse.cpp` |
| Prompt limits | `request_image.cpp` |
| Which source holds the prompt's start | `request_restore.cpp` |
| Sampling setup, turn and root parts | `request_prefill.cpp` |
| The window's order | `decode_loop.hpp`, `reply_rows.hpp` |
| Completion, slot admission and reports | `request_finish.cpp` |
| A model that offers only sessions | `session_engine.cpp` |

A model with its own serving derives from both classes in its own folder; see `src/models/qwen4exp/serve/`.

Request restore checks the model's slot-prefix compatibility before selecting slot state or checkpoints.

The protocol text and error paths are shared contracts with the Python server and the benchmark client. A structural
change preserves those contracts and is checked against a frozen engine with fixed expert placement, including batch
slots, cancellation and conversation restoration.

An optional `SCHEDQ <control> <admission> <parked> <oldest_ms> <free_pages>` line supplies advisory scheduler pressure. The input reader updates a timestamped snapshot. It does not enqueue the line with inference commands or send a reply.

A GEN request with `think_close` may contain widened decode state. Its final live state is ineligible for prompt reuse. Checkpoints produced while reading the prompt remain valid.

The internal `atomic_lora=0|1` and `swift_lora=0|1` selectors are validated before prompt preparation and cache lookup. The adapter mode includes both selectors. Native runtimes reject adapted requests unless they implement the binding hook.

The internal `rope_profile=0|1` selector is validated before adapter binding. Native runtimes reject an explicit profile until they implement the context-profile binding hook. Requests without a selector use the legacy startup configuration.

Request cache reads and spills use the active disk-store accessor. A model can select a different store before restore.

Context admission checks the expanded prompt and output cap after media placement, including eight cells of native slack. Oversized caps are rejected before arithmetic can overflow.

The shared KV worker advertises `shared_kv_grow=1`. At a command boundary, `BGROW` requests one additional slot. The reply is `BGROWN <slot>` or `BGROWN -1` when the worker cannot grow. The caller serializes the operation with request dispatch.

Disk snapshot freshness checks can take the namespace that owns an idle slot, so a model can spill that slot while another adapter is bound.

The Flash-Next worker may advertise independent vision. The stdin reader admits an ENCA job into an already fenced peer-prefill loan, or queues it for the scheduler to lend a tail between decode windows. The idle scheduler polls outstanding loans so experts can be restored if no request follows.

Cross-slot checkpoint selection can reject a candidate whose native backing is no longer published. Such candidates do not shadow usable RAM or disk prefixes.

The idle dispatcher publishes an engine's opt-in RAM recovery checkpoints before ordinary conversation spills. Prompt restore compares their prefix with the normal cache and uses the same snapshot validation. The write check also accepts an exact stored image, so a conversation with no earlier checkpoint is only captured once while it remains in the store.

Idle dispatch checks for pending commands before and after at most one background snapshot submission per timeout. Snapshot capture still runs on the owning engine thread.

An eligible Flash-Next BGEN can publish a slot with a deferred short text tail after restore. Its known tokens remain request-local. The internal group_prefill hint is advisory. It carries control-queue demand from the bridge. BADM transfers ownership to the slot. BPF reports completed prompt work before the first BT.

Idle dispatch checks protocol input between RAM recovery and any optional disk spill or KV parking.

Dispatch stamps each dequeued protocol command. Model caches can require an immediate successor before reusing a whole running-state image.

After reading a command, the model can reconcile a borrowed snapshot source that still owns an old slot. For Flash-Next text BGEN, final source retirement may be deferred to validated slot publication or cache namespace selection. BSTOP and the no-op BYIELD bypass the fence. Reader-side STOP remains out of band.

BGROW can carry a 0 or 1 RoPE hint on Flash-Next builds that advertise cold shared recovery routing. The result is still BGROWN. Older bridges continue to issue plain BGROW.
