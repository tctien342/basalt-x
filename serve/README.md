# Basalt API server

Configuration discovery searches for a literal double dash and checks the printable-byte boundaries of native flags. Inherited parser-only flag checks run up to four independent native processes per model and apply their decisions in configuration order. For older natives, checks still run the GPU-plan probe serially. Model configuration always runs a full GPU plan.
Set `BASALT_PARSER_PROBE_CACHE` to a writable bridge-owned file to reuse successful inherited-flag checks across bridge restarts. The last validation still comes from the full GPU plan.

The server owns model configuration, request scheduling, API parsing and child engine lifetime. Adaptive vision is wired into the existing model lifecycle. It starts the CPU sidecar and enables native ENC on the peer tier. The Python and native engine should ship together.

The `scheduler` object in `GET /metrics` reports the latest native autotuner snapshot, with its arrival timestamp. It is null until a compatible loaded worker publishes a snapshot. Routing counters have separate units for resident lookups, routed token/expert entries and distinct PCIe experts. The separate `scheduler_learning` object reports current qualified and provisional overlap scope counts, asynchronous profile write state, and the latest accepted outcome from the bridge. It is null without a loaded learner. Its timestamp uses Unix seconds and its chunk field counts tokens.

`startup_memory.kv_allocations` records loader-measured target and draft session allocations with logical device, backed bytes, configured bytes and context cells. The scheduler snapshot additionally reports current shared indexer mappings, pinned allocations and store usage.

Flash-Next accepts explicit reasoning effort `max` for the widened thinking block. The usual xhigh template still opens the block. The API passes its closing token to the native engine while reasoning is open.

For a Flash-Next entry, native `--atomic-lora /absolute/adapter.gguf` registers `-uncen` and `--swift-lora /absolute/swift.gguf` registers `-swift`. Configuring both also registers `-uncen-swift`. All aliases share the base model’s service, queue and native child. The alias sets the private adapter selectors and is returned in API responses.

With the server flag `--mini-tools-helper` (off by default), Flash-Next thinking modes see the private `engine_mini_batch` schema in their tool list and use standard tool calls for up to eight calculator, datetime and regex calls at a time. Successful calculator IDs are available to later calculations in the same batch. The parser hides raw batch frames and emits one English reasoning notice after the engine drains the current segment. It accepts one standalone batch after the thinking delimiter and before the first client tool call, including after visible introductory text. Inline examples and fenced code retain their literal text. After a standard tool result the assistant resumes thinking, including when the call followed the thinking delimiter. In max mode the resumed segment uses max thinking decode settings until its closing token. Results are appended to the exact token prefix and count against the remaining context/output budget. Standard calls receive a template-formatted tool response and a new assistant generation prefix, both internal to the request.

The server JSON equivalent is `"server": {"mini-tools-helper": true}`. Disabled requests retain their existing prompt and parser behavior. The instruction prefix is stable across turns. It asks the model to verify every calculated quantity it reports, including incidental totals, estimates and conversions. `datetime.difference` accepts `start` and `end` as two ISO dates (result in calendar days) or two ISO timestamps (elapsed seconds, including fractions). Timestamps accept Z, numeric offsets or local time in the specified IANA zone, defaulting to UTC. Mini results are request-local. Native cache policy is unchanged. Standard thinking may continue from live KV. Max thinking may reuse the ordinary prompt checkpoints and recompute widened rows using the existing prompt path.

Upstream session affinity, answer-now and logprob streaming use the same request context as the private alias and mini-tool continuation. Alias reporting follows the requested served name. A private tool continuation keeps the session key, counts inserted tokens against the output cap and hides the raw tool frame from stream events. PDF reading needs the pinned pypdf dependency from requirements.txt.

Shared KV runs the standard verifier for max reasoning requests. The max request budget and mini-tool behavior still apply.

The experimental shared-kv-pool model option keeps max-context as the total pool. Parallel controls the requested logical sessions at startup, independent of verifier window width. The native loader must still fit each requested session’s private state.

Shared KV can start with one session and grow on demand. The default seed count is one when the pool is enabled.

An adaptive image-only request may encode while text is running when the native worker advertises an independent vision arena. The service retains the image-byte admission limit and attaches completed embeddings in the request's original image order.

When the native worker advertises RAM recovery and self-healing is enabled, the service checks each generation stream before token events reach the API. It holds consecutive identical IDs or repeated text characters until the run breaks. The guard requires 256 consecutive generated tokens in the run. The run is discarded when it reaches that threshold.

The BASALT-X monitor reads the resolved model alias and normalized reasoning effort from active slots, the request queue and newly completed history. An enabled request with no explicit effort reports `default`. Requests whose template disables thinking report `off`. The monitor ID links the same request across these views. API model IDs and generation settings are unchanged.

Startup and dynamic shared-slot attachment require matching rotation, format and physical backing. For INT8 the residency and host bindings include K, V and both scale arrays.

The optional `context-profiles` model setting enables shared KV. Base, Uncen and Swift serve 262144 contexts in the native RoPE profile. Their `-512k` aliases serve 524288 contexts in the extended profile. The setting replaces the combined alias, lists per-alias capacity and leaves physical admission capacity with the shared pool.

## Speech input

Set `--stt-model /path/to/nemotron.gguf` and make the matching transcribe.cpp
Python binding and shared library available. One resident CPU worker defaults
to eight threads (`--stt-threads`) and Vietnamese (`--stt-language vi-VN`).
Audio and video share one bounded FIFO before generation routing. Audio uses
`input_audio` or `audio_url`. Videos also get timestamped transcripts when
they have audio, unless `--no-stt-video` is set.
