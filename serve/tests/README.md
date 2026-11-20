# API serving tests

The shared profile regression admits a 512K alias beside active 256K work and prefers its matching slot over a session hint from the other profile.

The four-alias regression checks HTTP model registration, response names and wire selectors. Slot selection also checks identical prefixes across all four adapter modes.

Run focused suites from the source root using the server Python environment. The tests use scripted engines for API, prompt, parser, cancellation and accounting contracts.

`test_independent_vision.py` covers tagged asynchronous replies alongside solo and batch output, engine EOF, refusal and timeout.

`test_thinking_batch.py` covers ordered exact arithmetic, ISO timestamp differences, batch-local names, tool limits, streamed tag handling, stable prompt instructions and exact token-prefix extension. It also covers private schema routing, reasoning after a tool result, chained batches and Responses streaming when a standalone batch follows visible text. The scripted engine does not measure native KV reuse.

The admission tests in `test_concurrency.py` cover capacity and adapter-mode blocking, one bypass, and granting after the active cohort drains. Slot-selection tests in `test_parallel.py` verify that a matching token prefix in a different adapter receives no affinity.

A shared admission regression covers a short request whose previous turn changed while a larger idle prefix remains in another slot.

The scripted shared-pool test grants an adapted request while a base request still holds a reservation and verifies the native command carries its adapter mode.

Cancellation after a successful growth reply leaves the new slot idle and available for the next request. The frozen-source fixture runs two consecutive replacements while slot 1 remains active, and verifies bypass with a fitting alternate plus fallback after pressure or growth refusal.

A growth-refusal test checks that the API keeps the waiter and resumes it on an existing slot after completion releases capacity.

Shared KV can start with one session and grow on demand. The default seed count is one when the pool is enabled.

The incremental shared-KV bridge test exercises two bounded streams beside an owner with a full grant, then verifies both streams resume with the same seed and emit each token exactly once after pages are returned. A second case cancels a parked stream and checks that it drops its admission ticket without reacquiring pages.

The shared backing retry test covers both indexer and draft ring refusals. Each provisional reservation is released and retried after another owner turns over.

Shared solo return tests retain slot zero's page charge through BDONE, exercise a waiter and late cancellation, and verify cleanup leaves a replacement engine's ledger intact.

The metrics HTTP tests cover the scheduler fields and the overlap learner's provisional, qualified and revoked states while profile persistence is in progress or has failed. They do not launch a GPU worker.

Context profile tests exercise six exact aliases, rejection of the combined alias, cap enforcement for text and token inputs, private wire keys, and the feature gate for older native workers.

Cold growth regressions cover independent routing, refusal, outgoing page accounting and ignoring a reply from an older worker generation.

The scripted mixed-alias admission cases check growth, reuse of an existing peer, return to the warm slot with or without a session key and the fallback when both page reservations do not fit.
