# Flash-Next native model adapter

This folder wires the Flash-Next engine into Basalt. The flags.cpp parser accepts architecture options, the loader selects target and drafter resources, and serve/ drives the native protocol. The qwen4exp.cmake file links the model's kernels, planner, loader and prefill libraries.

`SwiftLora` checks rank-32 BF16 factors against the supplied base shapes and hashes their contents for cache identity. `SwiftLoraDevice` owns immutable uploads for one device.

The --dflash2 DIR option expects a native Q4_0 pack and single-slot serving with a shared target head. The packed MTP layer is disabled before allocation. The target verifier retains its sampling and KV settings. See the load and decode guides for allocated resources.

`AtomicLora` validates the rank-one GGUF factors against the base writer shapes and owns compact host buffers. It shares one output vector across all writers and experts. The loader has no GPU or request side effects.

The optional device adapter owns one immutable upload per CUDA device and a fixed-address request-scale scalar. Before updating that scalar, the caller must drain previous users on the device. A zero device scale leaves output rows untouched.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The primary verifier applies the optional routed LoRA correction from the grouped gate/up scratch after down projection. Destination indices preserve router order for both cached and staged expert rows.

The optional native `--atomic-lora` path loads a validated Flash-Next adapter at startup. Embeddings, prefill and verifier bindings share the resident device factors. The startup scale is zero. Adapted requests still require the request binding and cache isolation gate.

AtomicChat disk state is keyed by the loaded factor contents.

The native max-thinking tuning flags select target K and the number of final layers at that width. Defaults are 16 experts and 16 layers.
