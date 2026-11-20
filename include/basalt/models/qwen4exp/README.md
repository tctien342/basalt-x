# Flash-Next public native headers

This folder defines the model geometry, options and contracts used by the Flash-Next adapter. Decode and prefill headers describe separate engine stages. Options carry model pack and draft selection into the loader.

`SwiftLora` checks rank-32 BF16 factors against the supplied base shapes and hashes their contents for cache identity. `SwiftLoraDevice` owns immutable uploads for one device.

The --dflash2 DIR option selects a native Q4_0 drafter and disables embedded MTP. The --no-mtp option can be used on its own. The verifier and target KV settings are controlled separately. These declarations perform no I/O.

`AtomicLora` validates the rank-one GGUF factors against the base writer shapes and owns compact host buffers. It shares one output vector across all writers and experts. The loader has no GPU or request side effects.

The optional device adapter owns one immutable upload per CUDA device and a fixed-address request-scale scalar. Before updating that scalar, the caller must drain previous users on the device. A zero device scale leaves output rows untouched.

The optional persistent peer LoRA path uses a resident per-layer factor table and applies the routed delta before writing its FP32 or FP16 result and signaling completion.

The optional native `--atomic-lora` path loads a validated Flash-Next adapter at startup. Embeddings, prefill and verifier bindings share the resident device factors. The startup scale is zero. Adapted requests still require the request binding and cache isolation gate.

AtomicChat exposes an identity hash over the loaded adapter factors for cache keys.

The native max-thinking tuning flags select target K and the number of final layers at that width. Defaults are 16 experts and 16 layers.
