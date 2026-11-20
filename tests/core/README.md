# Core tests

This folder covers model files, cache and memory ownership, and the native serving protocol. The source root CMake configuration registers the existing CPU and CUDA tests, with CUDA-only tests requiring a compatible device.

The standalone shared KV page and admission tests compile as C++20 with the repository include directory. They exercise capacity reservation and scheduler policy on CPU without launching a model or allocating GPU memory.

The CUDA conversation snapshot test also checks capture, restore and copy across different fragmented FP16 and rotated INT8 host page mappings, including rejection of an out-of-range page. The host transfer fixture checks mapped INT8 codes, scales and read-back before a real CUDA smoke run.

The CUDA snapshot fixture checks that selective prefill stages the correct physical host miss and resident hit for FP16 and INT8, leaving unselected pages untouched.

The disk conversation store test covers shared capacity borrowing and eviction across two and four adapter namespaces, stale match rejection and restore after a reopen.

The CPU shared KV table test verifies prefix coverage after partial publication, partial retirement, successful trim and a missing host page mapping.

The host snapshot validation test also checks that cutting a parked image to a retained checkpoint trims all target and draft layers, preserves the selected running state, and leaves device buffers untouched.

The conversation cache fixture covers whole recovery handoff, owner and command checks, dirty fallback and retained budget accounting.

The CUDA snapshot fixture also compares split shared INT8 capture with a synchronous capture across fragmented physical pages. The test changes the live directory and device index rows before the CPU worker copies the target KV. A frozen-index case checks deferred publication refusal, separate device rows for a replacement slot and byte-exact parity including the spare row.

The deferred-tail CUDA fixture overwrites the mutable draft prefix after seal and verifies all four dense draft arrays.

The host-transfer fixture injects pending input during cold and retained captures. It checks that capture cannot publish a partial image and that mapped INT8 copying yields between fragmented physical runs.
