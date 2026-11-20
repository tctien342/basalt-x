# Monitor web assets

Memory Detail reports RAM available to capture, committed and queued recovery separately, and both aggregate and current namespace prefix storage. The prefix budget applies per namespace. The recovery limit is shared.

The LoRA badges count active requests across the native and 512K context aliases.

The native API serves `index.html`, `app.js` and `app.css` as its status console. The browser polls metrics and slots, loads local fonts, and retains the active tab in session storage. The console displays the aggregate engine state, per-device metrics, request history and native logs without modifying request scheduling.

The Scheduler tab uses `GET /metrics` fields from the loaded worker. It reports live or last-sample state, policy choices, queue and shared KV, expert routing, VRAM and host memory. The GPU usage rows are whole-device observations. Engine-owned allocations are listed separately and loan rows refer to bytes already in an existing pool. A missing sample or measurement renders as a dash. The optional Overlap learning panel reports qualified and provisional scope counts, storage status and the last settled outcome. Profile write state is separate from in-memory qualification.

The KV startup section groups loader allocation records by device and separates target session state from draft KV. Its configured bytes are capacity figures. Runtime indexer mappings, expert pools, loans and snapshot stores retain their own measurement scopes.

The workspace rows retain arena ownership and borrowed-region labels. Pinned KV is the combined target and draft allocation counter.

The Scheduler labels a pending row-budget trial sourced from the process's previously validated workload settings.

The limited-window count and elapsed time cover completed joint decode windows where the row cap actually excluded requested work. They accumulate for the lifetime of the worker.

Scheduler Info lists warmed startup host-to-device copy measurements by CUDA ordinal and the simultaneous copy measurement when available. NVML physical GPU rows separate current negotiated link, sysfs path ceiling and live RX and TX rates. The PCIe box separates the batch and startup fractions, the trial value, the latest observed eligible and staged expert counts, and staged payload. Its window payload rate is staged bytes divided by full window time.
