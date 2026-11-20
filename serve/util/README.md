# Serving utilities

- `telemetry.py` uses NVML and optionally psutil to sample hardware for `GET /metrics`. Linux PCIe path capacity is read once for a direct endpoint-to-root-port path. Pick the common speed and width of the two link ends. More complex topology is reported as unavailable. It is a per-direction line-rate ceiling after line coding, before packet overhead. Runtime RX and TX readings are sampled independently.
- `console.py`, `loglevel.py`, and `logbuf.py` format, classify and retain bounded server or engine log messages.
- `text.py` implements tokenizer helpers, streaming detokenization and stop matching.

The telemetry sampler and optional log tail use background threads. Missing device fields, unsupported PCIe encodings and unreadable sysfs paths produce unavailable readings. No utility changes hardware settings.
