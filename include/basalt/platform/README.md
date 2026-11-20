# Platform interfaces

These headers expose file, mapping, CPU topology, process and system APIs implemented in `src/platform`.

A caller using shared memory regions owns population coordination. It may query or publish a ready marker while holding the population lock. The caller releases its mapping with `free_region`.

Request a ready-only mapping to require a published file and preserve its backing header.
