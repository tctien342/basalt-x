# Platform implementation

This directory implements platform memory mapping, system and process APIs, file I/O and CPU topology. Standalone callers use the interfaces under `include/basalt/platform`.

Linux shared host regions store a pack identity and a payload publication marker in their header. The expert source locks population and publishes only after writing the whole payload. The file survives the native process. Allocators unmap process-local views during teardown.

Ready-only open under the population lock requires an existing published file and validates the stored model identity before mapping it.
