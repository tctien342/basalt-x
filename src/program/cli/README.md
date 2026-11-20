# Basalt CLI

Argument parsing, GPU selection and help text are shared by Basalt's model adapters. The arguments.cpp parser dispatches model-specific flags and help.cpp documents them. The startup and serve modules prepare process state. GPU helpers select and configure the cards.

Put `--check-model-flags` first to select the model adapter and parse arguments without reading its vocabulary or planning GPU placement. The bridge uses this mode for inherited flag checks.

Flash-Next accepts --no-mtp and --dflash2 DIR. The latter also disables embedded MTP. The adapter validates the draft and serving topology after argument parsing. The max-thinking target K and final-layer count are documented in native help.
