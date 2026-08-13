# Artifact Quickstart

Run `bash scripts/run_smoke.sh` from the extracted package root. The check
verifies the figure/table manifest, the corrected SESRec latest-slot bootstrap,
three UniSAR latest-slot checks, the C0-versus-C1 observed-support control rows,
the nine-run planted focal control, the GenSAR PerSrc current-query
calibration, the GenSAR SeqRec native-quality boundary, the C2 source-slot
stratification, historical neutral-mask
provenance, reverse-source diagnostics, all six
common-surface SESRec/UniSAR rows, Python syntax, and local-module dependency
closure. It requires no private data, checkpoint, or third-party Python package.

Optional plotting dependencies are listed in `requirements.txt`. Raw datasets,
checkpoints, training logs, and row-level score exports are intentionally
omitted because they are either third-party, large, or potentially identifying.

After installing the optional dependencies, run
`python3 scripts/check_tool_imports.py` to load every packaged analysis module
without executing its data-dependent main routine.
