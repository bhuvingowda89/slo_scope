# SLOScope Publication Artifact

This artifact supports three reproduction levels.

## Level 1: Analysis reproduction
Verify scientific inputs and regenerate manuscript-facing registries, tables, figures, and consistency checks without running inference:

```bash
/private/tmp/slo_phase7a1_venv/bin/python paper/provenance/build_paper_artifacts.py
python3 paper/artifact/verify_artifact.py
python3 paper/provenance/verify_manuscript_numbers.py
```

## Level 2: Small smoke test
Use the benchmark CLI with a small local condition and a local llama.cpp server. This validates installation and artifact writing, not the formal claims.

## Level 3: Full reproduction
Follow the frozen campaign manifests for Phase6-V2, Phase7D, and Phase7F. Full reproduction requires substantial local model/runtime execution and should not be needed for manuscript-number verification.

The authoritative corrected analyses are Phase 7A.1, 7B, 7C, 7C.1a, 7D.2a, 7E, and 7F.
