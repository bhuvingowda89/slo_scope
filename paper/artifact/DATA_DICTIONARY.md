# Data Dictionary

`requests.parquet`: per measured request timing, requested output, request identity, and outcome.

`traces.parquet`: gateway, dependency, and llama span timing keyed by request identifier where enabled.

`runtime_metrics.parquet`: llama.cpp runtime metric scrape rows where enabled. Formal queue-depth rows required for queue RCA were unavailable in source artifacts.

`system_metrics.parquet`: host/system telemetry rows collected by the measurement stack.

`experimental_condition.json`: frozen condition contract and active mechanism labels.

`publication-run-index.jsonl`: authoritative eligible run list for a formal campaign.

Derived analysis tables include run-level metrics, factorial interactions, CSD signatures, RCA predictions, degradation predictions, and telemetry cost metrics. Ground-truth labels are stored separately from model feature matrices.
