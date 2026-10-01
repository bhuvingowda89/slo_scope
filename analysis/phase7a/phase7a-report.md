# Phase 7A Primary Analysis Report

campaign_id: `sloscope-phase6-v2`
phase7a_input_sha256: `6859c90df0efefef3ff0491aa7f4b9cad190af184802971fd1244920ad46af4b`
phase7a_analysis_spec_sha256: `91d4c95c839395f41f0fba31df18cbe6b7d2fb188bf8b1698b7d499ee3b407c7`
analyzed_runs: 104

## SLO Thresholds

- TTFT_SLO: 0.06512947314299491 s
- TOTAL_LATENCY_SLO: 0.26779416389490324 s
- DECODE_DURATION_SLO: 0.20162438542758712 s
- Throughput: diagnostic only

## RQ1

Direct mechanism evidence, SLO effects, and paired statistical evidence are reported separately in `direct-evidence.csv` and `isolated-effects.csv`. Diagnostic separability is deferred to RCA/CSD phases.

## RQ2

SLO violation frequencies are in `slo-violations.csv`; additive factorial interaction contrasts are in `factorial-interactions.csv`. No CSD/RCA analysis is performed here.

## Baseline/Control SLO Crossings

7 baseline/control runs crossed at least one counted SLO.

## Limitations

- Analysis uses run/repetition as the inferential unit (n=8 per condition).
- SciPy was unavailable, so Student-t p-values were computed with a deterministic numerical integration implementation recorded in the analysis code.
- Server process CPU/RSS fields are null in the formal artifacts because server PID telemetry was not supplied in the formal configs; host CPU telemetry remains available.
