# Phase 7A.1 Corrected Primary Analysis Report

campaign_id: `sloscope-phase6-v2`
phase7a1_input_sha256: `99482c58c4e30bc8316c42b61f4ed5af2f291b1528dcfbbe539e490faa59e51c`
phase7a1_analysis_spec_sha256: `135f1a170eb8640d7819a990a69c7978b087b8747b66d3a07f0222466cb8259f`
analyzed_runs: 104

## SLO Thresholds

- TTFT_SLO: 0.06512947314299491 s
- TOTAL_LATENCY_SLO: 0.26779416389490324 s
- DECODE_DURATION_SLO: 0.20162438542758712 s
- Throughput: diagnostic only

## RQ1

Direct mechanism evidence, matched-control metric effects, and absolute frozen-SLO crossings are reported separately. For OUTPUT, matched-control evidence is primary because formal OUTPUT_CONTROL crossed frozen SLO thresholds.

## RQ2

Matched factorial interaction contrasts are the primary RQ2 interaction evidence. Frozen compound-SLO frequencies are secondary descriptive evidence. No CSD/RCA analysis is performed here.

## Baseline/Control SLO Crossings

7 baseline/control runs crossed at least one counted SLO.

## Queue Metrics

Runtime queue metrics are extracted from long-form `metric_name`/`metric_value` rows for `llamacpp:requests_processing` and `llamacpp:requests_deferred`. Missing long-form metrics remain missing and are not inferred from latency.

## Limitations

- Analysis uses run/repetition as the inferential unit (n=8 per condition).
- SciPy is used for Student-t confidence intervals and tests in this corrected analysis.
- Control transportability changed for OUTPUT_CONTROL; frozen SLO thresholds were not recalibrated post hoc.
- Server process CPU/RSS fields are null in the formal artifacts because server PID telemetry was not supplied in the formal configs; host CPU telemetry remains available.
