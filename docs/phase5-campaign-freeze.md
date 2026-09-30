# Phase 5 Campaign Freeze

This document freezes the publication campaign specification. It is a plan only: no
formal campaign runs, compound campaign, CSD, RCA, memory/I/O, or cross-runtime
experiments are executed by Phase 5.

## Freeze Identity

- campaign_freeze_sha256: `82b61df32669f6731e31b1379d07d3d50291c1c000247251df2a4e14c60f0adb`
- campaign_manifest_sha256: `082ec774ea3ab514846a844f60f9c4adae2b2394dad5e6b9060fa70c6f38ce5c`
- git_revision: `0b73db598558b0df41b1b28654f5f568c65e7b35`
- git_dirty_at_freeze_generation: `True`
- source_tree_sha256_at_generation: `3a3eef468db5c1c4dcfbeeb67f55081e25d78d9bb621bb658c318cceb1dfc6fc`

Publication execution requires `git_dirty=false`; the Phase 5 repository may still
contain uncommitted freeze artifacts until the user commits them.

## Runtime Environment

- llama.cpp `0.5.0`, build `11146`, commit `7fe450e19`
- Darwin arm64 / Metal
- model `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`
- alias `sloscope-qwen2.5-0.5b`
- required flags: `-hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --alias sloscope-qwen2.5-0.5b --host 127.0.0.1 --port 8080 --metrics --parallel 2 --ctx-size 2048 --no-cache-prompt`
- prompt cache: disabled by `--no-cache-prompt`

Changing any runtime flag, model, quantization, build, or commit reopens the freeze.

## Frozen SLOs

SLO thresholds are frozen in `campaigns/phase5/slo-calibration.json` using
one-sided future-run prediction limits. The independent statistical unit is the
run/repetition. Formal Phase 6 outcomes must not be used to set thresholds.

- TTFT_SLO: p95 TTFT over successful requests; violation when above the calibrated
  baseline threshold.
- TOTAL_LATENCY_SLO: p95 total request latency; violation when above the calibrated
  baseline threshold.
- DECODE_DURATION_SLO: p95 post-first-token duration; violation when above the
  output matched-control threshold.
- observed seconds per output token: retained as a diagnostic decode metric, not a
  counted independent SLO, because it is highly redundant with decode duration here.
- THROUGHPUT_DIAGNOSTIC: run-level observed successful throughput is retained for
  capacity and RCA analysis, but it is demand-dependent and does not contribute to
  single/compound SLO violation counts.

Single SLO violation means exactly one frozen SLO is violated in a measured run.
Compound SLO violation means two or more frozen SLOs are violated in the same
measured run. Request rows are observations within a run, not independent
experimental repetitions.

The counted SLO set is exactly: `TTFT_SLO`, `TOTAL_LATENCY_SLO`, and
`DECODE_DURATION_SLO`.

## Frozen Mechanisms

- M1 input/prefill pressure: medium input, observed median prompt tokens about 267.
- M2 output/decode pressure: synthetic continuation with 32 requested output tokens.
- M3 queue/load pressure: 12 rps offered load; queueing is intentional.
- M4 downstream latency: synthetic dependency configured to 100 ms.
- CPU contention: DEFERRED for cross-configuration analysis.

## Single-Mechanism Matrix

Formal single/control conditions: `BASELINE, OUTPUT_CONTROL, OUTPUT_LOAD_CONTROL, OUTPUT_DOWNSTREAM_CONTROL, INPUT_MEDIUM, OUTPUT_MEDIUM, LOAD_MEDIUM, DOWNSTREAM_MEDIUM`.

## Compound Matrix

Primary pairwise compounds: `INPUT_LOAD, INPUT_DOWNSTREAM, OUTPUT_LOAD, OUTPUT_DOWNSTREAM, LOAD_DOWNSTREAM`.

Three-way and four-way compounds are not part of the primary Phase 5 campaign freeze.
They require an explicit reopen because interpretability is prioritized over
combinatorial coverage.

`INPUT_OUTPUT` is deferred because the accepted input and output references use
different prompt families. It may only be added after a matched continuation
small/medium input calibration demonstrates controllable output and no queue confound.

## Controls

Input, load, and downstream compare against `BASELINE`. Output compares against
`OUTPUT_CONTROL`. Output-containing compounds use the explicit four-cell mappings
in `factorial-design.json`, including `OUTPUT_LOAD_CONTROL` and
`OUTPUT_DOWNSTREAM_CONTROL`.

## Repetition, Order, Warm-Up, Run Length

- repetitions per condition: `8`
- deterministic order seed: `5150`
- order rule: block by repetition, randomize all formal conditions within each block
- measured requests per run: `40`
- unscored warm-up requests before each measured run: `5`
- fixed cooldown after each run: `5` seconds

## Telemetry Schema

Primary fields: TTFT, total latency, post-first-token duration, observed seconds per
output token, prompt tokens, output tokens, dependency duration, scheduler slip,
requests processing/deferred, request outcomes, observed throughput, process/system
metrics, and internal gateway/dependency/llama spans.

No primary-analysis telemetry field may be added after the formal campaign starts
without reopening the freeze.

## Confounds, Validity, and Reruns

For input and output, `requests_deferred_max > 0` is queue-confounded. For load,
queueing is the intended mechanism. Downstream queueing is a secondary effect unless
it trips a mandatory invalidation rule.

Invalid runs may be rerun only under a new attempt ID; valid surprising runs are data
and must not be silently rerun.

## Statistical Plan

Use 95% Student-t confidence intervals across run repetitions. Use paired comparisons
where pairing by repetition is valid. Apply Holm correction within predefined
hypothesis families. Report absolute deltas, ratios, and paired standardized effects
where appropriate, but interpret practical SLO boundary crossings first.

## Later CSD/RCA Contract

Phase 5 preserves active mechanisms, mechanism levels, compound degree, condition ID,
direct mechanism evidence, SLO aggregates, and telemetry features for later Compound
SLO Decomposition and RCA evaluation. Phase 5 does not execute those analyses.

## Campaign Size and Storage

- formal conditions: `13`
- formal runs: `104`
- estimated measured requests: `4160`
- estimated warm-up requests: `520`
- estimated storage guard: `32212254720` bytes minimum free
- estimated campaign storage: `21474836480` bytes

## Stop/Go Criteria

Go only when the source tree is clean, disk-space guard passes, runtime/model
provenance matches this freeze, service readiness passes, and the manifest/config
hashes match. Stop on dirty source, runtime mismatch, invalid manifest, insufficient
space, service contamination, or mandatory telemetry/trace failure.
