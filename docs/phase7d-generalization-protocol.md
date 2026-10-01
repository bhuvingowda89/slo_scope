# Phase 7D Unseen-Model Generalization Protocol

## RQ3A - Unseen Model Transfer

How well do diagnostic signatures and RCA methods learned on Qwen2.5-0.5B / llama.cpp / Metal transfer to an unseen larger model while keeping runtime, hardware, measurement stack, and mechanism semantics fixed?

## Target Model

- Model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`
- Alias: `sloscope-qwen2.5-1.5b`
- Runtime: llama.cpp 0.5.0, build 11146, commit 7fe450e19
- Server args: `-hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --alias sloscope-qwen2.5-1.5b --host 127.0.0.1 --port 8080 --metrics --parallel 2 --ctx-size 2048 --no-cache-prompt`

## Conditions

The formal campaign contains 11 RCA-transfer conditions: BASELINE, OUTPUT_CONTROL, INPUT_MEDIUM, OUTPUT_MEDIUM, LOAD_MEDIUM, DOWNSTREAM_MEDIUM, INPUT_LOAD, INPUT_DOWNSTREAM, OUTPUT_LOAD, OUTPUT_DOWNSTREAM, LOAD_DOWNSTREAM. `OUTPUT_LOAD_CONTROL` and `OUTPUT_DOWNSTREAM_CONTROL` are intentionally omitted because Phase 7D is not a factorial CSD campaign. `INPUT_OUTPUT` remains absent.

## Formal Campaign

- Campaign ID: `sloscope-generalization-qwen15b-v1`
- Conditions: 11
- Repetitions: 6
- Formal runs: 66
- Warm-up: 5 requests/run
- Measured: 40 requests/run
- Cooldown: 5 seconds
- Randomization seed: 7421

Six repetitions preserve independent repeated observations while allowing an exact paired/sign-flip test with minimum attainable two-sided p-value 0.03125, without duplicating the full 8-repetition source campaign cost.

## Zero-Shot Contract

Training uses only original Phase6-V2 Qwen-0.5B publication observations. No target-model labeled observation may fit imputation, scaling, logistic coefficients, MESR evidence distributions, thresholds, or method parameters. The Phase 7D.0 pilot is not training data.

## H4

H4: mechanism-oriented evidence transfers better than purely statistical signatures under configuration holdouts. The primary comparison is corrected D4-FULL vs M2/F2 on zero-shot target P1-COMPOUND exact set accuracy. M2/F2T must be reported prominently because it is the strongest in-domain baseline.

## SLO Policy

The Phase-5.2 SLO thresholds were calibrated for Qwen-0.5B and are not universal target-model objectives. Phase 7D primary RCA transfer does not depend on target SLO classification; any SLO-only baseline is non-transferable/descriptive unless explicitly labelled.

## Pilot Gate

Phase 7D.0 diagnostic pilot passed exact-source-setting viability; see `diagnostics/phase7d0-qwen15b-pilot/pilot-report.json`.
