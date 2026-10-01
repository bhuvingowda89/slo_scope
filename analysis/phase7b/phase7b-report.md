# Phase 7B Compound Signature Divergence

phase7b_input_sha256: `4254abc08167845541700d5ca15edfd1163b241d32c520cd6831bb40b9313bfa`
phase7b_csd_spec_sha256: `61cade7df4bfd64073fa8c91dee31476a9c490b464b2d25b60503384f17dd89c`

## Summary

- INPUT_LOAD: CSD=51.58, 95% bootstrap [50.8, 52.46], exact p=0.007812, Holm p=0.03906, Holm-supported.
- INPUT_DOWNSTREAM: CSD=0.7104, 95% bootstrap [0.3912, 1.499], exact p=0.1484, Holm p=0.3047, not Holm-supported.
- OUTPUT_LOAD: CSD=30.81, 95% bootstrap [29.67, 31.88], exact p=0.007812, Holm p=0.03906, Holm-supported.
- OUTPUT_DOWNSTREAM: CSD=0.793, 95% bootstrap [0.6039, 1.389], exact p=0.1016, Holm p=0.3047, not Holm-supported.
- LOAD_DOWNSTREAM: CSD=0.9488, 95% bootstrap [0.7336, 1.937], exact p=0.2578, Holm p=0.3047, not Holm-supported.

## Interpretation

CSD quantifies multivariate departure from additive matched-factorial predictions. It is not RCA and does not use ground-truth mechanism labels as features.
Queue metrics are unavailable in the formal artifacts, so queue-depth divergence is not directly measured.
Output-control drift is handled by matched contemporaneous factorial cells; absolute SLO crossings are secondary context only.
