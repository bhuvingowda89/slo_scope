# Claim Audit

| Claim | Classification | Wording guard |
|---|---|---|
| Some compound signatures are non-additive. | DIRECTLY_SUPPORTED | Say specific pairs, not all compounds. |
| Compound SLO violations imply non-additivity. | NEGATIVE_RESULT | Manuscript explicitly rejects this. |
| Single-fault-trained RCA loses complete-cause recall. | DIRECTLY_SUPPORTED | Scope to this runtime/model/workload. |
| MESR is the best RCA method. | PROHIBITED | Not supported; M2/F2T is strongest in-domain. |
| Temporal features improve in-domain diagnosis. | DIRECTLY_SUPPORTED | Pair with model-sensitivity caveat. |
| Mechanism-oriented evidence transfers better than statistical signatures. | NEGATIVE_RESULT | H4 not supported. |
| Reduced telemetry preserves quality at lower cost and meets live guard. | NEGATIVE_RESULT | H5 not supported because guard fails. |
| Feature shifts caused F2T transfer failure. | DESCRIPTIVE_ONLY | Use "consistent with", not causal language. |
