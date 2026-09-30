# Phase 6B.1 Missing Gateway Timing Forensics

Affected request: `phase6-r03-load-downstream-a01-req-000023`

Classification: B. RUNNER SNAPSHOT RACE
Recommendation: PATH 2 — SOURCE FIX REQUIRED
a02 safe under current frozen source: False

## Missing Fields
- `gateway_receive_time`
- `dependency_start_time`
- `dependency_end_time`
- `llama_dispatch_time`
- `llama_first_token_time`
- `gateway_completion_time`
- `dependency_duration`

## Basis
- Gateway publishes request_timings and trace_rows together only in _handle_completion finally block after streaming/upstream cleanup.
- Client runtime breaks out of the SSE read loop on [DONE], records completion, then immediately performs a single /sloscope/requests/{request_id} lookup.
- The single lookup returns empty timing if the gateway finally block has not yet inserted request_timings; there is no retry/eventual consistency wait.
- Trace rows for the same request were later collected after workload completion, proving the gateway ultimately finalized spans for the exact request ID.
- Identifier bytes match across workload, requests.parquet, and trace attributes; logs show no gateway/service exception.
