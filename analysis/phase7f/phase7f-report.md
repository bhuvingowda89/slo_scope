# Phase 7F Telemetry Cost Frontier

{
  "input_hash": "89776f070683ffbc68503bfcaba8dea40f6d6c7184d4d6dd0201a44ddd19ca5c",
  "cost_spec_hash": "254c69d2a425c36c2c90971a778943128522edabb99eefc4b6d2f532283d6f8e",
  "h5": {
    "criterion_A_source_quality": true,
    "criterion_B_target_quality": true,
    "criterion_C_measured_cost_reduction_all_12_pairs": true,
    "criterion_D_service_performance_guard": false,
    "final_H5_status": "NOT_SUPPORTED",
    "storage_reductions": [
      {
        "workload": "BASELINE",
        "repetition": 1,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.17082049845981517,
        "absolute_optional_bytes_per_request_saved": 106.75,
        "relative_total_latency_p95_change": -0.07098076680312404,
        "relative_ttft_p95_change": -0.12891835305465393,
        "relative_throughput_change": -0.0006333580027875207
      },
      {
        "workload": "BASELINE",
        "repetition": 2,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.1700476133317329,
        "absolute_optional_bytes_per_request_saved": 106.25,
        "relative_total_latency_p95_change": -0.12131473332020237,
        "relative_ttft_p95_change": -0.16717115035917718,
        "relative_throughput_change": 0.0004097414786812248
      },
      {
        "workload": "BASELINE",
        "repetition": 3,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.17609587132221194,
        "absolute_optional_bytes_per_request_saved": 110.57499999999993,
        "relative_total_latency_p95_change": -0.0620782402154374,
        "relative_ttft_p95_change": -0.04159680895805873,
        "relative_throughput_change": 0.0010446700486839156
      },
      {
        "workload": "BASELINE",
        "repetition": 4,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.17048050334629106,
        "absolute_optional_bytes_per_request_saved": 106.35000000000002,
        "relative_total_latency_p95_change": 0.09857674635089753,
        "relative_ttft_p95_change": 0.142025413571907,
        "relative_throughput_change": -0.0013598518883058963
      },
      {
        "workload": "BASELINE",
        "repetition": 5,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.17034548944337813,
        "absolute_optional_bytes_per_request_saved": 106.5,
        "relative_total_latency_p95_change": -0.028175006156895943,
        "relative_ttft_p95_change": -0.060687684736049685,
        "relative_throughput_change": 0.0015337982744716427
      },
      {
        "workload": "BASELINE",
        "repetition": 6,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.17802398538638697,
        "absolute_optional_bytes_per_request_saved": 112.07499999999993,
        "relative_total_latency_p95_change": 0.03270019947308356,
        "relative_ttft_p95_change": 0.04394383721845174,
        "relative_throughput_change": -0.0009102387121430189
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 1,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.14000863185153223,
        "absolute_optional_bytes_per_request_saved": 81.10000000000002,
        "relative_total_latency_p95_change": -0.11858997156318463,
        "relative_ttft_p95_change": -0.1494232052732134,
        "relative_throughput_change": 0.0332833388218301
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 2,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.14270377050588334,
        "absolute_optional_bytes_per_request_saved": 83.07499999999999,
        "relative_total_latency_p95_change": -0.06941955985228077,
        "relative_ttft_p95_change": -0.08297602512885427,
        "relative_throughput_change": 0.01886215020406934
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 3,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.14148381256196918,
        "absolute_optional_bytes_per_request_saved": 82.04999999999995,
        "relative_total_latency_p95_change": 0.035663762422604384,
        "relative_ttft_p95_change": 0.06346580485171671,
        "relative_throughput_change": -0.008394071170395145
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 4,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.13996634594641233,
        "absolute_optional_bytes_per_request_saved": 81.09999999999997,
        "relative_total_latency_p95_change": 0.06404016765855003,
        "relative_ttft_p95_change": 0.04353812349517816,
        "relative_throughput_change": -0.013982750436153157
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 5,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.13996126533247255,
        "absolute_optional_bytes_per_request_saved": 81.30000000000001,
        "relative_total_latency_p95_change": 0.0343137038743333,
        "relative_ttft_p95_change": 0.043155982122841774,
        "relative_throughput_change": -0.008288122741858861
      },
      {
        "workload": "LOAD_MEDIUM",
        "repetition": 6,
        "comparison": "T1_NO_HOST_vs_T0_FULL",
        "storage_reduction": 0.1387028475132871,
        "absolute_optional_bytes_per_request_saved": 80.25000000000006,
        "relative_total_latency_p95_change": 0.027262400396395936,
        "relative_ttft_p95_change": 0.035245672924521854,
        "relative_throughput_change": -0.005754027240517545
      }
    ],
    "service_performance_guard": [
      {
        "workload": "BASELINE",
        "mean_relative_total_latency_p95_change": -0.02521196677861311,
        "ci95_low": -0.10834408391735167,
        "ci95_high": 0.05792015036012546,
        "passes_plus_5pct_guard": false
      },
      {
        "workload": "LOAD_MEDIUM",
        "mean_relative_total_latency_p95_change": -0.004454916177263624,
        "ci95_low": -0.08021413184848826,
        "ci95_high": 0.07130429949396101,
        "passes_plus_5pct_guard": false
      }
    ]
  },
  "frontier": [
    {
      "telemetry_config": "T0_FULL",
      "source_M2F2_compound_exact": 0.575,
      "target_M2F2_compound_exact": 0.6,
      "source_control_false_alarm": 0.6875,
      "target_control_false_alarm": 0.9166666666666666,
      "optional_telemetry_bytes_per_request": 603.0375,
      "total_artifact_bytes_per_request": 1553.9708333333335,
      "storage_reduction_vs_FULL": 0.0,
      "baseline_latency_relative_change": 0.0,
      "load_latency_relative_change": 0.0,
      "H5_quality_screen_pass": true,
      "pareto_dominated": true
    },
    {
      "telemetry_config": "T1_NO_HOST",
      "source_M2F2_compound_exact": 0.6,
      "target_M2F2_compound_exact": 0.6,
      "source_control_false_alarm": 0.625,
      "target_control_false_alarm": 0.5833333333333334,
      "optional_telemetry_bytes_per_request": 508.25624999999997,
      "total_artifact_bytes_per_request": 1463.1812499999999,
      "storage_reduction_vs_FULL": 0.1571730613767801,
      "baseline_latency_relative_change": -0.02848387107795225,
      "load_latency_relative_change": -0.006618713542177135,
      "H5_quality_screen_pass": true,
      "pareto_dominated": true
    },
    {
      "telemetry_config": "T2_NO_RUNTIME",
      "source_M2F2_compound_exact": 0.575,
      "target_M2F2_compound_exact": 0.6,
      "source_control_false_alarm": 0.6875,
      "target_control_false_alarm": 0.9166666666666666,
      "optional_telemetry_bytes_per_request": 545.1625,
      "total_artifact_bytes_per_request": 1503.9604166666668,
      "storage_reduction_vs_FULL": 0.09597247269033848,
      "baseline_latency_relative_change": -0.033562426809393475,
      "load_latency_relative_change": -0.015379211528357528,
      "H5_quality_screen_pass": true,
      "pareto_dominated": true
    },
    {
      "telemetry_config": "T3_LIGHT",
      "source_M2F2_compound_exact": 0.6,
      "target_M2F2_compound_exact": 0.6,
      "source_control_false_alarm": 0.625,
      "target_control_false_alarm": 0.5833333333333334,
      "optional_telemetry_bytes_per_request": 448.3229166666667,
      "total_artifact_bytes_per_request": 1400.5916666666665,
      "storage_reduction_vs_FULL": 0.2565588099137007,
      "baseline_latency_relative_change": -0.011434646964370576,
      "load_latency_relative_change": 0.004800842398702487,
      "H5_quality_screen_pass": true,
      "pareto_dominated": false
    }
  ],
  "target_storage_projection": {
    "campaign": "target",
    "runs": 66,
    "system_bytes_per_run_mean": 12196.651515151516,
    "system_bytes_per_request_mean": 304.9162878787879,
    "campaign_total_system_bytes": 804979,
    "system_pct_optional_telemetry_storage": 0.4089573652094704,
    "trace_bytes_per_run_mean": 11602.015151515152,
    "trace_bytes_per_request_mean": 290.0503787878788
  },
  "trace_storage_context": [
    {
      "campaign": "source",
      "runs": 104,
      "system_bytes_per_run_mean": 11874.846153846154,
      "system_bytes_per_request_mean": 296.87115384615385,
      "campaign_total_system_bytes": 1234984,
      "system_pct_optional_telemetry_storage": 0.40140920834902966,
      "trace_bytes_per_run_mean": 11568.961538461539,
      "trace_bytes_per_request_mean": 289.2240384615385
    },
    {
      "campaign": "target",
      "runs": 66,
      "system_bytes_per_run_mean": 12196.651515151516,
      "system_bytes_per_request_mean": 304.9162878787879,
      "campaign_total_system_bytes": 804979,
      "system_pct_optional_telemetry_storage": 0.4089573652094704,
      "trace_bytes_per_run_mean": 11602.015151515152,
      "trace_bytes_per_request_mean": 290.0503787878788
    }
  ]
}
