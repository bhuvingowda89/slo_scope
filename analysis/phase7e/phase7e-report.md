# Phase 7E Degraded Telemetry Robustness

{
  "input_hash": "d71062e8c648becfabf1063948b7b36f37040e6387349bd6e957d2cd500ee272",
  "robustness_spec_hash": "979f56823140bd518d5b7f95663340767be5fb154f2e8837987c626d9e4f42dd",
  "clean_reference_audit": [
    {
      "domain": "source",
      "method": "M2/F2",
      "expected_compound_exact": 0.575,
      "actual_compound_exact": 0.575,
      "matches": true
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "expected_compound_exact": 0.825,
      "actual_compound_exact": 0.825,
      "matches": true
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "expected_compound_exact": 0.6,
      "actual_compound_exact": 0.6,
      "matches": true
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "expected_compound_exact": 0.8,
      "actual_compound_exact": 0.8,
      "matches": true
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "expected_compound_exact": 0.6,
      "actual_compound_exact": 0.6,
      "matches": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "expected_compound_exact": 0.2,
      "actual_compound_exact": 0.2,
      "matches": true
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "expected_compound_exact": 0.6,
      "actual_compound_exact": 0.6,
      "matches": true
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "expected_compound_exact": 0.6,
      "actual_compound_exact": 0.6,
      "matches": true
    }
  ],
  "h5_quality_component_status": "SUPPORTED_SCREENING",
  "h5_passing_candidates": [
    {
      "domain": "source",
      "method": "M2/F2",
      "candidate": "C1_HOST_CPU",
      "retention": 1.0434782608695652,
      "control_false_alarm_change": -0.0625,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "candidate": "C1_HOST_CPU",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "candidate": "C2_DEPENDENCY",
      "retention": 0.9696969696969698,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "candidate": "C3_TRACE_SPAN",
      "retention": 0.9583333333333333,
      "control_false_alarm_change": -0.0625,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "candidate": "C4_TEMPORAL_EVOLUTION",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "candidate": "C5_THROUGHPUT_SCHEDULER",
      "retention": 1.0,
      "control_false_alarm_change": -0.0625,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "candidate": "C5_THROUGHPUT_SCHEDULER",
      "retention": 0.9696969696969698,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "candidate": "C7_MEDIAN_LATENCY",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "candidate": "C1_HOST_CPU",
      "retention": 1.0,
      "control_false_alarm_change": -0.33333333333333326,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "candidate": "C1_HOST_CPU",
      "retention": 1.0,
      "control_false_alarm_change": -0.16666666666666663,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "candidate": "C1_HOST_CPU",
      "retention": 1.0,
      "control_false_alarm_change": -0.08333333333333337,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "candidate": "C3_TRACE_SPAN",
      "retention": 1.0,
      "control_false_alarm_change": -0.08333333333333326,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "candidate": "C4_TEMPORAL_EVOLUTION",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "candidate": "C5_THROUGHPUT_SCHEDULER",
      "retention": 1.0,
      "control_false_alarm_change": -0.33333333333333326,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "candidate": "C5_THROUGHPUT_SCHEDULER",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "candidate": "C5_THROUGHPUT_SCHEDULER",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "candidate": "C6_LATENCY_P95",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "candidate": "C6_LATENCY_P95",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "candidate": "C7_MEDIAN_LATENCY",
      "retention": 2.0,
      "control_false_alarm_change": 0.08333333333333337,
      "passes_H5_quality_screen": true
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "candidate": "C7_MEDIAN_LATENCY",
      "retention": 1.0,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": true
    }
  ],
  "auc_rows": [
    {
      "domain": "source",
      "method": "M2/F2",
      "degradation_family": "missingness",
      "robustness_auc": 0.45275
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "degradation_family": "missingness",
      "robustness_auc": 0.658125
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "degradation_family": "missingness",
      "robustness_auc": 0.5797499999999999
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "degradation_family": "missingness",
      "robustness_auc": 0.7444999999999999
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "degradation_family": "missingness",
      "robustness_auc": 0.3598333333333334
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "degradation_family": "missingness",
      "robustness_auc": 0.23716666666666664
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "degradation_family": "missingness",
      "robustness_auc": 0.5721666666666665
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "degradation_family": "missingness",
      "robustness_auc": 0.5823333333333333
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.575
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.7116666666666666
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.5995833333333332
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.775
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.5999999999999999
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.29277777777777775
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.5861111111111109
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "degradation_family": "temporal_sampling",
      "robustness_auc": 0.5872222222222221
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "degradation_family": "noise",
      "robustness_auc": 0.5817499999999999
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "degradation_family": "noise",
      "robustness_auc": 0.898
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "degradation_family": "noise",
      "robustness_auc": 0.56175
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "degradation_family": "noise",
      "robustness_auc": 0.647125
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "degradation_family": "noise",
      "robustness_auc": 0.5924999999999999
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "degradation_family": "noise",
      "robustness_auc": 0.2
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "degradation_family": "noise",
      "robustness_auc": 0.5208333333333333
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "degradation_family": "noise",
      "robustness_auc": 0.5473333333333332
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "degradation_family": "delay",
      "robustness_auc": 0.575
    },
    {
      "domain": "source",
      "method": "M2/F2T",
      "degradation_family": "delay",
      "robustness_auc": 0.71
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "degradation_family": "delay",
      "robustness_auc": 0.6
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "degradation_family": "delay",
      "robustness_auc": 0.8
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "degradation_family": "delay",
      "robustness_auc": 0.6
    },
    {
      "domain": "target",
      "method": "M2/F2T",
      "degradation_family": "delay",
      "robustness_auc": 0.3933333333333333
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "degradation_family": "delay",
      "robustness_auc": 0.6
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "degradation_family": "delay",
      "robustness_auc": 0.6
    }
  ],
  "top_channel_losses": [
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.225,
      "retention": 0.28125,
      "absolute_loss": 0.5750000000000001,
      "complete_cause_recall": 0.225,
      "jaccard": 0.5458333333333333,
      "macro_f1": 0.6005797101449276,
      "over_attribution": 0.4,
      "under_attribution": 0.775,
      "control_false_alarm_rate": 0.3125,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "source",
      "method": "D4-FULL-F3",
      "channel_removed": "C8_MECHANISM_ANCHOR",
      "compound_exact": 0.225,
      "retention": 0.28125,
      "absolute_loss": 0.5750000000000001,
      "complete_cause_recall": 0.225,
      "jaccard": 0.5125,
      "macro_f1": 0.5676190476190476,
      "over_attribution": 0.6,
      "under_attribution": 0.775,
      "control_false_alarm_rate": 0.4375,
      "control_false_alarm_change": 0.125,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "channel_removed": "C8_MECHANISM_ANCHOR",
      "compound_exact": 0.175,
      "retention": 0.2916666666666667,
      "absolute_loss": 0.425,
      "complete_cause_recall": 0.175,
      "jaccard": 0.3625,
      "macro_f1": 0.35270676691729325,
      "over_attribution": 0.525,
      "under_attribution": 0.825,
      "control_false_alarm_rate": 0.0,
      "control_false_alarm_change": -0.3125,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "source",
      "method": "M2/F2",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.175,
      "retention": 0.30434782608695654,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.175,
      "jaccard": 0.5875,
      "macro_f1": 0.6586021505376344,
      "over_attribution": 0.0,
      "under_attribution": 0.825,
      "control_false_alarm_rate": 0.6875,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "source",
      "method": "D4-FULL",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.2,
      "jaccard": 0.45,
      "macro_f1": 0.42920716112531976,
      "over_attribution": 0.425,
      "under_attribution": 0.8,
      "control_false_alarm_rate": 0.3125,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.2,
      "jaccard": 0.5,
      "macro_f1": 0.41428571428571426,
      "over_attribution": 0.2,
      "under_attribution": 0.8,
      "control_false_alarm_rate": 0.9166666666666666,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "target",
      "method": "D4-FULL",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.2,
      "jaccard": 0.43333333333333335,
      "macro_f1": 0.35714285714285715,
      "over_attribution": 0.6,
      "under_attribution": 0.8,
      "control_false_alarm_rate": 0.5833333333333334,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "target",
      "method": "D4-FULL-F3",
      "channel_removed": "C2_DEPENDENCY",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.2,
      "jaccard": 0.43333333333333335,
      "macro_f1": 0.35714285714285715,
      "over_attribution": 0.6,
      "under_attribution": 0.8,
      "control_false_alarm_rate": 0.5,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "channel_removed": "C3_TRACE_SPAN",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 0.2,
      "jaccard": 0.5333333333333333,
      "macro_f1": 0.5476190476190476,
      "over_attribution": 0.4,
      "under_attribution": 0.8,
      "control_false_alarm_rate": 0.8333333333333334,
      "control_false_alarm_change": -0.08333333333333326,
      "passes_H5_quality_screen": false
    },
    {
      "domain": "target",
      "method": "M2/F2",
      "channel_removed": "C6_LATENCY_P95",
      "compound_exact": 0.2,
      "retention": 0.33333333333333337,
      "absolute_loss": 0.39999999999999997,
      "complete_cause_recall": 1.0,
      "jaccard": 0.6666666666666666,
      "macro_f1": 0.7738095238095238,
      "over_attribution": 0.8,
      "under_attribution": 0.0,
      "control_false_alarm_rate": 0.9166666666666666,
      "control_false_alarm_change": 0.0,
      "passes_H5_quality_screen": false
    }
  ]
}
