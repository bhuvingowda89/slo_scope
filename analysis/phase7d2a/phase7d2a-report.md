# Phase 7D.2a Exact Source-Baseline Reproduction

{
  "analysis_spec_hash": "faca1733bab48f43d1fbe35e8eb01293763ce72db1ff6b38f6955d1a2251a57a",
  "d4_hash_verified": true,
  "d4_target_compound_exact": 0.6,
  "feature_shift_hash_verified": true,
  "h4": {
    "ci95": [
      NaN,
      NaN
    ],
    "differences": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "exact_signflip_p": 1.0,
    "h4_status": "NOT_SUPPORTED",
    "mean_difference": 0.0,
    "median_difference": 0.0,
    "secondary_D4_vs_M2F2T": {
      "ci95": [
        NaN,
        NaN
      ],
      "differences": [
        0.39999999999999997,
        0.39999999999999997,
        0.39999999999999997,
        0.39999999999999997,
        0.39999999999999997,
        0.39999999999999997
      ],
      "exact_signflip_p": 0.03125,
      "mean_difference": 0.39999999999999997
    }
  },
  "input_hash": "580ad72659387244129b27fefd5f7b4e97b4c7e19e697c05ceb91dae385cc75d",
  "material_changes": [
    "M2/F2 control false alarm",
    "M2/F2T target exact",
    "M2/F2T control false alarm"
  ],
  "optimizer_changed_cause_decisions": 36,
  "optimizer_changed_run_level_sets": 35,
  "result_status": "REVISED",
  "source_reproduction": {
    "f2_mismatch_examples": [],
    "f2_prediction_mismatches": 0,
    "f2t_mismatch_examples": [],
    "f2t_prediction_mismatches": 0,
    "new_f2_prediction_hash": "71a5289066bc2bbac4dd7e75ed8f7c743feda8a80a22a257476aef9f1a63bdc4",
    "new_f2t_prediction_hash": "d0c0cfadcc3d3b3ed4ae0e1853c86667871792a6ea185ce3771888340ab57b0f",
    "reference_f2_prediction_hash": "71a5289066bc2bbac4dd7e75ed8f7c743feda8a80a22a257476aef9f1a63bdc4",
    "reference_f2t_prediction_hash": "d0c0cfadcc3d3b3ed4ae0e1853c86667871792a6ea185ce3771888340ab57b0f",
    "row_count_f2": 104,
    "row_count_f2t": 104,
    "source_metrics": {
      "M2/F2": {
        "complete_cause_recall": 0.575,
        "control_false_alarm_rate": null,
        "exact_set_accuracy": 0.575,
        "jaccard": 0.7875,
        "macro_f1": 0.7419354838709677,
        "micro_f1": 0.881118881118881,
        "n": 40,
        "over_attribution": 0.0,
        "partial_cause_recall": 0.7875,
        "under_attribution": 0.425
      },
      "M2/F2T": {
        "complete_cause_recall": 0.85,
        "control_false_alarm_rate": null,
        "exact_set_accuracy": 0.825,
        "jaccard": 0.9166666666666667,
        "macro_f1": 0.9408045977011494,
        "micro_f1": 0.9548387096774195,
        "n": 40,
        "over_attribution": 0.025,
        "partial_cause_recall": 0.925,
        "under_attribution": 0.15
      }
    }
  },
  "table12": [
    {
      "absolute_drop": -0.025000000000000022,
      "method": "M2/F2",
      "retention": 1.0434782608695652,
      "source_compound_exact": 0.575,
      "target_complete_recall": 0.6,
      "target_compound_exact": 0.6,
      "target_control_false_alarm_rate": 0.9166666666666666,
      "target_jaccard": 0.75,
      "target_macro_f1": 0.6642857142857143,
      "target_over_attribution": 0.2,
      "target_under_attribution": 0.4
    },
    {
      "absolute_drop": 0.625,
      "method": "M2/F2T",
      "retention": 0.24242424242424246,
      "source_compound_exact": 0.825,
      "target_complete_recall": 1.0,
      "target_compound_exact": 0.2,
      "target_control_false_alarm_rate": 0.6666666666666666,
      "target_jaccard": 0.6555555555555556,
      "target_macro_f1": 0.7531328320802005,
      "target_over_attribution": 0.8,
      "target_under_attribution": 0.0
    },
    {
      "absolute_drop": 0.0,
      "method": "D4-FULL",
      "retention": 1.0,
      "source_compound_exact": 0.6,
      "target_complete_recall": 0.6,
      "target_compound_exact": 0.6,
      "target_control_false_alarm_rate": 0.5833333333333334,
      "target_jaccard": 0.7222222222222221,
      "target_macro_f1": 0.6506912442396313,
      "target_over_attribution": 0.4,
      "target_under_attribution": 0.4
    },
    {
      "absolute_drop": 0.20000000000000007,
      "method": "D4-FULL-F3",
      "retention": 0.7499999999999999,
      "source_compound_exact": 0.8,
      "target_complete_recall": 0.6,
      "target_compound_exact": 0.6,
      "target_control_false_alarm_rate": 0.5,
      "target_jaccard": 0.7222222222222221,
      "target_macro_f1": 0.6506912442396313,
      "target_over_attribution": 0.4,
      "target_under_attribution": 0.4
    }
  ]
}
