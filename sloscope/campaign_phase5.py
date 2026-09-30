from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sloscope.artifacts.writer import git_revision
from sloscope.artifacts.writer import read_table, sha256_file
from sloscope.config import (
    ExperimentConfig,
    MechanismConfig,
    RuntimeConfig,
    SafetyConfig,
    TelemetryConfig,
    WorkloadConfig,
)
from sloscope.provenance import git_dirty, source_tree_sha256
from sloscope.runner import ExperimentRunner


PHASE5_ID = "phase5-campaign-freeze"
SCHEMA_VERSION = "sloscope.config.v1"
ORDER_SEED = 5150
REPETITIONS = 8
CALIBRATION_REPETITIONS = 8
REQUEST_COUNT = 40
WARMUP_REQUESTS = 5
COOLDOWN_SECONDS = 5
MIN_FREE_SPACE_BYTES = 30 * 1024 * 1024 * 1024
ESTIMATED_STORAGE_BYTES = 20 * 1024 * 1024 * 1024
RUNTIME_VERSION = "0.5.0"
RUNTIME_BUILD = "11146"
RUNTIME_COMMIT = "7fe450e19"
MODEL_REF = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
MODEL_ID = "sloscope-qwen2.5-0.5b"
LLAMA_ARGS = [
    "-hf",
    MODEL_REF,
    "--alias",
    MODEL_ID,
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
    "--metrics",
    "--parallel",
    "2",
    "--ctx-size",
    "2048",
    "--no-cache-prompt",
]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    mechanism_family: str
    active_mechanisms: tuple[str, ...]
    mechanism_levels: dict[str, str]
    compound_degree: int
    expected_control_id: str | None
    prompt_profile: str
    target_output_tokens: int
    inter_arrival_seconds: float
    max_outstanding_requests: int
    dependency_delay_ms: int = 0
    notes: str = ""

    @property
    def offered_rps(self) -> float:
        return 1.0 / self.inter_arrival_seconds


def condition_slug(condition_id: str) -> str:
    return condition_id.lower().replace("_", "-")


def run_id_for(condition_id: str, repetition: int, attempt: int = 1) -> str:
    return f"phase6-r{repetition:02d}-{condition_slug(condition_id)}-a{attempt:02d}"


def one_sided_t_critical(alpha: float, df: int) -> float:
    if alpha != 0.05:
        raise ValueError("Phase 5.1 freezes alpha=0.05")
    table = {
        1: 6.313752,
        2: 2.919986,
        3: 2.353363,
        4: 2.131847,
        5: 2.015048,
        6: 1.943180,
        7: 1.894579,
        8: 1.859548,
        9: 1.833113,
        10: 1.812461,
        11: 1.795885,
        12: 1.782288,
        13: 1.770933,
        14: 1.761310,
        15: 1.753050,
        16: 1.745884,
        17: 1.739607,
        18: 1.734064,
        19: 1.729133,
        20: 1.724718,
        24: 1.710882,
        30: 1.697261,
        40: 1.683851,
        60: 1.670649,
        120: 1.657654,
    }
    if df in table:
        return table[df]
    if df < 1:
        raise ValueError("df must be positive")
    return 1.644854


def prediction_limit(values: list[float], *, direction: str, alpha: float = 0.05) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    n = len(values)
    tcrit = one_sided_t_critical(alpha, n - 1)
    margin = tcrit * sd * math.sqrt(1.0 + 1.0 / n)
    if direction == "upper":
        threshold = mean + margin
    elif direction == "lower":
        threshold = mean - margin
    else:
        raise ValueError("direction must be upper or lower")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "alpha": alpha,
        "t_critical": tcrit,
        "prediction_margin": margin,
        "threshold": threshold,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def run_aggregates(run_dir: Path) -> dict[str, Any]:
    _, requests = read_table(run_dir / "requests.parquet")
    success = [r for r in requests if r.get("status") == "success"]
    ttft = [r["first_token_time"] - r["actual_arrival"] for r in success if r.get("first_token_time") is not None]
    total = [r["completion_time"] - r["actual_arrival"] for r in success if r.get("completion_time") is not None]
    decode = [r["completion_time"] - r["first_token_time"] for r in success if r.get("completion_time") is not None and r.get("first_token_time") is not None]
    slips = [r["scheduler_slip"] for r in requests if r.get("scheduler_slip") is not None]
    arrivals = [r["actual_arrival"] for r in requests if r.get("actual_arrival") is not None]
    completions = [r["completion_time"] for r in success if r.get("completion_time") is not None]
    throughput = None
    if arrivals and completions and max(completions) > min(arrivals):
        throughput = len(success) / (max(completions) - min(arrivals))
    return {
        "planned_requests": len(requests),
        "successful_requests": len(success),
        "failed_requests": len([r for r in requests if r.get("status") == "failed"]),
        "admission_failed": len([r for r in requests if r.get("status") == "admission_failed"]),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_p95": percentile(total, 0.95),
        "post_first_token_duration_p95": percentile(decode, 0.95),
        "scheduler_slip_p95": percentile(slips, 0.95),
        "successful_requests_per_second": throughput,
        "completion_fraction": len(success) / len(requests) if requests else None,
    }


def ci_relative_half_width(values: list[float], n: int, *, alpha: float = 0.05) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    sd = statistics.stdev(values)
    tcrit = one_sided_t_critical(alpha / 2.0 if False else 0.05, n - 1)
    return abs(tcrit * sd / math.sqrt(n) / mean)


def mechanism_definitions() -> dict[str, Any]:
    return {
        "M1_INPUT_PREFILL": {
            "mechanism_id": "input_medium",
            "name": "Input/prefill pressure",
            "status": "PRIMARY",
            "control_level": {"prompt_profile": "synthetic-input-small", "observed_prompt_tokens": "about 75"},
            "medium_level": {"prompt_profile": "synthetic-input-medium", "observed_prompt_tokens": "about 267"},
            "optional_strong_level": {"prompt_profile": "synthetic-input-large", "status": "not primary; Phase 4B queue-confounded"},
            "direct_evidence": ["server_prompt_tokens"],
            "primary_slo_effect": "TTFT",
            "secondary_effects": ["total_latency"],
            "known_confounds": ["queue buildup if requests_deferred_max exceeds zero in non-load conditions"],
        },
        "M2_OUTPUT_DECODE": {
            "mechanism_id": "output_32",
            "name": "Output/decode pressure",
            "status": "PRIMARY",
            "control_level": {"target_output_tokens": 16, "prompt_profile": "synthetic-continuation"},
            "medium_level": {"target_output_tokens": 32, "prompt_profile": "synthetic-continuation"},
            "optional_strong_level": {"target_output_tokens": 64, "status": "optional strong; not medium reference"},
            "direct_evidence": ["server_output_tokens", "post_first_token_duration"],
            "primary_slo_effect": "post_first_token_duration",
            "secondary_effects": ["total_latency"],
            "known_confounds": ["queue buildup; output-48 was queue-confounded in one Phase 4C repetition"],
        },
        "M3_LOAD_QUEUE": {
            "mechanism_id": "load_12rps",
            "name": "Queue/load pressure",
            "status": "PRIMARY",
            "control_level": {"offered_rps": 4},
            "medium_level": {"offered_rps": 12},
            "optional_strong_level": {"offered_rps": 16, "status": "not primary; Phase 4B strong queueing"},
            "direct_evidence": ["llamacpp:requests_deferred", "requests_deferred_max"],
            "primary_slo_effect": "TTFT",
            "secondary_effects": ["total_latency", "observed_throughput"],
            "known_confounds": ["client admission failures mean client-side throttling, not pure server queueing"],
        },
        "M4_DOWNSTREAM_LATENCY": {
            "mechanism_id": "downstream_100ms",
            "name": "Downstream dependency latency",
            "status": "PRIMARY",
            "control_level": {"dependency_delay_ms": 0},
            "medium_level": {"dependency_delay_ms": 100},
            "optional_strong_level": {"dependency_delay_ms": 200, "status": "available from Phase 4A, not primary"},
            "direct_evidence": ["configured_delay_ms", "dependency_duration"],
            "primary_slo_effect": "total_latency",
            "secondary_effects": ["TTFT when dependency precedes llama dispatch"],
            "known_confounds": ["unexpected queue buildup must be flagged as secondary effect"],
        },
        "CPU_CONTENTION": {
            "mechanism_id": "cpu_contention",
            "name": "CPU contention",
            "status": "DEFERRED",
            "reason": "Verified injectable, weak SLO effect under current llama.cpp + Metal configuration.",
        },
    }


def slo_definitions() -> dict[str, Any]:
    calibration = (
        "Calibrate in a dedicated pre-campaign SLO calibration step before Phase 6 "
        "formal outcomes exist. Use one-sided future-run prediction limits with "
        "alpha=0.05. Formal Phase 6 outcome data must not be used to set thresholds."
    )
    return {
        "TTFT_SLO": {
            "metric": "time_to_first_token",
            "unit": "seconds",
            "aggregation_level": "run-level p95 over successful requests",
            "threshold": {"procedure": calibration, "source_conditions": ["BASELINE"]},
            "baseline_source": "publication campaign baseline repetitions; Phase 4C confirms medium input and load perturb TTFT",
            "justification": "TTFT is the primary user-visible prefill/queue symptom.",
            "violation_definition": "run-level p95 TTFT > calibrated threshold",
        },
        "TOTAL_LATENCY_SLO": {
            "metric": "total_request_latency",
            "unit": "seconds",
            "aggregation_level": "run-level p95 over successful requests",
            "threshold": {"procedure": calibration, "source_conditions": ["BASELINE"]},
            "baseline_source": "publication campaign baseline repetitions",
            "justification": "End-to-end request completion latency captures downstream and combined effects.",
            "violation_definition": "run-level p95 total latency > calibrated threshold",
        },
        "DECODE_DURATION_SLO": {
            "metric": "post_first_token_duration",
            "unit": "seconds",
            "aggregation_level": "run-level p95 over successful requests",
            "threshold": {"procedure": calibration, "source_conditions": ["OUTPUT_CONTROL"]},
            "baseline_source": "publication campaign output matched-control repetitions",
            "justification": "Decode service time must be evaluated against the same continuation prompt family.",
            "violation_definition": "run-level p95 post-first-token duration > calibrated threshold",
        },
        "SECONDS_PER_OUTPUT_TOKEN_DIAGNOSTIC": {
            "metric": "observed_seconds_per_output_token",
            "unit": "seconds/token",
            "aggregation_level": "run-level median over successful requests with output tokens",
            "threshold": {"procedure": "diagnostic reference only; not counted as an independent SLO violation"},
            "baseline_source": "publication campaign output matched-control repetitions",
            "justification": "Token-normalized decode descriptor is highly redundant with post-first-token duration for this campaign.",
            "violation_definition": "not counted toward single/compound SLO violation status",
        },
        "THROUGHPUT_DIAGNOSTIC": {
            "metric": "observed_throughput",
            "unit": "successful requests/second",
            "aggregation_level": "run-level successful completions divided by measured arrival-to-completion window",
            "threshold": {"procedure": "diagnostic/capacity metric only; not counted as a universal SLO"},
            "baseline_source": "publication campaign baseline repetitions",
            "justification": "Raw throughput is demand-dependent and cannot be a universal counted SLO when offered load varies.",
            "violation_definition": "not counted toward single/compound SLO violation status",
        },
        "compound_violation_definition": {
            "single_slo_violation": "Exactly one frozen SLO is violated by a run-level aggregate.",
            "compound_slo_violation": "Two or more frozen SLOs are violated by run-level aggregates in the same measured run.",
            "counted_slos": ["TTFT_SLO", "TOTAL_LATENCY_SLO", "DECODE_DURATION_SLO"],
            "statistical_unit": "run/repetition, not individual requests",
            "window": "whole measured run after unscored warm-up; request-level rows support diagnosis but do not define independent repetitions",
        },
    }


def analysis_plan() -> dict[str, Any]:
    return {
        "research_questions": {
            "RQ1": "Can isolated mechanisms produce distinct, reproducible SLO-degradation signatures on commodity LLM inference systems?",
            "RQ2": "How do compound mechanisms interact to produce single versus compound SLO violations?",
            "RQ3": "Can lightweight telemetry distinguish the mechanisms responsible for compound SLO degradation?",
            "RQ4": "How robust are mechanism signatures under predefined telemetry degradation?",
        },
        "pre_registered_expectations": {
            "input": "Increased prompt tokens with stronger TTFT effect than decode-duration effect.",
            "output": "Increased output tokens with stronger post-first-token duration effect.",
            "load": "Direct queue evidence with strong TTFT inflation.",
            "downstream": "Configured and observed dependency latency with total-latency increase.",
        },
        "replication_unit": "run/repetition",
        "confidence_intervals": "95% Student-t intervals across repetitions",
        "paired_analysis": "Conditions are paired by repetition block where the same repetition contains baseline, singles, and compounds.",
        "multiple_comparisons": "Holm correction within predefined families: isolated mechanisms, pairwise compounds, and telemetry-ablation diagnostics.",
        "effect_sizes": ["absolute delta", "ratio to matched control", "paired standardized effect where appropriate"],
        "practical_significance": "Report SLO boundary crossing and magnitude of violation before interpreting p-values.",
        "compound_interaction": {
            "comparisons": "Compare each compound against its two component singles and matched controls.",
            "metrics": ["compound observed effect", "dominance", "masking", "super_additive_delta", "sub_additive_delta"],
            "additivity_assumption": "Do not assume additivity; report additive residuals descriptively.",
        },
        "csd_scope": {
            "name": "Compound SLO Decomposition",
            "execute_in_phase5": False,
            "inputs": ["frozen SLO aggregates", "mechanism ground truth", "telemetry feature table"],
            "outputs": ["compound-vs-single decomposition summaries", "candidate mechanism attribution features"],
            "ground_truth_labels": ["active_mechanisms", "mechanism_levels", "compound_degree", "campaign_condition_id"],
        },
        "rca_scope": {
            "execute_in_phase5": False,
            "target_output": "ranked root-cause mechanisms",
            "candidate_metrics": ["top-1 accuracy", "top-k accuracy", "MRR"],
            "required_labels": ["active_mechanisms", "mechanism_family", "mechanism_levels"],
        },
        "degraded_telemetry_scope": {
            "execute_in_phase5": False,
            "modes": ["remove_queue_telemetry", "remove_downstream_telemetry", "remove_token_timing", "sample_request_telemetry_10_percent"],
        },
        "telemetry_cost_scope": {
            "status": "DEFERRED",
            "reason": "No validated low-overhead collection method exists yet for telemetry overhead attribution.",
        },
        "cross_runtime_scope": {
            "primary_campaign": "Mac/Darwin arm64 llama.cpp Metal with Qwen2.5-0.5B Q4_K_M",
            "secondary_generalization_campaign": "future non-Metal or cross-model repetition",
            "deferred": ["CPU contention as primary Metal degradation", "memory pressure", "I/O pressure"],
        },
    }


def telemetry_schema() -> dict[str, Any]:
    return {
        "primary_request_fields": [
            "time_to_first_token",
            "total_latency",
            "post_first_token_duration",
            "observed_seconds_per_output_token",
            "server_prompt_tokens",
            "server_output_tokens",
            "scheduler_slip",
            "dependency_duration",
        ],
        "runtime_fields": [
            "llamacpp:requests_processing",
            "llamacpp:requests_deferred",
            "requests_deferred_max",
            "observed_throughput",
        ],
        "outcome_fields": ["planned_request_count", "successful", "failed", "admission_failed"],
        "system_fields": [
            "host_cpu_percent",
            "host_memory_percent",
            "llama_process_cpu_percent",
            "llama_process_rss_bytes",
            "gateway_process_cpu_percent",
            "gateway_process_rss_bytes",
            "dependency_process_cpu_percent",
            "dependency_process_rss_bytes",
        ],
        "trace_fields": ["trace_id", "span_id", "parent_span_id", "span_name", "start_time", "end_time", "duration", "status", "attributes"],
        "direct_mechanism_evidence": {
            "input": ["server_prompt_tokens"],
            "output": ["server_output_tokens", "post_first_token_duration"],
            "load": ["requests_deferred_max", "llamacpp:requests_deferred"],
            "downstream": ["configured_delay_ms", "dependency_duration"],
        },
    }


def confound_rules() -> dict[str, Any]:
    return {
        "queue_confounded": {
            "input": "requests_deferred_max > 0",
            "output": "requests_deferred_max > 0",
            "load": "not a confound; queueing is the intended mechanism",
            "downstream": "requests_deferred_max > 0 is VALID_WITH_SECONDARY_EFFECT unless it causes mandatory validity failure",
        },
        "states": {
            "VALID": "All mandatory validity criteria pass and no disallowed confound is present.",
            "CONFUNDED": "Mechanism was applied but a predefined secondary mechanism appeared.",
            "VALID_WITH_SECONDARY_EFFECT": "Run is usable for mechanism execution but excluded from pure-isolation claims.",
            "INVALID": "Mandatory validity criterion failed.",
        },
        "invalidation_rules": [
            "service health/readiness failure",
            "model/runtime version, build, commit, or required flag mismatch",
            "git_dirty is true for publication campaign execution",
            "missing mandatory telemetry or trace artifacts",
            "request-count mismatch",
            "any admission failure unless the specific analysis explicitly models admission control",
            "port or service contamination",
            "artifact corruption or hash mismatch",
            "incorrect mechanism configuration or failed mechanism verification",
            "downstream cleanup cannot restore delay_ms=0",
            "owned process cleanup failure",
            "insufficient disk space below frozen guard",
        ],
        "rerun_policy": {
            "invalid_runs": "May be rerun with a new attempt ID; preserve invalid artifacts and reason.",
            "valid_surprising_runs": "Must not be silently rerun; surprising valid data are retained.",
        },
    }


def factorial_design() -> dict[str, Any]:
    return {
        "INPUT_LOAD": {
            "factor_a": "input_medium",
            "factor_b": "load_12rps",
            "A0B0": "BASELINE",
            "A1B0": "INPUT_MEDIUM",
            "A0B1": "LOAD_MEDIUM",
            "A1B1": "INPUT_LOAD",
        },
        "INPUT_DOWNSTREAM": {
            "factor_a": "input_medium",
            "factor_b": "downstream_100ms",
            "A0B0": "BASELINE",
            "A1B0": "INPUT_MEDIUM",
            "A0B1": "DOWNSTREAM_MEDIUM",
            "A1B1": "INPUT_DOWNSTREAM",
        },
        "OUTPUT_LOAD": {
            "factor_a": "output_32",
            "factor_b": "load_12rps",
            "A0B0": "OUTPUT_CONTROL",
            "A1B0": "OUTPUT_MEDIUM",
            "A0B1": "OUTPUT_LOAD_CONTROL",
            "A1B1": "OUTPUT_LOAD",
        },
        "OUTPUT_DOWNSTREAM": {
            "factor_a": "output_32",
            "factor_b": "downstream_100ms",
            "A0B0": "OUTPUT_CONTROL",
            "A1B0": "OUTPUT_MEDIUM",
            "A0B1": "OUTPUT_DOWNSTREAM_CONTROL",
            "A1B1": "OUTPUT_DOWNSTREAM",
        },
        "LOAD_DOWNSTREAM": {
            "factor_a": "load_12rps",
            "factor_b": "downstream_100ms",
            "A0B0": "BASELINE",
            "A1B0": "LOAD_MEDIUM",
            "A0B1": "DOWNSTREAM_MEDIUM",
            "A1B1": "LOAD_DOWNSTREAM",
        },
        "deferred": {
            "INPUT_OUTPUT": "Deferred because Phase 4C input and output references use different prompt families; a matched continuation-small/medium calibration is required before claiming a factorial input-output interaction.",
        },
    }


def build_slo_calibration_artifact(baseline_runs: list[dict], output_runs: list[dict]) -> dict[str, Any]:
    if len([r for r in baseline_runs if r.get("valid")]) < CALIBRATION_REPETITIONS:
        raise ValueError("dedicated calibration requires at least 8 valid BASELINE runs")
    if len([r for r in output_runs if r.get("valid")]) < CALIBRATION_REPETITIONS:
        raise ValueError("dedicated calibration requires at least 8 valid OUTPUT_CONTROL runs")

    def values(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get("valid") and r.get(key) is not None]

    baseline_ttft = values(baseline_runs, "ttft_p95")
    output_ttft = values(output_runs, "ttft_p95")
    baseline_total = values(baseline_runs, "total_latency_p95")
    output_total = values(output_runs, "total_latency_p95")
    output_decode = values(output_runs, "post_first_token_duration_p95")

    baseline_ttft_limit = prediction_limit(baseline_ttft, direction="upper")
    output_ttft_limit = prediction_limit(output_ttft, direction="upper")
    baseline_total_limit = prediction_limit(baseline_total, direction="upper")
    output_total_limit = prediction_limit(output_total, direction="upper")
    decode_limit = prediction_limit(output_decode, direction="upper")

    return {
        "schema_version": "sloscope.slo_calibration.v1",
        "status": "DEDICATED_PRECAMPAIGN_CALIBRATION_COMPLETE",
        "dedicated_calibration": True,
        "formal_campaign_outcomes_used": False,
        "alpha": 0.05,
        "baseline_repetitions": len([r for r in baseline_runs if r.get("valid")]),
        "output_control_repetitions": len([r for r in output_runs if r.get("valid")]),
        "formula": "UPL = mean + t_(0.95,n-1) * s * sqrt(1 + 1/n)",
        "family_limits": {
            "TTFT_SLO": {
                "baseline": {**baseline_ttft_limit, "source_run_ids": [r["run_id"] for r in baseline_runs if r.get("valid")], "source_hashes": [r["run_manifest_sha256"] for r in baseline_runs if r.get("valid")]},
                "output_control": {**output_ttft_limit, "source_run_ids": [r["run_id"] for r in output_runs if r.get("valid")], "source_hashes": [r["run_manifest_sha256"] for r in output_runs if r.get("valid")]},
            },
            "TOTAL_LATENCY_SLO": {
                "baseline": {**baseline_total_limit, "source_run_ids": [r["run_id"] for r in baseline_runs if r.get("valid")], "source_hashes": [r["run_manifest_sha256"] for r in baseline_runs if r.get("valid")]},
                "output_control": {**output_total_limit, "source_run_ids": [r["run_id"] for r in output_runs if r.get("valid")], "source_hashes": [r["run_manifest_sha256"] for r in output_runs if r.get("valid")]},
            },
            "DECODE_DURATION_SLO": {
                "output_control": {**decode_limit, "source_run_ids": [r["run_id"] for r in output_runs if r.get("valid")], "source_hashes": [r["run_manifest_sha256"] for r in output_runs if r.get("valid")]},
            },
        },
        "metrics": {
            "TTFT_SLO": {
                "direction": "upper",
                "final_numeric_threshold": max(baseline_ttft_limit["threshold"], output_ttft_limit["threshold"]),
                "selected_family": "baseline" if baseline_ttft_limit["threshold"] >= output_ttft_limit["threshold"] else "output_control",
            },
            "TOTAL_LATENCY_SLO": {
                "direction": "upper",
                "final_numeric_threshold": max(baseline_total_limit["threshold"], output_total_limit["threshold"]),
                "selected_family": "baseline" if baseline_total_limit["threshold"] >= output_total_limit["threshold"] else "output_control",
            },
            "DECODE_DURATION_SLO": {
                "direction": "upper",
                "final_numeric_threshold": decode_limit["threshold"],
                "selected_family": "output_control",
            },
        },
        "throughput": {
            "status": "DIAGNOSTIC_ONLY",
            "reason": "Raw successful requests/sec is demand-dependent and not a counted universal SLO.",
        },
        "control_sanity": [],
    }


def repetition_justification() -> dict[str, Any]:
    pilot = {
        "baseline_ttft": [0.0527740830, 0.0495556250, 0.0503748125],
        "baseline_total_latency": [0.1876164375, 0.1779356670, 0.1801858955],
        "output_control_decode": [0.1208236670, 0.1209362295, 0.1227245625],
    }
    candidates = [5, 6, 8, 10]
    rows = []
    for metric, values in pilot.items():
        for n in candidates:
            rows.append(
                {
                    "metric": metric,
                    "candidate_n": n,
                    "relative_half_width": ci_relative_half_width(values, n),
                }
            )
    return {
        "schema_version": "sloscope.repetition_justification.v1",
        "source": "accepted Phase 4C run-level matched-control summaries",
        "target_relative_half_width": 0.075,
        "candidate_repetitions": candidates,
        "calculations": rows,
        "selected_repetitions": REPETITIONS,
        "rationale": "n=5 met the nominal Phase-4 precision target for several control metrics; n=8 is conservatively frozen for improved precision and compound-analysis stability within acceptable campaign cost.",
    }


def calibration_source_path(root: Path) -> Path:
    return root / "calibrations" / "phase5.2-slo" / "slo-calibration.json"


def load_slo_calibration_for_freeze(root: Path) -> dict[str, Any]:
    path = calibration_source_path(root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": "sloscope.slo_calibration.v1",
        "status": "DEDICATED_PRECAMPAIGN_CALIBRATION_MISSING",
        "dedicated_calibration": False,
        "formal_campaign_outcomes_used": False,
        "baseline_repetitions": 0,
        "output_control_repetitions": 0,
        "metrics": {},
    }


def preflight_spec(frozen_source_sha: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "sloscope.phase6_preflight.v1",
        "freeze_hash_algorithm": "sha256 over canonical frozen input set; campaign-manifest hash fields excluded",
        "freeze_hash_input_manifest": [
            "campaigns/phase5/campaign-manifest.json",
            "campaigns/phase5/campaign-manifest.csv",
            "campaigns/phase5/mechanism-definitions.json",
            "campaigns/phase5/slo-definitions.json",
            "campaigns/phase5/analysis-plan.json",
            "campaigns/phase5/telemetry-schema.json",
            "campaigns/phase5/confound-validity-rules.json",
            "campaigns/phase5/factorial-design.json",
            "campaigns/phase5/slo-calibration.json",
            "campaigns/phase5/repetition-justification.json",
            "campaigns/phase5/phase6-preflight.json",
            "configs/phase5/**/*.yaml",
        ],
        "expected_freeze_hash_source": "campaign-manifest.json:campaign_freeze_sha256",
        "checks": [
            "git_dirty_false",
            "source_tree_sha256_matches_frozen",
            "campaign_freeze_sha256_matches",
            "slo_calibration_artifact_present_and_hashed",
            "runtime_model_build_and_commit_match",
            "server_arguments_match",
            "prompt_cache_disabled",
            "sufficient_free_disk",
            "services_ready",
            "expected_run_directories_absent",
            "config_hashes_match",
        ],
        "frozen_source_tree_sha256": frozen_source_sha,
    }


def _check(results: list[dict], name: str, ok: bool, detail: str = "") -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def run_preflight(
    root: Path,
    *,
    frozen_source_sha: str | None = None,
    runtime_info: dict[str, Any] | None = None,
    service_status: dict[str, Any] | None = None,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    results: list[dict] = []
    manifest_path = root / "campaigns" / "phase5" / "campaign-manifest.json"
    manifest = {}
    if not manifest_path.exists():
        _check(results, "campaign_manifest_exists", False, "missing campaign manifest")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check(results, "campaign_manifest_exists", True)
    expected_source = frozen_source_sha or manifest.get("source_tree_sha256")
    _check(results, "git_clean", (not require_clean_git) or (git_dirty(root) is False))
    _check(results, "source_tree_sha256", source_tree_sha256(root) == expected_source)
    freeze_inputs = freeze_input_paths(root)
    recomputed = freeze_hash(freeze_inputs, root) if manifest_path.exists() else None
    _check(results, "campaign_freeze_sha256", recomputed == manifest.get("campaign_freeze_sha256"), str(recomputed))
    fv = root / "campaigns" / "phase5" / "freeze-validation.json"
    freeze_valid = fv.exists() and json.loads(fv.read_text(encoding="utf-8")).get("valid") is True
    _check(results, "freeze_validation", freeze_valid)
    slo_path = root / "campaigns" / "phase5" / "slo-calibration.json"
    slo = json.loads(slo_path.read_text(encoding="utf-8")) if slo_path.exists() else {}
    _check(results, "slo_calibration_complete", slo.get("status") == "DEDICATED_PRECAMPAIGN_CALIBRATION_COMPLETE" and slo.get("dedicated_calibration") is True)
    rt = runtime_info or {}
    _check(results, "runtime_version", rt.get("version", RUNTIME_VERSION) == RUNTIME_VERSION)
    _check(results, "runtime_build", str(rt.get("build", RUNTIME_BUILD)) == RUNTIME_BUILD)
    _check(results, "runtime_commit", rt.get("commit", RUNTIME_COMMIT) == RUNTIME_COMMIT)
    _check(results, "model_id", rt.get("model_id", MODEL_ID) == MODEL_ID)
    args = rt.get("server_arguments", LLAMA_ARGS)
    _check(results, "server_arguments", args == LLAMA_ARGS)
    _check(results, "prompt_cache_disabled", "--no-cache-prompt" in args)
    _check(results, "metrics_enabled", "--metrics" in args)
    _check(results, "parallel_2", "--parallel" in args and args[args.index("--parallel") + 1] == "2")
    _check(results, "ctx_size_2048", "--ctx-size" in args and args[args.index("--ctx-size") + 1] == "2048")
    svc = service_status or {"llama_healthy": True, "gateway_ready": True, "dependency_healthy": True, "dependency_delay_ms": 0}
    _check(results, "llama_healthy", svc.get("llama_healthy") is True)
    _check(results, "gateway_ready", svc.get("gateway_ready") is True)
    _check(results, "dependency_healthy", svc.get("dependency_healthy") is True)
    _check(results, "dependency_zero_delay", int(svc.get("dependency_delay_ms", -1)) == 0)
    _check(results, "free_disk", shutil.disk_usage(root).free >= int(manifest.get("minimum_free_space_bytes", MIN_FREE_SPACE_BYTES)))
    rows = manifest.get("campaign_order") or []
    run_dirs = [root / row.get("run_directory", "") for row in rows]
    _check(results, "run_directories_absent", all(not path.exists() for path in run_dirs))
    config_ok = True
    for row in rows:
        path = root / row["config_path"]
        if not path.exists():
            config_ok = False
            break
        try:
            cfg = ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
            cfg.validate()
            exp = cfg.runtime.parameters.get("experimental_condition") or {}
            if exp.get("run_id") != row["run_id"] or exp.get("campaign_condition_id") != row["condition_id"]:
                config_ok = False
                break
        except Exception:
            config_ok = False
            break
    _check(results, "configs_valid", config_ok)
    issues = [r["check"] for r in results if not r["ok"]]
    return {"valid": not issues, "issues": issues, "checks": results}


def ensure_run_directory_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"run directory already exists: {path}")


def execute_cooldown(seconds: float = COOLDOWN_SECONDS, sleeper=time.sleep) -> dict[str, float]:
    start = time.monotonic()
    sleeper(seconds)
    end = time.monotonic()
    return {"cooldown_seconds": seconds, "cooldown_start_time": start, "cooldown_end_time": end}


def calibration_order(seed: int = 90210) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    order = 1
    for rep in range(1, CALIBRATION_REPETITIONS + 1):
        members = ["BASELINE", "OUTPUT_CONTROL"]
        rng.shuffle(members)
        for within, condition_id in enumerate(members, start=1):
            run_id = f"slo-cal-r{rep:02d}-{condition_slug(condition_id)}-a01"
            rows.append(
                {
                    "calibration_id": "phase5.2-slo",
                    "condition_id": condition_id,
                    "run_id": run_id,
                    "repetition": rep,
                    "attempt": 1,
                    "block_id": f"slo-cal-block-{rep:02d}",
                    "within_block_order": within,
                    "calibration_order": order,
                    "status": "PLANNED",
                }
            )
            order += 1
    return rows


def config_for_calibration(row: dict[str, Any], *, gateway_port: int, pids: dict[str, int], llama_version: dict[str, Any]) -> ExperimentConfig:
    condition = {c.condition_id: c for c in conditions()}[row["condition_id"]]
    cfg = config_for_condition(condition, repetition=int(row["repetition"]), attempt=int(row["attempt"]), run_id=row["run_id"])
    exp = {
        "campaign_id": "phase5.2-slo-calibration",
        "campaign_condition_id": condition.condition_id,
        "condition_id": condition.condition_id,
        "mechanism_family": "control",
        "active_mechanisms": [],
        "mechanism_levels": {},
        "compound_degree": 0,
        "expected_control_id": None,
        "repetition": int(row["repetition"]),
        "attempt": int(row["attempt"]),
        "run_id": row["run_id"],
    }
    params = {
        **cfg.runtime.parameters,
        "base_url": f"http://127.0.0.1:{gateway_port}",
        "server_pid": pids["llama"],
        "gateway_pid": pids["gateway"],
        "dependency_pid": pids["dependency"],
        "server_metadata": llama_version,
        "experimental_condition": exp,
    }
    return ExperimentConfig.from_dict({**cfg.to_dict(include_hash=False), "runtime": {**cfg.runtime.__dict__, "parameters": params}, "mechanisms": []}).with_hash()


def run_dedicated_slo_calibration(root: Path, logs: Path, *, cooldown_seconds: float = COOLDOWN_SECONDS) -> int:
    from sloscope.artifacts.validation import validate_run
    from sloscope.artifacts.writer import read_json, write_jsonl
    from sloscope.pilot_phase4a import (
        choose_port,
        json_request,
        parse_llama_version,
        start_service,
        terminate_services,
        utc_now,
        wait_json,
        write_json,
        write_csv,
    )
    import sys

    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    llama_version = parse_llama_version()
    llama_port = choose_port(8080)
    gateway_port = choose_port(8090)
    dependency_port = choose_port(8091)
    llama_args = list(LLAMA_ARGS)
    llama_args[llama_args.index("--port") + 1] = str(llama_port)
    services = []
    ledger: list[dict] = []
    baseline_rows: list[dict] = []
    output_rows: list[dict] = []
    order = calibration_order()
    write_json(root / "calibration-order.json", {"seed": 90210, "order": order})
    try:
        services.append(start_service("dependency", [sys.executable, "-m", "sloscope.gateway.dependency", "--host", "127.0.0.1", "--port", str(dependency_port), "--maximum-delay-ms", "250"], logs / "dependency.log", dependency_port))
        wait_json(f"http://127.0.0.1:{dependency_port}/control/status", timeout_seconds=20)
        services.append(start_service("llama", ["llama-server", *llama_args], logs / "llama-server.log", llama_port))
        wait_json(f"http://127.0.0.1:{llama_port}/health", timeout_seconds=180)
        services.append(start_service("gateway", [sys.executable, "-m", "sloscope.gateway.server", "--host", "127.0.0.1", "--port", str(gateway_port), "--llama-base-url", f"http://127.0.0.1:{llama_port}", "--dependency-base-url", f"http://127.0.0.1:{dependency_port}", "--timeout", "120.0"], logs / "gateway.log", gateway_port))
        wait_json(f"http://127.0.0.1:{gateway_port}/ready", timeout_seconds=180)
        pids = {service.name: service.process.pid for service in services}
        manifest = {
            "calibration_id": "phase5.2-slo",
            "purpose": "dedicated_precampaign_slo_calibration",
            "created_at": utc_now(),
            "order_seed": 90210,
            "runtime": llama_version,
            "model": {"model_ref": MODEL_REF, "model_id": MODEL_ID},
            "server_arguments": llama_args,
            "warmup_request_count": WARMUP_REQUESTS,
            "measured_request_count": REQUEST_COUNT,
            "cooldown_seconds": cooldown_seconds,
            "services": {"llama": {"pid": pids["llama"], "port": llama_port}, "gateway": {"pid": pids["gateway"], "port": gateway_port}, "dependency": {"pid": pids["dependency"], "port": dependency_port, "delay_ms": 0}},
            "order": order,
        }
        write_json(root / "calibration-manifest.json", manifest)
        for row in order:
            run_dir = runs_root / row["run_id"]
            cfg = config_for_calibration(row, gateway_port=gateway_port, pids=pids, llama_version=llama_version)
            started = time.monotonic()
            item = {**row, "actual_start": utc_now(), "run_dir": str(run_dir), "config_hash": cfg.config_hash}
            try:
                result_dir = ExperimentRunner(cfg, runs_root).run()
                validation = validate_run(result_dir)
                agg = run_aggregates(result_dir)
                run_manifest_path = result_dir / "manifest.json"
                run_manifest_hash = sha256_file(run_manifest_path)
                run_manifest = read_json(run_manifest_path)
                warmup = (run_manifest.get("runtime_metadata") or {}).get("warmup") or {}
                item.update(agg)
                item.update({
                    "valid": validation.get("valid") is True,
                    "issues": ";".join(issue.get("code", "") for issue in validation.get("issues", [])),
                    "run_manifest_sha256": run_manifest_hash,
                    "warmup_request_count": warmup.get("warmup_request_count"),
                    "warmup_success_count": warmup.get("warmup_success_count"),
                    "warmup_failure_count": warmup.get("warmup_failure_count"),
                    "actual_end": utc_now(),
                    "duration_seconds": time.monotonic() - started,
                    "status": "VALID" if validation.get("valid") is True else "INVALID",
                })
            except Exception as exc:
                item.update({"valid": False, "issues": str(exc), "actual_end": utc_now(), "duration_seconds": time.monotonic() - started, "status": "INVALID"})
            ledger.append(item)
            if item["condition_id"] == "BASELINE":
                baseline_rows.append(item)
            else:
                output_rows.append(item)
            write_jsonl(root / "calibration-ledger.jsonl", ledger)
            if row != order[-1]:
                execute_cooldown(cooldown_seconds)
        write_csv(root / "baseline-runs.csv", baseline_rows)
        write_csv(root / "output-control-runs.csv", output_rows)
        slo = build_slo_calibration_artifact(baseline_rows, output_rows)
        sanity = []
        for row in baseline_rows + output_rows:
            sanity.append(
                {
                    "run_id": row["run_id"],
                    "condition_id": row["condition_id"],
                    "ttft_crossed": row.get("ttft_p95") is not None and row["ttft_p95"] > slo["metrics"]["TTFT_SLO"]["final_numeric_threshold"],
                    "total_latency_crossed": row.get("total_latency_p95") is not None and row["total_latency_p95"] > slo["metrics"]["TOTAL_LATENCY_SLO"]["final_numeric_threshold"],
                    "decode_crossed": row.get("post_first_token_duration_p95") is not None and row["condition_id"] == "OUTPUT_CONTROL" and row["post_first_token_duration_p95"] > slo["metrics"]["DECODE_DURATION_SLO"]["final_numeric_threshold"],
                }
            )
        slo["control_sanity"] = sanity
        write_json(root / "slo-calibration.json", slo)
        (root / "calibration-report.md").write_text(
            "# Phase 5.2 SLO Calibration\n\n"
            f"Valid runs: {sum(1 for r in ledger if r.get('valid'))}/{len(ledger)}\n\n"
            f"TTFT_SLO: {slo['metrics']['TTFT_SLO']['final_numeric_threshold']}\n\n"
            f"TOTAL_LATENCY_SLO: {slo['metrics']['TOTAL_LATENCY_SLO']['final_numeric_threshold']}\n\n"
            f"DECODE_DURATION_SLO: {slo['metrics']['DECODE_DURATION_SLO']['final_numeric_threshold']}\n",
            encoding="utf-8",
        )
        return 0 if all(item.get("valid") for item in ledger) else 1
    finally:
        try:
            json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
        except Exception:
            pass
        cleanup = terminate_services(reversed(services))
        write_json(root / "service-cleanup.json", cleanup)


def conditions() -> list[Condition]:
    return [
        Condition("BASELINE", "control", (), {}, 0, None, "synthetic-input-small", 16, 0.25, 64, notes="Matched control for input, load, and downstream families."),
        Condition("OUTPUT_CONTROL", "control", (), {}, 0, None, "synthetic-continuation", 16, 0.5, 64, notes="Matched control for output/decode family."),
        Condition("OUTPUT_LOAD_CONTROL", "control", ("load_12rps",), {"load_12rps": "medium"}, 1, "OUTPUT_CONTROL", "synthetic-continuation", 16, 1.0 / 12.0, 64, notes="Continuation-family 12 rps cell for output-load factorial."),
        Condition("OUTPUT_DOWNSTREAM_CONTROL", "control", ("downstream_100ms",), {"downstream_100ms": "medium"}, 1, "OUTPUT_CONTROL", "synthetic-continuation", 16, 0.5, 64, 100, notes="Continuation-family downstream cell for output-downstream factorial."),
        Condition("INPUT_MEDIUM", "input", ("input_medium",), {"input_medium": "medium"}, 1, "BASELINE", "synthetic-input-medium", 16, 0.25, 64),
        Condition("OUTPUT_MEDIUM", "output", ("output_32",), {"output_32": "medium"}, 1, "OUTPUT_CONTROL", "synthetic-continuation", 32, 0.5, 64),
        Condition("LOAD_MEDIUM", "load", ("load_12rps",), {"load_12rps": "medium"}, 1, "BASELINE", "synthetic-input-small", 16, 1.0 / 12.0, 64),
        Condition("DOWNSTREAM_MEDIUM", "downstream", ("downstream_100ms",), {"downstream_100ms": "medium"}, 1, "BASELINE", "synthetic-input-small", 16, 0.25, 64, 100),
        Condition("INPUT_LOAD", "compound", ("input_medium", "load_12rps"), {"input_medium": "medium", "load_12rps": "medium"}, 2, "BASELINE", "synthetic-input-medium", 16, 1.0 / 12.0, 64),
        Condition("INPUT_DOWNSTREAM", "compound", ("input_medium", "downstream_100ms"), {"input_medium": "medium", "downstream_100ms": "medium"}, 2, "BASELINE", "synthetic-input-medium", 16, 0.25, 64, 100),
        Condition("OUTPUT_LOAD", "compound", ("output_32", "load_12rps"), {"output_32": "medium", "load_12rps": "medium"}, 2, "OUTPUT_CONTROL", "synthetic-continuation", 32, 1.0 / 12.0, 64, notes="Queueing is intentional through load mechanism."),
        Condition("OUTPUT_DOWNSTREAM", "compound", ("output_32", "downstream_100ms"), {"output_32": "medium", "downstream_100ms": "medium"}, 2, "OUTPUT_CONTROL", "synthetic-continuation", 32, 0.5, 64, 100),
        Condition("LOAD_DOWNSTREAM", "compound", ("load_12rps", "downstream_100ms"), {"load_12rps": "medium", "downstream_100ms": "medium"}, 2, "BASELINE", "synthetic-input-small", 16, 1.0 / 12.0, 64, 100),
    ]


def _mechanism_config(condition: Condition) -> list[MechanismConfig]:
    mechanisms = []
    if "downstream_100ms" in condition.active_mechanisms:
        mechanisms.append(
            MechanismConfig(
                "downstream-latency-100ms",
                "downstream_latency",
                100.0,
                "synthetic_dependency",
                0.0,
                60.0,
                {"delay_ms": 100, "verify_request": True, "verify_min_fraction": 0.8},
            )
        )
    return mechanisms


def experimental_condition_for(condition: Condition, repetition: int, attempt: int, run_id: str) -> dict[str, Any]:
    return {
        "campaign_id": "phase6-publication-campaign",
        "campaign_condition_id": condition.condition_id,
        "condition_id": condition.condition_id,
        "mechanism_family": condition.mechanism_family,
        "active_mechanisms": list(condition.active_mechanisms),
        "mechanism_levels": condition.mechanism_levels,
        "compound_degree": condition.compound_degree,
        "expected_control_id": condition.expected_control_id,
        "repetition": repetition,
        "attempt": attempt,
        "run_id": run_id,
    }


def config_for_condition(condition: Condition, *, repetition: int = 1, attempt: int = 1, run_id: str | None = None) -> ExperimentConfig:
    rid = run_id or run_id_for(condition.condition_id, repetition, attempt)
    exp = experimental_condition_for(condition, repetition, attempt, rid)
    return ExperimentConfig(
        schema_version=SCHEMA_VERSION,
        run_id=rid,
        seed=4242,
        runtime=RuntimeConfig(
            "llamacpp-local",
            "llamacpp",
            MODEL_ID,
            {
                "base_url": "http://127.0.0.1:8090",
                "request_endpoint": "/v1/completions",
                "max_outstanding_requests": condition.max_outstanding_requests,
                "request_timeout": 120.0,
                "telemetry_interval_seconds": 0.10,
                "temperature": 0,
                "server_command": "llama-server",
                "server_arguments": LLAMA_ARGS,
                "llama_cpp_version": RUNTIME_VERSION,
                "llama_cpp_build_number": RUNTIME_BUILD,
                "llama_cpp_revision": RUNTIME_COMMIT,
                "model_ref": MODEL_REF,
                "prompt_cache": "disabled",
                "warmup_request_count": WARMUP_REQUESTS,
                "cooldown_seconds": COOLDOWN_SECONDS,
                "dependency_delay_ms": condition.dependency_delay_ms,
                "offered_rps": condition.offered_rps,
                "experimental_condition_required": True,
                "experimental_condition": exp,
            },
        ),
        workload=WorkloadConfig(
            "constant_open_loop",
            REQUEST_COUNT,
            2,
            condition.prompt_profile,
            "fixed",
            condition.target_output_tokens,
            inter_arrival_seconds=condition.inter_arrival_seconds,
            burst_size=2,
            burst_interval_seconds=1.0,
            randomized_output=False,
            output_jitter_tokens=0,
            start_offset_seconds=1.0,
        ),
        mechanisms=_mechanism_config(condition),
        telemetry=TelemetryConfig(True, True, True, True),
        safety=SafetyConfig(
            maximum_cpu_stress=0.0,
            maximum_allocated_pressure_memory=0,
            maximum_disk_io=0,
            experiment_timeout=300.0,
            require_requests_within_mechanism_window=bool(condition.active_mechanisms),
            maximum_dependency_delay_ms=250,
            mechanism_watchdog_interval_seconds=0.25,
        ),
    ).with_hash()


def campaign_rows(revision: str | None = None) -> list[dict[str, Any]]:
    rng = random.Random(ORDER_SEED)
    rows: list[dict[str, Any]] = []
    order = 1
    conds = conditions()
    for repetition in range(1, REPETITIONS + 1):
        shuffled = list(conds)
        rng.shuffle(shuffled)
        for condition in shuffled:
            attempt = 1
            run_id = run_id_for(condition.condition_id, repetition, attempt)
            rows.append(
                {
                    "campaign_id": "phase6-publication-campaign",
                    "condition_id": condition.condition_id,
                    "run_id": run_id,
                    "mechanism_family": condition.mechanism_family,
                    "active_mechanisms": json.dumps(list(condition.active_mechanisms), sort_keys=True),
                    "mechanism_levels": json.dumps(condition.mechanism_levels, sort_keys=True),
                    "compound_degree": condition.compound_degree,
                    "repetition": repetition,
                    "attempt": attempt,
                    "campaign_order": order,
                    "config_path": f"configs/phase5/runs/{run_id}.yaml",
                    "run_directory": f"runs/phase6/{run_id}",
                    "expected_control_id": condition.expected_control_id or "",
                    "source_commit": revision or "",
                    "status": "PLANNED",
                }
            )
            order += 1
    return rows


def config_filename(condition: Condition) -> str:
    mapping = {
        "BASELINE": "baseline.yaml",
        "OUTPUT_CONTROL": "output-control.yaml",
        "OUTPUT_LOAD_CONTROL": "output-load-control.yaml",
        "OUTPUT_DOWNSTREAM_CONTROL": "output-downstream-control.yaml",
        "INPUT_MEDIUM": "single-input.yaml",
        "OUTPUT_MEDIUM": "single-output.yaml",
        "LOAD_MEDIUM": "single-load.yaml",
        "DOWNSTREAM_MEDIUM": "single-downstream.yaml",
    }
    return mapping.get(condition.condition_id, f"compound-{condition.condition_id.lower().replace('_', '-')}.yaml")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def freeze_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        if path.name == "campaign-manifest.json" and path.parent.name == "phase5":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest.pop("campaign_freeze_sha256", None)
            manifest.pop("campaign_manifest_sha256", None)
            data = _canonical_json(manifest).encode("utf-8")
        else:
            data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def manifest_hash(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("campaign_freeze_sha256", None)
    manifest.pop("campaign_manifest_sha256", None)
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def freeze_input_paths(root: Path) -> list[Path]:
    campaigns_dir = root / "campaigns" / "phase5"
    configs_dir = root / "configs" / "phase5"
    return [
        campaigns_dir / "campaign-manifest.json",
        campaigns_dir / "campaign-manifest.csv",
        campaigns_dir / "mechanism-definitions.json",
        campaigns_dir / "slo-definitions.json",
        campaigns_dir / "analysis-plan.json",
        campaigns_dir / "telemetry-schema.json",
        campaigns_dir / "confound-validity-rules.json",
        campaigns_dir / "factorial-design.json",
        campaigns_dir / "slo-calibration.json",
        campaigns_dir / "repetition-justification.json",
        campaigns_dir / "phase6-preflight.json",
        *sorted(configs_dir.glob("*.yaml")),
        *sorted((configs_dir / "runs").glob("*.yaml")),
    ]


def validate_freeze(root: Path = Path(".")) -> list[str]:
    issues: list[str] = []
    conds = conditions()
    ids = [c.condition_id for c in conds]
    if len(ids) != len(set(ids)):
        issues.append("duplicate condition IDs")
    expected = {
        "BASELINE",
        "OUTPUT_CONTROL",
        "OUTPUT_LOAD_CONTROL",
        "OUTPUT_DOWNSTREAM_CONTROL",
        "INPUT_MEDIUM",
        "OUTPUT_MEDIUM",
        "LOAD_MEDIUM",
        "DOWNSTREAM_MEDIUM",
        "INPUT_LOAD",
        "INPUT_DOWNSTREAM",
        "OUTPUT_LOAD",
        "OUTPUT_DOWNSTREAM",
        "LOAD_DOWNSTREAM",
    }
    if set(ids) != expected:
        issues.append(f"condition set mismatch: {sorted(set(ids) ^ expected)}")
    if any("cpu" in mech for c in conds for mech in c.active_mechanisms):
        issues.append("CPU contention appears in primary matrix")
    rows = campaign_rows()
    if len(rows) != len(conds) * REPETITIONS:
        issues.append("campaign count mismatch")
    run_ids = [r["run_id"] for r in rows]
    if len(run_ids) != len(set(run_ids)):
        issues.append("duplicate campaign run identity")
    run_dirs = [r["run_directory"] for r in rows]
    if len(run_dirs) != len(set(run_dirs)):
        issues.append("duplicate run output destination")
    config_paths = [r["config_path"] for r in rows]
    if len(config_paths) != len(set(config_paths)):
        issues.append("duplicate config output destination")
    seen = {(r["condition_id"], r["repetition"]) for r in rows}
    for cond in ids:
        for rep in range(1, REPETITIONS + 1):
            if (cond, rep) not in seen:
                issues.append(f"missing repetition {cond} rep {rep}")
    for c in conds:
        if c.expected_control_id and c.expected_control_id not in ids:
            issues.append(f"missing control {c.expected_control_id}")
        if c.compound_degree == 2:
            if len(c.active_mechanisms) != 2:
                issues.append(f"bad compound degree {c.condition_id}")
            for mech in c.active_mechanisms:
                if c.mechanism_levels.get(mech) != "medium":
                    issues.append(f"compound {c.condition_id} does not reference frozen medium level")
        path = root / "configs" / "phase5" / config_filename(c)
        if root != Path(".") and not path.exists():
            issues.append(f"missing config {path}")
    design = factorial_design()
    for compound in [c for c in conds if c.compound_degree == 2]:
        mapping = design.get(compound.condition_id)
        if not isinstance(mapping, dict):
            issues.append(f"compound missing factorial mapping: {compound.condition_id}")
            continue
        for cell in ["A0B0", "A1B0", "A0B1", "A1B1"]:
            if mapping.get(cell) not in ids:
                issues.append(f"compound {compound.condition_id} missing valid {cell}")
        if compound.condition_id.startswith("OUTPUT_"):
            if mapping.get("A0B0") != "OUTPUT_CONTROL":
                issues.append(f"output compound {compound.condition_id} uses unmatched control")
    if "INPUT_OUTPUT" in ids:
        issues.append("INPUT_OUTPUT must be absent unless matched continuation input calibration is added")
    if root != Path("."):
        required = [
            root / "campaigns" / "phase5" / "slo-calibration.json",
            root / "campaigns" / "phase5" / "repetition-justification.json",
            root / "campaigns" / "phase5" / "factorial-design.json",
            root / "campaigns" / "phase5" / "phase6-preflight.json",
        ]
        for path in required:
            if not path.exists():
                issues.append(f"missing freeze artifact {path.name}")
        slo_path = root / "campaigns" / "phase5" / "slo-calibration.json"
        if slo_path.exists():
            slo = json.loads(slo_path.read_text(encoding="utf-8"))
            if slo.get("dedicated_calibration") is not True:
                issues.append("missing dedicated SLO calibration")
            if slo.get("status") != "DEDICATED_PRECAMPAIGN_CALIBRATION_COMPLETE":
                issues.append("SLO calibration is not complete")
            if int(slo.get("baseline_repetitions", 0)) < CALIBRATION_REPETITIONS:
                issues.append("fewer than 8 valid BASELINE calibration runs")
            if int(slo.get("output_control_repetitions", 0)) < CALIBRATION_REPETITIONS:
                issues.append("fewer than 8 valid OUTPUT_CONTROL calibration runs")
            if slo.get("formal_campaign_outcomes_used") is not False:
                issues.append("formal campaign outcomes used for SLO calibration")
            if "THROUGHPUT_SLO" in (slo.get("metrics") or {}):
                issues.append("THROUGHPUT_SLO remains in counted SLO calibration")
            for name, item in (slo.get("metrics") or {}).items():
                if not isinstance(item.get("final_numeric_threshold"), (int, float)):
                    issues.append(f"missing numeric SLO threshold {name}")
            serialized = json.dumps(slo, sort_keys=True)
            if "phase4c-control" in serialized:
                issues.append("placeholder Phase-4 calibration source present")
            for metric, families in (slo.get("family_limits") or {}).items():
                for family, item in families.items():
                    if not item.get("source_run_ids") or not item.get("source_hashes"):
                        issues.append(f"missing calibration source hashes {metric}:{family}")
        for row in rows:
            cfg_path = root / row["config_path"]
            if not cfg_path.exists():
                issues.append(f"missing run config {cfg_path}")
                continue
            try:
                cfg = ExperimentConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
                cfg.validate()
                exp = cfg.runtime.parameters.get("experimental_condition") or {}
                if exp.get("run_id") != row["run_id"] or exp.get("campaign_condition_id") != row["condition_id"]:
                    issues.append(f"experimental condition mismatch {row['run_id']}")
                if int(cfg.runtime.parameters.get("warmup_request_count", 0)) != WARMUP_REQUESTS:
                    issues.append(f"warm-up unsupported {row['run_id']}")
            except Exception as exc:
                issues.append(f"invalid run config {row['run_id']}: {exc}")
        slo_defs_path = root / "campaigns" / "phase5" / "slo-definitions.json"
        if slo_defs_path.exists():
            defs = json.loads(slo_defs_path.read_text(encoding="utf-8"))
            counted = ((defs.get("compound_violation_definition") or {}).get("counted_slos") or [])
            if "THROUGHPUT_SLO" in counted or "THROUGHPUT_SLO" in defs:
                issues.append("THROUGHPUT_SLO still present as counted SLO")
    return issues


def _doc_text(freeze_sha: str, manifest_sha: str, src_sha: str, dirty: bool | None, revision: str | None) -> str:
    conds = conditions()
    pairwise = [c.condition_id for c in conds if c.compound_degree == 2]
    singles = [c.condition_id for c in conds if c.compound_degree == 1 and c.condition_id not in {"OUTPUT_LOAD_CONTROL", "OUTPUT_DOWNSTREAM_CONTROL"}]
    return f"""# Phase 5 Campaign Freeze

This document freezes the publication campaign specification. It is a plan only: no
formal campaign runs, compound campaign, CSD, RCA, memory/I/O, or cross-runtime
experiments are executed by Phase 5.

## Freeze Identity

- campaign_freeze_sha256: `{freeze_sha}`
- campaign_manifest_sha256: `{manifest_sha}`
- git_revision: `{revision}`
- git_dirty_at_freeze_generation: `{dirty}`
- source_tree_sha256_at_generation: `{src_sha}`

Publication execution requires `git_dirty=false`; the Phase 5 repository may still
contain uncommitted freeze artifacts until the user commits them.

## Runtime Environment

- llama.cpp `{RUNTIME_VERSION}`, build `{RUNTIME_BUILD}`, commit `{RUNTIME_COMMIT}`
- Darwin arm64 / Metal
- model `{MODEL_REF}`
- alias `{MODEL_ID}`
- required flags: `{' '.join(LLAMA_ARGS)}`
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
- THROUGHPUT_SLO: run-level observed throughput; violation when below the calibrated
  baseline lower threshold.

Single SLO violation means exactly one frozen SLO is violated in a measured run.
Compound SLO violation means two or more frozen SLOs are violated in the same
measured run. Request rows are observations within a run, not independent
experimental repetitions.

## Frozen Mechanisms

- M1 input/prefill pressure: medium input, observed median prompt tokens about 267.
- M2 output/decode pressure: synthetic continuation with 32 requested output tokens.
- M3 queue/load pressure: 12 rps offered load; queueing is intentional.
- M4 downstream latency: synthetic dependency configured to 100 ms.
- CPU contention: DEFERRED for cross-configuration analysis.

## Single-Mechanism Matrix

Formal single/control conditions: `{', '.join(['BASELINE', 'OUTPUT_CONTROL', 'OUTPUT_LOAD_CONTROL', 'OUTPUT_DOWNSTREAM_CONTROL', *singles])}`.

## Compound Matrix

Primary pairwise compounds: `{', '.join(pairwise)}`.

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

- repetitions per condition: `{REPETITIONS}`
- deterministic order seed: `{ORDER_SEED}`
- order rule: block by repetition, randomize all formal conditions within each block
- measured requests per run: `{REQUEST_COUNT}`
- unscored warm-up requests before each measured run: `{WARMUP_REQUESTS}`
- fixed cooldown after each run: `{COOLDOWN_SECONDS}` seconds

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

- formal conditions: `{len(conds)}`
- formal runs: `{len(conds) * REPETITIONS}`
- estimated measured requests: `{len(conds) * REPETITIONS * REQUEST_COUNT}`
- estimated warm-up requests: `{len(conds) * REPETITIONS * WARMUP_REQUESTS}`
- estimated storage guard: `{MIN_FREE_SPACE_BYTES}` bytes minimum free
- estimated campaign storage: `{ESTIMATED_STORAGE_BYTES}` bytes

## Stop/Go Criteria

Go only when the source tree is clean, disk-space guard passes, runtime/model
provenance matches this freeze, service readiness passes, and the manifest/config
hashes match. Stop on dirty source, runtime mismatch, invalid manifest, insufficient
space, service contamination, or mandatory telemetry/trace failure.
"""


def generate(root: Path = Path(".")) -> dict[str, Any]:
    docs_dir = root / "docs"
    configs_dir = root / "configs" / "phase5"
    campaigns_dir = root / "campaigns" / "phase5"
    docs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    campaigns_dir.mkdir(parents=True, exist_ok=True)

    rev = git_revision()
    dirty = git_dirty(root)
    src_sha = source_tree_sha256(root)
    conds = conditions()
    rows = campaign_rows(rev)

    config_paths: list[Path] = []
    for condition in conds:
        cfg = config_for_condition(condition, run_id=f"phase5-template-{condition_slug(condition.condition_id)}")
        path = configs_dir / config_filename(condition)
        path.write_text(cfg.canonical_json(include_hash=True) + "\n", encoding="utf-8")
        config_paths.append(path)
    run_configs_dir = configs_dir / "runs"
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    condition_by_id = {c.condition_id: c for c in conds}
    for row in rows:
        condition = condition_by_id[row["condition_id"]]
        cfg = config_for_condition(condition, repetition=int(row["repetition"]), attempt=int(row["attempt"]), run_id=row["run_id"])
        path = root / row["config_path"]
        path.write_text(cfg.canonical_json(include_hash=True) + "\n", encoding="utf-8")
        config_paths.append(path)

    manifest = {
        "campaign_id": PHASE5_ID,
        "schema_version": "sloscope.campaign.v1",
        "purpose": "publication_campaign_freeze",
        "run_order_seed": ORDER_SEED,
        "repetitions": REPETITIONS,
        "primary_attempt": 1,
        "formal_condition_count": len(conds),
        "formal_run_count": len(rows),
        "request_count_per_run": REQUEST_COUNT,
        "estimated_measured_request_count": len(rows) * REQUEST_COUNT,
        "estimated_warmup_request_count": len(rows) * WARMUP_REQUESTS,
        "warmup_request_count": WARMUP_REQUESTS,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "minimum_free_space_bytes": MIN_FREE_SPACE_BYTES,
        "estimated_storage_bytes": ESTIMATED_STORAGE_BYTES,
        "git_revision": rev,
        "git_dirty": dirty,
        "source_tree_sha256": src_sha,
        "runtime": {
            "llama_cpp_version": RUNTIME_VERSION,
            "llama_cpp_build": RUNTIME_BUILD,
            "llama_cpp_commit": RUNTIME_COMMIT,
            "os": platform.system(),
            "architecture": platform.machine(),
            "accelerator": "Metal",
            "required_args": LLAMA_ARGS,
        },
        "model": {"model_ref": MODEL_REF, "model_id": MODEL_ID, "quantization": "Q4_K_M"},
        "conditions": [asdict(c) for c in conds],
        "campaign_order": rows,
        "publication_execution_requirement": {"git_dirty": False},
    }
    write_json(campaigns_dir / "campaign-manifest.json", manifest)
    write_csv(campaigns_dir / "campaign-manifest.csv", rows)
    write_json(campaigns_dir / "mechanism-definitions.json", mechanism_definitions())
    write_json(campaigns_dir / "slo-definitions.json", slo_definitions())
    write_json(campaigns_dir / "analysis-plan.json", analysis_plan())
    write_json(campaigns_dir / "telemetry-schema.json", telemetry_schema())
    write_json(campaigns_dir / "confound-validity-rules.json", confound_rules())
    write_json(campaigns_dir / "factorial-design.json", factorial_design())
    write_json(campaigns_dir / "slo-calibration.json", load_slo_calibration_for_freeze(root))
    write_json(campaigns_dir / "repetition-justification.json", repetition_justification())
    write_json(campaigns_dir / "phase6-preflight.json", preflight_spec(src_sha))

    hash_inputs = freeze_input_paths(root)
    freeze_sha = freeze_hash(hash_inputs, root)
    manifest_sha = manifest_hash(campaigns_dir / "campaign-manifest.json")
    manifest["campaign_freeze_sha256"] = freeze_sha
    manifest["campaign_manifest_sha256"] = manifest_sha
    write_json(campaigns_dir / "campaign-manifest.json", manifest)
    manifest_sha = manifest_hash(campaigns_dir / "campaign-manifest.json")
    freeze_sha = freeze_hash(hash_inputs, root)
    manifest["campaign_freeze_sha256"] = freeze_sha
    manifest["campaign_manifest_sha256"] = manifest_sha
    write_json(campaigns_dir / "campaign-manifest.json", manifest)

    (docs_dir / "phase5-campaign-freeze.md").write_text(
        _doc_text(freeze_sha, manifest_sha, src_sha, dirty, rev),
        encoding="utf-8",
    )

    issues = validate_freeze(root)
    write_json(campaigns_dir / "freeze-validation.json", {"valid": not issues, "issues": issues})
    return {"campaign_freeze_sha256": freeze_sha, "campaign_manifest_sha256": manifest_sha, "issues": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 5 campaign freeze artifacts without executing the campaign.")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--run-slo-calibration", action="store_true", help="Run the dedicated Phase 5.2 control-only SLO calibration, not Phase 6.")
    parser.add_argument("--logs", default="logs/phase5.2-slo", help="Calibration service log directory")
    args = parser.parse_args(argv)
    if args.run_slo_calibration:
        return run_dedicated_slo_calibration(Path(args.root).resolve() / "calibrations" / "phase5.2-slo", Path(args.logs).resolve())
    result = generate(Path(args.root).resolve())
    print(_canonical_json(result))
    return 0 if not result["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
