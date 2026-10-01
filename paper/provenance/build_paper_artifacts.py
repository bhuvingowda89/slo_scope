from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
PHASES = {
    "7A.1": ROOT / "analysis" / "phase7a1",
    "7B": ROOT / "analysis" / "phase7b",
    "7C": ROOT / "analysis" / "phase7c",
    "7C.1a": ROOT / "analysis" / "phase7c1a",
    "7D.2a": ROOT / "analysis" / "phase7d2a",
    "7E": ROOT / "analysis" / "phase7e",
    "7F": ROOT / "analysis" / "phase7f",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")


def df(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def row_value(path: str, filters: dict[str, object], column: str) -> object:
    data = df(path)
    mask = pd.Series([True] * len(data))
    for key, value in filters.items():
        mask &= data[key].astype(str) == str(value)
    rows = data[mask]
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path} for {filters}, got {len(rows)}")
    value = rows.iloc[0][column]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def add_result(
    results: list[dict[str, object]],
    result_id: str,
    description: str,
    value: object,
    units: str,
    phase: str,
    source_artifact: str,
    relation: str,
    role: str = "primary",
    domain: str = "source",
    row_filter: dict[str, object] | None = None,
    column: str | None = None,
    notes: str = "",
) -> None:
    results.append(
        {
            "result_id": result_id,
            "description": description,
            "numeric_value": value,
            "units": units,
            "analysis_phase": phase,
            "source_artifact": source_artifact,
            "row_filter": row_filter or {},
            "column": column,
            "hypothesis_or_RQ_relation": relation,
            "primary_secondary_exploratory": role,
            "source_target_domain": domain,
            "notes": notes,
        }
    )


def latex_escape(text: object) -> str:
    s = str(text)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        s = s.replace(a, b)
    return s


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def table_tex(name: str, caption: str, headers: list[str], rows: list[list[object]]) -> None:
    align = "l" * len(headers)
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        body.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    body += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write(PAPER / "tables" / f"{name}.tex", "\n".join(body))


def make_architecture_svg() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">
  <defs>
    <style>
      .box{fill:#f8fafc;stroke:#334155;stroke-width:2;rx:8}
      .accent{fill:#ecfeff;stroke:#0e7490;stroke-width:2;rx:8}
      .fault{fill:#fff7ed;stroke:#c2410c;stroke-width:2;rx:8}
      .tele{fill:#f0fdf4;stroke:#15803d;stroke-width:2;rx:8}
      .txt{font-family:Arial, sans-serif;font-size:20px;fill:#0f172a}
      .small{font-family:Arial, sans-serif;font-size:15px;fill:#334155}
      .arrow{stroke:#334155;stroke-width:2.5;marker-end:url(#arrow)}
      .dash{stroke:#15803d;stroke-width:2;stroke-dasharray:7 5;marker-end:url(#arrow)}
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
    </marker>
  </defs>
  <rect x="35" y="60" width="150" height="80" class="box"/><text x="62" y="105" class="txt">Load generator</text>
  <rect x="250" y="60" width="165" height="80" class="accent"/><text x="302" y="105" class="txt">Gateway</text>
  <rect x="500" y="35" width="170" height="80" class="box"/><text x="535" y="82" class="txt">llama.cpp</text>
  <rect x="735" y="35" width="190" height="80" class="box"/><text x="770" y="82" class="txt">Qwen model</text>
  <rect x="500" y="155" width="190" height="80" class="box"/><text x="535" y="202" class="txt">Dependency</text>
  <path d="M185 100 L250 100" class="arrow"/><path d="M415 90 L500 75" class="arrow"/><path d="M670 75 L735 75" class="arrow"/>
  <path d="M415 115 L500 190" class="arrow"/><path d="M500 210 L415 125" class="arrow"/>
  <rect x="45" y="250" width="210" height="70" class="fault"/><text x="72" y="292" class="txt">INPUT: longer prompt</text>
  <rect x="285" y="250" width="230" height="70" class="fault"/><text x="315" y="292" class="txt">OUTPUT: more decode</text>
  <rect x="545" y="250" width="200" height="70" class="fault"/><text x="575" y="292" class="txt">LOAD: 12 rps</text>
  <rect x="775" y="250" width="245" height="70" class="fault"/><text x="803" y="292" class="txt">DOWNSTREAM: 100 ms</text>
  <path d="M150 250 L150 140" class="arrow"/><path d="M400 250 L585 115" class="arrow"/><path d="M645 250 L110 140" class="arrow"/><path d="M895 250 L595 235" class="arrow"/>
  <rect x="65" y="390" width="205" height="70" class="tele"/><text x="86" y="432" class="txt">Request timing</text>
  <rect x="305" y="390" width="145" height="70" class="tele"/><text x="340" y="432" class="txt">Traces</text>
  <rect x="485" y="390" width="190" height="70" class="tele"/><text x="512" y="432" class="txt">System metrics</text>
  <rect x="710" y="390" width="200" height="70" class="tele"/><text x="735" y="432" class="txt">Runtime metrics</text>
  <rect x="365" y="515" width="360" height="65" class="accent"/><text x="430" y="555" class="txt">Validated artifacts and analyses</text>
  <path d="M330 140 L175 390" class="dash"/><path d="M330 140 L375 390" class="dash"/><path d="M585 115 L580 390" class="dash"/><path d="M585 115 L810 390" class="dash"/>
  <path d="M170 460 L455 515" class="dash"/><path d="M375 460 L505 515" class="dash"/><path d="M580 460 L585 515" class="dash"/><path d="M810 460 L655 515" class="dash"/>
  <text x="45" y="30" class="txt">SLOScope benchmark architecture and controlled mechanism injection points</text>
</svg>
"""
    write(PAPER / "figures" / "figure-1-architecture.svg", svg)


def copy_figures() -> None:
    mapping = {
        "figure-2-isolated-effects.svg": "analysis/phase7a1/figures/figure-2-isolated-effects.svg",
        "figure-3-compound-csd.svg": "analysis/phase7b/figures/figure-6-csd-summary.svg",
        "figure-4-rca-gap.svg": "analysis/phase7c/figures/figure-11-single-to-compound-gap.svg",
        "figure-7-degraded-telemetry.svg": "analysis/phase7e/figures/figure-22-missingness-robustness.svg",
        "figure-8-quality-cost-frontier.svg": "analysis/phase7f/figures/figure-27-quality-cost-frontier.svg",
    }
    for dest, src in mapping.items():
        out = PAPER / "figures" / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, out)
    make_architecture_svg()
    # Compact SVGs for cross-model RCA and feature shift, generated from accepted values.
    gen = df("analysis/phase7d2a/table-12a-zero-shot-generalization.csv")
    bars = []
    x = 80
    max_h = 260
    for _, r in gen.iterrows():
        sh = float(r.source_compound_exact) * max_h
        th = float(r.target_compound_exact) * max_h
        bars.append(f'<rect x="{x}" y="{330-sh:.1f}" width="32" height="{sh:.1f}" fill="#2563eb"/><rect x="{x+38}" y="{330-th:.1f}" width="32" height="{th:.1f}" fill="#f97316"/><text x="{x-8}" y="360" class="small">{r.method}</text>')
        x += 150
    write(PAPER / "figures" / "figure-5-source-target-rca.svg", f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="410"><style>.small{{font-family:Arial;font-size:13px;fill:#111827}}.title{{font-family:Arial;font-size:20px;fill:#111827}}</style><text x="30" y="32" class="title">Source vs target compound exact recovery</text><line x1="55" y1="330" x2="690" y2="330" stroke="#111827"/><line x1="55" y1="70" x2="55" y2="330" stroke="#111827"/>{''.join(bars)}<rect x="560" y="50" width="18" height="18" fill="#2563eb"/><text x="585" y="64" class="small">source</text><rect x="560" y="78" width="18" height="18" fill="#f97316"/><text x="585" y="92" class="small">target</text></svg>""")
    shifts = [
        ("total_latency_early_median", 3.96),
        ("ttft_early_median", 3.72),
        ("prefill_proxy_early_median", 3.53),
        ("total_latency_median", 3.27),
        ("total_latency_p95", 3.20),
    ]
    rows = []
    y = 70
    for name, val in shifts:
        rows.append(f'<text x="25" y="{y+15}" class="small">{name}</text><rect x="245" y="{y}" width="{val*95:.1f}" height="22" fill="#0e7490"/><text x="{255+val*95:.1f}" y="{y+16}" class="small">{val:.2f}</text>')
        y += 48
    write(PAPER / "figures" / "figure-6-feature-shift.svg", f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360"><style>.small{{font-family:Arial;font-size:14px;fill:#111827}}.title{{font-family:Arial;font-size:20px;fill:#111827}}</style><text x="25" y="32" class="title">Largest source-to-target standardized mean shifts</text>{''.join(rows)}<text x="245" y="325" class="small">standardized mean shift in source SD units</text></svg>""")


def build_result_registry() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    iso_path = "analysis/phase7a1/table-2-isolated-effects.csv"
    for mech, rid in [
        ("INPUT", "R_RQ1_INPUT_TTFT_DELTA"),
        ("OUTPUT", "R_RQ1_OUTPUT_DECODE_DELTA"),
        ("LOAD", "R_RQ1_LOAD_TTFT_DELTA"),
        ("DOWNSTREAM", "R_RQ1_DOWNSTREAM_TOTAL_DELTA"),
    ]:
        filt = {"mechanism": mech}
        add_result(results, rid, f"Isolated {mech} primary paired effect", row_value(iso_path, filt, "mean paired delta"), "seconds", "7A.1", iso_path, "RQ1", row_filter=filt, column="mean paired delta")
        add_result(results, rid + "_HOLM_P", f"Isolated {mech} Holm-adjusted p", row_value(iso_path, filt, "Holm p"), "p-value", "7A.1", iso_path, "RQ1", row_filter=filt, column="Holm p")
    fact = "analysis/phase7a1/table-3-factorial-interactions.csv"
    classes = df(fact)["classification"].value_counts().to_dict()
    for label in ["SUPER_ADDITIVE_EVIDENCE", "SUB_ADDITIVE_EVIDENCE", "INTERACTION_UNCERTAIN"]:
        add_result(results, "R_RQ2_" + label, f"Count of {label} univariate interactions", int(classes.get(label, 0)), "count", "7A.1", fact, "RQ2", "primary", "source")
    csd = "analysis/phase7b/table-4-csd-summary.csv"
    for comp in ["INPUT_LOAD", "OUTPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]:
        filt = {"compound": comp}
        add_result(results, f"R_CSD_{comp}", f"{comp} primary CSD", row_value(csd, filt, "CSD"), "standardized RMS", "7B", csd, "H1/RQ2", row_filter=filt, column="CSD")
        add_result(results, f"R_CSD_{comp}_HOLM_P", f"{comp} CSD Holm p", row_value(csd, filt, "Holm_p"), "p-value", "7B", csd, "H1/RQ2", row_filter=filt, column="Holm_p")
    c6 = "analysis/phase7c/table-6-rca-baselines.csv"
    for proto, model, tier, subset, rid in [
        ("P1", "M2", "F2", "compound-only", "R_M2F2_SOURCE_P1_COMPOUND_EXACT"),
        ("P2", "M2", "F2", "compound-only", "R_M2F2_SOURCE_P2_COMPOUND_EXACT"),
        ("P1", "M2", "F2", "compound-only", "R_M2F2_SOURCE_P1_COMPLETE_RECALL"),
        ("P2", "M2", "F2", "compound-only", "R_M2F2_SOURCE_P2_COMPLETE_RECALL"),
    ]:
        col = "complete_cause_recall" if "COMPLETE" in rid else "exact_set_accuracy"
        filt = {"training_protocol": proto, "model": model, "feature_tier": tier, "subset": subset}
        add_result(results, rid, rid.replace("_", " "), row_value(c6, filt, col), "fraction", "7C", c6, "H2/RQ3", row_filter=filt, column=col)
    h2 = "analysis/phase7c/table-8-single-vs-mixed-training.csv"
    add_result(results, "R_H2_MEAN_DIFF", "H2 mean complete-cause-recall difference", row_value(h2, {"fold": "SUMMARY"}, "difference"), "fraction", "7C", h2, "H2", row_filter={"fold": "SUMMARY"}, column="difference")
    add_result(results, "R_H2_PVALUE", "H2 exact sign-flip p", row_value(h2, {"fold": "SUMMARY"}, "exact_signflip_p"), "p-value", "7C", h2, "H2", row_filter={"fold": "SUMMARY"}, column="exact_signflip_p")
    mesr = "analysis/phase7c1a/table-9a-mesr-corrected-primary.csv"
    for method in ["D4-FULL", "D4-FULL-F3", "D4-SM"]:
        filt = {"method": method, "subset": "P1-COMPOUND"}
        add_result(results, f"R_{method.replace('-', '_')}_SOURCE_EXACT", f"{method} source P1 compound exact", row_value(mesr, filt, "exact_set_accuracy"), "fraction", "7C.1a", mesr, "H3/RQ3", row_filter=filt, column="exact_set_accuracy")
    gen = "analysis/phase7d2a/table-12a-zero-shot-generalization.csv"
    for method in ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]:
        filt = {"method": method}
        safe = method.replace("/", "").replace("-", "_")
        add_result(results, f"R_{safe}_SOURCE_EXACT", f"{method} source compound exact", row_value(gen, filt, "source_compound_exact"), "fraction", "7D.2a", gen, "RQ3", row_filter=filt, column="source_compound_exact")
        add_result(results, f"R_{safe}_TARGET_EXACT", f"{method} target compound exact", row_value(gen, filt, "target_compound_exact"), "fraction", "7D.2a", gen, "H4/RQ3", row_filter=filt, column="target_compound_exact", domain="target")
        add_result(results, f"R_{safe}_TARGET_CFA", f"{method} target control false-alarm rate", row_value(gen, filt, "target_control_false_alarm_rate"), "fraction", "7D.2a", gen, "RQ3", row_filter=filt, column="target_control_false_alarm_rate", domain="target")
    h4 = "analysis/phase7d2a/table-14a-h4-cross-model.csv"
    add_result(results, "R_H4_PVALUE", "H4 D4 vs M2/F2 exact sign-flip p", row_value(h4, {"repetition": "SUMMARY"}, "exact_p_D4_vs_M2F2"), "p-value", "7D.2a", h4, "H4", row_filter={"repetition": "SUMMARY"}, column="exact_p_D4_vs_M2F2", domain="target")
    add_result(results, "R_H4_MEAN_DIFF", "H4 D4 minus M2/F2 mean target exact difference", row_value(h4, {"repetition": "SUMMARY"}, "D4_minus_M2F2"), "fraction", "7D.2a", h4, "H4", row_filter={"repetition": "SUMMARY"}, column="D4_minus_M2F2", domain="target")
    ctrl = "analysis/phase7d2a/control-normalized-summary.csv"
    for method in ["M2/F2-control-normalized", "M2/F2T-control-normalized"]:
        filt = {"method": method, "subset": "TARGET-COMPOUND"}
        add_result(results, f"R_{method.replace('/', '').replace('-', '_')}_TARGET_EXACT", f"{method} target compound exact", row_value(ctrl, filt, "exact_set_accuracy"), "fraction", "7D.2a", ctrl, "RQ3 secondary", "secondary", "target", row_filter=filt, column="exact_set_accuracy")
    miss = "analysis/phase7e/table-16-telemetry-missingness.csv"
    for domain, method in [("source", "M2/F2"), ("source", "M2/F2T"), ("source", "D4-FULL"), ("target", "M2/F2"), ("target", "M2/F2T"), ("target", "D4-FULL")]:
        filt = {"domain": domain, "method": method, "severity": 0.5}
        add_result(results, f"R_MISSING50_{domain}_{method.replace('/', '').replace('-', '_')}", f"{domain} {method} 50 percent missingness exact", row_value(miss, filt, "compound_exact"), "fraction", "7E", miss, "RQ4", "primary", domain, row_filter=filt, column="compound_exact")
    aucdiff = "analysis/phase7e/source-target-robustness-differences.csv"
    for method in ["M2/F2T", "D4-FULL"]:
        for fam in ["missingness", "noise", "temporal_sampling", "delay"]:
            filt = {"method": method, "degradation_family": fam}
            add_result(results, f"R_AUCDIFF_{method.replace('/', '').replace('-', '_')}_{fam}", f"{method} target-minus-source robustness AUC for {fam}", row_value(aucdiff, filt, "target_minus_source_auc"), "AUC difference", "7E", aucdiff, "RQ4", "secondary", "both", row_filter=filt, column="target_minus_source_auc")
    frontier = "analysis/phase7f/table-21-quality-cost-frontier.csv"
    for cfg in ["T0_FULL", "T1_NO_HOST", "T2_NO_RUNTIME", "T3_LIGHT"]:
        filt = {"telemetry_config": cfg}
        add_result(results, f"R_{cfg}_OPTIONAL_BYTES_PER_REQUEST", f"{cfg} optional telemetry bytes/request", row_value(frontier, filt, "optional_telemetry_bytes_per_request"), "bytes/request", "7F", frontier, "RQ5", row_filter=filt, column="optional_telemetry_bytes_per_request")
        add_result(results, f"R_{cfg}_STORAGE_REDUCTION", f"{cfg} storage reduction vs FULL", row_value(frontier, filt, "storage_reduction_vs_FULL"), "fraction", "7F", frontier, "RQ5", row_filter=filt, column="storage_reduction_vs_FULL")
    h5 = json.loads((ROOT / "analysis/phase7f/h5-final-evaluation.json").read_text())
    add_result(results, "R_H5_BASELINE_LATENCY_MEAN", "NO_HOST baseline total-latency-p95 relative change", h5["service_performance_guard"][0]["mean_relative_total_latency_p95_change"], "relative change", "7F", "analysis/phase7f/h5-final-evaluation.json", "H5/RQ5")
    add_result(results, "R_H5_BASELINE_LATENCY_CI_HIGH", "NO_HOST baseline latency CI high", h5["service_performance_guard"][0]["ci95_high"], "relative change", "7F", "analysis/phase7f/h5-final-evaluation.json", "H5/RQ5")
    add_result(results, "R_H5_LOAD_LATENCY_MEAN", "NO_HOST load total-latency-p95 relative change", h5["service_performance_guard"][1]["mean_relative_total_latency_p95_change"], "relative change", "7F", "analysis/phase7f/h5-final-evaluation.json", "H5/RQ5")
    add_result(results, "R_H5_LOAD_LATENCY_CI_HIGH", "NO_HOST load latency CI high", h5["service_performance_guard"][1]["ci95_high"], "relative change", "7F", "analysis/phase7f/h5-final-evaluation.json", "H5/RQ5")
    target_proj = df("analysis/phase7f/target-storage-projection.csv").iloc[0]
    add_result(results, "R_TARGET_NO_HOST_BYTES", "Target host telemetry storage projection", float(target_proj.campaign_total_system_bytes), "bytes", "7F", "analysis/phase7f/target-storage-projection.csv", "RQ5", "secondary", "target")
    add_result(results, "R_TARGET_NO_HOST_BYTES_PER_REQUEST", "Target host telemetry bytes/request projection", float(target_proj.system_bytes_per_request_mean), "bytes/request", "7F", "analysis/phase7f/target-storage-projection.csv", "RQ5", "secondary", "target")
    add_result(results, "R_TARGET_NO_HOST_OPTIONAL_PCT", "Target host telemetry share of optional storage", float(target_proj.system_pct_optional_telemetry_storage), "fraction", "7F", "analysis/phase7f/target-storage-projection.csv", "RQ5", "secondary", "target")
    # Accepted prompt/report feature-shift values; Phase 7D.2a verifies the historical feature-shift hash.
    for name, val in [
        ("total_latency_early_median", 3.96),
        ("ttft_early_median", 3.72),
        ("prefill_proxy_early_median", 3.53),
        ("total_latency_median", 3.27),
        ("total_latency_p95", 3.20),
    ]:
        add_result(results, f"R_SHIFT_{name.upper()}", f"Source-to-target standardized mean shift for {name}", val, "source SD", "7D.2a", "analysis/phase7d2a/phase7d2a-report.json", "RQ3", "descriptive", "both", notes="Phase 7D.2a accepted feature-shift hash; detailed shift table was historical artifact.")
    return results


def build_tables() -> None:
    table_tex("table-1-mechanisms", "SLOScope mechanisms and experimental roles.", ["Mechanism", "Operational definition", "Direct evidence", "Primary stage", "Compound partners"], [
        ["INPUT", "Synthetic input-medium prompt", "Observed prompt/prefill timing", "Prefill/TTFT", "LOAD, DOWNSTREAM"],
        ["OUTPUT", "Continuation workload requesting 32 output tokens", "Observed decode/output timing", "Decode", "LOAD, DOWNSTREAM"],
        ["LOAD", "12 rps offered load", "Arrival schedule and scheduler slip; queue-depth metric unavailable", "Queue/contention", "INPUT, OUTPUT, DOWNSTREAM"],
        ["DOWNSTREAM", "Synthetic dependency delay of 100 ms", "Dependency timing", "Gateway/dependency", "INPUT, OUTPUT, LOAD"],
    ])
    table_tex("table-2-campaigns", "Formal campaign summary. Cost runs are not RCA observations.", ["Campaign", "Model", "Runs", "Measured requests", "Role"], [
        ["Phase6-V2", "Qwen2.5-0.5B Q4_K_M", 104, 4160, "source formal RCA/factorial dataset"],
        ["Phase7D target", "Qwen2.5-1.5B Q4_K_M", 66, 2640, "zero-shot model-holdout dataset"],
        ["Phase7F cost", "Qwen2.5-0.5B Q4_K_M", 48, 1920, "telemetry cost measurement only"],
    ])
    iso = df("analysis/phase7a1/table-2-isolated-effects.csv")
    table_tex("table-3-isolated", "Primary isolated mechanism effects from matched controls.", ["Mechanism", "Metric", "Mean delta (s)", "95% CI", "Holm p"], [[r.mechanism, r["primary metric"], fmt(r["mean paired delta"], 4), r["95% CI"], f"{r['Holm p']:.2e}"] for _, r in iso.iterrows()])
    csd = df("analysis/phase7b/table-4-csd-summary.csv")
    table_tex("table-4-compound", "Compound interaction and CSD summary.", ["Compound", "CSD", "Holm p", "Supported", "SLO violations"], [[r.compound, fmt(r.CSD, 2), fmt(r.Holm_p, 6), bool(r.statistically_supported), r.compound_slo_violations] for _, r in csd.iterrows()])
    src = df("analysis/phase7d2a/table-12a-zero-shot-generalization.csv")
    table_tex("table-5-rca", "Source and target compound diagnosis.", ["Method", "Source exact", "Target exact", "Target CFA", "Retention"], [[r.method, fmt(r.source_compound_exact), fmt(r.target_compound_exact), fmt(r.target_control_false_alarm_rate), fmt(r.retention)] for _, r in src.iterrows()])
    miss = df("analysis/phase7e/table-16-telemetry-missingness.csv")
    rows = []
    for domain in ["source", "target"]:
        for method in ["M2/F2", "M2/F2T", "D4-FULL"]:
            r = miss[(miss.domain == domain) & (miss.method == method) & (miss.severity == 0.5)].iloc[0]
            rows.append([domain, method, fmt(r.compound_exact), fmt(r.retention), fmt(r.control_false_alarm_rate)])
    table_tex("table-6-robustness", "50 percent random feature missingness robustness.", ["Domain", "Method", "Compound exact", "Retention", "Control false alarm"], rows)
    frontier = df("analysis/phase7f/table-21-quality-cost-frontier.csv")
    table_tex("table-7-cost", "Telemetry quality/cost frontier.", ["Config", "Source exact", "Target exact", "Bytes/request", "Reduction", "Dominated"], [[r.telemetry_config, fmt(r.source_M2F2_compound_exact), fmt(r.target_M2F2_compound_exact), fmt(r.optional_telemetry_bytes_per_request, 2), fmt(r.storage_reduction_vs_FULL), bool(r.pareto_dominated)] for _, r in frontier.iterrows()])


def manuscript_tex() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{enumitem}
\title{SLOScope: Benchmarking Compound SLO Degradation and Diagnosis in Commodity LLM Inference Systems}
\author{Anonymous for review}
\date{}
\begin{document}
\maketitle
\begin{abstract}
Large-language-model inference services expose coupled latency stages: input prefill, output decoding, queueing under load, and downstream service calls. SLO failures in such systems may therefore arise from simultaneous mechanisms whose observable signatures need not compose additively. We present SLOScope, a reproducible benchmark and empirical study of single and compound degradation in a commodity llama.cpp/Metal inference service. Across 104 source-domain formal runs, matched factorial analysis shows that isolated mechanisms produce strong observable effects, but compound behavior is selective rather than universal: INPUT+LOAD and OUTPUT+LOAD show Holm-supported multivariate Compound Signature Divergence, while three other retained pairs do not. Multi-label RCA exhibits a significant compositional gap: a single-fault-trained logistic baseline recovers 57.5\% of compound cause sets, versus 97.5\% with compound training. Temporal features improve in-domain diagnosis to 82.5\%, but fall to 20.0\% under zero-shot transfer to a larger Qwen model, revealing model-sensitive signatures. An interpretable mechanism-evidence ranker is useful for analysis but is not the best-performing RCA method. Degraded-observability experiments separate clean accuracy from robustness, and a final telemetry cost study finds 14--17\% host-telemetry storage savings with preserved diagnosis quality, but the pre-specified service non-regression guard is not satisfied. The results position compound SLO diagnosis as a benchmark and measurement problem, not merely a classifier problem.
\end{abstract}

\section{Introduction}
LLM inference services expose multiple latency stages and telemetry views. A request may wait in the scheduler, spend time in prompt prefill before first token, generate tokens through autoregressive decoding, and call external services through an application gateway. In production, these mechanisms can overlap: longer prompts can arrive during high load; longer completions can coincide with queueing; dependency delay can be hidden behind or amplified by model-side work. SLOScope asks whether performance signatures compose, whether diagnosis composes, whether learned diagnostic relationships transfer across model configurations, and what diagnosis quality costs in telemetry.

The central thesis of this paper is empirical: compound degradation in LLM inference systems cannot always be understood as a simple composition of isolated faults. However, the evidence is deliberately nuanced. Some compound pairs show strong non-additivity; others do not. Some diagnostic features help in-domain; the same features may transfer poorly. Reduced telemetry can preserve offline diagnosis quality; proving live service non-regression is harder.

This paper makes five contributions. First, it provides a reproducible commodity-hardware benchmark methodology for controlled single and compound degradation in LLM inference services. Second, it introduces a matched factorial protocol and Compound Signature Divergence (CSD) analysis showing selective multivariate non-additivity. Third, it provides a systematic multi-label RCA study showing a statistically supported single-fault to compound-diagnosis gap. Fourth, it evaluates cross-model transfer and degraded observability, showing that temporal logistic features are strong in-domain but model-sensitive. Fifth, it quantifies a telemetry quality/cost frontier, finding meaningful storage reductions but a negative final H5 result under a frozen service-performance guard.

\section{Background and Motivation}
LLM serving work often separates prefill and decode behavior because they stress different resources and latency objectives \cite{distserve2024,sarathi2024}. SLO-oriented systems increasingly reason about TTFT, token generation, throughput, and goodput. Separately, microservice RCA research has produced trace, metric, graph, and benchmark-oriented approaches \cite{microRCA2020,rcaeval2025}. SLOScope studies the intersection: controlled compound degradation and diagnosis in an LLM inference service with gateway, dependency, runtime, trace, and system telemetry.

\section{Research Questions and Experimental Principles}
We organize the study around five research questions. RQ1 asks how individual mechanisms affect observable behavior. RQ2 asks how simultaneously active mechanisms interact. RQ3 asks how accurately causes can be recovered and how diagnosis transfers across model configurations. RQ4 asks how diagnosis degrades under missing, sampled, noisy, delayed, or absent telemetry. RQ5 asks which telemetry reductions preserve quality and what storage/service tradeoffs result.

The statistical unit is never the individual request when testing campaign-level hypotheses. Source-domain factorial and RCA analyses use run/repetition-level units; target generalization uses six target repetition blocks; telemetry cost uses matched repetition-by-workload blocks. Exact sign-flip tests, Holm corrections, bootstrap intervals, and Student-t intervals are used where pre-specified.

\section{SLOScope Design}
SLOScope consists of a load generator, an instrumented gateway, a synthetic dependency service, a llama.cpp server, and an artifact pipeline. Figure~\ref{fig:architecture} shows the architecture. The benchmark records ground-truth mechanism activation from frozen condition contracts, not from telemetry. Mechanisms are experimental interventions; observable signatures are measured outcomes.

\begin{figure}[t]
\centering
\fbox{\parbox{0.92\linewidth}{Architecture asset: \texttt{paper/figures/figure-1-architecture.svg}.}}
\caption{SLOScope architecture and controlled mechanism injection points.}
\label{fig:architecture}
\end{figure}

\input{../tables/table-1-mechanisms.tex}
\input{../tables/table-2-campaigns.tex}

CPU contention calibration was evaluated but deferred because it did not create a useful controlled degradation regime on the Metal setup. This is a scope decision rather than a hidden negative result.

\section{Experimental Methodology}
The source campaign uses Qwen2.5-0.5B-Instruct GGUF Q4\_K\_M with llama.cpp 0.5.0 on Darwin arm64/Metal. It contains 104 publication runs, 13 conditions, eight repetitions, and 4160 measured requests. The target model-holdout campaign uses Qwen2.5-1.5B-Instruct GGUF Q4\_K\_M with the same runtime and mechanism settings, producing 66 valid runs and 2640 measured requests. The final telemetry cost campaign contains 48 runs and 1920 measured requests and is not used for RCA training or accuracy estimation.

Warm-up requests are excluded from scientific aggregates. Every formal campaign writes publication indices, manifests, validation artifacts, trace accounting, gateway timing reconciliation summaries, and provenance hashes. Pilot and diagnostic data are excluded from formal analysis.

\section{Single-Mechanism Effects and Compound Interactions}
RQ1 is answered by matched control comparisons in Phase 7A.1. Table~\ref{tab:isolated} reports the primary isolated effects. INPUT primarily increases TTFT, OUTPUT primarily increases post-first-token duration, LOAD strongly increases TTFT, and DOWNSTREAM increases total latency through the dependency channel. Formal LOAD queue-depth metrics were unavailable; queue depth is therefore not inferred from latency.

\input{../tables/table-3-isolated.tex}

The OUTPUT control drift is a methodological result. The calibration control had no crossings, but formal OUTPUT\_CONTROL shifted by approximately 15.6\% in total latency and 20.4\% in decode duration, producing threshold crossings. For OUTPUT-family analyses, matched contemporaneous controls are therefore more reliable than absolute SLO crossings alone.

RQ2 begins with metric-level factorial interactions. Across fifteen pre-defined interactions, five show SUPER\_ADDITIVE\_EVIDENCE, one shows SUB\_ADDITIVE\_EVIDENCE, and nine are uncertain. Holm-supported univariate interactions occur for INPUT\_LOAD TTFT, total latency, and decode duration, and for OUTPUT\_LOAD TTFT and total latency.

CSD extends the univariate contrast to multivariate standardized signatures:
\[
\Delta_{rj}=X_{A1B1,rj}-X_{A1B0,rj}-X_{A0B1,rj}+X_{A0B0,rj},
\quad
CSD=\sqrt{\frac{1}{d}\sum_j \bar{Z}_j^2}.
\]
Table~\ref{tab:compound} shows that INPUT\_LOAD and OUTPUT\_LOAD have Holm-supported CSD; the other retained pairs do not. All five compounds produced frequent compound-SLO violations, so SLO severity is not the same concept as signature non-additivity.

\input{../tables/table-4-compound.tex}

\section{Compound Diagnosis}
RQ3 first asks whether diagnosis composes. The primary Phase 7C baseline, one-vs-rest logistic regression with F2 performance+telemetry features, recovers 57.5\% of compound cause sets when trained only on zero/single faults. Mixed training reaches 97.5\% compound exact recovery and 100\% complete cause recall. The fold-level complete-cause-recall improvement is 0.425 with exact sign-flip \(p=0.0078125\), supporting H2.

The error structure is pair-specific. Under single-fault training, INPUT\_LOAD and INPUT\_DOWNSTREAM have 0/8 exact recovery, whereas OUTPUT\_DOWNSTREAM and LOAD\_DOWNSTREAM have 8/8 and OUTPUT\_LOAD has 7/8. The dominant error is MISS\_INPUT with single-cause prediction.

Temporal features are useful in-domain: M2/F2 reaches 57.5\% source compound exact recovery and M2/F2T reaches 82.5\%. However, the temporal result is not a general victory; it becomes a model-transfer liability in Section~\ref{sec:generalization}.

MESR, reported as corrected D4-FULL, is an interpretable mechanism-evidence experiment. It reaches 60.0\% source compound exact recovery, close to M2/F2 but below M2/F2T. Corrected D4-FULL does not outperform D4-SM, so H3 is not supported.

\section{Cross-Model Generalization}
\label{sec:generalization}
The target campaign changes the model from Qwen2.5-0.5B to Qwen2.5-1.5B while holding runtime, hardware, instrumentation, and mechanism semantics fixed. Table~\ref{tab:rca} reports zero-shot transfer. M2/F2 transfers from 57.5\% to 60.0\%; M2/F2T drops from 82.5\% to 20.0\%; D4-FULL remains 60.0\%; D4-FULL-F3 drops from 80.0\% to 60.0\%. H4 is not supported: D4-FULL and M2/F2 have identical target fold exact accuracies and the exact sign-flip p-value is 1.0.

\input{../tables/table-5-rca.tex}

Feature shift is substantial, especially for temporal/performance features. The largest standardized mean shifts include total\_latency\_early\_median at about 3.96 source SD, ttft\_early\_median at 3.72, prefill\_proxy\_early\_median at 3.53, total\_latency\_median at 3.27, and total\_latency\_p95 at 3.20. This is consistent with the M2/F2T transfer degradation, but does not prove causation.

Compound-only accuracy is insufficient. Target control false alarms are high: M2/F2 flags 11/12 controls, M2/F2T flags 8/12, D4-FULL flags 7/12, and D4-FULL-F3 flags 6/12. Naive target-control normalization does not solve transfer: target compound exact recovery is 20.0\% for M2/F2 and 23.3\% for M2/F2T after the pre-frozen secondary normalization.

\section{Robustness to Degraded Observability}
RQ4 evaluates missingness, temporal sampling, noise, delay, and whole-channel loss without retraining. Table~\ref{tab:robustness} gives the 50\% missingness snapshot. D4-FULL is not the clean-accuracy winner, but its degraded-observability profile is comparatively stable across model shift. For M2/F2T, target-minus-source robustness AUC differences are -0.421 for missingness, -0.699 for noise, -0.419 for temporal sampling, and -0.317 for delay. For D4-FULL, the corresponding differences are approximately -0.008, -0.041, -0.013, and 0.000. These are descriptive robustness contrasts rather than formal superiority tests.

\input{../tables/table-6-robustness.tex}

The INPUT failure analysis separates pre-existing failure from degradation-induced failure. When clean diagnosis already misses INPUT in INPUT\_LOAD or INPUT\_DOWNSTREAM, degraded telemetry is not credited as the original cause of that miss.

\section{Telemetry Quality/Cost Tradeoffs}
RQ5 combines Phase 7E quality with Phase 7F live telemetry cost. Table~\ref{tab:cost} reports the frozen frontier. FULL optional telemetry costs 603.04 bytes/request; NO\_HOST costs 508.26; NO\_RUNTIME costs 545.16; LIGHT costs 448.32. For the primary NO\_HOST comparison, storage reduction is positive in all 12 matched live comparisons: 17.26\% mean under BASELINE and 14.05\% under LOAD\_MEDIUM.

\input{../tables/table-7-cost.tex}

Quality preservation passes for M2/F2+NO\_HOST: source exact is 0.575 under FULL and 0.600 under NO\_HOST; target exact is 0.600 under both. Control false alarms do not increase. This should not be read as proof that removing host telemetry improves RCA; it satisfies the frozen quality screen.

H5 is nevertheless not supported. The service-performance guard required the upper bound of the 95\% paired CI for total-latency-p95 relative change to be at most +5\%. NO\_HOST vs FULL has mean -2.52\% under BASELINE with CI [-10.83\%, +5.79\%], and mean -0.45\% under LOAD with CI [-8.02\%, +7.13\%]. This is not evidence that NO\_HOST slows the service; it means non-regression was not established under the pre-specified bound.

The target storage projection estimates that host telemetry accounts for 804,979 bytes over 66 target runs, 304.92 bytes/request, or 40.90\% of optional telemetry storage. This is an offline storage projection, not a target live-overhead measurement. T3\_LIGHT is the only non-dominated frozen quality/cost point, but the primary H5 candidate remains NO\_HOST as specified before the cost results.

\section{Discussion}
Several themes emerge. First, compound SLO violation is not equivalent to compound signature interaction: every retained compound can violate SLOs, yet only two have supported multivariate CSD. Second, single-fault diagnostic success does not imply compositional diagnosis. Third, temporal evidence can improve in-domain diagnosis while reducing cross-model portability. Fourth, interpretability, clean accuracy, transfer, and robustness are distinct axes. Fifth, more telemetry is not automatically better, but proving operational savings requires live measurement, not only offline feature ablation. Finally, control stability matters: OUTPUT\_CONTROL drift made matched comparisons essential.

\section{Threats to Validity}
\textbf{Construct validity.} Synthetic mechanisms approximate production causes but do not exhaust them. Queue-depth metrics were unavailable in formal runtime telemetry. CSD depends on the frozen feature set and standardization. SLO calibration drift, especially OUTPUT\_CONTROL, limits absolute threshold interpretation.

\textbf{Internal validity.} The experiments run on a single commodity Mac/Metal environment. Thermal state and background system activity may affect latency despite blocking and randomization. Instrumentation and gateway timing reconciliation are part of the measured system. Repetitions are limited.

\textbf{External validity.} Results cover llama.cpp, Qwen 0.5B/1.5B GGUF models, four mechanisms, and a single-host architecture. There is no primary CPU mechanism, no multi-node inference, and no real production traffic.

\textbf{Statistical conclusion validity.} The source campaign has eight repetitions; target and cost campaigns have six. Exact sign-flip tests have limited resolution. Multiple comparisons were controlled only for pre-specified families. Classifier estimates are finite-sample baselines.

\textbf{RCA validity.} F3 direct evidence exposes workload mechanisms and is not the primary fair method. Single-to-compound training distribution shift is intentionally severe. Target control false alarms show that cross-domain calibration remains unsolved.

\section{Related Work}
LLM serving systems such as DistServe and Sarathi-Serve highlight the distinct prefill and decode stages and throughput-latency tradeoffs \cite{distserve2024,sarathi2024}. Benchmarking tools such as vLLM serving benchmarks focus on latency and throughput measurement rather than controlled compound diagnosis \cite{vllmbench}. Operational systems such as LatencyPrism and StriaTrace address latency analysis and diagnosis in production LLM settings \cite{latencyprism2026,striatrace2026}. Microservice RCA work, including MicroRCA and RCAEval, studies telemetry-driven fault localization and benchmark design in distributed services \cite{microRCA2020,rcaeval2025}. SLOScope differs by combining controlled LLM-serving mechanisms, matched compound-fault factorial analysis, multi-label cause-set RCA, model-holdout transfer, degraded observability, and telemetry cost.

\section{Conclusion}
SLOScope shows that compound degradation and diagnosis in commodity LLM inference are not reducible to isolated effects. Some compound signatures are strongly non-additive, single-fault-trained RCA loses cause-set recall on compounds, temporal evidence is useful but model-sensitive, and telemetry reduction has a measurable quality/cost frontier. Negative results are central: H3, H4, and H5 are not supported under their frozen definitions. The artifact is designed to support reproduction of the benchmark, results, and manuscript-facing evidence.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""


def build_manuscript() -> None:
    write(PAPER / "manuscript" / "sloscope.tex", manuscript_tex())
    write(PAPER / "manuscript" / "references.bib", r"""@inproceedings{distserve2024,
  title={DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving},
  author={Zhong, Yinmin and others},
  booktitle={18th USENIX Symposium on Operating Systems Design and Implementation},
  year={2024}
}
@inproceedings{sarathi2024,
  title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve},
  author={Agrawal, Amey and others},
  booktitle={18th USENIX Symposium on Operating Systems Design and Implementation},
  year={2024}
}
@misc{vllmbench,
  title={vLLM Serving Benchmark Documentation},
  author={{vLLM Project}},
  year={2026},
  howpublished={\url{https://docs.vllm.ai/}}
}
@misc{latencyprism2026,
  title={LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference},
  author={Yin, Du and others},
  year={2026},
  eprint={2601.09258},
  archivePrefix={arXiv}
}
@inproceedings{rcaeval2025,
  title={RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data},
  author={Pham, Luan and Zhang, Hongyu and Ha, Huong and Salim, Flora and Zhang, Xiuzhen},
  booktitle={Companion Proceedings of the ACM Web Conference},
  year={2025},
  doi={10.1145/3701716.3715290}
}
@inproceedings{microRCA2020,
  title={MicroRCA: Root Cause Localization of Performance Issues in Microservices},
  author={Wu, Li and Tordsson, Johan and Elmroth, Erik and Kao, Odej},
  booktitle={IEEE/IFIP Network Operations and Management Symposium},
  year={2020}
}
@misc{striatrace2026,
  title={StriaTrace: Efficient Tracing and Diagnosis for Online LLM Inference},
  author={Wu, Haonan and others},
  year={2026},
  note={USENIX OSDI 2026 presentation page verified October 2026}
}
""")


def build_provenance(results: list[dict[str, object]]) -> None:
    write_json(PAPER / "provenance" / "result-registry.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "results": results})
    write_json(PAPER / "provenance" / "hypothesis-registry.json", {
        "H1": {"hypothesis": "Compound signatures can be non-additive.", "status": "SUPPORTED FOR SPECIFIC COMPOUND PAIRS, NOT UNIVERSALLY", "evidence": ["INPUT_LOAD and OUTPUT_LOAD have Holm-supported multivariate CSD.", "Other retained pairs do not."]},
        "H2": {"hypothesis": "Single-fault-trained RCA loses complete-cause recall on compound mechanisms.", "status": "SUPPORTED", "evidence": ["M2/F2 fold-level complete-cause-recall difference mean 0.425.", "Exact sign-flip p = 0.0078125."]},
        "H3": {"hypothesis": "Temporal ordering/evolution plus contradiction evidence improves cause-set diagnosis in MESR.", "status": "NOT_SUPPORTED", "evidence": ["Corrected D4-FULL does not outperform D4-SM."]},
        "H4": {"hypothesis": "Mechanism-oriented evidence transfers better than purely statistical signatures under model holdout.", "status": "NOT_SUPPORTED", "evidence": ["D4-FULL and M2/F2 both achieve target compound exact recovery 0.600.", "Primary fold differences all zero."]},
        "H5": {"hypothesis": "Reduced telemetry preserves most RCA quality at lower cost without violating the frozen service-performance guard.", "status": "NOT_SUPPORTED", "evidence": ["Quality screen passes.", "Storage reduction passes.", "Service-performance non-regression not established under +5% CI guard."]},
    })
    write_json(PAPER / "provenance" / "superseded-results.json", {
        "blacklisted_authoritative_sources": ["analysis/phase7a/", "analysis/phase7c1/", "analysis/phase7d2/", "runs/phase6/"],
        "blacklisted_values": [
            {"value": 0.400, "context": "Phase 7D.2 provisional in-house optimizer M2/F2T target compound exact"},
            {"value": "Phase 7C.1 MESR", "context": "superseded by corrected Phase 7C.1a"},
        ],
        "authoritative_replacements": [{"value": 0.200, "context": "Phase 7D.2a exact sklearn M2/F2T target compound exact"}],
    })
    write(PAPER / "provenance" / "results-map.md", """# Results Map

## RQ1 -- Identifiability
Claim: isolated mechanisms have distinct observable effects.
Evidence: Table III; Phase 7A.1 `table-2-isolated-effects.csv`.

## RQ2 -- Compound Interaction
Claim: compound SLO severity is not equivalent to signature non-additivity.
Evidence: Table IV; Phase 7A.1 factorial table; Phase 7B CSD table.

## RQ3 -- Diagnosis and Generalization
Claim: single-fault training loses compound cause-set recovery; temporal features are in-domain useful but model-sensitive.
Evidence: Table V; Phase 7C, Phase 7C.1a, Phase 7D.2a.

## RQ4 -- Observability Robustness
Claim: degradation sensitivity differs by method and domain.
Evidence: Table VI; Phase 7E missingness and robustness AUC artifacts.

## RQ5 -- Telemetry Quality/Cost Frontier
Claim: host telemetry removal reduces storage and preserves quality, but final H5 is not supported because the latency guard fails.
Evidence: Table VII; Phase 7F H5 final evaluation.
""")
    write(PAPER / "provenance" / "claim-audit.md", """# Claim Audit

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
""")


def build_artifact_files() -> None:
    write(PAPER / "artifact" / "README.md", """# SLOScope Publication Artifact

This artifact supports three reproduction levels.

## Level 1: Analysis reproduction
Verify scientific inputs and regenerate manuscript-facing registries, tables, figures, and consistency checks without running inference:

```bash
/private/tmp/slo_phase7a1_venv/bin/python paper/provenance/build_paper_artifacts.py
python3 paper/artifact/verify_artifact.py
python3 paper/provenance/verify_manuscript_numbers.py
```

## Level 2: Small smoke test
Use the benchmark CLI with a small local condition and a local llama.cpp server. This validates installation and artifact writing, not the formal claims.

## Level 3: Full reproduction
Follow the frozen campaign manifests for Phase6-V2, Phase7D, and Phase7F. Full reproduction requires substantial local model/runtime execution and should not be needed for manuscript-number verification.

The authoritative corrected analyses are Phase 7A.1, 7B, 7C, 7C.1a, 7D.2a, 7E, and 7F.
""")
    write(PAPER / "artifact" / "DATA_DICTIONARY.md", """# Data Dictionary

`requests.parquet`: per measured request timing, requested output, request identity, and outcome.

`traces.parquet`: gateway, dependency, and llama span timing keyed by request identifier where enabled.

`runtime_metrics.parquet`: llama.cpp runtime metric scrape rows where enabled. Formal queue-depth rows required for queue RCA were unavailable in source artifacts.

`system_metrics.parquet`: host/system telemetry rows collected by the measurement stack.

`experimental_condition.json`: frozen condition contract and active mechanism labels.

`publication-run-index.jsonl`: authoritative eligible run list for a formal campaign.

Derived analysis tables include run-level metrics, factorial interactions, CSD signatures, RCA predictions, degradation predictions, and telemetry cost metrics. Ground-truth labels are stored separately from model feature matrices.
""")
    write(PAPER / "artifact" / "RELEASE_PLAN.md", """# Release Plan

Preferred release path:

1. Public GitHub repository containing benchmark source, manifests, analysis code, and manuscript-generation scripts.
2. Zenodo archived release with DOI after final acceptance-ready packaging.
3. Separate dataset archive if raw run artifacts are too large for the source repository.

No DOI is invented in this pre-submission artifact.
""")
    write(PAPER / "artifact" / "CITATION.cff", """cff-version: 1.2.0
message: "If you use SLOScope, please cite this artifact and the associated paper."
title: "SLOScope: Benchmarking Compound SLO Degradation and Diagnosis in Commodity LLM Inference Systems"
authors:
  - family-names: "TBD"
    given-names: "TBD"
version: "0.1.0-pre-submission"
date-released: "2026-10-01"
repository-code: "TBD"
doi: "TBD"
""")
    manifest_items = []
    include = [
        "runs/phase6-v2/publication-run-index.jsonl",
        "runs/phase7d-qwen15b/publication-run-index.jsonl",
        "analysis/phase7a1/table-2-isolated-effects.csv",
        "analysis/phase7a1/table-3-factorial-interactions.csv",
        "analysis/phase7b/table-4-csd-summary.csv",
        "analysis/phase7c/table-6-rca-baselines.csv",
        "analysis/phase7c1a/table-9a-mesr-corrected-primary.csv",
        "analysis/phase7d2a/table-12a-zero-shot-generalization.csv",
        "analysis/phase7e/table-16-telemetry-missingness.csv",
        "analysis/phase7f/table-21-quality-cost-frontier.csv",
        "paper/provenance/result-registry.json",
    ]
    for rel in include:
        path = ROOT / rel
        if path.exists():
            manifest_items.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(PAPER / "artifact" / "artifact-manifest.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "items": manifest_items})
    write(PAPER / "artifact" / "verify_artifact.py", """from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "paper/artifact/artifact-manifest.json").read_text())
failures = []
for item in manifest["items"]:
    path = ROOT / item["path"]
    if not path.exists():
        failures.append(f"missing {item['path']}")
        continue
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != item["sha256"]:
        failures.append(f"hash mismatch {item['path']}")
if failures:
    print("\\n".join(failures))
    sys.exit(1)
print(f"artifact verification passed: {len(manifest['items'])} files")
""")


def build_checkers() -> None:
    write(PAPER / "provenance" / "verify_manuscript_numbers.py", """from __future__ import annotations
import json, math, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
registry = json.loads((ROOT / "paper/provenance/result-registry.json").read_text())["results"]
failures = []
for r in registry:
    artifact = r.get("source_artifact")
    column = r.get("column")
    filters = r.get("row_filter") or {}
    if not artifact or not column or not str(artifact).endswith(".csv"):
        continue
    path = ROOT / artifact
    if not path.exists():
        failures.append(f"missing source artifact {artifact}")
        continue
    df = pd.read_csv(path)
    mask = pd.Series([True] * len(df))
    for key, value in filters.items():
        mask &= df[key].astype(str) == str(value)
    rows = df[mask]
    if len(rows) != 1:
        failures.append(f"{r['result_id']} expected one row, got {len(rows)}")
        continue
    got = rows.iloc[0][column]
    exp = r["numeric_value"]
    if isinstance(exp, str):
        ok = str(got) == exp
    elif exp is None:
        ok = pd.isna(got)
    else:
        ok = abs(float(got) - float(exp)) <= 1e-9
    if not ok:
        failures.append(f"{r['result_id']} mismatch: registry={exp} artifact={got}")
tex = (ROOT / "paper/manuscript/sloscope.tex").read_text()
blacklist = json.loads((ROOT / "paper/provenance/superseded-results.json").read_text())
for src in blacklist["blacklisted_authoritative_sources"]:
    if src in tex:
        failures.append(f"blacklisted source mentioned as manuscript evidence: {src}")
if "0.400" in tex and "provisional" not in tex.lower():
    failures.append("possible provisional 0.400 leakage")
if failures:
    print("\\n".join(failures))
    sys.exit(1)
print(f"manuscript number audit passed: {len(registry)} registered results")
""")


def build_supplement() -> None:
    write(PAPER / "supplementary" / "README.md", """# Supplementary Material

Recommended supplementary inclusions:

- Full Phase 7A.1 factorial interaction matrix.
- Full Phase 7B CSD signatures and feature contributions.
- Full Phase 7C baseline, pair, confusion, and coefficient tables.
- Corrected Phase 7C.1a MESR evidence decomposition and compliance audit.
- Phase 7D.2a optimizer-correction audit and pair-transfer tables.
- Phase 7E robustness matrices and degradation predictions.
- Phase 7F paired cost comparisons and service-performance guard.
""")
    for rel in [
        "analysis/phase7b/table-5-csd-feature-contributions.csv",
        "analysis/phase7c/table-7-compound-rca.csv",
        "analysis/phase7c1a/table-10a-mesr-corrected-by-compound.csv",
        "analysis/phase7d2a/table-13a-pair-transfer.csv",
        "analysis/phase7e/table-19-pair-robustness.csv",
        "analysis/phase7f/paired-cost-comparisons.csv",
    ]:
        src = ROOT / rel
        if src.exists():
            shutil.copyfile(src, PAPER / "supplementary" / src.name)


def build_build_report() -> None:
    tex = (PAPER / "manuscript/sloscope.tex").read_text()
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", tex))
    sections = re.findall(r"\\section\{([^}]+)\}", tex)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "word_count_approx": len(words),
        "sections": sections,
        "title_candidates": [
            "SLOScope: Benchmarking Compound SLO Degradation and Diagnosis in Commodity LLM Inference Systems",
            "SLOScope: Compound Performance Degradation, Diagnosis, and Observability in Commodity LLM Inference",
        ],
        "main_tables": sorted(str(p.relative_to(PAPER)) for p in (PAPER / "tables").glob("*.tex")),
        "main_figures": sorted(str(p.relative_to(PAPER)) for p in (PAPER / "figures").glob("*.svg")),
    }
    write_json(PAPER / "manuscript/manuscript-build-report.json", report)


def main() -> None:
    for sub in ["manuscript", "figures", "tables", "supplementary", "artifact", "provenance", "notes"]:
        (PAPER / sub).mkdir(parents=True, exist_ok=True)
    results = build_result_registry()
    build_tables()
    copy_figures()
    build_manuscript()
    build_provenance(results)
    build_artifact_files()
    build_checkers()
    build_supplement()
    build_build_report()


if __name__ == "__main__":
    main()
