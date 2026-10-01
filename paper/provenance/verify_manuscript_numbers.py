from __future__ import annotations
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
    print("\n".join(failures))
    sys.exit(1)
print(f"manuscript number audit passed: {len(registry)} registered results")
