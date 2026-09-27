from __future__ import annotations

import json
import platform
import subprocess
import sys
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sloscope.config import ExperimentConfig
from sloscope.telemetry.schemas import REQUEST_COLUMNS, RUNTIME_METRIC_COLUMNS, SYSTEM_METRIC_COLUMNS, TRACE_COLUMNS

ARTIFACT_SCHEMA_VERSION = "phase3b.artifacts.v1"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {"phase3a.artifacts.v1", ARTIFACT_SCHEMA_VERSION}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_table(path: Path, rows: List[dict], schema: Dict[str, str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow>=15 is required to write genuine Apache Parquet artifacts") from exc

    arrays = {}
    for name, typ in schema.items():
        values = [row.get(name) for row in rows]
        pa_type = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64()}[typ]
        arrays[name] = pa.array(values, type=pa_type)
    pq.write_table(pa.table(arrays), path)


def read_table(path: Path) -> tuple[Dict[str, str], List[dict]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    schema = {field.name: str(field.type) for field in table.schema}
    return schema, table.to_pylist()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_run(
        self,
        config: ExperimentConfig,
        workload: list[dict],
        ground_truth: dict,
        requests: list[dict],
        system_metrics: list[dict],
        runtime_metrics: list[dict],
        traces: list[dict],
        events: list[Any],
        validation: dict,
        final_state: str,
        start_wall_time: str,
        end_wall_time: str,
        runtime_metadata: Optional[dict] = None,
    ) -> dict:
        cfg = config.with_hash()
        write_json(self.root / "config.json", cfg.to_dict(include_hash=True))
        write_json(self.root / "ground_truth.json", ground_truth)
        write_json(self.root / "workload.json", {"requests": workload})
        write_table(self.root / "requests.parquet", requests, REQUEST_COLUMNS)
        write_table(self.root / "system_metrics.parquet", system_metrics, SYSTEM_METRIC_COLUMNS)
        write_table(self.root / "runtime_metrics.parquet", runtime_metrics, RUNTIME_METRIC_COLUMNS)
        write_table(self.root / "traces.parquet", traces, TRACE_COLUMNS)
        write_jsonl(self.root / "events.jsonl", [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in events])
        if runtime_metadata is not None:
            write_json(self.root / "runtime_metadata.json", runtime_metadata)
        write_json(self.root / "validation.json", validation)
        inventory = sorted(p.name for p in self.root.iterdir() if p.is_file() and p.name != "manifest.json")
        hashed = [
            "config.json",
            "ground_truth.json",
            "workload.json",
            "requests.parquet",
            "system_metrics.parquet",
            "runtime_metrics.parquet",
            "traces.parquet",
            "events.jsonl",
        ]
        if runtime_metadata is not None:
            hashed.append("runtime_metadata.json")
        artifact_hashes = {name: sha256_file(self.root / name) for name in hashed if (self.root / name).exists()}
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": cfg.run_id,
            "seed": cfg.seed,
            "config_hash": cfg.config_hash,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git_revision": git_revision(),
            "start_timestamp": start_wall_time,
            "end_timestamp": end_wall_time,
            "final_lifecycle_state": final_state,
            "artifact_inventory": inventory + ["manifest.json"],
            "artifact_hashes": artifact_hashes,
            "runtime_metadata": runtime_metadata or {},
        }
        write_json(self.root / "manifest.json", manifest)
        return manifest
