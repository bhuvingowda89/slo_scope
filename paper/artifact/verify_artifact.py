from __future__ import annotations
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
    print("\n".join(failures))
    sys.exit(1)
print(f"artifact verification passed: {len(manifest['items'])} files")
