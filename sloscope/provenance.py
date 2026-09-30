from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


def git_dirty(root: Path | None = None) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root or Path.cwd(),
            text=True,
            capture_output=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def source_tree_files(root: Path | None = None) -> list[Path]:
    base = root or Path.cwd()
    files = [base / "pyproject.toml"]
    files.extend(sorted((base / "sloscope").glob("**/*.py")))
    return [path for path in files if path.is_file()]


def source_tree_sha256(root: Path | None = None, paths: Iterable[Path] | None = None) -> str:
    base = root or Path.cwd()
    selected = list(paths) if paths is not None else source_tree_files(base)
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.relative_to(base).as_posix()):
        rel = path.relative_to(base).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
