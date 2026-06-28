from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    """Return a filesystem-safe local timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory and all parents, returning a Path."""
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_path(root: str | Path, value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root).expanduser() / path
    return path.resolve()


def sanitize_experiment_id(value: str) -> str:
    """Restrict experiment identifiers to portable filename characters."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("experiment_id is empty after sanitization")
    return cleaned


def unique_experiment_dir(output_root: str | Path, experiment_id: str) -> Path:
    """Create a unique experiment directory without overwriting an existing run."""
    root = safe_mkdir(output_root)
    experiment_id = sanitize_experiment_id(experiment_id)
    candidate = root / experiment_id
    if not candidate.exists():
        candidate.mkdir()
        return candidate

    stamped = root / f"{experiment_id}_{timestamp()}"
    candidate = stamped
    counter = 1
    while candidate.exists():
        candidate = root / f"{stamped.name}_{counter:03d}"
        counter += 1
    candidate.mkdir()
    return candidate


def require_writable_targets(paths: list[str | Path], overwrite: bool = False) -> None:
    """Fail before a script overwrites any requested output file."""
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        joined = "\n  - ".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing output files. Use --overwrite only when "
            f"intentional:\n  - {joined}"
        )

