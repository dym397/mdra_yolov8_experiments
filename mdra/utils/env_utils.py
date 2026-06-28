from __future__ import annotations

import importlib.metadata
import importlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any


PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "numpy",
    "pandas",
    "matplotlib",
    "Pillow",
    "PyYAML",
    "tqdm",
    "scikit-learn",
)

IMPORT_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "opencv": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PIL": "PIL",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "scikit-learn": "sklearn",
}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(command: list[str], timeout: int = 15) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "output": "executable not found"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (completed.stdout or completed.stderr).strip()
        return {
            "available": True,
            "command": command,
            "returncode": completed.returncode,
            "output": output,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "command": command, "error": str(exc)}


def _check_import(module: str) -> dict[str, Any]:
    try:
        importlib.import_module(module)
        return {"importable": True}
    except Exception as exc:
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_env_info() -> dict[str, Any]:
    uname = platform.uname()
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}

    info: dict[str, Any] = {
        "collected_at": datetime.now().astimezone().isoformat(),
        "system": {
            "platform": platform.platform(),
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "os_release": os_release,
            "uname_a": _run(["uname", "-a"]),
        },
        "python": {
            "version": platform.python_version(),
            "version_full": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
            "DATA_ROOT": os.environ.get("DATA_ROOT"),
            "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT"),
        },
        "packages": {package: _package_version(package) for package in PACKAGES},
        "import_checks": {
            display_name: _check_import(module)
            for display_name, module in IMPORT_MODULES.items()
        },
    }

    torch_info: dict[str, Any] = {"importable": False}
    try:
        import torch

        torch_info.update(
            {
                "importable": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version_built": torch.version.cuda,
                "cudnn_available": torch.backends.cudnn.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
                "devices": [],
            }
        )
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            torch_info["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "total_memory_gib": round(properties.total_memory / (1024**3), 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    except Exception as exc:  # environment audit must survive broken CUDA installs
        torch_info["error"] = f"{type(exc).__name__}: {exc}"
    info["torch"] = torch_info

    info["nvidia_smi"] = {
        "general": _run(["nvidia-smi"]),
        "query": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    return info


def format_env_report(info: dict[str, Any]) -> str:
    system = info["system"]
    python = info["python"]
    torch = info["torch"]
    lines = [
        "MDRA-YOLOv8 environment report",
        "=" * 40,
        f"Collected at: {info['collected_at']}",
        f"Platform: {system['platform']}",
        f"Machine: {system['machine']}",
        f"Python: {python['version_full']}",
        f"Python executable: {python['executable']}",
        "",
        "Packages:",
    ]
    for package, version in info["packages"].items():
        lines.append(f"  {package}: {version or 'NOT INSTALLED'}")
    lines.extend(["", "Import checks:"])
    for package, result in info["import_checks"].items():
        status = "OK" if result["importable"] else f"FAILED ({result.get('error', '')})"
        lines.append(f"  {package}: {status}")

    lines.extend(
        [
            "",
            "PyTorch/CUDA:",
            f"  torch importable: {torch.get('importable')}",
            f"  torch version: {torch.get('version')}",
            f"  CUDA available: {torch.get('cuda_available')}",
            f"  CUDA built version: {torch.get('cuda_version_built')}",
            f"  cuDNN version: {torch.get('cudnn_version')}",
            f"  device count: {torch.get('device_count')}",
        ]
    )
    if torch.get("error"):
        lines.append(f"  error: {torch['error']}")
    for device in torch.get("devices", []):
        lines.append(
            "  GPU {index}: {name}, {total_memory_gib} GiB, compute {compute_capability}".format(
                **device
            )
        )

    query = info["nvidia_smi"]["query"]
    lines.extend(
        [
            "",
            "nvidia-smi:",
            f"  available: {query.get('available')}",
            f"  output: {query.get('output', query.get('error', ''))}",
            "",
        ]
    )
    return "\n".join(lines)
