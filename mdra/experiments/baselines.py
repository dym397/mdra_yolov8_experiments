from __future__ import annotations

from pathlib import Path
from typing import Any

from mdra.models.baselines import B_MODEL_VARIANTS, DEFAULT_CLASS_NAMES
from mdra.utils.io_utils import load_yaml


DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "batch": 16,
    "effective_batch": 16,
    "imgsz": 640,
    "device": "0",
    "workers": 4,
    "amp": True,
    "seed": 42,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 20,
    "hflip_prob": 0.5,
    "conf_thres": 0.001,
    "iou_thres": 0.7,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nc": 6,
    "class_names": DEFAULT_CLASS_NAMES,
    "pretrained": "weights/yolov8s.pt",
    "dra_lambda": 0.1,
    "dra_hidden_channels": 32,
    "dra_bbox_expansion": 1.2,
    "dra_edge_fusion": "max",
    "dra_edge_alpha": 0.5,
}


def load_b_experiment_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = dict(DEFAULTS)
    config.update(load_yaml(source))
    config["config_source"] = str(source)
    config.setdefault("experiment_id", source.stem)
    variant = str(config.get("variant", ""))
    if variant not in B_MODEL_VARIANTS:
        raise ValueError(f"variant must be one of {B_MODEL_VARIANTS}, got {variant!r}")
    if len(config["class_names"]) != int(config["nc"]):
        raise ValueError("class_names length must equal nc")
    for key in ("epochs", "batch", "effective_batch", "imgsz", "workers", "seed"):
        config[key] = int(config[key])
    for key in ("lr0", "lrf", "weight_decay", "warmup_epochs", "hflip_prob", "conf_thres", "iou_thres"):
        config[key] = float(config[key])
    for key in ("dra_lambda", "dra_bbox_expansion", "dra_edge_alpha"):
        config[key] = float(config[key])
    config["dra_hidden_channels"] = int(config["dra_hidden_channels"])
    if config["epochs"] <= 0 or config["batch"] <= 0 or config["effective_batch"] <= 0:
        raise ValueError("epochs, batch, and effective_batch must be positive")
    if config["imgsz"] <= 0 or config["imgsz"] % 32 != 0:
        raise ValueError("imgsz must be a positive multiple of 32")
    if config["optimizer"].lower() not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be AdamW or SGD")
    if config["dra_lambda"] < 0 or config["dra_bbox_expansion"] < 1.0:
        raise ValueError("dra_lambda must be non-negative and dra_bbox_expansion must be at least 1.0")
    if config["dra_edge_fusion"] not in {"max", "weighted"}:
        raise ValueError("dra_edge_fusion must be max or weighted")
    if not 0.0 <= config["dra_edge_alpha"] <= 1.0:
        raise ValueError("dra_edge_alpha must be within [0, 1]")
    return config


def apply_b_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def resolve_config_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    root = Path(project_root).expanduser().resolve()
    pretrained = resolved.get("pretrained")
    if pretrained:
        path = Path(pretrained).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved["pretrained"] = str(path.resolve())
    return resolved
