from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules import C2f, Conv, Detect, SPPF
from ultralytics.nn.tasks import DetectionModel

from mdra.models.lcmf import LightweightCrossModalFusion
from mdra.models.dra import DetailReconstructionHead


B_MODEL_VARIANTS = (
    "visible",
    "infrared",
    "early_fusion",
    "early_fusion_p2",
    "lcmf",
    "lcmf_p2",
    "early_fusion_p2_full_edge_dra",
    "early_fusion_p2_target_aware_dra",
)
DEFAULT_CLASS_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def _loss_args(config: dict[str, Any] | None = None) -> SimpleNamespace:
    config = config or {}
    return SimpleNamespace(
        box=float(config.get("box", 7.5)),
        cls=float(config.get("cls", 0.5)),
        dfl=float(config.get("dfl", 1.5)),
    )


class YOLOv8sBackbone(nn.Module):
    """YOLOv8s backbone exposing P2/P3/P4/P5 features."""

    source_indices = tuple(range(10))

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Conv(in_channels, 32, 3, 2),
                Conv(32, 64, 3, 2),
                C2f(64, 64, 1, True),
                Conv(64, 128, 3, 2),
                C2f(128, 128, 2, True),
                Conv(128, 256, 3, 2),
                C2f(256, 256, 2, True),
                Conv(256, 512, 3, 2),
                C2f(512, 512, 1, True),
                SPPF(512, 512, 5),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layers[0](x)
        x = self.layers[1](x)
        p2 = self.layers[2](x)
        x = self.layers[3](p2)
        p3 = self.layers[4](x)
        x = self.layers[5](p3)
        p4 = self.layers[6](x)
        x = self.layers[7](p4)
        x = self.layers[8](x)
        p5 = self.layers[9](x)
        return p2, p3, p4, p5


def _make_legacy_detect(nc: int, channels: tuple[int, ...], strides: tuple[int, ...]) -> Detect:
    previous = Detect.legacy
    Detect.legacy = True
    try:
        detect = Detect(nc=nc, ch=channels)
    finally:
        Detect.legacy = previous
    detect.legacy = True
    detect.stride = torch.tensor(strides, dtype=torch.float32)
    detect.bias_init()
    return detect


class ThreeScaleNeck(nn.Module):
    source_mapping = {
        "c2f_p4": 12,
        "c2f_p3": 15,
        "down_p3": 16,
        "c2f_n4": 18,
        "down_n4": 19,
        "c2f_n5": 21,
    }

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p4 = C2f(768, 256, 1)
        self.c2f_p3 = C2f(384, 128, 1)
        self.down_p3 = Conv(128, 128, 3, 2)
        self.c2f_n4 = C2f(384, 256, 1)
        self.down_n4 = Conv(256, 256, 3, 2)
        self.c2f_n5 = C2f(768, 512, 1)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n4_top = self.c2f_p4(torch.cat((self.up(p5), p4), dim=1))
        n3 = self.c2f_p3(torch.cat((self.up(n4_top), p3), dim=1))
        n4 = self.c2f_n4(torch.cat((self.down_p3(n3), n4_top), dim=1))
        n5 = self.c2f_n5(torch.cat((self.down_n4(n4), p5), dim=1))
        return n3, n4, n5


class FourScaleP2Neck(nn.Module):
    """P2/P3/P4/P5 PAN neck shared by controlled P2 baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_top_p4 = C2f(768, 256, 1)
        self.c2f_top_p3 = C2f(384, 128, 1)
        self.c2f_p2 = C2f(192, 64, 1)
        self.down_p2 = Conv(64, 64, 3, 2)
        self.c2f_out_p3 = C2f(192, 128, 1)
        self.down_p3 = Conv(128, 128, 3, 2)
        self.c2f_out_p4 = C2f(384, 256, 1)
        self.down_p4 = Conv(256, 256, 3, 2)
        self.c2f_out_p5 = C2f(768, 512, 1)

    def forward(
        self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        top_p4 = self.c2f_top_p4(torch.cat((self.up(p5), p4), dim=1))
        top_p3 = self.c2f_top_p3(torch.cat((self.up(top_p4), p3), dim=1))
        out_p2 = self.c2f_p2(torch.cat((self.up(top_p3), p2), dim=1))
        out_p3 = self.c2f_out_p3(torch.cat((self.down_p2(out_p2), top_p3), dim=1))
        out_p4 = self.c2f_out_p4(torch.cat((self.down_p3(out_p3), top_p4), dim=1))
        out_p5 = self.c2f_out_p5(torch.cat((self.down_p4(out_p4), p5), dim=1))
        return out_p2, out_p3, out_p4, out_p5


class EarlyFusionP2Detector(nn.Module):
    """Four-channel early-fusion backbone with a P2/P3/P4/P5 detection neck."""

    def __init__(
        self,
        nc: int,
        *,
        class_names: list[str],
        dra_mode: str | None = None,
        dra_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.variant = "early_fusion_p2" if dra_mode is None else f"early_fusion_p2_{dra_mode}_dra"
        self.dra_mode = dra_mode
        self.backbone = YOLOv8sBackbone(4)
        self.neck = FourScaleP2Neck()
        self.detect = _make_legacy_detect(nc, (64, 128, 256, 512), (4, 8, 16, 32))
        self.dra_head = (
            DetailReconstructionHead(64, dra_hidden_channels) if dra_mode is not None else None
        )
        self.model = [self.detect]
        self.stride = self.detect.stride
        self.names = {index: name for index, name in enumerate(class_names)}
        self.args = _loss_args()

    def _apply(self, fn):
        super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.stride = self.detect.stride
        if isinstance(self.detect.anchors, torch.Tensor):
            self.detect.anchors = fn(self.detect.anchors)
        if isinstance(self.detect.strides, torch.Tensor):
            self.detect.strides = fn(self.detect.strides)
        return self

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"early_fusion_p2 expects BCHW input with 4 channels, got {tuple(x.shape)}")
        features = self.backbone(x)
        pyramid = self.neck(*features)
        detection = self.detect(list(pyramid))
        if self.training and self.dra_head is not None:
            return {"det_preds": detection, "dra_pred": self.dra_head(pyramid[0])}
        return detection

    def forward_dra_diagnostic(self, x: torch.Tensor) -> tuple[Any, torch.Tensor]:
        """Explicit opt-in DRA output for validation diagnostics; detection metrics never call this."""
        if self.dra_head is None:
            raise RuntimeError("this model has no DRA head")
        features = self.backbone(x)
        pyramid = self.neck(*features)
        return self.detect(list(pyramid)), self.dra_head(pyramid[0])


class DualStreamLCMFDetector(nn.Module):
    def __init__(self, nc: int, *, with_p2: bool, class_names: list[str]) -> None:
        super().__init__()
        self.variant = "lcmf_p2" if with_p2 else "lcmf"
        self.vis_backbone = YOLOv8sBackbone(3)
        self.ir_backbone = YOLOv8sBackbone(1)
        scales = (("p2", 64), ("p3", 128), ("p4", 256), ("p5", 512)) if with_p2 else (
            ("p3", 128),
            ("p4", 256),
            ("p5", 512),
        )
        self.fusions = nn.ModuleDict({name: LightweightCrossModalFusion(channels) for name, channels in scales})
        self.with_p2 = bool(with_p2)
        if self.with_p2:
            self.neck = FourScaleP2Neck()
            self.detect = _make_legacy_detect(nc, (64, 128, 256, 512), (4, 8, 16, 32))
        else:
            self.neck = ThreeScaleNeck()
            self.detect = _make_legacy_detect(nc, (128, 256, 512), (8, 16, 32))
        self.model = [self.detect]
        self.stride = self.detect.stride
        self.names = {index: name for index, name in enumerate(class_names)}
        self.args = _loss_args()

    def _apply(self, fn):
        super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.stride = self.detect.stride
        if isinstance(self.detect.anchors, torch.Tensor):
            self.detect.anchors = fn(self.detect.anchors)
        if isinstance(self.detect.strides, torch.Tensor):
            self.detect.strides = fn(self.detect.strides)
        return self

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"{self.variant} expects BCHW input with 4 channels, got {tuple(x.shape)}")
        vis_features = self.vis_backbone(x[:, :3])
        ir_features = self.ir_backbone(x[:, 3:4])
        fused = {
            name: self.fusions[name](vis_features[index], ir_features[index])
            for index, name in enumerate(("p2", "p3", "p4", "p5"))
            if name in self.fusions
        }
        if self.with_p2:
            features = self.neck(fused["p2"], fused["p3"], fused["p4"], fused["p5"])
        else:
            features = self.neck(fused["p3"], fused["p4"], fused["p5"])
        return self.detect(list(features))


def _checkpoint_model(path: str | Path) -> nn.Module:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint.float()
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported pretrained checkpoint format: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, nn.Module):
        raise ValueError(f"checkpoint does not contain an Ultralytics model: {path}")
    return model.float()


def _copy_module(source: nn.Module, target: nn.Module, *, adapt_input_channels: int | None = None) -> int:
    source_state = copy.deepcopy(source.state_dict())
    if adapt_input_channels is not None and "conv.weight" in source_state:
        weight = source_state["conv.weight"]
        if adapt_input_channels == 1:
            source_state["conv.weight"] = weight.mean(dim=1, keepdim=True)
        elif adapt_input_channels == 4:
            expanded = torch.zeros(
                (weight.shape[0], 4, *weight.shape[2:]), dtype=weight.dtype, device=weight.device
            )
            expanded[:, :3] = weight
            expanded[:, 3:4] = weight.mean(dim=1, keepdim=True)
            source_state["conv.weight"] = expanded
    target_state = target.state_dict()
    compatible = {key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape}
    target.load_state_dict(compatible, strict=False)
    return len(compatible)


def _initialize_stock(model: DetectionModel, source: nn.Module, input_channels: int) -> dict[str, Any]:
    source_state = source.state_dict()
    target_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    adapted = False
    for key, value in source_state.items():
        if key == "model.0.conv.weight" and input_channels == 4:
            expanded = torch.zeros((value.shape[0], 4, *value.shape[2:]), dtype=value.dtype)
            expanded[:, :3] = value
            expanded[:, 3:4] = value.mean(dim=1, keepdim=True)
            if key in target_state and expanded.shape == target_state[key].shape:
                transferred[key] = expanded
                adapted = True
        elif key in target_state and value.shape == target_state[key].shape:
            transferred[key] = value
    model.load_state_dict(transferred, strict=False)
    return {"transferred_tensors": len(transferred), "adapted_first_conv": adapted}


def _initialize_early_fusion_p2(model: EarlyFusionP2Detector, source: nn.Module) -> dict[str, Any]:
    if not hasattr(source, "model"):
        raise ValueError("pretrained Ultralytics model does not expose model layers")
    transferred = 0
    for index, source_index in enumerate(YOLOv8sBackbone.source_indices):
        transferred += _copy_module(
            source.model[source_index],
            model.backbone.layers[index],
            adapt_input_channels=4 if index == 0 else None,
        )
    mapping = {
        "c2f_top_p4": 12,
        "c2f_top_p3": 15,
        "down_p3": 16,
        "c2f_out_p4": 18,
        "down_p4": 19,
        "c2f_out_p5": 21,
    }
    for attribute, source_index in mapping.items():
        transferred += _copy_module(source.model[source_index], getattr(model.neck, attribute))
    source_detect = source.model[-1]
    for source_index, target_index in enumerate((1, 2, 3)):
        transferred += _copy_module(source_detect.cv2[source_index], model.detect.cv2[target_index])
        transferred += _copy_module(source_detect.cv3[source_index], model.detect.cv3[target_index])
    transferred += _copy_module(source_detect.dfl, model.detect.dfl)
    return {
        "transferred_tensors": transferred,
        "adapted_first_conv": True,
        "new_modules": "P2 neck/head",
    }


def _initialize_dual(model: DualStreamLCMFDetector, source: nn.Module) -> dict[str, Any]:
    if not hasattr(source, "model"):
        raise ValueError("pretrained Ultralytics model does not expose model layers")
    transferred = 0
    for index, source_index in enumerate(YOLOv8sBackbone.source_indices):
        transferred += _copy_module(source.model[source_index], model.vis_backbone.layers[index])
        transferred += _copy_module(
            source.model[source_index],
            model.ir_backbone.layers[index],
            adapt_input_channels=1 if index == 0 else None,
        )

    if isinstance(model.neck, ThreeScaleNeck):
        for attribute, source_index in model.neck.source_mapping.items():
            transferred += _copy_module(source.model[source_index], getattr(model.neck, attribute))
        transferred += _copy_module(source.model[-1], model.detect)
    else:
        mapping = {
            "c2f_top_p4": 12,
            "c2f_top_p3": 15,
            "down_p3": 16,
            "c2f_out_p4": 18,
            "down_p4": 19,
            "c2f_out_p5": 21,
        }
        for attribute, source_index in mapping.items():
            transferred += _copy_module(source.model[source_index], getattr(model.neck, attribute))
        source_detect = source.model[-1]
        for source_index, target_index in enumerate((1, 2, 3)):
            transferred += _copy_module(source_detect.cv2[source_index], model.detect.cv2[target_index])
            transferred += _copy_module(source_detect.cv3[source_index], model.detect.cv3[target_index])
        transferred += _copy_module(source_detect.dfl, model.detect.dfl)
    return {
        "transferred_tensors": transferred,
        "infrared_first_conv": "mean of pretrained RGB kernels",
        "new_modules": "LCMF concat projections" + (", P2 neck/head" if model.with_p2 else ""),
    }


def build_b_model(
    variant: str,
    *,
    nc: int = 6,
    class_names: list[str] | None = None,
    pretrained: str | Path | None = None,
    loss_gains: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if variant not in B_MODEL_VARIANTS:
        raise ValueError(f"unsupported B model variant={variant!r}; expected one of {B_MODEL_VARIANTS}")
    names = list(class_names or DEFAULT_CLASS_NAMES)
    if len(names) != nc:
        raise ValueError(f"class_names length {len(names)} does not match nc={nc}")

    if variant in {"visible", "infrared", "early_fusion"}:
        channels = 4 if variant == "early_fusion" else 3
        model: nn.Module = DetectionModel("yolov8s.yaml", ch=channels, nc=nc, verbose=False)
        model.variant = variant
        model.args = _loss_args(loss_gains)
        model.names = {index: name for index, name in enumerate(names)}
    elif variant in {
        "early_fusion_p2",
        "early_fusion_p2_full_edge_dra",
        "early_fusion_p2_target_aware_dra",
    }:
        dra_mode = None
        if variant == "early_fusion_p2_full_edge_dra":
            dra_mode = "full_edge"
        elif variant == "early_fusion_p2_target_aware_dra":
            dra_mode = "target_aware"
        model = EarlyFusionP2Detector(
            nc,
            class_names=names,
            dra_mode=dra_mode,
            dra_hidden_channels=int((loss_gains or {}).get("dra_hidden_channels", 32)),
        )
        model.args = _loss_args(loss_gains)
    else:
        model = DualStreamLCMFDetector(nc, with_p2=variant == "lcmf_p2", class_names=names)
        model.args = _loss_args(loss_gains)

    report: dict[str, Any] = {
        "variant": variant,
        "pretrained": str(Path(pretrained).expanduser().resolve()) if pretrained else None,
        "transferred_tensors": 0,
        "adapted_first_conv": False,
    }
    if pretrained:
        source = _checkpoint_model(pretrained)
        if variant in {"visible", "infrared", "early_fusion"}:
            report.update(_initialize_stock(model, source, 4 if variant == "early_fusion" else 3))
        elif variant in {
            "early_fusion_p2",
            "early_fusion_p2_full_edge_dra",
            "early_fusion_p2_target_aware_dra",
        }:
            report.update(_initialize_early_fusion_p2(model, source))
        else:
            report.update(_initialize_dual(model, source))
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    report["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return model, report


def load_b_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("not an MDRA B-experiment checkpoint")
    config = checkpoint["config"]
    model, _ = build_b_model(
        str(config["variant"]),
        nc=int(config.get("nc", 6)),
        class_names=list(config["class_names"]),
        pretrained=None,
        loss_gains=config,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint
