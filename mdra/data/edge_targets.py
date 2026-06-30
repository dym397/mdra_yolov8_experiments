from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


DRA_MODES = ("full_edge", "target_aware")


@dataclass
class DRASupervision:
    target: torch.Tensor
    loss_mask: torch.Tensor
    edge_visible: torch.Tensor
    edge_infrared: torch.Tensor
    edge_multimodal: torch.Tensor
    input_mask: torch.Tensor
    valid_input_mask: torch.Tensor


def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Convert normalized RGB tensors to gray without changing channel order."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected RGB BCHW tensor, got {tuple(rgb.shape)}")
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def sobel_edge(image: torch.Tensor) -> torch.Tensor:
    """Return a fixed-scale Sobel magnitude in [0, 1] with no padding-frame artifact."""
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"expected single-channel BCHW tensor, got {tuple(image.shape)}")
    kernel_x = image.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    edge = ((gx.abs() + gy.abs()) / 8.0).clamp_(0.0, 1.0)
    edge = edge.clone()
    edge[:, :, 0, :] = 0
    edge[:, :, -1, :] = 0
    edge[:, :, :, 0] = 0
    edge[:, :, :, -1] = 0
    return edge


def _content_bounds(
    original_shape: tuple[int, int],
    ratio_pad: tuple[float, tuple[int, int]],
    input_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    original_h, original_w = (int(value) for value in original_shape)
    ratio, (pad_left, pad_top) = ratio_pad
    input_h, input_w = input_shape
    resized_w = max(1, int(round(original_w * float(ratio))))
    resized_h = max(1, int(round(original_h * float(ratio))))
    left = max(0, int(pad_left))
    top = max(0, int(pad_top))
    return left, top, min(input_w, left + resized_w), min(input_h, top + resized_h)


def build_valid_content_mask(
    original_shapes: list[tuple[int, int]],
    ratio_pads: list[tuple[float, tuple[int, int]]],
    *,
    batch_size: int,
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    shrink_border: int = 1,
) -> torch.Tensor:
    """Build masks that remove letterbox padding and its one-pixel transition edge."""
    if len(original_shapes) != batch_size or len(ratio_pads) != batch_size:
        raise ValueError("letterbox metadata length does not match batch size")
    height, width = size
    result = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    for index, (shape, ratio_pad) in enumerate(zip(original_shapes, ratio_pads)):
        left, top, right, bottom = _content_bounds(shape, ratio_pad, size)
        left += shrink_border
        top += shrink_border
        right -= shrink_border
        bottom -= shrink_border
        if right > left and bottom > top:
            result[index, 0, top:bottom, left:right] = 1.0
    return result


def _scale_bounds(start: int, end: int, source: int, target: int) -> tuple[int, int]:
    scaled_start = max(0, min(target, int(math.floor(start * target / source))))
    scaled_end = max(0, min(target, int(math.ceil(end * target / source))))
    return scaled_start, max(scaled_start, scaled_end)


def build_valid_output_mask(
    original_shapes: list[tuple[int, int]],
    ratio_pads: list[tuple[float, tuple[int, int]]],
    *,
    batch_size: int,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_h, input_w = input_size
    output_h, output_w = output_size
    result = torch.zeros((batch_size, 1, output_h, output_w), device=device, dtype=dtype)
    for index, (shape, ratio_pad) in enumerate(zip(original_shapes, ratio_pads)):
        left, top, right, bottom = _content_bounds(shape, ratio_pad, input_size)
        left, right = _scale_bounds(left + 1, max(left + 1, right - 1), input_w, output_w)
        top, bottom = _scale_bounds(top + 1, max(top + 1, bottom - 1), input_h, output_h)
        if right > left and bottom > top:
            result[index, 0, top:bottom, left:right] = 1.0
    return result


def build_bbox_mask(
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    batch_size: int,
    size: tuple[int, int],
    expansion: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize normalized xywh boxes directly on a target grid as a hard union mask."""
    if expansion < 1.0:
        raise ValueError("bbox expansion must be at least 1.0")
    height, width = size
    mask = torch.zeros((batch_size, 1, height, width), device=bboxes.device, dtype=dtype)
    for box, image_index_tensor in zip(bboxes, batch_idx):
        image_index = int(image_index_tensor.item())
        cx, cy, box_w, box_h = (float(value) for value in box)
        half_w = box_w * expansion * width / 2.0
        half_h = box_h * expansion * height / 2.0
        center_x = cx * width
        center_y = cy * height
        left = max(0, min(width - 1, int(math.floor(center_x - half_w))))
        top = max(0, min(height - 1, int(math.floor(center_y - half_h))))
        right = max(left + 1, min(width, int(math.ceil(center_x + half_w))))
        bottom = max(top + 1, min(height, int(math.ceil(center_y + half_h))))
        mask[image_index, 0, top:bottom, left:right] = 1.0
    return mask


@torch.no_grad()
def build_dra_supervision(
    images: torch.Tensor,
    batch: dict[str, Any],
    *,
    output_size: tuple[int, int],
    mode: str,
    bbox_expansion: float = 1.2,
    edge_fusion: str = "max",
    edge_alpha: float = 0.5,
) -> DRASupervision:
    if mode not in DRA_MODES:
        raise ValueError(f"unsupported DRA mode={mode!r}; expected one of {DRA_MODES}")
    if images.ndim != 4 or images.shape[1] != 4:
        raise ValueError(f"DRA supervision expects [R,G,B,IR] BCHW input, got {tuple(images.shape)}")
    batch_size, _, input_h, input_w = images.shape
    visible_gray = rgb_to_gray(images[:, :3])
    edge_visible = sobel_edge(visible_gray)
    edge_infrared = sobel_edge(images[:, 3:4])
    if edge_fusion == "max":
        edge_multimodal = torch.maximum(edge_visible, edge_infrared)
    elif edge_fusion == "weighted":
        if not 0.0 <= edge_alpha <= 1.0:
            raise ValueError("edge_alpha must be within [0, 1]")
        edge_multimodal = edge_alpha * edge_visible + (1.0 - edge_alpha) * edge_infrared
    else:
        raise ValueError("edge_fusion must be 'max' or 'weighted'")

    valid_input = build_valid_content_mask(
        batch["original_shape"],
        batch["ratio_pad"],
        batch_size=batch_size,
        size=(input_h, input_w),
        device=images.device,
        dtype=images.dtype,
    )
    valid_output = build_valid_output_mask(
        batch["original_shape"],
        batch["ratio_pad"],
        batch_size=batch_size,
        input_size=(input_h, input_w),
        output_size=output_size,
        device=images.device,
        dtype=images.dtype,
    )
    if mode == "target_aware":
        input_mask = build_bbox_mask(
            batch["bboxes"], batch["batch_idx"], batch_size=batch_size,
            size=(input_h, input_w), expansion=bbox_expansion, dtype=images.dtype,
        ) * valid_input
        output_mask = build_bbox_mask(
            batch["bboxes"], batch["batch_idx"], batch_size=batch_size,
            size=output_size, expansion=bbox_expansion, dtype=images.dtype,
        ) * valid_output
    else:
        input_mask = valid_input
        output_mask = valid_output

    target_input = edge_multimodal * input_mask
    target = F.interpolate(target_input, size=output_size, mode="area") * output_mask
    return DRASupervision(
        target=target,
        loss_mask=output_mask,
        edge_visible=edge_visible * valid_input,
        edge_infrared=edge_infrared * valid_input,
        edge_multimodal=edge_multimodal * valid_input,
        input_mask=input_mask,
        valid_input_mask=valid_input,
    )


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average each non-empty image over supervised cells, then average the valid images."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(
            f"prediction, target, and mask must share shape; got {prediction.shape}, {target.shape}, {mask.shape}"
        )
    numerator = ((prediction - target).abs() * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    valid = denominator > 0
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return (numerator[valid] / denominator[valid].clamp_min(1.0)).mean()
