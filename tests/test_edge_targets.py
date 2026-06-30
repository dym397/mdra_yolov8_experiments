from __future__ import annotations

import torch

from mdra.data.edge_targets import (
    build_bbox_mask,
    build_dra_supervision,
    masked_l1_loss,
    rgb_to_gray,
    sobel_edge,
)


def _metadata(batch_size: int = 1) -> dict:
    return {
        "original_shape": [(32, 32)] * batch_size,
        "ratio_pad": [(1.0, (0, 0))] * batch_size,
        "bboxes": torch.tensor([[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32),
        "batch_idx": torch.tensor([0], dtype=torch.long),
    }


def test_rgb_channel_order_is_explicit() -> None:
    rgb = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    assert torch.allclose(rgb_to_gray(rgb), torch.tensor([[[[0.299]]]]))


def test_replicate_sobel_has_no_constant_image_border_frame() -> None:
    image = torch.full((1, 1, 16, 16), 0.7)
    assert sobel_edge(image).max() < 1e-6


def test_direct_hard_mask_preserves_tiny_box() -> None:
    metadata = _metadata()
    mask = build_bbox_mask(
        metadata["bboxes"], metadata["batch_idx"], batch_size=1,
        size=(8, 8), expansion=1.2, dtype=torch.float32,
    )
    assert mask.sum() >= 1
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})


def test_target_aware_empty_image_has_zero_dra_loss() -> None:
    image = torch.rand(1, 4, 32, 32)
    metadata = _metadata()
    metadata["bboxes"] = torch.zeros((0, 4))
    metadata["batch_idx"] = torch.zeros((0,), dtype=torch.long)
    supervision = build_dra_supervision(
        image, metadata, output_size=(8, 8), mode="target_aware"
    )
    prediction = torch.rand_like(supervision.target, requires_grad=True)
    loss = masked_l1_loss(prediction, supervision.target, supervision.loss_mask)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None


def test_full_edge_excludes_letterbox_padding() -> None:
    image = torch.rand(1, 4, 32, 32)
    metadata = _metadata()
    metadata["original_shape"] = [(16, 32)]
    metadata["ratio_pad"] = [(1.0, (0, 8))]
    supervision = build_dra_supervision(image, metadata, output_size=(8, 8), mode="full_edge")
    assert supervision.valid_input_mask[:, :, :8].sum() == 0
    assert supervision.valid_input_mask[:, :, 24:].sum() == 0
