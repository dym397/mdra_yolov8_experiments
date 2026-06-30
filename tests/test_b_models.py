from __future__ import annotations

import pytest
import torch

from mdra.models.baselines import B_MODEL_VARIANTS, build_b_model
from mdra.models.lcmf import LightweightCrossModalFusion


@pytest.fixture(scope="module")
def built_models():
    return {variant: build_b_model(variant, pretrained=None) for variant in B_MODEL_VARIANTS}


@pytest.mark.parametrize("variant", B_MODEL_VARIANTS)
def test_b_model_static_detection_levels(variant: str, built_models) -> None:
    model, report = built_models[variant]
    detect = model.model[-1]
    assert detect.nl == (4 if variant in {"early_fusion_p2", "lcmf_p2"} else 3)
    assert report["parameters"] > 0


def test_input_channel_contracts(built_models) -> None:
    assert built_models["visible"][0].model[0].conv.in_channels == 3
    assert built_models["infrared"][0].model[0].conv.in_channels == 3
    assert built_models["early_fusion"][0].model[0].conv.in_channels == 4
    assert built_models["early_fusion_p2"][0].backbone.layers[0].conv.in_channels == 4
    for variant in ("lcmf", "lcmf_p2"):
        model = built_models[variant][0]
        assert model.vis_backbone.layers[0].conv.in_channels == 3
        assert model.ir_backbone.layers[0].conv.in_channels == 1


@pytest.mark.parametrize("variant", ["early_fusion_p2", "lcmf_p2"])
def test_p2_stride_contract(variant: str, built_models) -> None:
    assert torch.equal(
        built_models[variant][0].detect.stride,
        torch.tensor([4.0, 8.0, 16.0, 32.0]),
    )


def test_lcmf_concat_projection() -> None:
    fusion = LightweightCrossModalFusion(16)
    visible = torch.rand(2, 16, 8, 8)
    infrared = torch.rand(2, 16, 8, 8)
    assert fusion(visible, infrared).shape == visible.shape
    assert fusion.fusion.conv.in_channels == 32
    assert fusion.fusion.conv.out_channels == 16
