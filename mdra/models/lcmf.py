from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import Conv


class LightweightCrossModalFusion(nn.Module):
    """Simple Concat -> 1x1 Conv -> BN -> SiLU fusion baseline."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.fusion = Conv(channels * 2, channels, k=1, s=1)

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        if visible.shape != infrared.shape:
            raise ValueError(f"fusion shape mismatch: visible={visible.shape}, infrared={infrared.shape}")
        return self.fusion(torch.cat((visible, infrared), dim=1))
