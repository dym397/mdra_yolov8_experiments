from __future__ import annotations

import torch
from torch import nn


class DetailReconstructionHead(nn.Module):
    """Small training-only P2 edge reconstruction head."""

    def __init__(self, in_channels: int = 64, hidden_channels: int = 32) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        return self.layers(p2)
