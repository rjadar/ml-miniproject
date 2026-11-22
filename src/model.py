"""Small configurable 1D CNN for nine-channel SisFall windows."""
from __future__ import annotations

import torch
from torch import nn


class FallCNN1D(nn.Module):
    def __init__(self, in_channels: int = 9, filters: tuple[int, ...] = (16, 32, 64), kernel_size: int = 9, dropout: float = 0.30):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        blocks = []
        previous = in_channels
        for width in filters:
            blocks.extend([
                nn.Conv1d(previous, width, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ])
            previous = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(previous, 1)  # One logit: Fall (1) vs ADL (0).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x)).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
