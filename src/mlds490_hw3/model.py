"""Model required by the assignment."""

from __future__ import annotations

from torch import Tensor, nn


class TwoLayerNet(nn.Module):
    """Two fully connected layers with 128 hidden ReLU units."""

    def __init__(self, hidden_units: int = 128, n_classes: int = 62) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images.float())
