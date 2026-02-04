from __future__ import annotations

from torch import nn, Tensor


class PolyRegression(nn.Module):
    """
    Simple linear model for polynomial regression.

    This is equivalent to:
        y = w1 * x^1 + w2 * x^2 + ... + w_n * x^n + b,
    where the input features are [x, x^2, ..., x^n].
    """

    def __init__(self, in_dim: int = 4) -> None:
        """
        Args:
            in_dim: Input feature dimension (POLY_DEGREE).
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, in_dim).

        Returns:
            Tensor of shape (N, 1).
        """
        return self.fc(x)
