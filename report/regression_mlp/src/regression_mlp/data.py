from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# Degree of the polynomial we want to fit
POLY_DEGREE: int = 4

# Target polynomial parameters (same as the original example)
W_TARGET: Tensor = torch.randn(POLY_DEGREE, 1) * 5
B_TARGET: Tensor = torch.randn(1) * 5


def make_features(x: Tensor) -> Tensor:
    """
    Build polynomial features from a 1D input tensor.

    Given x of shape (N,), this returns a matrix with columns:
        [x, x^2, x^3, ..., x^POLY_DEGREE]

    Args:
        x: Input tensor of shape (N,).

    Returns:
        Tensor of shape (N, POLY_DEGREE).
    """
    x = x.unsqueeze(1)  # (N, 1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], dim=1)


def f(x: Tensor) -> Tensor:
    """
    Target polynomial function f(x) = x W + b.

    Args:
        x: Feature tensor of shape (N, POLY_DEGREE).

    Returns:
        Tensor of shape (N, 1).
    """
    return x @ W_TARGET + B_TARGET


def poly_desc(W: Tensor, b: Tensor) -> str:
    """
    Create a human-readable description of a polynomial.

    Args:
        W: Weight tensor of shape (POLY_DEGREE,).
        b: Bias tensor of shape (1,).

    Returns:
        String like 'y = +1.23 x^1 -0.45 x^2 ... +0.67'.
    """
    result = "y = "
    for i, w in enumerate(W):
        result += "{:+.2f} x^{} ".format(w.item(), i + 1)
    result += "{:+.2f}".format(b[0].item())
    return result


def get_batch(batch_size: int = 32) -> Tuple[Tensor, Tensor]:
    """
    Generate one mini-batch of training data (x, f(x)).

    Args:
        batch_size: Number of samples.

    Returns:
        features: Tensor of shape (batch_size, POLY_DEGREE).
        targets:  Tensor of shape (batch_size, 1).
    """
    random = torch.randn(batch_size)        # (batch_size,)
    x = make_features(random)               # (batch_size, POLY_DEGREE)
    y = f(x)                                # (batch_size, 1)
    return x, y
