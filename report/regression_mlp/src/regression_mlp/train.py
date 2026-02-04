from __future__ import annotations

from itertools import count
from typing import Tuple

import torch
from torch import Tensor, nn

from .data import POLY_DEGREE, W_TARGET, B_TARGET, get_batch, poly_desc
from .model import PolyRegression


def train(
    max_batches: int = 10_000,
    lr: float = 0.1,
    batch_size: int = 32,
    tol: float = 1e-3,
) -> Tuple[float, int, PolyRegression]:
    """
    Train the polynomial regression model.

    Args:
        max_batches: Maximum number of training iterations.
        lr: Learning rate for SGD.
        batch_size: Size of each mini-batch.
        tol: Stop training when loss goes below this value.

    Returns:
        final_loss: Final loss value.
        num_batches: Number of batches used before stopping.
        model: Trained model instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolyRegression(in_dim=POLY_DEGREE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    final_loss: float = float("inf")
    num_batches: int = 0

    for batch_idx in count(1):
        if batch_idx > max_batches:
            break

        model.train()
        x, y = get_batch(batch_size)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred: Tensor = model(x)
        loss: Tensor = criterion(pred, y)

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        num_batches = batch_idx

        if final_loss < tol:
            break

    return final_loss, num_batches, model


def print_learned_vs_target(model: PolyRegression) -> None:
    """
    Print the learned polynomial and the target polynomial
    in a human-readable form, similar to the original example.
    """
    with torch.no_grad():
        w = model.fc.weight.view(-1)  # (POLY_DEGREE,)
        b = model.fc.bias.view(-1)

    learned_str = poly_desc(w, b)
    target_str = poly_desc(W_TARGET.view(-1), B_TARGET.view(-1))

    print("==> Learned function:\t" + learned_str)
    print("==> Actual function:\t" + target_str)


if __name__ == "__main__":
    loss, batches, model = train()
    print(f"Loss: {loss:.6f} after {batches} batches")
    print_learned_vs_target(model)
