import torch

from regression_mlp.data import make_features
from regression_mlp.model import PolyRegression


def test_poly_regression_forward_shape() -> None:
    """
    Verify that the model produces outputs with the correct shape.
    """
    x_raw = torch.randn(10)          # (10,)
    x = make_features(x_raw)         # (10, POLY_DEGREE)

    model = PolyRegression(in_dim=x.shape[1])
    y = model(x)

    assert y.shape == (10, 1)
