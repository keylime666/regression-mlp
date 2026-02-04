# Polynomial Regression (Improved PyTorch Example)

This project refactors the official PyTorch regression demo into a cleaner and more modular structure.

Original code:
https://github.com/pytorch/examples/blob/main/regression/main.py

## Project structure

src/regression_mlp/
    data.py
    model.py
    train.py

tests/
    test_model.py

pyproject.toml

## How to run

Run training:
uv run python -m regression_mlp.train

Run tests:
uv run pytest

## Improvements

- Split original single-file script into modules (data / model / train)
- Added Python type hints
- Added pytest test
- Replaced manual parameter update with torch.optim.SGD
- Managed dependencies with uv and pyproject.toml

## Example output

Loss: 0.000696 after 408 batches
==> Learned function:   y = -6.14 x^1 +4.55 x^2 -1.84 x^3 -1.80 x^4 -3.33
==> Actual function:    y = -6.20 x^1 +4.60 x^2 -1.81 x^3 -1.80 x^4 -3.36
