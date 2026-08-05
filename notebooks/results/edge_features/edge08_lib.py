"""Support module for edge_08 (importable so inspect.getsource can fold it in the notebook)."""

import torch
import torch.nn as nn


class MultiHeadFM(nn.Module):
    """One shared FM interaction core (factors V), one linear+bias per head. Head 0 = primary (r2);
    extra heads = auxiliary targets that REGULARIZE / un-starve the shared V. Forcing V to also predict
    r1 (the pre-d8 residual) reintroduces the structure d8 took -> the primary head inherits it."""

    def __init__(self, d: int, heads: int, rank: int = 4):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, rank) * 0.01)
        self.lin = nn.Linear(d, heads)
        self.b = nn.Parameter(torch.zeros(heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x @ self.V
        ss = (x * x) @ (self.V * self.V)
        return self.lin(x) + self.b + 0.5 * (s * s - ss).sum(1, keepdim=True)
