"""Example: Differentiable Weighted PCA Low-Rank Denoising Layer on PyTorch MPS.

Computes principal subspaces in float64 on Apple Silicon's AMX coprocessor at zero
tensor-copy cost, with analytical projector gradients computed on the Metal GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import mps_metal_kit as mmk


class DifferentiablePCADenoiser(nn.Module):
    """Projects variable-length token sequences onto their learned top-k principal subspace."""

    def __init__(self, d_model: int = 128, rank: int = 4):
        super().__init__()
        self.importance = nn.Linear(d_model, 1)
        self.rank = rank

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        weights = torch.sigmoid(self.importance(x)).squeeze(-1)  # [B, M]
        U = mmk.weighted_gram_subspace(x, weights, rank=self.rank, lengths=lengths)  # [B, D, rank]
        # Orthogonal low-rank reconstruction: x_hat = (x @ U) @ U^T
        x_proj = torch.bmm(torch.bmm(x, U), U.transpose(1, 2))
        return x_proj


if __name__ == "__main__":
    dev = torch.device("mps" if mmk.is_mps_available() else "cpu")
    denoiser = DifferentiablePCADenoiser(d_model=128, rank=4).to(dev)

    x = torch.randn(8, 32, 128, device=dev, requires_grad=True)
    lengths = [28, 19, 32, 12, 25, 30, 15, 22]

    x_hat = denoiser(x, lengths)
    x_hat.pow(2).mean().backward()
    print(f"Success on {dev}! Reconstructed shape={tuple(x_hat.shape)}")
