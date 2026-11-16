"""Example: Drop-in O(T) Gated DeltaNet Attention Layer for Apple Silicon MPS.

Replaces O(T^2) softmax KV-cache growth with a constant-size [H, 64, DK] fast-weight state
per layer, running 36x faster than pure PyTorch on Apple Silicon GPUs via `mps-metal-kit`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import mps_metal_kit as mmk


class GatedDeltaNetAttention(nn.Module):
    """Multi-head causal Gated DeltaNet sequence mixer with constant-memory recurrent state."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_key: int = 64):
        super().__init__()
        assert d_key in (64, 128), "Metal register fast path supports d_key in {64, 128}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_val = 64

        self.q_proj = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_val, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(n_heads * self.d_val, d_model, bias=False)

        nn.init.constant_(self.beta_proj.bias, -1.5)
        nn.init.constant_(self.gamma_proj.bias, 2.5)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, DK, DV = self.n_heads, self.d_key, self.d_val

        q = F.normalize(self.q_proj(x).view(B, T, H, DK).transpose(1, 2), dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, H, DK).transpose(1, 2), dim=-1)
        v = self.v_proj(x).view(B, T, H, DV).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        gamma = torch.sigmoid(self.gamma_proj(x)).transpose(1, 2)

        y_heads, next_state = mmk.token_delta_scan(q, k, v, beta, gamma, S0=state)
        y = self.out_proj(y_heads.transpose(1, 2).reshape(B, T, H * DV))
        return x + y, next_state


if __name__ == "__main__":
    dev = torch.device("mps" if mmk.is_mps_available() else "cpu")
    layer = GatedDeltaNetAttention(d_model=512, n_heads=8, d_key=128).to(dev)

    # Process two consecutive streaming chunks of 512 tokens with O(1) state memory
    chunk1 = torch.randn(2, 512, 512, device=dev, requires_grad=True)
    chunk2 = torch.randn(2, 512, 512, device=dev, requires_grad=True)

    out1, st = layer(chunk1)
    out2, st = layer(chunk2, state=st)
    loss = out2.pow(2).mean()
    loss.backward()
    print(f"Success on {dev}! Output shape={tuple(out2.shape)}, state shape={tuple(st.shape)}")
