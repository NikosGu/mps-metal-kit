"""Numerical parity tests for single-encoder ragged multi-wave linear operators."""
from __future__ import annotations

import torch

import mps_metal_kit as mmk

DEV = torch.device("mps" if mmk.is_mps_available() else "cpu")


def test_ragged_waves_linear_single_encoder_parity():
    torch.manual_seed(606)
    B, T, D_in, D_out = 6, 128, 128, 96
    n_waves = 12
    x = torch.randn(B, T, D_in, device=DEV, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    weights = [torch.randn(B, D_out, D_in, device=DEV, requires_grad=True) for _ in range(n_waves)]
    weights_ref = [w.detach().clone().requires_grad_(True) for w in weights]

    cursors = [0] * B
    wave_intervals = []
    for w in range(n_waves):
        ivs = []
        for b in range(B):
            if cursors[b] < T and ((b + w) % 5 != 0):
                step = min(T - cursors[b], 5 + ((b * 7 + w * 3) % 13))
                ivs.append((b, cursors[b], cursors[b] + step))
                cursors[b] += step
        wave_intervals.append(ivs)

    gy = torch.randn(B, T, D_out, device=DEV)

    y_k = mmk.ragged_waves_linear(x, weights, wave_intervals)
    (y_k * gy).sum().backward()

    y_r = mmk.reference_ragged_waves_linear(x_ref, weights_ref, wave_intervals)
    (y_r * gy).sum().backward()

    assert (y_k - y_r).abs().max().item() < 2e-5
    assert (x.grad - x_ref.grad).abs().max().item() < 5e-5
    for wk, wr in zip(weights, weights_ref):
        assert (wk.grad - wr.grad).abs().max().item() < 8e-5
