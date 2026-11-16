"""Numerical parity tests for zero-copy AMX & Metal spectral operators."""
from __future__ import annotations

import pytest
import torch

import mps_metal_kit as mmk

DEV = torch.device("mps" if mmk.is_mps_available() else "cpu")


def test_unified_eigh_parity():
    torch.manual_seed(404)
    A = torch.randn(4, 48, 48, device=DEV)
    G = (torch.bmm(A, A.transpose(1, 2)) * 0.05).detach().requires_grad_(True)
    G_ref = G.detach().clone().requires_grad_(True)

    W_probe = torch.randn(4, 48, 48, device=DEV)
    W_sym = 0.5 * (W_probe + W_probe.transpose(1, 2))

    L_k, V_k = mmk.unified_eigh(G)
    quad_k = (V_k * torch.bmm(W_sym, V_k)).sum(dim=1)
    (torch.sin(L_k) * quad_k).sum().backward()

    G_cpu = G_ref.cpu().double().detach().requires_grad_(True)
    L_r, V_r = torch.linalg.eigh(0.5 * (G_cpu + G_cpu.transpose(1, 2)))
    W_cpu = W_sym.cpu().double()
    quad_r = (V_r * torch.bmm(W_cpu, V_r)).sum(dim=1)
    (torch.sin(L_r) * quad_r).sum().backward()

    assert (L_k.cpu() - L_r.float()).abs().max().item() < 2e-5
    assert (G.grad.cpu() - G_cpu.grad.float()).abs().max().item() < 2e-4


@pytest.mark.parametrize("m_len,d_dim", [(24, 128), (160, 128)])
def test_weighted_gram_subspace_dual_and_primal_parity(m_len: int, d_dim: int):
    torch.manual_seed(505 + m_len)
    B, rank = 4, 6
    X = (0.3 * torch.randn(B, m_len, d_dim, device=DEV)).requires_grad_(True)
    raw_w = torch.rand(B, m_len, device=DEV) + 0.1
    raw_w[3] = 0.0
    W = raw_w.detach().requires_grad_(True)
    lengths = [m_len, max(4, m_len - 7), max(3, m_len // 2), m_len]

    X_ref = X.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    probe = torch.randn(B, d_dim, d_dim, device=DEV)

    U_k = mmk.weighted_gram_subspace(X, W, rank=rank, lengths=lengths)
    P_k = torch.bmm(U_k, U_k.transpose(1, 2))
    (P_k * probe).sum().backward()

    U_r = mmk.reference_weighted_gram_subspace(X_ref, W_ref, rank=rank, lengths=lengths)
    P_r = torch.bmm(U_r, U_r.transpose(1, 2))
    (P_r * probe).sum().backward()

    assert (P_k - P_r).abs().max().item() < 2e-5
    assert (X.grad - X_ref.grad).abs().max().item() < 5e-5
    assert (W.grad - W_ref.grad).abs().max().item() < 5e-5
