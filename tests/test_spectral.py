"""Numerical parity tests for zero-copy AMX & Metal spectral operators."""
from __future__ import annotations

import pytest
import torch

import mps_metal_kit as mmk

DEV = torch.device("cuda" if mmk.is_cuda_available() else ("mps" if mmk.is_mps_available() else "cpu"))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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


def test_degenerate_collinear_and_zero_length_stability():
    torch.manual_seed(909)
    B, M, D, rank = 4, 64, 128, 6
    X_raw = 0.25 * torch.randn(B, M, D, device=DEV)
    # Row 0: repeated identical leading/trailing vectors (collinear rank-deficient Householder/QL stress test)
    X_raw[0, :20] = X_raw[0, 0:1].clone()
    X_raw[0, 20:40] = X_raw[0, 20:21].clone() + 1e-7 * torch.randn(20, D, device=DEV)
    # Row 1: zero-weighted prefix followed by low-rank span
    W_raw = torch.rand(B, M, device=DEV) + 0.1
    W_raw[1, :16] = 0.0
    lengths = [64, 52, 0, 12]

    X = X_raw.detach().requires_grad_(True)
    W = W_raw.detach().requires_grad_(True)
    probe = torch.randn(B, D, D, device=DEV)

    U_k = mmk.weighted_gram_subspace(X, W, rank=rank, lengths=lengths)
    P_k = torch.bmm(U_k, U_k.transpose(1, 2))
    (P_k * probe).sum().backward()

    assert torch.isfinite(U_k).all()
    assert torch.isfinite(X.grad).all()
    assert torch.isfinite(W.grad).all()
    assert U_k[2].abs().max().item() == 0.0
