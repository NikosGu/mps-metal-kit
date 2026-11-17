"""Numerical parity tests for associative & DeltaNet scan operators."""
from __future__ import annotations

import pytest
import torch

import mps_metal_kit as mmk

DEV = torch.device("cuda" if mmk.is_cuda_available() else ("mps" if mmk.is_mps_available() else "cpu"))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.mark.skipif(not mmk.is_mps_available(), reason="Apple Silicon MPS device required")
def test_extension_loads_and_registers_modules():
    ext = mmk.load_extension()
    assert hasattr(ext, "deltanet_chunk_scan_fwd")
    assert hasattr(ext, "token_delta_scan_fwd")


@pytest.mark.parametrize("dk,dv,n_chunks", [(64, 128, 16), (32, 64, 24), (48, 96, 11)])
def test_deltanet_chunk_scan_fwd_bwd_parity(dk: int, dv: int, n_chunks: int):
    torch.manual_seed(101)
    B, H = 2, 4
    M = (torch.eye(dk, device=DEV)[None, None, None] * 0.85 + 0.03 * torch.randn(B, H, n_chunks, dk, dk, device=DEV)).requires_grad_(True)
    B_in = (0.1 * torch.randn(B, H, n_chunks, dk, dv, device=DEV)).requires_grad_(True)
    dec = torch.sigmoid(1.5 + 0.5 * torch.randn(B, H, n_chunks, device=DEV)).requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, dk, dv, device=DEV)).requires_grad_(True)

    M_ref = M.detach().clone().requires_grad_(True)
    B_ref = B_in.detach().clone().requires_grad_(True)
    dec_ref = dec.detach().clone().requires_grad_(True)
    S0_ref = S0.detach().clone().requires_grad_(True)

    w_hist = torch.randn(B, H, n_chunks, dk, dv, device=DEV)
    w_end = torch.randn(B, H, dk, dv, device=DEV)

    hist_k, end_k = mmk.deltanet_chunk_scan(M, B_in, dec, S0)
    ((hist_k * w_hist).sum() + (end_k * w_end).sum()).backward()

    hist_r, end_r = mmk.reference_deltanet_chunk_scan(M_ref, B_ref, dec_ref, S0_ref)
    ((hist_r * w_hist).sum() + (end_r * w_end).sum()).backward()

    assert (hist_k - hist_r).abs().max().item() < 2e-5
    assert (end_k - end_r).abs().max().item() < 2e-5
    assert (M.grad - M_ref.grad).abs().max().item() < 5e-5
    assert (B_in.grad - B_ref.grad).abs().max().item() < 5e-5
    assert (dec.grad - dec_ref.grad).abs().max().item() < 5e-5
    assert (S0.grad - S0_ref.grad).abs().max().item() < 5e-5


@pytest.mark.parametrize("dk", [64, 128])
def test_token_delta_scan_fwd_bwd_parity(dk: int):
    torch.manual_seed(202)
    B, H, T, DV = 2, 3, 73, 64
    q = torch.nn.functional.normalize(torch.randn(B, H, T, dk, device=DEV), dim=-1).detach().requires_grad_(True)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, dk, device=DEV), dim=-1).detach().requires_grad_(True)
    v = (0.25 * torch.randn(B, H, T, DV, device=DEV)).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV) - 1.0).detach().requires_grad_(True)
    gamma = torch.sigmoid(torch.randn(B, H, T, device=DEV) + 2.0).detach().requires_grad_(True)
    S0 = (0.1 * torch.randn(B, H, DV, dk, device=DEV)).requires_grad_(True)

    q_r = q.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)
    b_r = beta.detach().clone().requires_grad_(True)
    g_r = gamma.detach().clone().requires_grad_(True)
    s0_r = S0.detach().clone().requires_grad_(True)

    go = torch.randn(B, H, T, DV, device=DEV)
    gs = torch.randn(B, H, DV, dk, device=DEV)

    out_k, st_k = mmk.token_delta_scan(q, k, v, beta, gamma, S0)
    ((out_k * go).sum() + (st_k * gs).sum()).backward()

    out_r, st_r = mmk.reference_token_delta_scan(q_r, k_r, v_r, b_r, g_r, s0_r)
    ((out_r * go).sum() + (st_r * gs).sum()).backward()

    assert (out_k - out_r).abs().max().item() < 2e-5
    assert (st_k - st_r).abs().max().item() < 2e-5
    assert (q.grad - q_r.grad).abs().max().item() < 5e-5
    assert (k.grad - k_r.grad).abs().max().item() < 5e-5
    assert (v.grad - v_r.grad).abs().max().item() < 5e-5
    assert (beta.grad - b_r.grad).abs().max().item() < 5e-5
    assert (gamma.grad - g_r.grad).abs().max().item() < 5e-5
    assert (S0.grad - s0_r.grad).abs().max().item() < 5e-5


def test_chunked_delta_rule_matches_token_recurrence():
    torch.manual_seed(303)
    B, H, T, DK, DV = 2, 2, 100, 64, 64
    q = torch.nn.functional.normalize(torch.randn(B, H, T, DK, device=DEV), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, DK, device=DEV), dim=-1)
    v = 0.2 * torch.randn(B, H, T, DV, device=DEV)
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV) - 1.2)
    S0 = 0.05 * torch.randn(B, H, DK, DV, device=DEV)

    out_chunk, st_chunk = mmk.chunked_delta_rule(q, k, v, beta, chunk_size=32, S0=S0)
    out_tok, st_tok_T = mmk.token_delta_scan(q, k, v, beta, gamma=None, S0=S0.transpose(-1, -2))

    assert (out_chunk - out_tok).abs().max().item() < 3e-5
    assert (st_chunk - st_tok_T.transpose(-1, -2)).abs().max().item() < 3e-5
