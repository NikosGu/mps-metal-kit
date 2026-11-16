"""Reproducible micro-benchmarks comparing mps-metal-kit against standard PyTorch MPS."""
from __future__ import annotations

import argparse
import platform
import time
import torch

import mps_metal_kit as mmk


def _sync(dev: torch.device):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _time_fwd_bwd(fn, warmup: int, iters: int, dev: torch.device) -> float:
    for _ in range(warmup):
        loss = fn()
        loss.backward()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = fn()
        loss.backward()
    _sync(dev)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser(description="Benchmark mps-metal-kit C++20/Metal/Accelerate kernels")
    ap.add_argument("--quick", action="store_true", help="Run fewer iterations for CI smoke checks")
    args = ap.parse_args()

    dev = torch.device("mps" if mmk.is_mps_available() else "cpu")
    warmup = 2 if args.quick else 5
    iters = 4 if args.quick else 15

    if dev.type == "mps":
        mmk.load_extension()

    print(f"Platform : {platform.platform()} ({platform.machine()})")
    print(f"PyTorch  : {torch.__version__} | Device: {dev}")
    print("-" * 78)
    print(f"{'Operator (Forward + Backward)':<42} | {'PyTorch MPS':>12} | {'mps-metal-kit':>13} | {'Speedup':>7}")
    print("-" * 78)

    # 1. Chunked Matrix-State Delta Scan (B=8, H=8, N=32, DK=64, DV=128)
    B, H, N, DK, DV = 8, 8, 32, 64, 128
    M = (torch.eye(DK, device=dev)[None, None, None] * 0.9 + 0.02 * torch.randn(B, H, N, DK, DK, device=dev)).requires_grad_(True)
    B_in = (0.1 * torch.randn(B, H, N, DK, DV, device=dev)).requires_grad_(True)
    dec = torch.sigmoid(torch.randn(B, H, N, device=dev) + 2.0).requires_grad_(True)
    S0 = torch.zeros(B, H, DK, DV, device=dev, requires_grad=True)

    def run_chunk_ref():
        M.grad = B_in.grad = dec.grad = S0.grad = None
        h, e = mmk.reference_deltanet_chunk_scan(M, B_in, dec, S0)
        return h.sum() + e.sum()

    def run_chunk_kit():
        M.grad = B_in.grad = dec.grad = S0.grad = None
        h, e = mmk.deltanet_chunk_scan(M, B_in, dec, S0)
        return h.sum() + e.sum()

    t_ref1 = _time_fwd_bwd(run_chunk_ref, warmup, iters, dev)
    t_kit1 = _time_fwd_bwd(run_chunk_kit, warmup, iters, dev)
    print(f"{'deltanet_chunk_scan [8x8, N=32, 64x128]':<42} | {t_ref1:>9.2f} ms | {t_kit1:>10.2f} ms | {t_ref1 / max(t_kit1, 1e-6):>6.2f}x")

    # 2. Token-Level Gated Delta Scan (B=4, H=8, T=512, DK=128, DV=64)
    Bt, Ht, Tt, DKt, DVt = 4, 8, 256 if args.quick else 512, 128, 64
    q = torch.nn.functional.normalize(torch.randn(Bt, Ht, Tt, DKt, device=dev), dim=-1).detach().requires_grad_(True)
    k = torch.nn.functional.normalize(torch.randn(Bt, Ht, Tt, DKt, device=dev), dim=-1).detach().requires_grad_(True)
    v = (0.2 * torch.randn(Bt, Ht, Tt, DVt, device=dev)).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(Bt, Ht, Tt, device=dev) - 1.0).detach().requires_grad_(True)
    gamma = torch.sigmoid(torch.randn(Bt, Ht, Tt, device=dev) + 2.0).detach().requires_grad_(True)

    def run_tok_ref():
        q.grad = k.grad = v.grad = beta.grad = gamma.grad = None
        out, st = mmk.reference_token_delta_scan(q, k, v, beta, gamma)
        return out.sum() + st.sum()

    def run_tok_kit():
        q.grad = k.grad = v.grad = beta.grad = gamma.grad = None
        out, st = mmk.token_delta_scan(q, k, v, beta, gamma)
        return out.sum() + st.sum()

    t_ref2 = _time_fwd_bwd(run_tok_ref, 1 if args.quick else 2, 2 if args.quick else 4, dev)
    t_kit2 = _time_fwd_bwd(run_tok_kit, warmup, iters, dev)
    print(f"{f'token_delta_scan [4x8, T={Tt}, 64x128]':<42} | {t_ref2:>9.2f} ms | {t_kit2:>10.2f} ms | {t_ref2 / max(t_kit2, 1e-6):>6.2f}x")

    # 3. Weighted Dual-Gram Spectral Subspace (B=16, M=32, D=192, rank=8)
    Bs, Ms, Ds = 16, 32, 192
    X_sp = (0.3 * torch.randn(Bs, Ms, Ds, device=dev)).requires_grad_(True)
    W_sp = (torch.rand(Bs, Ms, device=dev) + 0.1).detach().requires_grad_(True)
    lens_sp = [20 + (i % 12) for i in range(Bs)]

    def run_spec_ref():
        X_sp.grad = W_sp.grad = None
        U = mmk.reference_weighted_gram_subspace(X_sp, W_sp, rank=8, lengths=lens_sp)
        P = torch.bmm(U, U.transpose(1, 2))
        return P.pow(2).sum()

    def run_spec_kit():
        X_sp.grad = W_sp.grad = None
        U = mmk.weighted_gram_subspace(X_sp, W_sp, rank=8, lengths=lens_sp)
        P = torch.bmm(U, U.transpose(1, 2))
        return P.pow(2).sum()

    t_ref3 = _time_fwd_bwd(run_spec_ref, warmup, iters, dev)
    t_kit3 = _time_fwd_bwd(run_spec_kit, warmup, iters, dev)
    print(f"{'weighted_gram_subspace [B=16, M=32, D=192]':<42} | {t_ref3:>9.2f} ms | {t_kit3:>10.2f} ms | {t_ref3 / max(t_kit3, 1e-6):>6.2f}x")

    # 4. Single-Encoder Multi-Wave Ragged Segment Linear (B=16, T=512, 32 waves, D=192)
    Br, Tr, Dr, n_waves = 16, 512, 192, 32
    xr = torch.randn(Br, Tr, Dr, device=dev, requires_grad=True)
    ws = [torch.randn(Br, Dr, Dr, device=dev, requires_grad=True) for _ in range(n_waves)]
    cursors = [0] * Br
    wave_ivs = []
    for w in range(n_waves):
        ivs = []
        for b in range(Br):
            if cursors[b] < Tr:
                step = min(Tr - cursors[b], 8 + ((b * 5 + w * 3) % 15))
                ivs.append((b, cursors[b], cursors[b] + step))
                cursors[b] += step
        wave_ivs.append(ivs)

    def run_rag_ref():
        xr.grad = None
        for w in ws:
            w.grad = None
        return mmk.reference_ragged_waves_linear(xr, ws, wave_ivs).sum()

    def run_rag_kit():
        xr.grad = None
        for w in ws:
            w.grad = None
        return mmk.ragged_waves_linear(xr, ws, wave_ivs).sum()

    t_ref4 = _time_fwd_bwd(run_rag_ref, 2, 4 if args.quick else 8, dev)
    t_kit4 = _time_fwd_bwd(run_rag_kit, warmup, iters, dev)
    print(f"{'ragged_waves_linear [B=16, 32 waves, D=192]':<42} | {t_ref4:>9.2f} ms | {t_kit4:>10.2f} ms | {t_ref4 / max(t_kit4, 1e-6):>6.2f}x")
    print("-" * 78)


if __name__ == "__main__":
    main()
