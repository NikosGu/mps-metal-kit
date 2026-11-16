"""Register-resident matrix-state associative and DeltaNet scans on Apple Silicon Metal."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compiler import is_mps_available, load_extension


class _DeltaNetChunkScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M: torch.Tensor, B_in: torch.Tensor, decay: torch.Tensor, S0: torch.Tensor):
        ext = load_extension()
        orig_dt = B_in.dtype
        B, H, N, DK, DV = B_in.shape
        BH = B * H

        Mc = M.detach().float().contiguous()
        Bc = B_in.detach().float().contiguous()
        Dc = decay.detach().float().contiguous()
        S0c = S0.detach().float().contiguous()

        St_hist = torch.empty((B, H, N, DK, DV), device=B_in.device, dtype=torch.float32)
        St_end = torch.empty((B, H, DK, DV), device=B_in.device, dtype=torch.float32)
        ext.deltanet_chunk_scan_fwd(Mc, Bc, Dc, S0c, St_hist, St_end, BH, N, DK, DV)

        ctx.orig_dt = orig_dt
        ctx.shapes = (B, H, N, DK, DV)
        ctx.save_for_backward(Mc, Dc, St_hist)
        if orig_dt != torch.float32:
            return St_hist.to(orig_dt), St_end.to(orig_dt)
        return St_hist, St_end

    @staticmethod
    def backward(ctx, g_hist: torch.Tensor | None, g_end: torch.Tensor | None):
        ext = load_extension()
        Mc, Dc, St_hist = ctx.saved_tensors
        B, H, N, DK, DV = ctx.shapes
        BH = B * H
        dev = Mc.device
        f = dict(device=dev, dtype=torch.float32)

        gh = g_hist.float().contiguous() if g_hist is not None else torch.zeros((B, H, N, DK, DV), **f)
        ge = g_end.float().contiguous() if g_end is not None else torch.zeros((B, H, DK, DV), **f)

        GM = torch.empty_like(Mc)
        GB = torch.empty_like(St_hist)
        GDEC = torch.empty_like(Dc)
        GS0 = torch.empty((B, H, DK, DV), **f)
        ext.deltanet_chunk_scan_bwd(Mc, Dc, St_hist, gh, ge, GM, GB, GDEC, GS0, BH, N, DK, DV)

        dt = ctx.orig_dt
        if dt != torch.float32:
            return GM.to(dt), GB.to(dt), GDEC.to(dt), GS0.to(dt)
        return GM, GB, GDEC, GS0


def reference_deltanet_chunk_scan(
    M: torch.Tensor,
    B_in: torch.Tensor,
    decay: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for `S_{n+1} = decay_n * (M_n @ S_n) + B_n`."""
    B, H, N, DK, DV = B_in.shape
    if decay is None:
        decay = B_in.new_ones((B, H, N))
    if S0 is None:
        S0 = B_in.new_zeros((B, H, DK, DV))
    S = S0
    hist = []
    for n in range(N):
        hist.append(S)
        dec_n = decay[:, :, n, None, None]
        S = dec_n * torch.matmul(M[:, :, n], S) + B_in[:, :, n]
    St_hist = torch.stack(hist, dim=2) if N > 0 else B_in.new_zeros((B, H, 0, DK, DV))
    return St_hist, S


def deltanet_chunk_scan(
    M: torch.Tensor,
    B_in: torch.Tensor,
    decay: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register-resident chunkwise matrix transition scan on MPS (`S_{n+1} = decay_n * (M_n @ S_n) + B_n`).

    Args:
        M: Transition matrices of shape `[B, H, N, DK, DK]` (`1 <= DK <= 64`).
        B_in: Additive chunk writes of shape `[B, H, N, DK, DV]` (`1 <= DV <= 128`).
        decay: Optional scalar chunk retention gates `[B, H, N]` (defaults to ones).
        S0: Optional initial state `[B, H, DK, DV]` (defaults to zeros).

    Returns:
        `(St_hist, St_end)` where `St_hist[:, :, n]` is the pre-chunk state `S_n` (`[B, H, N, DK, DV]`)
        and `St_end` is the final state `S_N` (`[B, H, DK, DV]`).
    """
    B, H, N, DK, DV = B_in.shape
    if decay is None:
        decay = B_in.new_ones((B, H, N))
    if S0 is None:
        S0 = B_in.new_zeros((B, H, DK, DV))

    if B_in.device.type == "mps" and is_mps_available() and 1 <= DK <= 64 and 1 <= DV <= 128 and N > 0:
        return _DeltaNetChunkScanFn.apply(M, B_in, decay, S0)
    return reference_deltanet_chunk_scan(M, B_in, decay, S0)


class _TokenDeltaScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        S0: torch.Tensor,
    ):
        ext = load_extension()
        orig_dt = q.dtype
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        BH = B * H
        nck = (T + 15) // 16 + 1

        Qc = q.detach().float().contiguous()
        Kc = k.detach().float().contiguous()
        Vc = v.detach().float().contiguous()
        Bc = beta.detach().float().contiguous()
        Gc = gamma.detach().float().contiguous()
        S0c = S0.detach().float().contiguous()

        f = dict(device=q.device, dtype=torch.float32)
        OUT = torch.empty((B, H, T, DV), **f)
        CKPT = torch.empty((BH, nck, DV * DK), **f)
        ST_END = torch.empty((B, H, DV, DK), **f)

        ext.token_delta_scan_fwd(Qc, Kc, Vc, Bc, Gc, S0c, OUT, CKPT, ST_END, BH, T, DK, nck)

        ctx.orig_dt = orig_dt
        ctx.shapes = (B, H, T, DK, DV, nck)
        ctx.save_for_backward(Qc, Kc, Vc, Bc, Gc, CKPT)
        if orig_dt != torch.float32:
            return OUT.to(orig_dt), ST_END.to(orig_dt)
        return OUT, ST_END

    @staticmethod
    def backward(ctx, g_out: torch.Tensor | None, g_st_end: torch.Tensor | None):
        ext = load_extension()
        Qc, Kc, Vc, Bc, Gc, CKPT = ctx.saved_tensors
        B, H, T, DK, DV, nck = ctx.shapes
        BH = B * H
        f = dict(device=Qc.device, dtype=torch.float32)

        GO = g_out.float().contiguous() if g_out is not None else torch.zeros((B, H, T, DV), **f)
        GSE = g_st_end.float().contiguous() if g_st_end is not None else torch.zeros((B, H, DV, DK), **f)
        SCR = torch.empty((BH, 16, DV * DK), **f)

        GQ = torch.empty_like(Qc)
        GK = torch.empty_like(Kc)
        GV = torch.empty_like(Vc)
        GB = torch.empty_like(Bc)
        GG = torch.empty_like(Gc)
        GS0 = torch.empty((B, H, DV, DK), **f)

        ext.token_delta_scan_bwd(Qc, Kc, Vc, Bc, Gc, CKPT, GO, GSE, SCR, GQ, GK, GV, GB, GG, GS0, BH, T, DK, nck)

        dt = ctx.orig_dt
        if dt != torch.float32:
            return GQ.to(dt), GK.to(dt), GV.to(dt), GB.to(dt), GG.to(dt), GS0.to(dt)
        return GQ, GK, GV, GB, GG, GS0


def reference_token_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact token-by-token PyTorch reference for gated fast-weight / DeltaNet scan:
    `e_t = v_t - S_{t-1} k_t`, `S_t = gamma_t S_{t-1} + beta_t e_t k_t^T`, `o_t = S_t q_t`."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    if gamma is None:
        gamma = q.new_ones((B, H, T))
    if S0 is None:
        S0 = q.new_zeros((B, H, DV, DK))
    S = S0
    outs = []
    for t in range(T):
        kt = k[:, :, t]
        qt = q[:, :, t]
        vt = v[:, :, t]
        bt = beta[:, :, t, None]
        gt = gamma[:, :, t, None, None]
        v_hat = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)
        err = vt - v_hat
        S = gt * S + (bt * err).unsqueeze(-1) * kt.unsqueeze(-2)
        ot = torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1)
        outs.append(ot)
    out = torch.stack(outs, dim=2) if T > 0 else q.new_zeros((B, H, 0, DV))
    return out, S


def token_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register-resident token-level gated DeltaNet scan (`DK in {64, 128}`, `DV = 64`) with `CKI=16` checkpointing."""
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    if gamma is None:
        gamma = q.new_ones((B, H, T))
    if S0 is None:
        S0 = q.new_zeros((B, H, DV, DK))

    if q.device.type == "mps" and is_mps_available() and DK in (64, 128) and DV == 64 and T > 0:
        return _TokenDeltaScanFn.apply(q, k, v, beta, gamma, S0)
    return reference_token_delta_scan(q, k, v, beta, gamma, S0)


def chunked_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WY-factorized chunkwise causal DeltaNet (`S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`, `o_t = S_t q_t`).

    Combines batched intra-chunk GEMMs with `deltanet_chunk_scan` across chunks so `N = T // chunk_size`
    chunk state transitions execute in a single register-resident Metal kernel launch.
    """
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    C = int(chunk_size)
    if S0 is None:
        S0 = q.new_zeros((B, H, DK, DV))

    pad_t = (C - (T % C)) % C
    if pad_t > 0:
        q = F.pad(q, (0, 0, 0, pad_t))
        k = F.pad(k, (0, 0, 0, pad_t))
        v = F.pad(v, (0, 0, 0, pad_t))
        beta = F.pad(beta, (0, pad_t))

    Tp = q.shape[2]
    N = Tp // C
    qc = q.view(B, H, N, C, DK)
    kc = k.view(B, H, N, C, DK)
    vc = v.view(B, H, N, C, DV)
    bc = beta.view(B, H, N, C)

    kb = kc * bc.unsqueeze(-1)
    vb = vc * bc.unsqueeze(-1)

    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    strict_tril = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool), diagonal=-1)
    kk = torch.matmul(kb.float(), kc.float().transpose(-1, -2))
    A = eye + torch.where(strict_tril, kk, torch.zeros_like(kk))
    T_inv = torch.linalg.solve_triangular(A, eye.expand_as(A), upper=False).to(q.dtype)

    W = torch.matmul(T_inv, kb)      # [B, H, N, C, DK]
    U = torch.matmul(T_inv, vb)      # [B, H, N, C, DV]

    eye_dk = torch.eye(DK, device=q.device, dtype=q.dtype)
    M_chunk = eye_dk - torch.matmul(kc.transpose(-1, -2), W)   # [B, H, N, DK, DK]
    B_chunk = torch.matmul(kc.transpose(-1, -2), U)            # [B, H, N, DK, DV]

    St_hist, St_end = deltanet_chunk_scan(M_chunk, B_chunk, decay=None, S0=S0)

    U_eff = U - torch.matmul(W, St_hist)                       # [B, H, N, C, DV]
    causal = torch.tril(torch.ones((C, C), device=q.device, dtype=torch.bool))
    qk = torch.where(causal, torch.matmul(qc, kc.transpose(-1, -2)), torch.zeros((C, C), device=q.device, dtype=q.dtype))
    out_c = torch.matmul(qc, St_hist) + torch.matmul(qk, U_eff)
    out = out_c.reshape(B, H, Tp, DV)[:, :, :T, :]
    return out, St_end
