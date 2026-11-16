"""Zero-copy Apple AMX `Accelerate` spectral operators with Metal GPU eigenvector backward."""
from __future__ import annotations

import torch

from .compiler import is_mps_available, load_extension

EIGH_GRAD_CLIP = 100.0


def _safe_cpu_eigh(G64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Gs = 0.5 * (G64 + G64.transpose(-1, -2))
    try:
        return torch.linalg.eigh(Gs)
    except RuntimeError:
        D = Gs.shape[-1]
        flat = Gs.reshape(-1, D, D)
        eye = torch.eye(D, dtype=Gs.dtype, device=Gs.device)
        Ls, Vs = [], []
        for i in range(flat.shape[0]):
            Ai = flat[i]
            try:
                li, vi = torch.linalg.eigh(Ai)
            except RuntimeError:
                dg = Ai.diagonal()
                c = max(0.0, -float((dg - (Ai.abs().sum(-1) - dg.abs())).min())) + 1.0
                u, _, _ = torch.linalg.svd(Ai + c * eye)
                vi = u.flip(-1)
                li = (vi * (Ai @ vi)).sum(-2)
            Ls.append(li)
            Vs.append(vi)
        return torch.stack(Ls).reshape(Gs.shape[:-1]), torch.stack(Vs).reshape(Gs.shape)


class _UnifiedEighFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G: torch.Tensor):
        ext = load_extension()
        orig_shape = G.shape
        orig_dt = G.dtype
        D = orig_shape[-1]
        Gf = G.detach().float().reshape(-1, D, D).contiguous()
        B = Gf.shape[0]
        L_out = torch.empty((B, D), device=G.device, dtype=torch.float32)
        V_out = torch.empty((B, D, D), device=G.device, dtype=torch.float32)
        ext.amx_eigh_fwd(Gf, L_out, V_out)
        ctx.save_for_backward(L_out, V_out)
        ctx.orig_shape = orig_shape
        ctx.orig_dt = orig_dt
        L_res = L_out.view(*orig_shape[:-1])
        V_res = V_out.view(*orig_shape)
        if orig_dt != torch.float32:
            return L_res.to(orig_dt), V_res.to(orig_dt)
        return L_res, V_res

    @staticmethod
    def backward(ctx, gL: torch.Tensor | None, gV: torch.Tensor | None):
        L, V = ctx.saved_tensors
        B, D = L.shape
        gL_f = gL.float().reshape(B, D) if gL is not None else torch.zeros_like(L)
        if gV is not None:
            gV_f = gV.float().reshape(B, D, D)
            vt_gv = torch.bmm(V.transpose(1, 2), gV_f)
            skew = 0.5 * (vt_gv - vt_gv.transpose(1, 2))
            dl = L[:, None, :] - L[:, :, None]
            off = ~torch.eye(D, dtype=torch.bool, device=L.device).unsqueeze(0)
            inv_dl = torch.where(off, dl / (dl * dl + 1e-20), torch.zeros_like(dl))
            mid = torch.diag_embed(gL_f) + inv_dl * skew
        else:
            mid = torch.diag_embed(gL_f)
        gG = torch.bmm(torch.bmm(V, mid), V.transpose(1, 2))
        gG = 0.5 * (gG + gG.transpose(1, 2)).view(*ctx.orig_shape)
        return gG.to(ctx.orig_dt) if ctx.orig_dt != torch.float32 else gG


class _RefProjSubspaceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G_in: torch.Tensor, r: int, rel_tol: float, min_eig: float):
        D = G_in.shape[-1]
        L, V = _safe_cpu_eigh(G_in)
        top = L[..., -1:]
        active = (L > rel_tol * top) & (top > min_eig)
        top_mask = torch.zeros_like(active)
        top_mask[..., D - r :] = True
        keep = (active & top_mask).to(torch.float64)
        U_asc = V[..., D - r :] * keep[..., None, D - r :]
        ctx.save_for_backward(V, L, keep)
        ctx.r = r
        return U_asc.flip(-1)

    @staticmethod
    def backward(ctx, gU_desc: torch.Tensor):
        V, L, keep = ctx.saved_tensors
        r = ctx.r
        D = L.shape[-1]
        gU_asc = gU_desc.flip(-1)
        kf_top = keep[..., D - r :]
        gU_eff = gU_asc * kf_top[..., None, :]
        C_dk = 0.5 * torch.matmul(V.transpose(-1, -2), gU_eff)
        dropped = (1.0 - keep)[..., :, None]
        kept = kf_top[..., None, :]
        across = (dropped * kept) > 0.5
        dl = torch.where(across, L[..., None, D - r :] - L[..., :, None], torch.ones_like(C_dk))
        inv_dl = (dl / (dl * dl + 1e-12)).clamp(-EIGH_GRAD_CLIP, EIGH_GRAD_CLIP)
        W_dk = torch.where(across, inv_dl * C_dk, torch.zeros_like(C_dk))
        V_top = V[..., D - r :] * kf_top[..., None, :]
        H = torch.matmul(V, W_dk) @ V_top.transpose(-1, -2)
        return H + H.transpose(-1, -2), None, None, None


class _WeightedGramSubspaceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        weights: torch.Tensor,
        lengths_i32: torch.Tensor,
        rank: int,
        rel_tol: float,
        min_eig: float,
    ):
        ext = load_extension()
        orig_dt = X.dtype
        B, M, D = X.shape
        r = min(int(rank), D)
        k_pad = max(min(M, D), r, 1)

        Xc = X.detach().float().contiguous()
        Wc = weights.detach().float().contiguous()
        Lc = lengths_i32.to(device=X.device, dtype=torch.int32).contiguous()

        U_out = torch.empty((B, D, r), device=X.device, dtype=torch.float32)
        PACK_out = torch.empty((B, D + 3, k_pad), device=X.device, dtype=torch.float32)
        r_eff = ext.amx_dual_gram_subspace_fwd(Xc, Wc, Lc, U_out, PACK_out, r, k_pad, float(rel_tol), float(min_eig))

        ctx.orig_dt = orig_dt
        ctx.r_eff = int(r_eff)
        ctx.save_for_backward(Xc, Wc, Lc, PACK_out)
        return U_out.to(orig_dt) if orig_dt != torch.float32 else U_out

    @staticmethod
    def backward(ctx, gU: torch.Tensor):
        ext = load_extension()
        Xc, Wc, Lc, PACK = ctx.saved_tensors
        GU = gU.float().contiguous()
        GX = torch.zeros_like(Xc)
        GW = torch.zeros_like(Wc)
        if ctx.r_eff > 0:
            ext.spectral_subspace_bwd(Xc, Wc, Lc, PACK, GU, GX, GW, ctx.r_eff)
        dt = ctx.orig_dt
        if dt != torch.float32:
            return GX.to(dt), GW.to(dt), None, None, None, None
        return GX, GW, None, None, None, None


def reference_eigh(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference symmetric eigendecomposition via CPU float64 conversion."""
    L64, V64 = _safe_cpu_eigh(G.cpu().to(torch.float64))
    return L64.to(dtype=G.dtype).to(G.device), V64.to(dtype=G.dtype).to(G.device)


def unified_eigh(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-copy Apple AMX `float64` symmetric eigendecomposition for PyTorch MPS tensors."""
    if G.device.type == "mps" and is_mps_available() and G.dtype != torch.float64:
        return _UnifiedEighFn.apply(G)
    return torch.linalg.eigh(0.5 * (G + G.transpose(-1, -2)))


def reference_weighted_gram_subspace(
    X: torch.Tensor,
    weights: torch.Tensor,
    rank: int = 4,
    lengths: list[int] | torch.Tensor | None = None,
    rel_tol: float = 1e-5,
    min_eig: float = 1e-6,
) -> torch.Tensor:
    """Exact float64 reference for top-`rank` principal eigenvectors `U [B, D, rank]`."""
    B, M, D = X.shape
    r = min(int(rank), D)
    if lengths is None:
        lens_list = [M] * B
    elif isinstance(lengths, torch.Tensor):
        lens_list = lengths.detach().cpu().tolist()
    else:
        lens_list = [int(v) for v in lengths]

    X64 = X.cpu().to(torch.float64)
    W64 = weights.cpu().to(torch.float64)
    mask = torch.arange(M)[None, :] < torch.tensor(lens_list, dtype=torch.long)[:, None]
    W_eff = torch.where(mask, W64.clamp(min=0.0), torch.zeros_like(W64))
    G = torch.einsum("bm,bmd,bme->bde", W_eff, X64, X64)
    U64 = _RefProjSubspaceFn.apply(G, r, rel_tol, min_eig)
    return U64.to(dtype=X.dtype).to(X.device)


def weighted_gram_subspace(
    X: torch.Tensor,
    weights: torch.Tensor,
    rank: int = 4,
    lengths: list[int] | torch.Tensor | None = None,
    rel_tol: float = 1e-5,
    min_eig: float = 1e-6,
) -> torch.Tensor:
    """Extract top-`rank` principal orthonormal eigenvectors `U [B, D, rank]` of `G = sum_t w_t x_t x_t^T`."""
    B, M, D = X.shape
    r = min(int(rank), D)
    if lengths is None:
        lens_t = torch.full((B,), M, dtype=torch.int32, device=X.device)
    elif isinstance(lengths, torch.Tensor):
        lens_t = lengths.to(device=X.device, dtype=torch.int32)
    else:
        lens_t = torch.tensor(lengths, dtype=torch.int32, device=X.device)

    if (
        X.device.type == "mps"
        and is_mps_available()
        and X.dtype != torch.float64
        and D <= 256
        and r <= 12
    ):
        return _WeightedGramSubspaceFn.apply(X, weights, lens_t, r, rel_tol, min_eig)
    return reference_weighted_gram_subspace(X, weights, r, lengths=lengths, rel_tol=rel_tol, min_eig=min_eig)
