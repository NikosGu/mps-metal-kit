"""Prefix-sum ragged segment linear operators with single-encoder multi-wave MPSStream fusion."""
from __future__ import annotations

import torch

from .compiler import is_mps_available, load_extension


def _pack_wave_metadata(intervals: list[tuple[int, int, int]], dev: torch.device):
    """Pack active `(row_idx, start, stop)` triples into prefix-sum Metal buffers (`1` threadgroup = `1` active token)."""
    act_rows = []
    starts = []
    stops = []
    offs = [0]
    running = 0
    for b, s0, s1 in intervals:
        if s1 > s0:
            act_rows.append(int(b))
            starts.append(int(s0))
            stops.append(int(s1))
            running += int(s1 - s0)
            offs.append(running)
    if not act_rows:
        z = torch.zeros((0,), dtype=torch.int32, device=dev)
        z1 = torch.zeros((1,), dtype=torch.int32, device=dev)
        return z, z1, z, z, 0
    meta_cpu = torch.tensor(act_rows + offs + starts + stops, dtype=torch.int32)
    meta_dev = meta_cpu.to(dev)
    n = len(act_rows)
    return (
        meta_dev[:n].contiguous(),
        meta_dev[n : 2 * n + 1].contiguous(),
        meta_dev[2 * n + 1 : 3 * n + 1].contiguous(),
        meta_dev[3 * n + 1 :].contiguous(),
        running,
    )


class _RaggedWavesLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wave_intervals, x: torch.Tensor, *wave_weights: torch.Tensor):
        ext = load_extension()
        orig_dt = x.dtype
        B, T, _ = x.shape
        D_out = wave_weights[0].shape[1]
        dev = x.device

        Xc = x.detach().float().contiguous()
        Ws = [w.detach().float().contiguous() for w in wave_weights]
        Y = torch.zeros((B, T, D_out), device=dev, dtype=torch.float32)

        act_list, off_list, st_list, sp_list, tot_list = [], [], [], [], []
        for ivs in wave_intervals:
            act_t, off_t, st_t, sp_t, tot = _pack_wave_metadata(ivs, dev)
            act_list.append(act_t)
            off_list.append(off_t)
            st_list.append(st_t)
            sp_list.append(sp_t)
            tot_list.append(tot)

        ext.ragged_waves_mm_fwd(Xc, Ws, act_list, off_list, st_list, tot_list, Y)

        ctx.orig_dt = orig_dt
        ctx.meta = (act_list, off_list, st_list, sp_list, tot_list)
        ctx.save_for_backward(Xc, *Ws)
        return Y.to(orig_dt) if orig_dt != torch.float32 else Y

    @staticmethod
    def backward(ctx, gY: torch.Tensor):
        ext = load_extension()
        saved = ctx.saved_tensors
        Xc = saved[0]
        Ws = list(saved[1:])
        act_list, off_list, st_list, sp_list, tot_list = ctx.meta

        GY = gY.float().contiguous()
        GX = torch.zeros_like(Xc)
        GWs = [torch.zeros_like(w) for w in Ws]

        ext.ragged_waves_mm_bwd(GY, Xc, Ws, act_list, off_list, st_list, sp_list, tot_list, GX, GWs)

        dt = ctx.orig_dt
        if dt != torch.float32:
            return (None, GX.to(dt), *(gw.to(dt) for gw in GWs))
        return (None, GX, *GWs)


def reference_ragged_waves_linear(
    x: torch.Tensor,
    wave_weights: list[torch.Tensor],
    wave_intervals: list[list[tuple[int, int, int]]],
) -> torch.Tensor:
    """Pure PyTorch reference for multi-wave ragged segment linear projections."""
    B, T, _ = x.shape
    D_out = wave_weights[0].shape[1]
    y = x.new_zeros((B, T, D_out))
    for w_mat, ivs in zip(wave_weights, wave_intervals):
        for b, s0, s1 in ivs:
            if s1 > s0:
                y = y.clone()
                y[b, s0:s1] = torch.matmul(x[b, s0:s1], w_mat[b].t())
    return y


def ragged_waves_linear(
    x: torch.Tensor,
    wave_weights: list[torch.Tensor],
    wave_intervals: list[list[tuple[int, int, int]]],
) -> torch.Tensor:
    """Execute `len(wave_weights)` ragged variable-length segment projections in ONE single Metal command encoder.

    Args:
        x: Input sequence tensor `[B, T, D_in]` (`D_in <= 256`).
        wave_weights: List of per-wave batched weight matrices `[B, D_out, D_in]` (`D_out <= 256`).
        wave_intervals: Per-wave list of `(batch_row, start_tok, stop_tok)` non-overlapping active intervals.

    Returns:
        Output tensor `Y [B, T, D_out]` with exact active-token projections and zeros on untouched positions.
    """
    if not wave_weights:
        return x.new_zeros((x.shape[0], x.shape[1], 0))
    D_in = x.shape[-1]
    D_out = wave_weights[0].shape[1]
    if x.device.type == "mps" and is_mps_available() and D_in <= 256 and D_out <= 256:
        return _RaggedWavesLinearFn.apply(wave_intervals, x, *wave_weights)
    return reference_ragged_waves_linear(x, wave_weights, wave_intervals)
