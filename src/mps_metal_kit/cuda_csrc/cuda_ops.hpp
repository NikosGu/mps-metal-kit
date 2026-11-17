#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>

namespace mps_kit::cuda_ops {

void deltanet_chunk_scan_fwd(
    torch::Tensor M, torch::Tensor B_in, torch::Tensor DEC, torch::Tensor St0,
    torch::Tensor St_hist, torch::Tensor St_end,
    int64_t BH, int64_t N, int64_t DK, int64_t DV);

void deltanet_chunk_scan_bwd(
    torch::Tensor M, torch::Tensor DEC, torch::Tensor St_hist,
    torch::Tensor G_hist, torch::Tensor G_end,
    torch::Tensor GM, torch::Tensor GB, torch::Tensor GDEC, torch::Tensor G_St0,
    int64_t BH, int64_t N, int64_t DK, int64_t DV);

void token_delta_scan_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor ST0,
    torch::Tensor OUT, torch::Tensor CKPT, torch::Tensor ST_END,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK);

void token_delta_scan_bwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor CKPT,
    torch::Tensor GOUT, torch::Tensor GST_END, torch::Tensor SCR,
    torch::Tensor GQ, torch::Tensor GK, torch::Tensor GV,
    torch::Tensor GBETA, torch::Tensor GGAMMA, torch::Tensor GST0,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK);

void cuda_eigh_fwd(torch::Tensor G, torch::Tensor L_out, torch::Tensor V_out);

int64_t cuda_dual_gram_subspace_fwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor U_out, torch::Tensor PACK_out,
    int64_t rank, int64_t k_pad, double rel_tol, double min_eig);

void spectral_subspace_bwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor PACK, torch::Tensor GU,
    torch::Tensor GX, torch::Tensor GW, int64_t r_eff);

void ragged_waves_mm_fwd(
    torch::Tensor X,
    std::vector<torch::Tensor> W_waves,
    std::vector<torch::Tensor> act_waves,
    std::vector<torch::Tensor> offs_waves,
    std::vector<torch::Tensor> starts_waves,
    std::vector<int64_t> total_toks_waves,
    torch::Tensor Y);

void ragged_waves_mm_bwd(
    torch::Tensor GY,
    torch::Tensor X,
    std::vector<torch::Tensor> W_waves,
    std::vector<torch::Tensor> act_waves,
    std::vector<torch::Tensor> offs_waves,
    std::vector<torch::Tensor> starts_waves,
    std::vector<torch::Tensor> stops_waves,
    std::vector<int64_t> total_toks_waves,
    torch::Tensor GX,
    std::vector<torch::Tensor> GW_waves);

} // namespace mps_kit::cuda_ops
