#include "cuda_ops.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace mps_kit::cuda_ops {

namespace {

__device__ __forceinline__ int find_seg_idx(const int* __restrict__ offs, int n_act, int g_tok) {
    int lo = 0, hi = n_act;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (offs[mid] <= g_tok) lo = mid;
        else hi = mid;
    }
    return lo;
}

__global__ void __launch_bounds__(256) ragged_seg_mm_fwd_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const int* __restrict__ ACT,
    const int* __restrict__ OFFS,
    const int* __restrict__ STARTS,
    float* __restrict__ Y,
    int64_t T, int n_act, int64_t D_IN, int64_t D_OUT)
{
    __shared__ float sX[256];
    const int tid = threadIdx.x;
    const int g_tok = blockIdx.x;

    int i_seg = find_seg_idx(OFFS, n_act, g_tok);
    int b = ACT[i_seg];
    int t = STARTS[i_seg] + (g_tok - OFFS[i_seg]);

    if (tid < D_IN) sX[tid] = X[(b * T + t) * D_IN + tid];
    __syncthreads();

    if (tid < D_OUT) {
        const float* w_row = W + (b * D_OUT + tid) * D_IN;
        float acc = 0.0f;
        for (int j = 0; j < D_IN; ++j) acc += w_row[j] * sX[j];
        Y[(b * T + t) * D_OUT + tid] = acc;
    }
}

__global__ void __launch_bounds__(256) ragged_seg_mm_dx_bwd_kernel(
    const float* __restrict__ GY,
    const float* __restrict__ W,
    const int* __restrict__ ACT,
    const int* __restrict__ OFFS,
    const int* __restrict__ STARTS,
    float* __restrict__ GX,
    int64_t T, int n_act, int64_t D_IN, int64_t D_OUT)
{
    __shared__ float sGY[256];
    const int tid = threadIdx.x;
    const int g_tok = blockIdx.x;

    int i_seg = find_seg_idx(OFFS, n_act, g_tok);
    int b = ACT[i_seg];
    int t = STARTS[i_seg] + (g_tok - OFFS[i_seg]);

    if (tid < D_OUT) sGY[tid] = GY[(b * T + t) * D_OUT + tid];
    __syncthreads();

    if (tid < D_IN) {
        const float* w_b = W + b * D_OUT * D_IN + tid;
        float acc = 0.0f;
        for (int o = 0; o < D_OUT; ++o) acc += w_b[o * D_IN] * sGY[o];
        GX[(b * T + t) * D_IN + tid] = acc;
    }
}

__global__ void __launch_bounds__(256) ragged_seg_mm_dw_bwd_kernel(
    const float* __restrict__ GY,
    const float* __restrict__ X,
    const int* __restrict__ ACT,
    const int* __restrict__ STARTS,
    const int* __restrict__ STOPS,
    float* __restrict__ GW,
    int64_t T, int64_t D_IN, int64_t D_OUT)
{
    __shared__ float sX[256];
    __shared__ float sGY[256];

    const int tid = threadIdx.x;
    const int i_seg = blockIdx.x;
    const int b = ACT[i_seg];
    const int s0 = STARTS[i_seg];
    const int s1 = STOPS[i_seg];
    const int64_t total_oi = D_OUT * D_IN;
    float* gw_b = GW + b * total_oi;

    for (int t = s0; t < s1; ++t) {
        if (tid < D_IN) sX[tid] = X[(b * T + t) * D_IN + tid];
        if (tid < D_OUT) sGY[tid] = GY[(b * T + t) * D_OUT + tid];
        __syncthreads();

        for (int64_t idx = tid; idx < total_oi; idx += 256) {
            int64_t o = idx / D_IN;
            int64_t j = idx - o * D_IN;
            gw_b[idx] += sGY[o] * sX[j];
        }
        __syncthreads();
    }
}

} // namespace

void ragged_waves_mm_fwd(
    torch::Tensor X,
    std::vector<torch::Tensor> W_waves,
    std::vector<torch::Tensor> act_waves,
    std::vector<torch::Tensor> offs_waves,
    std::vector<torch::Tensor> starts_waves,
    std::vector<int64_t> total_toks_waves,
    torch::Tensor Y)
{
    int64_t T = X.size(1);
    int64_t D_IN = X.size(2);
    int64_t D_OUT = Y.size(2);
    size_t n_waves = W_waves.size();
    if (n_waves == 0) return;
    TORCH_CHECK(D_IN <= 256 && D_OUT <= 256, "ragged_waves_mm supports D_IN, D_OUT <= 256");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* d_X = X.data_ptr<float>();
    float* d_Y = Y.data_ptr<float>();

    for (size_t w = 0; w < n_waves; ++w) {
        int64_t tot = total_toks_waves[w];
        int n_act = (int)act_waves[w].size(0);
        if (tot <= 0 || n_act <= 0) continue;
        ragged_seg_mm_fwd_kernel<<<(unsigned)tot, 256, 0, stream>>>(
            d_X, W_waves[w].data_ptr<float>(),
            act_waves[w].data_ptr<int>(), offs_waves[w].data_ptr<int>(), starts_waves[w].data_ptr<int>(),
            d_Y, T, n_act, D_IN, D_OUT);
    }
}

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
    std::vector<torch::Tensor> GW_waves)
{
    int64_t T = X.size(1);
    int64_t D_IN = X.size(2);
    int64_t D_OUT = GY.size(2);
    size_t n_waves = W_waves.size();
    if (n_waves == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* d_GY = GY.data_ptr<float>();
    const float* d_X = X.data_ptr<float>();
    float* d_GX = GX.data_ptr<float>();

    for (size_t w = 0; w < n_waves; ++w) {
        int64_t tot = total_toks_waves[w];
        int n_act = (int)act_waves[w].size(0);
        if (tot > 0 && n_act > 0) {
            ragged_seg_mm_dx_bwd_kernel<<<(unsigned)tot, 256, 0, stream>>>(
                d_GY, W_waves[w].data_ptr<float>(),
                act_waves[w].data_ptr<int>(), offs_waves[w].data_ptr<int>(), starts_waves[w].data_ptr<int>(),
                d_GX, T, n_act, D_IN, D_OUT);
        }
        if (n_act > 0) {
            ragged_seg_mm_dw_bwd_kernel<<<(unsigned)n_act, 256, 0, stream>>>(
                d_GY, d_X,
                act_waves[w].data_ptr<int>(), starts_waves[w].data_ptr<int>(), stops_waves[w].data_ptr<int>(),
                GW_waves[w].data_ptr<float>(), T, D_IN, D_OUT);
        }
    }
}

} // namespace mps_kit::cuda_ops
