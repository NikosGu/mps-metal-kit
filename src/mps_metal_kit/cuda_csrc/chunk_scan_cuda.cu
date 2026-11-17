#include "cuda_ops.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace mps_kit::cuda_ops {

namespace {

__device__ __forceinline__ float warp_sum_cs(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__global__ void __launch_bounds__(256) chunk_scan_fwd_kernel(
    const float* __restrict__ M,
    const float* __restrict__ B_in,
    const float* __restrict__ DEC,
    const float* __restrict__ St0,
    float* __restrict__ St_hist,
    float* __restrict__ St_end,
    int64_t N, int64_t DK, int64_t DV)
{
    __shared__ float sM[64 * 64];
    __shared__ float sTile[64 * 32];

    const int tid = threadIdx.x;
    const int64_t bh = blockIdx.x;
    const int64_t total_kv = DK * DV;
    const int64_t total_kk = DK * DK;

    float rS[32];
    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        int64_t idx = tid + 256LL * p;
        rS[p] = (idx < total_kv) ? St0[bh * total_kv + idx] : 0.0f;
    }

    for (int64_t n = 0; n < N; ++n) {
        int64_t hist_base = (bh * N + n) * total_kv;
        #pragma unroll
        for (int p = 0; p < 32; ++p) {
            int64_t idx = tid + 256LL * p;
            if (idx < total_kv) St_hist[hist_base + idx] = rS[p];
        }

        const float* m_ptr = M + (bh * N + n) * total_kk;
        for (int64_t idx = tid; idx < total_kk; idx += 256) {
            sM[(idx / DK) * 64 + (idx % DK)] = m_ptr[idx];
        }
        float dec = DEC[bh * N + n];
        __syncthreads();

        float rS_next[32];
        #pragma unroll
        for (int p = 0; p < 32; ++p) rS_next[p] = 0.0f;

        const float* b_ptr = B_in + hist_base;
        for (int64_t v0 = 0; v0 < DV; v0 += 32) {
            int64_t tw = min((int64_t)32, DV - v0);
            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                int64_t idx = tid + 256LL * p;
                if (idx < total_kv) {
                    int64_t k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) sTile[k * 32 + (v - v0)] = rS[p];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                int64_t idx = tid + 256LL * p;
                if (idx < total_kv) {
                    int64_t k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        int64_t lv = v - v0;
                        float acc = 0.0f;
                        for (int64_t j = 0; j < DK; ++j) acc += sM[k * 64 + j] * sTile[j * 32 + lv];
                        rS_next[p] = dec * acc + b_ptr[idx];
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int p = 0; p < 32; ++p) rS[p] = rS_next[p];
    }

    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        int64_t idx = tid + 256LL * p;
        if (idx < total_kv) St_end[bh * total_kv + idx] = rS[p];
    }
}

__global__ void __launch_bounds__(256) chunk_scan_bwd_kernel(
    const float* __restrict__ M,
    const float* __restrict__ DEC,
    const float* __restrict__ St_hist,
    const float* __restrict__ G_hist,
    const float* __restrict__ G_end,
    float* __restrict__ GM,
    float* __restrict__ GB,
    float* __restrict__ GDEC,
    float* __restrict__ G_St0,
    int64_t N, int64_t DK, int64_t DV)
{
    __shared__ float sA[64 * 32];
    __shared__ float sB[64 * 32];
    __shared__ float sP[64 * 64];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sg = tid >> 5;
    const int64_t bh = blockIdx.x;
    const int64_t total_kv = DK * DV;
    const int64_t total_kk = DK * DK;

    float rG[32];
    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        int64_t idx = tid + 256LL * p;
        rG[p] = (idx < total_kv) ? G_end[bh * total_kv + idx] : 0.0f;
    }

    for (int64_t n = N - 1; n >= 0; --n) {
        int64_t hist_base = (bh * N + n) * total_kv;
        int64_t kk_base = (bh * N + n) * total_kk;
        float dec = DEC[bh * N + n];

        #pragma unroll
        for (int p = 0; p < 32; ++p) {
            int64_t idx = tid + 256LL * p;
            if (idx < total_kv) GB[hist_base + idx] = rG[p];
        }
        for (int64_t idx = tid; idx < total_kk; idx += 256) {
            sP[(idx / DK) * 64 + (idx % DK)] = 0.0f;
        }
        __syncthreads();

        const float* s_ptr = St_hist + hist_base;
        for (int64_t v0 = 0; v0 < DV; v0 += 32) {
            int64_t tw = min((int64_t)32, DV - v0);
            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                int64_t idx = tid + 256LL * p;
                if (idx < total_kv) {
                    int64_t k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        int64_t lv = v - v0;
                        sA[k * 32 + lv] = rG[p];
                        sB[k * 32 + lv] = s_ptr[idx];
                    }
                }
            }
            __syncthreads();

            for (int64_t idx = tid; idx < total_kk; idx += 256) {
                int64_t i = idx / DK, j = idx % DK;
                float acc = 0.0f;
                for (int64_t lv = 0; lv < tw; ++lv) acc += sA[i * 32 + lv] * sB[j * 32 + lv];
                sP[i * 64 + j] += acc;
            }
            __syncthreads();
        }

        const float* m_ptr = M + kk_base;
        float* gm_ptr = GM + kk_base;
        float loc_dec = 0.0f;
        for (int64_t idx = tid; idx < total_kk; idx += 256) {
            float pij = sP[(idx / DK) * 64 + (idx % DK)];
            float mij = m_ptr[idx];
            gm_ptr[idx] = dec * pij;
            loc_dec += pij * mij;
        }
        float sg_dec = warp_sum_cs(loc_dec);
        __syncthreads();
        if (lane == 0) sA[sg] = sg_dec;
        __syncthreads();
        if (tid == 0) {
            float gdec_total = 0.0f;
            for (int g = 0; g < 8; ++g) gdec_total += sA[g];
            GDEC[bh * N + n] = gdec_total;
        }
        __syncthreads();

        for (int64_t idx = tid; idx < total_kk; idx += 256) {
            sP[(idx / DK) * 64 + (idx % DK)] = m_ptr[idx];
        }
        __syncthreads();

        float rG_prev[32];
        #pragma unroll
        for (int p = 0; p < 32; ++p) rG_prev[p] = 0.0f;
        const float* gh_ptr = G_hist + hist_base;

        for (int64_t v0 = 0; v0 < DV; v0 += 32) {
            int64_t tw = min((int64_t)32, DV - v0);
            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                int64_t idx = tid + 256LL * p;
                if (idx < total_kv) {
                    int64_t k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) sA[k * 32 + (v - v0)] = rG[p];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                int64_t idx = tid + 256LL * p;
                if (idx < total_kv) {
                    int64_t k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        int64_t lv = v - v0;
                        float acc = 0.0f;
                        for (int64_t i = 0; i < DK; ++i) acc += sP[i * 64 + k] * sA[i * 32 + lv];
                        rG_prev[p] = dec * acc + gh_ptr[idx];
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int p = 0; p < 32; ++p) rG[p] = rG_prev[p];
    }

    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        int64_t idx = tid + 256LL * p;
        if (idx < total_kv) G_St0[bh * total_kv + idx] = rG[p];
    }
}

} // namespace

void deltanet_chunk_scan_fwd(
    torch::Tensor M, torch::Tensor B_in, torch::Tensor DEC, torch::Tensor St0,
    torch::Tensor St_hist, torch::Tensor St_end,
    int64_t BH, int64_t N, int64_t DK, int64_t DV)
{
    if (BH <= 0 || N <= 0) return;
    TORCH_CHECK(DK >= 1 && DK <= 64 && DV >= 1 && DV <= 128, "deltanet_chunk_scan expects 1<=DK<=64, 1<=DV<=128");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    chunk_scan_fwd_kernel<<<(unsigned)BH, 256, 0, stream>>>(
        M.data_ptr<float>(), B_in.data_ptr<float>(), DEC.data_ptr<float>(), St0.data_ptr<float>(),
        St_hist.data_ptr<float>(), St_end.data_ptr<float>(), N, DK, DV);
}

void deltanet_chunk_scan_bwd(
    torch::Tensor M, torch::Tensor DEC, torch::Tensor St_hist,
    torch::Tensor G_hist, torch::Tensor G_end,
    torch::Tensor GM, torch::Tensor GB, torch::Tensor GDEC, torch::Tensor G_St0,
    int64_t BH, int64_t N, int64_t DK, int64_t DV)
{
    if (BH <= 0 || N <= 0) return;
    TORCH_CHECK(DK >= 1 && DK <= 64 && DV >= 1 && DV <= 128, "deltanet_chunk_scan expects 1<=DK<=64, 1<=DV<=128");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    chunk_scan_bwd_kernel<<<(unsigned)BH, 256, 0, stream>>>(
        M.data_ptr<float>(), DEC.data_ptr<float>(), St_hist.data_ptr<float>(),
        G_hist.data_ptr<float>(), G_end.data_ptr<float>(),
        GM.data_ptr<float>(), GB.data_ptr<float>(), GDEC.data_ptr<float>(), G_St0.data_ptr<float>(),
        N, DK, DV);
}

} // namespace mps_kit::cuda_ops
