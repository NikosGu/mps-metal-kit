#include "cuda_ops.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace mps_kit::cuda_ops {

namespace {

constexpr int DV_TOK = 64;
constexpr int CKI_TOK = 16;

__device__ __forceinline__ float warp_sum_ts(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

template <int DK_TOK>
__global__ void __launch_bounds__(32 * (DK_TOK / 16)) token_scan_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ BETA,
    const float* __restrict__ GAMMA,
    const float* __restrict__ ST0,
    float* __restrict__ OUT,
    float* __restrict__ CKPT,
    float* __restrict__ ST_END,
    int64_t T, int64_t NCK)
{
    constexpr int NW_TOK = DK_TOK / 16;
    constexpr int TPB_TOK = 32 * NW_TOK;
    constexpr int64_t NS = DV_TOK * DK_TOK;

    __shared__ float tq[DK_TOK];
    __shared__ float tk[DK_TOK];
    __shared__ float tv[DV_TOK];
    __shared__ float te[DV_TOK];
    __shared__ float part[NW_TOK * DV_TOK];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sg = tid >> 5;
    const int64_t bh = blockIdx.x;

    float S0[16], S1[16];
    const float* st0 = ST0 + bh * NS;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        S0[j] = st0[lane * DK_TOK + 16 * sg + j];
        S1[j] = st0[(lane + 32) * DK_TOK + 16 * sg + j];
    }

    for (int64_t t = 0; t < T; ++t) {
        if ((t % CKI_TOK) == 0) {
            float* dst = CKPT + (bh * NCK + (t / CKI_TOK)) * NS;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                dst[lane * DK_TOK + 16 * sg + j] = S0[j];
                dst[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
            }
        }

        int64_t bt = bh * T + t;
        for (int i = tid; i < DK_TOK; i += TPB_TOK) {
            tq[i] = Q[bt * DK_TOK + i];
            tk[i] = K[bt * DK_TOK + i];
        }
        if (tid < DV_TOK) tv[tid] = V[bt * DV_TOK + tid];
        float beta = BETA[bt];
        float gam = GAMMA[bt];
        __syncthreads();

        float p0 = 0.0f, p1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float kj = tk[16 * sg + j];
            p0 += S0[j] * kj;
            p1 += S1[j] * kj;
        }
        part[sg * DV_TOK + lane] = p0;
        part[sg * DV_TOK + lane + 32] = p1;
        __syncthreads();

        if (tid < DV_TOK) {
            float vh = 0.0f;
            for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
            te[tid] = tv[tid] - vh;
        }
        __syncthreads();

        float be0 = beta * te[lane];
        float be1 = beta * te[lane + 32];
        float o0 = 0.0f, o1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float kj = tk[16 * sg + j];
            float qj = tq[16 * sg + j];
            S0[j] = gam * S0[j] + be0 * kj;
            S1[j] = gam * S1[j] + be1 * kj;
            o0 += S0[j] * qj;
            o1 += S1[j] * qj;
        }
        part[sg * DV_TOK + lane] = o0;
        part[sg * DV_TOK + lane + 32] = o1;
        __syncthreads();

        if (tid < DV_TOK) {
            float out_v = 0.0f;
            for (int g = 0; g < NW_TOK; ++g) out_v += part[g * DV_TOK + tid];
            OUT[bt * DV_TOK + tid] = out_v;
        }
        __syncthreads();
    }

    float* st_end = ST_END + bh * NS;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        st_end[lane * DK_TOK + 16 * sg + j] = S0[j];
        st_end[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
    }
}

template <int DK_TOK>
__global__ void __launch_bounds__(32 * (DK_TOK / 16)) token_scan_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ BETA,
    const float* __restrict__ GAMMA,
    const float* __restrict__ CKPT,
    const float* __restrict__ GOUT,
    const float* __restrict__ GST_END,
    float* __restrict__ SCR,
    float* __restrict__ GQ,
    float* __restrict__ GK,
    float* __restrict__ GV,
    float* __restrict__ GBETA,
    float* __restrict__ GGAMMA,
    float* __restrict__ GST0,
    int64_t T, int64_t NCK)
{
    constexpr int NW_TOK = DK_TOK / 16;
    constexpr int TPB_TOK = 32 * NW_TOK;
    constexpr int64_t NS = DV_TOK * DK_TOK;

    __shared__ float tq[DK_TOK];
    __shared__ float tk[DK_TOK];
    __shared__ float tv[DV_TOK];
    __shared__ float te[DV_TOK];
    __shared__ float tgo[DV_TOK];
    __shared__ float tge[DV_TOK];
    __shared__ float part[NW_TOK * DV_TOK];
    __shared__ float red[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sg = tid >> 5;
    const int64_t bh = blockIdx.x;

    float G0[16], G1[16];
    const float* gst_end = GST_END + bh * NS;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        G0[j] = gst_end[lane * DK_TOK + 16 * sg + j];
        G1[j] = gst_end[(lane + 32) * DK_TOK + 16 * sg + j];
    }

    int64_t n_chunks = (T + CKI_TOK - 1) / CKI_TOK;
    for (int64_t ck = n_chunks - 1; ck >= 0; --ck) {
        int64_t t0 = ck * CKI_TOK;
        int64_t clen = min((int64_t)CKI_TOK, T - t0);

        float S0[16], S1[16];
        const float* ck_src = CKPT + (bh * NCK + ck) * NS;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            S0[j] = ck_src[lane * DK_TOK + 16 * sg + j];
            S1[j] = ck_src[(lane + 32) * DK_TOK + 16 * sg + j];
        }

        for (int64_t lt = 0; lt < clen; ++lt) {
            float* scr_dst = SCR + (bh * CKI_TOK + lt) * NS;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                scr_dst[lane * DK_TOK + 16 * sg + j] = S0[j];
                scr_dst[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
            }
            if (lt + 1 < clen) {
                int64_t bt = bh * T + (t0 + lt);
                for (int i = tid; i < DK_TOK; i += TPB_TOK) tk[i] = K[bt * DK_TOK + i];
                if (tid < DV_TOK) tv[tid] = V[bt * DV_TOK + tid];
                float beta = BETA[bt];
                float gam = GAMMA[bt];
                __syncthreads();

                float p0 = 0.0f, p1 = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    float kj = tk[16 * sg + j];
                    p0 += S0[j] * kj;
                    p1 += S1[j] * kj;
                }
                part[sg * DV_TOK + lane] = p0;
                part[sg * DV_TOK + lane + 32] = p1;
                __syncthreads();

                if (tid < DV_TOK) {
                    float vh = 0.0f;
                    for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
                    te[tid] = tv[tid] - vh;
                }
                __syncthreads();

                float be0 = beta * te[lane], be1 = beta * te[lane + 32];
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    float kj = tk[16 * sg + j];
                    S0[j] = gam * S0[j] + be0 * kj;
                    S1[j] = gam * S1[j] + be1 * kj;
                }
                __syncthreads();
            }
        }
        __syncthreads();

        for (int64_t lt = clen - 1; lt >= 0; --lt) {
            int64_t bt = bh * T + (t0 + lt);
            const float* scr_src = SCR + (bh * CKI_TOK + lt) * NS;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                S0[j] = scr_src[lane * DK_TOK + 16 * sg + j];
                S1[j] = scr_src[(lane + 32) * DK_TOK + 16 * sg + j];
            }
            for (int i = tid; i < DK_TOK; i += TPB_TOK) {
                tq[i] = Q[bt * DK_TOK + i];
                tk[i] = K[bt * DK_TOK + i];
            }
            if (tid < DV_TOK) {
                tv[tid] = V[bt * DV_TOK + tid];
                tgo[tid] = GOUT[bt * DV_TOK + tid];
            }
            float beta = BETA[bt];
            float gam = GAMMA[bt];
            __syncthreads();

            float p0 = 0.0f, p1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                p0 += S0[j] * kj;
                p1 += S1[j] * kj;
            }
            part[sg * DV_TOK + lane] = p0;
            part[sg * DV_TOK + lane + 32] = p1;
            __syncthreads();

            if (tid < DV_TOK) {
                float vh = 0.0f;
                for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
                te[tid] = tv[tid] - vh;
            }
            __syncthreads();

            float go0 = tgo[lane], go1 = tgo[lane + 32];
            float be0 = beta * te[lane], be1 = beta * te[lane + 32];
            float loc_gam = 0.0f, u0 = 0.0f, u1 = 0.0f;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                float qj = tq[16 * sg + j];
                float St0_j = gam * S0[j] + be0 * kj;
                float St1_j = gam * S1[j] + be1 * kj;
                float gq_j = warp_sum_ts(St0_j * go0 + St1_j * go1);
                if (lane == 0) GQ[bt * DK_TOK + 16 * sg + j] = gq_j;
                G0[j] += go0 * qj;
                G1[j] += go1 * qj;
                loc_gam += G0[j] * S0[j] + G1[j] * S1[j];
                u0 += G0[j] * kj;
                u1 += G1[j] * kj;
            }

            float sg_gam = warp_sum_ts(loc_gam);
            if (lane == 0) red[sg] = sg_gam;
            part[sg * DV_TOK + lane] = u0;
            part[sg * DV_TOK + lane + 32] = u1;
            __syncthreads();

            if (tid == 0) {
                float gg = 0.0f;
                for (int g = 0; g < NW_TOK; ++g) gg += red[g];
                GGAMMA[bt] = gg;
            }
            if (tid < DV_TOK) {
                float ut = 0.0f;
                for (int g = 0; g < NW_TOK; ++g) ut += part[g * DV_TOK + tid];
                float ge = beta * ut;
                tge[tid] = ge;
                GV[bt * DV_TOK + tid] = ge;
                float sb = warp_sum_ts(te[tid] * ut);
                if (lane == 0) red[16 + sg] = sb;
            }
            __syncthreads();

            if (tid == 0) GBETA[bt] = red[16] + red[17];

            float ge0 = tge[lane], ge1 = tge[lane + 32];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                float gk_j = warp_sum_ts((G0[j] * be0 - S0[j] * ge0) + (G1[j] * be1 - S1[j] * ge1));
                if (lane == 0) GK[bt * DK_TOK + 16 * sg + j] = gk_j;
                G0[j] = gam * G0[j] - ge0 * kj;
                G1[j] = gam * G1[j] - ge1 * kj;
            }
            __syncthreads();
        }
    }

    float* gst0 = GST0 + bh * NS;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        gst0[lane * DK_TOK + 16 * sg + j] = G0[j];
        gst0[(lane + 32) * DK_TOK + 16 * sg + j] = G1[j];
    }
}

} // namespace

void token_delta_scan_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor ST0,
    torch::Tensor OUT, torch::Tensor CKPT, torch::Tensor ST_END,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK)
{
    if (BH <= 0 || T <= 0) return;
    TORCH_CHECK(DK == 64 || DK == 128, "token_delta_scan supports DK in {64, 128}");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (DK == 64) {
        token_scan_fwd_kernel<64><<<(unsigned)BH, 128, 0, stream>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            BETA.data_ptr<float>(), GAMMA.data_ptr<float>(), ST0.data_ptr<float>(),
            OUT.data_ptr<float>(), CKPT.data_ptr<float>(), ST_END.data_ptr<float>(), T, NCK);
    } else {
        token_scan_fwd_kernel<128><<<(unsigned)BH, 256, 0, stream>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            BETA.data_ptr<float>(), GAMMA.data_ptr<float>(), ST0.data_ptr<float>(),
            OUT.data_ptr<float>(), CKPT.data_ptr<float>(), ST_END.data_ptr<float>(), T, NCK);
    }
}

void token_delta_scan_bwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor CKPT,
    torch::Tensor GOUT, torch::Tensor GST_END, torch::Tensor SCR,
    torch::Tensor GQ, torch::Tensor GK, torch::Tensor GV,
    torch::Tensor GBETA, torch::Tensor GGAMMA, torch::Tensor GST0,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK)
{
    if (BH <= 0 || T <= 0) return;
    TORCH_CHECK(DK == 64 || DK == 128, "token_delta_scan supports DK in {64, 128}");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (DK == 64) {
        token_scan_bwd_kernel<64><<<(unsigned)BH, 128, 0, stream>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            BETA.data_ptr<float>(), GAMMA.data_ptr<float>(), CKPT.data_ptr<float>(),
            GOUT.data_ptr<float>(), GST_END.data_ptr<float>(), SCR.data_ptr<float>(),
            GQ.data_ptr<float>(), GK.data_ptr<float>(), GV.data_ptr<float>(),
            GBETA.data_ptr<float>(), GGAMMA.data_ptr<float>(), GST0.data_ptr<float>(), T, NCK);
    } else {
        token_scan_bwd_kernel<128><<<(unsigned)BH, 256, 0, stream>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            BETA.data_ptr<float>(), GAMMA.data_ptr<float>(), CKPT.data_ptr<float>(),
            GOUT.data_ptr<float>(), GST_END.data_ptr<float>(), SCR.data_ptr<float>(),
            GQ.data_ptr<float>(), GK.data_ptr<float>(), GV.data_ptr<float>(),
            GBETA.data_ptr<float>(), GGAMMA.data_ptr<float>(), GST0.data_ptr<float>(), T, NCK);
    }
}

} // namespace mps_kit::cuda_ops
