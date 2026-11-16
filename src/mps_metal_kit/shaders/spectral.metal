#include <metal_stdlib>
using namespace metal;

#define EIGH_GRAD_CLIP 100.0f

// Fused Daleckii-Krein + Orthogonal-Nullspace Spectral Subspace Backward
// Differentiates top-r orthonormal eigenvectors U [B, D, r] of
// G_b = sum_{t=0}^{L_b - 1} w_{b,t} x_{b,t} x_{b,t}^T in O(M * r * D).

kernel void spectral_subspace_bwd(
    device const float* X,        // [B, M, D]
    device const float* W,        // [B, M]
    device const int* LENS,       // [B]
    device const float* PACK,     // [B, D + 3, K_PAD] (Q [D, K_PAD], L, keep, valid)
    device const float* GU,       // [B, D, R]
    device float* GX,             // [B, M, D]
    device float* GW,             // [B, M]
    constant long& M,
    constant long& D,
    constant long& K_PAD,
    constant long& R,
    constant long& R_EFF,
    uint tid [[thread_position_in_threadgroup]],
    uint b [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sVtop[256 * 12];  // 12 KB
    threadgroup float sBmat[256 * 12];  // 12 KB
    threadgroup float sRed[256];        // 1 KB

    int L_b = LENS[b];
    device const float* pack_b = PACK + b * (D + 3) * K_PAD;
    device const float* Q = pack_b;
    device const float* L_vec = pack_b + D * K_PAD;
    device const float* kf_vec = pack_b + (D + 1) * K_PAD;
    device const float* vq_vec = pack_b + (D + 2) * K_PAD;
    device const float* gu_b = GU + b * D * R;

    float b_acc[12];
    for (int a = 0; a < 12; ++a) b_acc[a] = 0.0f;

    bool active_d = ((long)tid < D);
    if (active_d) {
        for (long a = 0; a < R_EFF; ++a) {
            float kfa = kf_vec[a];
            sVtop[tid * 12 + a] = Q[tid * K_PAD + a] * kfa;
            float la = L_vec[a];
            float inv_2l = (kfa > 0.5f && la > 1e-20f) ? (0.5f / la) : 0.0f;
            b_acc[a] = gu_b[tid * R + a] * kfa * inv_2l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (long i = 0; i < K_PAD; ++i) {
        if (vq_vec[i] > 0.5f) {
            float kfi = kf_vec[i];
            float li = L_vec[i];
            float qi = active_d ? Q[tid * K_PAD + i] : 0.0f;

            for (long a = 0; a < R_EFF; ++a) {
                float kfa = kf_vec[a];
                float prod = (active_d && kfa > 0.5f) ? (0.5f * qi * gu_b[tid * R + a]) : 0.0f;
                float sg_sum = simd_sum(prod);
                if (lane == 0) sRed[a * 8 + sg] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active_d) {
                for (long a = 0; a < R_EFF; ++a) {
                    float c_ia = 0.0f;
                    for (int g = 0; g < 8; ++g) c_ia += sRed[a * 8 + g];
                    float kfa = kf_vec[a];
                    float la = L_vec[a];
                    float inv_la = (kfa > 0.5f && la > 1e-20f) ? (1.0f / la) : 0.0f;
                    float dk_f = 0.0f;
                    if (kfi < 0.5f && kfa > 0.5f) {
                        float dl = la - li;
                        dk_f = clamp(dl / (dl * dl + 1e-12f), -EIGH_GRAD_CLIP, EIGH_GRAD_CLIP);
                    }
                    b_acc[a] += qi * c_ia * (dk_f - inv_la);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (active_d) {
        for (long a = 0; a < R_EFF; ++a) sBmat[tid * 12 + a] = b_acc[a];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (long m = 0; m < M; ++m) {
        long bm = b * M + m;
        if (m >= (long)L_b) {
            if (tid == 0) GW[bm] = 0.0f;
            if (active_d) GX[bm * D + tid] = 0.0f;
            continue;
        }
        float wm = max(W[bm], 0.0f);
        float xd = active_d ? X[bm * D + tid] : 0.0f;

        for (long a = 0; a < R_EFF; ++a) {
            float pb = simd_sum(active_d ? (xd * sBmat[tid * 12 + a]) : 0.0f);
            float pv = simd_sum(active_d ? (xd * sVtop[tid * 12 + a]) : 0.0f);
            if (lane == 0) {
                sRed[a * 8 + sg] = pb;
                sRed[96 + a * 8 + sg] = pv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float xb[12], xv[12];
        float gw_val = 0.0f;
        for (long a = 0; a < R_EFF; ++a) {
            float sb = 0.0f, sv = 0.0f;
            for (int g = 0; g < 8; ++g) {
                sb += sRed[a * 8 + g];
                sv += sRed[96 + a * 8 + g];
            }
            xb[a] = sb;
            xv[a] = sv;
            gw_val += 2.0f * sb * sv;
        }
        if (tid == 0) GW[bm] = gw_val;
        if (active_d) {
            float gx_d = 0.0f;
            for (long a = 0; a < R_EFF; ++a) {
                gx_d += xb[a] * sVtop[tid * 12 + a] + xv[a] * sBmat[tid * 12 + a];
            }
            GX[bm * D + tid] = 2.0f * wm * gx_d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
