#include <metal_stdlib>
using namespace metal;

#ifndef DK_TOK
#define DK_TOK 64
#endif
#define NW_TOK (DK_TOK / 16)
#define TPB_TOK (32 * NW_TOK)
#define DV_TOK 64
#define CKI_TOK 16

// Register-Resident Token-Level Gated Delta Scan (CKI=16 Checkpointed)
// e_t = v_t - S_{t-1} k_t,  S_t = gamma_t * S_{t-1} + beta_t * e_t * k_t^T,  o_t = S_t q_t

kernel void token_delta_scan_fwd(
    device const float* Q,        // [BH, T, DK_TOK]
    device const float* K,        // [BH, T, DK_TOK]
    device const float* V,        // [BH, T, 64]
    device const float* BETA,     // [BH, T]
    device const float* GAMMA,    // [BH, T]
    device const float* ST0,      // [BH, 64, DK_TOK]
    device float* OUT,            // [BH, T, 64]
    device float* CKPT,           // [BH, NCK, 64 * DK_TOK]
    device float* ST_END,         // [BH, 64, DK_TOK]
    constant long& T,
    constant long& NCK,
    uint tid [[thread_position_in_threadgroup]],
    uint bh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tq[DK_TOK];
    threadgroup float tk[DK_TOK];
    threadgroup float tv[DV_TOK];
    threadgroup float te[DV_TOK];
    threadgroup float part[NW_TOK * DV_TOK];

    const long NS = DV_TOK * DK_TOK;
    float S0[16], S1[16];
    device const float* st0 = ST0 + bh * NS;
    for (int j = 0; j < 16; ++j) {
        S0[j] = st0[lane * DK_TOK + 16 * sg + j];
        S1[j] = st0[(lane + 32) * DK_TOK + 16 * sg + j];
    }

    for (long t = 0; t < T; ++t) {
        if ((t % CKI_TOK) == 0) {
            device float* dst = CKPT + (bh * NCK + (t / CKI_TOK)) * NS;
            for (int j = 0; j < 16; ++j) {
                dst[lane * DK_TOK + 16 * sg + j] = S0[j];
                dst[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
            }
        }

        long bt = bh * T + t;
        for (uint i = tid; i < DK_TOK; i += TPB_TOK) {
            tq[i] = Q[bt * DK_TOK + i];
            tk[i] = K[bt * DK_TOK + i];
        }
        if (tid < DV_TOK) tv[tid] = V[bt * DV_TOK + tid];
        float beta = BETA[bt];
        float gam = GAMMA[bt];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float p0 = 0.0f, p1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            float kj = tk[16 * sg + j];
            p0 += S0[j] * kj;
            p1 += S1[j] * kj;
        }
        part[sg * DV_TOK + lane] = p0;
        part[sg * DV_TOK + lane + 32] = p1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < DV_TOK) {
            float vh = 0.0f;
            for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
            te[tid] = tv[tid] - vh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float be0 = beta * te[lane];
        float be1 = beta * te[lane + 32];
        float o0 = 0.0f, o1 = 0.0f;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < DV_TOK) {
            float out_v = 0.0f;
            for (int g = 0; g < NW_TOK; ++g) out_v += part[g * DV_TOK + tid];
            OUT[bt * DV_TOK + tid] = out_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* st_end = ST_END + bh * NS;
    for (int j = 0; j < 16; ++j) {
        st_end[lane * DK_TOK + 16 * sg + j] = S0[j];
        st_end[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
    }
}

kernel void token_delta_scan_bwd(
    device const float* Q,        // [BH, T, DK_TOK]
    device const float* K,        // [BH, T, DK_TOK]
    device const float* V,        // [BH, T, 64]
    device const float* BETA,     // [BH, T]
    device const float* GAMMA,    // [BH, T]
    device const float* CKPT,     // [BH, NCK, 64 * DK_TOK]
    device const float* GOUT,     // [BH, T, 64]
    device const float* GST_END,  // [BH, 64, DK_TOK]
    device float* SCR,            // [BH, CKI_TOK, 64 * DK_TOK]
    device float* GQ,             // [BH, T, DK_TOK]
    device float* GK,             // [BH, T, DK_TOK]
    device float* GV,             // [BH, T, 64]
    device float* GBETA,          // [BH, T]
    device float* GGAMMA,         // [BH, T]
    device float* GST0,           // [BH, 64, DK_TOK]
    constant long& T,
    constant long& NCK,
    uint tid [[thread_position_in_threadgroup]],
    uint bh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tq[DK_TOK];
    threadgroup float tk[DK_TOK];
    threadgroup float tv[DV_TOK];
    threadgroup float te[DV_TOK];
    threadgroup float tgo[DV_TOK];
    threadgroup float tge[DV_TOK];
    threadgroup float part[NW_TOK * DV_TOK];
    threadgroup float red[32];

    const long NS = DV_TOK * DK_TOK;
    float G0[16], G1[16];
    device const float* gst_end = GST_END + bh * NS;
    for (int j = 0; j < 16; ++j) {
        G0[j] = gst_end[lane * DK_TOK + 16 * sg + j];
        G1[j] = gst_end[(lane + 32) * DK_TOK + 16 * sg + j];
    }

    long n_chunks = (T + CKI_TOK - 1) / CKI_TOK;
    for (long ck = n_chunks - 1; ck >= 0; --ck) {
        long t0 = ck * CKI_TOK;
        long clen = min((long)CKI_TOK, T - t0);

        float S0[16], S1[16];
        device const float* ck_src = CKPT + (bh * NCK + ck) * NS;
        for (int j = 0; j < 16; ++j) {
            S0[j] = ck_src[lane * DK_TOK + 16 * sg + j];
            S1[j] = ck_src[(lane + 32) * DK_TOK + 16 * sg + j];
        }

        for (long lt = 0; lt < clen; ++lt) {
            device float* scr_dst = SCR + (bh * CKI_TOK + lt) * NS;
            for (int j = 0; j < 16; ++j) {
                scr_dst[lane * DK_TOK + 16 * sg + j] = S0[j];
                scr_dst[(lane + 32) * DK_TOK + 16 * sg + j] = S1[j];
            }
            if (lt + 1 < clen) {
                long bt = bh * T + (t0 + lt);
                for (uint i = tid; i < DK_TOK; i += TPB_TOK) tk[i] = K[bt * DK_TOK + i];
                if (tid < DV_TOK) tv[tid] = V[bt * DV_TOK + tid];
                float beta = BETA[bt];
                float gam = GAMMA[bt];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float p0 = 0.0f, p1 = 0.0f;
                for (int j = 0; j < 16; ++j) {
                    float kj = tk[16 * sg + j];
                    p0 += S0[j] * kj;
                    p1 += S1[j] * kj;
                }
                part[sg * DV_TOK + lane] = p0;
                part[sg * DV_TOK + lane + 32] = p1;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < DV_TOK) {
                    float vh = 0.0f;
                    for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
                    te[tid] = tv[tid] - vh;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float be0 = beta * te[lane];
                float be1 = beta * te[lane + 32];
                for (int j = 0; j < 16; ++j) {
                    float kj = tk[16 * sg + j];
                    S0[j] = gam * S0[j] + be0 * kj;
                    S1[j] = gam * S1[j] + be1 * kj;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        for (long lt = clen - 1; lt >= 0; --lt) {
            long bt = bh * T + (t0 + lt);
            device const float* scr_src = SCR + (bh * CKI_TOK + lt) * NS;
            for (int j = 0; j < 16; ++j) {
                S0[j] = scr_src[lane * DK_TOK + 16 * sg + j];
                S1[j] = scr_src[(lane + 32) * DK_TOK + 16 * sg + j];
            }
            for (uint i = tid; i < DK_TOK; i += TPB_TOK) {
                tq[i] = Q[bt * DK_TOK + i];
                tk[i] = K[bt * DK_TOK + i];
            }
            if (tid < DV_TOK) {
                tv[tid] = V[bt * DV_TOK + tid];
                tgo[tid] = GOUT[bt * DV_TOK + tid];
            }
            float beta = BETA[bt];
            float gam = GAMMA[bt];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float p0 = 0.0f, p1 = 0.0f;
            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                p0 += S0[j] * kj;
                p1 += S1[j] * kj;
            }
            part[sg * DV_TOK + lane] = p0;
            part[sg * DV_TOK + lane + 32] = p1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < DV_TOK) {
                float vh = 0.0f;
                for (int g = 0; g < NW_TOK; ++g) vh += part[g * DV_TOK + tid];
                te[tid] = tv[tid] - vh;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float go0 = tgo[lane], go1 = tgo[lane + 32];
            float be0 = beta * te[lane], be1 = beta * te[lane + 32];
            float loc_gam = 0.0f, u0 = 0.0f, u1 = 0.0f;

            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                float qj = tq[16 * sg + j];
                float St0_j = gam * S0[j] + be0 * kj;
                float St1_j = gam * S1[j] + be1 * kj;
                float gq_j = simd_sum(St0_j * go0 + St1_j * go1);
                if (lane == 0) GQ[bt * DK_TOK + 16 * sg + j] = gq_j;
                G0[j] += go0 * qj;
                G1[j] += go1 * qj;
                loc_gam += G0[j] * S0[j] + G1[j] * S1[j];
                u0 += G0[j] * kj;
                u1 += G1[j] * kj;
            }

            float sg_gam = simd_sum(loc_gam);
            if (lane == 0) red[sg] = sg_gam;
            part[sg * DV_TOK + lane] = u0;
            part[sg * DV_TOK + lane + 32] = u1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
                float sb = simd_sum(te[tid] * ut);
                if (lane == 0) red[16 + sg] = sb;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) GBETA[bt] = red[16] + red[17];

            float ge0 = tge[lane], ge1 = tge[lane + 32];
            for (int j = 0; j < 16; ++j) {
                float kj = tk[16 * sg + j];
                float gk_j = simd_sum((G0[j] * be0 - S0[j] * ge0) + (G1[j] * be1 - S1[j] * ge1));
                if (lane == 0) GK[bt * DK_TOK + 16 * sg + j] = gk_j;
                G0[j] = gam * G0[j] - ge0 * kj;
                G1[j] = gam * G1[j] - ge1 * kj;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device float* gst0 = GST0 + bh * NS;
    for (int j = 0; j < 16; ++j) {
        gst0[lane * DK_TOK + 16 * sg + j] = G0[j];
        gst0[(lane + 32) * DK_TOK + 16 * sg + j] = G1[j];
    }
}
