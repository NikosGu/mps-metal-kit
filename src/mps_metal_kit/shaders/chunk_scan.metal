#include <metal_stdlib>
using namespace metal;

// Register-Resident Chunked Matrix-State Scan (DeltaNet / Fast-Weight)
// Recurrence: S_{n+1} = decay_n * (M_n @ S_n) + B_n
// Supports arbitrary 1 <= DK <= 64, 1 <= DV <= 128 in 24-32 KB threadgroup SRAM.

kernel void deltanet_chunk_scan_fwd(
    device const float* M,        // [BH, N, DK, DK]
    device const float* B_in,     // [BH, N, DK, DV]
    device const float* DEC,      // [BH, N]
    device const float* St0,      // [BH, DK, DV]
    device float* St_hist,        // [BH, N, DK, DV]
    device float* St_end,         // [BH, DK, DV]
    constant long& N,
    constant long& DK,
    constant long& DV,
    uint tid [[thread_position_in_threadgroup]],
    uint bh [[threadgroup_position_in_grid]])
{
    threadgroup float sM[64 * 64];     // 16 KB
    threadgroup float sTile[64 * 32];  // 8 KB

    long total_kv = DK * DV;
    long total_kk = DK * DK;
    float rS[32];
    for (int p = 0; p < 32; ++p) {
        long idx = (long)tid + 256L * p;
        rS[p] = (idx < total_kv) ? St0[bh * total_kv + idx] : 0.0f;
    }

    for (long n = 0; n < N; ++n) {
        long hist_base = (bh * N + n) * total_kv;
        for (int p = 0; p < 32; ++p) {
            long idx = (long)tid + 256L * p;
            if (idx < total_kv) {
                St_hist[hist_base + idx] = rS[p];
            }
        }

        device const float* m_ptr = M + (bh * N + n) * total_kk;
        for (long idx = (long)tid; idx < total_kk; idx += 256L) {
            sM[(idx / DK) * 64 + (idx % DK)] = m_ptr[idx];
        }
        float dec = DEC[bh * N + n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rS_next[32];
        for (int p = 0; p < 32; ++p) rS_next[p] = 0.0f;

        device const float* b_ptr = B_in + hist_base;
        for (long v0 = 0; v0 < DV; v0 += 32L) {
            long tw = min(32L, DV - v0);
            for (int p = 0; p < 32; ++p) {
                long idx = (long)tid + 256L * p;
                if (idx < total_kv) {
                    long k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        sTile[k * 32 + (v - v0)] = rS[p];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int p = 0; p < 32; ++p) {
                long idx = (long)tid + 256L * p;
                if (idx < total_kv) {
                    long k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        long lv = v - v0;
                        float acc = 0.0f;
                        for (long j = 0; j < DK; ++j) {
                            acc += sM[k * 64 + j] * sTile[j * 32 + lv];
                        }
                        rS_next[p] = dec * acc + b_ptr[idx];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int p = 0; p < 32; ++p) rS[p] = rS_next[p];
    }

    for (int p = 0; p < 32; ++p) {
        long idx = (long)tid + 256L * p;
        if (idx < total_kv) {
            St_end[bh * total_kv + idx] = rS[p];
        }
    }
}

kernel void deltanet_chunk_scan_bwd(
    device const float* M,        // [BH, N, DK, DK]
    device const float* DEC,      // [BH, N]
    device const float* St_hist,  // [BH, N, DK, DV]
    device const float* G_hist,   // [BH, N, DK, DV]
    device const float* G_end,    // [BH, DK, DV]
    device float* GM,             // [BH, N, DK, DK]
    device float* GB,             // [BH, N, DK, DV]
    device float* GDEC,           // [BH, N]
    device float* G_St0,          // [BH, DK, DV]
    constant long& N,
    constant long& DK,
    constant long& DV,
    uint tid [[thread_position_in_threadgroup]],
    uint bh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sA[64 * 32];     // 8 KB
    threadgroup float sB[64 * 32];     // 8 KB
    threadgroup float sP[64 * 64];     // 16 KB

    long total_kv = DK * DV;
    long total_kk = DK * DK;
    float rG[32];
    for (int p = 0; p < 32; ++p) {
        long idx = (long)tid + 256L * p;
        rG[p] = (idx < total_kv) ? G_end[bh * total_kv + idx] : 0.0f;
    }

    for (long n = N - 1; n >= 0; --n) {
        long hist_base = (bh * N + n) * total_kv;
        long kk_base = (bh * N + n) * total_kk;
        float dec = DEC[bh * N + n];

        for (int p = 0; p < 32; ++p) {
            long idx = (long)tid + 256L * p;
            if (idx < total_kv) GB[hist_base + idx] = rG[p];
        }
        for (long idx = (long)tid; idx < total_kk; idx += 256L) {
            sP[(idx / DK) * 64 + (idx % DK)] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* s_ptr = St_hist + hist_base;
        for (long v0 = 0; v0 < DV; v0 += 32L) {
            long tw = min(32L, DV - v0);
            for (int p = 0; p < 32; ++p) {
                long idx = (long)tid + 256L * p;
                if (idx < total_kv) {
                    long k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        long lv = v - v0;
                        sA[k * 32 + lv] = rG[p];
                        sB[k * 32 + lv] = s_ptr[idx];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (long idx = (long)tid; idx < total_kk; idx += 256L) {
                long i = idx / DK, j = idx % DK;
                float acc = 0.0f;
                for (long lv = 0; lv < tw; ++lv) {
                    acc += sA[i * 32 + lv] * sB[j * 32 + lv];
                }
                sP[i * 64 + j] += acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        device const float* m_ptr = M + kk_base;
        device float* gm_ptr = GM + kk_base;
        float loc_dec = 0.0f;
        for (long idx = (long)tid; idx < total_kk; idx += 256L) {
            float pij = sP[(idx / DK) * 64 + (idx % DK)];
            float mij = m_ptr[idx];
            gm_ptr[idx] = dec * pij;
            loc_dec += pij * mij;
        }
        float sg_dec = simd_sum(loc_dec);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) sA[sg] = sg_dec;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float gdec_total = 0.0f;
            for (int g = 0; g < 8; ++g) gdec_total += sA[g];
            GDEC[bh * N + n] = gdec_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (long idx = (long)tid; idx < total_kk; idx += 256L) {
            sP[(idx / DK) * 64 + (idx % DK)] = m_ptr[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rG_prev[32];
        for (int p = 0; p < 32; ++p) rG_prev[p] = 0.0f;
        device const float* gh_ptr = G_hist + hist_base;

        for (long v0 = 0; v0 < DV; v0 += 32L) {
            long tw = min(32L, DV - v0);
            for (int p = 0; p < 32; ++p) {
                long idx = (long)tid + 256L * p;
                if (idx < total_kv) {
                    long k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        sA[k * 32 + (v - v0)] = rG[p];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int p = 0; p < 32; ++p) {
                long idx = (long)tid + 256L * p;
                if (idx < total_kv) {
                    long k = idx / DV, v = idx % DV;
                    if (v >= v0 && v < v0 + tw) {
                        long lv = v - v0;
                        float acc = 0.0f;
                        for (long i = 0; i < DK; ++i) {
                            acc += sP[i * 64 + k] * sA[i * 32 + lv];
                        }
                        rG_prev[p] = dec * acc + gh_ptr[idx];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int p = 0; p < 32; ++p) rG[p] = rG_prev[p];
    }

    for (int p = 0; p < 32; ++p) {
        long idx = (long)tid + 256L * p;
        if (idx < total_kv) {
            G_St0[bh * total_kv + idx] = rG[p];
        }
    }
}
