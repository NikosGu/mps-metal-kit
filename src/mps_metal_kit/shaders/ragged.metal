#include <metal_stdlib>
using namespace metal;

// Prefix-Sum Ragged Segment Linear Projection (1 Threadgroup = 1 Active Token)

inline int find_active_segment(device const int* TOK_OFFS, int n_act, uint g_tok) {
    int lo = 0, hi = n_act - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if ((int)g_tok >= TOK_OFFS[mid]) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

kernel void ragged_seg_mm_fwd(
    device const float* X,          // [B, T, D_IN]
    device const float* W_SEG,      // [B, D_OUT, D_IN]
    device const int* ACT_ROWS,     // [n_act]
    device const int* TOK_OFFS,     // [n_act + 1]
    device const int* SEG_STARTS,   // [n_act]
    device float* Y,                // [B, T, D_OUT]
    constant long& T,
    constant long& n_act,
    constant long& D_IN,
    constant long& D_OUT,
    uint tid [[thread_position_in_threadgroup]],
    uint g_tok [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sx[256];

    int i = find_active_segment(TOK_OFFS, (int)n_act, g_tok);
    int b = ACT_ROWS[i];
    long t = (long)SEG_STARTS[i] + (long)((int)g_tok - TOK_OFFS[i]);

    device const float* x_ptr = X + (b * T + t) * D_IN;
    for (long d = (long)tid; d < D_IN; d += 256L) {
        sx[d] = x_ptr[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* w_b = W_SEG + b * D_OUT * D_IN;
    device float* y_ptr = Y + (b * T + t) * D_OUT;
    for (long o0 = 0; o0 < D_OUT; o0 += 8L) {
        long o = o0 + (long)sg;
        if (o < D_OUT) {
            device const float* w_row = w_b + o * D_IN;
            float acc = 0.0f;
            for (long d = (long)lane; d < D_IN; d += 32L) {
                acc += w_row[d] * sx[d];
            }
            float sum = simd_sum(acc);
            if (lane == 0) y_ptr[o] = sum;
        }
    }
}

kernel void ragged_seg_mm_dx_bwd(
    device const float* GY,         // [B, T, D_OUT]
    device const float* W_SEG,      // [B, D_OUT, D_IN]
    device const int* ACT_ROWS,     // [n_act]
    device const int* TOK_OFFS,     // [n_act + 1]
    device const int* SEG_STARTS,   // [n_act]
    device float* GX,               // [B, T, D_IN]
    constant long& T,
    constant long& n_act,
    constant long& D_IN,
    constant long& D_OUT,
    uint tid [[thread_position_in_threadgroup]],
    uint g_tok [[threadgroup_position_in_grid]])
{
    threadgroup float sgy[256];

    int i = find_active_segment(TOK_OFFS, (int)n_act, g_tok);
    int b = ACT_ROWS[i];
    long t = (long)SEG_STARTS[i] + (long)((int)g_tok - TOK_OFFS[i]);

    device const float* gy_ptr = GY + (b * T + t) * D_OUT;
    for (long o = (long)tid; o < D_OUT; o += 256L) {
        sgy[o] = gy_ptr[o];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((long)tid < D_IN) {
        device const float* w_b = W_SEG + b * D_OUT * D_IN;
        float acc = 0.0f;
        for (long o = 0; o < D_OUT; ++o) {
            acc += sgy[o] * w_b[o * D_IN + tid];
        }
        GX[(b * T + t) * D_IN + tid] = acc;
    }
}

kernel void ragged_seg_mm_dw_bwd(
    device const float* GY,         // [B, T, D_OUT]
    device const float* X,          // [B, T, D_IN]
    device const int* ACT_ROWS,     // [n_act]
    device const int* SEG_STARTS,   // [n_act]
    device const int* SEG_STOPS,    // [n_act]
    device float* GW_SEG,           // [B, D_OUT, D_IN]
    constant long& T,
    constant long& D_IN,
    constant long& D_OUT,
    uint tid [[thread_position_in_threadgroup]],
    uint i [[threadgroup_position_in_grid]])
{
    int b = ACT_ROWS[i];
    long s0 = SEG_STARTS[i];
    long s1 = SEG_STOPS[i];
    long total_oi = D_OUT * D_IN;
    device float* gw_b = GW_SEG + b * total_oi;

    for (long idx = (long)tid; idx < total_oi; idx += 256L) {
        long o = idx / D_IN;
        long d = idx % D_IN;
        float acc = 0.0f;
        for (long t = s0; t < s1; ++t) {
            acc += GY[(b * T + t) * D_OUT + o] * X[(b * T + t) * D_IN + d];
        }
        gw_b[idx] += acc;
    }
}
