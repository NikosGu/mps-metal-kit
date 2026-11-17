#include "cuda_ops.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <math.h>

namespace mps_kit::cuda_ops {

namespace {

constexpr float EIGH_GRAD_CLIP_CU = 100.0f;

__device__ __forceinline__ float warp_sum_sp(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__device__ __forceinline__ double safe_hypot_f64(double a, double b) {
    double aa = fabs(a), bb = fabs(b);
    double mx = fmax(aa, bb);
    if (mx == 0.0 || !isfinite(mx)) return mx;
    double u = aa / mx, v = bb / mx;
    return mx * sqrt(u * u + v * v);
}

struct JacobiShared256 {
    double sA[32 * 32];
    double sV[32 * 32];
    double sd_raw[256];
    double sd_sort[256];
    double sw[256];
    double se[256];
    double scos_tql[256];
    double ssin_tql[256];
    double scos[16];
    double ssin[16];
    double smax_w[8];
    double mat_scale;
    double s_scale;
    double s_h;
    int top_rr[16];
    int bot_rr[16];
    int p_rr[16];
    int q_rr[16];
    int perm[256];
    int s_m;
    int s_brk;
    int s_done;
};

__device__ void parallel_jacobi_eigh_f64_device(
    int n, int tid,
    double* __restrict__ A_ptr,
    double* __restrict__ V_ptr,
    JacobiShared256& sj,
    double local_amax)
{
    const int lane = tid & 31;
    const int sg = tid >> 5;

    if (n == 1) {
        if (tid == 0) {
            double d0 = A_ptr[0];
            if (!isfinite(d0)) d0 = 0.0;
            V_ptr[0] = 1.0;
            sj.sd_sort[0] = d0;
            sj.perm[0] = 0;
        }
        __syncthreads();
        return;
    }

    if (n <= 32) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_amax = fmax(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, mask));
        }
        if (lane == 0) sj.smax_w[sg] = local_amax;
        __syncthreads();
        if (tid == 0) {
            double ms = 0.0;
            for (int g = 0; g < 8; ++g) ms = fmax(ms, sj.smax_w[g]);
            sj.mat_scale = fmax(1e-30, ms);
        }
        __syncthreads();

        const int np = (n + 1) & ~1;
        const int half = np >> 1;
        const int work_hn = half * n;
        const double conv_tol = 1e-12 * sj.mat_scale;
        if (tid < half) {
            sj.top_rr[tid] = tid;
            sj.bot_rr[tid] = np - 1 - tid;
        }
        __syncthreads();

        for (int sweep = 0; sweep < 20; ++sweep) {
            double sweep_max = 0.0;
            for (int step = 0; step < np - 1; ++step) {
                if (tid < half) {
                    int p = sj.top_rr[tid];
                    int q = sj.bot_rr[tid];
                    int pv = -1, qv = -1;
                    double cv = 1.0, sv = 0.0;
                    if (p < n && q < n) {
                        if (p > q) { int tmp = p; p = q; q = tmp; }
                        double apq = A_ptr[p * n + q];
                        double a_abs = fabs(apq);
                        sweep_max = fmax(sweep_max, a_abs);
                        double app = A_ptr[p * n + p];
                        double aqq = A_ptr[q * n + q];
                        if (a_abs > 1e-15 * (fabs(app) + fabs(aqq)) && a_abs > 1e-300) {
                            pv = p; qv = q;
                            double diff_pq = aqq - app;
                            double t;
                            if (fabs(diff_pq) >= 1e150 * a_abs) {
                                t = apq / diff_pq;
                            } else {
                                double tau = diff_pq / (2.0 * apq);
                                t = (tau >= 0.0) ? (1.0 / (tau + safe_hypot_f64(1.0, tau)))
                                                 : (-1.0 / (-tau + safe_hypot_f64(1.0, tau)));
                            }
                            cv = 1.0 / safe_hypot_f64(1.0, t);
                            sv = t * cv;
                        }
                    }
                    sj.p_rr[tid] = pv;
                    sj.q_rr[tid] = qv;
                    sj.scos[tid] = cv;
                    sj.ssin[tid] = sv;
                }
                __syncthreads();

                int nt = 0, nb = 0;
                if (half >= 2 && tid < half) {
                    nt = (tid == 0) ? sj.top_rr[0] : ((tid == 1) ? sj.bot_rr[0] : sj.top_rr[tid - 1]);
                    nb = (tid < half - 1) ? sj.bot_rr[tid + 1] : sj.top_rr[half - 1];
                }
                for (int idx = tid; idx < work_hn; idx += 256) {
                    int r = idx / half;
                    int k = idx - r * half;
                    int p = sj.p_rr[k];
                    if (p >= 0) {
                        int q = sj.q_rr[k];
                        double c = sj.scos[k], s = sj.ssin[k];
                        double arp = A_ptr[r * n + p], arq = A_ptr[r * n + q];
                        A_ptr[r * n + p] = c * arp - s * arq;
                        A_ptr[r * n + q] = s * arp + c * arq;
                        double vrp = V_ptr[r * n + p], vrq = V_ptr[r * n + q];
                        V_ptr[r * n + p] = c * vrp - s * vrq;
                        V_ptr[r * n + q] = s * vrp + c * vrq;
                    }
                }
                __syncthreads();

                if (half >= 2 && tid < half) {
                    sj.top_rr[tid] = nt;
                    sj.bot_rr[tid] = nb;
                }
                for (int idx = tid; idx < work_hn; idx += 256) {
                    int k = idx / n;
                    int col = idx - k * n;
                    int p = sj.p_rr[k];
                    if (p >= 0) {
                        int q = sj.q_rr[k];
                        double c = sj.scos[k], s = sj.ssin[k];
                        double apc = A_ptr[p * n + col], aqc = A_ptr[q * n + col];
                        A_ptr[p * n + col] = (col == q) ? 0.0 : (c * apc - s * aqc);
                        A_ptr[q * n + col] = (col == p) ? 0.0 : (s * apc + c * aqc);
                    }
                }
                __syncthreads();
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                sweep_max = fmax(sweep_max, __shfl_xor_sync(0xffffffffu, sweep_max, mask));
            }
            if (lane == 0) sj.smax_w[sg] = sweep_max;
            __syncthreads();
            if (tid == 0) {
                double sm = 0.0;
                for (int g = 0; g < 8; ++g) sm = fmax(sm, sj.smax_w[g]);
                sj.s_done = (sm <= conv_tol) ? 1 : 0;
            }
            __syncthreads();
            if (sj.s_done) break;
        }

        if (tid < n) {
            double dv = A_ptr[tid * n + tid];
            if (!isfinite(dv)) dv = 0.0;
            sj.sd_raw[tid] = dv;
        }
        __syncthreads();
    } else {
        // n in [33, 256]: Parallel symmetric Householder tred2 + unit-stride coalesced V_T tql2
        const int total_nn = n * n;
        if (tid < n) {
            sj.sd_raw[tid] = A_ptr[(n - 1) * n + tid];
            sj.se[tid] = 0.0;
        }
        __syncthreads();

        for (int i = n - 1; i >= 1; --i) {
            int l = i - 1;
            if (tid == 0) {
                double scale = 0.0;
                if (l > 0) {
                    for (int k = 0; k <= l; ++k) scale += fabs(sj.sd_raw[k]);
                }
                sj.s_scale = scale;
                if (scale == 0.0) {
                    sj.se[i] = sj.sd_raw[l];
                    sj.s_h = 0.0;
                } else {
                    double h = 0.0;
                    for (int k = 0; k <= l; ++k) {
                        sj.sd_raw[k] /= scale;
                        h += sj.sd_raw[k] * sj.sd_raw[k];
                    }
                    double f = sj.sd_raw[l];
                    double g = (f >= 0.0) ? -sqrt(h) : sqrt(h);
                    sj.se[i] = scale * g;
                    h -= f * g;
                    sj.sd_raw[l] = f - g;
                    sj.s_h = h;
                }
            }
            __syncthreads();

            if (sj.s_scale == 0.0) {
                if (tid <= l) {
                    sj.sd_raw[tid] = A_ptr[l * n + tid];
                    A_ptr[i * n + tid] = 0.0;
                    A_ptr[tid * n + i] = 0.0;
                }
            } else {
                if (tid <= l) {
                    int j = tid;
                    A_ptr[j * n + i] = sj.sd_raw[j];
                    double g = 0.0;
                    for (int k = 0; k <= l; ++k) g += A_ptr[k * n + j] * sj.sd_raw[k];
                    sj.se[j] = g / sj.s_h;
                }
                __syncthreads();

                if (tid == 0) {
                    double f = 0.0;
                    for (int j = 0; j <= l; ++j) f += sj.se[j] * sj.sd_raw[j];
                    double hh = f / (sj.s_h + sj.s_h);
                    for (int j = 0; j <= l; ++j) sj.se[j] -= hh * sj.sd_raw[j];
                }
                __syncthreads();

                int sub_nn = (l + 1) * (l + 1);
                for (int idx = tid; idx < sub_nn; idx += 256) {
                    int k = idx / (l + 1);
                    int j = idx - k * (l + 1);
                    A_ptr[k * n + j] -= (sj.sd_raw[j] * sj.se[k] + sj.se[j] * sj.sd_raw[k]);
                }
                __syncthreads();

                if (tid <= l) {
                    sj.sd_raw[tid] = A_ptr[l * n + tid];
                    A_ptr[i * n + tid] = 0.0;
                }
            }
            if (tid == 0) sj.sd_raw[i] = sj.s_h;
            __syncthreads();
        }

        for (int i = 1; i < n; ++i) {
            int l = i - 1;
            double h = sj.sd_raw[i];
            if (tid == 0) {
                A_ptr[(n - 1) * n + l] = A_ptr[l * n + l];
                A_ptr[l * n + l] = 1.0;
            }
            __syncthreads();
            if (h != 0.0) {
                if (tid <= l) {
                    double a_ti = A_ptr[tid * n + i];
                    sj.sd_raw[tid] = a_ti / h;
                    sj.scos_tql[tid] = a_ti;
                }
                __syncthreads();
                if (tid <= l) {
                    int j = tid;
                    double g = 0.0;
                    for (int k = 0; k <= l; ++k) g += sj.scos_tql[k] * A_ptr[k * n + j];
                    for (int k = 0; k <= l; ++k) A_ptr[k * n + j] -= g * sj.sd_raw[k];
                }
                __syncthreads();
            }
            if (tid <= l) A_ptr[tid * n + i] = 0.0;
            __syncthreads();
        }

        if (tid < n) {
            sj.sd_raw[tid] = A_ptr[(n - 1) * n + tid];
            A_ptr[(n - 1) * n + tid] = (tid == n - 1) ? 1.0 : 0.0;
        }
        if (tid == 0) {
            for (int i = 1; i < n; ++i) sj.se[i - 1] = sj.se[i];
            sj.se[n - 1] = 0.0;
        }
        __syncthreads();

        // Store Q^T in A_ptr temporarily -> copy to V_ptr as Q^T for coalesced tql2, then transpose back to V_ptr!
        for (int idx = tid; idx < total_nn; idx += 256) {
            int r = idx / n, c = idx - r * n;
            V_ptr[c * n + r] = A_ptr[idx];
        }
        __syncthreads();

        double f = 0.0;
        double tst1 = 0.0;
        const double eps = 2.2204460492503131e-16;
        if (tid == 0) {
            double norm_init = 0.0;
            for (int k = 0; k < n; ++k) {
                norm_init = fmax(norm_init, fabs(sj.sd_raw[k]) + fabs(sj.se[k]));
            }
            sj.mat_scale = norm_init;
        }
        __syncthreads();
        tst1 = sj.mat_scale;

        for (int l = 0; l < n; ++l) {
            if (tid == 0) {
                tst1 = fmax(tst1, fabs(sj.sd_raw[l]) + fabs(sj.se[l]));
                int m = l;
                while (m < n - 1) {
                    if (fabs(sj.se[m]) <= eps * tst1 || fabs(sj.se[m]) <= 1e-300) break;
                    ++m;
                }
                sj.s_m = m;
            }
            __syncthreads();

            if (sj.s_m > l) {
                int iter = 0;
                while (true) {
                    if (tid == 0) {
                        double e_l = sj.se[l];
                        if (++iter > 120 || fabs(e_l) <= eps * tst1 || fabs(e_l) <= 1e-300) {
                            sj.s_brk = 1;
                        } else {
                            sj.s_brk = 0;
                            int m = sj.s_m;
                            double g = sj.sd_raw[l];
                            double diff = sj.sd_raw[l + 1] - g;
                            if (fabs(diff) >= 1e150 * fabs(e_l)) {
                                sj.sd_raw[l] = e_l * (e_l / diff);
                                sj.sd_raw[l + 1] = diff;
                            } else {
                                double p = diff / (2.0 * e_l);
                                double r = safe_hypot_f64(p, 1.0);
                                if (p < 0.0) r = -r;
                                sj.sd_raw[l] = e_l / (p + r);
                                sj.sd_raw[l + 1] = e_l * (p + r);
                            }
                            double dl1 = sj.sd_raw[l + 1];
                            double h = g - sj.sd_raw[l];
                            for (int i = l + 2; i < n; ++i) sj.sd_raw[i] -= h;
                            f += h;

                            double p = sj.sd_raw[m];
                            double c = 1.0, c2 = 1.0, c3 = 1.0;
                            double el1 = sj.se[l + 1];
                            double s = 0.0, s2 = 0.0;
                            for (int i = m - 1; i >= l; --i) {
                                c3 = c2; c2 = c; s2 = s;
                                g = c * sj.se[i];
                                h = c * p;
                                double r = safe_hypot_f64(p, sj.se[i]);
                                sj.se[i + 1] = s * r;
                                if (r == 0.0) { s = 0.0; c = 1.0; }
                                else { s = sj.se[i] / r; c = p / r; }
                                sj.scos_tql[i] = c;
                                sj.ssin_tql[i] = s;
                                p = c * sj.sd_raw[i] - s * g;
                                sj.sd_raw[i + 1] = h + s * (c * g + s * sj.sd_raw[i]);
                            }
                            p = (dl1 == 0.0) ? 0.0 : (-s * s2 * c3 * el1 * e_l / dl1);
                            sj.se[l] = s * p;
                            sj.sd_raw[l] = c * p;
                        }
                    }
                    __syncthreads();
                    if (sj.s_brk) break;

                    if (tid < n) {
                        int m = sj.s_m;
                        double z_carry = V_ptr[m * n + tid];
                        for (int i = m - 1; i >= l; --i) {
                            double c = sj.scos_tql[i];
                            double s = sj.ssin_tql[i];
                            double z0 = V_ptr[i * n + tid];
                            V_ptr[(i + 1) * n + tid] = s * z0 + c * z_carry;
                            z_carry = c * z0 - s * z_carry;
                        }
                        V_ptr[l * n + tid] = z_carry;
                    }
                    __syncthreads();

                    if (tid == 0) {
                        if (fabs(sj.se[l]) <= eps * tst1 || fabs(sj.se[l]) <= 1e-300) {
                            sj.s_brk = 1;
                        } else {
                            int m = l;
                            while (m < n - 1) {
                                if (fabs(sj.se[m]) <= eps * tst1 || fabs(sj.se[m]) <= 1e-300) break;
                                ++m;
                            }
                            sj.s_m = m;
                            if (m <= l) sj.s_brk = 1;
                        }
                    }
                    __syncthreads();
                    if (sj.s_brk) break;
                }
            }
            if (tid == 0) {
                sj.sd_raw[l] += f;
                sj.se[l] = 0.0;
            }
            __syncthreads();
        }

        // Transpose V^T back into standard row-major V_ptr[row * n + col] via A_ptr
        for (int idx = tid; idx < total_nn; idx += 256) {
            A_ptr[idx] = V_ptr[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < total_nn; idx += 256) {
            int r = idx / n, c = idx - r * n;
            V_ptr[r * n + c] = A_ptr[c * n + r];
        }
        if (tid < n && !isfinite(sj.sd_raw[tid])) sj.sd_raw[tid] = 0.0;
        __syncthreads();
    }

    if (tid < n) {
        double my_d = sj.sd_raw[tid];
        int rank = 0;
        for (int j = 0; j < n; ++j) {
            double dj = sj.sd_raw[j];
            if (dj < my_d || (dj == my_d && j < tid)) ++rank;
        }
        sj.sd_sort[rank] = my_d;
        sj.perm[rank] = tid;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(256) sym_eigh_f64_kernel(
    const float* __restrict__ G, double* __restrict__ WORK,
    float* __restrict__ L_out, float* __restrict__ V_out, int D)
{
    __shared__ JacobiShared256 sj;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const float* gb = G + (int64_t)b * D * D;
    double* A_ptr = (D <= 32) ? sj.sA : (WORK + (int64_t)b * 2 * D * D);
    double* V_ptr = (D <= 32) ? sj.sV : (A_ptr + D * D);

    double local_amax = 0.0;
    for (int idx = tid; idx < D * D; idx += 256) {
        int r = idx / D, c = idx - r * D;
        double v = 0.5 * ((double)gb[r * D + c] + (double)gb[c * D + r]);
        A_ptr[idx] = v;
        V_ptr[idx] = (r == c) ? 1.0 : 0.0;
        local_amax = fmax(local_amax, fabs(v));
    }
    __syncthreads();

    parallel_jacobi_eigh_f64_device(D, tid, A_ptr, V_ptr, sj, local_amax);

    if (tid < D) L_out[(int64_t)b * D + tid] = (float)sj.sd_sort[tid];
    for (int idx = tid; idx < D * D; idx += 256) {
        int r = idx / D, c = idx - r * D;
        double vv = V_ptr[r * D + sj.perm[c]];
        if (!isfinite(vv) || fabs(vv) > 1.5) vv = 0.0;
        V_out[(int64_t)b * D * D + idx] = (float)vv;
    }
}

__global__ void __launch_bounds__(256) dual_gram_subspace_f64_kernel(
    const float* __restrict__ X, const float* __restrict__ W, const int* __restrict__ LENS,
    double* __restrict__ WORK, float* __restrict__ U_out, float* __restrict__ PACK_out,
    int M, int D, int r, int kp, double rel_tol, double min_eig)
{
    __shared__ JacobiShared256 sj;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    float* pack_b = PACK_out + (int64_t)b * (D + 3) * kp;
    float* u_b = U_out + (int64_t)b * D * r;
    for (int idx = tid; idx < (D + 3) * kp; idx += 256) pack_b[idx] = 0.0f;
    for (int idx = tid; idx < D * r; idx += 256) u_b[idx] = 0.0f;
    __syncthreads();

    int Lb = LENS[b];
    if (Lb < 0) Lb = 0;
    if (Lb > M) Lb = M;
    if (Lb <= 0) return;

    const float* xb = X + (int64_t)b * M * D;
    const float* wb = W + (int64_t)b * M;

    const bool is_dual = (Lb < D);
    const int n = is_dual ? Lb : D;
    double* A_ptr = (n <= 32) ? sj.sA : (WORK + (int64_t)b * 2 * D * D);
    double* V_ptr = (n <= 32) ? sj.sV : (A_ptr + D * D);

    if (is_dual) {
        if (tid < Lb) sj.sw[tid] = sqrt(fmax(0.0, (double)wb[tid]));
        __syncthreads();
    }

    double local_amax = 0.0;
    for (int idx = tid; idx < n * n; idx += 256) {
        int r0 = idx / n, c0 = idx - r0 * n;
        V_ptr[idx] = (r0 == c0) ? 1.0 : 0.0;
        if (r0 <= c0) {
            double acc = 0.0;
            if (is_dual) {
                double sw = sj.sw[r0] * sj.sw[c0];
                if (sw > 0.0) {
                    const float* xr = xb + (int64_t)r0 * D;
                    const float* xc = xb + (int64_t)c0 * D;
                    for (int col = 0; col < D; ++col) acc += (double)xr[col] * (double)xc[col];
                    acc *= sw;
                }
            } else {
                for (int t = 0; t < Lb; ++t) {
                    double wt = fmax(0.0, (double)wb[t]);
                    const float* xt = xb + (int64_t)t * D;
                    acc += wt * (double)xt[r0] * (double)xt[c0];
                }
            }
            A_ptr[r0 * n + c0] = acc;
            A_ptr[c0 * n + r0] = acc;
            local_amax = fmax(local_amax, fabs(acc));
        }
    }
    __syncthreads();

    parallel_jacobi_eigh_f64_device(n, tid, A_ptr, V_ptr, sj, local_amax);

    double top = fmax(0.0, sj.sd_sort[n - 1]);
    if (is_dual) {
        double eig_tol = fmax(1e-15, 1e-13 * top);
        if (tid < Lb) {
            int p = tid, src = Lb - 1 - p;
            double lp = fmax(0.0, sj.sd_sort[src]);
            bool pos = (lp > eig_tol);
            bool keep = pos && (top > min_eig) && (lp > rel_tol * top) && (p < r);
            sj.sd_raw[p] = pos ? (1.0 / sqrt(fmax(lp, 1e-30))) : 0.0;
            pack_b[(int64_t)D * kp + p] = (float)lp;
            pack_b[(int64_t)(D + 1) * kp + p] = keep ? 1.0f : 0.0f;
            pack_b[(int64_t)(D + 2) * kp + p] = pos ? 1.0f : 0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < D * Lb; idx += 256) {
            int col = idx / Lb, p = idx - col * Lb;
            double inv_s = sj.sd_raw[p], acc = 0.0;
            if (inv_s > 0.0) {
                int orig_c = sj.perm[Lb - 1 - p];
                for (int t = 0; t < Lb; ++t) acc += (sj.sw[t] * (double)xb[(int64_t)t * D + col]) * V_ptr[t * Lb + orig_c];
                acc *= inv_s;
                if (!isfinite(acc) || fabs(acc) > 1.5) acc = 0.0;
            }
            float q_dp = (float)acc;
            pack_b[(int64_t)col * kp + p] = q_dp;
            if (p < r) u_b[col * r + p] = q_dp * pack_b[(int64_t)(D + 1) * kp + p];
        }
    } else {
        if (tid < D) {
            int p = tid, src = D - 1 - p;
            double lp = fmax(0.0, sj.sd_sort[src]);
            bool keep = (top > min_eig) && (lp > rel_tol * top) && (p < r);
            pack_b[(int64_t)D * kp + p] = (float)lp;
            pack_b[(int64_t)(D + 1) * kp + p] = keep ? 1.0f : 0.0f;
            pack_b[(int64_t)(D + 2) * kp + p] = 1.0f;
        }
        __syncthreads();
        for (int idx = tid; idx < D * D; idx += 256) {
            int col = idx / D, p = idx - col * D;
            int orig_c = sj.perm[D - 1 - p];
            double vv = V_ptr[col * D + orig_c];
            if (!isfinite(vv) || fabs(vv) > 1.5) vv = 0.0;
            float q_dp = (float)vv;
            pack_b[(int64_t)col * kp + p] = q_dp;
            if (p < r) u_b[col * r + p] = q_dp * pack_b[(int64_t)(D + 1) * kp + p];
        }
    }
}

__global__ void __launch_bounds__(256) spectral_subspace_bwd_kernel(
    const float* __restrict__ X, const float* __restrict__ W, const int* __restrict__ LENS,
    const float* __restrict__ PACK, const float* __restrict__ GU,
    float* __restrict__ GX, float* __restrict__ GW,
    int64_t M, int64_t D, int64_t K_PAD, int64_t R, int64_t R_EFF)
{
    __shared__ float sVtop[256 * 12];
    __shared__ float sBmat[256 * 12];
    __shared__ float sRed[256];

    const int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    const int64_t b = blockIdx.x;
    const int L_b = LENS[b];
    const float* pack_b = PACK + b * (D + 3) * K_PAD;
    const float* Q = pack_b;
    const float* L_vec = pack_b + D * K_PAD;
    const float* kf_vec = pack_b + (D + 1) * K_PAD;
    const float* vq_vec = pack_b + (D + 2) * K_PAD;
    const float* gu_b = GU + b * D * R;

    float b_acc[12];
    for (int a = 0; a < 12; ++a) b_acc[a] = 0.0f;

    bool active_d = (tid < D);
    if (active_d) {
        for (int64_t a = 0; a < R_EFF; ++a) {
            float kfa = kf_vec[a];
            sVtop[tid * 12 + a] = Q[tid * K_PAD + a] * kfa;
            float la = L_vec[a];
            float inv_2l = (kfa > 0.5f && la > 1e-20f) ? (0.5f / la) : 0.0f;
            b_acc[a] = gu_b[tid * R + a] * kfa * inv_2l;
        }
    }
    __syncthreads();

    for (int64_t i = 0; i < K_PAD; ++i) {
        if (vq_vec[i] > 0.5f) {
            float kfi = kf_vec[i], li = L_vec[i];
            float qi = active_d ? Q[tid * K_PAD + i] : 0.0f;
            for (int64_t a = 0; a < R_EFF; ++a) {
                float kfa = kf_vec[a];
                float prod = (active_d && kfa > 0.5f) ? (0.5f * qi * gu_b[tid * R + a]) : 0.0f;
                float sg_sum = warp_sum_sp(prod);
                if (lane == 0) sRed[a * 8 + sg] = sg_sum;
            }
            __syncthreads();
            if (active_d) {
                for (int64_t a = 0; a < R_EFF; ++a) {
                    float c_ia = 0.0f;
                    for (int g = 0; g < 8; ++g) c_ia += sRed[a * 8 + g];
                    float kfa = kf_vec[a], la = L_vec[a];
                    float inv_la = (kfa > 0.5f && la > 1e-20f) ? (1.0f / la) : 0.0f;
                    float dk_f = 0.0f;
                    if (kfi < 0.5f && kfa > 0.5f) {
                        float dl = la - li;
                        dk_f = fminf(fmaxf(dl / (dl * dl + 1e-12f), -EIGH_GRAD_CLIP_CU), EIGH_GRAD_CLIP_CU);
                    }
                    b_acc[a] += qi * c_ia * (dk_f - inv_la);
                }
            }
            __syncthreads();
        }
    }

    if (active_d) {
        for (int64_t a = 0; a < R_EFF; ++a) sBmat[tid * 12 + a] = b_acc[a];
    }
    __syncthreads();

    for (int64_t m = 0; m < M; ++m) {
        int64_t bm = b * M + m;
        if (m >= L_b) {
            if (tid == 0) GW[bm] = 0.0f;
            if (active_d) GX[bm * D + tid] = 0.0f;
            continue;
        }
        float wm = fmaxf(W[bm], 0.0f);
        float xd = active_d ? X[bm * D + tid] : 0.0f;
        for (int64_t a = 0; a < R_EFF; ++a) {
            float pb = warp_sum_sp(active_d ? (xd * sBmat[tid * 12 + a]) : 0.0f);
            float pv = warp_sum_sp(active_d ? (xd * sVtop[tid * 12 + a]) : 0.0f);
            if (lane == 0) {
                sRed[a * 8 + sg] = pb;
                sRed[96 + a * 8 + sg] = pv;
            }
        }
        __syncthreads();
        float xb_v[12], xv_v[12], gw_val = 0.0f;
        for (int64_t a = 0; a < R_EFF; ++a) {
            float sb = 0.0f, sv = 0.0f;
            for (int g = 0; g < 8; ++g) { sb += sRed[a * 8 + g]; sv += sRed[96 + a * 8 + g]; }
            xb_v[a] = sb; xv_v[a] = sv;
            gw_val += 2.0f * sb * sv;
        }
        if (tid == 0) GW[bm] = gw_val;
        if (active_d) {
            float gx_d = 0.0f;
            for (int64_t a = 0; a < R_EFF; ++a) gx_d += xb_v[a] * sVtop[tid * 12 + a] + xv_v[a] * sBmat[tid * 12 + a];
            GX[bm * D + tid] = 2.0f * wm * gx_d;
        }
        __syncthreads();
    }
}

} // namespace

void cuda_eigh_fwd(torch::Tensor G, torch::Tensor L_out, torch::Tensor V_out) {
    int B = (int)G.size(0), D = (int)G.size(1);
    if (B <= 0 || D <= 0) return;
    TORCH_CHECK(D <= 256, "cuda_eigh_fwd supports D <= 256");
    int64_t w_el = (D > 32) ? ((int64_t)B * 2 * D * D) : 1;
    auto work = torch::empty({w_el}, G.options().dtype(torch::kFloat64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sym_eigh_f64_kernel<<<(unsigned)B, 256, 0, stream>>>(
        G.data_ptr<float>(), work.data_ptr<double>(), L_out.data_ptr<float>(), V_out.data_ptr<float>(), D);
}

int64_t cuda_dual_gram_subspace_fwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor U_out, torch::Tensor PACK_out,
    int64_t rank, int64_t k_pad, double rel_tol, double min_eig)
{
    int B = (int)X.size(0), M = (int)X.size(1), D = (int)X.size(2);
    if (B <= 0 || M <= 0 || D <= 0) return 0;
    TORCH_CHECK(D <= 256, "cuda_dual_gram_subspace_fwd supports D <= 256");
    U_out.zero_();
    PACK_out.zero_();
    int64_t w_el = (D > 32) ? ((int64_t)B * 2 * D * D) : 1;
    auto work = torch::empty({w_el}, X.options().dtype(torch::kFloat64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dual_gram_subspace_f64_kernel<<<(unsigned)B, 256, 0, stream>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), LENS.data_ptr<int>(),
        work.data_ptr<double>(), U_out.data_ptr<float>(), PACK_out.data_ptr<float>(),
        M, D, (int)rank, (int)k_pad, rel_tol, min_eig);
    return std::min<int64_t>(rank, std::min<int64_t>(M, D));
}

void spectral_subspace_bwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor PACK, torch::Tensor GU,
    torch::Tensor GX, torch::Tensor GW, int64_t r_eff)
{
    int64_t B = X.size(0), M = X.size(1), D = X.size(2), K_PAD = PACK.size(2), R = GU.size(2);
    if (B <= 0 || M <= 0 || D <= 0) return;
    TORCH_CHECK(D <= 256 && R <= 12, "spectral_subspace_bwd supports D <= 256, rank <= 12");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    spectral_subspace_bwd_kernel<<<(unsigned)B, 256, 0, stream>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), LENS.data_ptr<int>(),
        PACK.data_ptr<float>(), GU.data_ptr<float>(), GX.data_ptr<float>(), GW.data_ptr<float>(),
        M, D, K_PAD, R, r_eff);
}

} // namespace mps_kit::cuda_ops
