#include "mps_metal_kit/ops.hpp"
#include "mps_metal_kit/mps_bridge.hpp"

#include <cmath>
#include <cstring>

namespace mps_kit::ops {

void amx_eigh_fwd(torch::Tensor G, torch::Tensor L_out, torch::Tensor V_out) {
    TORCH_CHECK(G.dim() == 3 && G.size(1) == G.size(2), "G must have shape [B, D, D]");
    int B = (int)G.size(0);
    int D = (int)G.size(1);
    if (B == 0 || D == 0) return;

    sync_torch_stream();
    const float* g_ptr = UnifiedBuffer::from_tensor(G).data_as<float>();
    float* l_ptr = UnifiedBuffer::from_tensor(L_out).data_as<float>();
    float* v_ptr = UnifiedBuffer::from_tensor(V_out).data_as<float>();

    thread_local std::vector<double> workA;
    thread_local std::vector<double> workW;
    workA.resize((size_t)D * (size_t)D);
    workW.resize((size_t)D);

    for (int b = 0; b < B; ++b) {
        const float* gb = g_ptr + (size_t)b * D * D;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                workA[(size_t)j * D + i] = 0.5 * ((double)gb[i * D + j] + (double)gb[j * D + i]);
            }
        }
        amx_sym_eigh_f64(D, workA.data(), workW.data());
        float* lb = l_ptr + (size_t)b * D;
        float* vb = v_ptr + (size_t)b * D * D;
        for (int k = 0; k < D; ++k) {
            lb[k] = (float)workW[k];
            for (int i = 0; i < D; ++i) {
                vb[i * D + k] = (float)workA[(size_t)k * D + i];
            }
        }
    }
}

int64_t amx_dual_gram_subspace_fwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor U_out, torch::Tensor PACK_out,
    int64_t rank, int64_t k_pad, double rel_tol, double min_eig)
{
    int B = (int)X.size(0);
    int M = (int)X.size(1);
    int D = (int)X.size(2);
    int r = (int)rank;
    int kp = (int)k_pad;
    if (B <= 0 || M <= 0 || D <= 0) return 0;

    sync_torch_stream();
    const float* x_ptr = UnifiedBuffer::from_tensor(X).data_as<float>();
    const float* w_ptr = UnifiedBuffer::from_tensor(W).data_as<float>();
    const int32_t* l_ptr = UnifiedBuffer::from_tensor(LENS).data_as<int32_t>();
    float* u_ptr = UnifiedBuffer::from_tensor(U_out).data_as<float>();
    float* p_ptr = UnifiedBuffer::from_tensor(PACK_out).data_as<float>();

    std::memset(u_ptr, 0, (size_t)B * D * r * sizeof(float));
    std::memset(p_ptr, 0, (size_t)B * (D + 3) * kp * sizeof(float));

    int max_k = 1;
    thread_local std::vector<double> Aw, Gram, evals, Vscaled, Q_col;

    for (int b = 0; b < B; ++b) {
        int Lb = std::clamp((int)l_ptr[b], 0, M);
        if (Lb <= 0) continue;

        Aw.assign((size_t)Lb * D, 0.0);
        const float* xb = x_ptr + (size_t)b * M * D;
        const float* wb = w_ptr + (size_t)b * M;
        for (int t = 0; t < Lb; ++t) {
            double sw = std::sqrt(std::max(0.0, (double)wb[t]));
            const float* xt = xb + (size_t)t * D;
            double* aw_t = Aw.data() + (size_t)t * D;
            for (int d = 0; d < D; ++d) aw_t[d] = (double)xt[d] * sw;
        }

        float* pack_b = p_ptr + (size_t)b * (D + 3) * kp;
        float* Q_dst = pack_b;
        float* L_dst = pack_b + (size_t)D * kp;
        float* kf_dst = pack_b + (size_t)(D + 1) * kp;
        float* vq_dst = pack_b + (size_t)(D + 2) * kp;
        float* u_b = u_ptr + (size_t)b * D * r;

        if (Lb < D) {
            max_k = std::max(max_k, Lb);
            Gram.assign((size_t)Lb * Lb, 0.0);
            evals.assign((size_t)Lb, 0.0);
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, Lb, D, 1.0, Aw.data(), D, 0.0, Gram.data(), Lb);
            for (int col = 0; col < Lb; ++col) {
                for (int row = 0; row < col; ++row) {
                    Gram[(size_t)row * Lb + col] = Gram[(size_t)col * Lb + row];
                }
            }
            amx_sym_eigh_f64(Lb, Gram.data(), evals.data());

            double top = std::max(0.0, evals[Lb - 1]);
            double eig_tol = std::max(1e-15, 1e-13 * top);
            Vscaled.assign((size_t)Lb * Lb, 0.0);
            for (int p = 0; p < Lb; ++p) {
                int src_col = Lb - 1 - p;
                double lp = std::max(0.0, evals[src_col]);
                L_dst[p] = (float)lp;
                bool pos = (lp > eig_tol);
                vq_dst[p] = pos ? 1.0f : 0.0f;
                bool keep = pos && (top > min_eig) && (lp > rel_tol * top) && (p < r);
                kf_dst[p] = keep ? 1.0f : 0.0f;
                double inv_s = pos ? (1.0 / std::sqrt(std::max(lp, 1e-30))) : 0.0;
                for (int t = 0; t < Lb; ++t) {
                    Vscaled[(size_t)p * Lb + t] = Gram[(size_t)src_col * Lb + t] * inv_s;
                }
            }
            Q_col.assign((size_t)D * Lb, 0.0);
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                D, Lb, Lb, 1.0, Aw.data(), D, Vscaled.data(), Lb, 0.0, Q_col.data(), D
            );
            for (int p = 0; p < Lb; ++p) {
                float kfp = kf_dst[p];
                for (int d = 0; d < D; ++d) {
                    float q_dp = (float)Q_col[(size_t)p * D + d];
                    Q_dst[(size_t)d * kp + p] = q_dp;
                    if (p < r) u_b[d * r + p] = q_dp * kfp;
                }
            }
        } else {
            max_k = D;
            Gram.assign((size_t)D * D, 0.0);
            evals.assign((size_t)D, 0.0);
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, D, Lb, 1.0, Aw.data(), D, 0.0, Gram.data(), D);
            for (int col = 0; col < D; ++col) {
                for (int row = 0; row < col; ++row) {
                    Gram[(size_t)row * D + col] = Gram[(size_t)col * D + row];
                }
            }
            amx_sym_eigh_f64(D, Gram.data(), evals.data());

            double top = std::max(0.0, evals[D - 1]);
            for (int p = 0; p < D; ++p) {
                int src_col = D - 1 - p;
                double lp = std::max(0.0, evals[src_col]);
                L_dst[p] = (float)lp;
                vq_dst[p] = 1.0f;
                bool keep = (top > min_eig) && (lp > rel_tol * top) && (p < r);
                kf_dst[p] = keep ? 1.0f : 0.0f;
                float kfp = kf_dst[p];
                for (int d = 0; d < D; ++d) {
                    float q_dp = (float)Gram[(size_t)src_col * D + d];
                    Q_dst[(size_t)d * kp + p] = q_dp;
                    if (p < r) u_b[d * r + p] = q_dp * kfp;
                }
            }
        }
    }
    return std::min(r, max_k);
}

void spectral_subspace_bwd(
    torch::Tensor X, torch::Tensor W, torch::Tensor LENS,
    torch::Tensor PACK, torch::Tensor GU,
    torch::Tensor GX, torch::Tensor GW, int64_t r_eff)
{
    int64_t B = X.size(0);
    int64_t M = X.size(1);
    int64_t D = X.size(2);
    int64_t K_PAD = PACK.size(2);
    int64_t R = GU.size(2);
    if (B <= 0 || M <= 0 || D <= 0) return;
    TORCH_CHECK(D <= 256 && R <= 12, "spectral_subspace_bwd supports D <= 256, rank <= 12");

    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline("spectral", "spectral_subspace_bwd");
    auto bX = UnifiedBuffer::from_tensor(X);
    auto bW = UnifiedBuffer::from_tensor(W);
    auto bL = UnifiedBuffer::from_tensor(LENS);
    auto bP = UnifiedBuffer::from_tensor(PACK);
    auto bGU = UnifiedBuffer::from_tensor(GU);
    auto bGX = UnifiedBuffer::from_tensor(GX);
    auto bGW = UnifiedBuffer::from_tensor(GW);

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bX.bind(enc, 0);
        bW.bind(enc, 1);
        bL.bind(enc, 2);
        bP.bind(enc, 3);
        bGU.bind(enc, 4);
        bGX.bind(enc, 5);
        bGW.bind(enc, 6);
        long m_c = (long)M, d_c = (long)D, kp_c = (long)K_PAD, r_c = (long)R, re_c = (long)r_eff;
        [enc setBytes:&m_c length:sizeof(long) atIndex:7];
        [enc setBytes:&d_c length:sizeof(long) atIndex:8];
        [enc setBytes:&kp_c length:sizeof(long) atIndex:9];
        [enc setBytes:&r_c length:sizeof(long) atIndex:10];
        [enc setBytes:&re_c length:sizeof(long) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}

} // namespace mps_kit::ops
